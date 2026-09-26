package collective

import (
	"context"
	"net"
	"net/netip"
	"time"
)

// Exchange is an owned collective session. Close aborts it permanently and
// releases its sockets. Exchange preserves the Coordinator and Peer protocol.
type Exchange interface {
	Exchange(context.Context, uint32, []float32) ([]float32, error)
	Close() error
}

// ConnectionConfig specifies already-admitted TCP endpoints and setup budgets.
// Network must be tcp4 or tcp6 and both IPs must match that family. Address is
// rank zero's listening endpoint; LocalAddress selects the dialing interface and
// must equal Address.Addr() on rank zero. Addresses must be unicast and resolved;
// no DNS, environment, implicit interface or transport encryption is supplied.
// Every duration must be positive. ConnectTimeout bounds all setup attempts,
// DialTimeout bounds each peer attempt, and RetryInterval separates failures.
type ConnectionConfig struct {
	Network        string
	Address        netip.AddrPort
	LocalAddress   netip.Addr
	ConnectTimeout time.Duration
	DialTimeout    time.Duration
	RetryInterval  time.Duration
}

// Connect listens once for rank zero or retries peer dialing until setup ends.
// Spec is validated and its secret copied before I/O. Update is required on rank
// zero and unused on peers. No handshake, gradient or optimizer operation occurs
// here; those remain in Exchange. Established sessions are never reconnected.
//
// The earlier of ctx's deadline and ConnectTimeout bounds setup only. Once this
// function succeeds, the caller owns the returned session and must Close it;
// later Exchange calls carry their own contexts and Spec.PhaseTimeout. Every
// setup failure closes acquired sockets. Errors omit endpoints and credentials.
func Connect(ctx context.Context, config ConnectionConfig, spec Spec, update Update) (Exchange, error) {
	if ctx == nil {
		return nil, ErrSpec
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := config.validate(spec.Rank); err != nil {
		return nil, err
	}
	if spec.Rank == 0 && update == nil {
		return nil, ErrSpec
	}
	owned, err := configure(spec)
	if err != nil {
		return nil, err
	}
	spec = owned.spec
	defer clear(spec.Secret)
	ready, cancel := context.WithTimeout(ctx, config.ConnectTimeout)
	defer cancel()
	if spec.Rank == 0 {
		listener, err := (&net.ListenConfig{}).Listen(ready, config.Network, config.Address.String())
		if err != nil {
			return nil, transportError(ready, err)
		}
		coordinator, err := NewCoordinator(listener, spec, update)
		if err != nil {
			_ = listener.Close()
			return nil, err
		}
		if err = ready.Err(); err != nil {
			_ = coordinator.Close()
			return nil, err
		}
		return coordinator, nil
	}
	dialer := net.Dialer{Timeout: config.DialTimeout, LocalAddr: net.TCPAddrFromAddrPort(netip.AddrPortFrom(config.LocalAddress, 0))}
	for {
		conn, err := dialer.DialContext(ready, config.Network, config.Address.String())
		if err == nil {
			peer, err := NewPeer(conn, spec)
			if err != nil {
				_ = conn.Close()
				return nil, err
			}
			if err = ready.Err(); err != nil {
				_ = peer.Close()
				return nil, err
			}
			return peer, nil
		}
		if err := ready.Err(); err != nil {
			return nil, err
		}
		timer := time.NewTimer(config.RetryInterval)
		select {
		case <-ready.Done():
			timer.Stop()
			return nil, ready.Err()
		case <-timer.C:
		}
	}
}

func (config ConnectionConfig) validate(rank uint32) error {
	if !config.Address.IsValid() || config.Address.Port() == 0 || config.ConnectTimeout <= 0 || config.DialTimeout <= 0 || config.RetryInterval <= 0 {
		return ErrSpec
	}
	for _, address := range []netip.Addr{config.Address.Addr(), config.LocalAddress} {
		if !address.IsValid() || address.IsUnspecified() || address.IsMulticast() ||
			(config.Network != "tcp4" && config.Network != "tcp6") ||
			(config.Network == "tcp4" && !address.Is4()) ||
			(config.Network == "tcp6" && (!address.Is6() || address.Is4In6())) {
			return ErrSpec
		}
	}
	if rank == 0 && config.Address.Addr() != config.LocalAddress {
		return ErrSpec
	}
	return nil
}
