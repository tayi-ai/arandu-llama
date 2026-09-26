package unit_test

import (
	"bytes"
	"context"
	"errors"
	"net"
	"net/netip"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/tayi-ai/arandu-llama/training/collective"
)

func connectionConfig(t *testing.T) collective.ConnectionConfig {
	t.Helper()
	listener, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	address := listener.Addr().(*net.TCPAddr).AddrPort()
	if err := listener.Close(); err != nil {
		t.Fatal(err)
	}
	return collective.ConnectionConfig{Network: "tcp4", Address: address, LocalAddress: address.Addr(),
		ConnectTimeout: 3 * time.Second, DialTimeout: 100 * time.Millisecond, RetryInterval: 10 * time.Millisecond}
}

func connectionSpec(rank uint32) collective.Spec {
	return collective.Spec{World: 3, Rank: rank, Steps: 2, Dimension: 3, SessionID: "synthetic-transport-session",
		ContractSHA256: strings.Repeat("a", 64), TensorLayoutSHA256: strings.Repeat("b", 64),
		PhaseTimeout: 3 * time.Second, MaxFrameBytes: 1 << 20, MaxBufferedBytes: 1 << 20, Secret: bytes.Repeat([]byte{0x12}, 32)}
}

func TestConnectLoopbackPreservesOrderedExchangeAndSecretOwnership(t *testing.T) {
	config := connectionConfig(t)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	updates := 0
	coordinatorSpec := connectionSpec(0)
	coordinator, err := collective.Connect(ctx, config, coordinatorSpec, func(all [][]float32) ([]float32, error) {
		updates++
		if len(all) != 3 {
			return nil, errors.New("incomplete synthetic rank set")
		}
		values := make([]float32, 3)
		for rank, gradient := range all {
			if !reflect.DeepEqual(gradient, []float32{float32(rank), float32(updates), .25}) {
				return nil, errors.New("synthetic rank ordering or step differs")
			}
			for i, value := range gradient {
				values[i] += value
			}
		}
		return values, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	defer coordinator.Close()
	clear(coordinatorSpec.Secret)
	type result struct {
		step   uint32
		values []float32
		err    error
	}
	results := make(chan result, 4)
	// Arrival order must not select the numeric reduction order.
	for _, rank := range []uint32{2, 1} {
		spec := connectionSpec(rank)
		peer, err := collective.Connect(ctx, config, spec, nil)
		if err != nil {
			t.Fatal(err)
		}
		defer peer.Close()
		clear(spec.Secret)
		go func(rank uint32, peer collective.Exchange) {
			for step := uint32(1); step <= 2; step++ {
				values, err := peer.Exchange(ctx, step, []float32{float32(rank), float32(step), .25})
				results <- result{step, values, err}
				if err != nil {
					return
				}
			}
		}(rank, peer)
	}
	for step := uint32(1); step <= 2; step++ {
		values, err := coordinator.Exchange(ctx, step, []float32{0, float32(step), .25})
		if err != nil || !reflect.DeepEqual(values, []float32{3, float32(3 * step), .75}) {
			t.Fatalf("coordinator step %d: values=%v error=%v", step, values, err)
		}
	}
	for range 4 {
		select {
		case got := <-results:
			if got.err != nil || !reflect.DeepEqual(got.values, []float32{3, float32(3 * got.step), .75}) {
				t.Fatalf("peer result: %+v", got)
			}
		case <-ctx.Done():
			t.Fatal(ctx.Err())
		}
	}
	if updates != 2 {
		t.Fatalf("optimizer callback count changed: %d", updates)
	}
}

func TestConnectRetriesBeforeCoordinatorBecomesAvailable(t *testing.T) {
	config := connectionConfig(t)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	type result struct {
		exchange collective.Exchange
		err      error
	}
	connected := make(chan result, 1)
	go func() {
		peer, err := collective.Connect(ctx, config, connectionSpec(1), nil)
		connected <- result{peer, err}
	}()
	select {
	case got := <-connected:
		if got.exchange != nil {
			_ = got.exchange.Close()
		}
		t.Fatalf("peer completed before an endpoint existed: %v", got.err)
	case <-time.After(4 * config.RetryInterval):
	}
	coordinator, err := collective.Connect(ctx, config, connectionSpec(0), func(all [][]float32) ([]float32, error) { return all[0], nil })
	if err != nil {
		t.Fatal(err)
	}
	defer coordinator.Close()
	select {
	case got := <-connected:
		if got.err != nil || got.exchange == nil {
			t.Fatalf("peer did not become ready: %v", got.err)
		}
		if err := got.exchange.Close(); err != nil {
			t.Fatal(err)
		}
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
}

func TestConnectWaitHonorsCancellationAndBothDeadlines(t *testing.T) {
	for _, mode := range []string{"cancel", "parent-deadline", "setup-deadline"} {
		t.Run(mode, func(t *testing.T) {
			config := connectionConfig(t)
			config.ConnectTimeout, config.RetryInterval = time.Hour, time.Hour
			var ctx context.Context
			var cancel context.CancelFunc
			want := context.DeadlineExceeded
			if mode == "parent-deadline" {
				ctx, cancel = context.WithTimeout(context.Background(), 30*time.Millisecond)
			} else {
				ctx, cancel = context.WithCancel(context.Background())
				if mode == "setup-deadline" {
					config.ConnectTimeout = 30 * time.Millisecond
				}
			}
			defer cancel()
			result := make(chan error, 1)
			go func() {
				exchange, err := collective.Connect(ctx, config, connectionSpec(1), nil)
				if exchange != nil {
					_ = exchange.Close()
					result <- errors.New("unexpected connection")
					return
				}
				result <- err
			}()
			if mode == "cancel" {
				want = context.Canceled
				time.Sleep(10 * time.Millisecond)
				cancel()
			}
			select {
			case err := <-result:
				if !errors.Is(err, want) {
					t.Fatalf("got %v, want %v", err, want)
				}
			case <-time.After(2 * time.Second):
				t.Fatal("setup ignored cancellation during retry wait")
			}
		})
	}
}

func TestConnectRefusesInvalidInputsBeforeBinding(t *testing.T) {
	config := connectionConfig(t)
	update := func(all [][]float32) ([]float32, error) { return all[0], nil }
	for name, mutate := range map[string]func(*collective.ConnectionConfig){
		"implicit-network":              func(c *collective.ConnectionConfig) { c.Network = "" },
		"wrong-family":                  func(c *collective.ConnectionConfig) { c.Network = "tcp6" },
		"missing-address":               func(c *collective.ConnectionConfig) { c.Address = netip.AddrPort{} },
		"ephemeral-port":                func(c *collective.ConnectionConfig) { c.Address = netip.AddrPortFrom(c.Address.Addr(), 0) },
		"implicit-local":                func(c *collective.ConnectionConfig) { c.LocalAddress = netip.Addr{} },
		"wildcard-local":                func(c *collective.ConnectionConfig) { c.LocalAddress = netip.IPv4Unspecified() },
		"different-listening-interface": func(c *collective.ConnectionConfig) { c.LocalAddress = netip.MustParseAddr("127.0.0.2") },
		"setup-budget":                  func(c *collective.ConnectionConfig) { c.ConnectTimeout = 0 },
		"dial-budget":                   func(c *collective.ConnectionConfig) { c.DialTimeout = 0 },
		"retry-budget":                  func(c *collective.ConnectionConfig) { c.RetryInterval = 0 },
	} {
		t.Run(name, func(t *testing.T) {
			changed := config
			mutate(&changed)
			exchange, err := collective.Connect(context.Background(), changed, connectionSpec(0), update)
			if exchange != nil {
				_ = exchange.Close()
			}
			if exchange != nil || !errors.Is(err, collective.ErrSpec) {
				t.Fatalf("invalid config accepted: %v", err)
			}
		})
	}
	bad := connectionSpec(0)
	bad.Secret = nil
	if _, err := collective.Connect(context.Background(), config, bad, update); !errors.Is(err, collective.ErrSpec) {
		t.Fatal(err)
	}
	if _, err := collective.Connect(context.Background(), config, connectionSpec(0), nil); !errors.Is(err, collective.ErrSpec) {
		t.Fatal(err)
	}
	if _, err := collective.Connect(nil, config, connectionSpec(0), update); !errors.Is(err, collective.ErrSpec) {
		t.Fatal(err)
	}
	coordinator, err := collective.Connect(context.Background(), config, connectionSpec(0), update)
	if err != nil {
		t.Fatalf("invalid setup leaked a listener: %v", err)
	}
	if err := coordinator.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestConnectBindFailureIsRedactedAndCloseReleasesListener(t *testing.T) {
	config := connectionConfig(t)
	update := func(all [][]float32) ([]float32, error) { return all[0], nil }
	coordinator, err := collective.Connect(context.Background(), config, connectionSpec(0), update)
	if err != nil {
		t.Fatal(err)
	}
	defer coordinator.Close()
	other, err := collective.Connect(context.Background(), config, connectionSpec(0), update)
	if other != nil {
		_ = other.Close()
	}
	if other != nil || !errors.Is(err, collective.ErrTransport) || strings.Contains(err.Error(), config.Address.String()) {
		t.Fatalf("bind failure exposed an endpoint or returned a session: %v", err)
	}
	if err := coordinator.Close(); err != nil {
		t.Fatal(err)
	}
	replacement, err := collective.Connect(context.Background(), config, connectionSpec(0), update)
	if err != nil {
		t.Fatalf("closed coordinator retained its socket: %v", err)
	}
	if err := replacement.Close(); err != nil {
		t.Fatal(err)
	}
}
