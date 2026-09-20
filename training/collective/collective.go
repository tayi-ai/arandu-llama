// Package collective gathers ordered FP32 gradients and broadcasts parameters
// over connections supplied by an existing job. It never binds, dials, retries,
// launches processes, or reduces gradients implicitly. HMAC authenticates the
// fixed protocol; transport encryption and endpoint admission belong to callers.
package collective

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"math"
	"net"
	"slices"
	"strings"
	"sync"
	"time"
)

// ProtocolVersion identifies the fixed gradient/parameters/ack/commit protocol.
const ProtocolVersion = "tayi-gradient-exchange-v1"

// Errors classify failures without exposing endpoints, keys, or frame contents.
var (
	ErrSpec       = errors.New("collective: invalid specification")
	ErrBudget     = errors.New("collective: buffer budget exceeded or invalid")
	ErrProtocol   = errors.New("collective: frame metadata mismatch")
	ErrAuth       = errors.New("collective: frame authentication failed")
	ErrNonFinite  = errors.New("collective: nonfinite FP32 value")
	ErrTransport  = errors.New("collective: transport failed")
	ErrClosed     = errors.New("collective: session closed")
	ErrStep       = errors.New("collective: step is not the next admitted step")
	ErrConcurrent = errors.New("collective: overlapping exchanges are forbidden")
	ErrUpdate     = errors.New("collective: update failed or returned invalid parameters")
)

// Spec fixes the session and numeric layout. Rank zero coordinates; other ranks
// connect to it. Steps are numbered 1..Steps. SessionID must never be reused for
// another execution, including after failure. Secret is copied, kept only in
// memory, and must have 32..4096 bytes. It authenticates group membership, not
// individual holders of the same secret.
//
// PhaseTimeout bounds each complete Exchange, including first-step admission.
// The caller's earlier deadline also applies. MaxFrameBytes includes metadata
// and HMAC. MaxBufferedBytes bounds owned numeric/wire buffers, excluding the
// callback's allocations and Go/transport allocator overhead. World is at most
// 256, frames at most 64 MiB, and the buffer budget at most 1 GiB.
type Spec struct {
	World, Rank, Steps, Dimension                 uint32
	SessionID, ContractSHA256, TensorLayoutSHA256 string
	PhaseTimeout                                  time.Duration
	MaxFrameBytes, MaxBufferedBytes               uint64
	Secret                                        []byte
}

// Update receives exactly World independent gradients ordered by rank. It is
// called once only after every gradient is admitted. Inputs must not be retained
// or mutated. Its finite, dimension-matched result is copied before broadcast.
// The callback must terminate: Go cannot interrupt arbitrary callback code.
// A callback failure or later transport failure aborts the session without retry
// and cannot roll back external optimizer state that the callback already changed.
type Update func([][]float32) ([]float32, error)

type configuration struct {
	spec                      Spec
	session, contract, layout [sha256.Size]byte
	payload                   int
}

func configure(spec Spec) (configuration, error) {
	var c configuration
	if spec.World == 0 || spec.World > 256 || spec.Rank >= spec.World || spec.Steps == 0 ||
		spec.Dimension == 0 || spec.PhaseTimeout <= 0 || len(spec.Secret) < 32 || len(spec.Secret) > 4096 ||
		len(spec.SessionID) == 0 || len(spec.SessionID) > 128 {
		return c, ErrSpec
	}
	for _, b := range []byte(spec.SessionID) {
		if b < 33 || b > 126 {
			return c, ErrSpec
		}
	}
	for _, field := range []struct {
		text string
		out  *[sha256.Size]byte
	}{{spec.ContractSHA256, &c.contract}, {spec.TensorLayoutSHA256, &c.layout}} {
		value, err := hex.DecodeString(field.text)
		if err != nil || len(value) != sha256.Size || field.text != strings.ToLower(field.text) {
			return c, ErrSpec
		}
		copy(field.out[:], value)
	}
	payload := uint64(spec.Dimension) * 4
	requiredFrame := payload + headerBytes + sha256.Size
	requiredBuffers := (uint64(spec.World)+2)*payload + uint64(spec.World)*(headerBytes+sha256.Size+chunkBytes)
	if spec.MaxFrameBytes < requiredFrame || spec.MaxFrameBytes > 64<<20 ||
		spec.MaxBufferedBytes < requiredBuffers || spec.MaxBufferedBytes > 1<<30 {
		return c, ErrBudget
	}
	c.spec = spec
	c.spec.Secret = slices.Clone(spec.Secret)
	c.session = sha256.Sum256([]byte(spec.SessionID))
	c.payload = int(payload) // The checked frame ceiling also makes this safe on 32-bit Go.
	return c, nil
}

type session struct {
	c        configuration
	exchange sync.Mutex
	mu       sync.Mutex
	closed   bool
	conns    []net.Conn
	listener net.Listener
	next     uint32
	active   context.CancelFunc
}

func (s *session) close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	conns := slices.Clone(s.conns)
	listener := s.listener
	cancel := s.active
	s.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	if listener != nil {
		_ = listener.Close()
	}
	for _, conn := range conns {
		_ = conn.Close()
	}
	return nil
}

func (s *session) add(conn net.Conn) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		_ = conn.Close()
		return ErrClosed
	}
	s.conns = append(s.conns, conn)
	return nil
}

func (s *session) begin(ctx context.Context, step uint32) (context.Context, func(), error) {
	if ctx == nil {
		return nil, nil, ErrSpec
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	s.mu.Lock()
	closed := s.closed
	s.mu.Unlock()
	if closed {
		return nil, nil, ErrClosed
	}
	if step == 0 || step != s.next || step > s.c.spec.Steps {
		return nil, nil, ErrStep
	}
	phase, cancel := context.WithTimeout(ctx, s.c.spec.PhaseTimeout)
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		cancel()
		return nil, nil, ErrClosed
	}
	s.active = cancel
	s.mu.Unlock()
	stop := context.AfterFunc(phase, func() { _ = s.close() })
	return phase, func() {
		stop()
		s.mu.Lock()
		s.active = nil
		s.mu.Unlock()
		cancel()
	}, nil
}

func finite(values []float32, dimension uint32) error {
	if uint64(len(values)) != uint64(dimension) {
		return ErrProtocol
	}
	for _, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return ErrNonFinite
		}
	}
	return nil
}

func transportError(ctx context.Context, err error) error {
	if err == nil {
		return nil
	}
	if cause := ctx.Err(); cause != nil {
		return cause
	}
	var timeout net.Error
	if errors.As(err, &timeout) && timeout.Timeout() {
		return context.DeadlineExceeded
	}
	return ErrTransport
}

func deadline(ctx context.Context, conn net.Conn) error {
	when, ok := ctx.Deadline()
	if !ok {
		return ErrSpec
	}
	return transportError(ctx, conn.SetDeadline(when))
}

// Coordinator owns a caller-supplied listener and its accepted connections.
// Overlapping Exchange calls abort the session; Close interrupts transport.
type Coordinator struct {
	s        session
	update   Update
	peers    map[uint32]net.Conn
	admitted bool
}

// NewCoordinator validates without performing I/O. On success ownership of the
// listener transfers to the coordinator; on error it remains with the caller.
// The listener and connections must honor Close and connection deadlines.
func NewCoordinator(listener net.Listener, spec Spec, update Update) (*Coordinator, error) {
	if listener == nil || update == nil || spec.Rank != 0 {
		return nil, ErrSpec
	}
	c, err := configure(spec)
	if err != nil {
		return nil, err
	}
	return &Coordinator{s: session{c: c, listener: listener, next: 1}, update: update,
		peers: make(map[uint32]net.Conn, spec.World-1)}, nil
}

// Close permanently aborts the session and closes all owned transports.
func (c *Coordinator) Close() error {
	if c == nil {
		return nil
	}
	return c.s.close()
}

// Exchange gathers this rank-zero gradient and every peer gradient, invokes
// Update once, then sends identical parameter bytes. Every peer acknowledges the
// parameters before commit is sent. A failure closes the session permanently.
// A connection failure during commit can still leave peers with different
// knowledge of completion; the owning job must abort all ranks, never retry.
func (c *Coordinator) Exchange(ctx context.Context, step uint32, gradient []float32) (parameters []float32, err error) {
	if c == nil {
		return nil, ErrSpec
	}
	if !c.s.exchange.TryLock() {
		_ = c.Close()
		return nil, ErrConcurrent
	}
	defer c.s.exchange.Unlock()
	defer func() {
		if err != nil {
			_ = c.Close()
		}
	}()
	phase, done, err := c.s.begin(ctx, step)
	if err != nil {
		return nil, err
	}
	defer done()
	if err = finite(gradient, c.s.c.spec.Dimension); err != nil {
		return nil, err
	}
	ordered := make([][]float32, c.s.c.spec.World)
	ordered[0] = slices.Clone(gradient)
	var connections []rankConnection
	if !c.admitted {
		for i := uint32(1); i < c.s.c.spec.World; i++ {
			conn, acceptErr := c.s.listener.Accept()
			if acceptErr != nil {
				return nil, transportError(phase, acceptErr)
			}
			if err = c.s.add(conn); err != nil {
				return nil, transportError(phase, err)
			}
			connections = append(connections, rankConnection{rank: anyRank, conn: conn})
		}
		_ = c.s.listener.Close() // Admission is one-shot; reconnects cannot join.
	} else {
		for rank := uint32(1); rank < c.s.c.spec.World; rank++ {
			connections = append(connections, rankConnection{rank: rank, conn: c.peers[rank]})
		}
	}
	for _, peer := range connections {
		if err = deadline(phase, peer.conn); err != nil {
			return nil, err
		}
	}
	results := make(chan received, len(connections))
	for _, peer := range connections {
		go func(peer rankConnection) {
			frame, readErr := c.s.c.read(phase, peer.conn, gradientFrame, peer.rank, step)
			results <- received{rankConnection: rankConnection{rank: frame.rank, conn: peer.conn}, values: frame.values, err: readErr}
		}(peer)
	}
	var failure error
	for range connections {
		result := <-results
		if result.err == nil && (result.rank == 0 || result.rank >= c.s.c.spec.World || ordered[result.rank] != nil) {
			result.err = ErrProtocol
		}
		if result.err != nil {
			if failure == nil {
				failure = result.err
				_ = c.Close()
			}
			continue
		}
		ordered[result.rank] = result.values
		c.peers[result.rank] = result.conn
	}
	if failure != nil {
		return nil, failure
	}
	c.admitted = true
	if err = phase.Err(); err != nil {
		return nil, err
	}
	values, updateErr := invoke(c.update, ordered)
	if updateErr != nil || finite(values, c.s.c.spec.Dimension) != nil {
		return nil, ErrUpdate
	}
	if err = phase.Err(); err != nil {
		return nil, err
	}
	parameters = slices.Clone(values)
	payload := encode(parameters)
	if err = c.parallel(phase, func(rank uint32, conn net.Conn) error {
		if err := c.s.c.write(phase, conn, parametersFrame, 0, step, payload); err != nil {
			return err
		}
		_, err := c.s.c.read(phase, conn, ackFrame, rank, step)
		return err
	}); err != nil {
		return nil, err
	}
	if err = c.parallel(phase, func(_ uint32, conn net.Conn) error {
		return c.s.c.write(phase, conn, commitFrame, 0, step, nil)
	}); err != nil {
		return nil, err
	}
	if err = phase.Err(); err != nil {
		return nil, err
	}
	if step == c.s.c.spec.Steps {
		_ = c.Close()
	} else {
		c.s.next++
	}
	return parameters, nil
}

func invoke(update Update, gradients [][]float32) (values []float32, err error) {
	defer func() {
		if recover() != nil {
			values, err = nil, ErrUpdate
		}
	}()
	return update(gradients)
}

type rankConnection struct {
	rank uint32
	conn net.Conn
}
type received struct {
	rankConnection
	values []float32
	err    error
}

func (c *Coordinator) parallel(ctx context.Context, operation func(uint32, net.Conn) error) error {
	results := make(chan error, len(c.peers))
	for rank, conn := range c.peers {
		go func(rank uint32, conn net.Conn) { results <- operation(rank, conn) }(rank, conn)
	}
	var failure error
	for range c.peers {
		if err := <-results; err != nil && failure == nil {
			failure = err
			_ = c.Close()
		}
	}
	if failure != nil {
		return failure
	}
	return ctx.Err()
}

// Peer owns a caller-supplied connection to rank zero. It neither connects nor
// reconnects. Overlapping Exchange calls abort; Close interrupts transport.
type Peer struct {
	s    session
	conn net.Conn
}

// NewPeer validates without I/O and transfers connection ownership on success.
// On error ownership remains with the caller. The connection must honor Close
// and deadlines. The caller must not read or write it after ownership transfers.
func NewPeer(conn net.Conn, spec Spec) (*Peer, error) {
	if conn == nil || spec.Rank == 0 {
		return nil, ErrSpec
	}
	c, err := configure(spec)
	if err != nil {
		return nil, err
	}
	return &Peer{s: session{c: c, conns: []net.Conn{conn}, next: 1}, conn: conn}, nil
}

// Close permanently aborts the peer and closes its connection.
func (p *Peer) Close() error {
	if p == nil {
		return nil
	}
	return p.s.close()
}

// Exchange sends a finite gradient and returns owned FP32 parameters only after
// an authenticated commit. It does not mutate or retain the supplied gradient.
func (p *Peer) Exchange(ctx context.Context, step uint32, gradient []float32) (parameters []float32, err error) {
	if p == nil {
		return nil, ErrSpec
	}
	if !p.s.exchange.TryLock() {
		_ = p.Close()
		return nil, ErrConcurrent
	}
	defer p.s.exchange.Unlock()
	defer func() {
		if err != nil {
			_ = p.Close()
		}
	}()
	phase, done, err := p.s.begin(ctx, step)
	if err != nil {
		return nil, err
	}
	defer done()
	if err = finite(gradient, p.s.c.spec.Dimension); err != nil {
		return nil, err
	}
	if err = deadline(phase, p.conn); err != nil {
		return nil, err
	}
	if err = p.s.c.write(phase, p.conn, gradientFrame, p.s.c.spec.Rank, step, encode(gradient)); err != nil {
		return nil, err
	}
	frame, err := p.s.c.read(phase, p.conn, parametersFrame, 0, step)
	if err != nil {
		return nil, err
	}
	if err = p.s.c.write(phase, p.conn, ackFrame, p.s.c.spec.Rank, step, nil); err != nil {
		return nil, err
	}
	if _, err = p.s.c.read(phase, p.conn, commitFrame, 0, step); err != nil {
		return nil, err
	}
	if err = phase.Err(); err != nil {
		return nil, err
	}
	if step == p.s.c.spec.Steps {
		_ = p.Close()
	} else {
		p.s.next++
	}
	return frame.values, nil
}
