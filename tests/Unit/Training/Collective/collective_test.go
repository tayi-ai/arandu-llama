package collective_test

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"io"
	"math"
	"net"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/tayi-ai/arandu-llama/training/collective"
)

type pipeAddress struct{}

func (pipeAddress) Network() string { return "pipe" }
func (pipeAddress) String() string  { return "collective-test" }

type pipeListener struct {
	connections chan net.Conn
	closed      chan struct{}
	accepting   chan struct{}
	once        sync.Once
	acceptOnce  sync.Once
}

func newListener() *pipeListener {
	return &pipeListener{connections: make(chan net.Conn, 256), closed: make(chan struct{}), accepting: make(chan struct{})}
}
func (l *pipeListener) Accept() (net.Conn, error) {
	l.acceptOnce.Do(func() { close(l.accepting) })
	select {
	case conn := <-l.connections:
		return conn, nil
	case <-l.closed:
		return nil, net.ErrClosed
	}
}
func (l *pipeListener) Close() error   { l.once.Do(func() { close(l.closed) }); return nil }
func (l *pipeListener) Addr() net.Addr { return pipeAddress{} }

func specification(world, rank, dimension, steps uint32) collective.Spec {
	return collective.Spec{World: world, Rank: rank, Dimension: dimension, Steps: steps,
		SessionID: "fixture-session-once", ContractSHA256: strings.Repeat("a", 64), TensorLayoutSHA256: strings.Repeat("b", 64),
		PhaseTimeout: 2 * time.Second, MaxFrameBytes: 1 << 20, MaxBufferedBytes: 32 << 20, Secret: bytes.Repeat([]byte{0x83}, 32)}
}

func peer(t *testing.T, listener *pipeListener, spec collective.Spec) *collective.Peer {
	t.Helper()
	server, client := net.Pipe()
	listener.connections <- server
	p, err := collective.NewPeer(client, spec)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = p.Close(); _ = server.Close() })
	return p
}

func sameBits(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return false
		}
	}
	return true
}

func TestTwentyOneRanksOrderedAndBitwiseBroadcastAcrossSteps(t *testing.T) {
	const world, dimension, steps = 21, 1031, 2 // Exercise multiple wire chunks.
	listener := newListener()
	gradients := func(rank, step uint32) []float32 {
		values := make([]float32, dimension)
		for i := range values {
			values[i] = float32(rank*1000+step*100+uint32(i)) / 7
		}
		return values
	}
	var calls atomic.Int32
	coordinator, err := collective.NewCoordinator(listener, specification(world, 0, dimension, steps), func(all [][]float32) ([]float32, error) {
		step := uint32(calls.Add(1))
		if len(all) != world {
			t.Errorf("got %d ranks", len(all))
		}
		result := make([]float32, dimension)
		for rank, values := range all {
			if !sameBits(values, gradients(uint32(rank), step)) {
				t.Errorf("rank %d gradient was not preserved at step %d", rank, step)
			}
			for i, value := range values {
				result[i] += value // This callback, and only this callback, reduces.
			}
		}
		result[0], result[1], result[2] = math.Float32frombits(0x80000000), math.SmallestNonzeroFloat32, math.MaxFloat32
		return result, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = coordinator.Close() })
	type result struct {
		rank, step uint32
		values     []float32
		err        error
	}
	results := make(chan result, (world-1)*steps)
	// Deliberately admit connections in descending rank order.
	for rank := uint32(world - 1); rank > 0; rank-- {
		p := peer(t, listener, specification(world, rank, dimension, steps))
		go func(rank uint32) {
			for step := uint32(1); step <= steps; step++ {
				input := gradients(rank, step)
				before := slices.Clone(input)
				values, err := p.Exchange(context.Background(), step, input)
				if !sameBits(input, before) {
					t.Error("peer gradient mutated")
				}
				results <- result{rank, step, values, err}
				if err != nil {
					return
				}
			}
		}(rank)
	}
	outputs := make(map[uint32][]float32)
	for step := uint32(1); step <= steps; step++ {
		input := gradients(0, step)
		before := slices.Clone(input)
		values, err := coordinator.Exchange(context.Background(), step, input)
		if err != nil {
			t.Fatal(err)
		}
		if !sameBits(input, before) {
			t.Fatal("rank zero gradient mutated")
		}
		outputs[step] = values
	}
	for range (world - 1) * steps {
		r := <-results
		if r.err != nil || !sameBits(r.values, outputs[r.step]) {
			t.Fatalf("rank %d step %d: %v; bitwise broadcast mismatch", r.rank, r.step, r.err)
		}
		r.values[0] = 1
		if math.Float32bits(outputs[r.step][0]) != 0x80000000 {
			t.Fatal("results alias")
		}
	}
	if calls.Load() != steps {
		t.Fatalf("update calls = %d", calls.Load())
	}
	if _, err := coordinator.Exchange(context.Background(), steps+1, gradients(0, 1)); !errors.Is(err, collective.ErrClosed) {
		t.Fatalf("completed session reused: %v", err)
	}
}

// Independent frame construction permits malformed protocol tests through the
// public API. This is not an alternate transport implementation.
func wire(spec collective.Spec, kind, rank, step uint32, values []float32) []byte {
	const headerSize = 132
	body := make([]byte, headerSize+len(values)*4)
	copy(body, "TAYIGRD1")
	for i, value := range []uint32{kind, spec.World, rank, step, spec.Steps, spec.Dimension, uint32(len(values) * 4)} {
		binary.LittleEndian.PutUint32(body[8+i*4:], value)
	}
	session := sha256.Sum256([]byte(spec.SessionID))
	copy(body[36:68], session[:])
	contract, _ := hex.DecodeString(spec.ContractSHA256)
	layout, _ := hex.DecodeString(spec.TensorLayoutSHA256)
	copy(body[68:100], contract)
	copy(body[100:132], layout)
	for i, value := range values {
		binary.LittleEndian.PutUint32(body[headerSize+i*4:], math.Float32bits(value))
	}
	auth := hmac.New(sha256.New, spec.Secret)
	_, _ = auth.Write(body)
	return append(body, auth.Sum(nil)...)
}

func TestFailedAdmissionNeverCallsUpdate(t *testing.T) {
	for _, name := range []string{"bad HMAC", "wrong secret", "session", "contract", "layout", "world", "steps", "dimension", "stale step", "rank zero", "rank out of range", "length before allocation", "nonfinite", "truncated"} {
		t.Run(name, func(t *testing.T) {
			listener := newListener()
			spec := specification(2, 0, 2, 2)
			var calls atomic.Int32
			c, err := collective.NewCoordinator(listener, spec, func([][]float32) ([]float32, error) { calls.Add(1); return []float32{1, 2}, nil })
			if err != nil {
				t.Fatal(err)
			}
			defer c.Close()
			server, client := net.Pipe()
			listener.connections <- server
			defer client.Close()
			remote := spec
			rank, step := uint32(1), uint32(1)
			values := []float32{1, 2}
			want := collective.ErrProtocol
			switch name {
			case "wrong secret":
				remote.Secret = bytes.Repeat([]byte{1}, 32)
				want = collective.ErrAuth
			case "session":
				remote.SessionID = "other-session"
			case "contract":
				remote.ContractSHA256 = strings.Repeat("c", 64)
			case "layout":
				remote.TensorLayoutSHA256 = strings.Repeat("c", 64)
			case "world":
				remote.World++
			case "steps":
				remote.Steps++
			case "dimension":
				remote.Dimension++
			case "stale step":
				step = 2
			case "rank zero":
				rank = 0
			case "rank out of range":
				rank = 2
			case "nonfinite":
				values[1] = float32(math.Inf(1))
				want = collective.ErrNonFinite
			case "truncated":
				want = collective.ErrTransport
			case "bad HMAC":
				want = collective.ErrAuth
			}
			packet := wire(remote, 1, rank, step, values)
			if name == "bad HMAC" {
				packet[len(packet)-1] ^= 1
			}
			if name == "length before allocation" {
				binary.LittleEndian.PutUint32(packet[32:36], math.MaxUint32)
				packet = packet[:132]
			}
			if name == "truncated" {
				packet = packet[:140]
			}
			written := make(chan struct{})
			go func() { defer close(written); _, _ = client.Write(packet); _ = client.Close() }()
			result, err := c.Exchange(context.Background(), 1, []float32{3, 4})
			<-written
			if !errors.Is(err, want) || result != nil || calls.Load() != 0 {
				t.Fatalf("error=%v want=%v result=%v calls=%d", err, want, result, calls.Load())
			}
			if _, err := c.Exchange(context.Background(), 1, []float32{3, 4}); !errors.Is(err, collective.ErrClosed) {
				t.Fatalf("failed session reused: %v", err)
			}
		})
	}
}

func TestDuplicateRankAbortsAllPeersBeforeUpdate(t *testing.T) {
	listener := newListener()
	var calls atomic.Int32
	c, err := collective.NewCoordinator(listener, specification(3, 0, 2, 1), func([][]float32) ([]float32, error) { calls.Add(1); return []float32{1, 2}, nil })
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	results := make(chan error, 2)
	for range 2 {
		p := peer(t, listener, specification(3, 1, 2, 1))
		go func() { _, err := p.Exchange(context.Background(), 1, []float32{1, 2}); results <- err }()
	}
	if _, err := c.Exchange(context.Background(), 1, []float32{1, 2}); !errors.Is(err, collective.ErrProtocol) {
		t.Fatal(err)
	}
	for range 2 {
		if err := <-results; err == nil {
			t.Fatal("peer accepted duplicate rank")
		}
	}
	if calls.Load() != 0 {
		t.Fatal("update ran before unique-rank admission")
	}
}

func TestCancellationAndMandatoryPhaseDeadline(t *testing.T) {
	for _, mode := range []string{"cancel admission", "deadline admission", "cancel gradient", "deadline peer"} {
		t.Run(mode, func(t *testing.T) {
			spec := specification(2, 0, 2, 1)
			spec.PhaseTimeout = 40 * time.Millisecond
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			var calls atomic.Int32
			if mode == "deadline peer" {
				server, client := net.Pipe()
				defer server.Close()
				spec.Rank = 1
				p, err := collective.NewPeer(client, spec)
				if err != nil {
					t.Fatal(err)
				}
				defer p.Close()
				if _, err := p.Exchange(ctx, 1, []float32{1, 2}); !errors.Is(err, context.DeadlineExceeded) {
					t.Fatal(err)
				}
				return
			}
			listener := newListener()
			c, err := collective.NewCoordinator(listener, spec, func([][]float32) ([]float32, error) { calls.Add(1); return []float32{1, 2}, nil })
			if err != nil {
				t.Fatal(err)
			}
			defer c.Close()
			if mode == "cancel gradient" {
				server, client := net.Pipe()
				listener.connections <- server
				defer client.Close()
			}
			if strings.HasPrefix(mode, "cancel") {
				go func() { <-listener.accepting; cancel() }()
			}
			_, err = c.Exchange(ctx, 1, []float32{1, 2})
			want := context.DeadlineExceeded
			if strings.HasPrefix(mode, "cancel") {
				want = context.Canceled
			}
			if !errors.Is(err, want) || calls.Load() != 0 {
				t.Fatalf("error=%v calls=%d", err, calls.Load())
			}
		})
	}
}

func TestReplayOnEstablishedConnectionCannotUpdateAgain(t *testing.T) {
	listener := newListener()
	spec := specification(2, 0, 2, 2)
	var calls atomic.Int32
	c, err := collective.NewCoordinator(listener, spec, func([][]float32) ([]float32, error) { calls.Add(1); return []float32{4, 5}, nil })
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	server, client := net.Pipe()
	listener.connections <- server
	defer client.Close()
	done := make(chan error, 1)
	go func() {
		packet := wire(spec, 1, 1, 1, []float32{1, 2})
		if _, err := client.Write(packet); err != nil {
			done <- err
			return
		}
		if _, err := io.CopyN(io.Discard, client, 132+8+32); err != nil {
			done <- err
			return
		}
		if _, err := client.Write(wire(spec, 3, 1, 1, nil)); err != nil {
			done <- err
			return
		}
		if _, err := io.CopyN(io.Discard, client, 132+32); err != nil {
			done <- err
			return
		}
		_, err := client.Write(packet) // Identical, valid-HMAC replay of step one.
		done <- err
	}()
	if _, err := c.Exchange(context.Background(), 1, []float32{1, 2}); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Exchange(context.Background(), 2, []float32{1, 2}); !errors.Is(err, collective.ErrProtocol) {
		t.Fatal(err)
	}
	<-done
	if calls.Load() != 1 {
		t.Fatalf("replay produced %d updates", calls.Load())
	}
}

func TestSpecificationAndLocalGradientValidation(t *testing.T) {
	for _, mutate := range []func(*collective.Spec){
		func(s *collective.Spec) { s.World = 0 }, func(s *collective.Spec) { s.World = 257 },
		func(s *collective.Spec) { s.Dimension = math.MaxUint32 }, func(s *collective.Spec) { s.Steps = 0 },
		func(s *collective.Spec) { s.PhaseTimeout = 0 }, func(s *collective.Spec) { s.Secret = make([]byte, 31) },
		func(s *collective.Spec) { s.ContractSHA256 = strings.Repeat("A", 64) }, func(s *collective.Spec) { s.TensorLayoutSHA256 = "bad" },
		func(s *collective.Spec) { s.SessionID = "bad session" }, func(s *collective.Spec) { s.MaxFrameBytes = 1 },
		func(s *collective.Spec) { s.MaxBufferedBytes = 1 },
	} {
		listener := newListener()
		spec := specification(2, 0, 2, 1)
		mutate(&spec)
		if _, err := collective.NewCoordinator(listener, spec, func([][]float32) ([]float32, error) { return nil, nil }); err == nil {
			t.Fatal("invalid specification admitted")
		}
		select {
		case <-listener.closed:
			t.Fatal("failed constructor took ownership")
		default:
		}
		_ = listener.Close()
	}
	for _, input := range [][]float32{nil, {1}, {1, float32(math.NaN())}, {1, float32(math.Inf(-1))}} {
		listener := newListener()
		var calls atomic.Int32
		c, err := collective.NewCoordinator(listener, specification(1, 0, 2, 1), func([][]float32) ([]float32, error) { calls.Add(1); return []float32{1, 2}, nil })
		if err != nil {
			t.Fatal(err)
		}
		if _, err := c.Exchange(context.Background(), 1, input); err == nil || calls.Load() != 0 {
			t.Fatalf("invalid input: %v", err)
		}
	}
}

func TestUpdateErrorPanicAndInvalidResultAbortPeer(t *testing.T) {
	for _, mode := range []string{"error", "panic", "shape", "nonfinite"} {
		t.Run(mode, func(t *testing.T) {
			listener := newListener()
			var calls atomic.Int32
			c, err := collective.NewCoordinator(listener, specification(2, 0, 2, 1), func([][]float32) ([]float32, error) {
				calls.Add(1)
				switch mode {
				case "error":
					return nil, errors.New("private callback detail")
				case "panic":
					panic("private callback detail")
				case "shape":
					return []float32{1}, nil
				default:
					return []float32{1, float32(math.NaN())}, nil
				}
			})
			if err != nil {
				t.Fatal(err)
			}
			defer c.Close()
			p := peer(t, listener, specification(2, 1, 2, 1))
			result := make(chan error, 1)
			go func() { _, err := p.Exchange(context.Background(), 1, []float32{1, 2}); result <- err }()
			if values, err := c.Exchange(context.Background(), 1, []float32{1, 2}); values != nil || !errors.Is(err, collective.ErrUpdate) {
				t.Fatal(err)
			}
			if err := <-result; err == nil || calls.Load() != 1 {
				t.Fatalf("peer error=%v calls=%d", err, calls.Load())
			}
		})
	}
}

func TestPeerAuthenticatesParametersAndCommit(t *testing.T) {
	for _, mode := range []string{"parameters HMAC", "parameters nonfinite", "commit HMAC", "commit step"} {
		t.Run(mode, func(t *testing.T) {
			server, client := net.Pipe()
			defer server.Close()
			spec := specification(2, 1, 2, 1)
			p, err := collective.NewPeer(client, spec)
			if err != nil {
				t.Fatal(err)
			}
			defer p.Close()
			done := make(chan struct{})
			go func() {
				defer close(done)
				if _, err := io.CopyN(io.Discard, server, 132+8+32); err != nil {
					return
				}
				values := []float32{4, 5}
				if mode == "parameters nonfinite" {
					values[0] = float32(math.Inf(1))
				}
				packet := wire(spec, 2, 0, 1, values)
				if mode == "parameters HMAC" {
					packet[len(packet)-1] ^= 1
				}
				if _, err := server.Write(packet); err != nil {
					return
				}
				if strings.HasPrefix(mode, "parameters") {
					return
				}
				if _, err := io.CopyN(io.Discard, server, 132+32); err != nil {
					return
				}
				step := uint32(1)
				if mode == "commit step" {
					step = 2
				}
				packet = wire(spec, 4, 0, step, nil)
				if mode == "commit HMAC" {
					packet[len(packet)-1] ^= 1
				}
				_, _ = server.Write(packet)
			}()
			values, err := p.Exchange(context.Background(), 1, []float32{1, 2})
			<-done
			want := collective.ErrAuth
			if mode == "parameters nonfinite" {
				want = collective.ErrNonFinite
			}
			if mode == "commit step" {
				want = collective.ErrProtocol
			}
			if values != nil || !errors.Is(err, want) {
				t.Fatalf("error=%v want=%v", err, want)
			}
		})
	}
}

func TestBadAcknowledgementPreventsCommit(t *testing.T) {
	listener := newListener()
	spec := specification(2, 0, 2, 1)
	var calls atomic.Int32
	c, err := collective.NewCoordinator(listener, spec, func([][]float32) ([]float32, error) { calls.Add(1); return []float32{4, 5}, nil })
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	server, client := net.Pipe()
	listener.connections <- server
	defer client.Close()
	done := make(chan error, 1)
	go func() {
		if _, err := client.Write(wire(spec, 1, 1, 1, []float32{1, 2})); err != nil {
			done <- err
			return
		}
		if _, err := io.CopyN(io.Discard, client, 132+8+32); err != nil {
			done <- err
			return
		}
		packet := wire(spec, 3, 1, 1, nil)
		packet[len(packet)-1] ^= 1
		if _, err := client.Write(packet); err != nil {
			done <- err
			return
		}
		var next [1]byte
		_, err := client.Read(next[:])
		done <- err
	}()
	if values, err := c.Exchange(context.Background(), 1, []float32{1, 2}); values != nil || !errors.Is(err, collective.ErrAuth) {
		t.Fatal(err)
	}
	if err := <-done; !errors.Is(err, io.EOF) {
		t.Fatalf("commit sent after invalid ACK: %v", err)
	}
	if calls.Load() != 1 {
		t.Fatalf("update calls=%d", calls.Load())
	}
}

func TestConcurrentExchangeAbortsWithoutWaitingForAnotherPhase(t *testing.T) {
	listener := newListener()
	var calls atomic.Int32
	c, err := collective.NewCoordinator(listener, specification(2, 0, 2, 1), func([][]float32) ([]float32, error) { calls.Add(1); return []float32{1, 2}, nil })
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	done := make(chan error, 1)
	go func() { _, err := c.Exchange(context.Background(), 1, []float32{1, 2}); done <- err }()
	<-listener.accepting
	if _, err := c.Exchange(context.Background(), 1, []float32{1, 2}); !errors.Is(err, collective.ErrConcurrent) {
		t.Fatal(err)
	}
	if err := <-done; !errors.Is(err, context.Canceled) || calls.Load() != 0 {
		t.Fatalf("active exchange=%v calls=%d", err, calls.Load())
	}
}

func TestCloseDuringUpdateDoesNotPublishSuccess(t *testing.T) {
	listener := newListener()
	started, release := make(chan struct{}), make(chan struct{})
	c, err := collective.NewCoordinator(listener, specification(1, 0, 2, 1), func([][]float32) ([]float32, error) {
		close(started)
		<-release
		return []float32{4, 5}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	done := make(chan error, 1)
	go func() { _, err := c.Exchange(context.Background(), 1, []float32{1, 2}); done <- err }()
	<-started
	_ = c.Close()
	close(release)
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}

func TestConstructorsCopySecretAndAdmitProductionGeometry(t *testing.T) {
	listener := newListener()
	spec := specification(2, 0, 2, 1)
	c, err := collective.NewCoordinator(listener, spec, func([][]float32) ([]float32, error) { return []float32{4, 5}, nil })
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	spec.Rank = 1
	p := peer(t, listener, spec)
	clear(spec.Secret)
	done := make(chan error, 1)
	go func() { _, err := p.Exchange(context.Background(), 1, []float32{1, 2}); done <- err }()
	if _, err := c.Exchange(context.Background(), 1, []float32{1, 2}); err != nil {
		t.Fatal(err)
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	production := specification(21, 0, 557056, 18)
	production.MaxFrameBytes, production.MaxBufferedBytes = 4<<20, 64<<20
	production.PhaseTimeout = 73 * time.Second // There is no implicit fixed 300s phase.
	admitted, err := collective.NewCoordinator(newListener(), production, func([][]float32) ([]float32, error) { t.Fatal("constructor invoked update"); return nil, nil })
	if err != nil {
		t.Fatal(err)
	}
	_ = admitted.Close()
}
