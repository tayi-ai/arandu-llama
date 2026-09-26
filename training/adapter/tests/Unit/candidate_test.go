package unit_test

import (
	"github.com/tayi-ai/arandu-llama/training/zerothorder"

	"context"
	"errors"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	services "github.com/tayi-ai/arandu-llama/training/adapter"
	tests "github.com/tayi-ai/arandu-llama/training/adapter/tests/Fixtures"
)

// The candidate contract of the T02 addendum, proven through the methods
// zo:probe and zo:train actually call: zerothorder.Step, Snapshot.Perturb,
// Snapshot.Fold and BuildCandidate. The only thing faked is the C boundary --
// what llama.cpp does with a loaded LoRA -- and the fake reads the GGUF file
// the production code wrote, through the production reader.
//
// The scalar oracle: A = 2, B = 3, ZA = 5, ZB = -7, unit LoRA scale, eps =
// 0.001. A candidate built in factor space gives (3 - 7c)(2 + 5c) = 6 + c -
// 35c^2, so the probes read 6.000965 and 5.998965 and the central difference
// is 1. A probe that rescales the direction as a second adapter gives
// 6 - 35c, and the central difference is -35. A tolerance wide enough to hide
// 1 against -35 fails TestASecondAdapterProbeFailsTheOracle by construction.

// oracleTolerance is how far a gradient may sit from the oracle's derivative.
// The scalar candidate is rounded to float32 once per factor, which was
// measured to move the central difference by 2.3e-4; the number this has to
// tell apart is 36 away.
const oracleTolerance = 1e-2

// matrix is a row-major LoRA factor read back from a GGUF: ne[0] is the
// column count, ne[1] the row count, which is how ggml lays a 2-D tensor out.
type matrix struct {
	rows, cols int
	data       []float64
}

func (m matrix) at(row, col int) float64 { return m.data[row*m.cols+col] }

// factorHandle is what factorHost loaded: the bytes and digest of one adapter
// file and its factors decoded by name.
type factorHandle struct {
	digest string
	bytes  []byte
	a, b   map[string]matrix
	closed bool
}

func (h *factorHandle) Digest() string { return h.digest }

func (h *factorHandle) Close() error {
	h.closed = true
	return nil
}

// output is what the adapter adds to the model's output for x = ones(n_in),
// per module in name order: alpha/rank times B(Ax). This is the composition
// llama-adapter.cpp applies, with lora_a as [n_in, r] and lora_b as [r, n_out].
func (h *factorHandle) output(alpha float64) []float64 {
	modules := make([]string, 0, len(h.a))
	for name := range h.a {
		modules = append(modules, name)
	}
	sort.Strings(modules)
	var y []float64
	for _, name := range modules {
		a, b := h.a[name], h.b[name]
		scale := alpha / float64(a.rows)
		for o := 0; o < b.rows; o++ {
			sum := 0.0
			for k := 0; k < b.cols; k++ {
				for i := 0; i < a.cols; i++ {
					sum += b.at(o, k) * a.at(k, i)
				}
			}
			y = append(y, scale*sum)
		}
	}
	return y
}

// factorHost is the AdapterHost of the tests: it loads a file the way the
// binding would -- by reading it -- and records what it was asked to apply.
type factorHost struct {
	alpha float64
	// loads and sets count the calls, so a test can make the Nth one fail.
	loads, sets           int
	failLoadAt, failSetAt int
	errLoad, errSet       error
	// tamperFrom makes every load from the Nth on report a digest that is not
	// the file's, which is what a host that loaded the wrong bytes looks like.
	tamperFrom int
	applied    []services.AdapterHandle
	scales     []float32
	loaded     []*factorHandle
}

func (h *factorHost) LoadAdapter(path string) (services.AdapterHandle, error) {
	h.loads++
	if h.loads == h.failLoadAt {
		return nil, h.errLoad
	}
	body, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	container, err := services.ReadGGUF(path)
	if err != nil {
		return nil, err
	}
	handle := &factorHandle{digest: services.Digest(body), bytes: body, a: map[string]matrix{}, b: map[string]matrix{}}
	if h.tamperFrom > 0 && h.loads >= h.tamperFrom {
		handle.digest = "not what was written"
	}
	for _, tensor := range container.Tensors {
		if len(tensor.Dims) != 2 {
			return nil, errors.New("a LoRA factor has two dimensions")
		}
		values := make([]float64, tensor.Elements)
		for i := range values {
			at := container.DataOffset + tensor.Offset + int64(i)*4
			values[i] = float64(math.Float32frombits(uint32(body[at]) | uint32(body[at+1])<<8 | uint32(body[at+2])<<16 | uint32(body[at+3])<<24))
		}
		m := matrix{rows: int(tensor.Dims[1]), cols: int(tensor.Dims[0]), data: values}
		switch {
		case strings.HasSuffix(tensor.Name, ".lora_a"):
			handle.a[strings.TrimSuffix(tensor.Name, ".lora_a")] = m
		case strings.HasSuffix(tensor.Name, ".lora_b"):
			handle.b[strings.TrimSuffix(tensor.Name, ".lora_b")] = m
		}
	}
	h.loaded = append(h.loaded, handle)
	return handle, nil
}

func (h *factorHost) SetAdapters(handles []services.AdapterHandle, scales []float32) error {
	h.sets++
	if h.sets == h.failSetAt {
		return h.errSet
	}
	for _, handle := range handles {
		if handle.(*factorHandle).closed {
			return errors.New("a closed adapter was applied; on the card this reads released memory")
		}
	}
	h.applied = append([]services.AdapterHandle(nil), handles...)
	h.scales = append([]float32(nil), scales...)
	return nil
}

// factorEngine is the Engine zo:probe and zo:train use, with the card
// replaced by factorHost: Perturb is Snapshot.Perturb itself, and Score sums
// scale_i times the output of every applied handle.
type factorEngine struct {
	*services.Snapshot
	host    *factorHost
	loss    func(y []float64) float64
	onScore func()
	ctx     context.Context
}

func (e *factorEngine) Score(_ context.Context, _ zerothorder.Example) (zerothorder.Reading, error) {
	if e.onScore != nil {
		e.onScore()
	}
	if e.ctx != nil && e.ctx.Err() != nil {
		return zerothorder.Reading{}, e.ctx.Err()
	}
	var y []float64
	for i, handle := range e.host.applied {
		out := handle.(*factorHandle).output(e.host.alpha)
		if y == nil {
			y = make([]float64, len(out))
		}
		for k := range out {
			y[k] += float64(e.host.scales[i]) * out[k]
		}
	}
	return zerothorder.Reading{Loss: e.loss(y), Tokens: 1}, nil
}

// secondAdapterEngine is the engine before the fix, as a negative control: the
// policy at scale one and the direction loaded as a second adapter at scale c,
// summed in product space the way llama.cpp sums adapters. It is what
// EngineLlama.go did until T02, and it has to fail the oracle.
type secondAdapterEngine struct {
	snapshot *services.Snapshot
	z        *services.Displacement
	alpha    float64
	c        float64
	loss     func(y []float64) float64
}

func (e *secondAdapterEngine) Perturb(_ context.Context, d services.Direction, sign float64) error {
	e.c = sign * d.Scale
	return nil
}

func (e *secondAdapterEngine) Score(context.Context, zerothorder.Example) (zerothorder.Reading, error) {
	a, _ := e.snapshot.Values("blk.0.attn_q.weight.lora_a")
	b, _ := e.snapshot.Values("blk.0.attn_q.weight.lora_b")
	y := e.alpha*float64(b[0])*float64(a[0]) + e.c*e.alpha*float64(e.z.Tensors[1][0])*float64(e.z.Tensors[0][0])
	return zerothorder.Reading{Loss: e.loss([]float64{y}), Tokens: 1}, nil
}

const (
	loraA = "blk.0.attn_q.weight.lora_a"
	loraB = "blk.0.attn_q.weight.lora_b"
)

// scalarFixture is the oracle's adapter: A = 2, B = 3, alpha 1 over rank 1.
func scalarFixture(t *testing.T) (policy, scratch string) {
	t.Helper()
	dir := t.TempDir()
	policy = filepath.Join(dir, "policy.gguf")
	scratch = filepath.Join(dir, "scratch")
	tests.WriteLoRAGGUF(t, policy, "oracle", 1, []tests.LoRATensor{
		{Name: loraA, Dims: []uint64{1, 1}, Data: []float32{2}},
		{Name: loraB, Dims: []uint64{1, 1}, Data: []float32{3}},
	})
	return policy, scratch
}

// oracleDisplacement is ZA = 5, ZB = -7 over the scalar fixture.
func oracleDisplacement(seed uint64) *services.Displacement {
	return &services.Displacement{Direction: services.Direction{Seed: seed}, Tensors: [][]float32{{5}, {-7}}}
}

// matrixFixture is rectangular and non-unit: n_in = 3, r = 2, n_out = 2, alpha
// 4 so the scale is 2. Small distinct integers, so a transposition, a swap of
// A and B or a factor left unperturbed each give a different number.
func matrixFixture(t *testing.T) (policy, scratch string) {
	t.Helper()
	dir := t.TempDir()
	policy = filepath.Join(dir, "policy.gguf")
	scratch = filepath.Join(dir, "scratch")
	tests.WriteLoRAGGUF(t, policy, "oracle", 4, []tests.LoRATensor{
		{Name: loraA, Dims: []uint64{3, 2}, Data: []float32{1, 2, 3, 4, 5, 6}},
		{Name: loraB, Dims: []uint64{2, 2}, Data: []float32{1, 0, 2, 1}},
	})
	return policy, scratch
}

func matrixDisplacement(seed uint64) *services.Displacement {
	return &services.Displacement{Direction: services.Direction{Seed: seed}, Tensors: [][]float32{{1, -1, 2, 0, 1, 3}, {0.5, 1, -1, 2}}}
}

func open(t *testing.T, host *factorHost, policy, scratch string) *services.Snapshot {
	t.Helper()
	s, err := services.OpenSnapshot(host, policy, scratch)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

func sumOf(y []float64) float64 {
	total := 0.0
	for _, v := range y {
		total += v
	}
	return total
}

func fileDigest(t *testing.T, path string) string {
	t.Helper()
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return services.Digest(body)
}

func leftovers(t *testing.T, dir, prefix string) []string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		t.Fatal(err)
	}
	var found []string
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), prefix) {
			found = append(found, e.Name())
		}
	}
	return found
}

func TestTheProbeAndTheStepBuildTheSameCandidate(t *testing.T) {
	policy, scratch := scalarFixture(t)
	host := &factorHost{alpha: 1}
	s := open(t, host, policy, scratch)
	s.Use(oracleDisplacement(1))
	engine := &factorEngine{Snapshot: s, host: host, loss: func(y []float64) float64 { return y[0] }}
	ctx := context.Background()

	estimate, err := zerothorder.Step(ctx, engine, zerothorder.Example{ID: "oracle"}, services.Direction{Seed: 1, Scale: 0.001})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(estimate.Plus.Loss-6.000965) > 1e-5 {
		t.Errorf("the positive probe read %.9f, want 6.000965 = (3 - 0.007)(2 + 0.005)", estimate.Plus.Loss)
	}
	if math.Abs(estimate.Minus.Loss-5.998965) > 1e-5 {
		t.Errorf("the negative probe read %.9f, want 5.998965 = (3 + 0.007)(2 - 0.005)", estimate.Minus.Loss)
	}
	if math.Abs(estimate.Gradient-1) > oracleTolerance {
		t.Fatalf("the central difference is %.6f, want 1 = B*ZA + ZB*A; a second adapter would read -35", estimate.Gradient)
	}

	// A loss that is not linear in the output: the chain rule has to come
	// through the same candidate.
	engine.loss = func(y []float64) float64 { return 0.5 * y[0] * y[0] }
	estimate, err = zerothorder.Step(ctx, engine, zerothorder.Example{ID: "oracle"}, services.Direction{Seed: 1, Scale: 0.001})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(estimate.Gradient-6) > oracleTolerance {
		t.Fatalf("the gradient of 0.5*y^2 is %.6f, want y*dy = 6*1", estimate.Gradient)
	}

	// The accepted step, with a coefficient that is not eps, goes through the
	// same constructor as the probes: the bytes the host loaded for a probe at
	// that coefficient are the bytes the fold leaves on disk.
	step := services.Direction{Seed: 1, Scale: 0.1}
	if err := s.Perturb(ctx, step, -1); err != nil {
		t.Fatal(err)
	}
	probed := host.loaded[len(host.loaded)-1]
	probedReading, err := engine.Score(ctx, zerothorder.Example{})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Perturb(ctx, step, 0); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Fold(services.Direction{Seed: 1}, -0.1); err != nil {
		t.Fatal(err)
	}
	folded, err := os.ReadFile(policy)
	if err != nil {
		t.Fatal(err)
	}
	if string(folded) != string(probed.bytes) {
		t.Fatal("the folded policy is not the file the probe at the same coefficient loaded; the probe and the step built different candidates")
	}
	a, _ := s.Values(loraA)
	b, _ := s.Values(loraB)
	if a[0] != float32(2-0.5) || b[0] != float32(3+0.7) {
		t.Fatalf("the fold left A=%v B=%v, want A=1.5 B=3.7: A moved by c*ZA and B by c*ZB", a[0], b[0])
	}
	foldedReading, err := engine.Score(ctx, zerothorder.Example{})
	if err != nil {
		t.Fatal(err)
	}
	if foldedReading.Loss != probedReading.Loss {
		t.Fatalf("the folded policy scores %v where the probe scored %v; same bytes have to read the same", foldedReading.Loss, probedReading.Loss)
	}
}

func TestASecondAdapterProbeFailsTheOracle(t *testing.T) {
	policy, scratch := scalarFixture(t)
	host := &factorHost{alpha: 1}
	s := open(t, host, policy, scratch)
	z := oracleDisplacement(1)
	control := &secondAdapterEngine{snapshot: s, z: z, alpha: 1, loss: func(y []float64) float64 { return y[0] }}
	ctx := context.Background()

	estimate, err := zerothorder.Step(ctx, control, zerothorder.Example{}, services.Direction{Seed: 1, Scale: 0.001})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(estimate.Gradient-(-35)) > oracleTolerance {
		t.Fatalf("the second-adapter control estimated %.6f, want ZB*ZA = -35", estimate.Gradient)
	}
	control.loss = func(y []float64) float64 { return 0.5 * y[0] * y[0] }
	quadratic, err := zerothorder.Step(ctx, control, zerothorder.Example{}, services.Direction{Seed: 1, Scale: 0.001})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(quadratic.Gradient-(-210)) > oracleTolerance {
		t.Fatalf("the second-adapter control estimated %.6f for 0.5*y^2, want 6*(-35) = -210", quadratic.Gradient)
	}
	// The same predicate the oracle test applies has to reject this number. If
	// somebody widens oracleTolerance until -35 passes for 1, this is the line
	// that fails.
	if math.Abs(estimate.Gradient-1) <= oracleTolerance {
		t.Fatalf("oracleTolerance %v accepts %.6f as 1; it can no longer tell the product-space probe from the factor-space one", oracleTolerance, estimate.Gradient)
	}
}

func TestACandidateKeepsRectangularFactorsInPlace(t *testing.T) {
	policy, scratch := matrixFixture(t)
	host := &factorHost{alpha: 4}
	s := open(t, host, policy, scratch)
	z := matrixDisplacement(1)
	s.Use(z)

	// (a) Every element of both factors moves by c times its own displacement,
	// for the two probe coefficients and for one that is neither.
	for _, c := range []float64{0.001, -0.001, 0.37} {
		candidate, err := services.BuildCandidate(s, z, c)
		if err != nil {
			t.Fatal(err)
		}
		for i, name := range []string{loraA, loraB} {
			before, _ := s.Values(name)
			after, ok := candidate.Values(name)
			if !ok || len(after) != len(before) {
				t.Fatalf("the candidate lost %s", name)
			}
			for k := range before {
				want := float32(float64(before[k]) + c*float64(z.Tensors[i][k]))
				if after[k] != want {
					t.Fatalf("%s[%d] at c=%v is %v, want %v", name, k, c, after[k], want)
				}
			}
		}
		// (b) The header is the snapshot's, byte for byte: names, dims, alpha,
		// rank and offsets survive because nothing rewrote them.
		header := int(s.Container().DataOffset)
		if string(candidate.Bytes()[:header]) != string(s.Bytes()[:header]) {
			t.Fatal("the candidate's header differs from the snapshot's")
		}
		if candidate.Source != s.Digest() {
			t.Fatal("the candidate does not name the snapshot it was built from")
		}
	}
	for i, tensor := range s.Container().Tensors {
		want := [][]uint64{{3, 2}, {2, 2}}[i]
		if len(tensor.Dims) != 2 || tensor.Dims[0] != want[0] || tensor.Dims[1] != want[1] {
			t.Fatalf("%s reads back with dims %v, want %v", tensor.Name, tensor.Dims, want)
		}
	}

	// (c) The estimate is s * sum(B*ZA + ZB*A) = 2 * (10 + 42) = 104. With B
	// transposed it would be 2 * (14 + 42) = 112, and that is the control.
	engine := &factorEngine{Snapshot: s, host: host, loss: sumOf}
	for _, eps := range []float64{math.Ldexp(1, -10), 0.001} {
		estimate, err := zerothorder.Step(context.Background(), engine, zerothorder.Example{}, services.Direction{Seed: 1, Scale: eps})
		if err != nil {
			t.Fatal(err)
		}
		tolerance := oracleTolerance
		if eps == math.Ldexp(1, -10) {
			// Every sum is exact in float32 at this eps, so the estimate is too.
			tolerance = 1e-9
		}
		if math.Abs(estimate.Gradient-104) > tolerance {
			t.Fatalf("at eps=%v the estimate is %.9f, want 104 = 2*sum(B*ZA + ZB*A)", eps, estimate.Gradient)
		}
		if math.Abs(estimate.Gradient-112) < 1 {
			t.Fatalf("the estimate %.6f matches the transposed B", estimate.Gradient)
		}
	}

	// (d) A displacement whose slices do not fit the tensors is refused.
	swapped := &services.Displacement{Direction: services.Direction{Seed: 1}, Tensors: [][]float32{z.Tensors[1], z.Tensors[0]}}
	if _, err := services.BuildCandidate(s, swapped, 0.001); err == nil {
		t.Fatal("a displacement with the factors swapped was accepted")
	}
	short := &services.Displacement{Direction: services.Direction{Seed: 1}, Tensors: [][]float32{z.Tensors[0]}}
	if _, err := services.BuildCandidate(s, short, 0.001); err == nil {
		t.Fatal("a displacement over one tensor was accepted for two")
	}
}

func TestStepAndFoldUseTheSeededDisplacement(t *testing.T) {
	policy, scratch := scalarFixture(t)
	host := &factorHost{alpha: 1}
	s := open(t, host, policy, scratch)
	d := services.Direction{Seed: 7, Scale: 0.001}

	z := s.Draw(d)
	za, zb := float64(z.Tensors[0][0]), float64(z.Tensors[1][0])
	// The first two values of seed 7's stream, as TestTheSeedReproducesTheDirection
	// pins them; a different derivation here would move the fold off the probe.
	var pinned [2]float32
	d.Fill(pinned[:])
	if float32(za) != pinned[0] || float32(zb) != pinned[1] {
		t.Fatalf("Draw gave (%v, %v) where Fill gives (%v, %v)", za, zb, pinned[0], pinned[1])
	}

	engine := &factorEngine{Snapshot: s, host: host, loss: func(y []float64) float64 { return y[0] }}
	estimate, err := zerothorder.Step(context.Background(), engine, zerothorder.Example{}, d)
	if err != nil {
		t.Fatal(err)
	}
	want := 3*za + 2*zb
	if math.Abs(estimate.Gradient-want) > oracleTolerance {
		t.Fatalf("the estimate along seed 7 is %.6f, want B*ZA + ZB*A = %.6f", estimate.Gradient, want)
	}
	if math.Abs(estimate.Gradient-zb*za) < 0.5 {
		t.Fatalf("the estimate %.6f is the product-space derivative ZB*ZA = %.6f", estimate.Gradient, zb*za)
	}

	c := -0.1 * estimate.Gradient
	if _, err := s.Fold(d, c); err != nil {
		t.Fatal(err)
	}
	a, _ := s.Values(loraA)
	b, _ := s.Values(loraB)
	if a[0] != float32(2+c*za) || b[0] != float32(3+c*zb) {
		t.Fatalf("the fold left A=%v B=%v, want %v and %v", a[0], b[0], float32(2+c*za), float32(3+c*zb))
	}
}

func TestTheDrawnDisplacementIsTheFilledDirection(t *testing.T) {
	policy, scratch := matrixFixture(t)
	host := &factorHost{alpha: 4}
	s := open(t, host, policy, scratch)

	filled := make([]float32, 10)
	services.Direction{Seed: 42}.Fill(filled)
	var drawn []float32
	for _, tensor := range s.Draw(services.Direction{Seed: 42}).Tensors {
		drawn = append(drawn, tensor...)
	}
	if len(drawn) != len(filled) {
		t.Fatalf("Draw produced %d values over a 10-element adapter", len(drawn))
	}
	for i := range filled {
		if drawn[i] != filled[i] {
			t.Fatalf("Draw and Fill disagree at %d: %v against %v", i, drawn[i], filled[i])
		}
	}
	again := s.Draw(services.Direction{Seed: 42})
	other := s.Draw(services.Direction{Seed: 43})
	for i := range again.Tensors {
		for k := range again.Tensors[i] {
			if again.Tensors[i][k] != s.Draw(services.Direction{Seed: 42}).Tensors[i][k] {
				t.Fatal("the same seed drew different displacements")
			}
			if again.Tensors[i][k] == other.Tensors[i][k] {
				t.Fatal("two seeds drew the same value")
			}
		}
	}
}

func TestAStepRestoresTheSnapshotExactly(t *testing.T) {
	policy, scratch := scalarFixture(t)
	host := &factorHost{alpha: 1}
	s := open(t, host, policy, scratch)
	s.Use(oracleDisplacement(1))
	before := fileDigest(t, policy)
	engine := &factorEngine{Snapshot: s, host: host, loss: func(y []float64) float64 { return y[0] }}

	if _, err := zerothorder.Step(context.Background(), engine, zerothorder.Example{}, services.Direction{Seed: 1, Scale: 0.001}); err != nil {
		t.Fatal(err)
	}
	if len(host.applied) != 1 || host.applied[0] != s.Handle() || host.scales[0] != 1 {
		t.Fatalf("after the step the host holds %v at %v, want the snapshot's own handle at scale 1", host.applied, host.scales)
	}
	if fileDigest(t, policy) != before || s.Digest() != before {
		t.Fatal("a step changed the snapshot on disk or in memory")
	}
	if left := leftovers(t, scratch, ".candidate-"); len(left) != 0 {
		t.Fatalf("the step left %v in the scratch", left)
	}
	if left := leftovers(t, filepath.Dir(policy), ".fold-"); len(left) != 0 {
		t.Fatalf("the step left %v beside the policy", left)
	}
	if len(host.loaded) != 3 {
		t.Fatalf("the host loaded %d adapters, want the snapshot and two probes", len(host.loaded))
	}
	if host.loaded[0].closed || !host.loaded[1].closed || !host.loaded[2].closed {
		t.Fatal("the probe handles have to be closed after the step and the snapshot's kept")
	}
}

func TestAFailedOrCancelledProbeLeavesTheSnapshotApplied(t *testing.T) {
	loss := func(y []float64) float64 { return y[0] }
	snapshotApplied := func(t *testing.T, host *factorHost, s *services.Snapshot, policy, scratch, digest string) {
		t.Helper()
		if len(host.applied) != 1 || host.applied[0] != s.Handle() {
			t.Fatalf("the host holds %v, want the snapshot alone", host.applied)
		}
		if fileDigest(t, policy) != digest {
			t.Fatal("the snapshot on disk changed")
		}
		if left := leftovers(t, scratch, ".candidate-"); len(left) != 0 {
			t.Fatalf("%v was left in the scratch", left)
		}
		if left := leftovers(t, filepath.Dir(policy), ".fold-"); len(left) != 0 {
			t.Fatalf("%v was left beside the policy", left)
		}
	}

	t.Run("the load of a probe fails", func(t *testing.T) {
		policy, scratch := scalarFixture(t)
		injected := errors.New("the card refused the adapter")
		host := &factorHost{alpha: 1, failLoadAt: 2, errLoad: injected}
		s := open(t, host, policy, scratch)
		s.Use(oracleDisplacement(1))
		digest := s.Digest()
		engine := &factorEngine{Snapshot: s, host: host, loss: loss}
		_, err := zerothorder.Step(context.Background(), engine, zerothorder.Example{}, services.Direction{Seed: 1, Scale: 0.001})
		if !errors.Is(err, injected) {
			t.Fatalf("the step returned %v, want the host's error", err)
		}
		snapshotApplied(t, host, s, policy, scratch, digest)
	})

	t.Run("the apply of a probe fails", func(t *testing.T) {
		policy, scratch := scalarFixture(t)
		injected := errors.New("the context refused the adapter")
		host := &factorHost{alpha: 1, failSetAt: 2, errSet: injected}
		s := open(t, host, policy, scratch)
		s.Use(oracleDisplacement(1))
		digest := s.Digest()
		engine := &factorEngine{Snapshot: s, host: host, loss: loss}
		_, err := zerothorder.Step(context.Background(), engine, zerothorder.Example{}, services.Direction{Seed: 1, Scale: 0.001})
		if !errors.Is(err, injected) {
			t.Fatalf("the step returned %v, want the host's error", err)
		}
		snapshotApplied(t, host, s, policy, scratch, digest)
		if !host.loaded[1].closed {
			t.Fatal("the refused candidate's handle was not closed")
		}
	})

	t.Run("the context is cancelled before the step", func(t *testing.T) {
		policy, scratch := scalarFixture(t)
		host := &factorHost{alpha: 1}
		s := open(t, host, policy, scratch)
		s.Use(oracleDisplacement(1))
		digest := s.Digest()
		engine := &factorEngine{Snapshot: s, host: host, loss: loss}
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_, err := zerothorder.Step(ctx, engine, zerothorder.Example{}, services.Direction{Seed: 1, Scale: 0.001})
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("the step returned %v, want the cancellation", err)
		}
		if host.loads != 1 {
			t.Fatalf("a cancelled step still loaded %d candidates", host.loads-1)
		}
		snapshotApplied(t, host, s, policy, scratch, digest)
	})

	t.Run("the context is cancelled during the first probe", func(t *testing.T) {
		policy, scratch := scalarFixture(t)
		host := &factorHost{alpha: 1}
		s := open(t, host, policy, scratch)
		s.Use(oracleDisplacement(1))
		digest := s.Digest()
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		engine := &factorEngine{Snapshot: s, host: host, loss: loss, ctx: ctx, onScore: cancel}
		_, err := zerothorder.Step(ctx, engine, zerothorder.Example{}, services.Direction{Seed: 1, Scale: 0.001})
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("the step returned %v, want the cancellation", err)
		}
		// The +eps candidate was applied when the cancellation arrived; the step
		// has to put the snapshot back regardless.
		snapshotApplied(t, host, s, policy, scratch, digest)
		if !host.loaded[1].closed {
			t.Fatal("the cancelled probe's handle was not closed")
		}
	})

	t.Run("the load of a fold fails", func(t *testing.T) {
		policy, scratch := scalarFixture(t)
		injected := errors.New("the card refused the adapter")
		host := &factorHost{alpha: 1, failLoadAt: 2, errLoad: injected}
		s := open(t, host, policy, scratch)
		s.Use(oracleDisplacement(1))
		digest := s.Digest()
		if _, err := s.Fold(services.Direction{Seed: 1}, -0.1); !errors.Is(err, injected) {
			t.Fatalf("the fold returned %v, want the host's error", err)
		}
		if s.Digest() != digest {
			t.Fatal("a refused fold changed the snapshot in memory")
		}
		snapshotApplied(t, host, s, policy, scratch, digest)
	})

	t.Run("a fold with a coefficient that is not a number", func(t *testing.T) {
		policy, scratch := scalarFixture(t)
		host := &factorHost{alpha: 1}
		s := open(t, host, policy, scratch)
		s.Use(oracleDisplacement(1))
		digest := s.Digest()
		for _, c := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
			if _, err := s.Fold(services.Direction{Seed: 1}, c); err == nil {
				t.Fatalf("a fold by %v was accepted", c)
			}
		}
		if host.loads != 1 {
			t.Fatal("a refused fold still loaded something")
		}
		snapshotApplied(t, host, s, policy, scratch, digest)
	})
}

// A fold whose apply is refused leaves nothing behind: not in the engine, not
// in memory, and not on disk.
//
// The apply comes before the rename precisely so this is true. A file on disk
// is not evidence that a policy is in effect -- the addendum says a hash of the
// changed file, on its own, does not prove the adapter was applied. Written
// first, an apply that fails would leave the disk holding the new policy whilst
// the engine, the digest and the body still describe the old one, and the next
// reader would believe the file.
func TestAFoldThatIsNotAppliedChangesNothing(t *testing.T) {
	policy, scratch := scalarFixture(t)
	injected := errors.New("the context refused the adapter")
	host := &factorHost{alpha: 1, failSetAt: 2, errSet: injected}
	s := open(t, host, policy, scratch)
	z := oracleDisplacement(1)
	s.Use(z)
	digest := s.Digest()
	before := fileDigest(t, policy)
	if before != digest {
		t.Fatal("the fixture starts with the file and the snapshot disagreeing")
	}

	_, err := s.Fold(services.Direction{Seed: 1}, -0.1)
	if !errors.Is(err, injected) || !strings.Contains(err.Error(), "the policy on disk is unchanged") {
		t.Fatalf("the fold returned %v; a candidate that could not be applied has to say the disk was not touched", err)
	}
	if len(host.applied) != 1 || host.applied[0] != s.Handle() || s.Digest() != digest {
		t.Fatal("after the failed apply the old handle has to stay applied and the snapshot unchanged in memory")
	}
	if got := fileDigest(t, policy); got != before {
		t.Fatalf("the policy on disk moved to %s; the engine is on %s, and a reader would believe the file", got, digest)
	}
	if left := leftovers(t, filepath.Dir(policy), ".fold-"); len(left) != 0 {
		t.Fatalf("%v was left beside the policy", left)
	}
}

func TestAReloadedSnapshotScoresWhatWasCommitted(t *testing.T) {
	policy, scratch := scalarFixture(t)
	host := &factorHost{alpha: 1}
	s := open(t, host, policy, scratch)
	z := oracleDisplacement(1)
	s.Use(z)
	loss := func(y []float64) float64 { return y[0] }
	engine := &factorEngine{Snapshot: s, host: host, loss: loss}
	ctx := context.Background()

	// S is immutable through the probes: a candidate at coefficient zero is S.
	zero, err := services.BuildCandidate(s, z, 0)
	if err != nil {
		t.Fatal(err)
	}
	if zero.Digest != s.Digest() || zero.Moved != 0 {
		t.Fatal("a candidate at coefficient zero is not the snapshot")
	}
	if _, err := zerothorder.Step(ctx, engine, zerothorder.Example{}, services.Direction{Seed: 1, Scale: 0.001}); err != nil {
		t.Fatal(err)
	}
	if again, _ := services.BuildCandidate(s, z, 0); again.Digest != s.Digest() {
		t.Fatal("the probes moved the snapshot")
	}

	moved, err := s.Fold(services.Direction{Seed: 1}, -0.1)
	if err != nil {
		t.Fatal(err)
	}
	if moved != 2 {
		t.Fatalf("the fold moved %d elements, want both", moved)
	}
	committed, err := engine.Score(ctx, zerothorder.Example{})
	if err != nil {
		t.Fatal(err)
	}

	other := &factorHost{alpha: 1}
	reopened, err := services.OpenSnapshot(other, policy, scratch)
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	if reopened.Digest() != s.Digest() || other.applied[0].Digest() != s.Digest() {
		t.Fatalf("the reopened policy is %s, the folded snapshot %s, the host applied %s", reopened.Digest(), s.Digest(), other.applied[0].Digest())
	}
	reloaded, err := (&factorEngine{Snapshot: reopened, host: other, loss: loss}).Score(ctx, zerothorder.Example{})
	if err != nil {
		t.Fatal(err)
	}
	if reloaded.Loss != committed.Loss {
		t.Fatalf("the reloaded policy scores %v where the committed one scored %v", reloaded.Loss, committed.Loss)
	}

	// A host that loads something other than what the constructor wrote is
	// refused, at open and at every probe.
	tampered := &factorHost{alpha: 1, tamperFrom: 2}
	suspect := open(t, tampered, policy, scratch)
	suspect.Use(z)
	if err := suspect.Perturb(ctx, services.Direction{Seed: 1, Scale: 0.001}, 1); err == nil {
		t.Fatal("a probe whose loaded bytes are not the candidate was applied")
	}
	if len(tampered.applied) != 1 || tampered.applied[0] != suspect.Handle() {
		t.Fatal("the refused probe left something other than the snapshot applied")
	}
	if _, err := suspect.Fold(services.Direction{Seed: 1}, -0.1); err == nil {
		t.Fatal("a fold whose loaded bytes are not the candidate was committed")
	}
	if fileDigest(t, policy) != s.Digest() {
		t.Fatal("the refused fold changed the policy on disk")
	}
	if _, err := services.OpenSnapshot(&factorHost{alpha: 1, tamperFrom: 1}, policy, scratch); err == nil {
		t.Fatal("a policy the host loaded as different bytes was opened")
	}
}

func TestAHalfPrecisionSnapshotIsRefusedAtOpen(t *testing.T) {
	dir := t.TempDir()
	half := filepath.Join(dir, "half.gguf")
	tests.WriteLoRAGGUF(t, half, "oracle", 1, []tests.LoRATensor{
		{Name: loraA, Dims: []uint64{1, 1}, Type: 1, Data: []float32{2}},
		{Name: loraB, Dims: []uint64{1, 1}, Type: 1, Data: []float32{3}},
	})
	host := &factorHost{alpha: 1}
	if _, err := services.OpenSnapshot(host, half, dir); !errors.Is(err, services.ErrHalfPrecisionPolicy) {
		t.Fatalf("a half-precision policy was opened with %v; it has to be refused before the engine sees it", err)
	}
	if host.loads != 0 {
		t.Fatal("the refused policy was still loaded on the host")
	}

	quantised := filepath.Join(dir, "q8.gguf")
	tests.WriteLoRAGGUF(t, quantised, "oracle", 1, []tests.LoRATensor{
		{Name: loraA, Dims: []uint64{1, 1}, Type: 8, Data: []float32{2}},
		{Name: loraB, Dims: []uint64{1, 1}, Type: 8, Data: []float32{3}},
	})
	_, err := services.OpenSnapshot(host, quantised, dir)
	if err == nil || errors.Is(err, services.ErrHalfPrecisionPolicy) {
		t.Fatalf("a quantised policy was answered with %v; it is neither trainable nor half precision", err)
	}
}
