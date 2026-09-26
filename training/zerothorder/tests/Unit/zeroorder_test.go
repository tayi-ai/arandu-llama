package unit_test

import (
	"github.com/tayi-ai/arandu-llama/training/adapter"

	"context"
	"errors"
	"math"
	"testing"

	services "github.com/tayi-ai/arandu-llama/training/zerothorder"
)

// What the zeroth-order step has to get right before a card is involved: the
// arithmetic of the estimate, the fact that a seed reproduces a direction, and
// that a step never leaves the engine perturbed.

// quadratic is an engine with no model behind it: the loss is a known function
// of the perturbation, so the estimate has an answer to be checked against.
//
// L(x) = (x - m)^2 around the current point, which has derivative 2(x - m).
// Probing at +e and -e and taking the central difference has to return that
// derivative exactly, because the third derivative of a quadratic is zero and
// the central difference has no error term left.
type quadratic struct {
	minimum   float64
	at        float64
	scored    int
	perturbed []float64
	fail      error
	// failScore is returned by the first Score, so a step can be watched
	// failing between its probes.
	failScore error
}

func (q *quadratic) Perturb(_ context.Context, d adapter.Direction, sign float64) error {
	if q.fail != nil {
		return q.fail
	}
	q.at = sign * d.Scale
	q.perturbed = append(q.perturbed, q.at)
	return nil
}

func (q *quadratic) Score(context.Context, services.Example) (services.Reading, error) {
	q.scored++
	if q.failScore != nil && q.scored == 1 {
		return services.Reading{}, q.failScore
	}
	delta := q.at - q.minimum
	return services.Reading{Loss: delta * delta, Tokens: 1}, nil
}

func TestTheEstimateIsTheCentralDifference(t *testing.T) {
	engine := &quadratic{minimum: 0.25}
	got, err := services.Step(context.Background(), engine, services.Example{ID: "e1"}, adapter.Direction{Seed: 7, Scale: 0.01})
	if err != nil {
		t.Fatal(err)
	}
	// The derivative of (x - 0.25)^2 at x = 0 is -0.5.
	if math.Abs(got.Gradient-(-0.5)) > 1e-9 {
		t.Fatalf("the estimate is %v, want -0.5", got.Gradient)
	}
	if engine.scored != 2 {
		t.Fatalf("the step scored %d times, want exactly 2: a zeroth-order step is two forward passes", engine.scored)
	}
}

func TestAStepLeavesTheEngineUnperturbed(t *testing.T) {
	engine := &quadratic{minimum: 0.25}
	if _, err := services.Step(context.Background(), engine, services.Example{ID: "e1"}, adapter.Direction{Seed: 7, Scale: 0.01}); err != nil {
		t.Fatal(err)
	}
	// Plus, minus, and back to zero. An engine left perturbed is a model a later
	// measurement reads without anybody meaning to.
	want := []float64{0.01, -0.01, 0}
	if len(engine.perturbed) != len(want) {
		t.Fatalf("the step perturbed %v, want %v", engine.perturbed, want)
	}
	for i := range want {
		if engine.perturbed[i] != want[i] {
			t.Fatalf("the step perturbed %v, want %v", engine.perturbed, want)
		}
	}
}

func TestAZeroScaleIsRefused(t *testing.T) {
	engine := &quadratic{}
	for _, scale := range []float64{0, -0.01} {
		if _, err := services.Step(context.Background(), engine, services.Example{}, adapter.Direction{Seed: 1, Scale: scale}); err == nil {
			t.Errorf("scale %v was admitted; it measures the same point twice and estimates nothing", scale)
		}
	}
	if engine.scored != 0 {
		t.Fatalf("a refused step still scored %d times", engine.scored)
	}
}

func TestAnEngineThatFailsStopsTheStep(t *testing.T) {
	broken := errors.New("the card is busy")
	engine := &quadratic{fail: broken}
	if _, err := services.Step(context.Background(), engine, services.Example{}, adapter.Direction{Seed: 1, Scale: 0.01}); !errors.Is(err, broken) {
		t.Fatalf("the step returned %v, want the engine's own error", err)
	}
}

func TestTheSeedReproducesTheDirection(t *testing.T) {
	// Two machines holding the same seed hold the same direction.
	first := make([]float32, 4096)
	second := make([]float32, 4096)
	adapter.Direction{Seed: 42, Scale: 0.01}.Fill(first)
	adapter.Direction{Seed: 42, Scale: 0.01}.Fill(second)
	for i := range first {
		if first[i] != second[i] {
			t.Fatalf("the same seed gave different directions at %d: %v against %v", i, first[i], second[i])
		}
	}
	other := make([]float32, 4096)
	adapter.Direction{Seed: 43, Scale: 0.01}.Fill(other)
	same := 0
	for i := range first {
		if first[i] == other[i] {
			same++
		}
	}
	if same > len(first)/100 {
		t.Fatalf("two seeds agreed on %d of %d elements; they are not independent directions", same, len(first))
	}
}

func TestTheDirectionIsCenteredAndScaled(t *testing.T) {
	// The estimator assumes a standard normal. A direction whose spread is not one
	// is a learning rate error that looks like everything else.
	values := make([]float32, 1<<16)
	adapter.Direction{Seed: 11, Scale: 0.01}.Fill(values)
	sum, sumSquares := 0.0, 0.0
	for _, v := range values {
		sum += float64(v)
		sumSquares += float64(v) * float64(v)
	}
	n := float64(len(values))
	mean := sum / n
	variance := sumSquares/n - mean*mean
	if math.Abs(mean) > 0.02 {
		t.Errorf("the direction has mean %v, want about 0", mean)
	}
	if math.Abs(variance-1) > 0.03 {
		t.Errorf("the direction has variance %v, want about 1", variance)
	}
}

// TestAFailedProbeStillRestoresTheEngine is here because the first Step did
// not: a probe that failed at Score returned with the +eps candidate applied,
// and whatever measured next read a model nobody meant to evaluate.
func TestAFailedProbeStillRestoresTheEngine(t *testing.T) {
	broken := errors.New("the card is busy")
	engine := &quadratic{minimum: 0.25, failScore: broken}
	_, err := services.Step(context.Background(), engine, services.Example{}, adapter.Direction{Seed: 1, Scale: 0.01})
	if !errors.Is(err, broken) {
		t.Fatalf("the step returned %v, want the engine's own error", err)
	}
	want := []float64{0.01, 0}
	if len(engine.perturbed) != len(want) {
		t.Fatalf("the failed step perturbed %v, want %v: the probe and then the restore", engine.perturbed, want)
	}
	for i := range want {
		if engine.perturbed[i] != want[i] {
			t.Fatalf("the failed step perturbed %v, want %v", engine.perturbed, want)
		}
	}
}

func TestAnEngineWithNoDirectionSlotsRefusesToProbe(t *testing.T) {
	engine := services.NewLlamaEngine("http://127.0.0.1:1")
	if err := engine.Perturb(context.Background(), adapter.Direction{Seed: 1, Scale: 0.01}, 1); err == nil {
		t.Fatal("an engine with an empty bank accepted a perturbation; it would have probed the unperturbed model twice")
	}
}

// TestTheServerRefusalNamesTheProductSpace pins that the HTTP engine refuses
// to probe before it reaches the network: the address is unreachable, and a
// request would surface as a connection error rather than the sentinel.
func TestTheServerRefusalNamesTheProductSpace(t *testing.T) {
	engine := services.NewLlamaEngine("http://127.0.0.1:1")
	engine.Bank = []int{0, 1}
	err := engine.Perturb(context.Background(), adapter.Direction{Seed: 1, Scale: 0.01}, 1)
	if !errors.Is(err, services.ErrProductSpaceProbe) {
		t.Fatalf("the server engine answered %v to a probe; scaling a second adapter measures ZB*ZA, and the refusal has to say so", err)
	}
}

// weighingEngine scores each example by its id, so a test can say exactly what
// each one costs and check the arithmetic that combines them.
type weighingEngine struct {
	loss map[string]float64 // mean nll per example
	toks map[string]int     // positions each mean is over
	sign float64
}

func (e *weighingEngine) Perturb(_ context.Context, _ adapter.Direction, sign float64) error {
	e.sign = sign
	return nil
}

func (e *weighingEngine) Score(_ context.Context, ex services.Example) (services.Reading, error) {
	l, ok := e.loss[ex.ID]
	if !ok {
		return services.Reading{}, errors.New("no loss for " + ex.ID)
	}
	// The perturbation moves every example by the same amount, so a batch
	// gradient is exactly the mean of the per-example gradients and the test can
	// predict it.
	return services.Reading{Loss: l + e.sign, Tokens: e.toks[ex.ID]}, nil
}

// A batch loss weights each example by its length, and that is not the same
// number as the mean of the means.
//
// Two examples, one worth 0.2 over 90 positions and one worth 2.0 over 10:
//
//	weighted   (0.2*90 + 2.0*10) / 100 = 0.38
//	mean of means   (0.2 + 2.0) / 2    = 1.10
//
// Nearly three times apart. The weighted figure is the one a backpropagating
// trainer reports for the same data, and comparing against that reference is
// the only reason this loss is measured here at all.
func TestABatchLossWeightsByLength(t *testing.T) {
	engine := &weighingEngine{
		loss: map[string]float64{"long": 0.2, "short": 2.0},
		toks: map[string]int{"long": 90, "short": 10},
	}
	batch := []services.Example{
		{ID: "long", Tokens: make([]int32, 91), PromptTokens: 1},
		{ID: "short", Tokens: make([]int32, 11), PromptTokens: 1},
	}

	reading, err := services.ScoreBatch(context.Background(), engine, batch)
	if err != nil {
		t.Fatal(err)
	}
	if reading.Tokens != 100 {
		t.Fatalf("the batch covers %d positions, want 100", reading.Tokens)
	}
	const want = 0.38
	if math.Abs(reading.Loss-want) > 1e-12 {
		t.Fatalf("batch loss = %.12f, want %.12f", reading.Loss, want)
	}
	// The trap, stated so a future change that reintroduces it fails here.
	if math.Abs(reading.Loss-1.1) < 1e-9 {
		t.Fatal("the batch loss is the mean of the means; short completions are being weighted like long ones")
	}
}

// The order examples are named in does not change the batch loss.
//
// A chain that has to be reproducible cannot have its numbers depend on how
// somebody typed the ids, so the reader returns file order and the sum is taken
// in that order.
func TestTheBatchLossDoesNotDependOnOrder(t *testing.T) {
	engine := &weighingEngine{
		loss: map[string]float64{"a": 0.3, "b": 1.7, "c": 0.9},
		toks: map[string]int{"a": 17, "b": 4, "c": 31},
	}
	forward := []services.Example{{ID: "a"}, {ID: "b"}, {ID: "c"}}
	backward := []services.Example{{ID: "c"}, {ID: "b"}, {ID: "a"}}

	one, err := services.ScoreBatch(context.Background(), engine, forward)
	if err != nil {
		t.Fatal(err)
	}
	two, err := services.ScoreBatch(context.Background(), engine, backward)
	if err != nil {
		t.Fatal(err)
	}
	if one.Tokens != two.Tokens {
		t.Fatalf("the two orders cover %d and %d positions", one.Tokens, two.Tokens)
	}
	if math.Abs(one.Loss-two.Loss) > 1e-12 {
		t.Fatalf("order changed the batch loss: %.15f against %.15f", one.Loss, two.Loss)
	}
}

// A step over a batch estimates the gradient of the batch loss, not of one
// example's.
//
// The engine here moves every example by the perturbation's sign, so the
// central difference is exactly 1/scale and the test can name it. What it
// proves is that both probes scored the whole batch: a step that scored only
// the first example would still produce a finite number, and it would be the
// gradient of a different function.
func TestAStepOverABatchMeasuresTheBatch(t *testing.T) {
	engine := &weighingEngine{
		loss: map[string]float64{"a": 0.5, "b": 1.5},
		toks: map[string]int{"a": 50, "b": 50},
	}
	batch := []services.Example{{ID: "a"}, {ID: "b"}}
	d := adapter.Direction{Seed: 1, Scale: 1e-3}

	estimate, err := services.StepOver(context.Background(), engine, batch, d)
	if err != nil {
		t.Fatal(err)
	}
	// L(+eps) - L(-eps) = (base+1) - (base-1) = 2, over 2*scale.
	want := 1.0 / d.Scale
	if math.Abs(estimate.Gradient-want) > 1e-6 {
		t.Fatalf("gradient = %.6f, want %.6f", estimate.Gradient, want)
	}
	if math.Abs(estimate.Plus.Loss-2.0) > 1e-12 || math.Abs(estimate.Minus.Loss-0.0) > 1e-12 {
		t.Fatalf("the probes read %.12f and %.12f; the batch base is 1.0", estimate.Plus.Loss, estimate.Minus.Loss)
	}
	// And the engine is left on the snapshot, not on a probe.
	if engine.sign != 0 {
		t.Fatalf("the engine stayed perturbed at sign %v after the step", engine.sign)
	}
}

// An empty batch is refused rather than scored.
//
// A mean over nothing reads downstream as a perfect prediction, which is the
// same reason Score refuses a sequence it scored no position of.
func TestAnEmptyBatchIsRefused(t *testing.T) {
	engine := &weighingEngine{loss: map[string]float64{}, toks: map[string]int{}}
	if _, err := services.ScoreBatch(context.Background(), engine, nil); err == nil {
		t.Fatal("an empty batch was scored; a mean over nothing reads as a perfect prediction")
	}
	if _, err := services.StepOver(context.Background(), engine, nil, adapter.Direction{Seed: 1, Scale: 1e-3}); err == nil {
		t.Fatal("a step was taken over an empty batch")
	}
}
