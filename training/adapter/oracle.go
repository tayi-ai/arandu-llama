package adapter

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
)

// The independent arithmetic the engine is checked against.
//
// The controls that compare two adapters with each other cannot settle a uniform
// factor: if the engine applied a hundred times the right contribution, it would
// apply it to both sides of every equality and every one of them would still
// hold. Only a comparison against a number computed somewhere else closes that,
// and "somewhere else" has to mean this file -- it calls nothing the engine
// calls, and it works from the factors as they are on disk.
//
// llama.cpp computes, in llm_graph_context::build_lora_mm:
//
//	h  = A * x
//	u  = scale * (B * h),   scale = adapter_scale * alpha / rank
//	y  = W * x + u
//
// with rank taken from the tensor, not from metadata --
// llama_adapter_lora_weight::get_scale reads b->ne[0].
//
// Layout, and it is the part that is easy to get backwards: a ggml tensor of
// ne = [d0, d1] is flat with ne[0] varying fastest. So x of [n_in, n_tokens] has
// element (i, t) at t*n_in + i; A of [n_in, rank] has (i, r) at r*n_in + i; and
// B of [rank, n_out] has (r, j) at j*rank + r.

// LoRAReference is one projection's factors and the scale to apply them at.
type LoRAReference struct {
	A     []float32
	B     []float32
	NIn   int
	Rank  int
	NOut  int
	Scale float64
}

// Validate refuses a reference whose pieces do not describe the same projection.
func (r LoRAReference) Validate() error {
	if r.NIn <= 0 || r.Rank <= 0 || r.NOut <= 0 {
		return fmt.Errorf("cluster: the reference needs positive shapes, got n_in=%d rank=%d n_out=%d", r.NIn, r.Rank, r.NOut)
	}
	if len(r.A) != r.NIn*r.Rank {
		return fmt.Errorf("cluster: A holds %d elements where [n_in, rank] is %d", len(r.A), r.NIn*r.Rank)
	}
	if len(r.B) != r.Rank*r.NOut {
		return fmt.Errorf("cluster: B holds %d elements where [rank, n_out] is %d", len(r.B), r.Rank*r.NOut)
	}
	if math.IsNaN(r.Scale) || math.IsInf(r.Scale, 0) {
		return fmt.Errorf("cluster: the scale is %v", r.Scale)
	}
	return nil
}

// Where the rounding is, and why each stage is computed the way it is.
//
// This is declared before any result is looked at, because a reference that
// chose its arithmetic after seeing the engine's answer is not a reference.
//
//   h = A*x     accumulated in float64 and rounded once on the way out. The
//               engine sums n_in terms in float32 and rounds as it goes, so the
//               two agree exactly only when the sum is trivial. In the fixture
//               it is: A has one cell at 1, so 11007 of the terms are 0*x and
//               adding an exact zero to a finite float changes nothing, in any
//               order. Hence Exact for that fixture, and a declared relative
//               tolerance for the general one.
//   u_pre = B*h same, over rank terms. One cell again in the fixture.
//   u = s*u_pre one float32 multiplication on each side. In the fixture s is 2
//               and B's cell is 0.5, both exact in binary, so this rounds
//               nowhere.
//   y = y_b + u ONE float32 addition, and Go's float32 addition is IEEE single
//               precision, the same operation ggml performs. So the reference
//               here reproduces the rounding of the operation being compared
//               rather than computing a more precise answer the engine never
//               tried to produce.
//
// The staged comparison below matters more than any of it: each stage is fed
// the engine's OWN output from the stage before, never the reference's. A
// difference then belongs to one operation instead of propagating, which is
// what makes "stop at the first divergent operation" mean anything.

// Chain computes h, u before the scale, and u after it, for a captured input.
//
// This is the end-to-end path. It is the WEAKER of the two checks for finding a
// defect in the arithmetic: errors that compensate pass straight through it. An
// engine applying B twice and the scale at half gives 2*B*h * s/2 = s*B*h, and
// end to end that is the right answer. The staged comparison catches both,
// because each stage is judged against its own input.
//
// What it is for is the other direction: it consumes no captured intermediate,
// so it is sensitive to something the staged comparison is blind to. Stages that
// all agree pairwise while this one does not is a pattern that points at the
// capture -- a node matched that is not the one feeding the next operation. It
// points; it does not prove. Under a declared tolerance the same pattern can
// come from differences accepted at each stage adding up past the limit here, or
// from the reference itself being wrong. The pattern tells the next reader where
// to look first, not what the cause is.
func (r LoRAReference) Chain(x []float32, nTokens int) (h, uPre, u []float32, err error) {
	if err := r.Validate(); err != nil {
		return nil, nil, nil, err
	}
	if nTokens <= 0 {
		return nil, nil, nil, fmt.Errorf("cluster: the chain needs at least one token, got %d", nTokens)
	}
	if len(x) != nTokens*r.NIn {
		return nil, nil, nil, fmt.Errorf("cluster: the input holds %d elements where %d tokens of %d is %d", len(x), nTokens, r.NIn, nTokens*r.NIn)
	}
	for i, v := range x {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, nil, nil, fmt.Errorf("cluster: the captured input is not finite at index %d", i)
		}
	}

	h = make([]float32, nTokens*r.Rank)
	uPre = make([]float32, nTokens*r.NOut)
	u = make([]float32, nTokens*r.NOut)

	for t := 0; t < nTokens; t++ {
		hRow := make([]float64, r.Rank)
		for rr := 0; rr < r.Rank; rr++ {
			sum := 0.0
			for i := 0; i < r.NIn; i++ {
				sum += float64(r.A[rr*r.NIn+i]) * float64(x[t*r.NIn+i])
			}
			hRow[rr] = sum
			h[t*r.Rank+rr] = float32(sum)
		}
		for j := 0; j < r.NOut; j++ {
			sum := 0.0
			for rr := 0; rr < r.Rank; rr++ {
				sum += float64(r.B[j*r.Rank+rr]) * hRow[rr]
			}
			uPre[t*r.NOut+j] = float32(sum)
			u[t*r.NOut+j] = float32(r.Scale * sum)
		}
	}
	return h, uPre, u, nil
}

// ApplyA is h = A*x for one captured input, the first operation on its own.
func (r LoRAReference) ApplyA(x []float32, nTokens int) ([]float32, error) {
	if err := r.shapeFor(x, nTokens, r.NIn, "the input"); err != nil {
		return nil, err
	}
	h := make([]float32, nTokens*r.Rank)
	for t := 0; t < nTokens; t++ {
		for rr := 0; rr < r.Rank; rr++ {
			sum := 0.0
			for i := 0; i < r.NIn; i++ {
				sum += float64(r.A[rr*r.NIn+i]) * float64(x[t*r.NIn+i])
			}
			h[t*r.Rank+rr] = float32(sum)
		}
	}
	return h, nil
}

// ApplyB is u_pre = B*h, fed the h the engine produced rather than the one this
// reference would have.
func (r LoRAReference) ApplyB(h []float32, nTokens int) ([]float32, error) {
	if err := r.shapeFor(h, nTokens, r.Rank, "the intermediate"); err != nil {
		return nil, err
	}
	uPre := make([]float32, nTokens*r.NOut)
	for t := 0; t < nTokens; t++ {
		for j := 0; j < r.NOut; j++ {
			sum := 0.0
			for rr := 0; rr < r.Rank; rr++ {
				sum += float64(r.B[j*r.Rank+rr]) * float64(h[t*r.Rank+rr])
			}
			uPre[t*r.NOut+j] = float32(sum)
		}
	}
	return uPre, nil
}

// ApplyScale is u = s*u_pre, one float32 multiplication, fed the engine's u_pre.
//
// It is the stage that settles a uniform factor: if the engine multiplied by a
// hundred times the scale, this is where the hundred appears, and no comparison
// between two adapters can reach it.
func (r LoRAReference) ApplyScale(uPre []float32) []float32 {
	u := make([]float32, len(uPre))
	scale := float32(r.Scale)
	for i := range uPre {
		u[i] = scale * uPre[i]
	}
	return u
}

func (r LoRAReference) shapeFor(data []float32, nTokens, width int, what string) error {
	if err := r.Validate(); err != nil {
		return err
	}
	if nTokens <= 0 {
		return fmt.Errorf("cluster: at least one token is needed, got %d", nTokens)
	}
	if len(data) != nTokens*width {
		return fmt.Errorf("cluster: %s holds %d elements where %d tokens of %d is %d", what, len(data), nTokens, width, nTokens*width)
	}
	for i, v := range data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("cluster: %s is not finite at index %d", what, i)
		}
	}
	return nil
}

// AddTo is y = base + u, the last operation of the chain.
//
// Deliberately a float32 addition and not a float64 one rounded at the end: the
// operation under comparison is a single IEEE single-precision add, and a
// reference that computed it more precisely would report the engine's correct
// rounding as an error.
func AddTo(base, u []float32) ([]float32, error) {
	if len(base) != len(u) {
		return nil, fmt.Errorf("cluster: the base holds %d elements and the contribution %d", len(base), len(u))
	}
	y := make([]float32, len(base))
	for i := range base {
		y[i] = base[i] + u[i]
	}
	return y, nil
}

// Tolerance is what a comparison is allowed to differ by, declared before the
// numbers are seen.
//
// Floor exists because a relative error is meaningless where the expected value
// is near zero: dividing by it turns a rounding into a ratio of thousands. Below
// the floor only the absolute error is judged.
type Tolerance struct {
	Absolute float64
	Relative float64
	Floor    float64
}

// Exact is the tolerance for a fixture whose values are powers of two.
//
// With A[0,0] = 1 the sum that produces h is one term plus a row of zeroes, so
// it is exact whatever order it is added in; 0.5 and 2 are exact in binary; and
// y = base + u is a single IEEE addition. Nothing in the chain can round, so any
// difference at all is a defect and not a tolerance question.
var Exact = Tolerance{Absolute: 0, Relative: 0, Floor: 0}

// ActiveElement is one position the fixture was built to drive.
type ActiveElement struct {
	Index    int     `json:"index"`
	Expected float64 `json:"expected"`
	Observed float64 `json:"observed"`
}

// Comparison is what one stage of the chain came out as.
//
// Active carries the positions where the reference expects something other than
// zero. A sparse fixture drives few of them on purpose, and the maximum error is
// not diluted by the zeroes around them -- a wrong non-zero still shows up as a
// maximum. What the list adds is that the positions were exercised at all, with
// the numbers in the record rather than inferred from an error of zero.
type Comparison struct {
	Stage       string          `json:"stage"`
	Elements    int             `json:"elements"`
	ActiveCount int             `json:"active_count"`
	Active      []ActiveElement `json:"active"`
	Identical   bool            `json:"identical"`
	MaxAbsolute float64         `json:"max_absolute"`
	MaxRelative float64         `json:"max_relative"`
	WorstAt     int             `json:"worst_at"`
	Observed    float64         `json:"observed_at_worst"`
	Expected    float64         `json:"expected_at_worst"`
	Passed      bool            `json:"passed"`
}

// activeShown bounds what goes into the record. A dense fixture would otherwise
// write fourteen thousand numbers per stage into a file somebody has to read.
const activeShown = 12

// LoRAObserved is what the engine produced, as the capture hands it over.
//
// The activation fields can be large: x alone has n_in by n_tokens elements.
// A log line or a debug dump that rendered one could emit megabytes of
// numbers nobody reads, so the type says its shape and not its contents. A
// caller that wants the numbers takes the slices.
type LoRAObserved struct {
	X       []float32
	H       []float32
	UPre    []float32
	U       []float32
	YBase   []float32
	Y       []float32
	NTokens int
}

func (o LoRAObserved) shape() map[string]int {
	return map[string]int{
		"tokens": o.NTokens, "x": len(o.X), "h": len(o.H), "u_pre": len(o.UPre),
		"u": len(o.U), "y_base": len(o.YBase), "y": len(o.Y),
	}
}

// LogValue renders the shape, never the values.
func (o LoRAObserved) LogValue() slog.Value {
	s := o.shape()
	return slog.GroupValue(
		slog.Int("tokens", s["tokens"]),
		slog.Int("x_floats", s["x"]),
		slog.Int("h_floats", s["h"]),
		slog.Int("u_pre_floats", s["u_pre"]),
		slog.Int("u_floats", s["u"]),
		slog.Int("y_base_floats", s["y_base"]),
		slog.Int("y_floats", s["y"]),
	)
}

// MarshalJSON renders the shape, for the same reason.
func (o LoRAObserved) MarshalJSON() ([]byte, error) {
	return json.Marshal(o.shape())
}

// Stages compares every operation separately, each fed the engine's own input.
//
// The order is the order the graph computes in, so the first entry that did not
// pass names the first operation that disagrees. Feeding each stage the engine's
// previous output is what buys that: a reference chained to itself would carry a
// difference forward and report four failures for one defect, with no way to say
// which came first.
//
// The last entry is the end-to-end contribution, computed from the captured
// input alone. It is not there to catch compensating errors -- those escape it
// and are exactly what the local stages find. It consumes no intermediate, so
// it is sensitive where the staged comparison is blind: stages that all pass
// while it fails is a pattern that points at the capture. Under a tolerance it
// can equally come from per-stage differences accumulating, so what it gives the
// next reader is a place to look, not a cause.
func (r LoRAReference) Stages(o LoRAObserved, tolerance Tolerance) ([]Comparison, error) {
	if err := r.Validate(); err != nil {
		return nil, err
	}
	type stage struct {
		name     string
		observed []float32
		expected func() ([]float32, error)
	}
	stages := []stage{
		{"h = A*x", o.H, func() ([]float32, error) { return r.ApplyA(o.X, o.NTokens) }},
		{"u_pre = B*h", o.UPre, func() ([]float32, error) { return r.ApplyB(o.H, o.NTokens) }},
		{"u = scale*u_pre", o.U, func() ([]float32, error) { return r.ApplyScale(o.UPre), nil }},
		{"y = y_base + u", o.Y, func() ([]float32, error) { return AddTo(o.YBase, o.U) }},
		{"u end to end", o.U, func() ([]float32, error) {
			_, _, u, err := r.Chain(o.X, o.NTokens)
			return u, err
		}},
	}
	results := make([]Comparison, 0, len(stages))
	for _, s := range stages {
		expected, err := s.expected()
		if err != nil {
			return results, fmt.Errorf("cluster: %s: %w", s.name, err)
		}
		comparison, err := Compare(s.name, s.observed, expected, tolerance)
		if err != nil {
			return results, err
		}
		results = append(results, comparison)
		if !comparison.Passed {
			// Stopping here is the point: everything after this consumes a value
			// that is already wrong, and reporting those as failures too would
			// bury the one operation that has to be looked at.
			return results, nil
		}
	}
	return results, nil
}

// Compare measures one stage against its reference.
//
// Non-finite values on either side fail rather than compare: a NaN is equal to
// nothing, including itself, and a comparison that let one through would report
// a difference nobody could attribute.
func Compare(stage string, observed, expected []float32, tolerance Tolerance) (Comparison, error) {
	if len(observed) != len(expected) {
		return Comparison{}, fmt.Errorf("cluster: %s observed %d elements against %d expected", stage, len(observed), len(expected))
	}
	if len(observed) == 0 {
		return Comparison{}, fmt.Errorf("cluster: %s has nothing to compare", stage)
	}
	result := Comparison{Stage: stage, Elements: len(observed), Identical: true, WorstAt: -1}
	for i := range observed {
		got, want := float64(observed[i]), float64(expected[i])
		if math.IsNaN(got) || math.IsInf(got, 0) {
			return result, fmt.Errorf("cluster: %s observed %v at index %d", stage, got, i)
		}
		if math.IsNaN(want) || math.IsInf(want, 0) {
			return result, fmt.Errorf("cluster: %s expected %v at index %d", stage, want, i)
		}
		if got != want {
			result.Identical = false
		}
		absolute := math.Abs(got - want)
		relative := 0.0
		if math.Abs(want) > tolerance.Floor {
			relative = absolute / math.Abs(want)
		}
		if absolute > result.MaxAbsolute {
			result.MaxAbsolute = absolute
			result.WorstAt = i
			result.Observed = got
			result.Expected = want
		}
		if relative > result.MaxRelative {
			result.MaxRelative = relative
		}
		if want != 0 {
			result.ActiveCount++
			if len(result.Active) < activeShown {
				result.Active = append(result.Active, ActiveElement{Index: i, Expected: want, Observed: got})
			}
		}
	}
	if result.ActiveCount == 0 {
		// Every expected value is zero, so the stage compared nothing an
		// implementation returning zeroes would fail. Reported rather than
		// judged: a fixture can legitimately produce this, but a reader has to
		// be told before reading the pass as evidence of arithmetic.
		result.Stage += " (nenhum elemento ativo)"
	}
	result.Passed = result.MaxAbsolute <= tolerance.Absolute || result.MaxRelative <= tolerance.Relative
	if tolerance.Absolute == 0 && tolerance.Relative == 0 {
		result.Passed = result.Identical
	}
	return result, nil
}
