package unit_test

import (
	"math"
	"testing"

	services "github.com/tayi-ai/arandu-llama/training/adapter"
)

// The reference, checked before anything is checked against it.
//
// An independent reference is only worth what its own arithmetic is worth. If
// this file were wrong, the oracle would report the engine as broken and the
// next round would be spent looking in the wrong place -- or, worse, it would
// agree with a broken engine for a matching reason. So the reference is first
// exercised against cases small enough to work out by hand.
//
// Layout, restated because it is the part that is easy to invert: ne[0] varies
// fastest, so x of [n_in, n_tokens] has (i, t) at t*n_in+i, A of [n_in, rank]
// has (i, r) at r*n_in+i, and B of [rank, n_out] has (r, j) at j*rank+r.

func TestTheReferenceComputesTheChainByHand(t *testing.T) {
	// n_in 2, rank 2, n_out 2, one token.
	//
	//   A = [[1, 3],     as (i, r): A(0,0)=1 A(1,0)=2 A(0,1)=3 A(1,1)=4
	//        [2, 4]]
	//   x = [5, 6]
	//   h(r) = sum_i A(i,r) x(i)   ->  h0 = 1*5 + 2*6 = 17
	//                                  h1 = 3*5 + 4*6 = 39
	//   B as (r, j): B(0,0)=7 B(1,0)=8 B(0,1)=9 B(1,1)=10
	//   uPre(j) = sum_r B(r,j) h(r) -> u0 = 7*17 + 8*39 = 431
	//                                  u1 = 9*17 + 10*39 = 543
	//   scale 2                     -> u  = [862, 1086]
	reference := services.LoRAReference{
		A:   []float32{1, 2, 3, 4},  // r=0: (1,2); r=1: (3,4)
		B:   []float32{7, 8, 9, 10}, // j=0: (7,8); j=1: (9,10)
		NIn: 2, Rank: 2, NOut: 2, Scale: 2,
	}
	h, uPre, u, err := reference.Chain([]float32{5, 6}, 1)
	if err != nil {
		t.Fatal(err)
	}
	expect := func(name string, got []float32, want ...float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s holds %d elements where %d were worked out", name, len(got), len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("%s[%d] is %v where the hand calculation gives %v", name, i, got[i], want[i])
			}
		}
	}
	expect("h", h, 17, 39)
	expect("uPre", uPre, 431, 543)
	expect("u", u, 862, 1086)

	y, err := services.AddTo([]float32{100, 200}, u)
	if err != nil {
		t.Fatal(err)
	}
	expect("y", y, 962, 1286)
}

func TestTheReferenceKeepsTokensApart(t *testing.T) {
	// The same factors over two tokens: each token's row has to go through on
	// its own. A reference that summed across tokens would pass every
	// single-token test and fail nothing until the oracle ran.
	reference := services.LoRAReference{
		A: []float32{1, 0}, B: []float32{1}, NIn: 2, Rank: 1, NOut: 1, Scale: 1,
	}
	_, _, u, err := reference.Chain([]float32{5, 6, 7, 8}, 2)
	if err != nil {
		t.Fatal(err)
	}
	// A(0,0)=1, A(1,0)=0, so h = x(0) for each token: 5 and 7.
	if u[0] != 5 || u[1] != 7 {
		t.Fatalf("the two tokens came out as %v; each row is its own", u)
	}
}

func TestTheStructuredFixtureIsExactInTheReference(t *testing.T) {
	// The fixture the oracle uses: one component of A at 1, one of B at 0.5,
	// scale 2. The contribution is then exactly the input's first element, and
	// nothing in the chain can round -- which is why the declared tolerance for
	// it is zero.
	const nIn, rank, nOut = 8, 4, 3
	a := make([]float32, nIn*rank)
	b := make([]float32, rank*nOut)
	a[0] = 1   // (i=0, r=0)
	b[0] = 0.5 // (r=0, j=0)
	reference := services.LoRAReference{A: a, B: b, NIn: nIn, Rank: rank, NOut: nOut, Scale: 2}

	x := make([]float32, nIn)
	for i := range x {
		x[i] = float32(i) + 0.375 // not round numbers, and still exact through this chain
	}
	_, _, u, err := reference.Chain(x, 1)
	if err != nil {
		t.Fatal(err)
	}
	if u[0] != x[0] {
		t.Fatalf("the contribution is %v where the fixture makes it exactly x[0] = %v", u[0], x[0])
	}
	for j := 1; j < nOut; j++ {
		if u[j] != 0 {
			t.Fatalf("output %d is %v; only the named one is driven", j, u[j])
		}
	}
}

func TestTheReferenceRefusesWhatItCannotCompute(t *testing.T) {
	good := services.LoRAReference{A: []float32{1, 0}, B: []float32{1}, NIn: 2, Rank: 1, NOut: 1, Scale: 1}
	if _, _, _, err := good.Chain([]float32{1, 2, 3}, 1); err == nil {
		t.Fatal("an input of the wrong length was accepted")
	}
	if _, _, _, err := good.Chain([]float32{1, float32(math.NaN())}, 1); err == nil {
		t.Fatal("a non-finite input was accepted; it would propagate and read as a difference")
	}
	bad := services.LoRAReference{A: []float32{1}, B: []float32{1}, NIn: 2, Rank: 1, NOut: 1, Scale: 1}
	if err := bad.Validate(); err == nil {
		t.Fatal("factors that do not describe the declared shapes were accepted")
	}
	if _, err := services.AddTo([]float32{1, 2}, []float32{1}); err == nil {
		t.Fatal("adding tensors of different lengths was accepted")
	}
}

func TestTheComparisonCatchesAUniformFactor(t *testing.T) {
	// The defect the whole oracle exists for: a contribution that is a hundred
	// times too large. Every equality between two adapters survives it; a
	// comparison against the reference does not.
	expected := []float32{1, 2, 3, 4}
	observed := []float32{100, 200, 300, 400}
	result, err := services.Compare("u", observed, expected, services.Exact)
	if err != nil {
		t.Fatal(err)
	}
	if result.Passed {
		t.Fatal("a hundredfold contribution passed the comparison")
	}
	if result.MaxRelative < 98 {
		t.Fatalf("the relative error came out as %v for a hundredfold difference", result.MaxRelative)
	}
	if result.WorstAt != 3 || result.Observed != 400 || result.Expected != 4 {
		t.Fatalf("the worst element was reported as %d: observed %v against %v", result.WorstAt, result.Observed, result.Expected)
	}
}

func TestAnExactComparisonDoesNotAcceptRounding(t *testing.T) {
	expected := []float32{1, 2, 3}
	observed := []float32{1, 2, 3.0000002}
	result, err := services.Compare("u", observed, expected, services.Exact)
	if err != nil {
		t.Fatal(err)
	}
	if result.Passed || result.Identical {
		t.Fatal("a one-ulp difference passed an exact comparison; the fixture is built so nothing can round")
	}
	// The same numbers under a declared tolerance do pass, which is what the
	// non-power-of-two case runs under.
	loose, err := services.Compare("u", observed, expected, services.Tolerance{Relative: 1e-5, Floor: 1e-6})
	if err != nil {
		t.Fatal(err)
	}
	if !loose.Passed {
		t.Fatalf("a one-ulp difference failed a 1e-5 relative tolerance: %+v", loose)
	}
}

func TestTheComparisonRefusesNonFiniteValues(t *testing.T) {
	if _, err := services.Compare("u", []float32{float32(math.NaN())}, []float32{1}, services.Exact); err == nil {
		t.Fatal("a NaN observation was compared instead of refused")
	}
	if _, err := services.Compare("u", []float32{1}, []float32{float32(math.Inf(1))}, services.Exact); err == nil {
		t.Fatal("an infinite expectation was compared instead of refused")
	}
	if _, err := services.Compare("u", []float32{1, 2}, []float32{1}, services.Exact); err == nil {
		t.Fatal("tensors of different lengths were compared")
	}
}

func TestTheRelativeFloorKeepsNearZeroFromExploding(t *testing.T) {
	// Where the expectation is essentially zero, a relative error is a ratio of
	// two roundings and says nothing. Below the floor only the absolute error is
	// judged, and this case has to pass on it.
	expected := []float32{1e-9}
	observed := []float32{2e-9}
	result, err := services.Compare("u", observed, expected, services.Tolerance{Absolute: 1e-6, Relative: 1e-5, Floor: 1e-6})
	if err != nil {
		t.Fatal(err)
	}
	if result.MaxRelative != 0 {
		t.Fatalf("a value under the floor produced a relative error of %v", result.MaxRelative)
	}
	if !result.Passed {
		t.Fatalf("a difference of 1e-9 failed an absolute tolerance of 1e-6: %+v", result)
	}
}
