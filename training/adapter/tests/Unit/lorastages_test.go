package unit_test

import (
	"testing"

	services "github.com/tayi-ai/arandu-llama/training/adapter"
)

// The staged comparison, and the property that makes it worth having.
//
// A reference chained to itself carries a difference forward: one defect
// reports as four failures with no way to say which came first. Feeding each
// stage the engine's own output isolates it. These tests fabricate an engine
// that is wrong in one operation at a time and check that the first entry which
// does not pass names that operation.
//
// They also pin the direction the end-to-end check runs in, which is the
// opposite of what is easy to assume: compensating errors ESCAPE it, and the
// local stages are what find them.

func stagesReference() services.LoRAReference {
	// n_in 2, rank 1, n_out 2. A = [1, 0] over (i, r); B = [3, 5] over (r, j).
	return services.LoRAReference{
		A: []float32{1, 0}, B: []float32{3, 5},
		NIn: 2, Rank: 1, NOut: 2, Scale: 2,
	}
}

// correctEngine is what a faultless run of the reference's own arithmetic looks
// like, so a test can spoil exactly one field of it.
func correctEngine(t *testing.T, r services.LoRAReference, x []float32, base []float32) services.LoRAObserved {
	t.Helper()
	h, err := r.ApplyA(x, 1)
	if err != nil {
		t.Fatal(err)
	}
	uPre, err := r.ApplyB(h, 1)
	if err != nil {
		t.Fatal(err)
	}
	u := r.ApplyScale(uPre)
	y, err := services.AddTo(base, u)
	if err != nil {
		t.Fatal(err)
	}
	return services.LoRAObserved{X: x, H: h, UPre: uPre, U: u, YBase: base, Y: y, NTokens: 1}
}

func TestAFaultlessEngineClearsEveryStage(t *testing.T) {
	r := stagesReference()
	observed := correctEngine(t, r, []float32{4, 9}, []float32{100, 200})
	results, err := r.Stages(observed, services.Exact)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 5 {
		t.Fatalf("%d stages ran where five were expected: %+v", len(results), results)
	}
	for _, result := range results {
		if !result.Passed {
			t.Fatalf("stage %q failed on a faultless engine: %+v", result.Stage, result)
		}
	}
}

func TestTheFirstFailingStageNamesTheOperation(t *testing.T) {
	r := stagesReference()
	for _, c := range []struct {
		name  string
		spoil func(*services.LoRAObserved)
		stage string
	}{
		{"A", func(o *services.LoRAObserved) { o.H[0] *= 3 }, "h = A*x"},
		{"B", func(o *services.LoRAObserved) { o.UPre[0] *= 3 }, "u_pre = B*h"},
		{"scale", func(o *services.LoRAObserved) { o.U[0] *= 100 }, "u = scale*u_pre"},
		{"the sum", func(o *services.LoRAObserved) { o.Y[0] += 1 }, "y = y_base + u"},
	} {
		t.Run(c.name, func(t *testing.T) {
			observed := correctEngine(t, r, []float32{4, 9}, []float32{100, 200})
			c.spoil(&observed)
			results, err := r.Stages(observed, services.Exact)
			if err != nil {
				t.Fatal(err)
			}
			last := results[len(results)-1]
			if last.Passed {
				t.Fatalf("spoiling %s left every stage passing: %+v", c.name, results)
			}
			if last.Stage != c.stage {
				t.Fatalf("spoiling %s failed at %q where %q is the operation", c.name, last.Stage, c.stage)
			}
			// Everything before the failure has to have passed, or the report
			// does not localise anything.
			for _, earlier := range results[:len(results)-1] {
				if !earlier.Passed {
					t.Fatalf("spoiling %s also failed the earlier stage %q", c.name, earlier.Stage)
				}
			}
		})
	}
}

func TestCompensatingErrorsEscapeEndToEndAndTheStagesCatchThem(t *testing.T) {
	// The engine applies B twice and the scale at half. End to end that is
	// 2*B*h * s/2 = s*B*h, the right answer -- so a check that only looked at
	// the contribution as a whole would approve it.
	r := stagesReference()
	observed := correctEngine(t, r, []float32{4, 9}, []float32{100, 200})
	for i := range observed.UPre {
		observed.UPre[i] *= 2
	}
	// u is left at its correct value, which is what "compensating" means: the
	// halved scale undoes the doubled B.

	// End to end, from x alone, u is right.
	_, _, endToEnd, err := r.Chain(observed.X, observed.NTokens)
	if err != nil {
		t.Fatal(err)
	}
	whole, err := services.Compare("u end to end", observed.U, endToEnd, services.Exact)
	if err != nil {
		t.Fatal(err)
	}
	if !whole.Passed {
		t.Fatalf("the end-to-end check caught a compensating pair it cannot see: %+v", whole)
	}

	// The staged comparison does not let it through.
	results, err := r.Stages(observed, services.Exact)
	if err != nil {
		t.Fatal(err)
	}
	last := results[len(results)-1]
	if last.Passed {
		t.Fatal("the staged comparison passed a doubled B")
	}
	if last.Stage != "u_pre = B*h" {
		t.Fatalf("the first failing stage is %q where the doubled operation is u_pre = B*h", last.Stage)
	}
}

func TestTheEndToEndStageCatchesAMisCapturedIntermediate(t *testing.T) {
	// Every local stage agrees pairwise, because each was computed from the one
	// before -- but the whole chain does not follow from x. That is what a node
	// matched to the wrong tensor looks like.
	//
	// What this pins is that the case IS detected, here. It does not establish
	// that a mis-capture is the only thing that can produce the pattern: under a
	// declared tolerance, differences accepted at each stage can add up past the
	// limit at the end, and a wrong reference would do it too. The pattern
	// orients the next reader; it does not name the cause.
	r := stagesReference()
	observed := correctEngine(t, r, []float32{4, 9}, []float32{100, 200})
	// The captured h belongs to a different input than the captured x.
	observed.H[0] += 7
	uPre, err := r.ApplyB(observed.H, 1)
	if err != nil {
		t.Fatal(err)
	}
	observed.UPre = uPre
	observed.U = r.ApplyScale(uPre)
	y, err := services.AddTo(observed.YBase, observed.U)
	if err != nil {
		t.Fatal(err)
	}
	observed.Y = y

	results, err := r.Stages(observed, services.Exact)
	if err != nil {
		t.Fatal(err)
	}
	last := results[len(results)-1]
	if last.Passed {
		t.Fatal("a mis-captured intermediate passed every stage")
	}
	if last.Stage != "h = A*x" {
		t.Fatalf("the failure surfaced at %q; an h that does not follow from x fails there first", last.Stage)
	}
}
