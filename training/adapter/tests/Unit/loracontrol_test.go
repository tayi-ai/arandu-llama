package unit_test

import (
	"testing"

	services "github.com/tayi-ai/arandu-llama/training/adapter"
)

// The case table's claims, checked against the arithmetic they claim.
//
// The control runs on a card, inside an image that costs forty minutes to
// build, and it reports a case as failed when two captures that should match do
// not. A case table that claimed two contributions were equal when they were
// not would spend all of that and then blame the engine. So the claims are
// checked here first, against the collapsed matrix
//
//	M[j][i] = adapter_scale * alpha / rank * sum_r B[r][j] * A[i][r]
//
// which is what llm_graph_context::build_lora_mm applies and what
// llama_adapter_lora_weight::get_scale composes. Nothing here loads a model.

func controlShape() services.LoRAShape {
	// An explicit synthetic projection geometry.
	return services.LoRAShape{
		A: "blk.35.ffn_down.weight.lora_a", B: "blk.35.ffn_down.weight.lora_b",
		NIn: 11008, Rank: 4, Alpha: 8,
	}
}

func casesByName(t *testing.T, s services.LoRAShape) map[string]services.LoRAControlCase {
	t.Helper()
	cases, err := services.LoRAControlCases(s)
	if err != nil {
		t.Fatal(err)
	}
	byName := make(map[string]services.LoRAControlCase, len(cases))
	for _, c := range cases {
		if _, clash := byName[c.Name]; clash {
			t.Fatalf("two cases are named %s", c.Name)
		}
		byName[c.Name] = c
	}
	return byName
}

func TestEveryClaimedEqualityIsOne(t *testing.T) {
	s := controlShape()
	cases, err := services.LoRAControlCases(s)
	if err != nil {
		t.Fatal(err)
	}
	byName := casesByName(t, s)

	for _, c := range cases {
		want := map[[2]int64]float64{}
		if c.Against != "" {
			against, ok := byName[c.Against]
			if !ok {
				t.Fatalf("%s is measured against %s, which is not a case", c.Name, c.Against)
			}
			want = against.Contribution(s)
		}
		got := c.Contribution(s)

		same := len(got) == len(want)
		if same {
			for k, v := range want {
				if got[k] != v {
					same = false
					break
				}
			}
		}
		// A case claiming identity whose matrix differs would fail on the card
		// and be read as a defect in the engine. A case claiming a difference
		// whose matrix is the same would pass against an engine that applies no
		// adapter at all.
		if same != c.Identical {
			t.Fatalf("%s claims identical=%v against %q, but its contribution is %v where the reference is %v",
				c.Name, c.Identical, c.Against, got, want)
		}
	}
}

func TestTheActiveCaseAppliesTheScaleTheFleetUses(t *testing.T) {
	s := controlShape()
	byName := casesByName(t, s)

	// alpha 8 over rank 4 at adapter_scale 1 is 2, and B[0][0] is 1/2, so the
	// single entry of the matrix is exactly 1. The number matters: it is what
	// V4 and V5 have to reproduce by other routes, and an engine that dropped
	// alpha would apply 1/2 while one that took the rank from the wrong tensor
	// would apply something else again.
	got := byName["V3_ativo"].Contribution(s)
	if len(got) != 1 {
		t.Fatalf("the active case has %d entries; one cell per factor is one entry", len(got))
	}
	if v := got[[2]int64{0, 0}]; v != 1 {
		t.Fatalf("the active case applies %v at (0,0) where alpha/rank = 2 over B = 0.5 gives 1", v)
	}
}

func TestTheNullCasesContributeNothing(t *testing.T) {
	s := controlShape()
	byName := casesByName(t, s)
	for _, name := range []string{"V1_nulo_original", "V2_nulo_espelhado", "V6_par_de_rank_cruzado"} {
		if got := byName[name].Contribution(s); len(got) != 0 {
			t.Fatalf("%s contributes %v; it is a null control and has to collapse to nothing", name, got)
		}
	}
}

func TestTheSignCaseInvertsAndDoesNotMerelyDiffer(t *testing.T) {
	s := controlShape()
	byName := casesByName(t, s)
	active := byName["V3_ativo"].Contribution(s)
	flipped := byName["V8_sinal"].Contribution(s)
	if len(active) != len(flipped) {
		t.Fatalf("the sign case has %d entries against the active case's %d", len(flipped), len(active))
	}
	for k, v := range active {
		if flipped[k] != -v {
			t.Fatalf("at %v the sign case applies %v where the active case applies %v", k, flipped[k], v)
		}
	}
}

func TestTheCellsLandWhereTheLayoutSaysTheyDo(t *testing.T) {
	s := controlShape()
	// ne[0] varies fastest, so A is [n_in, r] with (i, r) at r*n_in+i and B is
	// [r, n_out] with (r, j) at j*rank+r. Reading this back from the index is
	// what Contribution does, and getting it backwards is exactly the confusion
	// the rank cases exist to catch in the engine.
	if cell := s.CellA(7, 3, 1); cell.Index != 3*11008+7 {
		t.Fatalf("A(7,3) landed at %d where the layout puts it at %d", cell.Index, 3*11008+7)
	}
	if cell := s.CellB(3, 5, 1); cell.Index != 5*4+3 {
		t.Fatalf("B(3,5) landed at %d where the layout puts it at %d", cell.Index, 5*4+3)
	}
	if _, err := services.LoRAControlCases(services.LoRAShape{A: "a", B: "b", NIn: 8, Rank: 1, Alpha: 8}); err == nil {
		t.Fatal("a rank of one was accepted; the rank pairing cases need two components")
	}
	if _, err := services.LoRAControlCases(services.LoRAShape{A: "a", B: "b", NIn: 8, Rank: 4, Alpha: 0}); err == nil {
		t.Fatal("an alpha of zero was accepted; llama.cpp would fall back to the bare adapter scale")
	}
}
