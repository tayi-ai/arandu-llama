package adapter

import "fmt"

// The cases of the positive control, and the arithmetic that says what each one
// is supposed to equal.
//
// They live here, with no build tag, rather than inside the command that runs
// them, for one reason: the command needs a card and forty minutes of image
// build, and a case table that claims two contributions are equal when they are
// not would burn both and report the engine as broken. Here the claim is
// checkable in milliseconds against the same formula llama.cpp implements.
//
// llama.cpp computes, in llm_graph_context::build_lora_mm:
//
//	ab = scale * B * (A * x),   scale = adapter_scale * alpha / rank
//
// with rank read from the tensor -- llama_adapter_lora_weight::get_scale takes
// it as b->ne[0], not from metadata. Collapsing the two factors, the
// contribution is M*x for
//
//	M[j][i] = scale * sum_r B[r][j] * A[i][r]
//
// and x never appears. That is what makes the control possible at all: the
// projection's input is not observable from outside the graph, but two adapters
// with the same M have to produce the same output whatever it is fed.

// LoRAControlCase is one fixture of the control, and the relation it has to the
// case named in Against.
type LoRAControlCase struct {
	Name string
	// Against names the case this one is compared with. Empty means the base,
	// with no adapter applied at all.
	Against string
	// Identical says which way the expectation runs: true when the two
	// contributions are equal by construction, false when the case exists to
	// prove the engine reacts to a contribution that is not zero.
	Identical bool
	Why       string
	// Alpha is written into the fixture's metadata; Scale is handed to
	// llama_set_adapters_lora. Their product over the rank is the scale applied.
	Alpha float32
	Scale float32
	Cells []StructuredCell
}

// LoRAShape is what the template says about the pair under test, read from the
// file rather than assumed.
type LoRAShape struct {
	A     string
	B     string
	NIn   int64
	Rank  int64
	Alpha float32
}

// CellA names element (i, r) of the A factor, which is stored [n_in, r] with
// ne[0] varying fastest.
func (s LoRAShape) CellA(i, r int64, value float32) StructuredCell {
	return StructuredCell{Tensor: s.A, Index: r*s.NIn + i, Value: value}
}

// CellB names element (r, j) of the B factor, stored [r, n_out].
func (s LoRAShape) CellB(r, j int64, value float32) StructuredCell {
	return StructuredCell{Tensor: s.B, Index: j*s.Rank + r, Value: value}
}

// LoRAControlCases builds the case table for one projection.
//
// Every case sets at most one cell per factor, so the contribution of each is a
// single outer product and the equalities are checkable by hand. The two that
// expect a difference are there because a table of equalities alone would pass
// against an engine that applied no adapter at all.
func LoRAControlCases(s LoRAShape) ([]LoRAControlCase, error) {
	if s.NIn <= 0 || s.Rank < 2 {
		return nil, fmt.Errorf("cluster: the control needs n_in above zero and rank at least two; the template says %d and %d", s.NIn, s.Rank)
	}
	if s.Alpha <= 0 {
		return nil, fmt.Errorf("cluster: the template declares alpha %v; the effective scale would not be a scale", s.Alpha)
	}
	return []LoRAControlCase{
		{
			Name: "V1_nulo_original", Against: "", Identical: true,
			Why:   "A finito com B em zero: s*B*A e a matriz nula, entao a captura tem de ser a da base",
			Alpha: s.Alpha, Scale: 1,
			Cells: []StructuredCell{s.CellA(0, 0, 1)},
		},
		{
			Name: "V2_nulo_espelhado", Against: "", Identical: true,
			Why:   "B finito com A em zero: a matriz nula pelo outro lado",
			Alpha: s.Alpha, Scale: 1,
			Cells: []StructuredCell{s.CellB(0, 0, 0.5)},
		},
		{
			Name: "V3_ativo", Against: "", Identical: false,
			Why:   "a contribuicao nao e nula; a captura tem de mudar em relacao a base",
			Alpha: s.Alpha, Scale: 1,
			Cells: []StructuredCell{s.CellA(0, 0, 1), s.CellB(0, 0, 0.5)},
		},
		{
			Name: "V4_escala_do_adaptador", Against: "V3_ativo", Identical: true,
			Why:   "adapter_scale dobrado com B pela metade: o produto scale*B nao muda",
			Alpha: s.Alpha, Scale: 2,
			Cells: []StructuredCell{s.CellA(0, 0, 1), s.CellB(0, 0, 0.25)},
		},
		{
			Name: "V5_composicao_do_alpha", Against: "V3_ativo", Identical: true,
			Why:   "alpha pela metade com B dobrado: alpha/rank compensa e a matriz e a mesma",
			Alpha: s.Alpha / 2, Scale: 1,
			Cells: []StructuredCell{s.CellA(0, 0, 1), s.CellB(0, 0, 1)},
		},
		{
			Name: "V6_par_de_rank_cruzado", Against: "", Identical: true,
			Why:   "A na componente 0 e B na componente 1: nenhum r tem os dois, entao a soma e nula",
			Alpha: s.Alpha, Scale: 1,
			Cells: []StructuredCell{s.CellA(0, 0, 1), s.CellB(1, 0, 0.5)},
		},
		{
			Name: "V7_par_de_rank_deslocado", Against: "V3_ativo", Identical: true,
			Why:   "A e B ambos na componente 1: a mesma matriz de V3, por outro indice de rank",
			Alpha: s.Alpha, Scale: 1,
			Cells: []StructuredCell{s.CellA(0, 1, 1), s.CellB(1, 0, 0.5)},
		},
		{
			Name: "V8_sinal", Against: "V3_ativo", Identical: false,
			Why:   "B com o sinal trocado: a matriz inverte, entao a captura tem de diferir de V3",
			Alpha: s.Alpha, Scale: 1,
			Cells: []StructuredCell{s.CellA(0, 0, 1), s.CellB(0, 0, -0.5)},
		},
	}, nil
}

// Contribution is the matrix M the engine should apply for this case, computed
// from the case's own cells by the formula above.
//
// It is the independent reference: it does not call anything the engine calls,
// and it is what a test compares the case table's claims against. Returned as a
// dense map from (j, i) so that a case touching one cell per factor produces a
// handful of entries rather than n_out by n_in zeroes.
func (c LoRAControlCase) Contribution(s LoRAShape) map[[2]int64]float64 {
	a := map[[2]int64]float64{}
	b := map[[2]int64]float64{}
	for _, cell := range c.Cells {
		switch cell.Tensor {
		case s.A:
			// index = r*n_in + i
			a[[2]int64{cell.Index % s.NIn, cell.Index / s.NIn}] = float64(cell.Value)
		case s.B:
			// index = j*rank + r
			b[[2]int64{cell.Index % s.Rank, cell.Index / s.Rank}] = float64(cell.Value)
		}
	}
	scale := float64(c.Scale) * float64(c.Alpha) / float64(s.Rank)
	m := map[[2]int64]float64{}
	for bk, bv := range b {
		r, j := bk[0], bk[1]
		for ak, av := range a {
			i, ar := ak[0], ak[1]
			if ar != r {
				continue
			}
			if v := scale * bv * av; v != 0 {
				m[[2]int64{j, i}] += v
			}
		}
	}
	for k, v := range m {
		if v == 0 {
			delete(m, k)
		}
	}
	return m
}
