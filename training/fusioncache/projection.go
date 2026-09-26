package fusioncache

import (
	"fmt"
	"math"
)

// SampleIdentity binds a feature pair to one teacher-forced gold prefix.
type SampleIdentity struct {
	DatasetSHA256       string `json:"dataset_sha256"`
	ExampleID           string `json:"example_id"`
	Role                string `json:"role"`
	TargetIndex         int    `json:"target_index"`
	TeacherPrefixSHA256 string `json:"teacher_prefix_sha256"`
	StudentPrefixSHA256 string `json:"student_prefix_sha256"`
}

// FeatureSample contains matched feature vectors from independently identified models.
type FeatureSample struct {
	Identity SampleIdentity
	Source   []float64
	Target   []float64
}

// MatchingPlan freezes layer correspondence, ridge strength and example splits
// before fitting. Heldout contains calibration validation rows, never final eval.
type MatchingPlan struct {
	ID                 string           `json:"id"`
	SourceModel        ModelIdentity    `json:"source_model"`
	TargetModel        ModelIdentity    `json:"target_model"`
	TokenMappingSHA256 string           `json:"token_mapping_sha256"`
	Source             FeatureSpec      `json:"source"`
	Target             FeatureSpec      `json:"target"`
	Ridge              float64          `json:"ridge"`
	Fit                []SampleIdentity `json:"fit"`
	Heldout            []SampleIdentity `json:"heldout"`
}

// ProjectionLimits bounds the input dimensions and working/storage float64 counts.
type ProjectionLimits struct {
	MaxSamples   int
	MaxDimension int
	MaxElements  int64
}

// Projection stores P for min ||XP^T-Y||_F^2 + ridge*||P||_F^2.
// Dual stores X and (XX^T+ridge*I)^-1 Y instead of materializing a wide P.
// SHA256 binds all fields except itself; Apply refuses any mutated coefficients.
type Projection struct {
	PlanSHA256      string      `json:"plan_sha256"`
	SourceDimension int         `json:"source_dimension"`
	TargetDimension int         `json:"target_dimension"`
	Dual            bool        `json:"dual"`
	Basis           [][]float64 `json:"basis,omitempty"`
	Coefficients    [][]float64 `json:"coefficients"`
	SHA256          string      `json:"sha256"`
}

// FitReport contains reconstruction error on disjoint, predeclared partitions.
// These errors are not evidence of language-model capability improvement.
type FitReport struct {
	PlanSHA256          string
	FitSquaredError     float64
	FitValues           int64
	HeldoutSquaredError float64
	HeldoutValues       int64
	Dual                bool
}

func validFeatureSpec(f FeatureSpec, maximum int) bool {
	return f.Layer >= 0 && f.Tensor != "" && (f.DType == "float32" || f.DType == "float64") && f.Dimension > 0 && f.Dimension <= maximum
}

func sampleExampleKey(s SampleIdentity) string { return s.DatasetSHA256 + "/" + s.ExampleID }

func validatePlan(p MatchingPlan, expected string, fit, heldout []FeatureSample, limits ProjectionLimits) error {
	if limits.MaxSamples < 1 || limits.MaxDimension < 1 || limits.MaxElements < 1 {
		return fmt.Errorf("%w: positive projection bounds required", ErrContract)
	}
	digest, err := Digest(p)
	if err != nil || !validSHA(expected) || digest != expected || p.ID == "" || !validModel(p.SourceModel) || !validModel(p.TargetModel) || !validSHA(p.TokenMappingSHA256) || !validFeatureSpec(p.Source, limits.MaxDimension) || !validFeatureSpec(p.Target, limits.MaxDimension) || !finite(p.Ridge) || p.Ridge <= 0 {
		return fmt.Errorf("%w: invalid or changed matching plan", ErrContract)
	}
	if len(fit) == 0 || len(heldout) == 0 || len(fit) > limits.MaxSamples || len(heldout) > limits.MaxSamples-len(fit) || len(p.Fit) != len(fit) || len(p.Heldout) != len(heldout) {
		return fmt.Errorf("%w: fitting and disjoint heldout required within sample budget", ErrContract)
	}
	seenRows := map[string]bool{}
	fitExamples := map[string]bool{}
	for partition, samples := range [][]FeatureSample{fit, heldout} {
		identities := p.Fit
		if partition == 1 {
			identities = p.Heldout
		}
		for i, sample := range samples {
			id := sample.Identity
			key := sampleExampleKey(id)
			rowKey := fmt.Sprintf("%s/%d", key, id.TargetIndex)
			if id != identities[i] || !allowedRole(id.Role) || !validSHA(id.DatasetSHA256) || id.ExampleID == "" || id.TargetIndex < 1 || !validSHA(id.TeacherPrefixSHA256) || !validSHA(id.StudentPrefixSHA256) || seenRows[rowKey] || (partition == 1 && fitExamples[key]) {
				return fmt.Errorf("%w: feature sample identity, role or split differs", ErrContract)
			}
			seenRows[rowKey] = true
			if partition == 0 {
				fitExamples[key] = true
			}
			if len(sample.Source) != p.Source.Dimension || len(sample.Target) != p.Target.Dimension {
				return fmt.Errorf("%w: feature sample dimensions differ", ErrContract)
			}
			for _, vector := range [][]float64{sample.Source, sample.Target} {
				for _, value := range vector {
					if !finite(value) {
						return fmt.Errorf("%w: nonfinite feature sample", ErrContract)
					}
				}
			}
		}
	}
	return nil
}

// FitProjection fits only plan.Fit and evaluates plan.Heldout without selecting
// layers, ridge or candidates. The caller must separately approve those choices.
// The solver uses min(samples, source dimension)^2 memory for its Gram matrix.
func FitProjection(plan MatchingPlan, expectedPlanSHA256 string, fit, heldout []FeatureSample, limits ProjectionLimits) (Projection, FitReport, error) {
	if err := validatePlan(plan, expectedPlanSHA256, fit, heldout, limits); err != nil {
		return Projection{}, FitReport{}, err
	}
	n, d, out := len(fit), plan.Source.Dimension, plan.Target.Dimension
	dual := n < d
	size := d
	if dual {
		size = n
	}
	// Count input vectors, Gram, solve RHS, stored basis and application scratch.
	remaining := limits.MaxElements
	addProduct := func(a, b int) bool {
		if int64(a) > remaining/int64(b) {
			return false
		}
		remaining -= int64(a) * int64(b)
		return true
	}
	if !addProduct(n+len(heldout), d) || !addProduct(n+len(heldout), out) || !addProduct(size, size) || !addProduct(size, out) || !addProduct(1, size) || !addProduct(1, out) || (dual && !addProduct(n, d)) {
		return Projection{}, FitReport{}, fmt.Errorf("%w: projection memory budget exceeded", ErrContract)
	}
	gram := make([]float64, size*size)
	right := make([][]float64, size)
	for i := range right {
		right[i] = make([]float64, out)
	}
	if dual {
		for i := 0; i < n; i++ {
			copy(right[i], fit[i].Target)
			for j := 0; j <= i; j++ {
				value := dot(fit[i].Source, fit[j].Source)
				gram[i*size+j], gram[j*size+i] = value, value
			}
		}
	} else {
		for _, sample := range fit {
			for i, value := range sample.Source {
				for j := 0; j <= i; j++ {
					gram[i*size+j] += value * sample.Source[j]
				}
				for j, target := range sample.Target {
					right[i][j] += value * target
				}
			}
		}
		for i := 0; i < size; i++ {
			for j := 0; j < i; j++ {
				gram[j*size+i] = gram[i*size+j]
			}
		}
	}
	for i := 0; i < size; i++ {
		gram[i*size+i] += plan.Ridge
	}
	if err := choleskySolve(gram, right); err != nil {
		return Projection{}, FitReport{}, err
	}
	p := Projection{PlanSHA256: expectedPlanSHA256, SourceDimension: d, TargetDimension: out, Dual: dual, Coefficients: right}
	if dual {
		for _, sample := range fit {
			p.Basis = append(p.Basis, append([]float64(nil), sample.Source...))
		}
	}
	var err error
	p.SHA256, err = projectionDigest(p)
	if err != nil {
		return Projection{}, FitReport{}, err
	}
	report := FitReport{PlanSHA256: expectedPlanSHA256, Dual: dual}
	for partition, samples := range [][]FeatureSample{fit, heldout} {
		squared := 0.0
		for _, sample := range samples {
			prediction, err := p.apply(sample.Source)
			if err != nil {
				return Projection{}, FitReport{}, err
			}
			for i, target := range sample.Target {
				difference := prediction[i] - target
				squared += difference * difference
			}
		}
		if !finite(squared) {
			return Projection{}, FitReport{}, fmt.Errorf("%w: nonfinite reconstruction error", ErrContract)
		}
		if partition == 0 {
			report.FitSquaredError, report.FitValues = squared, int64(len(samples))*int64(out)
		} else {
			report.HeldoutSquaredError, report.HeldoutValues = squared, int64(len(samples))*int64(out)
		}
	}
	return p, report, nil
}

func dot(a, b []float64) float64 {
	value := 0.0
	for i := range a {
		value += a[i] * b[i]
	}
	return value
}

// choleskySolve overwrites a with L and b with the deterministic SPD solution.
func choleskySolve(a []float64, b [][]float64) error {
	n := len(b)
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			value := a[i*n+j]
			for k := 0; k < j; k++ {
				value -= a[i*n+k] * a[j*n+k]
			}
			if !finite(value) {
				return fmt.Errorf("%w: nonfinite ridge system", ErrContract)
			}
			if i == j {
				if value <= 0 {
					return fmt.Errorf("%w: ridge system is numerically nonpositive", ErrContract)
				}
				a[i*n+j] = math.Sqrt(value)
			} else {
				a[i*n+j] = value / a[j*n+j]
			}
		}
	}
	for column := range b[0] {
		for i := 0; i < n; i++ {
			value := b[i][column]
			for j := 0; j < i; j++ {
				value -= a[i*n+j] * b[j][column]
			}
			b[i][column] = value / a[i*n+i]
		}
		for i := n - 1; i >= 0; i-- {
			value := b[i][column]
			for j := i + 1; j < n; j++ {
				value -= a[j*n+i] * b[j][column]
			}
			b[i][column] = value / a[i*n+i]
			if !finite(b[i][column]) {
				return fmt.Errorf("%w: nonfinite ridge solution", ErrContract)
			}
		}
	}
	return nil
}

func projectionDigest(p Projection) (string, error) { p.SHA256 = ""; return Digest(p) }

// Validate checks coefficient identity, representation and explicit storage bounds.
func (p Projection) Validate(expectedPlanSHA256 string, limits ProjectionLimits) error {
	if !validSHA(expectedPlanSHA256) || p.PlanSHA256 != expectedPlanSHA256 || !validSHA(p.SHA256) || p.SourceDimension <= 0 || p.SourceDimension > limits.MaxDimension || p.TargetDimension <= 0 || p.TargetDimension > limits.MaxDimension || limits.MaxElements <= 0 {
		return fmt.Errorf("%w: projection identity or dimensions differ", ErrContract)
	}
	rows := p.SourceDimension
	if p.Dual {
		rows = len(p.Basis)
		if rows <= 0 || rows >= p.SourceDimension || rows > limits.MaxSamples {
			return fmt.Errorf("%w: invalid dual basis", ErrContract)
		}
	} else if len(p.Basis) != 0 {
		return fmt.Errorf("%w: primal projection has a basis", ErrContract)
	}
	if len(p.Coefficients) != rows {
		return fmt.Errorf("%w: projection coefficient rows differ", ErrContract)
	}
	remaining := limits.MaxElements
	for _, matrix := range []struct {
		values  [][]float64
		columns int
	}{{p.Basis, p.SourceDimension}, {p.Coefficients, p.TargetDimension}} {
		for _, row := range matrix.values {
			if len(row) != matrix.columns || int64(len(row)) > remaining {
				return fmt.Errorf("%w: projection storage bound exceeded", ErrContract)
			}
			remaining -= int64(len(row))
			for _, value := range row {
				if !finite(value) {
					return fmt.Errorf("%w: nonfinite projection", ErrContract)
				}
			}
		}
	}
	digest, err := projectionDigest(p)
	if err != nil || digest != p.SHA256 {
		return fmt.Errorf("%w: projection digest differs", ErrContract)
	}
	return nil
}

// Apply maps one source vector after validating coefficient and plan identities.
func (p Projection) Apply(source []float64, expectedPlanSHA256 string, limits ProjectionLimits) ([]float64, error) {
	if err := p.Validate(expectedPlanSHA256, limits); err != nil {
		return nil, err
	}
	return p.apply(source)
}

func (p Projection) apply(source []float64) ([]float64, error) {
	if len(source) != p.SourceDimension {
		return nil, fmt.Errorf("%w: projection input dimension differs", ErrContract)
	}
	for _, value := range source {
		if !finite(value) {
			return nil, fmt.Errorf("%w: nonfinite projection input", ErrContract)
		}
	}
	result := make([]float64, p.TargetDimension)
	for i, coefficients := range p.Coefficients {
		value := source[i]
		if p.Dual {
			value = dot(p.Basis[i], source)
		}
		for j, coefficient := range coefficients {
			result[j] += value * coefficient
		}
	}
	for _, value := range result {
		if !finite(value) {
			return nil, fmt.Errorf("%w: nonfinite projection output", ErrContract)
		}
	}
	return result, nil
}
