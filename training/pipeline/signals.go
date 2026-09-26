package pipeline

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"slices"

	"github.com/tayi-ai/arandu-llama/training/fusioncache"
)

// SignalAdmission independently pins the student and the exact training row.
// Only the train role is admitted; evaluation and protection are separate inputs.
type SignalAdmission struct {
	Student         fusioncache.ModelIdentity
	DatasetID, Role string
	Training        Example
}

// ProjectionInput pins an explicit correspondence and fitted coefficients.
// Weight multiplies each position's mean squared error; it is not normalized.
type ProjectionInput struct {
	Plan       fusioncache.MatchingPlan
	PlanSHA256 string
	Projection fusioncache.Projection
	SHA256     string
	Weight     float64
}

// CacheInput identifies persisted teacher-forced bytes and an external expectation.
// Bytes is the exact artifact length. Expectation must come from the caller's
// admitted dataset/model manifest, never be reconstructed from the cache itself.
type CacheInput struct {
	Path, SHA256 string
	Bytes        int64
	Expectation  fusioncache.Expectation
	Weight       float64
	Projections  []ProjectionInput
}

// SignalLimits bounds all sources together as well as individual cache/projection
// validation. MaxBytes covers cache files and encoded expectations/plans/projections.
// MaxFeatures and MaxFeatureValues bound the projected targets retained in memory.
type SignalLimits struct {
	MaxBytes                      int64
	MaxFeatures, MaxFeatureValues int
	Cache                         fusioncache.Limits
	Projection                    fusioncache.ProjectionLimits
}

// SignalProbability is unrenormalized top-k mass in the student's token mapping.
type SignalProbability struct {
	TokenID     int64
	Probability float64
}

// TeacherPosition binds one distribution to the exact next-token target index.
type TeacherPosition struct {
	TargetIndex   int
	RetainedMass  float64
	Probabilities []SignalProbability
}

// TeacherSignal contains the admitted teacher's completion distributions.
type TeacherSignal struct {
	Identity    fusioncache.ModelIdentity
	CacheSHA256 string
	Weight      float64
	Positions   []TeacherPosition
}

// FeatureSignal is an explicitly projected decoder output at an input position.
type FeatureSignal struct {
	Source, PlanSHA256, ProjectionSHA256 string
	Target                               fusioncache.FeatureSpec
	Position                             int64
	Weight                               float64
	Values                               []float32
}

// Signals is an immutable admission result. The zero value is not admitted.
// Hash validation proves identity consistency, not teacher execution or quality.
type Signals struct {
	admission SignalAdmission
	teachers  []TeacherSignal
	features  []FeatureSignal
	admitted  bool
}

// Student returns the externally admitted student identity.
func (s *Signals) Student() fusioncache.ModelIdentity {
	if s == nil {
		return fusioncache.ModelIdentity{}
	}
	return s.admission.Student
}

// Training returns a detached copy of the exact training row.
func (s *Signals) Training() Example {
	if s == nil {
		return Example{}
	}
	row := s.admission.Training
	row.Tokens = slices.Clone(row.Tokens)
	return row
}

// Admitted reports successful construction with positive teacher and feature paths.
func (s *Signals) Admitted() bool { return s != nil && s.admitted }

// Teachers returns detached distributions; modifying them cannot change Signals.
func (s *Signals) Teachers() []TeacherSignal {
	if s == nil {
		return nil
	}
	result := slices.Clone(s.teachers)
	for i := range result {
		result[i].Positions = slices.Clone(result[i].Positions)
		for j := range result[i].Positions {
			result[i].Positions[j].Probabilities = slices.Clone(result[i].Positions[j].Probabilities)
		}
	}
	return result
}

// Features returns detached projected targets.
func (s *Signals) Features() []FeatureSignal {
	if s == nil {
		return nil
	}
	result := slices.Clone(s.features)
	for i := range result {
		result[i].Values = slices.Clone(result[i].Values)
	}
	return result
}

type signalByteBudget struct{ remaining int64 }

func (b *signalByteBudget) Write(p []byte) (int, error) {
	if int64(len(p)) > b.remaining {
		return 0, errors.New("pipeline: aggregate signal byte limit exceeded")
	}
	b.remaining -= int64(len(p))
	return len(p), nil
}

// NewSignals verifies persisted bytes against independent expectations, selects
// exactly the admitted gold row, and applies only explicitly pinned projections.
// At most eight teachers are admitted. It never fits or selects a matching plan.
func NewSignals(admission SignalAdmission, sources []CacheInput, limits SignalLimits) (*Signals, error) {
	row := admission.Training
	if !identifier(row.ID) || !digest(row.DatasetDigest) || !identifier(admission.DatasetID) || admission.Role != "train" || row.PromptTokens < 1 || row.PromptTokens >= len(row.Tokens) ||
		len(sources) == 0 || len(sources) > 8 || limits.MaxBytes <= 0 || limits.MaxFeatures <= 0 || limits.MaxFeatureValues <= 0 ||
		limits.Projection.MaxSamples <= 0 || limits.Projection.MaxDimension <= 0 || limits.Projection.MaxElements <= 0 {
		return nil, errors.New("pipeline: invalid signal admission or bounds")
	}
	admission.Training.Tokens = slices.Clone(row.Tokens)
	result := &Signals{admission: admission}
	budget := &signalByteBudget{remaining: limits.MaxBytes}
	encoder := json.NewEncoder(budget)
	seenTeachers := map[string]bool{}
	totalWeight := 0.0
	remainingValues := limits.MaxFeatureValues
	for _, source := range sources {
		e := source.Expectation
		if e.Student != admission.Student || !finite(source.Weight) || source.Weight <= 0 || seenTeachers[e.Teacher.Name] ||
			source.Bytes <= 0 || source.Bytes > limits.Cache.MaxBytes || source.Bytes > budget.remaining || len(source.Projections) > limits.MaxFeatures {
			return nil, errors.New("pipeline: teacher identity, positive weight or cache budget differs")
		}
		seenTeachers[e.Teacher.Name] = true
		totalWeight += source.Weight
		if !finite(totalWeight) || totalWeight > 1 {
			return nil, errors.New("pipeline: teacher weights exceed one")
		}
		budget.remaining -= source.Bytes
		if err := fusioncache.ValidateExpectation(e, limits.Cache); err != nil {
			return nil, err
		}
		if err := encoder.Encode(e); err != nil {
			return nil, err
		}
		info, err := os.Stat(source.Path)
		if err != nil {
			return nil, err
		}
		if !info.Mode().IsRegular() || info.Size() != source.Bytes {
			return nil, errors.New("pipeline: cache byte length differs")
		}
		cacheLimits := limits.Cache
		cacheLimits.MaxBytes = source.Bytes
		cache, err := fusioncache.Read(source.Path, source.SHA256, e, cacheLimits)
		if err != nil {
			return nil, err
		}
		var record *fusioncache.Record
		for i := range cache.Records {
			x := &cache.Records[i]
			if x.Example.DatasetID == admission.DatasetID && x.Example.DatasetSHA256 == row.DatasetDigest && x.Example.ID == row.ID {
				if record != nil || x.Example.Role != admission.Role || x.Example.PromptTokens != row.PromptTokens || !slices.Equal(x.Example.StudentTokens, admission.Training.Tokens) {
					return nil, errors.New("pipeline: exact training row differs")
				}
				record = x
			}
		}
		if record == nil {
			return nil, errors.New("pipeline: admitted training row absent")
		}
		teacher := TeacherSignal{Identity: e.Teacher, CacheSHA256: source.SHA256, Weight: source.Weight}
		for _, position := range record.Positions {
			mass := 0.0
			for _, probability := range position.Probabilities {
				mass += probability.Probability
			}
			if position.RetainedMass <= 0 || source.Weight*mass <= 0 {
				return nil, errors.New("pipeline: zero teacher mass at training position")
			}
			out := TeacherPosition{TargetIndex: position.TargetIndex, RetainedMass: position.RetainedMass}
			for _, p := range position.Probabilities {
				out.Probabilities = append(out.Probabilities, SignalProbability{TokenID: p.StudentTokenID, Probability: p.Probability})
			}
			teacher.Positions = append(teacher.Positions, out)
		}
		result.teachers = append(result.teachers, teacher)
		seenTargets := map[int]bool{}
		for _, projection := range source.Projections {
			plan := projection.Plan
			if seenTargets[plan.Target.Layer] {
				return nil, errors.New("pipeline: duplicate teacher target layer")
			}
			seenTargets[plan.Target.Layer] = true
			if err := validateSignalProjection(projection, e, admission, limits.Projection); err != nil {
				return nil, err
			}
			if err := encoder.Encode(plan); err != nil {
				return nil, err
			}
			if err := encoder.Encode(projection.Projection); err != nil {
				return nil, err
			}
			for _, position := range record.Positions {
				if len(result.features) >= limits.MaxFeatures || plan.Target.Dimension > remainingValues {
					return nil, errors.New("pipeline: aggregate projected feature bound exceeded")
				}
				var feature *fusioncache.Feature
				for i := range position.Features {
					if position.Features[i].Spec == plan.Source {
						feature = &position.Features[i]
						break
					}
				}
				if feature == nil {
					return nil, errors.New("pipeline: planned source feature absent")
				}
				values, err := projection.Projection.Apply(feature.Values, projection.PlanSHA256, limits.Projection)
				if err != nil {
					return nil, err
				}
				out := FeatureSignal{Source: e.Teacher.Name, PlanSHA256: projection.PlanSHA256, ProjectionSHA256: projection.SHA256, Target: plan.Target, Position: int64(position.TargetIndex - 1), Weight: projection.Weight, Values: make([]float32, len(values))}
				for i, value := range values {
					if math.Abs(value) > math.MaxFloat32 {
						return nil, errors.New("pipeline: projected feature exceeds float32 range")
					}
					out.Values[i] = float32(value)
				}
				remainingValues -= len(values)
				result.features = append(result.features, out)
			}
		}
	}
	if len(result.features) == 0 {
		return nil, errors.New("pipeline: positive feature path required")
	}
	result.admitted = true
	return result, nil
}

func validateSignalProjection(input ProjectionInput, e fusioncache.Expectation, admission SignalAdmission, limits fusioncache.ProjectionLimits) error {
	p := input.Plan
	if !finite(input.Weight) || input.Weight <= 0 || !identifier(p.ID) || p.SourceModel != e.Teacher || p.TargetModel != e.Student || p.TokenMappingSHA256 != e.MappingSHA256 ||
		!finite(p.Ridge) || p.Ridge <= 0 || p.Target.Tensor != "decoder_output" || p.Target.Layer < 0 || p.Target.Dimension <= 0 || p.Target.Dimension > limits.MaxDimension ||
		(p.Target.DType != "float32" && p.Target.DType != "float64") || !slices.Contains(e.Features, p.Source) ||
		len(p.Fit) == 0 || len(p.Heldout) == 0 || len(p.Fit) > limits.MaxSamples || len(p.Heldout) > limits.MaxSamples-len(p.Fit) ||
		input.Projection.SourceDimension != p.Source.Dimension || input.Projection.TargetDimension != p.Target.Dimension || input.Projection.SHA256 != input.SHA256 {
		return errors.New("pipeline: unqualified feature projection")
	}
	fitExamples := map[string]bool{}
	seenRows := map[string]bool{}
	for partition, identities := range [][]fusioncache.SampleIdentity{p.Fit, p.Heldout} {
		for _, id := range identities {
			key := id.DatasetSHA256 + "/" + id.ExampleID
			rowKey := fmt.Sprintf("%s/%d", key, id.TargetIndex)
			if !digest(id.DatasetSHA256) || !identifier(id.ExampleID) || (id.Role != "train" && id.Role != "calibration") || id.TargetIndex < 1 || !digest(id.TeacherPrefixSHA256) || !digest(id.StudentPrefixSHA256) || seenRows[rowKey] ||
				(partition == 1 && (fitExamples[key] || (id.DatasetSHA256 == admission.Training.DatasetDigest && id.ExampleID == admission.Training.ID))) {
				return errors.New("pipeline: projection split or role differs")
			}
			seenRows[rowKey] = true
			if partition == 0 {
				fitExamples[key] = true
			}
		}
	}
	actual, err := fusioncache.Digest(p)
	if err != nil || !digest(input.PlanSHA256) || actual != input.PlanSHA256 {
		return errors.New("pipeline: matching plan digest differs")
	}
	return input.Projection.Validate(input.PlanSHA256, limits)
}
