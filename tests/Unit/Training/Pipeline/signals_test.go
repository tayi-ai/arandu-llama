package pipeline_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/fusioncache"
	"github.com/tayi-ai/arandu-llama/training/ornith"
	"github.com/tayi-ai/arandu-llama/training/pipeline"
)

func signalDigest(t *testing.T, value any) string {
	t.Helper()
	d, err := fusioncache.Digest(value)
	if err != nil {
		t.Fatal(err)
	}
	return d
}

func signalModel(name string) fusioncache.ModelIdentity {
	return fusioncache.ModelIdentity{Name: name, Revision: strings.Repeat("a", 40), WeightsSHA256: strings.Repeat("b", 64), TokenizerSHA256: strings.Repeat("c", 64), TemplateSHA256: strings.Repeat("d", 64), RuntimeSHA256: strings.Repeat("e", 64), Vocabulary: 8}
}

func writeSignalBytes(t *testing.T, source *pipeline.CacheInput, value any) {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(source.Path, data, 0600); err != nil {
		t.Fatal(err)
	}
	d := sha256.Sum256(data)
	source.SHA256 = hex.EncodeToString(d[:])
	source.Bytes = int64(len(data))
}

func signalFixture(t *testing.T) (pipeline.SignalAdmission, []pipeline.CacheInput, pipeline.SignalLimits) {
	t.Helper()
	limits := pipeline.SignalLimits{MaxBytes: 1 << 20, MaxFeatures: 16, MaxFeatureValues: 128,
		Cache:      fusioncache.Limits{MaxBytes: 1 << 18, MaxExamples: 8, MaxTokens: 32, MaxPositions: 16, MaxTopK: 8, MaxFeatureValues: 128, MaxMappingPairs: 8},
		Projection: fusioncache.ProjectionLimits{MaxSamples: 8, MaxDimension: 16, MaxElements: 256}}
	e := fusioncache.Expectation{Teacher: signalModel("teacher"), Student: signalModel("student"), Features: []fusioncache.FeatureSpec{{Layer: 1, Tensor: "source_output", DType: "float32", Dimension: 2}}}
	e.Mapping = fusioncache.TokenMapping{TeacherTokenizerSHA256: e.Teacher.TokenizerSHA256, StudentTokenizerSHA256: e.Student.TokenizerSHA256, Identity: true, EvidenceSHA256: strings.Repeat("f", 64)}
	e.MappingSHA256 = signalDigest(t, e.Mapping)
	x := fusioncache.Example{DatasetID: "training", DatasetSHA256: strings.Repeat("1", 64), ID: "row", Role: "train", TeacherTokens: []int64{1, 2, 3, 4}, StudentTokens: []int64{1, 2, 3, 4}, PromptTokens: 2}
	e.Examples = []fusioncache.Example{x}
	record := fusioncache.Record{Example: x}
	for pos := 2; pos < 4; pos++ {
		record.Positions = append(record.Positions, fusioncache.Position{TargetIndex: pos, TeacherPrefixSHA256: fusioncache.PrefixDigest(x.TeacherTokens[:pos]), StudentPrefixSHA256: fusioncache.PrefixDigest(x.StudentTokens[:pos]), RetainedMass: 0.75, Probabilities: []fusioncache.Probability{{TeacherTokenID: 2, StudentTokenID: 2, Probability: 0.25}, {TeacherTokenID: 4, StudentTokenID: 4, Probability: 0.5}}, Features: []fusioncache.Feature{{Spec: e.Features[0], Values: []float64{2, 4}}}})
	}
	cache := fusioncache.Cache{Schema: 1, Mode: "teacher_forced", Teacher: e.Teacher, Student: e.Student, Mapping: e.Mapping, MappingSHA256: e.MappingSHA256, Features: e.Features, Records: []fusioncache.Record{record}}
	sample := func(id string) fusioncache.SampleIdentity {
		return fusioncache.SampleIdentity{DatasetSHA256: x.DatasetSHA256, ExampleID: id, Role: "calibration", TargetIndex: 2, TeacherPrefixSHA256: record.Positions[0].TeacherPrefixSHA256, StudentPrefixSHA256: record.Positions[0].StudentPrefixSHA256}
	}
	plan := fusioncache.MatchingPlan{ID: "fixed-match", SourceModel: e.Teacher, TargetModel: e.Student, TokenMappingSHA256: e.MappingSHA256, Source: e.Features[0], Target: fusioncache.FeatureSpec{Layer: 2, Tensor: "decoder_output", DType: "float32", Dimension: 2}, Ridge: 1, Fit: []fusioncache.SampleIdentity{sample("fit1"), sample("fit2")}, Heldout: []fusioncache.SampleIdentity{sample("heldout")}}
	planSHA := signalDigest(t, plan)
	projection, _, err := fusioncache.FitProjection(plan, planSHA, []fusioncache.FeatureSample{{Identity: plan.Fit[0], Source: []float64{1, 0}, Target: []float64{2, 0}}, {Identity: plan.Fit[1], Source: []float64{0, 1}, Target: []float64{0, 3}}}, []fusioncache.FeatureSample{{Identity: plan.Heldout[0], Source: []float64{1, 1}, Target: []float64{2, 3}}}, limits.Projection)
	if err != nil {
		t.Fatal(err)
	}
	source := pipeline.CacheInput{Path: filepath.Join(t.TempDir(), "cache.json"), Expectation: e, Weight: 0.5, Projections: []pipeline.ProjectionInput{{Plan: plan, PlanSHA256: planSHA, Projection: projection, SHA256: projection.SHA256, Weight: 0.25}}}
	writeSignalBytes(t, &source, cache)
	admission := pipeline.SignalAdmission{Student: e.Student, DatasetID: x.DatasetID, Role: x.Role, Training: pipeline.Example{ID: x.ID, DatasetDigest: x.DatasetSHA256, Tokens: append([]int64(nil), x.StudentTokens...), PromptTokens: x.PromptTokens}}
	return admission, []pipeline.CacheInput{source}, limits
}

func TestSignalsUseExactPrefixPositionAndDetachedCopies(t *testing.T) {
	admission, sources, limits := signalFixture(t)
	signals, err := pipeline.NewSignals(admission, sources, limits)
	if err != nil {
		t.Fatal(err)
	}
	if !signals.Admitted() || signals.Student() != admission.Student {
		t.Fatal("admission identity lost")
	}
	features := signals.Features()
	for i, f := range features {
		if f.Position != int64(i+1) || f.Target.Layer != 2 || f.PlanSHA256 != sources[0].Projections[0].PlanSHA256 || math.Abs(float64(f.Values[0])-2) > 1e-6 || math.Abs(float64(f.Values[1])-6) > 1e-6 {
			t.Fatalf("wrong mapped feature: %+v", f)
		}
	}
	if len(features) != 2 || signals.Teachers()[0].Positions[1].TargetIndex != 3 {
		t.Fatal("missing supervised positions")
	}
	admission.Training.Tokens[0] = 7
	sources[0].Expectation.Examples[0].StudentTokens[0] = 7
	sources[0].Projections[0].Projection.Coefficients[0][0] = 99
	features[0].Values[0] = 99
	teachers := signals.Teachers()
	teachers[0].Positions[0].Probabilities[0].Probability = 99
	row := signals.Training()
	row.Tokens[0] = 99
	if signals.Training().Tokens[0] != 1 || signals.Features()[0].Values[0] != 2 || signals.Teachers()[0].Positions[0].Probabilities[0].Probability != 0.25 {
		t.Fatal("caller mutated immutable admitted signals")
	}
}

func repinSignalProjection(t *testing.T, p *pipeline.ProjectionInput) {
	t.Helper()
	p.PlanSHA256 = signalDigest(t, p.Plan)
	p.Projection.PlanSHA256 = p.PlanSHA256
	p.Projection.SHA256 = ""
	p.Projection.SHA256 = signalDigest(t, p.Projection)
	p.SHA256 = p.Projection.SHA256
}

func TestSignalsRejectUnqualifiedAdmissions(t *testing.T) {
	tests := map[string]func(*pipeline.SignalAdmission, []pipeline.CacheInput, *pipeline.SignalLimits){
		"wrong_student": func(a *pipeline.SignalAdmission, _ []pipeline.CacheInput, _ *pipeline.SignalLimits) {
			a.Student.Revision = strings.Repeat("0", 40)
		},
		"wrong_gold": func(a *pipeline.SignalAdmission, _ []pipeline.CacheInput, _ *pipeline.SignalLimits) {
			a.Training.Tokens[3] = 7
		},
		"wrong_prompt": func(a *pipeline.SignalAdmission, _ []pipeline.CacheInput, _ *pipeline.SignalLimits) {
			a.Training.PromptTokens = 1
		},
		"wrong_example": func(a *pipeline.SignalAdmission, _ []pipeline.CacheInput, _ *pipeline.SignalLimits) {
			a.Training.ID = "absent"
		},
		"wrong_dataset": func(a *pipeline.SignalAdmission, _ []pipeline.CacheInput, _ *pipeline.SignalLimits) {
			a.DatasetID = "absent"
		},
		"protection": func(a *pipeline.SignalAdmission, _ []pipeline.CacheInput, _ *pipeline.SignalLimits) {
			a.Role = "protection"
		},
		"calibration_target": func(a *pipeline.SignalAdmission, _ []pipeline.CacheInput, _ *pipeline.SignalLimits) {
			a.Role = "calibration"
		},
		"zero_teacher": func(_ *pipeline.SignalAdmission, s []pipeline.CacheInput, _ *pipeline.SignalLimits) { s[0].Weight = 0 },
		"nonfinite_teacher": func(_ *pipeline.SignalAdmission, s []pipeline.CacheInput, _ *pipeline.SignalLimits) {
			s[0].Weight = math.NaN()
		},
		"excess_teacher_mass": func(_ *pipeline.SignalAdmission, s []pipeline.CacheInput, _ *pipeline.SignalLimits) {
			s[0].Weight = 1.01
		},
		"zero_feature": func(_ *pipeline.SignalAdmission, s []pipeline.CacheInput, _ *pipeline.SignalLimits) {
			s[0].Projections[0].Weight = 0
		},
		"no_feature": func(_ *pipeline.SignalAdmission, s []pipeline.CacheInput, _ *pipeline.SignalLimits) {
			s[0].Projections = nil
		},
		"wrong_cache_sha": func(_ *pipeline.SignalAdmission, s []pipeline.CacheInput, _ *pipeline.SignalLimits) {
			s[0].SHA256 = strings.Repeat("0", 64)
		},
		"wrong_cache_length": func(_ *pipeline.SignalAdmission, s []pipeline.CacheInput, _ *pipeline.SignalLimits) { s[0].Bytes-- },
		"external_teacher": func(_ *pipeline.SignalAdmission, s []pipeline.CacheInput, _ *pipeline.SignalLimits) {
			s[0].Expectation.Teacher.Revision = strings.Repeat("0", 40)
		},
		"wrong_plan_digest": func(_ *pipeline.SignalAdmission, s []pipeline.CacheInput, _ *pipeline.SignalLimits) {
			s[0].Projections[0].PlanSHA256 = strings.Repeat("0", 64)
		},
		"changed_coefficients": func(_ *pipeline.SignalAdmission, s []pipeline.CacheInput, _ *pipeline.SignalLimits) {
			s[0].Projections[0].Projection.Coefficients[0][0] = 99
		},
		"wrong_projection_sha": func(_ *pipeline.SignalAdmission, s []pipeline.CacheInput, _ *pipeline.SignalLimits) {
			s[0].Projections[0].SHA256 = strings.Repeat("0", 64)
		},
		"wrong_target_tensor": func(_ *pipeline.SignalAdmission, s []pipeline.CacheInput, _ *pipeline.SignalLimits) {
			p := &s[0].Projections[0]
			p.Plan.Target.Tensor = "post_norm"
			repinSignalProjection(t, p)
		},
		"wrong_target_model": func(_ *pipeline.SignalAdmission, s []pipeline.CacheInput, _ *pipeline.SignalLimits) {
			p := &s[0].Projections[0]
			p.Plan.TargetModel.Revision = strings.Repeat("0", 40)
			repinSignalProjection(t, p)
		},
		"wrong_source_tensor": func(_ *pipeline.SignalAdmission, s []pipeline.CacheInput, _ *pipeline.SignalLimits) {
			p := &s[0].Projections[0]
			p.Plan.Source.Layer++
			repinSignalProjection(t, p)
		},
		"protection_fit": func(_ *pipeline.SignalAdmission, s []pipeline.CacheInput, _ *pipeline.SignalLimits) {
			p := &s[0].Projections[0]
			p.Plan.Fit[0].Role = "protection"
			repinSignalProjection(t, p)
		},
		"heldout_target": func(a *pipeline.SignalAdmission, s []pipeline.CacheInput, _ *pipeline.SignalLimits) {
			p := &s[0].Projections[0]
			p.Plan.Heldout[0].ExampleID = a.Training.ID
			repinSignalProjection(t, p)
		},
		"overlapping_splits": func(_ *pipeline.SignalAdmission, s []pipeline.CacheInput, _ *pipeline.SignalLimits) {
			p := &s[0].Projections[0]
			p.Plan.Heldout[0] = p.Plan.Fit[0]
			repinSignalProjection(t, p)
		},
		"duplicate_target": func(_ *pipeline.SignalAdmission, s []pipeline.CacheInput, _ *pipeline.SignalLimits) {
			s[0].Projections = append(s[0].Projections, s[0].Projections[0])
		},
		"aggregate_features": func(_ *pipeline.SignalAdmission, _ []pipeline.CacheInput, l *pipeline.SignalLimits) {
			l.MaxFeatures = 1
		},
		"aggregate_values": func(_ *pipeline.SignalAdmission, _ []pipeline.CacheInput, l *pipeline.SignalLimits) {
			l.MaxFeatureValues = 3
		},
		"aggregate_bytes": func(_ *pipeline.SignalAdmission, s []pipeline.CacheInput, l *pipeline.SignalLimits) {
			l.MaxBytes = s[0].Bytes
		},
	}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			a, s, l := signalFixture(t)
			mutate(&a, s, &l)
			if got, err := pipeline.NewSignals(a, s, l); err == nil || got != nil {
				t.Fatal("accepted unqualified signals")
			}
		})
	}
}

func TestSignalsRefuseRehashedForeignCacheContents(t *testing.T) {
	tests := map[string]func(*fusioncache.Cache){
		"free_generation": func(c *fusioncache.Cache) { c.Mode = "free_generation" },
		"wrong_prefix": func(c *fusioncache.Cache) {
			c.Records[0].Positions[1].TeacherPrefixSHA256 = fusioncache.PrefixDigest([]int64{1, 2, 7})
		},
		"wrong_student_prefix": func(c *fusioncache.Cache) { c.Records[0].Positions[0].StudentPrefixSHA256 = strings.Repeat("0", 64) },
		"wrong_position":       func(c *fusioncache.Cache) { c.Records[0].Positions[0].TargetIndex-- },
		"wrong_order":          func(c *fusioncache.Cache) { p := c.Records[0].Positions; p[0], p[1] = p[1], p[0] },
		"wrong_teacher":        func(c *fusioncache.Cache) { c.Teacher.Revision = strings.Repeat("0", 40) },
		"wrong_token_mapping":  func(c *fusioncache.Cache) { c.Records[0].Positions[0].Probabilities[0].StudentTokenID = 7 },
		"zero_mass": func(c *fusioncache.Cache) {
			for i := range c.Records[0].Positions {
				p := &c.Records[0].Positions[i]
				p.RetainedMass = 0
				for j := range p.Probabilities {
					p.Probabilities[j].Probability = 0
				}
			}
		},
	}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			a, s, l := signalFixture(t)
			c, err := fusioncache.Read(s[0].Path, s[0].SHA256, s[0].Expectation, l.Cache)
			if err != nil {
				t.Fatal(err)
			}
			mutate(&c)
			writeSignalBytes(t, &s[0], c)
			if _, err := pipeline.NewSignals(a, s, l); err == nil {
				t.Fatal("accepted inconsistent rehashed cache")
			}
		})
	}
}

func TestSignalsRejectDuplicateAndExcessSources(t *testing.T) {
	a, s, l := signalFixture(t)
	for _, sources := range [][]pipeline.CacheInput{nil, {s[0], s[0]}, make([]pipeline.CacheInput, 9)} {
		if _, err := pipeline.NewSignals(a, sources, l); err == nil {
			t.Fatal("accepted invalid teacher count or duplicate")
		}
	}
}

func TestNativeBackendCannotBypassAdmission(t *testing.T) {
	a, s, l := signalFixture(t)
	signals, err := pipeline.NewSignals(a, s, l)
	if err != nil {
		t.Fatal(err)
	}
	for _, typ := range []reflect.Type{reflect.TypeFor[pipeline.Signals](), reflect.TypeFor[ornith.TrainingBackend]()} {
		for i := 0; i < typ.NumField(); i++ {
			if typ.Field(i).PkgPath == "" {
				t.Fatalf("mutable exported bypass: %s.%s", typ.Name(), typ.Field(i).Name)
			}
		}
	}
	var zero ornith.TrainingBackend
	if _, err := zero.Objective(context.Background()); err == nil {
		t.Fatal("zero backend admitted")
	}
	if _, err := zero.Parameters(context.Background()); err == nil {
		t.Fatal("zero backend parameters admitted")
	}
	if err := zero.Install(context.Background(), []float32{1}); err == nil {
		t.Fatal("zero backend install admitted")
	}
	identity := ornith.AssemblyIdentity{IndexSHA256: strings.Repeat("1", 64), ConfigSHA256: strings.Repeat("2", 64), ReferenceSHA256: strings.Repeat("3", 64), InitialAdapterSHA256: strings.Repeat("4", 64)}
	loaded := &ornith.LoadedTextModel{Model: &ornith.TextModel{}, Summary: ornith.AssemblySummary{Identity: identity}}
	external := ornith.TrainingAdmission{Assembly: identity, Student: a.Student}
	external.Assembly.ConfigSHA256 = strings.Repeat("5", 64)
	_, err = ornith.NewTrainingBackend(loaded, signals, ornith.Limits{MaxTokens: 8, MaxCheckpointBytes: 1024}, nil, external)
	if err == nil || !strings.Contains(err.Error(), "assembly or student") {
		t.Fatalf("identity mismatch reached native geometry: %v", err)
	}
	external.Assembly = identity
	external.Student.Revision = strings.Repeat("0", 40)
	if _, err := ornith.NewTrainingBackend(loaded, signals, ornith.Limits{}, nil, external); err == nil || !strings.Contains(err.Error(), "assembly or student") {
		t.Fatalf("student mismatch reached native geometry: %v", err)
	}
	if _, err := ornith.NewTrainingBackend(loaded, &pipeline.Signals{}, ornith.Limits{}, nil, ornith.TrainingAdmission{Assembly: identity, Student: a.Student}); err == nil {
		t.Fatal("zero signals admitted")
	}
}
