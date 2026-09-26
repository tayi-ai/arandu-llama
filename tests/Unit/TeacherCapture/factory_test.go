package teachercapture_test

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	llama "github.com/tayi-ai/arandu-llama"
	"github.com/tayi-ai/arandu-llama/training/fusioncache"
	"github.com/tayi-ai/arandu-llama/training/teachers"
)

func cacheDigest(t *testing.T, value any) string {
	t.Helper()
	digest, err := fusioncache.Digest(value)
	if err != nil {
		t.Fatal(err)
	}
	return digest
}

func cacheArtifact(t *testing.T, value any) (string, string) {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "artifact.json")
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	return path, fmt.Sprintf("%x", sha256.Sum256(data))
}

func nativeJobs(t *testing.T) ([]fusioncache.Expectation, []teachers.NativeConfig, teachers.NativeLimits) {
	t.Helper()
	modelPath, modelSHA, _ := tinyTeacher(t)
	tokenizerPath, tokenizerSHA := cacheArtifact(t, map[string]any{"fixture_token_ids": true, "vocabulary": fixtureVocabulary})
	templatePath, templateSHA := cacheArtifact(t, map[string]string{"template": "exact supplied IDs"})
	runtimePath, runtimeSHA := cacheArtifact(t, map[string]string{"backend": "llama.cpp", "capture_version": llama.TeacherCaptureVersion})
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	file, err := os.Open(executable)
	if err != nil {
		t.Fatal(err)
	}
	hash := sha256.New()
	_, copyErr := io.Copy(hash, file)
	if err := errors.Join(copyErr, file.Close()); err != nil {
		t.Fatal(err)
	}
	binding := teachers.NativeBindingEvidence{Backend: "llama.cpp", CaptureVersion: llama.TeacherCaptureVersion,
		RuntimeSHA256: runtimeSHA, ExecutableSHA256: fmt.Sprintf("%x", hash.Sum(nil))}
	bindingPath, bindingSHA := cacheArtifact(t, binding)
	limits := teachers.NativeLimits{Cache: fusioncache.Limits{MaxBytes: 1 << 20, MaxExamples: 8, MaxTokens: 32,
		MaxPositions: 32, MaxTopK: fixtureVocabulary, MaxFeatureValues: 2048, MaxMappingPairs: 32},
		MaxTeachers: 2, MaxTotalTokens: 64, MaxResidentBytes: 1 << 20}
	var jobs []fusioncache.Expectation
	var configs []teachers.NativeConfig
	for _, name := range []string{"arbitrary-source-a", "other-source-b"} {
		teacher := fusioncache.ModelIdentity{Name: name, Revision: strings.Repeat("a", 40), WeightsSHA256: modelSHA,
			TokenizerSHA256: tokenizerSHA, TemplateSHA256: templateSHA, RuntimeSHA256: runtimeSHA, Vocabulary: fixtureVocabulary}
		student := teacher
		student.Name = "independently-admitted-student"
		mapping := fusioncache.TokenMapping{TeacherTokenizerSHA256: tokenizerSHA, StudentTokenizerSHA256: tokenizerSHA,
			Identity: true, EvidenceSHA256: strings.Repeat("f", 64)}
		jobs = append(jobs, fusioncache.Expectation{Teacher: teacher, Student: student, Mapping: mapping, MappingSHA256: cacheDigest(t, mapping),
			Examples: []fusioncache.Example{{DatasetID: "fixture-gold", DatasetSHA256: strings.Repeat("d", 64), ID: "one", Role: "calibration",
				TeacherTokens: []int64{1, 2, 3, 4, 5, 6}, StudentTokens: []int64{1, 2, 3, 4, 5, 6}, PromptTokens: 2}},
			Features: []fusioncache.FeatureSpec{{Layer: 0, Tensor: "l_out-0", DType: "float32", Dimension: fixtureWidth},
				{Layer: 1, Tensor: "l_out-1", DType: "float32", Dimension: fixtureWidth}}})
		options := captureOptions()
		options.TopK = 3
		configs = append(configs, teachers.NativeConfig{Identity: teacher, ModelPath: modelPath, TokenizerPath: tokenizerPath,
			TemplatePath: templatePath, RuntimePath: runtimePath, BindingEvidencePath: bindingPath, BindingEvidenceSHA256: bindingSHA,
			CaptureVersion: llama.TeacherCaptureVersion, GPULayers: 0, Capture: options})
	}
	return jobs, configs, limits
}

func nativeFactory(t *testing.T, jobs []fusioncache.Expectation, configs []teachers.NativeConfig, limits teachers.NativeLimits) *teachers.NativeFactory {
	t.Helper()
	factory, err := teachers.NewNativeFactory(jobs, configs, limits)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := factory.Close(); err != nil {
			t.Error(err)
		}
	})
	return factory
}

func prefixRequest(job fusioncache.Expectation, target int) fusioncache.PrefixRequest {
	tokens := append([]int64(nil), job.Examples[0].TeacherTokens[:target]...)
	return fusioncache.PrefixRequest{Tokens: tokens, PrefixSHA256: fusioncache.PrefixDigest(tokens), TargetIndex: target,
		Features: append([]fusioncache.FeatureSpec(nil), job.Features...)}
}

func TestNativeFactorySequentialCacheRoundTrip(t *testing.T) {
	jobs, configs, limits := nativeJobs(t)
	factory := nativeFactory(t, jobs, configs, limits)
	receipts, err := fusioncache.ProduceSequential(context.Background(), jobs, factory, fusioncache.DirectorySink{Directory: t.TempDir()}, limits.Cache)
	if err != nil {
		t.Fatal(err)
	}
	if len(receipts) != 2 {
		t.Fatalf("expected two receipts: %+v", receipts)
	}
	// Independently request the native causal rows. This catches a factory
	// that labels the right prefix while copying a shifted tensor/logit row.
	reference, err := llama.OpenTeacherModel(configs[0].ModelPath, jobs[0].Teacher.WeightsSHA256, llama.WithGPULayers(0))
	if err != nil {
		t.Fatal(err)
	}
	defer reference.Close()
	options := configs[0].Capture
	options.Tensor = "l_out-0"
	firstLayer := capture(t, reference, []int32{1, 2, 3, 4, 5, 6}, []int32{1, 2, 3, 4}, options)
	options.Tensor = "l_out-1"
	secondLayer := capture(t, reference, []int32{1, 2, 3, 4, 5, 6}, []int32{1, 2, 3, 4}, options)
	for i, receipt := range receipts {
		cache, err := fusioncache.Read(receipt.Location, receipt.SHA256, jobs[i], limits.Cache)
		if err != nil {
			t.Fatal(err)
		}
		if cache.Mode != "teacher_forced" || len(cache.Records[0].Positions) != 4 || cache.Teacher != jobs[i].Teacher {
			t.Fatalf("wrong cache: %+v", cache)
		}
		for rowIndex, row := range cache.Records[0].Positions {
			if row.RetainedMass <= 0 || row.RetainedMass >= 1 || len(row.Features) != 2 || len(row.Probabilities) != 3 {
				t.Fatalf("wrong native signals: %+v", row)
			}
			for _, probability := range row.Probabilities {
				found := false
				for _, native := range firstLayer.Rows[rowIndex].TopK {
					if int64(native.TokenID) == probability.TeacherTokenID {
						found = native.Probability == probability.Probability
					}
				}
				if !found {
					t.Fatal("persisted probability differs from the exact native causal row")
				}
			}
			for layer, native := range []*llama.TeacherCapture{firstLayer, secondLayer} {
				for column, value := range row.Features[layer].Values {
					if value != float64(native.Rows[rowIndex].Features[column]) {
						t.Fatal("persisted feature differs from the exact native causal row")
					}
				}
			}
		}
	}
	stats := factory.Stats()
	if stats.OpenedTeachers != 2 || stats.ClosedTeachers != 2 || stats.ResidentTeachers != 0 || stats.NativeCaptures != 4 || stats.CapturedExamples != 2 {
		t.Fatalf("did not capture once per example/tensor or release sequentially: %+v", stats)
	}
	t.Logf("native sequential cache: %+v; receipts=%d", stats, len(receipts))
}

func TestNativeFactoryFreezesInputsCachesRowsAndOwnsOneTeacher(t *testing.T) {
	jobs, configs, limits := nativeJobs(t)
	factory := nativeFactory(t, jobs, configs, limits)
	admitted := jobs[0]
	admitted.Examples = append([]fusioncache.Example(nil), admitted.Examples...)
	admitted.Examples[0].TeacherTokens = append([]int64(nil), admitted.Examples[0].TeacherTokens...)
	jobs[0].Examples[0].TeacherTokens[2] = 10
	configs[0].ModelPath = "caller mutation"
	producer, err := factory.Open(context.Background(), admitted.Teacher)
	if err != nil {
		t.Fatal(err)
	}
	if other, err := factory.Open(context.Background(), jobs[1].Teacher); err == nil || other != nil {
		t.Fatal("two teachers became resident")
	}
	request := prefixRequest(admitted, 3)
	first, err := producer.TeacherForce(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	first.Probabilities[0].Probability = 0
	first.Features[0].Values[0] = 999
	request.Tokens[0] = 15
	again, err := producer.TeacherForce(context.Background(), prefixRequest(admitted, 3))
	if err != nil {
		t.Fatal(err)
	}
	if again.Probabilities[0].Probability == 0 || again.Features[0].Values[0] == 999 {
		t.Fatal("caller mutated cached signals")
	}
	if stats := factory.Stats(); stats.NativeCaptures != 2 {
		t.Fatalf("prefix was recomputed: %+v", stats)
	}
	if err := producer.Close(); err != nil {
		t.Fatal(err)
	}
	if err := producer.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := producer.TeacherForce(context.Background(), prefixRequest(admitted, 3)); err == nil {
		t.Fatal("closed producer answered")
	}
	second, err := factory.Open(context.Background(), jobs[1].Teacher)
	if err != nil {
		t.Fatal(err)
	}
	if err := factory.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := second.TeacherForce(context.Background(), prefixRequest(jobs[1], 3)); err == nil {
		t.Fatal("factory failed to close active producer")
	}
	if _, err := factory.Open(context.Background(), jobs[1].Teacher); err == nil {
		t.Fatal("closed factory reopened")
	}
}

func TestNativeFactoryRejectsUnadmittedPrefixesWithoutNativeCalls(t *testing.T) {
	jobs, configs, limits := nativeJobs(t)
	factory := nativeFactory(t, jobs, configs, limits)
	unknown := jobs[0].Teacher
	unknown.Revision = strings.Repeat("c", 40)
	if _, err := factory.Open(context.Background(), unknown); !errors.Is(err, fusioncache.ErrContract) {
		t.Fatalf("unknown identity admitted: %v", err)
	}
	producer, err := factory.Open(context.Background(), jobs[0].Teacher)
	if err != nil {
		t.Fatal(err)
	}
	for name, mutate := range map[string]func(*fusioncache.PrefixRequest){
		"digest": func(r *fusioncache.PrefixRequest) { r.PrefixSHA256 = strings.Repeat("0", 64) },
		"target": func(r *fusioncache.PrefixRequest) { r.TargetIndex++ },
		"token": func(r *fusioncache.PrefixRequest) {
			r.Tokens[0] = 8
			r.PrefixSHA256 = fusioncache.PrefixDigest(r.Tokens)
		},
		"feature":         func(r *fusioncache.PrefixRequest) { r.Features[0].Layer++ },
		"missing-feature": func(r *fusioncache.PrefixRequest) { r.Features = nil },
	} {
		t.Run(name, func(t *testing.T) {
			request := prefixRequest(jobs[0], 3)
			mutate(&request)
			if _, err := producer.TeacherForce(context.Background(), request); !errors.Is(err, fusioncache.ErrContract) {
				t.Fatalf("accepted invalid request: %v", err)
			}
		})
	}
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := producer.TeacherForce(cancelled, prefixRequest(jobs[0], 3)); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if _, err := producer.TeacherForce(nil, prefixRequest(jobs[0], 3)); err == nil {
		t.Fatal("nil context admitted")
	}
	if stats := factory.Stats(); stats.NativeCaptures != 0 {
		t.Fatalf("invalid request executed native model: %+v", stats)
	}
}

func TestNativeFactorySharedPrefixesCaptureEachExampleOnce(t *testing.T) {
	jobs, configs, limits := nativeJobs(t)
	jobs, configs = jobs[:1], configs[:1]
	second := jobs[0].Examples[0]
	second.ID = "two"
	second.TeacherTokens = []int64{1, 2, 3, 9, 5, 6}
	second.StudentTokens = append([]int64(nil), second.TeacherTokens...)
	jobs[0].Examples = append(jobs[0].Examples, second)
	factory := nativeFactory(t, jobs, configs, limits)
	receipts, err := fusioncache.ProduceSequential(context.Background(), jobs, factory, fusioncache.DirectorySink{Directory: t.TempDir()}, limits.Cache)
	if err != nil {
		t.Fatal(err)
	}
	cache, err := fusioncache.Read(receipts[0].Location, receipts[0].SHA256, jobs[0], limits.Cache)
	if err != nil {
		t.Fatal(err)
	}
	if stats := factory.Stats(); stats.NativeCaptures != 4 || stats.CapturedExamples != 2 {
		t.Fatalf("native prefixes were recomputed: %+v", stats)
	}
	if !reflect.DeepEqual(cache.Records[0].Positions[0], cache.Records[1].Positions[0]) ||
		reflect.DeepEqual(cache.Records[0].Positions[2].Probabilities, cache.Records[1].Positions[2].Probabilities) {
		t.Fatal("shared or diverged gold prefixes were mishandled")
	}
}

func TestNativeFactoryRejectsBindingsArtifactsAndBounds(t *testing.T) {
	jobs, configs, limits := nativeJobs(t)
	for name, mutate := range map[string]func([]teachers.NativeConfig, *teachers.NativeLimits){
		"version":             func(c []teachers.NativeConfig, _ *teachers.NativeLimits) { c[0].CaptureVersion = "unqualified" },
		"identity":            func(c []teachers.NativeConfig, _ *teachers.NativeLimits) { c[0].Identity.Name = "different" },
		"binding-missing":     func(c []teachers.NativeConfig, _ *teachers.NativeLimits) { c[0].BindingEvidenceSHA256 = "" },
		"aggregate-teachers":  func(_ []teachers.NativeConfig, l *teachers.NativeLimits) { l.MaxTeachers = 1 },
		"aggregate-tokens":    func(_ []teachers.NativeConfig, l *teachers.NativeLimits) { l.MaxTotalTokens = 11 },
		"aggregate-examples":  func(_ []teachers.NativeConfig, l *teachers.NativeLimits) { l.Cache.MaxExamples = 1 },
		"aggregate-positions": func(_ []teachers.NativeConfig, l *teachers.NativeLimits) { l.Cache.MaxPositions = 7 },
		"aggregate-features":  func(_ []teachers.NativeConfig, l *teachers.NativeLimits) { l.Cache.MaxFeatureValues = 127 },
		"aggregate-bytes":     func(_ []teachers.NativeConfig, l *teachers.NativeLimits) { l.Cache.MaxBytes = 100 },
		"resident-payload":    func(_ []teachers.NativeConfig, l *teachers.NativeLimits) { l.MaxResidentBytes = 64 },
	} {
		t.Run(name, func(t *testing.T) {
			changedConfigs, changedLimits := append([]teachers.NativeConfig(nil), configs...), limits
			mutate(changedConfigs, &changedLimits)
			if factory, err := teachers.NewNativeFactory(jobs, changedConfigs, changedLimits); err == nil || factory != nil {
				t.Fatal("invalid constructor contract admitted")
			}
		})
	}
	for name, mutate := range map[string]func(*teachers.NativeConfig){
		"tokenizer":     func(c *teachers.NativeConfig) { c.TokenizerPath = c.TemplatePath },
		"template":      func(c *teachers.NativeConfig) { c.TemplatePath = c.RuntimePath },
		"runtime":       func(c *teachers.NativeConfig) { c.RuntimePath = c.TemplatePath },
		"weights":       func(c *teachers.NativeConfig) { c.ModelPath = c.TemplatePath },
		"evidence-hash": func(c *teachers.NativeConfig) { c.BindingEvidenceSHA256 = strings.Repeat("0", 64) },
		"other-backend": func(c *teachers.NativeConfig) {
			c.BindingEvidencePath, c.BindingEvidenceSHA256 = cacheArtifact(t, teachers.NativeBindingEvidence{Backend: "other-runtime",
				CaptureVersion: llama.TeacherCaptureVersion, RuntimeSHA256: c.Identity.RuntimeSHA256, ExecutableSHA256: strings.Repeat("a", 64)})
		},
		"other-executable": func(c *teachers.NativeConfig) {
			c.BindingEvidencePath, c.BindingEvidenceSHA256 = cacheArtifact(t, teachers.NativeBindingEvidence{Backend: "llama.cpp",
				CaptureVersion: llama.TeacherCaptureVersion, RuntimeSHA256: c.Identity.RuntimeSHA256, ExecutableSHA256: strings.Repeat("a", 64)})
		},
	} {
		t.Run(name, func(t *testing.T) {
			changed := append([]teachers.NativeConfig(nil), configs...)
			mutate(&changed[0])
			factory := nativeFactory(t, jobs, changed, limits)
			if producer, err := factory.Open(context.Background(), jobs[0].Teacher); err == nil || producer != nil {
				t.Fatal("unverified artifacts admitted")
			}
			if stats := factory.Stats(); stats.ResidentTeachers != 0 || stats.OpenedTeachers != 0 {
				t.Fatalf("failed open retained teacher: %+v", stats)
			}
		})
	}
}

func TestNativeFactoryProbabilityOnlyAndFailureRelease(t *testing.T) {
	jobs, configs, limits := nativeJobs(t)
	jobs = jobs[:1]
	configs = configs[:1]
	jobs[0].Features = nil
	configs[0].Capture.TopK = fixtureVocabulary
	factory := nativeFactory(t, jobs, configs, limits)
	receipts, err := fusioncache.ProduceSequential(context.Background(), jobs, factory, fusioncache.DirectorySink{Directory: t.TempDir()}, limits.Cache)
	if err != nil {
		t.Fatal(err)
	}
	cache, err := fusioncache.Read(receipts[0].Location, receipts[0].SHA256, jobs[0], limits.Cache)
	if err != nil || len(cache.Records[0].Positions[0].Features) != 0 || factory.Stats().NativeCaptures != 1 {
		t.Fatalf("probability-only capture: %v %+v", err, factory.Stats())
	}
	jobs[0].Features = []fusioncache.FeatureSpec{{Layer: 0, Tensor: "l_out-0", DType: "float32", Dimension: fixtureWidth + 1}}
	badFactory := nativeFactory(t, jobs, configs, limits)
	failed, err := fusioncache.ProduceSequential(context.Background(), jobs, badFactory, fusioncache.DirectorySink{Directory: t.TempDir()}, limits.Cache)
	if err == nil || len(failed) != 0 {
		t.Fatal("wrong feature shape persisted")
	}
	if stats := badFactory.Stats(); stats.OpenedTeachers != 1 || stats.ClosedTeachers != 1 || stats.ResidentTeachers != 0 {
		t.Fatalf("failure leaked teacher: %+v", stats)
	}
	if reflect.DeepEqual(factory.Stats(), badFactory.Stats()) {
		t.Fatal("failed capture was reported as completed example")
	}
}
