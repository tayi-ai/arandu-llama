package fusioncache_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/fusioncache"
)

func digest(t *testing.T, value any) string {
	t.Helper()
	digest, err := fusioncache.Digest(value)
	if err != nil {
		t.Fatal(err)
	}
	return digest
}

func model(name string) fusioncache.ModelIdentity {
	return fusioncache.ModelIdentity{Name: name, Revision: strings.Repeat("a", 40), WeightsSHA256: strings.Repeat("b", 64), TokenizerSHA256: strings.Repeat("c", 64), TemplateSHA256: strings.Repeat("d", 64), RuntimeSHA256: strings.Repeat("e", 64), Vocabulary: 8}
}

func limits() fusioncache.Limits {
	return fusioncache.Limits{MaxBytes: 1 << 20, MaxExamples: 8, MaxTokens: 32, MaxPositions: 16, MaxTopK: 8, MaxFeatureValues: 128, MaxMappingPairs: 8}
}

func expectation(t *testing.T) fusioncache.Expectation {
	t.Helper()
	m := model("teacher")
	e := fusioncache.Expectation{Teacher: m, Student: model("student"), Mapping: fusioncache.TokenMapping{TeacherTokenizerSHA256: m.TokenizerSHA256, StudentTokenizerSHA256: m.TokenizerSHA256, Identity: true, EvidenceSHA256: strings.Repeat("f", 64)}, Features: []fusioncache.FeatureSpec{{Layer: 1, Tensor: "post_norm", DType: "float32", Dimension: 2}}}
	e.MappingSHA256 = digest(t, e.Mapping)
	e.Examples = []fusioncache.Example{{DatasetID: "admitted", DatasetSHA256: strings.Repeat("1", 64), ID: "one", Role: "calibration", TeacherTokens: []int64{1, 2, 3, 4}, StudentTokens: []int64{1, 2, 3, 4}, PromptTokens: 2}}
	return e
}

func fixture(t *testing.T) (fusioncache.Cache, fusioncache.Expectation) {
	t.Helper()
	e := expectation(t)
	c := fusioncache.Cache{Schema: 1, Mode: "teacher_forced", Teacher: e.Teacher, Student: e.Student, Mapping: e.Mapping, MappingSHA256: e.MappingSHA256, Features: e.Features}
	for _, x := range e.Examples {
		r := fusioncache.Record{Example: x}
		for position := x.PromptTokens; position < len(x.TeacherTokens); position++ {
			r.Positions = append(r.Positions, fusioncache.Position{TargetIndex: position, TeacherPrefixSHA256: fusioncache.PrefixDigest(x.TeacherTokens[:position]), StudentPrefixSHA256: fusioncache.PrefixDigest(x.StudentTokens[:position]), RetainedMass: 0.75, Probabilities: []fusioncache.Probability{{TeacherTokenID: 2, StudentTokenID: 2, Probability: 0.25}, {TeacherTokenID: 4, StudentTokenID: 4, Probability: 0.5}}, Features: []fusioncache.Feature{{Spec: e.Features[0], Values: []float64{1, 2}}}})
		}
		c.Records = append(c.Records, r)
	}
	if err := fusioncache.ValidateAgainstStudent(c, e, limits()); err != nil {
		t.Fatal(err)
	}
	return c, e
}

func clone[T any](t *testing.T, source T) T {
	t.Helper()
	data, err := json.Marshal(source)
	if err != nil {
		t.Fatal(err)
	}
	var result T
	if err := json.Unmarshal(data, &result); err != nil {
		t.Fatal(err)
	}
	return result
}

func TestCacheRejectsDifferentPrefixOrderMappingAndRoles(t *testing.T) {
	c, e := fixture(t)
	cases := map[string]func(*fusioncache.Cache){
		"free_generation": func(c *fusioncache.Cache) { c.Mode = "free_generation" },
		"generated_prefix": func(c *fusioncache.Cache) {
			c.Records[0].Positions[1].TeacherPrefixSHA256 = fusioncache.PrefixDigest([]int64{1, 2, 7})
		},
		"student_prefix": func(c *fusioncache.Cache) { c.Records[0].Positions[0].StudentPrefixSHA256 = strings.Repeat("0", 64) },
		"position_order": func(c *fusioncache.Cache) {
			c.Records[0].Positions[0], c.Records[0].Positions[1] = c.Records[0].Positions[1], c.Records[0].Positions[0]
		},
		"token_mapping":         func(c *fusioncache.Cache) { c.Records[0].Positions[0].Probabilities[0].StudentTokenID = 1 },
		"mapping_digest":        func(c *fusioncache.Cache) { c.MappingSHA256 = strings.Repeat("0", 64) },
		"mapping_contents":      func(c *fusioncache.Cache) { c.Mapping.EvidenceSHA256 = strings.Repeat("0", 64) },
		"revision":              func(c *fusioncache.Cache) { c.Teacher.Revision = strings.Repeat("0", 40) },
		"tokenizer":             func(c *fusioncache.Cache) { c.Teacher.TokenizerSHA256 = strings.Repeat("0", 64) },
		"runtime":               func(c *fusioncache.Cache) { c.Teacher.RuntimeSHA256 = strings.Repeat("0", 64) },
		"gold_tokens":           func(c *fusioncache.Cache) { c.Records[0].Example.TeacherTokens[3] = 5 },
		"mass":                  func(c *fusioncache.Cache) { c.Records[0].Positions[0].RetainedMass = 0.5 },
		"duplicate_token":       func(c *fusioncache.Cache) { c.Records[0].Positions[0].Probabilities[1].TeacherTokenID = 2 },
		"nonfinite_probability": func(c *fusioncache.Cache) { c.Records[0].Positions[0].Probabilities[0].Probability = math.NaN() },
		"nonfinite_feature":     func(c *fusioncache.Cache) { c.Records[0].Positions[0].Features[0].Values[0] = math.Inf(1) },
		"feature_layer":         func(c *fusioncache.Cache) { c.Records[0].Positions[0].Features[0].Spec.Layer = 2 },
		"missing_position":      func(c *fusioncache.Cache) { c.Records[0].Positions = c.Records[0].Positions[:1] },
	}
	for name, change := range cases {
		t.Run(name, func(t *testing.T) {
			bad := clone(t, c)
			change(&bad)
			if err := fusioncache.ValidateAgainstStudent(bad, e, limits()); !errors.Is(err, fusioncache.ErrContract) {
				t.Fatalf("expected contract refusal, got %v", err)
			}
		})
	}
	for _, role := range []string{"protection", "sealed-final", "selection", "pending", "test", ""} {
		t.Run("role_"+role, func(t *testing.T) {
			bad := clone(t, e)
			bad.Examples[0].Role = role
			if err := fusioncache.ValidateExpectation(bad, limits()); err == nil {
				t.Fatal("reserved role admitted")
			}
		})
	}
}

func TestExternalStudentGoldAndExampleOrderAreRequired(t *testing.T) {
	c, e := fixture(t)
	changed := clone(t, e)
	changed.Examples[0].TeacherTokens[3] = 5
	changed.Examples[0].StudentTokens[3] = 5
	if err := fusioncache.ValidateAgainstStudent(c, changed, limits()); err == nil {
		t.Fatal("different student gold admitted")
	}
	c.Records = append(c.Records, clone(t, c.Records[0]))
	c.Records[1].Example.ID = "two"
	e.Examples = append(e.Examples, clone(t, e.Examples[0]))
	e.Examples[1].ID = "two"
	if err := fusioncache.ValidateAgainstStudent(c, e, limits()); err != nil {
		t.Fatal(err)
	}
	c.Records[0], c.Records[1] = c.Records[1], c.Records[0]
	if err := fusioncache.ValidateAgainstStudent(c, e, limits()); err == nil {
		t.Fatal("different example order admitted")
	}
}

func TestQualifiedBijectionAndExactPositionCounts(t *testing.T) {
	c, e := fixture(t)
	e.Student.TokenizerSHA256 = strings.Repeat("8", 64)
	e.Mapping.Identity = false
	e.Mapping.StudentTokenizerSHA256 = e.Student.TokenizerSHA256
	for token := int64(0); token < 8; token++ {
		e.Mapping.Pairs = append(e.Mapping.Pairs, fusioncache.TokenPair{Teacher: token, Student: 7 - token})
	}
	e.MappingSHA256 = digest(t, e.Mapping)
	for i, token := range e.Examples[0].TeacherTokens {
		e.Examples[0].StudentTokens[i] = 7 - token
	}
	c.Student, c.Mapping, c.MappingSHA256 = e.Student, e.Mapping, e.MappingSHA256
	c.Records[0].Example = clone(t, e.Examples[0])
	for i := range c.Records[0].Positions {
		p := &c.Records[0].Positions[i]
		p.StudentPrefixSHA256 = fusioncache.PrefixDigest(e.Examples[0].StudentTokens[:p.TargetIndex])
		for j := range p.Probabilities {
			p.Probabilities[j].StudentTokenID = 7 - p.Probabilities[j].TeacherTokenID
		}
	}
	if err := fusioncache.ValidateAgainstStudent(c, e, limits()); err != nil {
		t.Fatal(err)
	}
	bad := clone(t, e)
	bad.Mapping.Pairs[1].Student = bad.Mapping.Pairs[0].Student
	bad.MappingSHA256 = digest(t, bad.Mapping)
	if err := fusioncache.ValidateExpectation(bad, limits()); err == nil {
		t.Fatal("nonbijective mapping admitted")
	}
	bad = clone(t, e)
	bad.Examples[0].StudentTokens = append(bad.Examples[0].StudentTokens, 1)
	if err := fusioncache.ValidateExpectation(bad, limits()); err == nil {
		t.Fatal("different token boundaries admitted")
	}
}

func TestAtomicRoundTripDigestBoundsAndTornFile(t *testing.T) {
	c, e := fixture(t)
	path := filepath.Join(t.TempDir(), "cache.json")
	receipt, err := fusioncache.WriteAtomic(path, c, e, limits())
	if err != nil {
		t.Fatal(err)
	}
	read, err := fusioncache.Read(path, receipt.SHA256, e, limits())
	if err != nil || !reflect.DeepEqual(read, c) {
		t.Fatalf("round trip differs: %v", err)
	}
	again, err := fusioncache.WriteAtomic(path, c, e, limits())
	if err != nil || again != receipt {
		t.Fatalf("idempotent write differs: %v", err)
	}
	altered := clone(t, c)
	altered.Records[0].Positions[0].Features[0].Values[0] = 9
	if _, err := fusioncache.WriteAtomic(path, altered, e, limits()); err == nil {
		t.Fatal("immutable artifact replaced")
	}
	if _, err := fusioncache.Read(path, strings.Repeat("0", 64), e, limits()); err == nil {
		t.Fatal("wrong external digest accepted")
	}
	bound := limits()
	bound.MaxBytes = 8
	if _, err := fusioncache.Read(path, receipt.SHA256, e, bound); err == nil {
		t.Fatal("oversized file read")
	}
	if _, err := fusioncache.WriteAtomic(filepath.Join(t.TempDir(), "small.json"), c, e, bound); err == nil {
		t.Fatal("oversized file written")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	torn := data[:len(data)/2]
	broken := filepath.Join(t.TempDir(), "broken.json")
	if err := os.WriteFile(broken, torn, 0600); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(torn)
	if _, err := fusioncache.Read(broken, hex.EncodeToString(sum[:]), e, limits()); err == nil {
		t.Fatal("torn JSON with matching bytes digest accepted")
	}
	for _, suffix := range []string{"{}", "garbage"} {
		bad := append(append([]byte(nil), data...), []byte(suffix)...)
		if err := os.WriteFile(broken, bad, 0600); err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(bad)
		if _, err := fusioncache.Read(broken, hex.EncodeToString(sum[:]), e, limits()); err == nil {
			t.Fatal("trailing JSON admitted")
		}
	}
	entries, err := os.ReadDir(filepath.Dir(path))
	if err != nil || len(entries) != 1 {
		t.Fatalf("temporary files remain: %v %v", entries, err)
	}
}

type fakeFactory struct {
	events []string
	active bool
	failAt string
	cancel context.CancelFunc
	opens  int
}
type fakeProducer struct {
	factory  *fakeFactory
	identity fusioncache.ModelIdentity
	calls    int
}

func (f *fakeFactory) Open(_ context.Context, identity fusioncache.ModelIdentity) (fusioncache.Producer, error) {
	if f.active {
		return nil, fmt.Errorf("overlapping residency")
	}
	f.active = true
	f.opens++
	f.events = append(f.events, "open:"+identity.Name)
	return &fakeProducer{factory: f, identity: identity}, nil
}
func (p *fakeProducer) Identity() fusioncache.ModelIdentity { return p.identity }
func (p *fakeProducer) TeacherForce(_ context.Context, request fusioncache.PrefixRequest) (fusioncache.Signal, error) {
	p.calls++
	p.factory.events = append(p.factory.events, "force:"+p.identity.Name)
	if request.TargetIndex != len(request.Tokens) || request.PrefixSHA256 != fusioncache.PrefixDigest(request.Tokens) {
		return fusioncache.Signal{}, fmt.Errorf("invalid causal prefix")
	}
	if p.calls == 1 && !reflect.DeepEqual(request.Tokens, []int64{1, 2}) {
		return fusioncache.Signal{}, fmt.Errorf("future gold leaked into first request")
	}
	if p.calls == 2 && !reflect.DeepEqual(request.Tokens, []int64{1, 2, 3}) {
		return fusioncache.Signal{}, fmt.Errorf("previous gold missing")
	}
	if p.factory.failAt == "force" {
		return fusioncache.Signal{}, fmt.Errorf("backend failed")
	}
	if p.factory.cancel != nil {
		p.factory.cancel()
	}
	signal := fusioncache.Signal{PrefixSHA256: request.PrefixSHA256, TargetIndex: request.TargetIndex, RetainedMass: 0.75, Probabilities: []fusioncache.TeacherProbability{{TokenID: 4, Probability: 0.5}, {TokenID: 2, Probability: 0.25}}}
	for _, spec := range request.Features {
		signal.Features = append(signal.Features, fusioncache.Feature{Spec: spec, Values: []float64{1, 2}})
	}
	if p.factory.failAt == "prefix" {
		signal.PrefixSHA256 = fusioncache.PrefixDigest([]int64{7, 7})
	}
	return signal, nil
}
func (p *fakeProducer) Close() error {
	p.factory.events = append(p.factory.events, "close:"+p.identity.Name)
	if p.factory.failAt == "close" {
		return fmt.Errorf("close failed")
	}
	p.factory.active = false
	return nil
}

type fakeSink struct {
	factory *fakeFactory
	sink    fusioncache.DirectorySink
}

func (s fakeSink) Store(ctx context.Context, c fusioncache.Cache, e fusioncache.Expectation, l fusioncache.Limits) (fusioncache.Receipt, error) {
	if s.factory.active {
		return fusioncache.Receipt{}, fmt.Errorf("teacher still resident while persisting")
	}
	s.factory.events = append(s.factory.events, "store:"+e.Teacher.Name)
	if s.factory.failAt == "store" {
		return fusioncache.Receipt{}, fmt.Errorf("persist failed")
	}
	return s.sink.Store(ctx, c, e, l)
}

func TestSequentialProducerClosesBeforePersistenceAndNextTeacher(t *testing.T) {
	one := expectation(t)
	two := clone(t, one)
	two.Teacher.Name = "second"
	factory := &fakeFactory{}
	sink := fakeSink{factory: factory, sink: fusioncache.DirectorySink{Directory: t.TempDir()}}
	receipts, err := fusioncache.ProduceSequential(context.Background(), []fusioncache.Expectation{one, two}, factory, sink, limits())
	if err != nil || len(receipts) != 2 {
		t.Fatalf("production failed: %v", err)
	}
	want := []string{"open:teacher", "force:teacher", "force:teacher", "close:teacher", "store:teacher", "open:second", "force:second", "force:second", "close:second", "store:second"}
	if !reflect.DeepEqual(factory.events, want) {
		t.Fatalf("unexpected order %v", factory.events)
	}
	if _, err := fusioncache.Read(receipts[0].Location, receipts[0].SHA256, one, limits()); err != nil {
		t.Fatal(err)
	}
	for _, failure := range []string{"force", "prefix", "close", "store"} {
		t.Run(failure, func(t *testing.T) {
			f := &fakeFactory{failAt: failure}
			s := fakeSink{factory: f, sink: fusioncache.DirectorySink{Directory: t.TempDir()}}
			r, err := fusioncache.ProduceSequential(context.Background(), []fusioncache.Expectation{one, two}, f, s, limits())
			if err == nil || len(r) != 0 || f.opens != 1 {
				t.Fatalf("failure did not stop sequence: %v %v", err, f.events)
			}
			closed := false
			for _, event := range f.events {
				if event == "close:teacher" {
					closed = true
				}
			}
			if !closed {
				t.Fatal("failed teacher not closed")
			}
		})
	}
}

func TestCancellationClosesTeacherAndInvalidJobsNeverOpen(t *testing.T) {
	e := expectation(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	factory := &fakeFactory{cancel: cancel}
	sink := fakeSink{factory: factory, sink: fusioncache.DirectorySink{Directory: t.TempDir()}}
	if _, err := fusioncache.ProduceSequential(ctx, []fusioncache.Expectation{e}, factory, sink, limits()); !errors.Is(err, context.Canceled) || factory.active {
		t.Fatalf("cancellation did not close: %v", err)
	}
	factory = &fakeFactory{}
	sink.factory = factory
	bad := clone(t, e)
	bad.Examples[0].Role = "sealed-final"
	if _, err := fusioncache.ProduceSequential(context.Background(), []fusioncache.Expectation{e, bad}, factory, sink, limits()); err == nil || factory.opens != 0 {
		t.Fatal("backend opened before all jobs admitted")
	}
}
