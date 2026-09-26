package teacherstage_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tayi-ai/arandu-llama/training/fusioncache"
	"github.com/tayi-ai/arandu-llama/training/pipeline"
	"github.com/tayi-ai/arandu-llama/training/teachers"
)

func digest(t *testing.T, value any) string {
	t.Helper()
	h, e := fusioncache.Digest(value)
	if e != nil {
		t.Fatal(e)
	}
	return h
}

type factory struct {
	identity      fusioncache.ModelIdentity
	opens, closes int
	active        bool
}

func (f *factory) Open(context.Context, fusioncache.ModelIdentity) (fusioncache.Producer, error) {
	if f.active {
		return nil, errors.New("overlapping teacher")
	}
	f.active = true
	f.opens++
	return &producer{f}, nil
}

type producer struct{ owner *factory }

func (p *producer) Identity() fusioncache.ModelIdentity { return p.owner.identity }
func (p *producer) Close() error                        { p.owner.active = false; p.owner.closes++; return nil }
func (p *producer) TeacherForce(_ context.Context, r fusioncache.PrefixRequest) (fusioncache.Signal, error) {
	return fusioncache.Signal{PrefixSHA256: r.PrefixSHA256, TargetIndex: r.TargetIndex, RetainedMass: .5, Probabilities: []fusioncache.TeacherProbability{{TokenID: 0, Probability: .5}}, Features: []fusioncache.Feature{{Spec: r.Features[0], Values: []float64{1, 2}}}}, nil
}
func ref(id, letter string) pipeline.ArtifactRef {
	return pipeline.ArtifactRef{ID: id, SHA256: strings.Repeat(letter, 64)}
}
func fixture(t *testing.T) (teachers.CacheStageConfig, pipeline.StageContext, []*factory) {
	t.Helper()
	r := pipeline.Recipe{Version: 1, ID: "arbitrary-method", Method: "generational-fusion-v1", Student: ref("student", "a"), Teachers: []pipeline.ArtifactRef{ref("teacher-one", "b"), ref("teacher-two", "c")}, Training: ref("train", "d"), Recovery: ref("recovery", "e"), Calibration: ref("calibration", "f"), Protection: ref("protection", "1"), Heldout: ref("heldout", "2")}
	phases := []pipeline.Phase{pipeline.PhaseSFT, pipeline.PhaseTeacherCache, pipeline.PhaseAlignment, pipeline.PhaseFusion, pipeline.PhaseRecovery, pipeline.PhaseMaster, pipeline.PhaseCalibration, pipeline.PhaseQuantize, pipeline.PhaseVariantRecovery, pipeline.PhaseQuantize, pipeline.PhaseVariantRecovery, pipeline.PhaseQuantize, pipeline.PhaseVariantRecovery}
	for i, phase := range phases {
		s := pipeline.Stage{ID: []string{"sft", "cache", "alignment", "fusion", "recovery", "master", "calibration", "q8", "r8", "q6", "r6", "q4", "r4"}[i], Phase: phase, MaxSteps: 2, MaxTokens: 32, TimeoutSeconds: 10}
		if i > 0 {
			s.ParentStage = r.Stages[i-1].ID
		}
		data := r.Training
		switch phase {
		case pipeline.PhaseRecovery, pipeline.PhaseVariantRecovery:
			data = r.Recovery
		case pipeline.PhaseMaster:
			data = r.Heldout
		case pipeline.PhaseCalibration, pipeline.PhaseQuantize:
			data = r.Calibration
		}
		s.Inputs = []pipeline.ArtifactRef{data, r.Student}
		if phase == pipeline.PhaseTeacherCache {
			s.Inputs = append(s.Inputs, r.Teachers...)
		}
		if phase == pipeline.PhaseFusion || phase == pipeline.PhaseRecovery || phase == pipeline.PhaseVariantRecovery {
			s.Inputs = append(s.Inputs, r.Protection)
		}
		if phase == pipeline.PhaseQuantize {
			s.ParentStage = "master"
		}
		if i >= 7 {
			s.Format = []string{"Q8_0", "Q6_K", "Q4_K_M"}[(i-7)/2]
		}
		r.Stages = append(r.Stages, s)
	}
	rhash, e := r.Digest()
	if e != nil {
		t.Fatal(e)
	}
	qualification := strings.Repeat("3", 64)
	placement := pipeline.Placement{Backend: "fixture", MemoryBytes: 512 << 20, MaxTokens: 32, RuntimeSHA256: strings.Repeat("8", 64), QualificationSHA256: qualification}
	config := teachers.CacheStageConfig{MaxPlanBytes: 1 << 20, Plan: teachers.CacheStagePlan{Version: 1, StageID: "cache", RecipeSHA256: rhash, QualificationSHA256: qualification, Placement: placement, Limits: fusioncache.Limits{MaxBytes: 1 << 20, MaxExamples: 4, MaxTokens: 32, MaxPositions: 32, MaxTopK: 4, MaxFeatureValues: 1024, MaxMappingPairs: 32}}}
	student := fusioncache.ModelIdentity{Name: "student", Revision: strings.Repeat("a", 40), WeightsSHA256: r.Student.SHA256, TokenizerSHA256: strings.Repeat("4", 64), TemplateSHA256: strings.Repeat("5", 64), RuntimeSHA256: strings.Repeat("6", 64), Vocabulary: 8}
	mapping := fusioncache.TokenMapping{Identity: true, TeacherTokenizerSHA256: student.TokenizerSHA256, StudentTokenizerSHA256: student.TokenizerSHA256, EvidenceSHA256: strings.Repeat("7", 64)}
	var factories []*factory
	for _, teacherRef := range r.Teachers {
		teacher := student
		teacher.Name = teacherRef.ID
		teacher.WeightsSHA256 = teacherRef.SHA256
		ex := fusioncache.Expectation{Teacher: teacher, Student: student, Mapping: mapping, MappingSHA256: digest(t, mapping), Features: []fusioncache.FeatureSpec{{Layer: 0, Tensor: "result_norm", DType: "float32", Dimension: 2}}, Examples: []fusioncache.Example{{ID: "row", DatasetID: "training", DatasetSHA256: r.Training.SHA256, Role: "train", TeacherTokens: []int64{1, 2, 3}, StudentTokens: []int64{1, 2, 3}, PromptTokens: 1}}}
		config.Plan.Expectations = append(config.Plan.Expectations, ex)
		f := &factory{identity: teacher}
		factories = append(factories, f)
		config.Factories = append(config.Factories, f)
	}
	config.PlanSHA256 = digest(t, config.Plan)
	return config, pipeline.StageContext{Execution: pipeline.Execution{Recipe: r, Placement: placement}, Stage: r.Stages[1], ArtifactDirectory: t.TempDir()}, factories
}
func stage(t *testing.T, c teachers.CacheStageConfig) *teachers.CacheStage {
	t.Helper()
	s, e := teachers.NewCacheStage(c)
	if e != nil {
		t.Fatal(e)
	}
	return s
}

func TestCacheStageRecoversCommitFailureWithoutReopeningTeacher(t *testing.T) {
	c, x, f := fixture(t)
	s := stage(t, c)
	interrupted := errors.New("lost receipt transaction")
	ctx := context.Background()
	if e := s.Run(ctx, x, func(context.Context, pipeline.StageResult) error {
		if f[0].active {
			t.Fatal("teacher resident at commit")
		}
		return interrupted
	}); !errors.Is(e, interrupted) {
		t.Fatal(e)
	}
	if f[0].opens != 1 || f[1].opens != 0 {
		t.Fatal("production continued after failed commit")
	}
	s = stage(t, c)
	var recovered []pipeline.StageReceipt
	if e := s.Reconcile(ctx, x, func(_ context.Context, r pipeline.StageResult) error {
		recovered = append(recovered, pipeline.StageReceipt{StageID: x.Stage.ID, Result: r})
		return nil
	}); e != nil {
		t.Fatal(e)
	}
	if len(recovered) != 1 || recovered[0].Result.Complete {
		t.Fatal("wrong resumed boundary")
	}
	x.Previous = &recovered[0]
	if e := s.Run(ctx, x, func(_ context.Context, r pipeline.StageResult) error {
		if r.Steps != 2 || !r.Complete || f[1].active || len(r.Artifacts) != 2 {
			t.Fatal("bad completion")
		}
		return s.Verify(ctx, x, pipeline.StageReceipt{StageID: x.Stage.ID, Result: r})
	}); e != nil {
		t.Fatal(e)
	}
	if f[0].opens != 1 || f[0].closes != 1 || f[1].opens != 1 || f[1].closes != 1 {
		t.Fatal("teacher ownership differs")
	}
}

func TestCacheStageRejectsCorruptCheckpointBeforeResume(t *testing.T) {
	c, x, f := fixture(t)
	s := stage(t, c)
	ctx := context.Background()
	if e := s.Run(ctx, x, func(context.Context, pipeline.StageResult) error { return errors.New("stop") }); e == nil {
		t.Fatal("expected interruption")
	}
	if e := os.WriteFile(filepath.Join(x.ArtifactDirectory, "cache-000001.json"), []byte("{}\n"), 0600); e != nil {
		t.Fatal(e)
	}
	if e := s.Reconcile(ctx, x, func(context.Context, pipeline.StageResult) error { return nil }); e == nil {
		t.Fatal("corrupt checkpoint accepted")
	}
	if f[0].opens != 1 || f[1].opens != 0 {
		t.Fatal("reconcile launched work")
	}
}

func TestCacheStageRefusesDataRoleOrMethodChangesWithoutCompute(t *testing.T) {
	for _, change := range []string{"role", "method", "budget", "qualification", "memory", "backend", "runtime", "nodes", "missing-teacher"} {
		t.Run(change, func(t *testing.T) {
			c, x, f := fixture(t)
			switch change {
			case "role":
				c.Plan.Expectations[0].Examples[0].Role = "calibration"
			case "method":
				x.Execution.Recipe.ID = "changed"
			case "budget":
				x.Stage.MaxTokens = 2
			case "qualification":
				x.Execution.Placement.QualificationSHA256 = strings.Repeat("9", 64)
			case "memory":
				x.Execution.Placement.MemoryBytes = 1
			case "backend":
				x.Execution.Placement.Backend = "unqualified"
			case "runtime":
				x.Execution.Placement.RuntimeSHA256 = strings.Repeat("9", 64)
			case "nodes":
				x.Execution.Placement.Nodes = []string{"one"}
			case "missing-teacher":
				c.Plan.Expectations = c.Plan.Expectations[:1]
				c.Factories = c.Factories[:1]
				x.Stage.MaxSteps = 1
			}
			c.PlanSHA256 = digest(t, c.Plan)
			s := stage(t, c)
			if e := s.Admit(context.Background(), x.Execution.Recipe, x.Stage, x.Execution.Placement); e == nil {
				t.Fatal("changed plan admitted")
			}
			if f[0].opens != 0 {
				t.Fatal("admission started teacher")
			}
		})
	}
}

func TestCacheStageFinalReceiptProtectsEveryTeacher(t *testing.T) {
	c, x, _ := fixture(t)
	s := stage(t, c)
	var final pipeline.StageReceipt
	if err := s.Run(context.Background(), x, func(_ context.Context, r pipeline.StageResult) error {
		final = pipeline.StageReceipt{StageID: x.Stage.ID, Result: r}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if len(final.Result.Artifacts) != 2 {
		t.Fatal("completed stage omitted teacher evidence")
	}
	if err := os.WriteFile(filepath.Join(x.ArtifactDirectory, final.Result.Artifacts[0].Path), []byte("{}\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := s.Verify(context.Background(), x, final); err == nil {
		t.Fatal("completed stage accepted corrupted earlier teacher")
	}
}

func TestCacheStageSnapshotsPlanAndStopsOnCancellation(t *testing.T) {
	c, x, f := fixture(t)
	s := stage(t, c)
	c.Plan.Expectations[0].Examples[0].TeacherTokens[0] = 7
	if e := s.Admit(context.Background(), x.Execution.Recipe, x.Stage, x.Execution.Placement); e != nil {
		t.Fatal(e)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if e := s.Run(ctx, x, func(context.Context, pipeline.StageResult) error { return nil }); !errors.Is(e, context.Canceled) {
		t.Fatal(e)
	}
	if f[0].opens != 0 {
		t.Fatal("cancelled production started")
	}
}

type blockingFactory struct {
	*factory
	entered chan struct{}
	release chan struct{}
}

func (f *blockingFactory) Open(ctx context.Context, id fusioncache.ModelIdentity) (fusioncache.Producer, error) {
	close(f.entered)
	select {
	case <-f.release:
		return f.factory.Open(ctx, id)
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func TestCacheStageSerializesRunsWithCancelableWait(t *testing.T) {
	c, x, f := fixture(t)
	blocking := &blockingFactory{factory: f[0], entered: make(chan struct{}), release: make(chan struct{})}
	c.Factories[0] = blocking
	s := stage(t, c)
	done := make(chan error, 1)
	go func() {
		done <- s.Run(context.Background(), x, func(context.Context, pipeline.StageResult) error { return nil })
	}()
	<-blocking.entered
	x.ArtifactDirectory = t.TempDir()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	err := s.Run(ctx, x, func(context.Context, pipeline.StageResult) error { return nil })
	close(blocking.release)
	if first := <-done; first != nil {
		t.Fatal(first)
	}
	if !errors.Is(err, context.DeadlineExceeded) || f[0].opens != 1 || f[1].opens != 1 {
		t.Fatalf("overlapping run was not fenced: %v", err)
	}
}

func TestCacheStageRefusesOversizedMetadata(t *testing.T) {
	c, _, _ := fixture(t)
	c.Plan.Placement.Nodes = []string{strings.Repeat("x", 1<<20)}
	if _, err := teachers.NewCacheStage(c); err == nil {
		t.Fatal("oversized metadata accepted")
	}
}
