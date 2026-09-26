package alignment_test

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

	"github.com/tayi-ai/arandu-llama/training/alignment"
	"github.com/tayi-ai/arandu-llama/training/fusioncache"
	"github.com/tayi-ai/arandu-llama/training/pipeline"
)

func hash(body []byte) string { h := sha256.Sum256(body); return hex.EncodeToString(h[:]) }
func digest(v any) string {
	h, err := fusioncache.Digest(v)
	if err != nil {
		panic(err)
	}
	return h
}
func ref(id string) pipeline.ArtifactRef {
	return pipeline.ArtifactRef{ID: id, SHA256: hash([]byte(id))}
}
func model(id string) fusioncache.ModelIdentity {
	return fusioncache.ModelIdentity{Name: id, Revision: strings.Repeat("1", 40), WeightsSHA256: ref(id).SHA256, TokenizerSHA256: ref("tokens").SHA256, TemplateSHA256: ref("template").SHA256, RuntimeSHA256: ref("runtime").SHA256, Vocabulary: 16}
}
func seal(r pipeline.StageReceipt) pipeline.StageReceipt {
	r.SHA256 = ""
	r.SHA256 = digest(r)
	return r
}

func recipe() pipeline.Recipe {
	r := pipeline.Recipe{Version: 1, ID: "synthetic", Method: "generational-fusion-v1", Student: ref("student"), Teachers: []pipeline.ArtifactRef{ref("teacher")}, Training: ref("train"), Recovery: ref("recovery"), Calibration: ref("calibration"), Protection: ref("protection"), Heldout: ref("heldout")}
	add := func(phase pipeline.Phase, input pipeline.ArtifactRef, format string) {
		parent := ""
		if len(r.Stages) > 0 {
			parent = r.Stages[len(r.Stages)-1].ID
		}
		if phase == pipeline.PhaseQuantize {
			parent = r.Stages[5].ID
		}
		inputs := []pipeline.ArtifactRef{input}
		if phase == pipeline.PhaseFusion || phase == pipeline.PhaseRecovery || phase == pipeline.PhaseVariantRecovery {
			inputs = append(inputs, r.Protection)
		}
		if phase == pipeline.PhaseAlignment {
			inputs = append(inputs, r.Student, r.Teachers[0])
		}
		r.Stages = append(r.Stages, pipeline.Stage{ID: fmt.Sprintf("stage-%d", len(r.Stages)), Phase: phase, Inputs: inputs, ParentStage: parent, Format: format, MaxSteps: 1, MaxTokens: 128, TimeoutSeconds: 60})
	}
	for _, phase := range []pipeline.Phase{pipeline.PhaseSFT, pipeline.PhaseTeacherCache, pipeline.PhaseAlignment, pipeline.PhaseFusion} {
		add(phase, r.Training, "")
	}
	add(pipeline.PhaseRecovery, r.Recovery, "")
	add(pipeline.PhaseMaster, r.Heldout, "")
	add(pipeline.PhaseCalibration, r.Calibration, "")
	for _, format := range []string{"Q8_0", "Q6_K", "Q4_K_M"} {
		add(pipeline.PhaseQuantize, r.Calibration, format)
		add(pipeline.PhaseVariantRecovery, r.Recovery, format)
	}
	return r
}

func writeJSON(t *testing.T, path string, v any) []byte {
	t.Helper()
	body, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	body = append(body, '\n')
	if err := os.WriteFile(path, body, 0600); err != nil {
		t.Fatal(err)
	}
	return body
}
func pin(t *testing.T, c *alignment.Config) {
	t.Helper()
	sha, err := c.Protocol.Digest()
	if err != nil {
		t.Fatal(err)
	}
	c.ProtocolSHA256 = sha
}
func handler(t *testing.T, c alignment.Config) *alignment.Stage {
	t.Helper()
	h, err := alignment.NewStage(c)
	if err != nil {
		t.Fatal(err)
	}
	return h
}
func noCommit(context.Context, pipeline.StageResult) error { return nil }

func fixture(t *testing.T, count int) (alignment.Config, pipeline.StageContext) {
	t.Helper()
	r := recipe()
	r.Stages[2].MaxSteps = count
	rsha, err := r.Digest()
	if err != nil {
		t.Fatal(err)
	}
	pl := pipeline.Placement{Backend: "fixture-cpu", MemoryBytes: 64 << 20, MaxTokens: 128, RuntimeSHA256: ref("runtime").SHA256, QualificationSHA256: ref("qualification").SHA256}
	x := pipeline.Execution{RunID: "run-1", TenantID: "tenant-1", Generation: 1, Recipe: r, Placement: pl, TargetSHA256: ref("target").SHA256}
	id := pipeline.ReceiptIdentity{RunID: x.RunID, TenantID: x.TenantID, RecipeSHA256: rsha, TargetSHA256: x.TargetSHA256, PlacementSHA256: digest(pl), Generation: 1}
	c := pipeline.StageContext{Execution: x, Stage: r.Stages[2], ArtifactDirectory: t.TempDir()}
	p := alignment.Protocol{Version: 1, RecipeSHA256: rsha, PlacementSHA256: digest(pl), QualificationSHA256: pl.QualificationSHA256, StageID: c.Stage.ID, Limits: alignment.Limits{MaxProtocolBytes: 1 << 20, MaxSourceBytes: 1 << 16, MaxArtifactBytes: 1 << 16, MaxTotalBytes: int64(count) << 16, MaxFits: count, MaxSolveWork: 10000, Projection: fusioncache.ProjectionLimits{MaxSamples: 10, MaxDimension: 8, MaxElements: 1000}}}
	parent := pipeline.StageReceipt{Version: 1, Identity: id, StageID: c.Stage.ParentStage, Result: pipeline.StageResult{Steps: 1, Complete: true}}
	for i := 0; i < count; i++ {
		sample := func(name string, source, target []float64) fusioncache.FeatureSample {
			return fusioncache.FeatureSample{Identity: fusioncache.SampleIdentity{DatasetSHA256: r.Training.SHA256, ExampleID: name, Role: "train", TargetIndex: 2, TeacherPrefixSHA256: fusioncache.PrefixDigest([]int64{1, 2}), StudentPrefixSHA256: fusioncache.PrefixDigest([]int64{1, 2})}, Source: source, Target: target}
		}
		fit := []fusioncache.FeatureSample{sample("fit-one", []float64{1, 0}, []float64{2, 3}), sample("fit-two", []float64{0, 1}, []float64{-1, 4})}
		held := []fusioncache.FeatureSample{sample("validation", []float64{2, 1}, []float64{3, 10})}
		plan := fusioncache.MatchingPlan{ID: fmt.Sprintf("projection-%d", i), SourceModel: model("teacher"), TargetModel: model("student"), TokenMappingSHA256: ref("mapping").SHA256, Source: fusioncache.FeatureSpec{Layer: 3, Tensor: "post_norm", DType: "float64", Dimension: 2}, Target: fusioncache.FeatureSpec{Layer: 1, Tensor: "post_norm", DType: "float64", Dimension: 2}, Ridge: .5, Fit: []fusioncache.SampleIdentity{fit[0].Identity, fit[1].Identity}, Heldout: []fusioncache.SampleIdentity{held[0].Identity}}
		samples := alignment.Samples{Version: 1, MatchingPlanSHA256: digest(plan), SourceModel: plan.SourceModel, TargetModel: plan.TargetModel, TokenMappingSHA256: plan.TokenMappingSHA256, Source: plan.Source, Target: plan.Target, Fit: fit, Heldout: held}
		name := fmt.Sprintf("samples-%d.json", i)
		body := writeJSON(t, filepath.Join(c.ArtifactDirectory, name), samples)
		a := pipeline.StageArtifact{Path: name, SHA256: hash(body), Bytes: int64(len(body))}
		parent.Result.Artifacts = append(parent.Result.Artifacts, a)
		p.Fits = append(p.Fits, alignment.Fit{MatchingPlan: plan, MatchingPlanSHA256: digest(plan), Source: alignment.Source{Artifact: a, StageID: parent.StageID}, Qualification: alignment.Qualification{MaxFitMSE: 1, MaxHeldoutMSE: 7}})
	}
	parent = seal(parent)
	c.Parent = &parent
	c.Completed = []pipeline.StageReceipt{parent}
	for i := range p.Fits {
		p.Fits[i].Source.ReceiptSHA256 = parent.SHA256
	}
	config := alignment.Config{Recipe: r, Placement: pl, Protocol: p, SourceDirectory: t.TempDir()}
	pin(t, &config)
	return config, c
}

func commit(t *testing.T, h *alignment.Stage, c *pipeline.StageContext) func(context.Context, pipeline.StageResult) error {
	t.Helper()
	return func(ctx context.Context, result pipeline.StageResult) error {
		previous := c.Parent.SHA256
		if c.Previous != nil {
			previous = c.Previous.SHA256
		}
		id := c.Parent.Identity
		id.Generation = c.Execution.Generation
		r := seal(pipeline.StageReceipt{Version: 1, Identity: id, StageID: c.Stage.ID, Result: result, PreviousSHA256: previous})
		if err := h.Verify(ctx, *c, r); err != nil {
			return err
		}
		c.Previous = &r
		return nil
	}
}

func evidence(t *testing.T, c pipeline.StageContext, step int) (string, alignment.Evidence) {
	t.Helper()
	name := filepath.Join(c.ArtifactDirectory, fmt.Sprintf("alignment-%s-%06d.json", c.Stage.ID, step))
	b, err := os.ReadFile(name)
	if err != nil {
		t.Fatal(err)
	}
	var e alignment.Evidence
	if err := json.Unmarshal(b, &e); err != nil {
		t.Fatal(err)
	}
	return name, e
}

func TestStageFitsKnownRidgeMatrixAndPersistsProvenance(t *testing.T) {
	c, x := fixture(t, 1)
	h := handler(t, c)
	if err := h.Run(context.Background(), x, commit(t, h, &x)); err != nil {
		t.Fatal(err)
	}
	_, e := evidence(t, x, 1)
	want := [][]float64{{4.0 / 3, 2}, {-2.0 / 3, 8.0 / 3}}
	for i, row := range want {
		for j, v := range row {
			if math.Abs(e.Projection.Coefficients[i][j]-v) > 1e-12 {
				t.Fatal(e.Projection.Coefficients)
			}
		}
	}
	if e.Report.FitValues != 4 || e.Report.HeldoutValues != 2 || math.Abs(e.Report.FitSquaredError-30.0/9) > 1e-12 || math.Abs(e.Report.HeldoutSquaredError-109.0/9) > 1e-12 {
		t.Fatal(e.Report)
	}
	if e.Source != c.Protocol.Fits[0].Source || e.ProtocolSHA256 != c.ProtocolSHA256 || e.ParentSHA256 != x.Parent.SHA256 || x.Previous == nil || !x.Previous.Result.Complete {
		t.Fatal("provenance or completion missing")
	}
	if err := h.Verify(context.Background(), x, *x.Previous); err != nil {
		t.Fatal(err)
	}
}

func TestRestartRecoversLostCallbackWithoutReplacingProjection(t *testing.T) {
	c, x := fixture(t, 2)
	h := handler(t, c)
	lost := errors.New("receipt transaction lost")
	if err := h.Run(context.Background(), x, func(context.Context, pipeline.StageResult) error { return lost }); !errors.Is(err, lost) {
		t.Fatal(err)
	}
	name, e := evidence(t, x, 1)
	before, err := os.Stat(name)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(x.ArtifactDirectory, "alignment-stage-2-000002.json")); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("continued after callback error")
	}
	x.Execution.Generation = 2
	h = handler(t, c)
	if err := h.Reconcile(context.Background(), x, commit(t, h, &x)); err != nil {
		t.Fatal(err)
	}
	if x.Previous == nil || x.Previous.Result.Steps != 1 || x.Previous.Result.Complete {
		t.Fatal("reconcile fitted a missing projection")
	}
	after, err := os.Stat(name)
	if err != nil || !os.SameFile(before, after) || before.ModTime() != after.ModTime() {
		t.Fatal("projection was refitted or replaced")
	}
	_, again := evidence(t, x, 1)
	if !reflect.DeepEqual(e, again) {
		t.Fatal("projection evidence changed")
	}
	if err := h.Run(context.Background(), x, commit(t, h, &x)); err != nil {
		t.Fatal(err)
	}
	if x.Previous.Result.Steps != 2 || !x.Previous.Result.Complete || len(x.Previous.Result.Artifacts) != 2 {
		t.Fatal(x.Previous)
	}
	_, second := evidence(t, x, 2)
	if second.Identity.Generation != 2 || second.PreviousSHA256 != x.Previous.Result.Artifacts[0].SHA256 {
		t.Fatal("continuation lineage differs")
	}
	if err := h.Run(context.Background(), x, func(context.Context, pipeline.StageResult) error {
		t.Fatal("completed result reported twice")
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

func TestStageRejectsTamperedEvidenceAndSource(t *testing.T) {
	for _, kind := range []string{"coefficient", "report", "protocol", "generation", "source", "prefix", "parent"} {
		t.Run(kind, func(t *testing.T) {
			c, x := fixture(t, 1)
			h := handler(t, c)
			if err := h.Run(context.Background(), x, commit(t, h, &x)); err != nil {
				t.Fatal(err)
			}
			name, e := evidence(t, x, 1)
			switch kind {
			case "coefficient":
				e.Projection.Coefficients[0][0]++
			case "report":
				e.Report.HeldoutSquaredError = 0
			case "protocol":
				e.ProtocolSHA256 = ref("other").SHA256
			case "generation":
				e.Identity.Generation = 2
			case "parent":
				e.ParentSHA256 = ref("other").SHA256
			case "source":
				if err := os.WriteFile(filepath.Join(x.ArtifactDirectory, c.Protocol.Fits[0].Source.Artifact.Path), []byte("{}"), 0600); err != nil {
					t.Fatal(err)
				}
			case "prefix":
				var s alignment.Samples
				path := filepath.Join(x.ArtifactDirectory, c.Protocol.Fits[0].Source.Artifact.Path)
				body, _ := os.ReadFile(path)
				_ = json.Unmarshal(body, &s)
				s.Fit[0].Identity.TeacherPrefixSHA256 = ref("different-prefix").SHA256
				writeJSON(t, path, s)
			}
			writeJSON(t, name, e)
			if err := h.Reconcile(context.Background(), x, noCommit); err == nil {
				t.Fatal("tampered state accepted")
			}
		})
	}
}

func TestAdmissionRejectsChangedProtocolAndResourceLimits(t *testing.T) {
	for _, kind := range []string{"hash", "recipe", "placement", "role", "model", "overlap", "duplicate-row", "samples", "elements", "work", "file-bytes", "total-bytes", "negative-threshold", "nonfinite-threshold", "path"} {
		t.Run(kind, func(t *testing.T) {
			c, _ := fixture(t, 1)
			switch kind {
			case "hash":
				c.ProtocolSHA256 = ref("different").SHA256
			case "recipe":
				c.Recipe.ID = "different"
			case "placement":
				c.Placement.MemoryBytes++
			case "role":
				c.Protocol.Fits[0].MatchingPlan.Fit[0].Role = "protection"
			case "model":
				c.Protocol.Fits[0].MatchingPlan.SourceModel.WeightsSHA256 = ref("unadmitted").SHA256
			case "overlap":
				c.Protocol.Fits[0].MatchingPlan.Heldout[0].ExampleID = "fit-one"
			case "duplicate-row":
				c.Protocol.Fits[0].MatchingPlan.Fit[1] = c.Protocol.Fits[0].MatchingPlan.Fit[0]
				c.Protocol.Fits[0].MatchingPlan.Fit[1].TeacherPrefixSHA256 = ref("different").SHA256
			case "samples":
				c.Protocol.Limits.Projection.MaxSamples = 2
			case "elements":
				c.Protocol.Limits.Projection.MaxElements = 1
			case "work":
				c.Protocol.Limits.MaxSolveWork = 1
			case "file-bytes":
				c.Protocol.Limits.MaxSourceBytes = 1
			case "total-bytes":
				c.Protocol.Limits.MaxTotalBytes = 1
			case "negative-threshold":
				c.Protocol.Fits[0].Qualification.MaxFitMSE = -1
			case "nonfinite-threshold":
				c.Protocol.Fits[0].Qualification.MaxHeldoutMSE = math.Inf(1)
			case "path":
				c.Protocol.Fits[0].Source.Artifact.Path = "../escape.json"
			}
			if kind != "hash" && kind != "recipe" && kind != "placement" {
				c.Protocol.Fits[0].MatchingPlanSHA256 = digest(c.Protocol.Fits[0].MatchingPlan)
				if sha, err := c.Protocol.Digest(); err == nil {
					c.ProtocolSHA256 = sha
				} else {
					return
				}
			}
			if _, err := alignment.NewStage(c); err == nil {
				t.Fatal("invalid installation admitted")
			}
		})
	}
}

func TestSourceReceiptMustBeCompletedAndBelongToThisExecution(t *testing.T) {
	for _, kind := range []string{"missing", "incomplete", "other-run", "future-generation", "forged-hash", "artifact-missing"} {
		t.Run(kind, func(t *testing.T) {
			c, x := fixture(t, 1)
			switch kind {
			case "missing":
				x.Completed = nil
			case "incomplete":
				x.Completed[0].Result.Complete = false
			case "other-run":
				x.Completed[0].Identity.RunID = "other"
			case "future-generation":
				x.Completed[0].Identity.Generation = 2
			case "forged-hash":
				x.Completed[0].SHA256 = ref("forged").SHA256
			case "artifact-missing":
				x.Completed[0].Result.Artifacts = nil
			}
			if kind != "missing" && kind != "forged-hash" {
				x.Completed[0] = seal(x.Completed[0])
				c.Protocol.Fits[0].Source.ReceiptSHA256 = x.Completed[0].SHA256
				pin(t, &c)
			}
			if err := handler(t, c).Run(context.Background(), x, noCommit); err == nil {
				t.Fatal("unadmitted predecessor accepted")
			}
		})
	}
}

func TestCancellationAndFailedQualificationNeverReportSuccess(t *testing.T) {
	c, x := fixture(t, 1)
	h := handler(t, c)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := h.Run(ctx, x, noCommit); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if err := h.Reconcile(ctx, x, noCommit); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	c.Protocol.Fits[0].Qualification.MaxHeldoutMSE = 0
	pin(t, &c)
	h = handler(t, c)
	if err := h.Run(context.Background(), x, func(context.Context, pipeline.StageResult) error {
		t.Fatal("failed qualification completed")
		return nil
	}); err == nil {
		t.Fatal("threshold ignored")
	}
	files, _ := filepath.Glob(filepath.Join(x.ArtifactDirectory, "alignment-*.json"))
	if len(files) != 0 {
		t.Fatal("unqualified artifact persisted", files)
	}
}

func TestCancellationAfterFirstCommitStopsFurtherFitting(t *testing.T) {
	c, x := fixture(t, 2)
	h := handler(t, c)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := h.Run(ctx, x, func(context.Context, pipeline.StageResult) error { cancel(); return nil }); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	files, _ := filepath.Glob(filepath.Join(x.ArtifactDirectory, "alignment-*.json"))
	if len(files) != 1 {
		t.Fatal(files)
	}
}

func TestSourceAndOutputSymlinksFailClosed(t *testing.T) {
	for _, which := range []string{"source", "output", "directory", "source-parent"} {
		t.Run(which, func(t *testing.T) {
			c, x := fixture(t, 1)
			h := handler(t, c)
			source := filepath.Join(x.ArtifactDirectory, c.Protocol.Fits[0].Source.Artifact.Path)
			if which == "directory" {
				old := x.ArtifactDirectory
				x.ArtifactDirectory = filepath.Join(t.TempDir(), "link")
				if err := os.Symlink(old, x.ArtifactDirectory); err != nil {
					t.Fatal(err)
				}
			} else if which == "source-parent" {
				dir := t.TempDir()
				body, _ := os.ReadFile(source)
				if err := os.WriteFile(filepath.Join(dir, "source.json"), body, 0600); err != nil {
					t.Fatal(err)
				}
				if err := os.Symlink(dir, filepath.Join(x.ArtifactDirectory, "nested")); err != nil {
					t.Fatal(err)
				}
				c.Protocol.Fits[0].Source.Artifact.Path = "nested/source.json"
				x.Parent.Result.Artifacts[0] = c.Protocol.Fits[0].Source.Artifact
				*x.Parent = seal(*x.Parent)
				x.Completed[0] = *x.Parent
				c.Protocol.Fits[0].Source.ReceiptSHA256 = x.Parent.SHA256
				pin(t, &c)
				h = handler(t, c)
			} else {
				name := source
				if which == "output" {
					name = filepath.Join(x.ArtifactDirectory, "alignment-stage-2-000001.json")
				} else if err := os.Remove(name); err != nil {
					t.Fatal(err)
				}
				target := filepath.Join(t.TempDir(), "sentinel")
				if err := os.WriteFile(target, []byte("private"), 0600); err != nil {
					t.Fatal(err)
				}
				if err := os.Symlink(target, name); err != nil {
					t.Fatal(err)
				}
			}
			if err := h.Run(context.Background(), x, noCommit); err == nil {
				t.Fatal("symlink admitted")
			}
		})
	}
}

func TestInstallationSnapshotAndChangedExecutionAreEnforced(t *testing.T) {
	c, x := fixture(t, 1)
	h := handler(t, c)
	c.Protocol.Fits[0].MatchingPlan.Fit[0].ExampleID = "mutated"
	c.Protocol.Fits[0].Qualification.MaxHeldoutMSE = 0
	if err := h.Run(context.Background(), x, commit(t, h, &x)); err != nil {
		t.Fatal("caller mutated installed protocol", err)
	}
	for _, kind := range []string{"run", "tenant", "target", "generation", "recipe", "placement"} {
		t.Run(kind, func(t *testing.T) {
			changed := x
			switch kind {
			case "run":
				changed.Execution.RunID = "different"
			case "tenant":
				changed.Execution.TenantID = "different"
			case "target":
				changed.Execution.TargetSHA256 = ref("different").SHA256
			case "generation":
				changed.Execution.Generation = 0
			case "recipe":
				changed.Execution.Recipe.ID = "different"
			case "placement":
				changed.Execution.Placement.Backend = "different"
			}
			if err := h.Reconcile(context.Background(), changed, noCommit); err == nil {
				t.Fatal("changed execution accepted")
			}
		})
	}
}

func TestReconcileRejectsMissingOrGappedArtifacts(t *testing.T) {
	c, x := fixture(t, 2)
	h := handler(t, c)
	if err := h.Run(context.Background(), x, commit(t, h, &x)); err != nil {
		t.Fatal(err)
	}
	first, _ := evidence(t, x, 1)
	if err := os.Remove(first); err != nil {
		t.Fatal(err)
	}
	if err := h.Reconcile(context.Background(), x, noCommit); err == nil {
		t.Fatal("gap accepted")
	}
	second, _ := evidence(t, x, 2)
	if err := os.Remove(second); err != nil {
		t.Fatal(err)
	}
	if err := h.Reconcile(context.Background(), x, noCommit); err == nil {
		t.Fatal("completed receipt lost artifacts")
	}
}

func TestStrictSourceJSONRejectsUnknownFields(t *testing.T) {
	c, x := fixture(t, 1)
	path := filepath.Join(x.ArtifactDirectory, c.Protocol.Fits[0].Source.Artifact.Path)
	body, _ := os.ReadFile(path)
	body = []byte(strings.Replace(string(body), "{", "{\"unregistered\":true,", 1))
	if err := os.WriteFile(path, body, 0600); err != nil {
		t.Fatal(err)
	}
	a := c.Protocol.Fits[0].Source.Artifact
	a.SHA256 = hash(body)
	a.Bytes = int64(len(body))
	c.Protocol.Fits[0].Source.Artifact = a
	x.Parent.Result.Artifacts[0] = a
	*x.Parent = seal(*x.Parent)
	x.Completed[0] = *x.Parent
	c.Protocol.Fits[0].Source.ReceiptSHA256 = x.Parent.SHA256
	pin(t, &c)
	if err := handler(t, c).Run(context.Background(), x, noCommit); err == nil {
		t.Fatal("unknown provenance field accepted")
	}
}

// replaceSamples changes only an explicitly repinned test installation. Runtime
// mutation of any of these same bytes is refused by the production handler.
func replaceSamples(t *testing.T, c *alignment.Config, x *pipeline.StageContext, change func(*fusioncache.MatchingPlan, *alignment.Samples)) {
	t.Helper()
	f := &c.Protocol.Fits[0]
	path := filepath.Join(x.ArtifactDirectory, f.Source.Artifact.Path)
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var samples alignment.Samples
	if err := json.Unmarshal(body, &samples); err != nil {
		t.Fatal(err)
	}
	change(&f.MatchingPlan, &samples)
	f.MatchingPlanSHA256 = digest(f.MatchingPlan)
	samples.MatchingPlanSHA256 = f.MatchingPlanSHA256
	samples.Source = f.MatchingPlan.Source
	samples.Target = f.MatchingPlan.Target
	body = writeJSON(t, path, samples)
	f.Source.Artifact.SHA256 = hash(body)
	f.Source.Artifact.Bytes = int64(len(body))
	x.Parent.Result.Artifacts[0] = f.Source.Artifact
	*x.Parent = seal(*x.Parent)
	x.Completed[0] = *x.Parent
	f.Source.ReceiptSHA256 = x.Parent.SHA256
	pin(t, c)
}

func TestStageAcceptsExplicitZeroThresholdForExactZeroTargets(t *testing.T) {
	c, x := fixture(t, 1)
	replaceSamples(t, &c, &x, func(_ *fusioncache.MatchingPlan, s *alignment.Samples) {
		for _, rows := range [][]fusioncache.FeatureSample{s.Fit, s.Heldout} {
			for i := range rows {
				for j := range rows[i].Target {
					rows[i].Target[j] = 0
				}
			}
		}
	})
	c.Protocol.Fits[0].Qualification = alignment.Qualification{MaxFitMSE: 0, MaxHeldoutMSE: 0}
	pin(t, &c)
	h := handler(t, c)
	if err := h.Run(context.Background(), x, commit(t, h, &x)); err != nil {
		t.Fatal(err)
	}
	_, e := evidence(t, x, 1)
	if e.Report.FitSquaredError != 0 || e.Report.HeldoutSquaredError != 0 {
		t.Fatal(e.Report)
	}
}

func TestStageFitsAndRecoversWideDualProjection(t *testing.T) {
	c, x := fixture(t, 1)
	replaceSamples(t, &c, &x, func(p *fusioncache.MatchingPlan, s *alignment.Samples) {
		p.Source.Dimension = 3
		for _, rows := range [][]fusioncache.FeatureSample{s.Fit, s.Heldout} {
			for i := range rows {
				rows[i].Source = append(rows[i].Source, 0)
			}
		}
	})
	h := handler(t, c)
	if err := h.Run(context.Background(), x, commit(t, h, &x)); err != nil {
		t.Fatal(err)
	}
	_, e := evidence(t, x, 1)
	if !e.Projection.Dual || len(e.Projection.Basis) != 2 {
		t.Fatal("wide solve did not retain dual basis")
	}
	got, err := e.Projection.Apply([]float64{2, 1, 0}, c.Protocol.Fits[0].MatchingPlanSHA256, c.Protocol.Limits.Projection)
	if err != nil || math.Abs(got[0]-2) > 1e-12 || math.Abs(got[1]-20.0/3) > 1e-12 {
		t.Fatal(got, err)
	}
	x.Execution.Generation = 2
	if err := handler(t, c).Reconcile(context.Background(), x, noCommit); err != nil {
		t.Fatal(err)
	}
}

func TestRepinnedSourceCannotChangeFrozenCausalPrefixOrDimensions(t *testing.T) {
	for _, kind := range []string{"prefix", "dimension", "model"} {
		t.Run(kind, func(t *testing.T) {
			c, x := fixture(t, 1)
			replaceSamples(t, &c, &x, func(_ *fusioncache.MatchingPlan, s *alignment.Samples) {
				switch kind {
				case "prefix":
					s.Fit[0].Identity.StudentPrefixSHA256 = ref("different-prefix").SHA256
				case "dimension":
					s.Fit[0].Source = append(s.Fit[0].Source, 1)
				case "model":
					s.TargetModel.WeightsSHA256 = ref("different-model").SHA256
				}
			})
			if err := handler(t, c).Run(context.Background(), x, noCommit); err == nil {
				t.Fatal("source contradicted frozen plan")
			}
		})
	}
}

func TestExplicitStageInputSourceWorksWithoutForgingPredecessorArtifact(t *testing.T) {
	c, x := fixture(t, 1)
	// The source is a derived feature pool, not the raw dataset's bytes. Both
	// identities are independently pinned, without any self-referential hash.
	f := &c.Protocol.Fits[0]
	body, err := os.ReadFile(filepath.Join(x.ArtifactDirectory, f.Source.Artifact.Path))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(c.SourceDirectory, f.Source.Artifact.Path), body, 0600); err != nil {
		t.Fatal(err)
	}
	f.Source.Input = c.Recipe.Training
	f.Source.StageID = ""
	f.Source.ReceiptSHA256 = ""
	pin(t, &c)
	h := handler(t, c)
	if err := h.Run(context.Background(), x, commit(t, h, &x)); err != nil {
		t.Fatal(err)
	}
	// An otherwise valid artifact reference absent from this exact stage is not
	// authorization to read an arbitrary external feature pool.
	f.Source.Input = c.Recipe.Recovery
	pin(t, &c)
	if _, err := alignment.NewStage(c); err == nil {
		t.Fatal("unadmitted external input accepted")
	}
}

func TestMemoryAdmissionCannotBeBypassedByRepinningPlacement(t *testing.T) {
	c, _ := fixture(t, 1)
	c.Placement.MemoryBytes = 1
	c.Protocol.PlacementSHA256 = digest(c.Placement)
	pin(t, &c)
	if _, err := alignment.NewStage(c); err == nil {
		t.Fatal("one byte admitted for the feature solve")
	}
}

func TestZeroValueStageRefusesWithoutWaiting(t *testing.T) {
	var h alignment.Stage
	if err := h.Run(context.Background(), pipeline.StageContext{}, noCommit); !errors.Is(err, alignment.ErrContract) {
		t.Fatal(err)
	}
	if err := h.Reconcile(context.Background(), pipeline.StageContext{}, noCommit); !errors.Is(err, alignment.ErrContract) {
		t.Fatal(err)
	}
}

func TestOutputByteBoundPreventsPublishingOversizedProjection(t *testing.T) {
	c, x := fixture(t, 1)
	c.Protocol.Limits.MaxArtifactBytes = 1
	pin(t, &c)
	if err := handler(t, c).Run(context.Background(), x, func(context.Context, pipeline.StageResult) error { t.Fatal("oversized artifact reported"); return nil }); err == nil {
		t.Fatal("output byte bound ignored")
	}
	files, err := filepath.Glob(filepath.Join(x.ArtifactDirectory, "alignment-*.json"))
	if err != nil || len(files) != 0 {
		t.Fatal("oversized evidence was published", files, err)
	}
}
