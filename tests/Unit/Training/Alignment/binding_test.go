package alignment_test

import (
	"bytes"
	"context"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/alignment"
	"github.com/tayi-ai/arandu-llama/training/pipeline"
)

func bindingFixture(t *testing.T) (alignment.Config, pipeline.StageContext) {
	t.Helper()
	config, c := fixture(t, 1)
	config.Protocol.Version = 2
	config.Protocol.Limits.MaxTotalBytes *= 2
	for i := range config.Protocol.Fits {
		old := config.Protocol.Fits[i].Source
		config.Protocol.Fits[i].Source = alignment.Source{Binding: pipeline.ArtifactBinding{Kind: pipeline.BindingPredecessor, Predecessor: pipeline.PredecessorOutput{StageID: old.StageID, Path: old.Artifact.Path, Schema: "alignment-samples-v1", MaxBytes: config.Protocol.Limits.MaxSourceBytes}}}
	}
	pin(t, &config)
	return config, c
}

func TestBindingV2ConstructsBeforeOutputsFitsRealMatrixAndConsumesProjection(t *testing.T) {
	config, c := bindingFixture(t)
	oldCompleted := c.Completed
	c.Completed = nil
	source := filepath.Join(c.ArtifactDirectory, "samples-0.json")
	body, err := os.ReadFile(source)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.Remove(source); err != nil {
		t.Fatal(err)
	}
	h := handler(t, config)
	if err = h.Run(context.Background(), c, func(context.Context, pipeline.StageResult) error { t.Fatal("missing source committed"); return nil }); err == nil {
		t.Fatal("missing predecessor accepted")
	}
	if err = os.WriteFile(source, body, 0600); err != nil {
		t.Fatal(err)
	}
	c.Completed = oldCompleted
	if err = h.Run(context.Background(), c, commit(t, h, &c)); err != nil {
		t.Fatal(err)
	}
	if len(c.Previous.Result.Artifacts) != 2 || !c.Previous.Result.Complete {
		t.Fatal("intent and projection missing")
	}
	_, e := evidence(t, c, 1)
	if e.Version != 2 || e.Binding.ReceiptSHA256 != c.Parent.SHA256 || e.Intent != c.Previous.Result.Artifacts[0] {
		t.Fatal("bindings not retained", e)
	}
	want := [][]float64{{4.0 / 3, 2}, {-2.0 / 3, 8.0 / 3}}
	for row := range want {
		for col := range want[row] {
			if math.Abs(e.Projection.Coefficients[row][col]-want[row][col]) > 1e-12 {
				t.Fatal("ridge coefficients differ", e.Projection.Coefficients)
			}
		}
	}
	// The consumer is constructed from a selector, never a future projection SHA.
	consumer := c
	consumer.Stage = c.Execution.Recipe.Stages[3]
	consumer.Parent = c.Previous
	consumer.Previous = nil
	consumer.Completed = append(append([]pipeline.StageReceipt(nil), c.Completed...), *c.Previous)
	selector := pipeline.ArtifactBinding{Kind: pipeline.BindingPredecessor, Predecessor: pipeline.PredecessorOutput{StageID: c.Stage.ID, Path: c.Previous.Result.Artifacts[1].Path, Schema: alignment.EvidenceSchema, MaxBytes: config.Protocol.Limits.MaxArtifactBytes}}
	bound, err := pipeline.ResolveArtifact(context.Background(), consumer, config.SourceDirectory, selector, config.Protocol.Limits.MaxArtifactBytes)
	if err != nil {
		t.Fatal(err)
	}
	fit := config.Protocol.Fits[0]
	projection, err := alignment.ReadBoundProjection(context.Background(), bound, fit.MatchingPlan, fit.MatchingPlanSHA256, config.Protocol.Limits.Projection, config.Protocol.Limits.MaxArtifactBytes)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(projection, e.Projection) {
		t.Fatal("consumer changed projection")
	}
	mapped, err := projection.Apply([]float64{2, 1}, fit.MatchingPlanSHA256, config.Protocol.Limits.Projection)
	if err != nil || len(mapped) != 2 || math.Abs(mapped[0]-2) > 1e-12 || math.Abs(mapped[1]-20.0/3) > 1e-12 {
		t.Fatal("consumer projection not applied", mapped, err)
	}
	changed := fit.MatchingPlan
	changed.Source.Layer++
	if _, err = alignment.ReadBoundProjection(context.Background(), bound, changed, fit.MatchingPlanSHA256, config.Protocol.Limits.Projection, config.Protocol.Limits.MaxArtifactBytes); err == nil {
		t.Fatal("changed matching plan accepted")
	}
	if _, err = alignment.ReadBoundProjection(context.Background(), bound, fit.MatchingPlan, fit.MatchingPlanSHA256, config.Protocol.Limits.Projection, bound.Binding.Artifact.Bytes-1); err == nil {
		t.Fatal("oversized output accepted")
	}
	bad := bound
	bad.Binding.Identity.RunID = "other-run"
	if _, err = alignment.ReadBoundProjection(context.Background(), bad, fit.MatchingPlan, fit.MatchingPlanSHA256, config.Protocol.Limits.Projection, config.Protocol.Limits.MaxArtifactBytes); err == nil {
		t.Fatal("wrong producer identity accepted")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err = alignment.ReadBoundProjection(ctx, bound, fit.MatchingPlan, fit.MatchingPlanSHA256, config.Protocol.Limits.Projection, config.Protocol.Limits.MaxArtifactBytes); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if err = os.WriteFile(bound.Path, []byte("corrupt"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err = alignment.ReadBoundProjection(context.Background(), bound, fit.MatchingPlan, fit.MatchingPlanSHA256, config.Protocol.Limits.Projection, config.Protocol.Limits.MaxArtifactBytes); err == nil {
		t.Fatal("corrupt coefficients accepted")
	}
}

func TestBindingV2RecoversLostCommitAndRefusesReboundSource(t *testing.T) {
	config, c := bindingFixture(t)
	h := handler(t, config)
	lost := errors.New("lost app receipt")
	if err := h.Run(context.Background(), c, func(context.Context, pipeline.StageResult) error { return lost }); !errors.Is(err, lost) {
		t.Fatal(err)
	}
	path, e := evidence(t, c, 1)
	before, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	intentPath := filepath.Join(c.ArtifactDirectory, e.Intent.Path)
	intentBefore, err := os.Stat(intentPath)
	if err != nil {
		t.Fatal(err)
	}
	h = handler(t, config)
	c.Execution.Generation++
	if err = h.Reconcile(context.Background(), c, commit(t, h, &c)); err != nil {
		t.Fatal(err)
	}
	after, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	intentAfter, err := os.Stat(intentPath)
	if err != nil {
		t.Fatal(err)
	}
	if !os.SameFile(before, after) || !before.ModTime().Equal(after.ModTime()) || !os.SameFile(intentBefore, intentAfter) {
		t.Fatal("reconciliation refitted")
	}
	c.Completed[0].Identity.Generation++
	c.Completed[0] = seal(c.Completed[0])
	if err = h.Verify(context.Background(), c, *c.Previous); err == nil {
		t.Fatal("different source receipt accepted")
	}
}

func TestBindingV2QualificationFailurePersistsIntentAndNeverRepeatsFit(t *testing.T) {
	config, c := bindingFixture(t)
	config.Protocol.Fits[0].Qualification = alignment.Qualification{MaxFitMSE: 0, MaxHeldoutMSE: 0}
	pin(t, &config)
	h := handler(t, config)
	if err := h.Run(context.Background(), c, noCommit); err == nil {
		t.Fatal("nonzero ridge error passed exact threshold")
	}
	entries, err := os.ReadDir(c.ArtifactDirectory)
	if err != nil {
		t.Fatal(err)
	}
	count := 0
	for _, e := range entries {
		if e.Name() == "alignment-"+c.Stage.ID+"-000001-intent.json" {
			count++
		}
	}
	if count != 1 {
		t.Fatal("attempt not persisted before numerical fit", entries)
	}
	if err := handler(t, config).Run(context.Background(), c, noCommit); !errors.Is(err, alignment.ErrAttempt) {
		t.Fatal("uncertain/rejected fit retried", err)
	}
}

func TestBindingV2DerivedSamplesKeepUnderlyingDatasetProvenance(t *testing.T) {
	config, c := bindingFixture(t)
	a := c.Parent.Result.Artifacts[0]
	body, err := os.ReadFile(filepath.Join(c.ArtifactDirectory, a.Path))
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(config.SourceDirectory, a.Path), body, 0600); err != nil {
		t.Fatal(err)
	}
	config.Protocol.Fits[0].Source = alignment.Source{Binding: pipeline.ArtifactBinding{Kind: pipeline.BindingDerivedInput, Input: config.Recipe.Training, Artifact: a}}
	if a.SHA256 == config.Recipe.Training.SHA256 {
		t.Fatal("fixture does not distinguish derived bytes")
	}
	pin(t, &config)
	h := handler(t, config)
	if err = h.Run(context.Background(), c, commit(t, h, &c)); err != nil {
		t.Fatal(err)
	}
	_, e := evidence(t, c, 1)
	if e.Binding.StageID != "" || e.Binding.Artifact != a {
		t.Fatal("derived origin changed")
	}
	changed := config
	changed.Protocol.Fits = append([]alignment.Fit(nil), config.Protocol.Fits...)
	changed.Protocol.Fits[0].Source.Binding.Input = config.Recipe.Calibration
	pin(t, &changed)
	if _, err = alignment.NewStage(changed); err == nil {
		t.Fatal("mismatched sample dataset admitted")
	}
}

func TestBindingV2CancelledBeforeIntentAndCorruptIntentRefused(t *testing.T) {
	config, c := bindingFixture(t)
	h := handler(t, config)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := h.Run(ctx, c, noCommit); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(c.ArtifactDirectory)
	if err != nil || len(entries) != 1 {
		t.Fatal("cancelled stage persisted attempt", err)
	}
	if err = h.Run(context.Background(), c, commit(t, h, &c)); err != nil {
		t.Fatal(err)
	}
	_, e := evidence(t, c, 1)
	path := filepath.Join(c.ArtifactDirectory, e.Intent.Path)
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	body = bytes.Replace(body, []byte(e.Binding.ReceiptSHA256), []byte(ref("different").SHA256), 1)
	if err = os.WriteFile(path, body, 0600); err != nil {
		t.Fatal(err)
	}
	if err = h.Verify(context.Background(), c, *c.Previous); err == nil {
		t.Fatal("corrupt durable binding accepted")
	}
}
