//go:build libtorch && cgo

package decoder

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/alignment"
	"github.com/tayi-ai/arandu-llama/training/fusioncache"
	"github.com/tayi-ai/arandu-llama/training/pipeline"
)

func sessionRepinV2(t *testing.T, c *ConstrainedFactoryConfig) {
	t.Helper()
	var err error
	c.Recipe.Protocol.RecipeSHA256, err = c.Recipe.Recipe.Digest()
	if err != nil {
		t.Fatal(err)
	}
	c.Recipe.ProtocolSHA256, err = c.Recipe.Protocol.Digest()
	if err != nil {
		t.Fatal(err)
	}
	c.SHA256, err = c.Digest()
	if err != nil {
		t.Fatal(err)
	}
}

func sessionBoundSource(t *testing.T, c pipeline.StageContext, source pipeline.ConstrainedSource, limit int64) pipeline.ConstrainedSourceFile {
	t.Helper()
	file, err := pipeline.ResolveArtifact(context.Background(), c, c.ArtifactDirectory, source.Binding, limit)
	if err != nil {
		t.Fatal(err)
	}
	return pipeline.ConstrainedSourceFile{Source: source, Path: file.Path, Artifact: file.Binding.Artifact, Binding: file.Binding}
}

func sessionRequestV2(t *testing.T, config ConstrainedFactoryConfig, c pipeline.StageContext, initialSHA string) pipeline.ConstrainedRequest {
	t.Helper()
	r := pipeline.ConstrainedRequest{Context: c, Step: 1}
	for _, source := range config.Recipe.Protocol.Updates[0].Sources {
		r.Sources = append(r.Sources, sessionBoundSource(t, c, source, config.Recipe.Protocol.Limits.MaxSourceBytes))
	}
	r.Initial = pipeline.ConstrainedInitialCheckpoint{File: sessionBoundSource(t, c, pipeline.ConstrainedSource{Binding: config.Recipe.Protocol.InitialSource}, config.Recipe.Protocol.Limits.MaxCheckpointBytes), ParametersSHA256: initialSHA}
	return r
}

// Output paths are known before execution; no checkpoint or projection SHA is
// smuggled into the factory. The initial adapter differs from initialization.
func recoverySessionFixture(t *testing.T) (ConstrainedFactoryConfig, pipeline.StageContext, constrainedAssemblyLoader, **LoadedTextModel, func(), string) {
	t.Helper()
	c, request, loader, last := sessionFixture(t)
	initialSHA := c.Recipe.Initial.ParametersSHA256
	if initialSHA == c.Recipe.Initializer.ExpectedSHA256 {
		t.Fatal("fixture did not change initial parameters")
	}
	initialBody, err := os.ReadFile(c.Recipe.Initial.Path)
	if err != nil {
		t.Fatal(err)
	}
	initialArtifact := pipeline.StageArtifact{Path: "initial.safetensors", SHA256: assemblyHash(initialBody), Bytes: int64(len(initialBody))}
	if err := os.Remove(c.Recipe.Initial.Path); err != nil {
		t.Fatal(err)
	}
	d := RecoveryBatchDocument{Version: 1, Dataset: c.Recipe.Recipe.Recovery, Student: c.Recipe.Admission.Student, Aggregation: RecoveryTokenMean, SupervisedTokens: 3,
		Examples: []pipeline.Example{{ID: "long", DatasetDigest: c.Recipe.Recipe.Recovery.SHA256, Tokens: []int64{1, 2, 3}, PromptTokens: 1}, {ID: "short", DatasetDigest: c.Recipe.Recipe.Recovery.SHA256, Tokens: []int64{1, 4}, PromptTokens: 1}}}
	limits := RecoveryLimits{MaxBytes: 1 << 20, MaxExamples: 4, MaxTotalTokens: 12, MaxTokens: 3, WorkingBytes: 4 << 20}
	targets, err := d.TargetsDigest(limits)
	if err != nil {
		t.Fatal(err)
	}
	body, err := json.Marshal(d)
	if err != nil {
		t.Fatal(err)
	}
	artifact := pipeline.StageArtifact{Path: "recovery.json", SHA256: assemblyHash(body), Bytes: int64(len(body))}
	if artifact.SHA256 == d.Dataset.SHA256 {
		t.Fatal("derived artifact must differ from underlying dataset")
	}
	c.Recipe.Signals = nil
	c.Recipe.Recovery = []ConstrainedRecovery{{SourceIndex: 0, Expectation: RecoveryExpectation{Dataset: d.Dataset, Student: d.Student, Aggregation: d.Aggregation, TargetsSHA256: targets, Examples: 2, SupervisedTokens: 3}, Limits: limits}}
	c.Recipe.Initial.Path = ""
	c.Recipe.Initial.FileSHA256 = ""
	c.Recipe.Initial.ParametersSHA256 = ""
	c.Recipe.Protocol.Version = 2
	c.Recipe.Protocol.StageID = "stage-4"
	c.Recipe.Protocol.InitialParametersSHA256 = ""
	c.Recipe.Protocol.InitialSource = pipeline.ArtifactBinding{Kind: pipeline.BindingPredecessor, Predecessor: pipeline.PredecessorOutput{StageID: "stage-3", Path: initialArtifact.Path, Schema: "adapter-safetensors-v1", MaxBytes: 1 << 20}}
	c.Recipe.Protocol.Updates[0].Sources = []pipeline.ConstrainedSource{{Binding: pipeline.ArtifactBinding{Kind: pipeline.BindingDerivedInput, Input: d.Dataset, Artifact: artifact}}}
	sessionRepinV2(t, &c)
	context := request.Context
	context.Stage = c.Recipe.Recipe.Stages[4]
	context.Execution.Recipe = c.Recipe.Recipe
	parent := sessionReceipt(pipeline.StageReceipt{Version: 1, Identity: context.Parent.Identity, StageID: "stage-3", Result: pipeline.StageResult{Steps: 1, Complete: true, Artifacts: []pipeline.StageArtifact{initialArtifact}}})
	context.Parent = &parent
	context.Completed = []pipeline.StageReceipt{parent}
	publish := func() {
		if err := os.WriteFile(filepath.Join(context.ArtifactDirectory, initialArtifact.Path), initialBody, 0600); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(context.ArtifactDirectory, artifact.Path), body, 0600); err != nil {
			t.Fatal(err)
		}
	}
	return c, context, loader, last, publish, initialSHA
}

func TestNativeConstrainedFactoryV2RecoveryLoadsProducedCheckpointAndCommits(t *testing.T) {
	c, stage, loader, last, publish, initialSHA := recoverySessionFixture(t)
	factory, err := newConstrainedFactory(c, loader)
	if err != nil {
		t.Fatal(err)
	}
	handler, err := pipeline.NewConstrainedStage(pipeline.ConstrainedConfig{Recipe: c.Recipe.Recipe, Placement: c.Recipe.Placement, Protocol: c.Recipe.Protocol, ProtocolSHA256: c.Recipe.ProtocolSHA256, SourceDirectory: stage.ArtifactDirectory, Factory: factory})
	if err != nil {
		t.Fatal(err)
	}
	// Both factory and stage exist before predecessor bytes have been produced.
	if _, err := os.Stat(filepath.Join(stage.ArtifactDirectory, "initial.safetensors")); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("initial output already exists", err)
	}
	publish()
	var result pipeline.StageResult
	if err := handler.Run(context.Background(), stage, func(_ context.Context, r pipeline.StageResult) error { result = r; return nil }); err != nil {
		t.Fatal(err)
	}
	if !result.Complete || result.Steps != 1 || (*last).Model != nil {
		t.Fatal("recovery did not commit and close")
	}
	body, err := os.ReadFile(filepath.Join(stage.ArtifactDirectory, result.Artifacts[1].Path))
	if err != nil {
		t.Fatal(err)
	}
	var evidence pipeline.ConstrainedEvidence
	if err := json.Unmarshal(body, &evidence); err != nil {
		t.Fatal(err)
	}
	receipt := evidence.Receipt
	if evidence.PriorParametersSHA256 != initialSHA || evidence.ParametersSHA256 == initialSHA || !receipt.Evaluation.Accepted || receipt.Objective.FeatureLoss != 0 || receipt.CandidateObjective.Loss >= receipt.Objective.Loss {
		t.Fatal("causal recovery did not use produced parameters and decrease loss", evidence)
	}
	if evidence.Bindings == nil || evidence.Bindings.InitialParametersSHA256 != initialSHA || evidence.Bindings.Sources[0].Artifact.SHA256 == c.Recipe.Recipe.Recovery.SHA256 {
		t.Fatal("initial and derived dataset provenance missing")
	}
}

func TestNativeConstrainedFactoryV2RecoveryRefusesUnqualifiedInputsBeforeLoad(t *testing.T) {
	for _, mode := range []string{"bad-targets", "bad-dataset", "corrupt-batch", "corrupt-initial", "source-binding", "initial-binding"} {
		t.Run(mode, func(t *testing.T) {
			c, stage, loader, _, publish, initialSHA := recoverySessionFixture(t)
			if mode == "bad-targets" {
				c.Recipe.Recovery[0].Expectation.TargetsSHA256 = assemblyHash([]byte("wrong targets"))
				sessionRepinV2(t, &c)
			}
			if mode == "bad-dataset" {
				c.Recipe.Recovery[0].Expectation.Dataset = c.Recipe.Recipe.Training
				if _, err := c.Digest(); err == nil {
					t.Fatal("training admitted as recovery")
				}
				return
			}
			calls := 0
			factory, err := newConstrainedFactory(c, func(ctx context.Context, p *AssemblyPlan, s ShardProvider, a *InitialAdapter) (*LoadedTextModel, error) {
				calls++
				return loader(ctx, p, s, a)
			})
			if err != nil {
				t.Fatal(err)
			}
			publish()
			request := sessionRequestV2(t, c, stage, initialSHA)
			switch mode {
			case "corrupt-batch":
				if err := os.WriteFile(request.Sources[0].Path, []byte("corrupt"), 0600); err != nil {
					t.Fatal(err)
				}
			case "corrupt-initial":
				if err := os.WriteFile(request.Initial.File.Path, []byte("corrupt"), 0600); err != nil {
					t.Fatal(err)
				}
			case "source-binding":
				request.Sources[0].Binding.Identity.RunID = "other"
			case "initial-binding":
				request.Initial.File.Binding.Identity.RunID = "other"
			}
			if session, err := factory(context.Background(), request); err == nil {
				session.Close()
				t.Fatal("unqualified input admitted")
			}
			if calls != 0 {
				t.Fatal("bad input reached native loader", calls)
			}
		})
	}
}

func TestNativeConstrainedFactoryV2RecoveryStillRefusesVariant(t *testing.T) {
	c, _, _, _, _, _ := recoverySessionFixture(t)
	c.Recipe.Protocol.StageID = "stage-8"
	c.Recipe.Protocol.InitialSource.Predecessor.StageID = "stage-7"
	c.Recipe.ProtocolSHA256, _ = c.Recipe.Protocol.Digest()
	if _, err := c.Digest(); err == nil {
		t.Fatal("quantized recovery falsely admitted")
	}
}

func fusionSessionV2Fixture(t *testing.T) (ConstrainedFactoryConfig, pipeline.StageContext, constrainedAssemblyLoader, **LoadedTextModel, func() pipeline.StageContext, string) {
	t.Helper()
	c, request, loader, last := sessionFixture(t)
	initialSHA := c.Recipe.Initial.ParametersSHA256
	initialBody, err := os.ReadFile(c.Recipe.Initial.Path)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(c.Recipe.Initial.Path); err != nil {
		t.Fatal(err)
	}
	initialArtifact := pipeline.StageArtifact{Path: "initial.safetensors", SHA256: assemblyHash(initialBody), Bytes: int64(len(initialBody))}
	c.Recipe.Initial.Path = ""
	c.Recipe.Initial.FileSHA256 = ""
	c.Recipe.Initial.ParametersSHA256 = ""
	c.Recipe.Protocol.Version = 2
	c.Recipe.Protocol.InitialParametersSHA256 = ""
	c.Recipe.Protocol.InitialSource = pipeline.ArtifactBinding{Kind: pipeline.BindingPredecessor, Predecessor: pipeline.PredecessorOutput{StageID: "stage-0", Path: initialArtifact.Path, Schema: "adapter-safetensors-v1", MaxBytes: 1 << 20}}
	c.Recipe.Recipe.Stages[2].Inputs = append(c.Recipe.Recipe.Stages[2].Inputs, c.Recipe.Recipe.Student, c.Recipe.Recipe.Teachers[0])
	projection := c.Recipe.Signals[0].Caches[0].Projections[0]
	plan := projection.Plan
	for i := range plan.Fit {
		plan.Fit[i].DatasetSHA256 = c.Recipe.Recipe.Training.SHA256
		plan.Fit[i].Role = "train"
	}
	for i := range plan.Heldout {
		plan.Heldout[i].DatasetSHA256 = c.Recipe.Recipe.Training.SHA256
		plan.Heldout[i].Role = "train"
	}
	planSHA, _ := fusioncache.Digest(plan)
	// Targets intentionally differ from the old embedded projection so the
	// objective can prove that newly fitted coefficients were consumed.
	samples := alignment.Samples{Version: 1, MatchingPlanSHA256: planSHA, SourceModel: plan.SourceModel, TargetModel: plan.TargetModel, TokenMappingSHA256: plan.TokenMappingSHA256, Source: plan.Source, Target: plan.Target,
		Fit:     []fusioncache.FeatureSample{{Identity: plan.Fit[0], Source: []float64{1, 0}, Target: []float64{.8, .7, .6, .5}}, {Identity: plan.Fit[1], Source: []float64{0, 1}, Target: []float64{.5, .6, .7, .8}}},
		Heldout: []fusioncache.FeatureSample{{Identity: plan.Heldout[0], Source: []float64{1, 1}, Target: []float64{1.3, 1.3, 1.3, 1.3}}}}
	body, _ := json.Marshal(samples)
	body = append(body, '\n')
	sampleArtifact := pipeline.StageArtifact{Path: "samples.json", SHA256: assemblyHash(body), Bytes: int64(len(body))}
	cacheSource := request.Sources[0].Source.Artifact
	c.Recipe.Protocol.Updates[0].Sources = []pipeline.ConstrainedSource{
		{Binding: pipeline.ArtifactBinding{Kind: pipeline.BindingPredecessor, Predecessor: pipeline.PredecessorOutput{StageID: "stage-1", Path: cacheSource.Path, Schema: "teacher-cache-v1", MaxBytes: 1 << 20}}},
		{Binding: pipeline.ArtifactBinding{Kind: pipeline.BindingPredecessor, Predecessor: pipeline.PredecessorOutput{StageID: "stage-2", Path: "alignment-stage-2-000001.json", Schema: alignment.EvidenceSchema, MaxBytes: 1 << 20}}},
	}
	c.Recipe.Protocol.Limits.MaxSourceBytes = 2 << 20
	c.Recipe.Signals[0].Caches[0].Projections = nil
	c.Recipe.Signals[0].Caches[0].BoundProjections = []ConstrainedProjection{{SourceIndex: 1, Plan: plan, PlanSHA256: planSHA, Weight: projection.Weight, Limits: c.Recipe.Signals[0].Limits.Projection, MaxBytes: 1 << 20}}
	sessionRepinV2(t, &c)
	stage := request.Context
	stage.Execution.Recipe = c.Recipe.Recipe
	stage.Stage = c.Recipe.Recipe.Stages[3]
	id := stage.Parent.Identity
	id.RecipeSHA256 = c.Recipe.Protocol.RecipeSHA256
	teacher := sessionReceipt(pipeline.StageReceipt{Version: 1, Identity: id, StageID: "stage-1", Result: pipeline.StageResult{Steps: 1, Complete: true, Artifacts: []pipeline.StageArtifact{cacheSource}}})
	initial := sessionReceipt(pipeline.StageReceipt{Version: 1, Identity: id, StageID: "stage-0", Result: pipeline.StageResult{Steps: 1, Complete: true, Artifacts: []pipeline.StageArtifact{initialArtifact}}})
	stage.Completed = []pipeline.StageReceipt{initial, teacher}
	stage.Parent = nil
	ap := alignment.Protocol{Version: 2, RecipeSHA256: c.Recipe.Protocol.RecipeSHA256, PlacementSHA256: c.Recipe.Protocol.PlacementSHA256, QualificationSHA256: c.Recipe.Placement.QualificationSHA256, StageID: "stage-2",
		Fits:   []alignment.Fit{{MatchingPlan: plan, MatchingPlanSHA256: planSHA, Source: alignment.Source{Binding: pipeline.ArtifactBinding{Kind: pipeline.BindingDerivedInput, Input: c.Recipe.Recipe.Training, Artifact: sampleArtifact}}, Qualification: alignment.Qualification{MaxFitMSE: 1, MaxHeldoutMSE: 1}}},
		Limits: alignment.Limits{MaxProtocolBytes: 1 << 16, MaxSourceBytes: 1 << 16, MaxArtifactBytes: 1 << 16, MaxTotalBytes: 2 << 16, MaxFits: 1, MaxSolveWork: 10000, Projection: c.Recipe.Signals[0].Limits.Projection}}
	apSHA, err := ap.Digest()
	if err != nil {
		t.Fatal("alignment protocol", err)
	}
	ah, err := alignment.NewStage(alignment.Config{Recipe: c.Recipe.Recipe, Placement: c.Recipe.Placement, Protocol: ap, ProtocolSHA256: apSHA, SourceDirectory: stage.ArtifactDirectory})
	if err != nil {
		t.Fatal("alignment stage", err)
	}
	publish := func() pipeline.StageContext {
		if err := os.WriteFile(filepath.Join(stage.ArtifactDirectory, initialArtifact.Path), initialBody, 0600); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(stage.ArtifactDirectory, sampleArtifact.Path), body, 0600); err != nil {
			t.Fatal(err)
		}
		ac := stage
		ac.Stage = c.Recipe.Recipe.Stages[2]
		ac.Parent = &teacher
		var result pipeline.StageResult
		if err := ah.Run(context.Background(), ac, func(_ context.Context, r pipeline.StageResult) error { result = r; return nil }); err != nil {
			t.Fatal(err)
		}
		parent := sessionReceipt(pipeline.StageReceipt{Version: 1, Identity: id, StageID: "stage-2", Result: result})
		if err := ah.Verify(context.Background(), ac, parent); err != nil {
			t.Fatal(err)
		}
		stage.Parent = &parent
		stage.Completed = append(stage.Completed, parent)
		return stage
	}
	return c, stage, loader, last, publish, initialSHA
}

func TestNativeConstrainedFactoryV2ConsumesNewlyFittedProjectionAndInitial(t *testing.T) {
	c, stage, loader, last, publish, initialSHA := fusionSessionV2Fixture(t)
	factory, err := newConstrainedFactory(c, loader)
	if err != nil {
		t.Fatal(err)
	}
	handler, err := pipeline.NewConstrainedStage(pipeline.ConstrainedConfig{Recipe: c.Recipe.Recipe, Placement: c.Recipe.Placement, Protocol: c.Recipe.Protocol, ProtocolSHA256: c.Recipe.ProtocolSHA256, SourceDirectory: stage.ArtifactDirectory, Factory: factory})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(stage.ArtifactDirectory, "alignment-stage-2-000001.json")); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("projection already exists", err)
	}
	stage = publish()
	request := sessionRequestV2(t, c, stage, initialSHA)
	actualSignals, err := c.signals(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	projection := c.Recipe.Signals[0].Caches[0].BoundProjections[0]
	fitted, _, err := fusioncache.FitProjection(projection.Plan, projection.PlanSHA256,
		[]fusioncache.FeatureSample{{Identity: projection.Plan.Fit[0], Source: []float64{1, 0}, Target: []float64{.8, .7, .6, .5}}, {Identity: projection.Plan.Fit[1], Source: []float64{0, 1}, Target: []float64{.5, .6, .7, .8}}},
		[]fusioncache.FeatureSample{{Identity: projection.Plan.Heldout[0], Source: []float64{1, 1}, Target: []float64{1.3, 1.3, 1.3, 1.3}}}, projection.Limits)
	if err != nil {
		t.Fatal(err)
	}
	want, err := fitted.Apply([]float64{.1, .2}, projection.PlanSHA256, projection.Limits)
	wantFP32 := make([]float32, len(want))
	for i, value := range want {
		wantFP32[i] = float32(value)
	}
	if err != nil || len(actualSignals.Features()) != 2 || actualSignals.Features()[0].ProjectionSHA256 != fitted.SHA256 || !reflect.DeepEqual(actualSignals.Features()[0].Values, wantFP32) {
		t.Fatal("new fitted projection was not consumed", err)
	}
	var result pipeline.StageResult
	if err := handler.Run(context.Background(), stage, func(_ context.Context, r pipeline.StageResult) error { result = r; return nil }); err != nil {
		t.Fatal(err)
	}
	if !result.Complete || (*last).Model != nil {
		t.Fatal("fusion did not commit and close")
	}
	body, err := os.ReadFile(filepath.Join(stage.ArtifactDirectory, result.Artifacts[1].Path))
	if err != nil {
		t.Fatal(err)
	}
	var evidence pipeline.ConstrainedEvidence
	if err := json.Unmarshal(body, &evidence); err != nil {
		t.Fatal(err)
	}
	if evidence.PriorParametersSHA256 != initialSHA || evidence.ParametersSHA256 == initialSHA || !evidence.Receipt.Evaluation.Accepted || evidence.Receipt.Objective.FeatureLoss <= 0 || evidence.Bindings == nil || len(evidence.Bindings.Sources) != 2 {
		t.Fatal("bound fusion evidence absent")
	}
}

func TestNativeConstrainedFactoryV2ProjectionRefusesBeforeLoad(t *testing.T) {
	for _, mode := range []string{"corrupt", "plan", "source-phase", "source-schema"} {
		t.Run(mode, func(t *testing.T) {
			c, _, loader, _, publish, initialSHA := fusionSessionV2Fixture(t)
			if mode == "source-phase" {
				c.Recipe.Protocol.Updates[0].Sources[1].Binding.Predecessor.StageID = "stage-1"
			}
			if mode == "source-schema" {
				c.Recipe.Protocol.Updates[0].Sources[1].Binding.Predecessor.Schema = "other"
			}
			if mode == "source-phase" || mode == "source-schema" {
				c.Recipe.ProtocolSHA256, _ = c.Recipe.Protocol.Digest()
				if _, err := c.Digest(); err == nil {
					t.Fatal("non-alignment output admitted")
				}
				return
			}
			if mode == "plan" {
				p := &c.Recipe.Signals[0].Caches[0].BoundProjections[0]
				p.Plan.Ridge = 2
				p.PlanSHA256, _ = fusioncache.Digest(p.Plan)
				sessionRepinV2(t, &c)
			}
			calls := 0
			factory, err := newConstrainedFactory(c, func(ctx context.Context, p *AssemblyPlan, s ShardProvider, a *InitialAdapter) (*LoadedTextModel, error) {
				calls++
				return loader(ctx, p, s, a)
			})
			if err != nil {
				t.Fatal(err)
			}
			stage := publish()
			request := sessionRequestV2(t, c, stage, initialSHA)
			if mode == "corrupt" {
				if err := os.WriteFile(request.Sources[1].Path, []byte("corrupt"), 0600); err != nil {
					t.Fatal(err)
				}
			}
			if session, err := factory(context.Background(), request); err == nil {
				session.Close()
				t.Fatal("bad projection admitted")
			}
			if calls != 0 {
				t.Fatal("bad projection reached native loader")
			}
		})
	}
}

func TestNativeConstrainedFactoryV2RecoveryClosesFailedRestoreAndReleasesGate(t *testing.T) {
	c, stage, loader, last, publish, initialSHA := recoverySessionFixture(t)
	factory, err := newConstrainedFactory(c, loader)
	if err != nil {
		t.Fatal(err)
	}
	publish()
	request := sessionRequestV2(t, c, stage, initialSHA)
	request.Initial.ParametersSHA256 = assemblyHash([]byte("different ordered parameters"))
	if session, err := factory(context.Background(), request); err == nil {
		session.Close()
		t.Fatal("bad parameter digest admitted")
	}
	if *last == nil || (*last).Model != nil {
		t.Fatal("failed restore retained native tensors")
	}
	request.Initial.ParametersSHA256 = initialSHA
	session, err := factory(context.Background(), request)
	if err != nil {
		t.Fatal("failed restore did not release session gate", err)
	}
	if _, ok := session.Model.(*RecoveryBackend); !ok {
		t.Fatal("recovery did not construct causal backend")
	}
	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
	if (*last).Model != nil {
		t.Fatal("native state retained after successful close")
	}
}
