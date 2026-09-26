package pipeline_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"reflect"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/pipeline"
)

func dispatchDigest(v any) string {
	body, _ := json.Marshal(v)
	sum := sha256.Sum256(body)
	return hex.EncodeToString(sum[:])
}

func dispatchSeal(r pipeline.StageReceipt) pipeline.StageReceipt {
	r.SHA256 = ""
	r.SHA256 = dispatchDigest(r)
	return r
}

func dispatchFixture(t *testing.T) (*pipeline.StageDispatch, pipeline.StageContext, pipeline.StageReceipt, map[string]*fixturePhase, map[string]pipeline.StageHandler) {
	t.Helper()
	_, x, _ := durableFixture(t.TempDir())
	x.Generation = 2
	stage := x.Recipe.Stages[8]
	stage.Inputs = append([]pipeline.ArtifactRef(nil), stage.Inputs...)
	c := pipeline.StageContext{Execution: x, Stage: stage, ArtifactDirectory: t.TempDir()}
	implementations := map[string]*fixturePhase{}
	wiring := map[string]pipeline.StageHandler{}
	for _, s := range x.Recipe.Stages {
		if s.Phase == pipeline.PhaseVariantRecovery {
			h := &fixturePhase{calls: map[string]int{}}
			implementations[s.ID], wiring[s.ID] = h, h
		}
	}
	d, err := pipeline.NewStageDispatch(pipeline.PhaseVariantRecovery, wiring)
	if err != nil {
		t.Fatal(err)
	}
	recipeSHA, _ := x.Recipe.Digest()
	r := dispatchSeal(pipeline.StageReceipt{Version: 1, StageID: stage.ID, Identity: pipeline.ReceiptIdentity{RunID: x.RunID, TenantID: x.TenantID, RecipeSHA256: recipeSHA, TargetSHA256: x.TargetSHA256, PlacementSHA256: dispatchDigest(x.Placement), Generation: 1}, Result: pipeline.StageResult{Steps: 1, Complete: true}})
	return d, c, r, implementations, wiring
}

func TestStageDispatchRunsDistinctHandlersInCompleteDurableRecipe(t *testing.T) {
	config, execution, _ := durableFixture(t.TempDir())
	groups := map[pipeline.Phase]map[string]pipeline.StageHandler{}
	implementations := map[string]*fixturePhase{}
	admitted := map[string]bool{}
	for _, stage := range execution.Recipe.Stages {
		exact := stage
		h := &fixturePhase{calls: map[string]int{}}
		h.admit = func(_ context.Context, _ pipeline.Recipe, got pipeline.Stage, _ pipeline.Placement) error {
			if !reflect.DeepEqual(got, exact) {
				t.Errorf("wrong variant admitted: %+v want %+v", got, exact)
			}
			admitted[exact.ID] = true
			return nil
		}
		h.run = func(ctx context.Context, c pipeline.StageContext, commit func(context.Context, pipeline.StageResult) error) error {
			if len(admitted) != 13 || c.Stage.ID != exact.ID {
				t.Errorf("effects before complete admission or wrong handler: %s", c.Stage.ID)
			}
			result, err := writeFixtureResult(c, 1, true)
			if err != nil {
				return err
			}
			return commit(ctx, result)
		}
		if groups[stage.Phase] == nil {
			groups[stage.Phase] = map[string]pipeline.StageHandler{}
		}
		groups[stage.Phase][stage.ID], implementations[stage.ID] = h, h
	}
	for phase, handlers := range groups {
		d, err := pipeline.NewStageDispatch(phase, handlers)
		if err != nil {
			t.Fatal(err)
		}
		config.Handlers[phase] = d
		clear(handlers) // Installation map mutation cannot remove admitted handlers.
	}
	runtime := runtimeFixture(t, config)
	if err := runtime.Run(context.Background(), execution, noReport); err != nil {
		t.Fatal(err)
	}
	execution.Generation++
	progress, err := runtime.Reconcile(context.Background(), execution)
	if err != nil || progress.CompletedStages != 13 || progress.CompletedSteps != 13 {
		t.Fatalf("historical receipt replay: %+v %v", progress, err)
	}
	if err := runtime.Run(context.Background(), execution, noReport); err != nil {
		t.Fatal(err)
	}
	for id, h := range implementations {
		if len(h.calls) != 1 || h.calls[id] != 1 {
			t.Fatalf("wrong handler or duplicate work for %s: %v", id, h.calls)
		}
	}
}

func TestStageDispatchMissingOrDeniedFutureVariantPreventsAllRuntimeEffects(t *testing.T) {
	for _, missing := range []bool{false, true} {
		name := "denied"
		if missing {
			name = "missing"
		}
		t.Run(name, func(t *testing.T) {
			config, execution, ordinary := durableFixture(t.TempDir())
			bindings := map[string]pipeline.StageHandler{}
			var variants []*fixturePhase
			denied := errors.New("variant qualification denied")
			for _, stage := range execution.Recipe.Stages {
				if stage.Phase != pipeline.PhaseVariantRecovery {
					continue
				}
				h := &fixturePhase{calls: map[string]int{}}
				if stage.ID == "stage-12" {
					if missing {
						continue
					}
					h.admit = func(context.Context, pipeline.Recipe, pipeline.Stage, pipeline.Placement) error { return denied }
				}
				bindings[stage.ID], variants = h, append(variants, h)
			}
			d, err := pipeline.NewStageDispatch(pipeline.PhaseVariantRecovery, bindings)
			if err != nil {
				t.Fatal(err)
			}
			config.Handlers[pipeline.PhaseVariantRecovery] = d
			err = runtimeFixture(t, config).Run(context.Background(), execution, noReport)
			want := denied
			if missing {
				want = pipeline.ErrStageDispatch
			}
			if !errors.Is(err, want) {
				t.Fatal(err)
			}
			if len(ordinary.calls) != 0 {
				t.Fatal("earlier phases ran before variant admission")
			}
			for _, h := range variants {
				if len(h.calls) != 0 {
					t.Fatal("variant ran before complete admission")
				}
			}
			files, err := os.ReadDir(config.Root)
			if err != nil || len(files) != 0 {
				t.Fatal("failed admission created execution state", files, err)
			}
		})
	}
}

func TestStageDispatchVerifiesExactReceiptAndContextBeforeDelegating(t *testing.T) {
	for _, test := range []struct {
		name   string
		change func(*pipeline.StageContext, *pipeline.StageReceipt)
	}{
		{"stage-id", func(_ *pipeline.StageContext, r *pipeline.StageReceipt) { r.StageID = "stage-10" }},
		{"tenant", func(_ *pipeline.StageContext, r *pipeline.StageReceipt) { r.Identity.TenantID = "other" }},
		{"run", func(_ *pipeline.StageContext, r *pipeline.StageReceipt) { r.Identity.RunID = "other" }},
		{"recipe", func(_ *pipeline.StageContext, r *pipeline.StageReceipt) {
			r.Identity.RecipeSHA256 = ref("other").SHA256
		}},
		{"target", func(_ *pipeline.StageContext, r *pipeline.StageReceipt) {
			r.Identity.TargetSHA256 = ref("other").SHA256
		}},
		{"placement", func(_ *pipeline.StageContext, r *pipeline.StageReceipt) {
			r.Identity.PlacementSHA256 = ref("other").SHA256
		}},
		{"future-generation", func(c *pipeline.StageContext, r *pipeline.StageReceipt) {
			r.Identity.Generation = c.Execution.Generation + 1
		}},
		{"zero-generation", func(_ *pipeline.StageContext, r *pipeline.StageReceipt) { r.Identity.Generation = 0 }},
		{"version", func(_ *pipeline.StageContext, r *pipeline.StageReceipt) { r.Version = 2 }},
		{"zero-steps", func(_ *pipeline.StageContext, r *pipeline.StageReceipt) { r.Result.Steps = 0 }},
		{"step-budget", func(c *pipeline.StageContext, r *pipeline.StageReceipt) { r.Result.Steps = int64(c.Stage.MaxSteps) + 1 }},
		{"stage-definition", func(c *pipeline.StageContext, _ *pipeline.StageReceipt) { c.Stage.MaxSteps++ }},
		{"stage-phase", func(c *pipeline.StageContext, _ *pipeline.StageReceipt) { c.Stage.Phase = pipeline.PhaseQuantize }},
		{"stage-format", func(c *pipeline.StageContext, _ *pipeline.StageReceipt) { c.Stage.Format = "Q4_K_M" }},
		{"stage-input", func(c *pipeline.StageContext, _ *pipeline.StageReceipt) { c.Stage.Inputs[0] = ref("other") }},
		{"unknown-stage", func(c *pipeline.StageContext, _ *pipeline.StageReceipt) { c.Stage.ID = "unknown" }},
		{"directory", func(c *pipeline.StageContext, _ *pipeline.StageReceipt) { c.ArtifactDirectory = "relative" }},
	} {
		t.Run(test.name, func(t *testing.T) {
			d, c, receipt, handlers, _ := dispatchFixture(t)
			called := false
			for _, h := range handlers {
				h.verify = func(context.Context, pipeline.StageContext, pipeline.StageReceipt) error { called = true; return nil }
			}
			test.change(&c, &receipt)
			receipt = dispatchSeal(receipt) // The envelope is validly hashed but contradicts execution.
			if err := d.Verify(context.Background(), c, receipt); !errors.Is(err, pipeline.ErrStageDispatch) || called {
				t.Fatalf("divergent receipt/context forwarded: %v called=%v", err, called)
			}
		})
	}
	d, c, receipt, handlers, _ := dispatchFixture(t)
	called := 0
	handlers[c.Stage.ID].verify = func(_ context.Context, got pipeline.StageContext, r pipeline.StageReceipt) error {
		called++
		if !reflect.DeepEqual(got, c) || !reflect.DeepEqual(r, receipt) {
			t.Fatal("valid envelope changed")
		}
		return nil
	}
	if err := d.Verify(context.Background(), c, receipt); err != nil || called != 1 {
		t.Fatal("historical generation refused", err)
	}
	receipt.SHA256 = ref("bad-hash").SHA256
	if err := d.Verify(context.Background(), c, receipt); !errors.Is(err, pipeline.ErrStageDispatch) || called != 1 {
		t.Fatal("bad hash forwarded", err)
	}
}

func TestStageDispatchSnapshotsMapAndPropagatesSelectedHandlerFailures(t *testing.T) {
	d, c, _, handlers, source := dispatchFixture(t)
	clear(source)
	source[c.Stage.ID] = (*fixturePhase)(nil)
	selected := handlers[c.Stage.ID]
	selected.run = func(context.Context, pipeline.StageContext, func(context.Context, pipeline.StageResult) error) error {
		return io.ErrClosedPipe
	}
	selected.reconcile = selected.run
	if err := d.Run(context.Background(), c, func(context.Context, pipeline.StageResult) error { return nil }); !errors.Is(err, io.ErrClosedPipe) {
		t.Fatal(err)
	}
	if err := d.Reconcile(context.Background(), c, func(context.Context, pipeline.StageResult) error { return nil }); !errors.Is(err, io.ErrClosedPipe) {
		t.Fatal(err)
	}
	if selected.calls[c.Stage.ID] != 1 {
		t.Fatal("wrong handler selected")
	}
	denied := errors.New("future variant denied")
	handlers["stage-12"].admit = func(context.Context, pipeline.Recipe, pipeline.Stage, pipeline.Placement) error { return denied }
	if err := d.Run(context.Background(), c, func(context.Context, pipeline.StageResult) error { return nil }); !errors.Is(err, denied) || selected.calls[c.Stage.ID] != 1 {
		t.Fatal("direct Run bypassed future variant admission", err)
	}
}

type nilDispatchFunction func()

func (nilDispatchFunction) Admit(context.Context, pipeline.Recipe, pipeline.Stage, pipeline.Placement) error {
	panic("nil handler called")
}
func (nilDispatchFunction) Verify(context.Context, pipeline.StageContext, pipeline.StageReceipt) error {
	panic("nil handler called")
}
func (nilDispatchFunction) Reconcile(context.Context, pipeline.StageContext, func(context.Context, pipeline.StageResult) error) error {
	panic("nil handler called")
}
func (nilDispatchFunction) Run(context.Context, pipeline.StageContext, func(context.Context, pipeline.StageResult) error) error {
	panic("nil handler called")
}

func TestStageDispatchRefusesInvalidWiringZeroValueAndCancelledCalls(t *testing.T) {
	for _, h := range []pipeline.StageHandler{nil, (*fixturePhase)(nil), nilDispatchFunction(nil)} {
		if _, err := pipeline.NewStageDispatch(pipeline.PhaseSFT, map[string]pipeline.StageHandler{"stage-0": h}); !errors.Is(err, pipeline.ErrStageDispatch) {
			t.Fatal("nil handler admitted", err)
		}
	}
	if _, err := pipeline.NewStageDispatch("unknown", map[string]pipeline.StageHandler{"stage-0": &fixturePhase{}}); !errors.Is(err, pipeline.ErrStageDispatch) {
		t.Fatal(err)
	}
	if _, err := pipeline.NewStageDispatch(pipeline.PhaseSFT, nil); !errors.Is(err, pipeline.ErrStageDispatch) {
		t.Fatal(err)
	}
	if _, err := pipeline.NewStageDispatch(pipeline.PhaseSFT, map[string]pipeline.StageHandler{"../stage": &fixturePhase{}}); !errors.Is(err, pipeline.ErrStageDispatch) {
		t.Fatal(err)
	}
	d, c, receipt, _, _ := dispatchFixture(t)
	commit := func(context.Context, pipeline.StageResult) error { return nil }
	for _, invalid := range []*pipeline.StageDispatch{nil, {}} {
		if err := invalid.Admit(context.Background(), c.Execution.Recipe, c.Stage, c.Execution.Placement); !errors.Is(err, pipeline.ErrStageDispatch) {
			t.Fatal(err)
		}
		if err := invalid.Verify(context.Background(), c, receipt); !errors.Is(err, pipeline.ErrStageDispatch) {
			t.Fatal(err)
		}
		if err := invalid.Reconcile(context.Background(), c, commit); !errors.Is(err, pipeline.ErrStageDispatch) {
			t.Fatal(err)
		}
		if err := invalid.Run(context.Background(), c, commit); !errors.Is(err, pipeline.ErrStageDispatch) {
			t.Fatal(err)
		}
	}
	if err := d.Run(context.Background(), c, nil); !errors.Is(err, pipeline.ErrStageDispatch) {
		t.Fatal(err)
	}
	if err := d.Reconcile(context.Background(), c, nil); !errors.Is(err, pipeline.ErrStageDispatch) {
		t.Fatal(err)
	}
	if err := d.Run(nil, c, commit); !errors.Is(err, pipeline.ErrStageDispatch) {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := d.Run(ctx, c, commit); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}
