package pipeline_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/pipeline"
)

func masterRecipe() pipeline.Recipe {
	r := recipe()
	r.Version, r.Scope = 2, "master"
	r.ID = "master-delivery-v2"
	r.Calibration = pipeline.ArtifactRef{}
	r.Stages = r.Stages[:6]
	return r
}

// This independent wire definition is copied from b8a854c's Recipe. A roundtrip
// through only the new type would not detect an accidental v1 digest change.
type legacyRecipeWire struct {
	Version     int                    `json:"schema_version"`
	ID          string                 `json:"id"`
	Method      string                 `json:"method"`
	Student     pipeline.ArtifactRef   `json:"student"`
	Teachers    []pipeline.ArtifactRef `json:"teachers"`
	Training    pipeline.ArtifactRef   `json:"training"`
	Recovery    pipeline.ArtifactRef   `json:"recovery"`
	Calibration pipeline.ArtifactRef   `json:"calibration"`
	Protection  pipeline.ArtifactRef   `json:"protection"`
	Heldout     pipeline.ArtifactRef   `json:"heldout"`
	Stages      []pipeline.Stage       `json:"stages"`
}

func TestMasterScopePreservesV1WireDigestAndRequiredVariants(t *testing.T) {
	for _, r := range []pipeline.Recipe{{}, recipe()} {
		legacy := legacyRecipeWire{r.Version, r.ID, r.Method, r.Student, r.Teachers, r.Training, r.Recovery, r.Calibration, r.Protection, r.Heldout, r.Stages}
		before, err := json.Marshal(legacy)
		if err != nil {
			t.Fatal(err)
		}
		after, err := json.Marshal(r)
		if err != nil || !bytes.Equal(before, after) {
			t.Fatal("v1 wire identity changed", err)
		}
		if r.Version == 1 {
			sum := sha256.Sum256(before)
			want := hex.EncodeToString(sum[:])
			got, err := r.Digest()
			if err != nil || got != want {
				t.Fatal("v1 digest changed", got, want, err)
			}
			decoded, err := pipeline.DecodeRecipe(before, want)
			if err != nil || decoded.Version != 1 || decoded.Scope != "" || len(decoded.Stages) != 13 {
				t.Fatal("v1 implicitly converted", decoded, err)
			}
		}
	}
	for n := 0; n < 13; n++ {
		r := recipe()
		r.Stages = r.Stages[:n]
		if err := r.Validate(); err == nil {
			t.Fatalf("v1 silently shortened to %d phases", n)
		}
	}
	r := recipe()
	r.Calibration = pipeline.ArtifactRef{}
	if err := r.Validate(); err == nil {
		t.Fatal("v1 calibration requirement relaxed")
	}
	for _, value := range []string{`""`, `null`, `"master"`} {
		legacy, err := json.Marshal(recipe())
		if err != nil {
			t.Fatal(err)
		}
		body := append([]byte(`{"scope":`+value+`,`), legacy[1:]...)
		sum := sha256.Sum256(body)
		if _, err := pipeline.DecodeRecipe(body, hex.EncodeToString(sum[:])); err == nil {
			t.Fatal("v1 decoder accepted a previously unknown scope field", value)
		}
	}
}

func TestMasterScopeRequiresExplicitVersionAndExactlySixCompletePhases(t *testing.T) {
	r := masterRecipe()
	if err := r.Validate(); err != nil {
		t.Fatal(err)
	}
	body, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(body)
	decoded, err := pipeline.DecodeRecipe(body, hex.EncodeToString(sum[:]))
	if err != nil || decoded.Version != 2 || decoded.Scope != "master" || decoded.Calibration != (pipeline.ArtifactRef{}) {
		t.Fatal("explicit Master document refused", err)
	}
	for name, change := range map[string]func(*pipeline.Recipe){
		"missing-scope":        func(r *pipeline.Recipe) { r.Scope = "" },
		"arbitrary-prefix":     func(r *pipeline.Recipe) { r.Scope = "through_fusion" },
		"unknown-version":      func(r *pipeline.Recipe) { r.Version = 3 },
		"v1-with-master-scope": func(r *pipeline.Recipe) { r.Version = 1 },
		"new-numerical-method": func(r *pipeline.Recipe) { r.Method = "replacement-method" },
		"missing-evaluation":   func(r *pipeline.Recipe) { r.Stages = r.Stages[:5] },
		"reordered":            func(r *pipeline.Recipe) { r.Stages[1], r.Stages[2] = r.Stages[2], r.Stages[1] },
		"different-last-phase": func(r *pipeline.Recipe) {
			r.Stages[5].Phase = pipeline.PhaseCalibration
		},
		"variant-added": func(r *pipeline.Recipe) {
			r.Stages = append(r.Stages, recipe().Stages[6:]...)
		},
		"wrong-lineage": func(r *pipeline.Recipe) { r.Stages[5].ParentStage = r.Stages[3].ID },
		"zero-budget":   func(r *pipeline.Recipe) { r.Stages[2].MaxSteps = 0 },
		"format-set":    func(r *pipeline.Recipe) { r.Stages[3].Format = "Q4_K_M" },
		"unprotected-fusion": func(r *pipeline.Recipe) {
			r.Stages[3].Inputs = []pipeline.ArtifactRef{r.Training}
		},
		"unprotected-recovery": func(r *pipeline.Recipe) {
			r.Stages[4].Inputs = []pipeline.ArtifactRef{r.Recovery}
		},
	} {
		t.Run(name, func(t *testing.T) {
			r := masterRecipe()
			change(&r)
			if err := r.Validate(); err == nil {
				t.Fatal("invalid Master scope accepted")
			}
		})
	}
}

func TestMasterScopeOptionalCalibrationKeepsDataRolesSeparated(t *testing.T) {
	r := masterRecipe()
	r.Calibration = ref("alignment-validation")
	if err := r.Validate(); err != nil {
		t.Fatal("unused optional calibration ref refused", err)
	}
	r.Stages[2].Inputs = append(r.Stages[2].Inputs, r.Calibration)
	if err := r.Validate(); err != nil {
		t.Fatal("explicit alignment calibration refused", err)
	}
	for name, cal := range map[string]pipeline.ArtifactRef{
		"missing-id":     {SHA256: ref("calibration").SHA256},
		"missing-sha":    {ID: "calibration"},
		"student-id":     {ID: r.Student.ID, SHA256: ref("calibration").SHA256},
		"train-sha":      {ID: "calibration", SHA256: r.Training.SHA256},
		"recovery-sha":   {ID: "calibration", SHA256: r.Recovery.SHA256},
		"protection-sha": {ID: "calibration", SHA256: r.Protection.SHA256},
		"heldout-sha":    {ID: "calibration", SHA256: r.Heldout.SHA256},
	} {
		t.Run(name, func(t *testing.T) {
			r := masterRecipe()
			r.Calibration = cal
			if err := r.Validate(); err == nil {
				t.Fatal("invalid or overlapping calibration accepted")
			}
		})
	}
	for _, phase := range []int{0, 1, 3, 4, 5} {
		t.Run(fmt.Sprintf("calibration-leak-stage-%d", phase), func(t *testing.T) {
			r := masterRecipe()
			r.Calibration = ref("alignment-validation")
			r.Stages[phase].Inputs = append(r.Stages[phase].Inputs, r.Calibration)
			if err := r.Validate(); err == nil {
				t.Fatal("calibration admitted outside alignment")
			}
		})
	}
	for phase := 0; phase < 5; phase++ {
		r := masterRecipe()
		r.Stages[phase].Inputs = append(r.Stages[phase].Inputs, r.Heldout)
		if err := r.Validate(); err == nil {
			t.Fatal("final heldout admitted for adaptation", phase)
		}
	}
}

// These coordinator tests verify durable order and evidence delivery with
// synthetic file handlers. They do not qualify numerical training or a Master.
func TestMasterScopeCoordinatorCompletesAndReconcilesWithoutVariantHandlers(t *testing.T) {
	config, execution, handler := durableFixture(t.TempDir())
	execution.Recipe = masterRecipe()
	delete(config.Handlers, pipeline.PhaseCalibration)
	delete(config.Handlers, pipeline.PhaseQuantize)
	delete(config.Handlers, pipeline.PhaseVariantRecovery)
	runtime := runtimeFixture(t, config)
	lost := errors.New("application receipt failed after Master publication")
	var reported []string
	err := runtime.Run(context.Background(), execution, func(_ context.Context, p pipeline.Progress) error {
		reported = append(reported, p.StageID)
		if p.StageID == execution.Recipe.Stages[5].ID {
			return lost
		}
		return nil
	})
	if !errors.Is(err, lost) || len(reported) != 6 || len(handler.calls) != 6 {
		t.Fatal("wrong delivery sequence", reported, handler.calls, err)
	}
	execution.Generation++
	p, err := runtime.Reconcile(context.Background(), execution)
	if err != nil || p.CompletedStages != 6 || p.StageID != execution.Recipe.Stages[5].ID || p.CompletedSteps != 6 || p.Checkpoint == "" {
		t.Fatal("Master snapshot not reconciled", p, err)
	}
	if err := runtime.Run(context.Background(), execution, noReport); err != nil {
		t.Fatal(err)
	}
	for i, stage := range execution.Recipe.Stages {
		if reported[i] != stage.ID || handler.calls[stage.ID] != 1 {
			t.Fatal("phase reordered or recomputed", reported, handler.calls)
		}
	}
}

func TestMasterScopeCoordinatorRequiresMasterHandlerAndAcceptedEvaluation(t *testing.T) {
	config, execution, handler := durableFixture(t.TempDir())
	execution.Recipe = masterRecipe()
	delete(config.Handlers, pipeline.PhaseMaster)
	if err := runtimeFixture(t, config).Admit(context.Background(), execution.Recipe, execution.Placement); err == nil {
		t.Fatal("missing Master handler admitted")
	}
	refused := errors.New("Master evaluation did not pass")
	config.Handlers[pipeline.PhaseMaster] = &fixturePhase{calls: map[string]int{}, run: func(context.Context, pipeline.StageContext, func(context.Context, pipeline.StageResult) error) error {
		return refused
	}}
	runtime := runtimeFixture(t, config)
	if err := runtime.Run(context.Background(), execution, noReport); !errors.Is(err, refused) {
		t.Fatal(err)
	}
	p, err := runtime.Reconcile(context.Background(), execution)
	if err != nil || p.CompletedStages != 5 || p.StageID != execution.Recipe.Stages[4].ID || len(handler.calls) != 5 {
		t.Fatal("rejected evaluation became complete", p, err)
	}
}

func TestMasterScopeCannotReinterpretPersistedV1Receipts(t *testing.T) {
	config, execution, handler := durableFixture(t.TempDir())
	runtime := runtimeFixture(t, config)
	stop := errors.New("stop after durable v1 receipt")
	if err := runtime.Run(context.Background(), execution, func(context.Context, pipeline.Progress) error { return stop }); !errors.Is(err, stop) {
		t.Fatal(err)
	}
	originalSHA, err := execution.Recipe.Digest()
	if err != nil {
		t.Fatal(err)
	}
	// Even an explicit document with the same human-facing ID is a new identity.
	execution.Recipe.Version, execution.Recipe.Scope = 2, "master"
	execution.Recipe.Stages = execution.Recipe.Stages[:6]
	execution.Generation++
	newSHA, err := execution.Recipe.Digest()
	if err != nil || newSHA == originalSHA {
		t.Fatal("v2 did not bind a distinct identity", err)
	}
	if _, err := runtime.Reconcile(context.Background(), execution); !errors.Is(err, pipeline.ErrDurableIdentity) {
		t.Fatal("v1 receipt adopted by v2", err)
	}
	if len(handler.calls) != 1 || handler.calls[execution.Recipe.Stages[0].ID] != 1 {
		t.Fatal("identity refusal started work", handler.calls)
	}
}
