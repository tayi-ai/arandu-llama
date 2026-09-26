package pipeline_test

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/pipeline"
)

func ref(id string) pipeline.ArtifactRef {
	s := sha256.Sum256([]byte(id))
	return pipeline.ArtifactRef{ID: id, SHA256: hex.EncodeToString(s[:])}
}
func recipe() pipeline.Recipe {
	r := pipeline.Recipe{Version: 1, ID: "admitted-v1", Method: "generational-fusion-v1", Student: ref("student"), Teachers: []pipeline.ArtifactRef{ref("teacher")}, Training: ref("train"), Calibration: ref("calibration"), Protection: ref("protection"), Heldout: ref("heldout")}
	add := func(phase pipeline.Phase, input pipeline.ArtifactRef, format string) {
		parent := ""
		if len(r.Stages) > 0 {
			parent = r.Stages[len(r.Stages)-1].ID
		}
		if phase == pipeline.PhaseQuantize {
			parent = r.Stages[5].ID
		}
		inputs := []pipeline.ArtifactRef{input}
		if phase == pipeline.PhaseFusion {
			inputs = append(inputs, r.Protection)
		}
		r.Stages = append(r.Stages, pipeline.Stage{ID: fmt.Sprintf("stage-%d", len(r.Stages)), Phase: phase, Inputs: inputs, MaxSteps: 1, MaxTokens: 128, TimeoutSeconds: 60, Format: format, ParentStage: parent})
	}
	for _, p := range []pipeline.Phase{pipeline.PhaseSFT, pipeline.PhaseTeacherCache, pipeline.PhaseAlignment, pipeline.PhaseFusion} {
		add(p, r.Training, "")
	}
	add(pipeline.PhaseRecovery, r.Protection, "")
	add(pipeline.PhaseMaster, r.Heldout, "")
	add(pipeline.PhaseCalibration, r.Calibration, "")
	for _, f := range []string{"Q8_0", "Q6_K", "Q4_K_M"} {
		add(pipeline.PhaseQuantize, r.Calibration, f)
		add(pipeline.PhaseVariantRecovery, r.Protection, f)
	}
	return r
}
func TestRecipePinsEveryPhaseAndPrecision(t *testing.T) {
	r := recipe()
	if err := r.Validate(); err != nil {
		t.Fatal(err)
	}
	d, err := r.Digest()
	if err != nil || len(d) != 64 {
		t.Fatal(d, err)
	}
}
func TestRecipeRejectsMissingPhasesAndHeldoutLeak(t *testing.T) {
	for _, kind := range []string{"missing", "reorder", "leak", "digest", "precision", "role", "protection_leak", "lineage"} {
		t.Run(kind, func(t *testing.T) {
			r := recipe()
			switch kind {
			case "missing":
				r.Stages = r.Stages[:11]
			case "reorder":
				r.Stages[1], r.Stages[2] = r.Stages[2], r.Stages[1]
			case "leak":
				r.Stages[0].Inputs = append(r.Stages[0].Inputs, r.Heldout)
			case "digest":
				r.Stages[0].Inputs[0].SHA256 = ref("different").SHA256
			case "precision":
				r.Stages[9].Format = "Q8_0"
			case "role":
				r.Stages[0].Inputs = []pipeline.ArtifactRef{r.Calibration}
			case "protection_leak":
				r.Stages[1].Inputs = append(r.Stages[1].Inputs, r.Protection)
			case "lineage":
				r.Stages[9].ParentStage = r.Stages[7].ID
			}
			if err := r.Validate(); err == nil {
				t.Fatal("invalid recipe accepted")
			}
		})
	}
}
func TestDecodeRecipeRefusesChangedBytesAndUnknownFields(t *testing.T) {
	body, _ := json.Marshal(recipe())
	sum := sha256.Sum256(body)
	if _, err := pipeline.DecodeRecipe(body, hex.EncodeToString(sum[:])); err != nil {
		t.Fatal(err)
	}
	if _, err := pipeline.DecodeRecipe(append(body, ' '), hex.EncodeToString(sum[:])); err == nil {
		t.Fatal("changed bytes admitted")
	}
	body = append([]byte(`{"extra":1,`), body[1:]...)
	sum = sha256.Sum256(body)
	if _, err := pipeline.DecodeRecipe(body, hex.EncodeToString(sum[:])); err == nil {
		t.Fatal("unknown field admitted")
	}
}
