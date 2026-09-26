package native_test

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	services "github.com/tayi-ai/arandu-llama/training/native"
)

func nativeCanaryResultsFixture(t *testing.T) (*nativeFixture, services.Job, []services.ExperimentNativeResultEnvelope) {
	t.Helper()
	f := newNativeFixture(t, true)
	admission, err := services.LoadExperimentNativeAdmissionForRelease(f.experiment, f.root, "canary", f.release)
	if err != nil {
		t.Fatal(err)
	}
	original := admission.Job
	results := make([]services.ExperimentNativeResultEnvelope, 0, 20)
	f.canary = services.ExperimentNativeCanaryProof{SchemaVersion: 1, Job: f.contract.Admission.Jobs["canary"], CompletedAtUTC: f.now.Add(-time.Minute).Format(time.RFC3339)}
	exit := 0
	for rank, id := range services.ExperimentNativeNodeIDs(f.experiment) {
		_, outcome := nativeResultFixture("canary", rank)
		recipe := services.ExperimentNativeRecipe(f.experiment)
		outcome.Node = id
		outcome.ModelManifestSHA256 = recipe.Model.ManifestSHA256
		outcome.InitialAdapterSHA256 = recipe.LoRA.ExpectedInitialDigest
		outcome.JobID, outcome.Generation, outcome.ReleaseSHA256 = original.ID, original.Generation, original.RuntimeDigest
		outcome.ContractSHA256, outcome.RuntimeSHA256 = f.manifest.Files["contract.json"], f.manifest.RuntimeSHA256
		outcome.CalibrationReferenceSHA256 = f.manifest.Files["calibration-reference.json"]
		outcome.ResultSHA256, outcome.TelemetrySHA256 = hash([]byte("remote-result-"+id)), hash([]byte("remote-telemetry-"+id))
		results = append(results, outcome)
		f.canary.Nodes = append(f.canary.Nodes, services.ExperimentNativeCanaryNode{Node: id, Rank: rank, State: "succeeded", Exit: &exit,
			GPUAdmission: true, ResultSHA256: outcome.ResultSHA256, TelemetrySHA256: outcome.TelemetrySHA256,
			CalibrationExamples: 8, CalibrationReferenceSHA256: outcome.CalibrationReferenceSHA256, ReferenceProbabilitiesMatch: true})
	}
	f.seal(t, true)
	if f.release.ManifestSHA256 == original.RuntimeDigest {
		t.Fatal("adding the canary proof did not change the release identity")
	}
	return f, original, results
}

func TestNativeCanaryProofBindingPreservesOriginalRelease(t *testing.T) {
	f, original, results := nativeCanaryResultsFixture(t)
	if err := services.ValidateExperimentNativeCanaryResultsForRelease(f.experiment, f.root, original, results, f.release); err != nil {
		t.Fatalf("matching canary evidence rejected after manifest transition: %v", err)
	}
	rebuilt, err := services.LoadExperimentNativeAdmissionForRelease(f.experiment, f.root, "canary", f.release)
	if err != nil {
		t.Fatal(err)
	}
	if err := services.ValidateExperimentNativeCanaryResultsForRelease(f.experiment, f.root, rebuilt.Job, results, f.release); err == nil {
		t.Fatal("canary identity reconstructed from train release was admitted")
	}
	if err := services.ValidateExperimentNativeCanaryResults(f.experiment, f.root, original, results); err == nil {
		t.Fatal("production release pin accepted a fixture bundle")
	}
	noOutput(t, f)
}

func TestNativeCanaryProofBindingRejectsIndividuallyValidDifferentEvidence(t *testing.T) {
	f, original, results := nativeCanaryResultsFixture(t)
	for name, mutate := range map[string]func(*services.ExperimentNativeResultEnvelope){
		"runtime": func(r *services.ExperimentNativeResultEnvelope) { r.RuntimeSHA256 = strings.Repeat("d", 64) },
		"calibration": func(r *services.ExperimentNativeResultEnvelope) {
			r.CalibrationReferenceSHA256 = strings.Repeat("d", 64)
		},
		"remote_result": func(r *services.ExperimentNativeResultEnvelope) { r.ResultSHA256 = strings.Repeat("d", 64) },
		"telemetry":     func(r *services.ExperimentNativeResultEnvelope) { r.TelemetrySHA256 = strings.Repeat("d", 64) },
		"fleet_json_instead_of_remote_file": func(r *services.ExperimentNativeResultEnvelope) {
			body, _ := json.Marshal(struct {
				Output string `json:"output"`
			}{Output: "different hash domain"})
			r.ResultSHA256 = hash(body)
		},
	} {
		t.Run(name, func(t *testing.T) {
			changed := append([]services.ExperimentNativeResultEnvelope(nil), results...)
			mutate(&changed[19])
			if _, err := services.ParseExperimentNativeResult(f.experiment, nativeResultOutput(t, changed[19]), original, changed[19].Node, 19, "canary"); err != nil {
				t.Fatalf("test requires an independently valid summary: %v", err)
			}
			if _, err := services.LoadExperimentNativeAdmissionForRelease(f.experiment, f.root, "train", f.release); err != nil {
				t.Fatalf("test requires an independently valid train proof: %v", err)
			}
			if err := services.ValidateExperimentNativeCanaryResultsForRelease(f.experiment, f.root, original, changed, f.release); err == nil {
				t.Fatal("different collected evidence was admitted")
			}
		})
	}
	for _, field := range []string{"result", "telemetry"} {
		t.Run("changed_pinned_proof_"+field, func(t *testing.T) {
			prior := f.canary.Nodes[19]
			if field == "result" {
				f.canary.Nodes[19].ResultSHA256 = strings.Repeat("e", 64)
			} else {
				f.canary.Nodes[19].TelemetrySHA256 = strings.Repeat("e", 64)
			}
			f.seal(t, true)
			if _, err := services.LoadExperimentNativeAdmissionForRelease(f.experiment, f.root, "train", f.release); err != nil {
				t.Fatalf("test requires individually valid proof: %v", err)
			}
			if err := services.ValidateExperimentNativeCanaryResultsForRelease(f.experiment, f.root, original, results, f.release); err == nil {
				t.Fatal("proof unbound from original evidence was admitted")
			}
			f.canary.Nodes[19] = prior
			f.seal(t, true)
		})
	}
}

func TestNativeCanaryProofBindingRejectsWrongJobOrderAndIncompleteResults(t *testing.T) {
	f, original, results := nativeCanaryResultsFixture(t)
	for name, mutate := range map[string]func(*services.Job){
		"id":            func(j *services.Job) { j.ID += "-other" },
		"generation":    func(j *services.Job) { j.Generation++ },
		"action":        func(j *services.Job) { j.Action = "other" },
		"contract":      func(j *services.Job) { j.ContractVersion = "other" },
		"recipe":        func(j *services.Job) { j.ModelRecipe = "other" },
		"model":         func(j *services.Job) { j.ModelDigest = strings.Repeat("e", 64) },
		"release":       func(j *services.Job) { j.RuntimeDigest = strings.Repeat("e", 64) },
		"request_bytes": func(j *services.Job) { j.Request = append([]byte(" "), j.Request...) },
	} {
		t.Run(name, func(t *testing.T) {
			job := original
			mutate(&job)
			if err := services.ValidateExperimentNativeCanaryResultsForRelease(f.experiment, f.root, job, results, f.release); err == nil {
				t.Fatal("changed original canary identity admitted")
			}
		})
	}
	for name, mutate := range map[string]func([]services.ExperimentNativeResultEnvelope) []services.ExperimentNativeResultEnvelope{
		"missing": func(r []services.ExperimentNativeResultEnvelope) []services.ExperimentNativeResultEnvelope {
			return r[:19]
		},
		"extra": func(r []services.ExperimentNativeResultEnvelope) []services.ExperimentNativeResultEnvelope {
			return append(r, r[0])
		},
		"duplicate_rank": func(r []services.ExperimentNativeResultEnvelope) []services.ExperimentNativeResultEnvelope {
			r[19] = r[0]
			return r
		},
		"swapped_order": func(r []services.ExperimentNativeResultEnvelope) []services.ExperimentNativeResultEnvelope {
			r[0], r[1] = r[1], r[0]
			return r
		},
		"failed": func(r []services.ExperimentNativeResultEnvelope) []services.ExperimentNativeResultEnvelope {
			r[19].Status = "FAILED"
			return r
		},
		"different_final_adapter": func(r []services.ExperimentNativeResultEnvelope) []services.ExperimentNativeResultEnvelope {
			r[19].FinalAdapterSHA256 = strings.Repeat("e", 64)
			return r
		},
	} {
		t.Run(name, func(t *testing.T) {
			changed := mutate(append([]services.ExperimentNativeResultEnvelope(nil), results...))
			if err := services.ValidateExperimentNativeCanaryResultsForRelease(f.experiment, f.root, original, changed, f.release); err == nil {
				t.Fatal("incomplete or inconsistent canary set admitted")
			}
		})
	}
	noOutput(t, f)
}
