package native_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"math"
	"strings"
	"testing"

	services "github.com/tayi-ai/arandu-llama/training/native"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

func nativeResultFixture(mode string, rank int) (services.Job, services.ExperimentNativeResultEnvelope) {
	recipe := services.ExperimentNativeRecipe(testNativeExperiment())
	request, _ := json.Marshal(map[string]string{"mode": mode, "contract_sha256": strings.Repeat("1", 64)})
	job := services.Job{ID: "native-result-" + mode, Generation: 7, Action: "experiment.train", ContractVersion: services.ExperimentNativeContractVersion,
		RuntimeDigest: strings.Repeat("2", 64), ModelRecipe: "fixture-decoder", ModelDigest: recipe.Model.ManifestSHA256, Request: request}
	result := services.ExperimentNativeResultEnvelope{Kind: services.ExperimentNativeResultKind, Status: "PASSED", Mode: mode,
		Node: services.ExperimentNativeNodeIDs(testNativeExperiment())[rank], Rank: rank, JobID: job.ID, Generation: job.Generation,
		ContractSHA256: strings.Repeat("1", 64), ReleaseSHA256: job.RuntimeDigest, RuntimeSHA256: strings.Repeat("3", 64),
		ModelManifestSHA256: job.ModelDigest, WorldSize: 20, GPUWorldSize: 40, OptimizerSteps: 4, LocalExamples: 4, GlobalUniqueExamples: 4,
		InitialAdapterSHA256: recipe.LoRA.ExpectedInitialDigest, FinalAdapterSHA256: strings.Repeat("4", 64),
		BaseBeforeSHA256: strings.Repeat("5", 64), BaseAfterSHA256: strings.Repeat("5", 64), AllFinite: true, GPUAdmission: true,
		CalibrationExamples: 8, CalibrationReferenceSHA256: strings.Repeat("6", 64), ReferenceProbabilitiesMatch: true,
		ResultSHA256: strings.Repeat("7", 64), TelemetrySHA256: strings.Repeat("8", 64), MinimumHostAvailableBytes: 6 << 30,
		MinimumCgroupAvailableBytes: 512 << 20, Checkpoints: []services.ExperimentNativeCheckpointReceipt{}}
	if rank < 4 {
		result.LocalExamples++
	}
	for device := 0; device < 2; device++ {
		result.GPUs = append(result.GPUs, torch.CUDAMemoryStats{Device: device, FreeBytes: 5 << 30, TotalBytes: 16 << 30,
			AllocatedBytes: 9 << 30, ReservedBytes: 10 << 30, PeakAllocatedBytes: 10 << 30, PeakReservedBytes: 11 << 30,
			AllocatorBackend: "native", AllocatorEnabled: true, AllocatorStatsValid: true, AllocatorFraction: .80})
	}
	if mode == "train" {
		result.OptimizerSteps, result.LocalExamples, result.GlobalUniqueExamples = 18, 18, 378
		if rank < 18 {
			result.LocalExamples++
		}
		result.CalibrationExamples, result.ReferenceProbabilitiesMatch = 0, false
		if rank == 0 {
			result.Checkpoints = []services.ExperimentNativeCheckpointReceipt{
				{Step: 9, Candidate: false, AdapterSHA256: strings.Repeat("9", 64), ArtifactSHA256: strings.Repeat("a", 64)},
				{Step: 18, Candidate: true, AdapterSHA256: result.FinalAdapterSHA256, ArtifactSHA256: strings.Repeat("b", 64)},
			}
		}
	}
	return job, result
}

func nativeResultOutput(t *testing.T, result services.ExperimentNativeResultEnvelope) string {
	t.Helper()
	var output bytes.Buffer
	if err := services.EmitExperimentNativeResult(&output, result); err != nil {
		t.Fatal(err)
	}
	return output.String()
}

func TestNativeResultRoundTripForEveryRankAndMode(t *testing.T) {
	for _, mode := range []string{"canary", "train"} {
		for rank := range len(services.ExperimentNativeNodeIDs(testNativeExperiment())) {
			job, expected := nativeResultFixture(mode, rank)
			output := "native diagnostic line\n{\"phase\":\"completed\"}\n" + nativeResultOutput(t, expected)
			actual, err := services.ParseExperimentNativeResult(testNativeExperiment(), output, job, expected.Node, rank, mode)
			if err != nil || actual.FinalAdapterSHA256 != expected.FinalAdapterSHA256 || actual.ReleaseSHA256 == actual.RuntimeSHA256 ||
				actual.GlobalUniqueExamples != expected.GlobalUniqueExamples || len(actual.Checkpoints) != len(expected.Checkpoints) {
				t.Fatalf("mode=%s rank=%d: result=%+v err=%v", mode, rank, actual, err)
			}
		}
	}
}

func TestNativeResultRequiresVersionedContributionCounts(t *testing.T) {
	for _, mode := range []string{"canary", "train"} {
		total := 0
		for rank := range len(services.ExperimentNativeNodeIDs(testNativeExperiment())) {
			job, result := nativeResultFixture(mode, rank)
			total += result.LocalExamples
			result.LocalExamples++
			if _, err := services.ParseExperimentNativeResult(testNativeExperiment(), nativeResultOutput(t, result), job, result.Node, rank, mode); err == nil {
				t.Fatal("invented extra contribution was accepted")
			}
			result.LocalExamples -= 2
			if _, err := services.ParseExperimentNativeResult(testNativeExperiment(), nativeResultOutput(t, result), job, result.Node, rank, mode); err == nil {
				t.Fatal("missing local contribution was accepted")
			}
		}
		expected := 378
		if mode == "canary" {
			expected = 84
		}
		if total != expected {
			t.Fatalf("%s contribution total: got %d, want %d", mode, total, expected)
		}
	}
}

func TestNativeResultRejectsHistoricalFleetAndExcludedNodes(t *testing.T) {
	job, result := nativeResultFixture("canary", 0)
	for _, node := range []string{"zero", "nine"} {
		changed := result
		changed.Node = node
		if _, err := services.ParseExperimentNativeResult(testNativeExperiment(), nativeResultOutput(t, changed), job, node, 0, "canary"); err == nil {
			t.Fatal("excluded node result was admitted")
		}
	}
	result.WorldSize, result.GPUWorldSize = 21, 42
	if _, err := services.ParseExperimentNativeResult(testNativeExperiment(), nativeResultOutput(t, result), job, result.Node, 0, "canary"); err == nil {
		t.Fatal("historical fleet result was admitted")
	}
	job.ContractVersion = "tayi.experiment.native.v2"
	if _, err := services.ParseExperimentNativeResult(testNativeExperiment(), nativeResultOutput(t, result), job, "nine", 0, "canary"); err == nil {
		t.Fatal("historical contract was admitted for a new execution")
	}
}

func TestNativeResultAdmitsT10RoundedAllocatorWithoutIncreasingCeiling(t *testing.T) {
	const total int64 = 16703356928
	const ceiling int64 = 13362685542
	for _, mode := range []string{"canary", "train"} {
		t.Run(mode, func(t *testing.T) {
			job, result := nativeResultFixture(mode, 0)
			for device := range result.GPUs {
				result.GPUs[device].TotalBytes = total
				result.GPUs[device].AllocatorFraction = float64(ceiling) / float64(total)
				result.GPUs[device].PeakReservedBytes = ceiling
			}
			if _, err := services.ParseExperimentNativeResult(testNativeExperiment(), nativeResultOutput(t, result), job, result.Node, 0, mode); err != nil {
				t.Fatalf("valid T10 allocator result rejected: %v", err)
			}
			result.GPUs[0].PeakReservedBytes++
			if _, err := services.ParseExperimentNativeResult(testNativeExperiment(), nativeResultOutput(t, result), job, result.Node, 0, mode); err == nil {
				t.Fatal("peak one byte above the unchanged ceiling was admitted")
			}
			result.GPUs[0].PeakReservedBytes = ceiling
			result.GPUs[0].AllocatorFraction = float64(ceiling+1) / float64(total)
			if _, err := services.ParseExperimentNativeResult(testNativeExperiment(), nativeResultOutput(t, result), job, result.Node, 0, mode); err == nil {
				t.Fatal("allocator limit one byte above the unchanged ceiling was admitted")
			}
		})
	}
}

func TestNativeResultRejectsIncompleteAmbiguousAndLegacyOutput(t *testing.T) {
	job, result := nativeResultFixture("canary", 0)
	valid := nativeResultOutput(t, result)
	legacy := strings.ReplaceAll(valid, services.ExperimentNativeResultKind, "EXPERIMENT_DDP21_ALIGNMENT_V1")
	for name, output := range map[string]string{
		"empty": "", "legacy": legacy, "mixed_legacy": legacy + valid, "duplicate_summary": valid + valid,
		"truncated": valid[:len(valid)-5], "truncated_then_valid": valid[:len(valid)-5] + "\n" + valid,
		"duplicate_kind":     strings.Replace(valid, `"kind":`, `"kind":"ignored","kind":`, 1),
		"duplicate_status":   strings.Replace(valid, `"status":`, `"status":"FAILED","status":`, 1),
		"unknown_field":      strings.Replace(valid, "{", `{"unknown":true,`, 1),
		"case_alias":         strings.Replace(valid, "{", `{"RANK":0,`, 1),
		"oversized_output":   strings.Repeat("x", 16<<20) + valid,
		"oversized_envelope": strings.Replace(valid, "{", `{`+strings.Repeat(" ", 64<<10), 1),
	} {
		t.Run(name, func(t *testing.T) {
			if got, err := services.ParseExperimentNativeResult(testNativeExperiment(), output, job, result.Node, result.Rank, result.Mode); err == nil || got.Kind != "" {
				t.Fatalf("accepted invalid output: %v", err)
			}
		})
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal([]byte(valid), &fields); err != nil {
		t.Fatal(err)
	}
	for name, original := range fields {
		t.Run("missing_"+name, func(t *testing.T) {
			delete(fields, name)
			body, _ := json.Marshal(fields)
			fields[name] = original
			if _, err := services.ParseExperimentNativeResult(testNativeExperiment(), string(body), job, result.Node, result.Rank, result.Mode); err == nil {
				t.Fatal("accepted missing field")
			}
		})
		t.Run("null_"+name, func(t *testing.T) {
			fields[name] = json.RawMessage("null")
			body, _ := json.Marshal(fields)
			fields[name] = original
			if _, err := services.ParseExperimentNativeResult(testNativeExperiment(), string(body), job, result.Node, result.Rank, result.Mode); err == nil {
				t.Fatal("accepted null field")
			}
		})
	}
	for _, output := range []string{
		strings.Replace(valid, `"device":0,`, "", 1),
		strings.Replace(valid, `"allocator_fraction":0.8`, `"allocator_fraction":0.8,"allocator_fraction":0.8`, 1),
	} {
		if _, err := services.ParseExperimentNativeResult(testNativeExperiment(), output, job, result.Node, result.Rank, result.Mode); err == nil {
			t.Fatal("accepted incomplete or duplicate GPU telemetry")
		}
	}
}

func TestNativeResultRejectsWrongIdentityMeasurementsAndCanaryClaims(t *testing.T) {
	mutations := map[string]func(*services.ExperimentNativeResultEnvelope){
		"failed":                func(r *services.ExperimentNativeResultEnvelope) { r.Status = "FAILED" },
		"node":                  func(r *services.ExperimentNativeResultEnvelope) { r.Node = "other" },
		"rank":                  func(r *services.ExperimentNativeResultEnvelope) { r.Rank = 1 },
		"job":                   func(r *services.ExperimentNativeResultEnvelope) { r.JobID += "-other" },
		"generation":            func(r *services.ExperimentNativeResultEnvelope) { r.Generation++ },
		"mode":                  func(r *services.ExperimentNativeResultEnvelope) { r.Mode = "train" },
		"contract":              func(r *services.ExperimentNativeResultEnvelope) { r.ContractSHA256 = strings.Repeat("e", 64) },
		"release":               func(r *services.ExperimentNativeResultEnvelope) { r.ReleaseSHA256 = r.RuntimeSHA256 },
		"binary":                func(r *services.ExperimentNativeResultEnvelope) { r.RuntimeSHA256 = "not-a-digest" },
		"model":                 func(r *services.ExperimentNativeResultEnvelope) { r.ModelManifestSHA256 = r.RuntimeSHA256 },
		"world":                 func(r *services.ExperimentNativeResultEnvelope) { r.WorldSize = 42 },
		"gpu_world":             func(r *services.ExperimentNativeResultEnvelope) { r.GPUWorldSize = 21 },
		"updates":               func(r *services.ExperimentNativeResultEnvelope) { r.OptimizerSteps = 1 },
		"local_examples":        func(r *services.ExperimentNativeResultEnvelope) { r.LocalExamples = 21 },
		"false_unique_coverage": func(r *services.ExperimentNativeResultEnvelope) { r.GlobalUniqueExamples = 84 },
		"initial_adapter":       func(r *services.ExperimentNativeResultEnvelope) { r.InitialAdapterSHA256 = r.FinalAdapterSHA256 },
		"unchanged_adapter":     func(r *services.ExperimentNativeResultEnvelope) { r.FinalAdapterSHA256 = r.InitialAdapterSHA256 },
		"mutated_base":          func(r *services.ExperimentNativeResultEnvelope) { r.BaseAfterSHA256 = r.FinalAdapterSHA256 },
		"missing_base":          func(r *services.ExperimentNativeResultEnvelope) { r.BaseBeforeSHA256, r.BaseAfterSHA256 = "", "" },
		"nonfinite":             func(r *services.ExperimentNativeResultEnvelope) { r.AllFinite = false },
		"gpu_unadmitted":        func(r *services.ExperimentNativeResultEnvelope) { r.GPUAdmission = false },
		"calibration_count":     func(r *services.ExperimentNativeResultEnvelope) { r.CalibrationExamples = 7 },
		"calibration_reference": func(r *services.ExperimentNativeResultEnvelope) { r.CalibrationReferenceSHA256 = "" },
		"calibration_failed":    func(r *services.ExperimentNativeResultEnvelope) { r.ReferenceProbabilitiesMatch = false },
		"result_hash":           func(r *services.ExperimentNativeResultEnvelope) { r.ResultSHA256 = "" },
		"telemetry_hash":        func(r *services.ExperimentNativeResultEnvelope) { r.TelemetrySHA256 = strings.Repeat("0", 64) },
		"host_reserve":          func(r *services.ExperimentNativeResultEnvelope) { r.MinimumHostAvailableBytes-- },
		"cgroup_reserve":        func(r *services.ExperimentNativeResultEnvelope) { r.MinimumCgroupAvailableBytes-- },
		"missing_gpu":           func(r *services.ExperimentNativeResultEnvelope) { r.GPUs = r.GPUs[:1] },
		"duplicate_gpu":         func(r *services.ExperimentNativeResultEnvelope) { r.GPUs[1].Device = 0 },
		"allocator_disabled":    func(r *services.ExperimentNativeResultEnvelope) { r.GPUs[0].AllocatorEnabled = false },
		"invalid_stats":         func(r *services.ExperimentNativeResultEnvelope) { r.GPUs[0].AllocatorStatsValid = false },
		"wrong_backend":         func(r *services.ExperimentNativeResultEnvelope) { r.GPUs[0].AllocatorBackend = "cudaMallocAsync" },
		"wrong_cap":             func(r *services.ExperimentNativeResultEnvelope) { r.GPUs[0].AllocatorFraction = .90 },
		"zero_peak":             func(r *services.ExperimentNativeResultEnvelope) { r.GPUs[0].PeakReservedBytes = 0 },
		"peak_over_cap":         func(r *services.ExperimentNativeResultEnvelope) { r.GPUs[0].PeakReservedBytes = 13 << 30 },
		"peak_below_current":    func(r *services.ExperimentNativeResultEnvelope) { r.GPUs[0].PeakAllocatedBytes = 8 << 30 },
		"negative_allocated":    func(r *services.ExperimentNativeResultEnvelope) { r.GPUs[0].AllocatedBytes = -1 },
		"impossible_free":       func(r *services.ExperimentNativeResultEnvelope) { r.GPUs[0].FreeBytes = r.GPUs[0].TotalBytes + 1 },
		"canary_checkpoint": func(r *services.ExperimentNativeResultEnvelope) {
			r.Checkpoints = []services.ExperimentNativeCheckpointReceipt{{Step: 9}}
		},
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			job, result := nativeResultFixture("canary", 0)
			mutate(&result)
			if _, err := services.ParseExperimentNativeResult(testNativeExperiment(), nativeResultOutput(t, result), job, services.ExperimentNativeNodeIDs(testNativeExperiment())[0], 0, "canary"); err == nil {
				t.Fatal("accepted invalid result")
			}
		})
	}
}

func TestNativeResultRequiresAdmittedJobAndRank(t *testing.T) {
	mutations := map[string]func(*services.Job){
		"action":            func(j *services.Job) { j.Action = "other" },
		"generation":        func(j *services.Job) { j.Generation = 0 },
		"legacy_contract":   func(j *services.Job) { j.ContractVersion = "tayi.experiment.ddp21.v1" },
		"runtime":           func(j *services.Job) { j.RuntimeDigest = strings.Repeat("f", 64) },
		"recipe":            func(j *services.Job) { j.ModelRecipe = "other" },
		"model":             func(j *services.Job) { j.ModelDigest = strings.Repeat("f", 64) },
		"extra_request":     func(j *services.Job) { j.Request = append([]byte(`{"extra":1,`), j.Request[1:]...) },
		"duplicate_request": func(j *services.Job) { j.Request = append([]byte(`{"mode":"train",`), j.Request[1:]...) },
		"missing_request":   func(j *services.Job) { j.Request = json.RawMessage(`{"mode":"canary"}`) },
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			job, result := nativeResultFixture("canary", 0)
			mutate(&job)
			if _, err := services.ParseExperimentNativeResult(testNativeExperiment(), nativeResultOutput(t, result), job, result.Node, 0, "canary"); err == nil {
				t.Fatal("accepted unadmitted job")
			}
		})
	}
	job, result := nativeResultFixture("canary", 0)
	for _, rank := range []int{-1, 1, 21} {
		if _, err := services.ParseExperimentNativeResult(testNativeExperiment(), nativeResultOutput(t, result), job, result.Node, rank, "canary"); err == nil {
			t.Fatal("accepted inconsistent rank")
		}
	}
}

func TestNativeResultRequiresRankZeroTrainingCheckpoints(t *testing.T) {
	for name, mutate := range map[string]func(*services.ExperimentNativeResultEnvelope){
		"missing":          func(r *services.ExperimentNativeResultEnvelope) { r.Checkpoints = r.Checkpoints[:1] },
		"wrong_step":       func(r *services.ExperimentNativeResultEnvelope) { r.Checkpoints[0].Step = 8 },
		"wrong_candidate":  func(r *services.ExperimentNativeResultEnvelope) { r.Checkpoints[0].Candidate = true },
		"missing_artifact": func(r *services.ExperimentNativeResultEnvelope) { r.Checkpoints[0].ArtifactSHA256 = "" },
		"wrong_final_adapter": func(r *services.ExperimentNativeResultEnvelope) {
			r.Checkpoints[1].AdapterSHA256 = r.Checkpoints[0].AdapterSHA256
		},
		"fake_local_calibration": func(r *services.ExperimentNativeResultEnvelope) { r.CalibrationExamples = 8 },
		"fake_reference_match":   func(r *services.ExperimentNativeResultEnvelope) { r.ReferenceProbabilitiesMatch = true },
		"partial_epoch":          func(r *services.ExperimentNativeResultEnvelope) { r.GlobalUniqueExamples = 377 },
	} {
		t.Run(name, func(t *testing.T) {
			job, result := nativeResultFixture("train", 0)
			mutate(&result)
			if _, err := services.ParseExperimentNativeResult(testNativeExperiment(), nativeResultOutput(t, result), job, result.Node, 0, "train"); err == nil {
				t.Fatal("accepted invalid training checkpoint result")
			}
		})
	}
	job, other := nativeResultFixture("train", 1)
	_, zero := nativeResultFixture("train", 0)
	other.Checkpoints = zero.Checkpoints
	if _, err := services.ParseExperimentNativeResult(testNativeExperiment(), nativeResultOutput(t, other), job, other.Node, 1, "train"); err == nil {
		t.Fatal("accepted peer checkpoint ownership")
	}
	job, zero = nativeResultFixture("train", 0)
	output := strings.Replace(nativeResultOutput(t, zero), `"candidate":false,`, "", 1)
	if _, err := services.ParseExperimentNativeResult(testNativeExperiment(), output, job, zero.Node, 0, "train"); err == nil {
		t.Fatal("accepted omitted checkpoint flag")
	}
}

type nativeResultWriter struct{ err error }

func (w nativeResultWriter) Write(body []byte) (int, error) { return len(body) / 2, w.err }

func TestNativeResultEmitPropagatesWriterAndEncodingFailures(t *testing.T) {
	_, result := nativeResultFixture("canary", 0)
	failure := errors.New("write failed")
	if err := services.EmitExperimentNativeResult(nativeResultWriter{err: failure}, result); !errors.Is(err, failure) {
		t.Fatalf("writer error lost: %v", err)
	}
	if err := services.EmitExperimentNativeResult(nativeResultWriter{}, result); !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("short write accepted: %v", err)
	}
	if err := services.EmitExperimentNativeResult(nil, result); err == nil {
		t.Fatal("nil writer accepted")
	}
	var output bytes.Buffer
	result.GPUs[0].AllocatorFraction = math.NaN()
	if err := services.EmitExperimentNativeResult(&output, result); err == nil || output.Len() != 0 {
		t.Fatal("nonfinite JSON emitted")
	}
}
