package native

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"strings"

	"github.com/tayi-ai/arandu-llama/training/torch"
)

// ExperimentNativeResultKind identifies the versioned native result envelope.
const ExperimentNativeResultKind = "EXPERIMENT_NATIVE_ALIGNMENT_V1"

const (
	experimentNativeOutputLimit   = 16 << 20
	experimentNativeEnvelopeLimit = 64 << 10
)

// ExperimentNativeResultEnvelope is one node's measured terminal result. ReleaseSHA256
// identifies the admitted manifest; RuntimeSHA256 identifies its executable.
// The file digests name persisted evidence, not bytes transported by Fleet.
// GlobalUniqueExamples describes the fixed schedule, not independently observed
// remote completion. The coordinator must reconcile the complete admitted fleet.
// GPUs contain cumulative allocator peaks; they are not whole-process GPU peaks.
type ExperimentNativeResultEnvelope struct {
	Kind                        string                              `json:"kind"`
	Status                      string                              `json:"status"`
	Mode                        string                              `json:"mode"`
	Node                        string                              `json:"node"`
	Rank                        int                                 `json:"rank"`
	JobID                       string                              `json:"job_id"`
	Generation                  uint64                              `json:"job_generation"`
	ContractSHA256              string                              `json:"contract_sha256"`
	ReleaseSHA256               string                              `json:"release_sha256"`
	RuntimeSHA256               string                              `json:"runtime_sha256"`
	ModelManifestSHA256         string                              `json:"model_manifest_sha256"`
	WorldSize                   int                                 `json:"world_size"`
	GPUWorldSize                int                                 `json:"gpu_world_size"`
	OptimizerSteps              int                                 `json:"optimizer_steps"`
	LocalExamples               int                                 `json:"local_examples"`
	GlobalUniqueExamples        int                                 `json:"global_unique_examples"`
	InitialAdapterSHA256        string                              `json:"initial_adapter_sha256"`
	FinalAdapterSHA256          string                              `json:"final_adapter_sha256"`
	BaseBeforeSHA256            string                              `json:"base_before_sha256"`
	BaseAfterSHA256             string                              `json:"base_after_sha256"`
	AllFinite                   bool                                `json:"all_finite"`
	GPUAdmission                bool                                `json:"gpu_admission"`
	CalibrationExamples         int                                 `json:"calibration_examples"`
	CalibrationReferenceSHA256  string                              `json:"calibration_reference_sha256"`
	ReferenceProbabilitiesMatch bool                                `json:"reference_probabilities_match"`
	ResultSHA256                string                              `json:"result_sha256"`
	TelemetrySHA256             string                              `json:"telemetry_sha256"`
	MinimumHostAvailableBytes   int64                               `json:"minimum_host_available_bytes"`
	MinimumCgroupAvailableBytes int64                               `json:"minimum_cgroup_available_bytes"`
	GPUs                        []torch.CUDAMemoryStats             `json:"gpus"`
	Checkpoints                 []ExperimentNativeCheckpointReceipt `json:"checkpoints"`
}

// ParseExperimentNativeResult admits exactly one complete native success envelope in
// bounded Fleet output. It performs no filesystem, CUDA or network operation.
// The caller must first verify Fleet's terminal state, exit and output digest.
// Binary/library/input provenance comes from the admitted release; this parser
// does not independently open or rehash the evidence files named by the result.
func ParseExperimentNativeResult(experiment *NativeExperiment, output string, job Job, node string, rank int, mode string) (ExperimentNativeResultEnvelope, error) {
	if experiment == nil {
		return ExperimentNativeResultEnvelope{}, errors.New("native experiment: installation is not configured")
	}

	fail := func() (ExperimentNativeResultEnvelope, error) {
		return ExperimentNativeResultEnvelope{}, errors.New("experiment native: terminal result rejected")
	}
	ids := ExperimentNativeNodeIDs(experiment)
	if len(output) > experimentNativeOutputLimit || rank < 0 || rank >= len(ids) || node != ids[rank] ||
		(mode != "canary" && mode != "train") || job.Action != "experiment.train" || !experimentJobID.MatchString(job.ID) || job.Generation == 0 ||
		job.ContractVersion != ExperimentNativeContractVersion || !experimentNativeResultHash(job.RuntimeDigest) ||
		job.ModelRecipe !=
			experiment.config.ModelRecipe ||
		job.ModelDigest !=
			experiment.config.Recipe.Model.ManifestSHA256 {
		return fail()
	}
	var request experimentRequest
	if experimentDecode(job.Request, &request, true) != nil || request.Mode != mode || !experimentNativeResultHash(request.ContractSHA256) ||
		!experimentNativeResultFields(job.Request, "mode contract_sha256") {
		return fail()
	}
	var result ExperimentNativeResultEnvelope
	scanner := bufio.NewScanner(strings.NewReader(output))
	scanner.Buffer(make([]byte, 4096), experimentNativeOutputLimit+1)
	found := 0
	for scanner.Scan() {
		line := scanner.Bytes()
		var marker struct {
			Kind string `json:"kind"`
		}
		markerErr := json.Unmarshal(line, &marker)
		if marker.Kind == "EXPERIMENT_DDP21_ALIGNMENT_V1" {
			return fail()
		}
		if marker.Kind != ExperimentNativeResultKind && !bytes.Contains(line, []byte(ExperimentNativeResultKind)) {
			continue
		}
		found++
		if markerErr != nil || found != 1 || len(line) > experimentNativeEnvelopeLimit || experimentDecode(line, &result, true) != nil ||
			!experimentNativeResultFields(line, "kind status mode node rank job_id job_generation contract_sha256 release_sha256 runtime_sha256 model_manifest_sha256 world_size gpu_world_size optimizer_steps local_examples global_unique_examples initial_adapter_sha256 final_adapter_sha256 base_before_sha256 base_after_sha256 all_finite gpu_admission calibration_examples calibration_reference_sha256 reference_probabilities_match result_sha256 telemetry_sha256 minimum_host_available_bytes minimum_cgroup_available_bytes gpus checkpoints") {
			return fail()
		}
		var fields struct {
			GPUs        []json.RawMessage `json:"gpus"`
			Checkpoints []json.RawMessage `json:"checkpoints"`
		}
		if json.Unmarshal(line, &fields) != nil {
			return fail()
		}
		for _, gpu := range fields.GPUs {
			if !experimentNativeResultFields(gpu, "device free_bytes total_bytes allocated_bytes reserved_bytes peak_allocated_bytes peak_reserved_bytes allocator_backend allocator_enabled allocator_stats_valid allocator_fraction") {
				return fail()
			}
		}
		for _, checkpoint := range fields.Checkpoints {
			if !experimentNativeResultFields(checkpoint, "step candidate adapter_sha256 artifact_sha256") {
				return fail()
			}
		}
	}
	if scanner.Err() != nil || found != 1 || result.Kind != ExperimentNativeResultKind || result.Status != "PASSED" ||
		result.Mode != mode || result.Node != node || result.Rank != rank || result.JobID != job.ID || result.Generation != job.Generation ||
		result.ContractSHA256 != request.ContractSHA256 || result.ReleaseSHA256 != job.RuntimeDigest || result.ModelManifestSHA256 != job.ModelDigest ||
		result.WorldSize != len(ids) || result.GPUWorldSize != 2*len(ids) || result.InitialAdapterSHA256 !=
		experiment.config.Recipe.LoRA.ExpectedInitialDigest ||
		result.FinalAdapterSHA256 == result.InitialAdapterSHA256 || result.BaseBeforeSHA256 != result.BaseAfterSHA256 || !result.AllFinite || !result.GPUAdmission ||
		result.MinimumHostAvailableBytes < 6<<30 || result.MinimumCgroupAvailableBytes < 512<<20 || len(result.GPUs) != 2 {
		return fail()
	}
	for _, digest := range []string{result.RuntimeSHA256, result.FinalAdapterSHA256, result.BaseBeforeSHA256,
		result.CalibrationReferenceSHA256, result.ResultSHA256, result.TelemetrySHA256} {
		if !experimentNativeResultHash(digest) {
			return fail()
		}
	}
	for device, gpu := range result.GPUs {
		if gpu.Device != device || gpu.TotalBytes <= 0 || gpu.FreeBytes < 0 || gpu.FreeBytes > gpu.TotalBytes ||
			ValidateExperimentNativeAllocator(gpu) != nil || gpu.AllocatorBackend != "native" ||
			gpu.AllocatedBytes < 0 || gpu.ReservedBytes < gpu.AllocatedBytes || gpu.PeakAllocatedBytes <= 0 || gpu.PeakReservedBytes <= 0 ||
			gpu.PeakAllocatedBytes < gpu.AllocatedBytes || gpu.PeakReservedBytes < gpu.ReservedBytes || gpu.PeakReservedBytes < gpu.PeakAllocatedBytes {
			return fail()
		}
	}
	if mode == "canary" {
		localExamples := 4
		if rank < 4 {
			localExamples++
		}
		if result.OptimizerSteps != 4 || result.LocalExamples != localExamples || result.GlobalUniqueExamples != 4 || result.CalibrationExamples != 8 ||
			!result.ReferenceProbabilitiesMatch || len(result.Checkpoints) != 0 {
			return fail()
		}
	} else {
		// The versioned 21-example batch uses every rank once and assigns the
		// extra example to rank step-1. Across 18 updates ranks 0..17 each
		// process 19 unique examples; the other ranks process 18.
		localExamples := 18
		if rank < 18 {
			localExamples++
		}
		if result.OptimizerSteps != 18 || result.LocalExamples != localExamples || result.GlobalUniqueExamples != 378 || result.CalibrationExamples != 0 || result.ReferenceProbabilitiesMatch {
			return fail()
		}
		if rank != 0 && len(result.Checkpoints) != 0 || rank == 0 && len(result.Checkpoints) != 2 {
			return fail()
		}
		for i, checkpoint := range result.Checkpoints {
			if checkpoint.Step != (i+1)*9 || checkpoint.Candidate != (i == 1) || !experimentNativeResultHash(checkpoint.AdapterSHA256) ||
				!experimentNativeResultHash(checkpoint.ArtifactSHA256) || i == 1 && checkpoint.AdapterSHA256 != result.FinalAdapterSHA256 {
				return fail()
			}
		}
	}
	return result, nil
}

// EmitExperimentNativeResult writes one compact JSON line. The caller must finish
// resource monitoring, persist/sync result and telemetry, and fill their hashes
// before emitting success. This encoder does not manufacture or qualify facts.
func EmitExperimentNativeResult(writer io.Writer, result ExperimentNativeResultEnvelope) error {
	if writer == nil {
		return errors.New("experiment native: result writer required")
	}
	body, err := json.Marshal(result)
	if err != nil {
		return err
	}
	if len(body) > experimentNativeEnvelopeLimit {
		return errors.New("experiment native: result envelope exceeds limit")
	}
	body = append(body, '\n')
	n, err := writer.Write(body)
	if err == nil && n != len(body) {
		err = io.ErrShortWrite
	}
	return err
}

func experimentNativeResultHash(value string) bool {
	return experimentSHA256(value) && value != strings.Repeat("0", 64)
}

func experimentNativeResultFields(body []byte, names string) bool {
	var fields map[string]json.RawMessage
	expected := strings.Fields(names)
	if json.Unmarshal(body, &fields) != nil || len(fields) != len(expected) {
		return false
	}
	for _, name := range expected {
		value, ok := fields[name]
		if !ok || bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			return false
		}
	}
	return true
}
