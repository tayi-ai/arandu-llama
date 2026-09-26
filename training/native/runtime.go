package native

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime/debug"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/tayi-ai/arandu-llama/checkpoint"
	decoder "github.com/tayi-ai/arandu-llama/training/decoder"
	"github.com/tayi-ai/arandu-llama/training/sequence"
	"github.com/tayi-ai/arandu-llama/training/tokenizer"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

// The embedding input, every decoder boundary and the final activation remain
// FP16 on the T10 path qualified by the R6 forward calibration. The count is
// the 32 decoder inputs plus the final activation and one conservative live
// boundary retained while vocabulary projection completes.
const experimentNativeCheckpointBytesPerElement = 2 * (32 + 2)

// ExperimentNativeMemoryPlan separates admission before loading from the free host
// reserve during execution. Tensor element budgets conservatively count a
// phase's intermediates, not simultaneous physical allocations. Only measured
// allocator peaks can qualify GPU fit; the allocator cap remains 0.80.
type ExperimentNativeMemoryPlan struct {
	MaximumSequenceLength int64 `json:"maximum_sequence_length"`
	LargestTensorBytes    int64 `json:"largest_tensor_bytes"`
	TensorCopyBytes       int64 `json:"tensor_copy_bytes"`
	HostOverheadBytes     int64 `json:"host_overhead_bytes"`
	HostReserveBytes      int64 `json:"host_reserve_bytes"`
	HostPreloadBytes      int64 `json:"host_preload_bytes"`
	CheckpointBytes       int64 `json:"checkpoint_bytes"`
	WorkingElements       int64 `json:"working_elements"`
	SequenceElements      int64 `json:"sequence_elements"`
}

// ExperimentNativeMemory fixes the observed corpus maximum (1128 plus appended A),
// the largest 248320x4096 FP16 embedding/head and the explicit three-copy loader.
// Two GiB cover Go/tokenizer/header/allocator overhead separately from the six
// GiB host reserve. This is a conservative admission, not an observed RSS peak.
func ExperimentNativeMemory(maximumSequenceLength int64) (ExperimentNativeMemoryPlan, error) {
	if maximumSequenceLength < 2 || maximumSequenceLength > 1129 {
		return ExperimentNativeMemoryPlan{}, errors.New("experiment native: sequence exceeds the frozen corpus admission")
	}
	t := maximumSequenceLength
	state, values, queries := int64(32*128*128), t*32*128, t*32*128
	chunks := (t + 7) / 8
	forward := 3*values + (chunks+1)*state
	reverse := 2*values + (chunks+3)*state + 2*queries + values + 2*t*32
	concatenate := 2*values + 3*state + max(values, queries) + 2*queries + values + 2*t*32
	plan := ExperimentNativeMemoryPlan{MaximumSequenceLength: t, LargestTensorBytes: 248320 * 4096 * 2,
		HostOverheadBytes: 2 << 30, HostReserveBytes: 6 << 30, CheckpointBytes: t * 4096 * experimentNativeCheckpointBytesPerElement,
		WorkingElements:  32*t*4096 + 32*t*8192 + 32*t*4096 + 64*t*32*128 + 32*t*32 + 4*state,
		SequenceElements: max(forward, reverse, concatenate)}
	plan.TensorCopyBytes = 3 * plan.LargestTensorBytes
	plan.HostPreloadBytes = plan.TensorCopyBytes + plan.HostOverheadBytes + plan.HostReserveBytes
	return plan, nil
}

func (p ExperimentNativeMemoryPlan) assemblyLimits(experiment *NativeExperiment) decoder.AssemblyLimits {
	limits := experiment.config.Assembly
	limits.PersistentBytes = append([]int64(nil), limits.PersistentBytes...)
	limits.DeviceByLayer = append([]int(nil), limits.DeviceByLayer...)
	limits.HeaderLimits = checkpoint.DefaultLimits()
	limits.TensorCopyBytes = p.TensorCopyBytes
	limits.HashChunkBytes = 4 << 20
	limits.MaxInputElements = p.MaximumSequenceLength * 4096
	limits.MaxScoreElements = 16 * p.MaximumSequenceLength * p.MaximumSequenceLength
	limits.MaxWorkingElements = p.WorkingElements
	limits.Sequence = sequence.SequenceLimits{ChunkTokens: 8, MaxTokens: p.MaximumSequenceLength, MaxOwnedElements: p.SequenceElements}
	return limits
}

type experimentNativeResources struct {
	At                   time.Time               `json:"at"`
	Phase                string                  `json:"phase"`
	HostAvailableBytes   int64                   `json:"host_available_bytes"`
	CgroupAvailableBytes int64                   `json:"cgroup_available_bytes"`
	GPUs                 []torch.CUDAMemoryStats `json:"gpus,omitempty"`
}

type experimentNativeMonitor struct {
	mu            sync.Mutex
	file          *os.File
	resources     RuntimeConfig
	minimum       int64
	minimumCgroup int64
	gpus          []torch.CUDAMemoryStats
}

// ValidateExperimentNativeAllocator checks the effective byte limit and measured peak.
// LibTorch 2.14 reports round(fraction * total) / total, rather than the original
// fraction. Compare its reconstructed byte limit with the unchanged floor(0.80
// * total) ceiling; a rounded limit or measured peak above that ceiling fails.
func ValidateExperimentNativeAllocator(stats torch.CUDAMemoryStats) error {
	if !stats.AllocatorEnabled || !stats.AllocatorStatsValid || stats.TotalBytes <= 0 || stats.TotalBytes > 1<<53 ||
		math.IsNaN(stats.AllocatorFraction) || math.IsInf(stats.AllocatorFraction, 0) ||
		stats.AllocatorFraction <= 0 || stats.AllocatorFraction > 1 || stats.PeakReservedBytes < 0 {
		return errors.New("experiment native: allocator guard failed")
	}
	ceiling := int64(float64(stats.TotalBytes) * .80)
	allocatorLimit := math.Round(stats.AllocatorFraction * float64(stats.TotalBytes))
	if allocatorLimit != float64(ceiling) || stats.PeakReservedBytes > ceiling {
		return errors.New("experiment native: allocator guard failed")
	}
	return nil
}

func (m *experimentNativeMonitor) sample(phase string, cuda bool) error {
	config := m.resources
	available, err := config.HostAvailableBytes()
	if err != nil {
		return err
	}
	cgroup, err := config.CgroupAvailableBytes()
	if err != nil {
		return err
	}
	sample := experimentNativeResources{At: time.Now().UTC(), Phase: phase, HostAvailableBytes: available, CgroupAvailableBytes: cgroup}
	var guardErr error
	if cuda {
		for device := 0; device < 2; device++ {
			stats, err := torch.CUDAMemory(device)
			if err != nil {
				return err
			}
			sample.GPUs = append(sample.GPUs, stats)
			if err := ValidateExperimentNativeAllocator(stats); err != nil {
				guardErr = errors.Join(guardErr, err)
			}
		}
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.minimum == 0 || available < m.minimum {
		m.minimum = available
	}
	if m.minimumCgroup == 0 || cgroup < m.minimumCgroup {
		m.minimumCgroup = cgroup
	}
	if len(sample.GPUs) == 2 {
		m.gpus = slices.Clone(sample.GPUs)
	}
	if err = json.NewEncoder(m.file).Encode(sample); err != nil {
		return err
	}
	if err = m.file.Sync(); err != nil {
		return err
	}
	if available < 6<<30 || cgroup < 512<<20 {
		guardErr = errors.Join(guardErr, errors.New("experiment native: host or container memory reserve exhausted"))
	}
	return guardErr
}

type experimentNativeShard struct {
	*os.File
	size int64
}

func (s *experimentNativeShard) Size() int64 { return s.size }

type experimentNativeSources struct {
	root  string
	files map[string]experimentNativeSource
}
type experimentNativeSource struct {
	Path   string `json:"path"`
	Bytes  int64  `json:"bytes"`
	SHA256 string `json:"sha256"`
}

func (s *experimentNativeSources) OpenShard(ctx context.Context, name string) (decoder.Shard, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	entry, ok := s.files[name]
	if !ok || filepath.Base(name) != name || !strings.HasSuffix(name, ".safetensors") {
		return nil, errors.New("experiment native: unadmitted shard")
	}
	path := filepath.Join(s.root, name)
	before, err := os.Lstat(path)
	if err != nil || !before.Mode().IsRegular() || before.Size() != entry.Bytes {
		return nil, errors.New("experiment native: invalid shard file")
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	actual, err := f.Stat()
	if err != nil || !os.SameFile(before, actual) {
		_ = f.Close()
		return nil, errors.New("experiment native: shard changed while opening")
	}
	return &experimentNativeShard{File: f, size: entry.Bytes}, nil
}

func experimentNativeVerifySources(experiment *NativeExperiment, ctx context.Context, root string, body []byte) (*experimentNativeSources, error) {
	if experiment == nil {
		return nil, errors.New("native experiment: installation is not configured")
	}

	if experimentHash(body) !=
		experiment.config.Recipe.Model.ManifestSHA256 ||
		experimentDirectory(root) != nil {
		return nil, errors.New("experiment native: frozen model manifest or root differs")
	}
	var manifest struct {
		Files []experimentNativeSource `json:"files"`
	}
	if err := json.Unmarshal(body, &manifest); err != nil || len(manifest.Files) != 18 {
		return nil, errors.New("experiment native: frozen model inventory differs")
	}
	s := &experimentNativeSources{root: root, files: make(map[string]experimentNativeSource)}
	buffer := make([]byte, 4<<20)
	for _, entry := range manifest.Files {
		if !filepath.IsLocal(entry.Path) || filepath.Clean(entry.Path) != entry.Path || entry.Bytes <= 0 || !experimentSHA256(entry.SHA256) {
			return nil, errors.New("experiment native: unsafe source entry")
		}
		path := filepath.Join(root, entry.Path)
		if experimentDirectory(filepath.Dir(path)) != nil {
			return nil, errors.New("experiment native: unsafe source directory")
		}
		before, err := os.Lstat(path)
		if err != nil || !before.Mode().IsRegular() || before.Size() != entry.Bytes {
			return nil, errors.New("experiment native: source file missing or changed")
		}
		f, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		opened, statErr := f.Stat()
		if statErr != nil || !os.SameFile(before, opened) {
			_ = f.Close()
			return nil, errors.New("experiment native: source changed while opening")
		}
		h := sha256.New()
		var read int64
		for {
			if err = ctx.Err(); err != nil {
				break
			}
			var n int
			n, err = f.Read(buffer)
			if n > 0 {
				_, _ = h.Write(buffer[:n])
				read += int64(n)
			}
			if err != nil {
				break
			}
		}
		after, statErr := f.Stat()
		closeErr := f.Close()
		if !errors.Is(err, io.EOF) || statErr != nil || closeErr != nil || after.Size() != before.Size() || !after.ModTime().Equal(before.ModTime()) || read != entry.Bytes || hex.EncodeToString(h.Sum(nil)) != entry.SHA256 {
			return nil, errors.New("experiment native: source integrity verification failed")
		}
		s.files[entry.Path] = entry
	}
	return s, nil
}

// RunExperimentNative executes one admitted native invocation. The application
// supplies identity, resources and an authenticated collective transport.
// It creates no daemon and performs no download, retry, fallback or promotion.
func RunExperimentNative(experiment *NativeExperiment, parent context.Context, config RuntimeConfig) (err error) {
	config = config.snapshot()
	mode, contractPath, out := config.Mode, config.ContractPath, config.OutputDirectory
	if experiment == nil {
		return errors.New("native experiment: installation is not configured")
	}

	if !RuntimeAvailable() {
		return ErrExperimentNativeBackendUnqualified
	}
	root := filepath.Dir(filepath.Dir(contractPath))
	if contractPath != filepath.Join(root, experiment.config.BundleName, "contract.json") {
		return errors.New("experiment native: fixed contract path required")
	}
	if err := config.validate(parent); err != nil {
		return err
	}
	release := config.Release
	bundle, err := experimentLoadNativeBundle(experiment, root, mode, release)
	if err != nil {
		return err
	}
	admission, err := experimentNativeAdmission(experiment, bundle.contract, bundle.bytes, mode, release)
	if err != nil {
		return err
	}
	if err = admission.ValidateStart(time.Now()); err != nil {
		return err
	}
	job := admission.Job
	if !sameJob(config.Job, job) || config.ContractSHA256 != experimentHash(bundle.bytes) || config.RuntimeSHA256 != bundle.manifest.RuntimeSHA256 {
		return errors.New("experiment native: supervisor identity differs")
	}
	if out != filepath.Join(root, "runs", job.ID, mode) || experimentDirectory(filepath.Dir(out)) != nil {
		return errors.New("experiment native: output must belong to the admitted job")
	}
	digest, err := experimentNativeFileHash(config.ExecutablePath, 512<<20, true)
	if err != nil || digest != bundle.manifest.RuntimeSHA256 {
		return errors.New("experiment native: executable identity differs")
	}
	rank, world := config.Rank, bundle.contract.Admission.ProcessWorldSize
	if rank < 0 || rank >= len(admission.Nodes) || config.NodeID != admission.Nodes[rank] || config.World != world || !slices.Equal(config.GPUIds, []int{0, 1}) || config.VRAMFraction != .80 {
		return errors.New("experiment native: supervisor topology differs")
	}

	ctx, cancel := context.WithTimeout(parent, admission.Deadline)
	defer cancel()
	ctx, abort := context.WithCancelCause(ctx)
	defer abort(nil)
	if err = os.Mkdir(out, 0700); err != nil {
		return errors.New("experiment native: output already exists or cannot be created")
	}
	log, err := os.OpenFile(filepath.Join(out, "resources.jsonl"), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	defer log.Close()
	monitor := &experimentNativeMonitor{file: log, resources: config}
	result := map[string]any{"schema_version": 2, "mode": mode, "job_id": job.ID, "node": admission.Nodes[rank], "rank": rank, "world": world, "contract_sha256": experimentHash(bundle.bytes), "runtime_sha256": bundle.manifest.RuntimeSHA256, "completed": false, "optimizer_steps": 0}
	outcome := ExperimentNativeResultEnvelope{Kind: ExperimentNativeResultKind, Status: "FAILED", Mode: mode, Node: admission.Nodes[rank], Rank: rank, JobID: job.ID, Generation: job.Generation,
		ContractSHA256: experimentHash(bundle.bytes), ReleaseSHA256: release.ManifestSHA256, RuntimeSHA256: bundle.manifest.RuntimeSHA256, ModelManifestSHA256: experiment.config.Recipe.Model.ManifestSHA256,

		WorldSize: world, GPUWorldSize: world * bundle.contract.Admission.GPUsPerNode, CalibrationReferenceSHA256: bundle.manifest.Files["calibration-reference.json"], Checkpoints: []ExperimentNativeCheckpointReceipt{}}
	defer func() {
		if cause := context.Cause(ctx); cause != nil {
			err = errors.Join(err, cause)
		}
		if err != nil {
			result["error"] = err.Error()
			result["completed"] = false
		}
		result["finished_at"] = time.Now().UTC()
		monitor.mu.Lock()
		result["minimum_host_available_bytes"] = monitor.minimum
		result["minimum_cgroup_available_bytes"] = monitor.minimumCgroup
		outcome.MinimumHostAvailableBytes = monitor.minimum
		outcome.MinimumCgroupAvailableBytes = monitor.minimumCgroup
		outcome.GPUs = slices.Clone(monitor.gpus)
		monitor.mu.Unlock()
		err = errors.Join(err, experimentNativeWriteJSON(filepath.Join(out, "result.json"), result))
		if err == nil {
			outcome.ResultSHA256, err = experimentNativeFileHash(filepath.Join(out, "result.json"), 16<<20, false)
			if err == nil {
				outcome.TelemetrySHA256, err = experimentNativeFileHash(filepath.Join(out, "resources.jsonl"), 16<<20, false)
			}
		}
		if err == nil {
			outcome.Status = "PASSED"
			outcome.GPUAdmission = true
			outcome.AllFinite = true
		}
		if err == nil {
			body, marshalErr := json.Marshal(outcome)
			if marshalErr == nil {
				_, marshalErr = ParseExperimentNativeResult(experiment, string(body), job, admission.Nodes[rank], rank, mode)
			}
			err = marshalErr
		}
		if err != nil {
			outcome.Status = "FAILED"
		}
		err = errors.Join(err, EmitExperimentNativeResult(config.Output, outcome))
	}()
	if err = monitor.sample("preflight", false); err != nil {
		return err
	}
	stopMonitor := make(chan struct{})
	monitorDone := make(chan struct{})
	defer func() { close(stopMonitor); <-monitorDone }()
	go func() {
		defer close(monitorDone)
		ticker := time.NewTicker(2 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-stopMonitor:
				return
			case <-ctx.Done():
				return
			case <-ticker.C:
				if failure := monitor.sample("host_watch", false); failure != nil {
					abort(failure)
					return
				}
			}
		}
	}()
	path := filepath.Join(root, experiment.config.BundleName)
	read := func(name string) ([]byte, error) {
		body, e := experimentReadFile(filepath.Join(path, name), 32<<20)
		if e != nil || experimentHash(body) != bundle.manifest.Files[name] {
			return nil, errors.New("experiment native: admitted input changed")
		}
		return body, nil
	}
	trainingBytes, err := read("training.json")
	if err != nil {
		return err
	}
	data, err := ReadExperimentTraining(experiment, strings.NewReader(string(trainingBytes)))
	if err != nil {
		return err
	}
	codecBytes, err := read("tokenizer.json")
	if err != nil {
		return err
	}
	codec, err := tokenizer.Load(ctx, strings.NewReader(string(codecBytes)), bundle.manifest.Files["tokenizer.json"], tokenizer.DefaultLimits())
	if err != nil {
		return err
	}
	order, err := ExperimentTrainingOrder(experiment, data.Examples)
	if err != nil {
		return err
	}
	maximum := int64(0)
	longest := ExperimentTrainingRow{}
	for _, row := range order {
		prompt, e := RenderExperimentTrainingPrompt(row, data.Demos)
		if e != nil {
			return e
		}
		ids, e := codec.Encode(ctx, prompt)
		if e != nil {
			return e
		}
		n := int64(len(ids) + 1)
		if n > maximum || (n == maximum && row.ExampleID < longest.ExampleID) {
			maximum = n
			longest = row
		}
	}
	memory, err := ExperimentNativeMemory(maximum)
	if err != nil {
		return err
	}
	result["memory_plan"] = memory
	available, err := config.HostAvailableBytes()
	if err != nil {
		return err
	}
	headroom, err := config.CgroupAvailableBytes()
	if err != nil {
		return err
	}
	if available < memory.HostPreloadBytes || headroom < memory.TensorCopyBytes+memory.HostOverheadBytes+(512<<20) {
		return errors.New("experiment native: preload RAM admission failed for tensor copies, overhead and fixed reserve")
	}
	probeCtx, probeCancel := context.WithTimeout(ctx, 15*time.Second)
	probe, err := config.ProbeGPUs(probeCtx, .80)
	probeCancel()
	if err != nil {
		return err
	}
	result["driver_admission"] = probe
	if len(probe.GPUs) != 2 {
		return errors.New("experiment native: two measured GPUs required")
	}
	for device, gpu := range probe.GPUs {
		if gpu.Index != device || gpu.Capability != "7.5" || !strings.Contains(gpu.Name, "T10") || gpu.FreeBytes < gpu.CapBytes+(512<<20) || memory.assemblyLimits(experiment).PersistentBytes[device]+(512<<20) > gpu.CapBytes {
			return errors.New("experiment native: measured GPU capacity or SM75 identity rejected")
		}
	}
	index, err := read("model.safetensors.index.json")
	if err != nil {
		return err
	}
	modelConfig, err := read("config.json")
	if err != nil {
		return err
	}
	reference, err := read("initial-reference.json")
	if err != nil {
		return err
	}
	plan, err := decoder.PlanTextAssembly(index, modelConfig, reference, decoder.AssemblyIdentity{IndexSHA256: bundle.manifest.Files["model.safetensors.index.json"], ConfigSHA256: bundle.manifest.Files["config.json"], ReferenceSHA256: bundle.manifest.Files["initial-reference.json"], InitialAdapterSHA256: experiment.config.Recipe.LoRA.ExpectedInitialDigest}, memory.assemblyLimits(experiment))
	if err != nil {
		return err
	}
	if err = validateNativeRotary(experiment, plan); err != nil {
		return err
	}
	baseBytes, err := read("base-manifest.json")
	if err != nil {
		return err
	}
	sources, err := experimentNativeVerifySources(experiment, ctx, bundle.contract.BasePath, baseBytes)
	if err != nil {
		return err
	}
	inspection, err := decoder.InspectAssemblySources(ctx, plan, sources)
	if err != nil {
		return err
	}
	result["source_inspection"] = inspection
	if err = experimentNativeBarrier(experiment, config, ctx, rank, world, job.ID+"-preload", experimentHash(bundle.bytes)); err != nil {
		return err
	}
	count, err := torch.CUDADeviceCount()
	if err != nil || count != 2 {
		return errors.New("experiment native: CUDA device count differs")
	}
	for device := 0; device < 2; device++ {
		if err = torch.SetCUDAMemoryFraction(device, .80); err != nil {
			return err
		}
	}
	if err = monitor.sample("allocator_admitted", true); err != nil {
		return err
	}
	// Bound Go payload retention between large checkpoint tensors. Native
	// allocations are admitted independently; this is not a native RSS limit.
	previousLimit := debug.SetMemoryLimit(2 << 30)
	defer debug.SetMemoryLimit(previousLimit)
	initial, err := decoder.InitializeAdapter(ctx, experiment.config.Initializer)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, initial.Close()) }()
	loaded, err := decoder.LoadTextAssembly(ctx, plan, sources, initial)
	if err != nil {
		return err
	}
	if err = monitor.sample("model_loaded", true); err != nil {
		_ = loaded.Close()
		return err
	}
	replica, err := NewExperimentDecoderReplica(experiment, loaded, codec, plan, ExperimentDecoderReplicaOptions{Limits: decoder.Limits{MaxTokens: maximum, LogitRows: 2, MaxCheckpointBytes: memory.CheckpointBytes}, HashChunkBytes: 4 << 20, ParameterCopyBytes: 557056 * 4 * 3, SourceManifestSHA256: experiment.config.Recipe.Model.ManifestSHA256, ExpectedRotaryFrequencySHA256: experiment.config.RotarySHA256})
	if err != nil {
		_ = loaded.Close()
		return err
	}
	defer func() { err = errors.Join(err, replica.Close()) }()
	initialReceipt, err := replica.Inspect(ctx)
	if err != nil {
		return err
	}
	result["initial"] = initialReceipt
	outcome.InitialAdapterSHA256 = initialReceipt.AdapterSHA256
	outcome.BaseBeforeSHA256 = initialReceipt.BaseSHA256
	if err = experimentNativeBarrier(experiment, config, ctx, rank, world, job.ID+"-loaded", experimentHash(bundle.bytes)); err != nil {
		return err
	}
	if mode == "canary" {
		calibration, readErr := read("calibration-reference.json")
		if readErr != nil {
			return readErr
		}
		if err = experimentNativeCalibrate(ctx, replica, order, data.Demos, calibration, out, monitor); err != nil {
			return err
		}
		result["calibration_examples"] = 8
		result["calibration_reference_sha256"] = bundle.manifest.Files["calibration-reference.json"]
		result["reference_scores_match"] = true
		outcome.CalibrationExamples = 8
		outcome.ReferenceProbabilitiesMatch = true
	}
	parameters, err := replica.InitialParameters(ctx)
	if err != nil {
		return err
	}
	optimizer, err := NewExperimentAdamW(parameters)
	if err != nil {
		return err
	}
	clear(parameters)
	steps := uint32(bundle.contract.Admission.OptimizerSteps)
	if mode == "canary" {
		steps = 4
	}
	layoutBytes, _ := json.Marshal(initialReceipt.Layout)
	update, err := ExperimentNativeAdamWUpdate(experiment, optimizer)
	if err != nil {
		return err
	}
	exchange, err := experimentNativeExchange(experiment, config, ctx, rank, world, job.ID+"-gradient", experimentHash(bundle.bytes), experimentHash(layoutBytes), steps, 557056, update)
	if err != nil {
		return err
	}
	defer exchange.Close()
	if mode == "train" {
		var writer *ExperimentNativeFileCheckpointWriter
		if rank == 0 {
			writer, err = NewExperimentNativeCheckpointWriter(experiment, out, optimizer)
			if err != nil {
				return err
			}
			defer writer.Close()
		}
		training, err := NewExperimentNativeTraining(experiment, rank, world, replica, exchange, writer, func(ctx context.Context, p ExperimentNativeProgress) error {
			if err := monitor.sample("update_installed", true); err != nil {
				return err
			}
			return experimentNativeWriteJSON(filepath.Join(out, fmt.Sprintf("step-%02d.json", p.Step)), p)
		})
		if err != nil {
			return err
		}
		trained, err := training.RunFrozen(ctx, strings.NewReader(string(trainingBytes)))
		result["training"] = trained
		result["optimizer_steps"] = trained.InstalledUpdates
		outcome.OptimizerSteps = trained.InstalledUpdates
		outcome.LocalExamples = trained.LocalExamples
		outcome.GlobalUniqueExamples = trained.GlobalScheduledExamples
		outcome.FinalAdapterSHA256 = trained.Final.AdapterSHA256
		outcome.BaseAfterSHA256 = trained.Final.BaseSHA256
		if rank == 0 {
			outcome.Checkpoints = trained.Checkpoints
		}
		if err != nil {
			return err
		}
	} else {
		rows := []ExperimentTrainingRow{longest}
		for _, row := range order {
			if row.ExampleID != longest.ExampleID && len(rows) < 4 {
				rows = append(rows, row)
			}
		}
		for step, row := range rows {
			prompt, e := RenderExperimentTrainingPrompt(row, data.Demos)
			if e != nil {
				return e
			}
			contributions := 1
			if rank == step {
				contributions++
			}
			gradient, losses, e := experimentNativeCanaryGradient(experiment, ctx, replica, row, prompt, contributions)
			if e != nil {
				return e
			}
			next, e := exchange.Exchange(ctx, uint32(step+1), gradient.ValuesF32)
			if e != nil {
				return e
			}
			installed, e := replica.Install(ctx, next)
			if e != nil {
				return e
			}
			result["optimizer_steps"] = step + 1
			outcome.OptimizerSteps = step + 1
			outcome.LocalExamples += contributions
			outcome.GlobalUniqueExamples = 4
			if err = monitor.sample("canary_update_installed", true); err != nil {
				return err
			}
			if err = experimentNativeWriteJSON(filepath.Join(out, fmt.Sprintf("step-%02d.json", step+1)), map[string]any{"step": step + 1, "example_id": row.ExampleID, "loss": losses[0], "contribution_losses": losses, "local_contributions": contributions, "local_examples": outcome.LocalExamples, "adapter_sha256": installed, "sequence_length": gradient.InputSequenceLength}); err != nil {
				return err
			}
		}
		final, e := replica.Inspect(ctx)
		if e != nil {
			return e
		}
		if final.BaseSHA256 != initialReceipt.BaseSHA256 || final.AdapterSHA256 == initialReceipt.AdapterSHA256 {
			return errors.New("experiment native: canary base or adapter transition failed")
		}
		result["final"] = final
		outcome.FinalAdapterSHA256 = final.AdapterSHA256
		outcome.BaseAfterSHA256 = final.BaseSHA256
	}
	if err = exchange.Close(); err != nil {
		return err
	}
	if _, err = experimentNativeVerifySources(experiment, ctx, bundle.contract.BasePath, baseBytes); err != nil {
		return err
	}
	if err = monitor.sample("completed", true); err != nil {
		return err
	}
	if err = experimentNativeBarrier(experiment, config, ctx, rank, world, job.ID+"-completed", experimentHash(bundle.bytes)); err != nil {
		return err
	}
	result["gpu_admission"] = true
	result["completed"] = true
	return nil
}

// The canary repeats the same measured example across the global batch. The
// assigned rank computes its extra contribution sequentially before install;
// local sums are neither averaged nor unscaled before the global optimizer.
func experimentNativeCanaryGradient(experiment *NativeExperiment, ctx context.Context, replica ExperimentNativeReplica, row ExperimentTrainingRow, prompt string, contributions int) (ExperimentNativeGradient, []ExperimentLoss, error) {
	if experiment == nil {
		return ExperimentNativeGradient{}, nil, errors.New("native experiment: installation is not configured")
	}

	fail := func() (ExperimentNativeGradient, []ExperimentLoss, error) {
		return ExperimentNativeGradient{}, nil, errors.New("experiment native: invalid canary gradient contributions")
	}
	if ctx == nil || ctx.Err() != nil || replica == nil || contributions < 1 || contributions > 2 {
		return fail()
	}
	var combined ExperimentNativeGradient
	losses := make([]ExperimentLoss, 0, contributions)
	for contribution := 0; contribution < contributions; contribution++ {
		if err := ctx.Err(); err != nil {
			return ExperimentNativeGradient{}, nil, err
		}
		var loss ExperimentLoss
		calls := 0
		gradient, err := replica.Gradient(ctx, ExperimentNativeExample{ExampleID: row.ExampleID, Prompt: prompt, CandidateVocabularyIDs: [4]int{357, 417, 351, 414}, LogitRows: 2}, func(logits [4]float64) ([4]float64, error) {
			calls++
			var failure error
			loss, failure = MultiTeacherLoss(experiment, logits, row)
			derivative := loss.LogitGradient
			for i := range derivative {
				derivative[i] *= ExperimentAdamWLossScale
			}
			return derivative, failure
		})
		if err != nil {
			return ExperimentNativeGradient{}, nil, err
		}
		if calls != 1 || len(gradient.ValuesF32) == 0 || gradient.InputSequenceLength < 1 {
			return fail()
		}
		if contribution == 0 {
			combined = gradient
			combined.ValuesF32 = slices.Clone(gradient.ValuesF32)
		} else {
			if len(gradient.ValuesF32) != len(combined.ValuesF32) || gradient.InputSequenceLength != combined.InputSequenceLength {
				return fail()
			}
			for i, value := range gradient.ValuesF32 {
				if !experimentFinite32(value) {
					return fail()
				}
				combined.ValuesF32[i] += value
			}
		}
		for _, value := range combined.ValuesF32 {
			if !experimentFinite32(value) {
				return fail()
			}
		}
		losses = append(losses, loss)
	}
	if err := ctx.Err(); err != nil {
		return ExperimentNativeGradient{}, nil, err
	}
	return combined, losses, nil
}

func experimentNativeWriteJSON(path string, value any) error {
	body, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	body = append(body, '\n')
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	_, writeErr := f.Write(body)
	syncErr := f.Sync()
	if err := errors.Join(writeErr, syncErr, f.Close()); err != nil {
		return err
	}
	directory, err := os.Open(filepath.Dir(path))
	if err != nil {
		return err
	}
	return errors.Join(directory.Sync(), directory.Close())
}

func experimentNativeBarrier(experiment *NativeExperiment, config RuntimeConfig, ctx context.Context, rank, world int, session, contract string) error {
	if experiment == nil {
		return errors.New("native experiment: installation is not configured")
	}

	layout := experimentHash([]byte("tayi-native-admission-one-f32-v1"))
	exchange, err := experimentNativeExchange(experiment, config, ctx, rank, world, session, contract, layout, 1, 1, func(values [][]float32) ([]float32, error) {
		if len(values) != world {
			return nil, errors.New("experiment native: incomplete admission")
		}
		for _, value := range values {
			if !slices.Equal(value, []float32{1}) {
				return nil, errors.New("experiment native: rank admission failed")
			}
		}
		return []float32{1}, nil
	})
	if err != nil {
		return err
	}
	defer exchange.Close()
	_, err = exchange.Exchange(ctx, 1, []float32{1})
	return err
}

func experimentNativeCalibrate(ctx context.Context, replica *ExperimentDecoderReplica, order, demos []ExperimentTrainingRow, body []byte, out string, monitor *experimentNativeMonitor) error {
	var reference struct {
		Status    string  `json:"status"`
		Tolerance float64 `json:"tolerance"`
		RowSource string  `json:"row_source"`
		Rows      []struct {
			ExampleID    string `json:"example_id"`
			PromptLength int64  `json:"prompt_tokens"`
			Checks       []struct {
				Label     string             `json:"label"`
				Reference map[string]float64 `json:"reference"`
			} `json:"checks"`
		} `json:"rows"`
	}
	if json.Unmarshal(body, &reference) != nil || reference.Status != "PASSED" || reference.Tolerance != .03 || reference.RowSource != "first_eight_training_order" || len(reference.Rows) != 8 {
		return errors.New("experiment native: frozen calibration reference differs")
	}
	for i, expected := range reference.Rows {
		row := order[i]
		if row.ExampleID != expected.ExampleID {
			return errors.New("experiment native: calibration example differs")
		}
		var scores map[string]float64
		for _, check := range expected.Checks {
			if check.Label == "wrapper_disabled" {
				scores = check.Reference
			}
		}
		if len(scores) != 4 {
			return errors.New("experiment native: calibration scores missing")
		}
		prompt, err := RenderExperimentTrainingPrompt(row, demos)
		if err != nil {
			return err
		}
		trace, err := NewExperimentNativeForwardTrace(out, i)
		if err != nil {
			return err
		}
		measured, calibrationErr := replica.CalibrationObserved(ctx, ExperimentNativeExample{ExampleID: row.ExampleID, Prompt: prompt, CandidateVocabularyIDs: [4]int{357, 417, 351, 414}, LogitRows: 2}, trace.Observe)
		traceReceipt, traceErr := trace.Close()
		if err = errors.Join(calibrationErr, traceErr); err != nil {
			return err
		}
		if err != nil {
			return err
		}
		var referenceValues, differences [4]float64
		bestReference, bestMeasured := 0, 0
		passed := measured.InputSequenceLength == expected.PromptLength+1
		for index, letter := range []string{"A", "B", "C", "D"} {
			value, ok := scores[letter]
			if !ok || math.IsNaN(value) || math.IsInf(value, 0) {
				return errors.New("experiment native: nonfinite calibration reference")
			}
			referenceValues[index] = value
			differences[index] = math.Abs(value - measured.LogProbabilities[index])
			passed = passed && differences[index] <= .03 && !math.IsNaN(measured.LogProbabilities[index]) && !math.IsInf(measured.LogProbabilities[index], 0)
			if referenceValues[index] > referenceValues[bestReference] {
				bestReference = index
			}
			if measured.LogProbabilities[index] > measured.LogProbabilities[bestMeasured] {
				bestMeasured = index
			}
		}
		passed = passed && bestReference == bestMeasured
		if err = experimentNativeWriteJSON(filepath.Join(out, fmt.Sprintf("calibration-%02d.json", i)), map[string]any{"example_id": row.ExampleID, "reference": referenceValues, "measured": measured, "absolute_differences": differences, "tolerance": .03, "passed": passed, "forward_trace": traceReceipt}); err != nil {
			return err
		}
		if !passed {
			return errors.New("experiment native: full-vocabulary score calibration failed")
		}
		if err = monitor.sample("calibration_passed", true); err != nil {
			return err
		}
	}
	return nil
}
