//go:build libtorch && libtorch_cuda

package native

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strings"
	"testing"
	"time"

	decoder "github.com/tayi-ai/arandu-llama/training/decoder"
	"github.com/tayi-ai/arandu-llama/training/layers"
	"github.com/tayi-ai/arandu-llama/training/loading"
	"github.com/tayi-ai/arandu-llama/training/tokenizer"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

// This opt-in measurement never starts a fleet job or installs an optimizer update.
func TestExperimentDecoder16FrozenForwardDiagnosis(t *testing.T) {
	root := os.Getenv("TAYI_EXPERIMENT_DECODER_PROBE_ROOT")
	if root == "" {
		t.Skip("requires explicitly admitted two-GPU diagnostic fixture")
	}
	experiment, evidence := privateNativeEvidence(t)
	ctx, cancel := context.WithTimeout(context.Background(), 12*time.Minute)
	defer cancel()
	check := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
	}
	body, err := os.ReadFile(filepath.Join(root, "manifest.json"))
	check(err)
	if experimentHash(body) != evidence.DiagnosticRelease {
		t.Fatal("diagnostic release differs")
	}
	var manifest ExperimentNativeManifest
	check(json.Unmarshal(body, &manifest))
	read := func(name string) []byte {
		b, e := os.ReadFile(filepath.Join(root, name))
		check(e)
		if experimentHash(b) != manifest.Files[name] {
			t.Fatalf("input identity differs: %s", name)
		}
		return b
	}
	memory, err := ExperimentNativeMemory(1129)
	check(err)
	available, err := diagnosticHostAvailableRAM("/proc/meminfo")
	check(err)
	headroom, err := experimentNativeCgroupAvailable()
	check(err)
	if available < memory.HostPreloadBytes || headroom < memory.TensorCopyBytes+memory.HostOverheadBytes+(512<<20) {
		t.Fatal("diagnostic RAM admission refused")
	}
	probe, err := diagnosticProbeGPUs(ctx, .80)
	check(err)
	if len(probe.GPUs) != 2 {
		t.Fatal("diagnostic requires two measured GPUs")
	}
	for i, gpu := range probe.GPUs {
		if gpu.Index != i || gpu.Capability != "7.5" || !strings.Contains(gpu.Name, "T10") || gpu.FreeBytes < gpu.CapBytes+(512<<20) {
			t.Fatal("diagnostic GPU admission refused")
		}
		check(torch.SetCUDAMemoryFraction(i, .80))
	}
	previous := debug.SetMemoryLimit(2 << 30)
	defer debug.SetMemoryLimit(previous)
	plan, err := decoder.PlanTextAssembly(read("model.safetensors.index.json"), read("config.json"), read("initial-reference.json"), decoder.AssemblyIdentity{IndexSHA256: manifest.Files["model.safetensors.index.json"], ConfigSHA256: manifest.Files["config.json"], ReferenceSHA256: manifest.Files["initial-reference.json"], InitialAdapterSHA256: experiment.config.Recipe.LoRA.ExpectedInitialDigest}, memory.assemblyLimits(experiment))
	check(err)
	sources, err := experimentNativeVerifySources(experiment, ctx, experiment.config.BasePath, read("base-manifest.json"))
	check(err)
	initial, err := decoder.InitializeAdapter(ctx, experiment.config.Initializer)
	check(err)
	defer initial.Close()
	loaded, err := decoder.LoadTextAssembly(ctx, plan, sources, initial)
	check(err)
	codec, err := tokenizer.Load(ctx, strings.NewReader(string(read("tokenizer.json"))), manifest.Files["tokenizer.json"], tokenizer.DefaultLimits())
	check(err)
	replica, err := NewExperimentDecoderReplica(experiment, loaded, codec, plan, ExperimentDecoderReplicaOptions{Limits: decoder.Limits{MaxTokens: 1129, LogitRows: 2, MaxCheckpointBytes: memory.CheckpointBytes}, HashChunkBytes: 4 << 20, ParameterCopyBytes: 557056 * 4 * 3, SourceManifestSHA256: experiment.config.Recipe.Model.ManifestSHA256, ExpectedRotaryFrequencySHA256: experiment.config.RotarySHA256})
	if err != nil {
		_ = loaded.Close()
		t.Fatal(err)
	}
	defer replica.Close()
	_, err = replica.Inspect(ctx)
	check(err)
	data, err := ReadExperimentTraining(experiment, strings.NewReader(string(read("training.json"))))
	check(err)
	order, err := ExperimentTrainingOrder(experiment, data.Examples)
	check(err)
	if order[0].ExampleID != "mmlu-development-8ebfccf18df2260b" {
		t.Fatal("calibration example differs")
	}
	prompt, err := RenderExperimentTrainingPrompt(order[0], data.Demos)
	check(err)
	ids, restore, err := replica.prepare(ctx, ExperimentNativeExample{ExampleID: order[0].ExampleID, Prompt: prompt, CandidateVocabularyIDs: [4]int{357, 417, 351, 414}, LogitRows: 2})
	check(err)
	defer restore()
	if len(ids) != 879 {
		t.Fatal("calibration token geometry differs")
	}
	report := func(name string, v *torch.Tensor) {
		info, e := v.Info()
		check(e)
		raw, e := v.Bytes()
		check(e)
		digest := experimentHash(raw)
		raw = nil
		values, e := v.Float32Values()
		check(e)
		low, high := math.Inf(1), math.Inf(-1)
		bad := 0
		for _, x := range values {
			f := float64(x)
			if math.IsNaN(f) || math.IsInf(f, 0) {
				bad++
				continue
			}
			low = math.Min(low, f)
			high = math.Max(high, f)
		}
		t.Logf("stage=%s dtype=%d device=%s:%d shape=%v nonfinite=%d min=%g max=%g sha256=%s", name, info.DType, info.Device.Kind, info.Device.Index, info.Shape, bad, low, high, digest)
		for i := 0; i < 2; i++ {
			stats, e := torch.CUDAMemory(i)
			check(e)
			check(ValidateExperimentNativeAllocator(stats))
		}
		if bad != 0 {
			t.Fatalf("FIRST_NONFINITE_STAGE=%s", name)
		}
	}
	rawIDs := make([]byte, len(ids)*8)
	for i, id := range ids {
		binary.LittleEndian.PutUint64(rawIDs[i*8:], uint64(id))
	}
	index, err := torch.FromBytes(rawIDs, []int64{int64(len(ids))}, torch.Int64, torch.CUDADevice(0), false)
	check(err)
	defer index.Close()
	rows, err := loaded.Model.Embedding.IndexSelect(0, index)
	check(err)
	current, err := rows.Reshape([]int64{1, int64(len(ids)), 4096})
	check(err)
	_ = rows.Close()
	defer func() { _ = current.Close() }()
	for i := 0; i < 16; i++ {
		layer := loaded.Model.Layers[i]
		next, e := layers.DecoderForward(ctx, current, layer.Weights, layer.Adapter, layer.Cosine, layer.Sine, layer.Config)
		check(e)
		_ = current.Close()
		current = next
		t.Logf("prefix_layer_completed=%d", i)
	}
	report("layer15_output", current)
	var owned []*torch.Tensor
	defer func() {
		for i := len(owned) - 1; i >= 0; i-- {
			_ = owned[i].Close()
		}
	}()
	keep := func(v *torch.Tensor, e error) *torch.Tensor { check(e); owned = append(owned, v); return v }
	placed := keep(current.To(torch.CUDADevice(1), torch.Float16))
	report("layer16_input", placed)
	layer := loaded.Model.Layers[16]
	if capture := os.Getenv("TAYI_EXPERIMENT_DECODER_PROBE_CAPTURE"); capture != "" {
		check(os.Mkdir(capture, 0700))
		raw, e := placed.Bytes()
		check(e)
		if experimentHash(raw) != evidence.CapturedInput {
			t.Fatal("captured input does not reproduce the original failure")
		}
		check(os.WriteFile(filepath.Join(capture, "layer16-input-f16.bin"), raw, 0600))
		t.Log("ORIGINAL_INPUT_CAPTURED_FOR_BOUNDED_REPLAY")
	}
	for _, locked := range []bool{false, true} {
		for iteration := 0; iteration < 4; iteration++ {
			func() {
				if locked {
					runtime.LockOSThread()
					defer runtime.UnlockOSThread()
				}
				value, e := layers.DecoderForward(ctx, placed, layer.Weights, layer.Adapter, layer.Cosine, layer.Sine, layer.Config)
				if e != nil {
					t.Logf("production_decoder locked=%t iteration=%d error=%v", locked, iteration, e)
					return
				}
				defer value.Close()
				finite, e := value.AllFinite()
				check(e)
				raw, e := value.Bytes()
				check(e)
				t.Logf("production_decoder locked=%t iteration=%d all_finite=%t sha256=%s", locked, iteration, finite, experimentHash(raw))
			}()
		}
	}
	source := keep(placed.To(layer.Device, torch.Float32))
	norm := keep(layers.RMSNorm(source, layer.Weights.InputNorm, layer.Config.Epsilon))
	report("input_normalization", norm)
	attention := keep(layers.ForwardLinearAttention(ctx, norm, *layer.Weights.Linear, layer.Config.Linear))
	report("linear_attention_float32", attention)
	storedAttention := keep(attention.To(layer.Device, torch.Float16))
	report("attention_float16_storage", storedAttention)
	residual := keep(placed.Add(storedAttention))
	report("attention_residual", residual)
	post := keep(layers.RMSNorm(residual, layer.Weights.PostAttentionNorm, layer.Config.Epsilon))
	report("post_attention_normalization", post)
	ff := keep(layers.FeedForwardPromoted(post, layer.Weights.Gate, layer.Weights.Up, layer.Weights.Down))
	report("feed_forward_float32", ff)
	storedFF := keep(ff.To(layer.Device, torch.Float16))
	report("feed_forward_float16_storage", storedFF)
	output := keep(residual.Add(storedFF))
	report("decoder16_output", output)
	t.Log("NO_REPRODUCTION: this diagnostic does not qualify the canary or training")
}

// Captured replay loads only fourteen frozen weights, never the complete model.
func TestExperimentDecoder16CapturedReplay(t *testing.T) {
	root, capture := os.Getenv("TAYI_EXPERIMENT_DECODER_PROBE_ROOT"), os.Getenv("TAYI_EXPERIMENT_DECODER_CAPTURE_INPUT")
	if root == "" || capture == "" {
		t.Skip("requires exact private R16 capture")
	}
	experiment, evidence := privateNativeEvidence(t)
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	check := func(e error) {
		t.Helper()
		if e != nil {
			t.Fatal(e)
		}
	}
	body, e := os.ReadFile(filepath.Join(root, "manifest.json"))
	check(e)
	if experimentHash(body) != evidence.DiagnosticRelease {
		t.Fatal("release differs")
	}
	var manifest ExperimentNativeManifest
	check(json.Unmarshal(body, &manifest))
	read := func(name string) []byte {
		b, e := os.ReadFile(filepath.Join(root, name))
		check(e)
		if experimentHash(b) != manifest.Files[name] {
			t.Fatal("input hash differs", name)
		}
		return b
	}
	raw, e := os.ReadFile(capture)
	check(e)
	if len(raw) != 879*4096*2 || experimentHash(raw) != evidence.CapturedInput {
		t.Fatal("captured tensor differs")
	}
	memory, e := ExperimentNativeMemory(1129)
	check(e)
	available, e := diagnosticHostAvailableRAM("/proc/meminfo")
	check(e)
	if available < memory.HostPreloadBytes {
		t.Fatal("RAM reserve admission refused")
	}
	probe, e := diagnosticProbeGPUs(ctx, .80)
	check(e)
	if len(probe.GPUs) != 2 {
		t.Fatal("two measured GPUs required")
	}
	for i, g := range probe.GPUs {
		if g.Index != i || g.Capability != "7.5" || !strings.Contains(g.Name, "T10") || g.FreeBytes < g.CapBytes+(512<<20) {
			t.Fatal("GPU admission refused")
		}
		check(torch.SetCUDAMemoryFraction(i, .80))
	}
	limits := memory.assemblyLimits(experiment)
	plan, e := decoder.PlanTextAssembly(read("model.safetensors.index.json"), read("config.json"), read("initial-reference.json"), decoder.AssemblyIdentity{IndexSHA256: manifest.Files["model.safetensors.index.json"], ConfigSHA256: manifest.Files["config.json"], ReferenceSHA256: manifest.Files["initial-reference.json"], InitialAdapterSHA256: experiment.config.Recipe.LoRA.ExpectedInitialDigest}, limits)
	check(e)
	f, e := os.Open("/cache/tayi/checkpoints/fixture/model-00003-of-00004.safetensors")
	check(e)
	defer f.Close()
	stat, e := f.Stat()
	check(e)
	for _, deviceIndex := range []int{1, 0} {
		t.Run(fmt.Sprintf("gpu%d", deviceIndex), func(t *testing.T) {
			check := func(e error) {
				t.Helper()
				if e != nil {
					t.Fatal(e)
				}
			}
			var owned []*torch.Tensor
			defer func() {
				for i := len(owned) - 1; i >= 0; i-- {
					_ = owned[i].Close()
				}
			}()
			keep := func(v *torch.Tensor, e error) *torch.Tensor { check(e); owned = append(owned, v); return v }
			device := torch.CUDADevice(deviceIndex)
			weights := map[string]*torch.Tensor{}
			specs := map[string]decoder.AssemblyTensor{}
			prefix := "model.language_model.layers.16."
			for _, spec := range plan.Tensors() {
				if !strings.HasPrefix(spec.SourceName, prefix) {
					continue
				}
				if spec.Shard != "model-00003-of-00004.safetensors" {
					t.Fatal("unexpected shard")
				}
				storage, _, e := loading.LoadTensor(ctx, f, stat.Size(), spec.SourceName, loading.Options{HeaderLimits: limits.HeaderLimits, BudgetBytes: limits.TensorCopyBytes, DType: torch.Float16, Device: torch.CPUDevice(), Expected: &loading.Expectation{Shape: spec.Shape}})
				check(e)
				placed, e := storage.To(device, spec.DType)
				_ = storage.Close()
				check(e)
				owned = append(owned, placed)
				spec.Device = device
				digest, e := ExperimentNativeTensorDigest(ctx, placed, spec, 4<<20)
				check(e)
				if digest != spec.SHA256 {
					t.Fatal("frozen weight differs", spec.SourceName)
				}
				name := strings.TrimPrefix(spec.SourceName, prefix)
				weights[name] = placed
				specs[name] = spec
			}
			if len(weights) != 14 {
				t.Fatal("expected fourteen layer16 weights")
			}
			report := func(name string, v *torch.Tensor) string {
				info, e := v.Info()
				check(e)
				raw, e := v.Bytes()
				check(e)
				digest := experimentHash(raw)
				values, e := v.Float32Values()
				check(e)
				low, high := math.Inf(1), math.Inf(-1)
				bad := 0
				for _, x := range values {
					q := float64(x)
					if math.IsNaN(q) || math.IsInf(q, 0) {
						bad++
						continue
					}
					low = math.Min(low, q)
					high = math.Max(high, q)
				}
				t.Logf("stage=%s dtype=%d nonfinite=%d min=%g max=%g sha256=%s", name, info.DType, bad, low, high, digest)
				stats, e := torch.CUDAMemory(deviceIndex)
				check(e)
				check(ValidateExperimentNativeAllocator(stats))
				return digest
			}
			w := layers.DecoderWeights{InputNorm: weights["input_layernorm.weight"], PostAttentionNorm: weights["post_attention_layernorm.weight"], Gate: weights["mlp.gate_proj.weight"], Up: weights["mlp.up_proj.weight"], Down: weights["mlp.down_proj.weight"], Linear: &layers.LinearAttentionWeights{QKV: weights["linear_attn.in_proj_qkv.weight"], Z: weights["linear_attn.in_proj_z.weight"], Beta: weights["linear_attn.in_proj_b.weight"], Alpha: weights["linear_attn.in_proj_a.weight"], Convolution: weights["linear_attn.conv1d.weight"], ALog: weights["linear_attn.A_log"], DTBias: weights["linear_attn.dt_bias"], Norm: weights["linear_attn.norm.weight"], Output: weights["linear_attn.out_proj.weight"]}}
			cfg := layers.DecoderConfig{Epsilon: 1e-6, MaxInputElements: limits.MaxInputElements, Linear: layers.LinearAttentionConfig{KeyHeads: 16, ValueHeads: 32, KeyDimension: 128, ValueDimension: 128, Epsilon: 1e-6, MaxWorkingElements: limits.MaxWorkingElements, Sequence: limits.Sequence}}
			x := keep(torch.FromBytes(raw, []int64{1, 879, 4096}, torch.Float16, device, false))
			if os.Getenv("TAYI_EXPERIMENT_DECODER_REPLAY_PRESSURE") == "1" {
				zero := make([]byte, 64<<20)
				for d := 0; d < 2; d++ {
					stats, e := torch.CUDAMemory(d)
					check(e)
					target := int64(11050000000)
					for remaining := target - stats.AllocatedBytes; remaining > 0; {
						n := min(remaining, int64(len(zero)))
						keep(torch.FromBytes(zero[:int(n)], []int64{n}, torch.Bool, torch.CUDADevice(d), false))
						remaining -= n
					}
					stats, e = torch.CUDAMemory(d)
					check(e)
					check(ValidateExperimentNativeAllocator(stats))
					t.Logf("controlled_pressure device=%d allocated=%d reserved=%d", d, stats.AllocatedBytes, stats.ReservedBytes)
				}
			}
			for attempt := 0; attempt < 8; attempt++ {
				direct := keep(layers.DecoderForward(ctx, x, w, nil, nil, nil, cfg))
				report(fmt.Sprintf("production_decoder_%d", attempt), direct)
			}
			xf := keep(x.To(device, torch.Float32))
			normalized := keep(layers.RMSNorm(xf, w.InputNorm, 1e-6))
			attention := keep(layers.ForwardLinearAttention(ctx, normalized, *w.Linear, cfg.Linear))
			attentionHalf := keep(attention.To(device, torch.Float16))
			residual := keep(x.Add(attentionHalf))
			post := keep(layers.RMSNorm(residual, w.PostAttentionNorm, 1e-6))
			if report("post_attention_normalization", post) != evidence.PostAttention {
				t.Fatal("captured replay prefix differs")
			}
			if target := os.Getenv("TAYI_EXPERIMENT_DECODER_REPLAY_CAPTURE"); target != "" && deviceIndex == 1 {
				check(os.Mkdir(target, 0700))
				b, e := post.Bytes()
				check(e)
				check(os.WriteFile(filepath.Join(target, "post-normalization-f16.bin"), b, 0600))
			}
			whole := keep(layers.FeedForwardPromoted(post, w.Gate, w.Up, w.Down))
			report("whole_feed_forward", whole)
			input := keep(post.To(device, torch.Float32))
			report("ff_input_promoted", input)
			gateBase := keep(w.Gate.To(device, torch.Float32))
			gateHash := report("gate_weight_promoted", gateBase)
			upBase := keep(w.Up.To(device, torch.Float32))
			upHash := report("up_weight_promoted", upBase)
			downBase := keep(w.Down.To(device, torch.Float32))
			downHash := report("down_weight_promoted", downBase)
			gate := keep(layers.Linear(input, gateBase))
			report("gate_projection", gate)
			activated := keep(gate.SiLU())
			report("gate_silu", activated)
			up := keep(layers.Linear(input, upBase))
			report("up_projection", up)
			product := keep(activated.Mul(up))
			report("gated_product", product)
			down := keep(layers.Linear(product, downBase))
			report("down_projection", down)
			if report("gate_weight_after", gateBase) != gateHash || report("up_weight_after", upBase) != upHash || report("down_weight_after", downBase) != downHash {
				t.Fatal("promoted weights changed during calculation")
			}
			for name, spec := range specs {
				digest, e := ExperimentNativeTensorDigest(ctx, weights[name], spec, 4<<20)
				check(e)
				if digest != spec.SHA256 {
					t.Fatal("frozen base changed", name)
				}
			}
			cpuInput := keep(input.Slice(1, 0, 1, 1))
			cpuInput = keep(cpuInput.To(torch.CPUDevice(), torch.Float32))
			cpuGate := keep(gateBase.To(torch.CPUDevice(), torch.Float32))
			cpuUp := keep(upBase.To(torch.CPUDevice(), torch.Float32))
			cpuDown := keep(downBase.To(torch.CPUDevice(), torch.Float32))
			cpuResult := keep(layers.FeedForwardPromoted(cpuInput, cpuGate, cpuUp, cpuDown))
			report("cpu_first_token_reference", cpuResult)
			first := keep(down.Slice(1, 0, 1, 1))
			firstCPU := keep(first.To(torch.CPUDevice(), torch.Float32))
			delta := keep(firstCPU.Sub(cpuResult))
			report("gpu_minus_cpu_first_token", delta)
			t.Log("BOUNDED_REPLAY_COMPLETE: diagnostic only, not a successful canary")
		})
	}
}

// This test separates immutable CUDA storage from all model and matrix code.
func TestExperimentCUDAImmutableStoragePressure(t *testing.T) {
	if os.Getenv("TAYI_EXPERIMENT_STORAGE_PRESSURE_PROBE") != "1" {
		t.Skip("requires explicit private GPU diagnostic admission")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	check := func(e error) {
		t.Helper()
		if e != nil {
			t.Fatal(e)
		}
	}
	available, e := diagnosticHostAvailableRAM("/proc/meminfo")
	check(e)
	if available < 14<<30 {
		t.Fatal("host memory reserve refused")
	}
	probe, e := diagnosticProbeGPUs(ctx, .80)
	check(e)
	if len(probe.GPUs) != 2 {
		t.Fatal("requires two measured GPUs")
	}
	for i, g := range probe.GPUs {
		if g.Index != i || g.Capability != "7.5" || g.FreeBytes < g.CapBytes+(512<<20) {
			t.Fatal("GPU admission refused")
		}
		check(torch.SetCUDAMemoryFraction(i, .80))
	}
	for _, d := range []int{1, 0} {
		t.Run(fmt.Sprintf("gpu%d", d), func(t *testing.T) {
			var owned []*torch.Tensor
			defer func() {
				for i := len(owned) - 1; i >= 0; i-- {
					_ = owned[i].Close()
				}
			}()
			keep := func(v *torch.Tensor, e error) *torch.Tensor {
				if e != nil {
					t.Fatal(e)
				}
				owned = append(owned, v)
				return v
			}
			device := torch.CUDADevice(d)
			zeros := make([]byte, 64<<20)
			target := int64(11050000000)
			for remain := target; remain > 0; {
				n := min(remain, int64(len(zeros)))
				keep(torch.FromBytes(zeros[:int(n)], []int64{n}, torch.Bool, device, false))
				remain -= n
			}
			const count = 12288 * 4096
			raw16 := make([]byte, count*2)
			raw32 := make([]byte, count*4)
			for i := 0; i < count; i++ {
				binary.LittleEndian.PutUint16(raw16[i*2:], 0x3000)
				binary.LittleEndian.PutUint32(raw32[i*4:], 0x3e000000)
			}
			expected16, expected32 := experimentHash(raw16), experimentHash(raw32)
			raw32 = nil
			var promoted []*torch.Tensor
			for iteration := 0; iteration < 3; iteration++ {
				half := keep(torch.FromBytes(raw16, []int64{12288, 4096}, torch.Float16, device, false))
				raw, e := half.Bytes()
				check(e)
				t.Logf("immutable_half iteration=%d matches=%t sha256=%s", iteration, experimentHash(raw) == expected16, experimentHash(raw))
				promoted = append(promoted, keep(half.To(device, torch.Float32)))
				check(torch.CUDASynchronize(d))
				for j, v := range promoted {
					for read := 0; read < 2; read++ {
						raw, e := v.Bytes()
						check(e)
						got := experimentHash(raw)
						t.Logf("immutable_promoted allocation=%d tensor=%d read=%d matches=%t sha256=%s", iteration, j, read, got == expected32, got)
						if got != expected32 {
							t.Error("IMMUTABLE_STORAGE_CONTENT_MISMATCH")
						}
					}
				}
				stats, e := torch.CUDAMemory(d)
				check(e)
				check(ValidateExperimentNativeAllocator(stats))
				t.Logf("storage_pressure allocated=%d reserved=%d", stats.AllocatedBytes, stats.ReservedBytes)
				if os.Getenv("TAYI_EXPERIMENT_STORAGE_MATMUL_PROBE") == "1" {
					inputValues := make([]float32, 879*4096)
					for i := range inputValues {
						inputValues[i] = .125
					}
					input := keep(torch.FromFloat32(inputValues, []int64{1, 879, 4096}, device, false))
					result := keep(layers.Linear(input, promoted[iteration]))
					values, e := result.Float32Values()
					check(e)
					bad := 0
					for _, v := range values {
						if v != 64 {
							bad++
						}
					}
					t.Logf("constant_matmul iteration=%d mismatches=%d of=%d expected=64", iteration, bad, len(values))
					if bad > 0 {
						t.Error("CONSTANT_MATMUL_INCORRECT")
					}
					for j, v := range promoted {
						raw, e := v.Bytes()
						check(e)
						got := experimentHash(raw)
						t.Logf("weight_after_matmul iteration=%d tensor=%d matches=%t sha256=%s", iteration, j, got == expected32, got)
						if got != expected32 {
							t.Error("MATMUL_MODIFIED_READONLY_WEIGHT")
						}
					}
				}
			}
		})
	}
}
