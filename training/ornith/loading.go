package ornith

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"sort"
	"strings"

	"github.com/tayi-ai/arandu-llama/checkpoint"
	"github.com/tayi-ai/arandu-llama/training/layers"
	"github.com/tayi-ai/arandu-llama/training/loading"
	"github.com/tayi-ai/arandu-llama/training/sequence"
	"github.com/tayi-ai/arandu-llama/training/tensor"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

// ErrAssembly identifies invalid geometry, identities, placement or admission.
var ErrAssembly = errors.New("ornith: text assembly rejected")

// AssemblyIdentity contains mandatory caller-admitted identities. Document
// hashes cover exact bytes, and InitialAdapterSHA256 uses name/content order.
// The caller pins the intended checkpoint; matching geometry alone is not an
// identity. No default identity or alternate initialization is accepted.
type AssemblyIdentity struct {
	IndexSHA256, ConfigSHA256, ReferenceSHA256, InitialAdapterSHA256 string
}

// AssemblyLimits admits persistent weight bytes per GPU and temporary explicit
// tensor copies. TensorCopyBytes must cover three times the larger source/final
// tensor storage, including bounded hash copies. These are payload estimates,
// not measured RAM/VRAM caps. Headers, allocator overhead, caller-owned initial
// tensors, activations, RoPE and kernel workspaces require separate admission.
// All execution budgets are explicit; no sequence or attention defaults apply.
type AssemblyLimits struct {
	HeaderLimits       checkpoint.Limits
	TensorCopyBytes    int64
	PersistentBytes    [2]int64
	HashChunkBytes     int
	MaxInputElements   int64
	MaxScoreElements   int64
	MaxWorkingElements int64
	Sequence           sequence.SequenceLimits
}

// AssemblyTensor specifies a checkpoint weight and its required final identity.
// All base tensors pass through Float16 CPU storage before final placement,
// including A_log; attention tensors are then promoted to Float32. Adapter
// tensors have an empty SourceName/Shard and are copied from the admitted CPU set.
type AssemblyTensor struct {
	SourceName, ReferenceName, Shard, SHA256 string
	Shape                                    []int64
	DType                                    torch.DType
	Device                                   torch.Device
	Bytes                                    int64
	Trainable                                bool
}

// AssemblySummary records the admitted geometry without allocating tensor data.
type AssemblySummary struct {
	BaseTensors, AdapterTensors int
	AdapterElements             int64
	PersistentBytes             [2]int64
	LocalMPSBytes               int64
	Identity                    AssemblyIdentity
}

// AssemblyPlan is an immutable, validated text-only weight and placement plan.
// Vision and MTP weights are not executed by the text-only forward. This plan
// does not support image/video inputs or claim equivalence for multimodal calls.
type AssemblyPlan struct {
	weights, adapters []AssemblyTensor
	limits            AssemblyLimits
	summary           AssemblySummary
	localMPS          bool
}

// Summary returns value-only counts, budgets and document identities.
func (p *AssemblyPlan) Summary() AssemblySummary {
	if p == nil {
		return AssemblySummary{}
	}
	return p.summary
}

// Tensors returns detached metadata copies in base-then-adapter loading order.
func (p *AssemblyPlan) Tensors() []AssemblyTensor {
	if p == nil {
		return nil
	}
	result := append(append([]AssemblyTensor(nil), p.weights...), p.adapters...)
	for i := range result {
		result[i].Shape = slices.Clone(result[i].Shape)
	}
	return result
}

// Shard is an immutable opened checkpoint source. Size reports its full size;
// the loader closes each successfully opened source exactly once per operation.
type Shard interface {
	io.ReaderAt
	Size() int64
	Close() error
}

// ShardProvider opens a caller-authorized shard by its validated basename. It
// performs no implicit download or discovery. Separate opens must return handles
// with independent Close ownership over identical, immutable source bytes.
type ShardProvider interface {
	OpenShard(context.Context, string) (Shard, error)
}

// AssemblyInspection records header-only source checks, not content verification.
type AssemblyInspection struct {
	Shards, BaseTensors int
	SourceBytes         int64
	LargestCopyBytes    int64
}

// AssemblyTensorReceipt records both checkpoint payload and final native hashes.
// Initial adapter receipts have no checkpoint payload receipt.
type AssemblyTensorReceipt struct {
	Name, SHA256 string
	Source       *loading.Receipt
}

// LoadedTextModel owns every placed weight and adapter. Model and Parameters
// borrow those handles; close this owner only after all forward/VJP work ends.
// Rotary values remain nil until the caller supplies prompt-specific tables.
// The caller owns those rotary handles independently and must close them itself.
type LoadedTextModel struct {
	Model      *TextModel
	Parameters []InitialParameter
	Receipts   []AssemblyTensorReceipt
	Summary    AssemblySummary
	owned      []*torch.Tensor
}

// Close releases all assembled weights; borrowed CPU initial tensors and caller
// rotary values remain untouched. Closing this owner invalidates its TextModel.
func (m *LoadedTextModel) Close() error {
	if m == nil {
		return nil
	}
	var failures []error
	for _, value := range m.owned {
		failures = append(failures, value.Close())
	}
	m.owned, m.Parameters = nil, nil
	if m.Model != nil {
		*m.Model = TextModel{}
	}
	m.Model = nil
	return errors.Join(failures...)
}

// PlanTextAssembly validates exact document hashes, fixed Ornith geometry,
// every required reference tensor and both persistent budgets before any source
// is opened. No payload is read and no native backend is required by this call.
func PlanTextAssembly(indexJSON, configJSON, referenceJSON []byte, identity AssemblyIdentity, limits AssemblyLimits) (*AssemblyPlan, error) {
	for _, document := range []struct {
		data   []byte
		digest string
	}{
		{indexJSON, identity.IndexSHA256}, {configJSON, identity.ConfigSHA256}, {referenceJSON, identity.ReferenceSHA256},
	} {
		if len(document.data) == 0 || len(document.data) > 16<<20 || !validAssemblyHash(document.digest) || assemblyHash(document.data) != document.digest {
			return nil, fmt.Errorf("%w: document identity or size differs", ErrAssembly)
		}
	}
	if !validAssemblyHash(identity.InitialAdapterSHA256) {
		return nil, fmt.Errorf("%w: initial adapter identity is required", ErrAssembly)
	}
	if err := validateAssemblyLimits(limits); err != nil {
		return nil, err
	}
	if err := validateAssemblyConfig(configJSON); err != nil {
		return nil, err
	}
	var index struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(indexJSON, &index); err != nil {
		return nil, fmt.Errorf("%w: index: %v", ErrAssembly, err)
	}
	var reference struct {
		Tensors []assemblyReference `json:"tensors"`
	}
	if err := json.Unmarshal(referenceJSON, &reference); err != nil {
		return nil, fmt.Errorf("%w: reference: %v", ErrAssembly, err)
	}
	if len(index.WeightMap) > 4096 || len(reference.Tensors) > 4096 {
		return nil, fmt.Errorf("%w: too many source/reference entries", ErrAssembly)
	}
	refs := make(map[string]assemblyReference, len(reference.Tensors))
	for _, row := range reference.Tensors {
		if _, exists := refs[row.Name]; exists {
			return nil, fmt.Errorf("%w: duplicate reference %q", ErrAssembly, row.Name)
		}
		refs[row.Name] = row
	}
	p := &AssemblyPlan{limits: limits, summary: AssemblySummary{Identity: identity}}
	p.weights, p.adapters = assemblyGeometry()
	admittedNames := make(map[string]bool, len(p.weights)+len(p.adapters))
	sourceNames := make(map[string]bool, len(p.weights))
	for _, group := range [][]AssemblyTensor{p.weights, p.adapters} {
		for i := range group {
			item := &group[i]
			row, exists := refs[item.ReferenceName]
			if !exists || row.Kind != "parameter" || row.RequiresGrad != item.Trainable || !sameShape(row.Shape, item.Shape) ||
				row.Bytes != item.Bytes || row.DType != assemblyDTypeName(item.DType) || row.Device != fmt.Sprintf("cuda:%d", item.Device.Index) || !validAssemblyHash(row.SHA256) {
				return nil, fmt.Errorf("%w: reference metadata differs for %s", ErrAssembly, item.ReferenceName)
			}
			item.SHA256 = row.SHA256
			admittedNames[item.ReferenceName] = true
			if !item.Trainable {
				shard, exists := index.WeightMap[item.SourceName]
				if !exists || !validAssemblyShard(shard) {
					return nil, fmt.Errorf("%w: invalid or missing shard for %s", ErrAssembly, item.SourceName)
				}
				item.Shard = shard
				sourceNames[item.SourceName] = true
			}
			p.summary.PersistentBytes[item.Device.Index] += item.Bytes
			if item.Bytes > limits.TensorCopyBytes/3 {
				return nil, fmt.Errorf("%w: tensor copy budget for %s", ErrAssembly, item.ReferenceName)
			}
		}
	}
	for name, shard := range index.WeightMap {
		if !validAssemblyShard(shard) {
			return nil, fmt.Errorf("%w: invalid shard basename", ErrAssembly)
		}
		if sourceNames[name] || strings.HasPrefix(name, "model.visual.") || strings.HasPrefix(name, "mtp.") {
			continue
		}
		return nil, fmt.Errorf("%w: unexpected source tensor %s", ErrAssembly, name)
	}
	for name, row := range refs {
		if admittedNames[name] || strings.HasPrefix(name, "base_model.model.model.visual.") {
			continue
		}
		if (name == "base_model.model.model.language_model.rotary_emb.inv_freq" || name == "base_model.model.model.language_model.rotary_emb.original_inv_freq") && row.Kind == "buffer" {
			continue
		}
		return nil, fmt.Errorf("%w: unexpected reference tensor %s", ErrAssembly, name)
	}
	for device, needed := range p.summary.PersistentBytes {
		if needed > limits.PersistentBytes[device] {
			return nil, fmt.Errorf("%w: GPU %d persistent budget needs %d bytes", ErrAssembly, device, needed)
		}
	}
	p.summary.BaseTensors, p.summary.AdapterTensors, p.summary.AdapterElements = len(p.weights), len(p.adapters), 557056
	return p, nil
}

// PlanLocalMPSAssembly retains the frozen two-CUDA reference validation and
// content hashes, then explicitly places the same tensors on one Apple device.
// maxMPSBytes is a caller-admitted payload cap, not an available-memory probe;
// activations, allocator overhead and system reserve need separate admission.
func PlanLocalMPSAssembly(indexJSON, configJSON, referenceJSON []byte, identity AssemblyIdentity, limits AssemblyLimits, maxMPSBytes int64) (*AssemblyPlan, error) {
	p, err := PlanTextAssembly(indexJSON, configJSON, referenceJSON, identity, limits)
	if err != nil {
		return nil, err
	}
	needed := p.summary.PersistentBytes[0] + p.summary.PersistentBytes[1]
	if maxMPSBytes <= 0 || needed > maxMPSBytes {
		return nil, fmt.Errorf("%w: local MPS persistent budget needs %d bytes", ErrAssembly, needed)
	}
	for i := range p.weights {
		p.weights[i].Device = torch.MPSDevice()
	}
	for i := range p.adapters {
		p.adapters[i].Device = torch.MPSDevice()
	}
	p.summary.LocalMPSBytes = needed
	p.localMPS = true
	return p, nil
}

// InspectAssemblySources opens only required shards, validates headers and all
// source shapes/dtypes/copy admissions, then closes every source. No payload or
// native tensor is read/created. Successful inspection is not a content attestation.
func InspectAssemblySources(ctx context.Context, plan *AssemblyPlan, provider ShardProvider) (result AssemblyInspection, err error) {
	sources, result, err := openAssemblySources(ctx, plan, provider)
	return result, errors.Join(err, closeAssemblySources(sources))
}

// LoadTextAssembly assembles the fixed text model from caller-owned
// immutable shard providers and an exact borrowed CPU initializer. It verifies
// all source headers before payloads, then every final tensor hash after the
// historical Float16 intermediate and Float32 attention promotion. A mismatch
// closes all placed tensors; no partial model is returned. GPU resource caps,
// process deadlines and live reserve monitoring remain caller responsibilities.
// Initial tensors must remain open and unchanged until this operation returns.
func LoadTextAssembly(ctx context.Context, plan *AssemblyPlan, provider ShardProvider, initial *InitialAdapter) (_ *LoadedTextModel, err error) {
	if err := assemblyContext(ctx, plan, provider); err != nil {
		return nil, err
	}
	if err := validateAssemblyInitial(ctx, plan, initial); err != nil {
		return nil, err
	}
	if plan.localMPS {
		if !torch.MPSAvailable() {
			return nil, fmt.Errorf("%w: local MPS unavailable", ErrAssembly)
		}
		probe, probeErr := torch.FromFloat32([]float32{0}, []int64{1}, torch.MPSDevice(), false)
		if probeErr != nil {
			return nil, fmt.Errorf("%w: local MPS unavailable: %v", ErrAssembly, probeErr)
		}
		if closeErr := probe.Close(); closeErr != nil {
			return nil, closeErr
		}
	} else {
		if !torch.CUDAEnabled() {
			return nil, torch.ErrCUDAUnavailable
		}
		count, countErr := torch.CUDADeviceCount()
		if countErr != nil {
			return nil, countErr
		}
		if count != 2 {
			return nil, fmt.Errorf("%w: exactly two visible CUDA devices required", ErrAssembly)
		}
	}
	sources, _, err := openAssemblySources(ctx, plan, provider)
	if err != nil {
		return nil, errors.Join(err, closeAssemblySources(sources))
	}
	defer func() { err = errors.Join(err, closeAssemblySources(sources)) }()
	result := &LoadedTextModel{Model: &TextModel{Layers: make([]Layer, 32), Epsilon: 1e-6}, Summary: plan.summary}
	complete := false
	defer func() {
		if !complete || err != nil {
			err = errors.Join(err, result.Close())
		}
	}()
	values := make(map[string]*torch.Tensor, len(plan.weights)+len(plan.adapters))
	for _, spec := range plan.weights {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		shard := sources[spec.Shard]
		storage, receipt, err := loading.LoadTensor(ctx, shard, shard.Size(), spec.SourceName, loading.Options{
			HeaderLimits: plan.limits.HeaderLimits, BudgetBytes: plan.limits.TensorCopyBytes,
			DType: torch.Float16, Device: torch.CPUDevice(), Expected: &loading.Expectation{Shape: spec.Shape},
		})
		if err != nil {
			return nil, fmt.Errorf("%w: %s: %w", ErrAssembly, spec.SourceName, err)
		}
		placed, err := storage.To(spec.Device, spec.DType)
		closeErr := storage.Close()
		if err != nil || closeErr != nil {
			if placed != nil {
				_ = placed.Close()
			}
			return nil, errors.Join(err, closeErr)
		}
		result.owned = append(result.owned, placed)
		if err := verifyAssemblyTensor(ctx, placed, spec, plan.limits.HashChunkBytes, nil); err != nil {
			return nil, err
		}
		values[spec.ReferenceName] = placed
		result.Receipts = append(result.Receipts, AssemblyTensorReceipt{Name: spec.ReferenceName, SHA256: spec.SHA256, Source: &receipt})
	}
	for i, spec := range plan.adapters {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		detached, err := initial.Parameters[i].Value.Detach()
		if err != nil {
			return nil, err
		}
		placed, err := detached.To(spec.Device, torch.Float32)
		closeErr := detached.Close()
		if err != nil || closeErr != nil {
			if placed != nil {
				_ = placed.Close()
			}
			return nil, errors.Join(err, closeErr)
		}
		leaf, err := placed.SetRequiresGrad(true)
		closeErr = placed.Close()
		if err != nil || closeErr != nil {
			if leaf != nil {
				_ = leaf.Close()
			}
			return nil, errors.Join(err, closeErr)
		}
		result.owned = append(result.owned, leaf)
		if err := verifyAssemblyTensor(ctx, leaf, spec, plan.limits.HashChunkBytes, nil); err != nil {
			return nil, err
		}
		values[spec.ReferenceName] = leaf
		result.Parameters = append(result.Parameters, InitialParameter{Name: spec.ReferenceName, Value: leaf})
		result.Receipts = append(result.Receipts, AssemblyTensorReceipt{Name: spec.ReferenceName, SHA256: spec.SHA256})
	}
	wireAssembly(result.Model, values, plan.limits, plan.localMPS)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	// Close sources before returning, so even a provider close failure cannot
	// return an apparently successful model or leak its owned native handles.
	if err := closeAssemblySources(sources); err != nil {
		return nil, err
	}
	complete = true
	return result, nil
}

type assemblyReference struct {
	Name         string  `json:"name"`
	Kind         string  `json:"kind"`
	DType        string  `json:"dtype"`
	Device       string  `json:"device"`
	Shape        []int64 `json:"shape"`
	Bytes        int64   `json:"bytes"`
	SHA256       string  `json:"content_sha256"`
	RequiresGrad bool    `json:"requires_grad"`
}

func assemblyGeometry() (weights, adapters []AssemblyTensor) {
	add := func(source string, shape []int64, dtype torch.DType, device int) {
		ref := "base_model.model." + source
		if strings.Contains(source, ".self_attn.q_proj.") || strings.Contains(source, ".self_attn.v_proj.") {
			ref = strings.TrimSuffix(ref, ".weight") + ".base_layer.weight"
		}
		count := int64(1)
		for _, d := range shape {
			count *= d
		}
		width := int64(2)
		if dtype == torch.Float32 {
			width = 4
		}
		weights = append(weights, AssemblyTensor{SourceName: source, ReferenceName: ref, Shape: shape, DType: dtype, Device: torch.CUDADevice(device), Bytes: count * width})
	}
	add("model.language_model.embed_tokens.weight", []int64{248320, 4096}, torch.Float16, 0)
	for layer := 0; layer < 32; layer++ {
		device := layer / 16
		prefix := fmt.Sprintf("model.language_model.layers.%d.", layer)
		if layer%4 == 3 {
			for _, item := range []struct {
				suffix string
				shape  []int64
			}{
				{"q_proj.weight", []int64{8192, 4096}}, {"k_proj.weight", []int64{1024, 4096}},
				{"v_proj.weight", []int64{1024, 4096}}, {"o_proj.weight", []int64{4096, 4096}},
				{"q_norm.weight", []int64{256}}, {"k_norm.weight", []int64{256}},
			} {
				add(prefix+"self_attn."+item.suffix, item.shape, torch.Float32, device)
			}
			for _, item := range []struct {
				projection, letter string
				shape              []int64
			}{
				{"q_proj", "A", []int64{4, 4096}}, {"q_proj", "B", []int64{8192, 4}},
				{"v_proj", "A", []int64{4, 4096}}, {"v_proj", "B", []int64{1024, 4}},
			} {
				name := "base_model.model." + prefix + "self_attn." + item.projection + ".lora_" + item.letter + ".default.weight"
				adapters = append(adapters, AssemblyTensor{ReferenceName: name, Shape: item.shape, DType: torch.Float32, Device: torch.CUDADevice(device), Bytes: item.shape[0] * item.shape[1] * 4, Trainable: true})
			}
		} else {
			for _, item := range []struct {
				suffix string
				shape  []int64
			}{
				{"dt_bias", []int64{32}}, {"A_log", []int64{32}}, {"conv1d.weight", []int64{8192, 1, 4}},
				{"norm.weight", []int64{128}}, {"out_proj.weight", []int64{4096, 4096}},
				{"in_proj_qkv.weight", []int64{8192, 4096}}, {"in_proj_z.weight", []int64{4096, 4096}},
				{"in_proj_b.weight", []int64{32, 4096}}, {"in_proj_a.weight", []int64{32, 4096}},
			} {
				add(prefix+"linear_attn."+item.suffix, item.shape, torch.Float32, device)
			}
		}
		for _, item := range []struct {
			suffix string
			shape  []int64
		}{
			{"mlp.gate_proj.weight", []int64{12288, 4096}}, {"mlp.up_proj.weight", []int64{12288, 4096}}, {"mlp.down_proj.weight", []int64{4096, 12288}},
			{"input_layernorm.weight", []int64{4096}}, {"post_attention_layernorm.weight", []int64{4096}},
		} {
			add(prefix+item.suffix, item.shape, torch.Float16, device)
		}
	}
	add("model.language_model.norm.weight", []int64{4096}, torch.Float16, 1)
	add("lm_head.weight", []int64{248320, 4096}, torch.Float16, 1)
	return
}

func validateAssemblyLimits(limits AssemblyLimits) error {
	h := limits.HeaderLimits
	if h.MaxHeaderBytes <= 0 || h.MaxHeaderBytes > 100_000_000 || h.MaxTensors <= 0 || h.MaxDimensions < 3 || h.MaxMetadataEntries <= 0 || h.MaxChunkBytes <= 0 ||
		limits.HashChunkBytes < 4 || limits.HashChunkBytes > 4<<20 || limits.TensorCopyBytes <= 0 || limits.PersistentBytes[0] <= 0 || limits.PersistentBytes[1] <= 0 ||
		limits.MaxInputElements <= 0 || limits.MaxScoreElements <= 0 || limits.MaxWorkingElements <= 0 || limits.Sequence.ChunkTokens <= 0 || limits.Sequence.ChunkTokens > tensor.MaxChunkTokens ||
		limits.Sequence.MaxTokens <= 0 || limits.Sequence.MaxOwnedElements <= 0 {
		return fmt.Errorf("%w: explicit positive parsing, copy, persistent and execution limits required", ErrAssembly)
	}
	return nil
}

func validateAssemblyConfig(data []byte) error {
	var config struct {
		ModelType string `json:"model_type"`
		Tie       bool   `json:"tie_word_embeddings"`
		Text      struct {
			ModelType     string   `json:"model_type"`
			Hidden        int      `json:"hidden_size"`
			Vocab         int      `json:"vocab_size"`
			MLP           int      `json:"intermediate_size"`
			Layers        int      `json:"num_hidden_layers"`
			Heads         int      `json:"num_attention_heads"`
			KVHeads       int      `json:"num_key_value_heads"`
			Dimension     int      `json:"head_dim"`
			LayerTypes    []string `json:"layer_types"`
			KHeads        int      `json:"linear_num_key_heads"`
			VHeads        int      `json:"linear_num_value_heads"`
			KDimension    int      `json:"linear_key_head_dim"`
			VDimension    int      `json:"linear_value_head_dim"`
			Conv          int      `json:"linear_conv_kernel_dim"`
			Epsilon       float64  `json:"rms_norm_eps"`
			Activation    string   `json:"hidden_act"`
			AttentionBias bool     `json:"attention_bias"`
			Dropout       float64  `json:"attention_dropout"`
			Gate          bool     `json:"attn_output_gate"`
			Tie           bool     `json:"tie_word_embeddings"`
			RoPE          struct {
				Theta       float64 `json:"rope_theta"`
				Type        string  `json:"rope_type"`
				Partial     float64 `json:"partial_rotary_factor"`
				Interleaved bool    `json:"mrope_interleaved"`
				Sections    []int   `json:"mrope_section"`
			} `json:"rope_parameters"`
		} `json:"text_config"`
	}
	if err := json.Unmarshal(data, &config); err != nil {
		return fmt.Errorf("%w: config: %v", ErrAssembly, err)
	}
	c := config.Text
	if config.ModelType != "qwen3_5" || config.Tie || c.ModelType != "qwen3_5_text" || c.Hidden != 4096 || c.Vocab != 248320 || c.MLP != 12288 || c.Layers != 32 || c.Heads != 16 || c.KVHeads != 4 || c.Dimension != 256 || len(c.LayerTypes) != 32 || c.KHeads != 16 || c.VHeads != 32 || c.KDimension != 128 || c.VDimension != 128 || c.Conv != 4 || c.Epsilon != 1e-6 || c.Activation != "silu" || c.AttentionBias || c.Dropout != 0 || !c.Gate || c.Tie || c.RoPE.Theta != 1e7 || c.RoPE.Type != "default" || c.RoPE.Partial != .25 || !c.RoPE.Interleaved || !slices.Equal(c.RoPE.Sections, []int{11, 11, 10}) {
		return fmt.Errorf("%w: fixed text geometry/config differs", ErrAssembly)
	}
	for i, kind := range c.LayerTypes {
		expected := "linear_attention"
		if i%4 == 3 {
			expected = "full_attention"
		}
		if kind != expected {
			return fmt.Errorf("%w: layer %d type differs", ErrAssembly, i)
		}
	}
	return nil
}

func assemblyContext(ctx context.Context, plan *AssemblyPlan, provider ShardProvider) error {
	if ctx == nil || plan == nil || len(plan.weights) != 427 || len(plan.adapters) != 32 || provider == nil {
		return fmt.Errorf("%w: context, validated plan and provider required", ErrAssembly)
	}
	return ctx.Err()
}

func openAssemblySources(ctx context.Context, plan *AssemblyPlan, provider ShardProvider) (map[string]Shard, AssemblyInspection, error) {
	sources := make(map[string]Shard)
	result := AssemblyInspection{}
	if err := assemblyContext(ctx, plan, provider); err != nil {
		return sources, result, err
	}
	names := make(map[string]bool)
	for _, spec := range plan.weights {
		names[spec.Shard] = true
	}
	ordered := make([]string, 0, len(names))
	for name := range names {
		ordered = append(ordered, name)
	}
	sort.Strings(ordered)
	indices := make(map[string]*checkpoint.Safetensors)
	for _, name := range ordered {
		if err := ctx.Err(); err != nil {
			return sources, result, err
		}
		source, err := provider.OpenShard(ctx, name)
		if source != nil {
			sources[name] = source
		}
		if err != nil {
			return sources, result, err
		}
		if source == nil {
			return sources, result, fmt.Errorf("%w: provider returned nil shard", ErrAssembly)
		}
		index, err := checkpoint.OpenSafetensors(assemblyReader{ctx, source}, source.Size(), plan.limits.HeaderLimits)
		if err != nil {
			return sources, result, err
		}
		indices[name] = index
		result.Shards++
	}
	for _, spec := range plan.weights {
		if err := ctx.Err(); err != nil {
			return sources, result, err
		}
		row, found := indices[spec.Shard].Tensor(spec.SourceName)
		shape := make([]int64, len(row.Shape))
		for i, d := range row.Shape {
			if d > 1<<62 {
				return sources, result, fmt.Errorf("%w: oversized source shape", ErrAssembly)
			}
			shape[i] = int64(d)
		}
		if !found || !sameShape(shape, spec.Shape) || (row.DType != "F16" && row.DType != "BF16" && row.DType != "F32" && row.DType != "F64") {
			return sources, result, fmt.Errorf("%w: source metadata differs for %s", ErrAssembly, spec.SourceName)
		}
		needed := max(row.Size(), spec.Bytes)
		if needed > plan.limits.TensorCopyBytes/3 {
			return sources, result, fmt.Errorf("%w: source conversion exceeds copy budget for %s", ErrAssembly, spec.SourceName)
		}
		result.LargestCopyBytes = max(result.LargestCopyBytes, needed*3)
		result.SourceBytes += row.Size()
		result.BaseTensors++
	}
	return sources, result, nil
}

func closeAssemblySources(sources map[string]Shard) error {
	var failures []error
	for name, source := range sources {
		failures = append(failures, source.Close())
		delete(sources, name)
	}
	return errors.Join(failures...)
}

type assemblyReader struct {
	context context.Context
	source  io.ReaderAt
}

func (r assemblyReader) ReadAt(data []byte, offset int64) (int, error) {
	if err := r.context.Err(); err != nil {
		return 0, err
	}
	return r.source.ReadAt(data, offset)
}

func validateAssemblyInitial(ctx context.Context, plan *AssemblyPlan, initial *InitialAdapter) error {
	if initial == nil || len(initial.Parameters) != 32 || initial.SHA256 != plan.summary.Identity.InitialAdapterSHA256 {
		return fmt.Errorf("%w: exact initial adapter required", ErrAssembly)
	}
	digest := sha256.New()
	for i, spec := range plan.adapters {
		if initial.Parameters[i].Name != spec.ReferenceName {
			return fmt.Errorf("%w: adapter order/names differ", ErrAssembly)
		}
		cpuSpec := spec
		cpuSpec.Device = torch.CPUDevice()
		_, _ = digest.Write([]byte(spec.ReferenceName))
		if err := verifyAssemblyTensor(ctx, initial.Parameters[i].Value, cpuSpec, plan.limits.HashChunkBytes, digest); err != nil {
			return err
		}
	}
	if hex.EncodeToString(digest.Sum(nil)) != plan.summary.Identity.InitialAdapterSHA256 {
		return fmt.Errorf("%w: initial adapter aggregate differs", ErrAssembly)
	}
	return ctx.Err()
}

func verifyAssemblyTensor(ctx context.Context, value *torch.Tensor, spec AssemblyTensor, chunkBytes int, aggregate io.Writer) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if value == nil {
		return fmt.Errorf("%w: missing tensor", ErrAssembly)
	}
	info, err := value.Info()
	if err != nil {
		return err
	}
	if info.Device != spec.Device || info.DType != spec.DType || info.RequiresGrad != spec.Trainable || !sameShape(info.Shape, spec.Shape) {
		return fmt.Errorf("%w: native metadata differs for %s", ErrAssembly, spec.ReferenceName)
	}
	flat, err := value.Reshape([]int64{info.Elements})
	if err != nil {
		return err
	}
	defer flat.Close()
	width := int64(2)
	if spec.DType == torch.Float32 {
		width = 4
	}
	count := int64(chunkBytes) / width
	digest := sha256.New()
	for offset := int64(0); offset < info.Elements; {
		if err := ctx.Err(); err != nil {
			return err
		}
		end := min(info.Elements, offset+count)
		piece, err := flat.Slice(0, offset, end, 1)
		if err != nil {
			return err
		}
		finite, finiteErr := piece.AllFinite()
		content, copyErr := piece.Bytes()
		closeErr := piece.Close()
		if finiteErr != nil || copyErr != nil || closeErr != nil {
			return errors.Join(finiteErr, copyErr, closeErr)
		}
		if !finite {
			return fmt.Errorf("%w: nonfinite tensor %s", ErrAssembly, spec.ReferenceName)
		}
		_, _ = digest.Write(content)
		if aggregate != nil {
			if _, err := aggregate.Write(content); err != nil {
				return err
			}
		}
		offset = end
	}
	if hex.EncodeToString(digest.Sum(nil)) != spec.SHA256 {
		return fmt.Errorf("%w: materialized hash differs for %s", ErrAssembly, spec.ReferenceName)
	}
	return ctx.Err()
}

func wireAssembly(model *TextModel, values map[string]*torch.Tensor, limits AssemblyLimits, localMPS bool) {
	get := func(source string) *torch.Tensor { return values["base_model.model."+source] }
	model.Embedding = get("model.language_model.embed_tokens.weight")
	model.FinalNorm = get("model.language_model.norm.weight")
	model.Head = get("lm_head.weight")
	for i := range model.Layers {
		prefix := fmt.Sprintf("model.language_model.layers.%d.", i)
		g := func(s string) *torch.Tensor { return get(prefix + s) }
		layer := &model.Layers[i]
		layer.Device = torch.CUDADevice(i / 16)
		if localMPS {
			layer.Device = torch.MPSDevice()
		}
		layer.Config = layers.DecoderConfig{Epsilon: 1e-6, MaxInputElements: limits.MaxInputElements,
			Full:   layers.AttentionConfig{Heads: 16, KVHeads: 4, HeadDimension: 256, RotaryDimension: 64, Epsilon: 1e-6, MaxScoreElements: limits.MaxScoreElements},
			Linear: layers.LinearAttentionConfig{KeyHeads: 16, ValueHeads: 32, KeyDimension: 128, ValueDimension: 128, Epsilon: 1e-6, MaxWorkingElements: limits.MaxWorkingElements, Sequence: limits.Sequence}}
		layer.Weights = layers.DecoderWeights{InputNorm: g("input_layernorm.weight"), PostAttentionNorm: g("post_attention_layernorm.weight"), Gate: g("mlp.gate_proj.weight"), Up: g("mlp.up_proj.weight"), Down: g("mlp.down_proj.weight")}
		if i%4 == 3 {
			layer.Weights.Full = &layers.AttentionWeights{Query: g("self_attn.q_proj.base_layer.weight"), Key: g("self_attn.k_proj.weight"), Value: g("self_attn.v_proj.base_layer.weight"), Output: g("self_attn.o_proj.weight"), QueryNorm: g("self_attn.q_norm.weight"), KeyNorm: g("self_attn.k_norm.weight")}
			layer.Adapter = &layers.AttentionLoRA{QueryA: g("self_attn.q_proj.lora_A.default.weight"), QueryB: g("self_attn.q_proj.lora_B.default.weight"), ValueA: g("self_attn.v_proj.lora_A.default.weight"), ValueB: g("self_attn.v_proj.lora_B.default.weight"), Alpha: 8}
		} else {
			layer.Weights.Linear = &layers.LinearAttentionWeights{QKV: g("linear_attn.in_proj_qkv.weight"), Z: g("linear_attn.in_proj_z.weight"), Beta: g("linear_attn.in_proj_b.weight"), Alpha: g("linear_attn.in_proj_a.weight"), Convolution: g("linear_attn.conv1d.weight"), ALog: g("linear_attn.A_log"), DTBias: g("linear_attn.dt_bias"), Norm: g("linear_attn.norm.weight"), Output: g("linear_attn.out_proj.weight")}
		}
	}
}

func validAssemblyHash(value string) bool {
	decoded, err := hex.DecodeString(value)
	return err == nil && len(decoded) == 32 && strings.ToLower(value) == value
}
func assemblyHash(value []byte) string { return fmt.Sprintf("%x", sha256.Sum256(value)) }
func assemblyDTypeName(dtype torch.DType) string {
	if dtype == torch.Float32 {
		return "torch.float32"
	}
	return "torch.float16"
}
func validAssemblyShard(name string) bool {
	for i := 1; i <= 4; i++ {
		if name == fmt.Sprintf("model-%05d-of-00004.safetensors", i) {
			return true
		}
	}
	return false
}

// Keep this compile-time assertion near the source adapter: no Seek, path,
// filesystem mutation or process interface is required by the loader.
var _ io.ReaderAt = assemblyReader{}
