package assembly_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"testing"

	"github.com/tayi-ai/arandu-llama/checkpoint"
	"github.com/tayi-ai/arandu-llama/training/ornith"
	"github.com/tayi-ai/arandu-llama/training/sequence"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

const initialDigest = "75185d68fcfdd09bb2dcb6dffe0902a35300f52d9fda716b408189991773aa7c"

type reference struct {
	Name   string  `json:"name"`
	Kind   string  `json:"kind"`
	DType  string  `json:"dtype"`
	Device string  `json:"device"`
	Shape  []int64 `json:"shape"`
	Bytes  int64   `json:"bytes"`
	Hash   string  `json:"content_sha256"`
	Grad   bool    `json:"requires_grad"`
}

type documents struct {
	Index     map[string]string
	Reference []reference
	Config    map[string]any
}

func fixture() documents {
	d := documents{Index: map[string]string{}}
	add := func(source string, shape []int64, dtype string, device int, grad bool) {
		name := "base_model.model." + source
		if !grad && (strings.Contains(source, ".self_attn.q_proj.") || strings.Contains(source, ".self_attn.v_proj.")) {
			name = strings.TrimSuffix(name, ".weight") + ".base_layer.weight"
		}
		width := int64(2)
		if dtype == "torch.float32" {
			width = 4
		}
		size := width
		for _, n := range shape {
			size *= n
		}
		d.Reference = append(d.Reference, reference{name, "parameter", dtype, fmt.Sprintf("cuda:%d", device), shape, size, strings.Repeat("0", 64), grad})
		if !grad {
			d.Index[source] = fmt.Sprintf("model-%05d-of-00004.safetensors", len(d.Index)%4+1)
		}
	}
	add("model.language_model.embed_tokens.weight", []int64{248320, 4096}, "torch.float16", 0, false)
	kinds := make([]string, 32)
	for layer := 0; layer < 32; layer++ {
		prefix := fmt.Sprintf("model.language_model.layers.%d.", layer)
		device := layer / 16
		kinds[layer] = "linear_attention"
		if layer%4 == 3 {
			kinds[layer] = "full_attention"
			for name, out := range map[string]int64{"q_proj": 8192, "k_proj": 1024, "v_proj": 1024, "o_proj": 4096} {
				add(prefix+"self_attn."+name+".weight", []int64{out, 4096}, "torch.float32", device, false)
			}
			for _, name := range []string{"q_norm", "k_norm"} {
				add(prefix+"self_attn."+name+".weight", []int64{256}, "torch.float32", device, false)
			}
			for _, projection := range []struct {
				name string
				out  int64
			}{{"q_proj", 8192}, {"v_proj", 1024}} {
				add(prefix+"self_attn."+projection.name+".lora_A.default.weight", []int64{4, 4096}, "torch.float32", device, true)
				add(prefix+"self_attn."+projection.name+".lora_B.default.weight", []int64{projection.out, 4}, "torch.float32", device, true)
			}
		} else {
			for name, out := range map[string]int64{"in_proj_qkv": 8192, "in_proj_z": 4096, "in_proj_b": 32, "in_proj_a": 32, "out_proj": 4096} {
				add(prefix+"linear_attn."+name+".weight", []int64{out, 4096}, "torch.float32", device, false)
			}
			add(prefix+"linear_attn.conv1d.weight", []int64{8192, 1, 4}, "torch.float32", device, false)
			add(prefix+"linear_attn.norm.weight", []int64{128}, "torch.float32", device, false)
			for _, name := range []string{"dt_bias", "A_log"} {
				add(prefix+"linear_attn."+name, []int64{32}, "torch.float32", device, false)
			}
		}
		for _, name := range []string{"gate_proj", "up_proj"} {
			add(prefix+"mlp."+name+".weight", []int64{12288, 4096}, "torch.float16", device, false)
		}
		add(prefix+"mlp.down_proj.weight", []int64{4096, 12288}, "torch.float16", device, false)
		for _, name := range []string{"input_layernorm", "post_attention_layernorm"} {
			add(prefix+name+".weight", []int64{4096}, "torch.float16", device, false)
		}
	}
	add("model.language_model.norm.weight", []int64{4096}, "torch.float16", 1, false)
	add("lm_head.weight", []int64{248320, 4096}, "torch.float16", 1, false)
	d.Config = map[string]any{"model_type": "qwen3_5", "tie_word_embeddings": false, "text_config": map[string]any{
		"model_type": "qwen3_5_text", "hidden_size": 4096, "vocab_size": 248320, "intermediate_size": 12288, "num_hidden_layers": 32,
		"num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256, "layer_types": kinds,
		"linear_num_key_heads": 16, "linear_num_value_heads": 32, "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
		"rms_norm_eps": 1e-6, "hidden_act": "silu", "attention_bias": false, "attention_dropout": 0, "attn_output_gate": true, "tie_word_embeddings": false,
		"rope_parameters": map[string]any{"rope_theta": 1e7, "rope_type": "default", "partial_rotary_factor": .25, "mrope_interleaved": true, "mrope_section": []int{11, 11, 10}},
	}}
	return d
}

func admittedLimits() ornith.AssemblyLimits {
	return ornith.AssemblyLimits{HeaderLimits: checkpoint.DefaultLimits(), TensorCopyBytes: 6_102_712_320, PersistentBytes: [2]int64{11_042_374_656, 11_042_382_848}, HashChunkBytes: 4096,
		MaxInputElements: 4096 * 4096, MaxScoreElements: 16 * 4096 * 4096, MaxWorkingElements: 256 << 20, Sequence: sequence.DefaultLimits()}
}

func (d documents) encode(t *testing.T) ([]byte, []byte, []byte, ornith.AssemblyIdentity) {
	t.Helper()
	marshal := func(v any) []byte {
		data, err := json.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		return data
	}
	index := marshal(map[string]any{"weight_map": d.Index})
	config := marshal(d.Config)
	reference := marshal(map[string]any{"tensors": d.Reference})
	return index, config, reference, ornith.AssemblyIdentity{IndexSHA256: fmt.Sprintf("%x", sha256.Sum256(index)), ConfigSHA256: fmt.Sprintf("%x", sha256.Sum256(config)), ReferenceSHA256: fmt.Sprintf("%x", sha256.Sum256(reference)), InitialAdapterSHA256: initialDigest}
}

func plan(t *testing.T, d documents) *ornith.AssemblyPlan {
	t.Helper()
	i, c, r, id := d.encode(t)
	p, err := ornith.PlanTextAssembly(i, c, r, id, admittedLimits())
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestLocalMPSAssemblyPreservesFrozenReferenceAndRequiresAggregateBudget(t *testing.T) {
	i, c, r, id := fixture().encode(t)
	limits := admittedLimits()
	needed := limits.PersistentBytes[0] + limits.PersistentBytes[1]
	if p, err := ornith.PlanLocalMPSAssembly(i, c, r, id, limits, needed-1); p != nil || !errors.Is(err, ornith.ErrAssembly) {
		t.Fatalf("under-budget MPS plan accepted: %v", err)
	}
	p, err := ornith.PlanLocalMPSAssembly(i, c, r, id, limits, needed)
	if err != nil {
		t.Fatal(err)
	}
	if p.Summary().LocalMPSBytes != needed || p.Summary().PersistentBytes != limits.PersistentBytes {
		t.Fatalf("local placement budget or frozen reference changed: %+v", p.Summary())
	}
	for _, item := range p.Tensors() {
		if item.Device != torch.MPSDevice() {
			t.Fatalf("tensor %s not on MPS", item.ReferenceName)
		}
	}
	badReference := fixture()
	badReference.Reference[0].Device = "mps"
	i, c, r, id = badReference.encode(t)
	if p, err := ornith.PlanLocalMPSAssembly(i, c, r, id, limits, needed); p != nil || !errors.Is(err, ornith.ErrAssembly) {
		t.Fatalf("modified frozen reference accepted: %v", err)
	}
}

func TestFixedTextAssemblyPlanAndOwnedMetadata(t *testing.T) {
	p := plan(t, fixture())
	summary := p.Summary()
	if summary.BaseTensors != 427 || summary.AdapterTensors != 32 || summary.AdapterElements != 557056 || summary.PersistentBytes != admittedLimits().PersistentBytes {
		t.Fatalf("geometry:%+v", summary)
	}
	metadata := p.Tensors()
	if len(metadata) != 459 {
		t.Fatal(len(metadata))
	}
	var perDevice [2]int
	var adapters int
	var trainableElements int64
	for _, row := range metadata {
		perDevice[row.Device.Index]++
		if row.Trainable {
			adapters++
			trainableElements += row.Bytes / 4
			if row.DType != torch.Float32 || row.Shard != "" || row.SourceName != "" {
				t.Fatal("invalid adapter placement")
			}
		}
		if strings.HasSuffix(row.SourceName, "A_log") && row.DType != torch.Float32 {
			t.Fatal("A_log was not promoted after storage load")
		}
	}
	if perDevice != [2]int{229, 230} || adapters != 32 || trainableElements != 557056 {
		t.Fatalf("placement:%v/%d/%d", perDevice, adapters, trainableElements)
	}
	if metadata[0].SourceName != "model.language_model.embed_tokens.weight" || metadata[426].SourceName != "lm_head.weight" {
		t.Fatal("embedding/head order changed")
	}
	metadata[0].Shape[0] = 1
	metadata[0].SHA256 = "changed"
	metadata[0].Device = torch.CPUDevice()
	if p.Tensors()[0].Shape[0] != 248320 || p.Tensors()[0].Device != torch.CUDADevice(0) {
		t.Fatal("caller modified private plan")
	}
}

func TestPlanRejectsIdentityGeometryAndBudgetBeforeOpeningSources(t *testing.T) {
	cases := []struct {
		name   string
		change func(*documents, *ornith.AssemblyLimits)
	}{
		{"missing_reference", func(d *documents, l *ornith.AssemblyLimits) { d.Reference = d.Reference[1:] }},
		{"duplicate_reference", func(d *documents, l *ornith.AssemblyLimits) { d.Reference = append(d.Reference, d.Reference[0]) }},
		{"shape", func(d *documents, l *ornith.AssemblyLimits) { d.Reference[0].Shape = []int64{1, 4096} }},
		{"dtype", func(d *documents, l *ornith.AssemblyLimits) { d.Reference[0].DType = "torch.float32" }},
		{"device", func(d *documents, l *ornith.AssemblyLimits) { d.Reference[0].Device = "cuda:1" }},
		{"trainable_base", func(d *documents, l *ornith.AssemblyLimits) { d.Reference[0].Grad = true }},
		{"byte_count", func(d *documents, l *ornith.AssemblyLimits) { d.Reference[0].Bytes-- }},
		{"malformed_tensor_hash", func(d *documents, l *ornith.AssemblyLimits) { d.Reference[0].Hash = "missing" }},
		{"missing_shard", func(d *documents, l *ornith.AssemblyLimits) { delete(d.Index, "lm_head.weight") }},
		{"traversal", func(d *documents, l *ornith.AssemblyLimits) { d.Index["lm_head.weight"] = "../weights.safetensors" }},
		{"unexpected_text", func(d *documents, l *ornith.AssemblyLimits) {
			d.Index["model.language_model.extra.weight"] = "model-00001-of-00004.safetensors"
		}},
		{"hidden_geometry", func(d *documents, l *ornith.AssemblyLimits) {
			d.Config["text_config"].(map[string]any)["hidden_size"] = 2048
		}},
		{"wrong_layers", func(d *documents, l *ornith.AssemblyLimits) {
			d.Config["text_config"].(map[string]any)["layer_types"].([]string)[3] = "linear_attention"
		}},
		{"copy_budget", func(d *documents, l *ornith.AssemblyLimits) { l.TensorCopyBytes-- }},
		{"persistent_budget", func(d *documents, l *ornith.AssemblyLimits) { l.PersistentBytes[1]-- }},
		{"zero_working", func(d *documents, l *ornith.AssemblyLimits) { l.MaxWorkingElements = 0 }},
		{"unavailable_chunk", func(d *documents, l *ornith.AssemblyLimits) { l.Sequence.ChunkTokens = 33 }},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			d := fixture()
			limits := admittedLimits()
			test.change(&d, &limits)
			i, c, r, id := d.encode(t)
			p, err := ornith.PlanTextAssembly(i, c, r, id, limits)
			if p != nil || !errors.Is(err, ornith.ErrAssembly) {
				t.Fatalf("invalid plan accepted:%v", err)
			}
		})
	}
	i, c, r, id := fixture().encode(t)
	c = append(c, ' ')
	if p, err := ornith.PlanTextAssembly(i, c, r, id, admittedLimits()); p != nil || !errors.Is(err, ornith.ErrAssembly) {
		t.Fatal("changed document identity accepted")
	}
	i, c, r, id = fixture().encode(t)
	id.InitialAdapterSHA256 = ""
	if p, err := ornith.PlanTextAssembly(i, c, r, id, admittedLimits()); p != nil || !errors.Is(err, ornith.ErrAssembly) {
		t.Fatal("missing initialization identity accepted")
	}
}

type virtualShard struct {
	header                      []byte
	size                        int64
	reads, payloadReads, closes int
	closeErr                    error
	afterRead                   func()
}

func (s *virtualShard) Size() int64  { return s.size }
func (s *virtualShard) Close() error { s.closes++; return s.closeErr }
func (s *virtualShard) ReadAt(p []byte, offset int64) (int, error) {
	s.reads++
	if offset >= int64(len(s.header)) {
		s.payloadReads++
		return 0, errors.New("payload read is forbidden in a header inspection")
	}
	n, err := bytes.NewReader(s.header).ReadAt(p, offset)
	if s.afterRead != nil {
		s.afterRead()
	}
	return n, err
}

type virtualProvider struct {
	shards  map[string]*virtualShard
	opens   int
	openErr error
}

func (p *virtualProvider) OpenShard(ctx context.Context, name string) (ornith.Shard, error) {
	p.opens++
	return p.shards[name], p.openErr
}

func provider(t *testing.T, p *ornith.AssemblyPlan, mutate func(string, *checkpoint.Tensor)) *virtualProvider {
	t.Helper()
	result := &virtualProvider{shards: map[string]*virtualShard{}}
	byShard := map[string]map[string]any{}
	sizes := map[string]int64{}
	for _, spec := range p.Tensors() {
		if spec.Trainable {
			continue
		}
		row := checkpoint.Tensor{Name: spec.SourceName, DType: "BF16"}
		for _, dim := range spec.Shape {
			row.Shape = append(row.Shape, uint64(dim))
		}
		if mutate != nil {
			mutate(spec.SourceName, &row)
		}
		width := int64(2)
		if row.DType == "F64" {
			width = 8
		}
		if row.DType == "BOOL" {
			width = 1
		}
		size := width
		for _, dim := range row.Shape {
			size *= int64(dim)
		}
		if byShard[spec.Shard] == nil {
			byShard[spec.Shard] = map[string]any{}
		}
		start := sizes[spec.Shard]
		byShard[spec.Shard][spec.SourceName] = map[string]any{"dtype": row.DType, "shape": row.Shape, "data_offsets": []int64{start, start + size}}
		sizes[spec.Shard] += size
	}
	for name, rows := range byShard {
		raw, err := json.Marshal(rows)
		if err != nil {
			t.Fatal(err)
		}
		header := make([]byte, 8+len(raw))
		binary.LittleEndian.PutUint64(header, uint64(len(raw)))
		copy(header[8:], raw)
		result.shards[name] = &virtualShard{header: header, size: int64(len(header)) + sizes[name]}
	}
	return result
}

func assertOnlyHeadersAndClosed(t *testing.T, source *virtualProvider) {
	t.Helper()
	for name, shard := range source.shards {
		if shard.payloadReads != 0 {
			t.Fatalf("payload read:%s", name)
		}
		if shard.reads > 0 && shard.closes != 1 {
			t.Fatalf("close ownership:%s reads%d closes%d", name, shard.reads, shard.closes)
		}
	}
}

func TestInspectVirtualMultiGigabyteShardsOnlyReadsHeaders(t *testing.T) {
	p := plan(t, fixture())
	source := provider(t, p, nil)
	inspection, err := ornith.InspectAssemblySources(context.Background(), p, source)
	if err != nil {
		t.Fatal(err)
	}
	if inspection.BaseTensors != 427 || inspection.Shards != 4 || inspection.LargestCopyBytes != 6_102_712_320 || inspection.SourceBytes <= 15<<30 {
		t.Fatalf("inspection:%+v", inspection)
	}
	for _, shard := range source.shards {
		if shard.reads != 2 || shard.size <= 1<<30 {
			t.Fatalf("unexpected virtual shard:%+v", shard)
		}
	}
	assertOnlyHeadersAndClosed(t, source)
}

func TestInspectRejectsSourceTamperingAndLargerConversionBeforePayload(t *testing.T) {
	for _, test := range []struct {
		name   string
		mutate func(string, *checkpoint.Tensor)
	}{
		{"wrong_shape", func(name string, row *checkpoint.Tensor) {
			if strings.HasSuffix(name, ".dt_bias") {
				row.Shape = []uint64{31}
			}
		}},
		{"wrong_dtype", func(name string, row *checkpoint.Tensor) {
			if name == "lm_head.weight" {
				row.DType = "BOOL"
			}
		}},
		{"source_copy_budget", func(name string, row *checkpoint.Tensor) {
			if name == "lm_head.weight" {
				row.DType = "F64"
			}
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			p := plan(t, fixture())
			source := provider(t, p, test.mutate)
			_, err := ornith.InspectAssemblySources(context.Background(), p, source)
			if !errors.Is(err, ornith.ErrAssembly) {
				t.Fatalf("tampered source:%v", err)
			}
			assertOnlyHeadersAndClosed(t, source)
		})
	}
}

func TestInspectionCancellationOpenErrorsAndTruncationReleaseSources(t *testing.T) {
	p := plan(t, fixture())
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	source := provider(t, p, nil)
	if _, err := ornith.InspectAssemblySources(ctx, p, source); !errors.Is(err, context.Canceled) || source.opens != 0 {
		t.Fatal("canceled inspection opened sources")
	}
	ctx, cancel = context.WithCancel(context.Background())
	defer cancel()
	source = provider(t, p, nil)
	for _, shard := range source.shards {
		shard.afterRead = cancel
	}
	if _, err := ornith.InspectAssemblySources(ctx, p, source); !errors.Is(err, context.Canceled) {
		t.Fatalf("midheader cancellation:%v", err)
	}
	assertOnlyHeadersAndClosed(t, source)
	marker := errors.New("provider failed")
	source = provider(t, p, nil)
	source.openErr = marker
	if _, err := ornith.InspectAssemblySources(context.Background(), p, source); !errors.Is(err, marker) {
		t.Fatal(err)
	}
	if source.shards["model-00001-of-00004.safetensors"].closes != 1 {
		t.Fatal("handle accompanying open error leaked")
	}
	source = provider(t, p, nil)
	source.shards["model-00001-of-00004.safetensors"].header = []byte{1, 2, 3}
	if _, err := ornith.InspectAssemblySources(context.Background(), p, source); err == nil {
		t.Fatal("truncated header accepted")
	}
	assertOnlyHeadersAndClosed(t, source)
	source = provider(t, p, nil)
	source.shards["model-00001-of-00004.safetensors"].closeErr = marker
	if _, err := ornith.InspectAssemblySources(context.Background(), p, source); !errors.Is(err, marker) {
		t.Fatal("provider close failure swallowed")
	}
	assertOnlyHeadersAndClosed(t, source)
}

func TestLoadRejectsPartialOrReorderedInitialSetWithoutOpeningShards(t *testing.T) {
	p := plan(t, fixture())
	for _, initial := range []*ornith.InitialAdapter{nil, {SHA256: initialDigest, Parameters: make([]ornith.InitialParameter, 31)}, {SHA256: initialDigest, Parameters: make([]ornith.InitialParameter, 32)}} {
		source := provider(t, p, nil)
		result, err := ornith.LoadTextAssembly(context.Background(), p, source, initial)
		if result != nil || !errors.Is(err, ornith.ErrAssembly) || source.opens != 0 {
			t.Fatalf("partial initial set reached shards:%v", err)
		}
	}
	var empty *ornith.LoadedTextModel
	if err := empty.Close(); err != nil {
		t.Fatal(err)
	}
}

var _ io.ReaderAt = (*virtualShard)(nil)
