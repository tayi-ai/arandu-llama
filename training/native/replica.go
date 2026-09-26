package native

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"math"
	"reflect"
	"slices"
	"strings"
	"sync"

	decoder "github.com/tayi-ai/arandu-llama/training/decoder"
	"github.com/tayi-ai/arandu-llama/training/tokenizer"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

// ErrExperimentNativeReplica identifies invalid native identity, state or metadata.
var ErrExperimentNativeReplica = errors.New("experiment: native replica rejected")

// ExperimentDecoderReplicaOptions fixes explicit operation budgets and source identity.
// HashChunkBytes is in [4,4MiB]. ParameterCopyBytes admits three full adapter
// payloads (at least 6684672 bytes), separately from checkpoint, rotary, full
// logit seed and native workspace budgets. These bounds are payload accounting,
// not measured process RAM/VRAM admission. Limits must retain exactly two rows
// and permit 2..4096 sequence positions, including the appended A candidate.
type ExperimentDecoderReplicaOptions struct {
	Limits                        decoder.Limits
	HashChunkBytes                int
	ParameterCopyBytes            int64
	SourceManifestSHA256          string
	ExpectedRotaryFrequencySHA256 string
}

// ExperimentDecoderReplica implements the application replica using an already loaded
// native model. It owns loaded after successful construction; a rejected
// constructor leaves ownership with the caller. The tokenizer is immutable and
// borrowed. The caller must not directly mutate/use/close loaded after transfer.
// Operations and Close are serialized. A derivative callback must not reenter
// this replica. No operation changes the loss, optimizer or fleet admission.
type ExperimentDecoderReplica struct {
	experiment *NativeExperiment

	mu       sync.Mutex
	loaded   *decoder.LoadedTextModel
	codec    *tokenizer.Tokenizer
	options  ExperimentDecoderReplicaOptions
	specs    []decoder.AssemblyTensor
	summary  decoder.AssemblySummary
	admitted bool
	updated  bool
	closed   bool
}

var _ ExperimentNativeReplica = (*ExperimentDecoderReplica)(nil)

// NewExperimentDecoderReplica validates fixed model, tokenizer and assembly metadata.
// It creates no native tensor, loads no model and invokes no CUDA operation.
// The caller supplies the independently pinned assembly plan and previously
// admitted CUDA model. Inspect must verify live bytes before calculation.
func NewExperimentDecoderReplica(experiment *NativeExperiment, loaded *decoder.LoadedTextModel, codec *tokenizer.Tokenizer, plan *decoder.AssemblyPlan, options ExperimentDecoderReplicaOptions) (*ExperimentDecoderReplica, error) {
	if experiment == nil {
		return nil, errors.New("native experiment: installation is not configured")
	}

	if loaded == nil || loaded.Model == nil || codec == nil || plan == nil || codec.SHA256() !=
		experiment.config.TokenizerSHA256 ||
		options.SourceManifestSHA256 != ExperimentNativeRecipe(experiment).Model.ManifestSHA256 || options.ExpectedRotaryFrequencySHA256 !=
		experiment.config.RotarySHA256 ||
		options.HashChunkBytes < 4 || options.HashChunkBytes > 4<<20 || options.ParameterCopyBytes < int64(ExperimentNativeRecipe(experiment).LoRA.Parameters)*4*3 ||
		options.Limits.LogitRows != 2 || options.Limits.MaxTokens < 2 || options.Limits.MaxTokens > 4096 || options.Limits.MaxCheckpointBytes <= 0 {
		return nil, fmt.Errorf("%w: fixed identities and explicit budgets required", ErrExperimentNativeReplica)
	}
	if err := validateNativeRotary(experiment, plan); err != nil {
		return nil, err
	}
	summary := plan.Summary()
	if !reflect.DeepEqual(summary, loaded.Summary) || summary.BaseTensors != 427 || summary.AdapterTensors != 32 || summary.AdapterElements != 557056 ||
		summary.Identity.InitialAdapterSHA256 !=
			experiment.config.Recipe.LoRA.ExpectedInitialDigest ||
		summary.Identity.IndexSHA256 !=
			experiment.config.IndexSHA256 ||
		summary.Identity.ConfigSHA256 !=
			experiment.config.ConfigSHA256 ||
		summary.Identity.ReferenceSHA256 !=
			experiment.config.ReferenceSHA256 {
		return nil, fmt.Errorf("%w: assembly plan differs from the frozen reference", ErrExperimentNativeReplica)
	}
	specs := plan.Tensors()
	if len(specs) != 459 || len(loaded.Receipts) != len(specs) {
		return nil, fmt.Errorf("%w: incomplete assembly receipts", ErrExperimentNativeReplica)
	}
	for i, spec := range specs {
		if loaded.Receipts[i].Name != spec.ReferenceName || loaded.Receipts[i].SHA256 != spec.SHA256 {
			return nil, fmt.Errorf("%w: assembly receipt differs", ErrExperimentNativeReplica)
		}
	}
	result := &ExperimentDecoderReplica{experiment: experiment, loaded: loaded, codec: codec, options: options, specs: specs, summary: summary}
	if _, err := result.registry(); err != nil {
		return nil, err
	}
	return result, nil
}

// Inspect hashes live base and adapter values using bounded native copies.
// BaseSHA256 is SHA256(name+raw bytes), in the assembly plan's 427-text-weight
// order. It excludes vision/MTP/rotary and is not the historical full-model hash.
// Every base content hash must equal its pinned reference on every inspection.
// The first successful inspection also verifies all 32 initial tensor hashes
// and the historical adapter aggregate before enabling Gradient or Install.
func (replica *ExperimentDecoderReplica) Inspect(ctx context.Context) (ExperimentNativeReplicaReceipt, error) {
	if replica == nil {
		return ExperimentNativeReplicaReceipt{}, ErrExperimentNativeReplica
	}
	replica.mu.Lock()
	defer replica.mu.Unlock()
	return replica.inspect(ctx)
}

func (replica *ExperimentDecoderReplica) inspect(ctx context.Context) (ExperimentNativeReplicaReceipt, error) {
	var receipt ExperimentNativeReplicaReceipt
	if err := replica.ready(ctx, false); err != nil {
		return receipt, err
	}
	registry, err := replica.registry()
	if err != nil {
		return receipt, err
	}
	base := sha256.New()
	for _, spec := range replica.specs {
		if spec.Trainable {
			continue
		}
		_, _ = base.Write([]byte(spec.ReferenceName))
		if _, err := experimentNativeScanTensor(ctx, registry[spec.ReferenceName], spec, replica.options.HashChunkBytes, base, nil); err != nil {
			return receipt, err
		}
	}
	digest, _, err := replica.readParameters(ctx, false)
	if err != nil {
		return receipt, err
	}
	if !replica.updated && digest !=
		replica.experiment.config.Recipe.LoRA.ExpectedInitialDigest {
		return receipt, fmt.Errorf("%w: live initial adapter identity differs", ErrExperimentNativeReplica)
	}
	replica.admitted = true
	return ExperimentNativeReplicaReceipt{BaseSHA256: hex.EncodeToString(base.Sum(nil)), AdapterSHA256: digest,
		SourceManifestSHA256: replica.options.SourceManifestSHA256, Layout: ExperimentNativeParameterLayout()}, nil
}

// InitialParameters returns a detached, finite Go FP32 copy for rank-zero AdamW.
// It performs initial Inspect if necessary and refuses any updated adapter.
// Its name/content aggregate must equal the fixed historical initialization.
func (replica *ExperimentDecoderReplica) InitialParameters(ctx context.Context) ([]float32, error) {
	if replica == nil {
		return nil, ErrExperimentNativeReplica
	}
	replica.mu.Lock()
	defer replica.mu.Unlock()
	if err := replica.ready(ctx, false); err != nil {
		return nil, err
	}
	if replica.updated {
		return nil, fmt.Errorf("%w: initial parameters requested after update", ErrExperimentNativeReplica)
	}
	if !replica.admitted {
		if _, err := replica.inspect(ctx); err != nil {
			return nil, err
		}
	}
	digest, values, err := replica.readParameters(ctx, true)
	if err != nil {
		return nil, err
	}
	if digest !=
		replica.experiment.config.Recipe.LoRA.ExpectedInitialDigest {
		return nil, fmt.Errorf("%w: initial parameters changed", ErrExperimentNativeReplica)
	}
	return values, nil
}

// Gradient encodes the raw prompt, appends candidate A once, binds temporary
// device-specific rotary tables and delegates forward/VJP to CandidateGradient.
// It preserves the callback derivative exactly; scaling belongs to the caller.
// All temporary tables close after restoring every borrowed rotary reference.
func (replica *ExperimentDecoderReplica) Gradient(ctx context.Context, example ExperimentNativeExample, derivative ExperimentNativeLossGradient) (result ExperimentNativeGradient, err error) {
	if replica == nil {
		return result, ErrExperimentNativeReplica
	}
	replica.mu.Lock()
	defer replica.mu.Unlock()
	if err := replica.ready(ctx, true); err != nil {
		return result, err
	}
	if derivative == nil {
		return result, fmt.Errorf("%w: derivative required", ErrExperimentNativeReplica)
	}
	ids, restore, err := replica.prepare(ctx, example)
	if err != nil {
		return result, err
	}
	defer func() {
		err = errors.Join(err, restore())
		if err != nil {
			result = ExperimentNativeGradient{}
		}
	}()
	candidates := [4]int64{}
	for i, id := range example.CandidateVocabularyIDs {
		candidates[i] = int64(id)
	}
	native, err := decoder.CandidateGradient(ctx, replica.loaded.Model, ids, candidates, replica.options.Limits, decoder.LossGradient(derivative))
	if err != nil {
		return result, err
	}
	layout := ExperimentNativeParameterLayout()
	if len(native.Gradients) != len(layout) {
		return result, fmt.Errorf("%w: incomplete native gradient set", ErrExperimentNativeReplica)
	}
	result.Logits, result.InputSequenceLength = native.Logits, int64(len(ids))
	result.ValuesF32 = make([]float32, 0, ExperimentNativeRecipe(replica.experiment).LoRA.Parameters)
	for i, gradient := range native.Gradients {
		if err := ctx.Err(); err != nil {
			return result, err
		}
		if gradient.Name != layout[i].Name || !slices.Equal(gradient.Shape, layout[i].Shape) || int64(len(gradient.ValuesF32)) != layout[i].Shape[0]*layout[i].Shape[1] {
			return result, fmt.Errorf("%w: gradient name, shape or order differs", ErrExperimentNativeReplica)
		}
		result.ValuesF32 = append(result.ValuesF32, gradient.ValuesF32...)
	}
	if err := experimentNativeValues(replica.experiment, result.ValuesF32); err != nil {
		return result, err
	}
	return result, ctx.Err()
}

// ExperimentNativeCalibration records four scores normalized over the full 248320
// vocabulary, using FP32 native logits and stable Go float64 log-sum-exp. This
// explicitly differs in arithmetic precision from the historical scorer; the
// caller must compare the frozen argmax and tolerance, not claim bitwise parity.
type ExperimentNativeCalibration struct {
	LogProbabilities    [4]float64 `json:"log_probabilities_abcd"`
	InputSequenceLength int64      `json:"input_sequence_length"`
}

// Calibration scores the inspected fresh adapter without gradients. It selects
// the first of two retained rows, copies exactly 248320 FP32 values, and closes
// its snapshot, row view and rotary tables on every return. An updated adapter
// cannot be compared as if it were the frozen initial calibration.
func (replica *ExperimentDecoderReplica) Calibration(ctx context.Context, example ExperimentNativeExample) (result ExperimentNativeCalibration, err error) {
	return replica.CalibrationObserved(ctx, example, nil)
}

// CalibrationObserved preserves the calibration calculation and optionally
// reports bounded activation statistics through the native model contract.
// The observer receives no tensor handles and must not reenter this replica.
func (replica *ExperimentDecoderReplica) CalibrationObserved(ctx context.Context, example ExperimentNativeExample, observer decoder.ForwardObserver) (result ExperimentNativeCalibration, err error) {
	if replica == nil {
		return result, ErrExperimentNativeReplica
	}
	replica.mu.Lock()
	defer replica.mu.Unlock()
	if err := replica.ready(ctx, true); err != nil {
		return result, err
	}
	if replica.updated {
		return result, fmt.Errorf("%w: calibration requires the initial adapter", ErrExperimentNativeReplica)
	}
	ids, restore, err := replica.prepare(ctx, example)
	if err != nil {
		return result, err
	}
	var snapshot *decoder.Snapshot
	var row *torch.Tensor
	defer func() {
		err = errors.Join(err, row.Close(), snapshot.Close(), restore())
		if err != nil {
			result = ExperimentNativeCalibration{}
		}
	}()
	snapshot, err = replica.loaded.Model.ForwardObserved(ctx, ids, replica.options.Limits, observer)
	if err != nil {
		return result, err
	}
	info, err := snapshot.Logits.Info()
	if err != nil {
		return result, err
	}
	if info.DType != torch.Float32 || !slices.Equal(info.Shape, []int64{1, 2, 248320}) {
		return result, fmt.Errorf("%w: calibration logit geometry differs", ErrExperimentNativeReplica)
	}
	row, err = snapshot.Logits.Select(1, 0)
	if err != nil {
		return result, err
	}
	values, err := row.Float32Values()
	if err != nil {
		return result, err
	}
	if err := ctx.Err(); err != nil {
		return result, err
	}
	result.LogProbabilities, err = ExperimentNativeVocabularyLogProbabilities(replica.experiment, values)
	result.InputSequenceLength = int64(len(ids))
	return result, errors.Join(err, ctx.Err())
}

// ExperimentNativeVocabularyLogProbabilities computes initial-calibration scores
// using all 248320 supplied FP32 logits, not a four-choice softmax. The returned
// values are Go float64 log probabilities in the recipe's fixed A/B/C/D order.
func ExperimentNativeVocabularyLogProbabilities(experiment *NativeExperiment, logits []float32) ([4]float64, error) {
	if experiment == nil {
		return [4]float64{}, errors.New("native experiment: installation is not configured")
	}

	var result [4]float64
	if len(logits) != 248320 {
		return result, fmt.Errorf("%w: complete fixed vocabulary required", ErrExperimentNativeReplica)
	}
	maximum := float64(logits[0])
	for _, value := range logits {
		if !experimentFinite32(value) {
			return result, fmt.Errorf("%w: nonfinite calibration logit", ErrExperimentNativeReplica)
		}
		maximum = math.Max(maximum, float64(value))
	}
	sum := 0.0
	for _, value := range logits {
		sum += math.Exp(float64(value) - maximum)
	}
	logSum := math.Log(sum)
	for i, id := range ExperimentNativeRecipe(experiment).Data.CandidateVocabularyIDs {
		result[i] = (float64(logits[id]) - maximum) - logSum
	}
	return result, nil
}

func (replica *ExperimentDecoderReplica) prepare(ctx context.Context, example ExperimentNativeExample) ([]int64, func() error, error) {
	if example.ExampleID == "" || example.LogitRows != 2 || example.CandidateVocabularyIDs != ExperimentNativeRecipe(replica.experiment).Data.CandidateVocabularyIDs || !strings.HasSuffix(example.Prompt, "\nAnswer:") {
		return nil, nil, fmt.Errorf("%w: raw prompt or candidates differ", ErrExperimentNativeReplica)
	}
	if _, err := replica.registry(); err != nil {
		return nil, nil, err
	}
	ids, err := replica.codec.Encode(ctx, example.Prompt)
	if err != nil {
		return nil, nil, err
	}
	if len(ids) == 0 || int64(len(ids)) >= replica.options.Limits.MaxTokens {
		return nil, nil, fmt.Errorf("%w: encoded prompt exceeds explicit sequence budget", ErrExperimentNativeReplica)
	}
	ids = append(ids, int64(example.CandidateVocabularyIDs[0]))
	var tables [2]*decoder.Rotary
	for device := range tables {
		tables[device], err = decoder.TextRotary(ctx, len(ids), torch.CUDADevice(device), replica.experiment.config.Rotary)
		if err != nil {
			return nil, nil, errors.Join(err, tables[0].Close(), tables[1].Close())
		}
	}
	model := replica.loaded.Model
	var previous [32][2]*torch.Tensor
	for i := range model.Layers {
		previous[i] = [2]*torch.Tensor{model.Layers[i].Cosine, model.Layers[i].Sine}
		if i%4 == 3 {
			model.Layers[i].Cosine, model.Layers[i].Sine = tables[i/16].Cosine, tables[i/16].Sine
		}
	}
	restore := func() error {
		for i := range model.Layers {
			model.Layers[i].Cosine, model.Layers[i].Sine = previous[i][0], previous[i][1]
		}
		return errors.Join(tables[0].Close(), tables[1].Close())
	}
	return ids, restore, nil
}

// Install delegates atomic, verified FP32 leaf replacement to the loaded owner.
// A release error after commit remains an error and must abort the owning job.
func (replica *ExperimentDecoderReplica) Install(ctx context.Context, values []float32) (string, error) {
	if replica == nil {
		return "", ErrExperimentNativeReplica
	}
	replica.mu.Lock()
	defer replica.mu.Unlock()
	if err := replica.ready(ctx, true); err != nil {
		return "", err
	}
	if _, err := replica.registry(); err != nil {
		return "", err
	}
	digest, err := replica.loaded.ReplaceParameters(ctx, values)
	if digest != "" {
		replica.updated = true
	}
	return digest, err
}

// Close releases the transferred loaded model once; it never closes tokenizer
// data or preexisting caller-owned rotary tables. A nil receiver is harmless.
func (replica *ExperimentDecoderReplica) Close() error {
	if replica == nil {
		return nil
	}
	replica.mu.Lock()
	defer replica.mu.Unlock()
	if replica.closed {
		return nil
	}
	replica.closed = true
	err := replica.loaded.Close()
	replica.loaded, replica.codec, replica.specs = nil, nil, nil
	return err
}

func (replica *ExperimentDecoderReplica) ready(ctx context.Context, admitted bool) error {
	if ctx == nil || replica.closed || replica.loaded == nil || replica.loaded.Model == nil || replica.codec == nil || admitted && !replica.admitted {
		return fmt.Errorf("%w: open, inspected replica and context required", ErrExperimentNativeReplica)
	}
	return ctx.Err()
}

func (replica *ExperimentDecoderReplica) readParameters(ctx context.Context, copyValues bool) (string, []float32, error) {
	registry, err := replica.registry()
	if err != nil {
		return "", nil, err
	}
	digest := sha256.New()
	var values []float32
	var consume func([]byte)
	if copyValues {
		values = make([]float32, 0, ExperimentNativeRecipe(replica.experiment).LoRA.Parameters)
		consume = func(content []byte) {
			for i := 0; i < len(content); i += 4 {
				values = append(values, math.Float32frombits(binary.LittleEndian.Uint32(content[i:])))
			}
		}
	}
	count := 0
	for _, spec := range replica.specs {
		if !spec.Trainable {
			continue
		}
		if replica.updated {
			spec.SHA256 = "" // Updated values have no initial per-tensor identity.
		}
		_, _ = digest.Write([]byte(spec.ReferenceName))
		if _, err := experimentNativeScanTensor(ctx, registry[spec.ReferenceName], spec, replica.options.HashChunkBytes, digest, consume); err != nil {
			return "", nil, err
		}
		count++
	}
	if count != 32 || copyValues && len(values) != ExperimentNativeRecipe(replica.experiment).LoRA.Parameters {
		return "", nil, fmt.Errorf("%w: incomplete live parameter registry", ErrExperimentNativeReplica)
	}
	return hex.EncodeToString(digest.Sum(nil)), values, ctx.Err()
}

// ExperimentNativeTensorDigest verifies one live FP16/FP32 tensor against its required
// assembly identity. Slice/Select views bound every Bytes and AllFinite operation
// to chunkBytes, including noncontiguous input; it never flattens/copies an entire
// tensor before chunking. Caller owns value and keeps it immutable for the call.
// Cancellation is checked between chunks, not inside running native operations.
func ExperimentNativeTensorDigest(ctx context.Context, value *torch.Tensor, expected decoder.AssemblyTensor, chunkBytes int) (string, error) {
	if !experimentNativeHash(expected.SHA256) {
		return "", fmt.Errorf("%w: tensor content identity required", ErrExperimentNativeReplica)
	}
	return experimentNativeScanTensor(ctx, value, expected, chunkBytes, nil, nil)
}

func experimentNativeScanTensor(ctx context.Context, value *torch.Tensor, expected decoder.AssemblyTensor, chunkBytes int, aggregate hash.Hash, consume func([]byte)) (result string, err error) {
	if ctx == nil || value == nil || chunkBytes < 4 || chunkBytes > 4<<20 || (expected.DType != torch.Float16 && expected.DType != torch.Float32) {
		return "", fmt.Errorf("%w: tensor, FP16/FP32 and bounded chunk required", ErrExperimentNativeReplica)
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	info, err := value.Info()
	if err != nil {
		return "", err
	}
	width := int64(2)
	if info.DType == torch.Float32 {
		width = 4
	}
	if info.DType != expected.DType || info.Device != expected.Device || info.RequiresGrad != expected.Trainable || !slices.Equal(info.Shape, expected.Shape) || info.Elements <= 0 || info.Elements > math.MaxInt64/width || info.Elements*width != expected.Bytes {
		return "", fmt.Errorf("%w: live tensor metadata differs for %s", ErrExperimentNativeReplica, expected.ReferenceName)
	}
	digest := sha256.New()
	var visit func(*torch.Tensor, []int64, int64) error
	visit = func(part *torch.Tensor, shape []int64, elements int64) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if elements*width <= int64(chunkBytes) {
			finite, err := part.AllFinite()
			if err != nil || !finite {
				return errors.Join(fmt.Errorf("%w: nonfinite live tensor", ErrExperimentNativeReplica), err)
			}
			content, err := part.Bytes()
			if err != nil {
				return err
			}
			if int64(len(content)) != elements*width {
				return fmt.Errorf("%w: native tensor byte count differs", ErrExperimentNativeReplica)
			}
			_, _ = digest.Write(content)
			if aggregate != nil {
				_, _ = aggregate.Write(content)
			}
			if consume != nil {
				consume(content)
			}
			return ctx.Err()
		}
		if len(shape) == 0 || shape[0] <= 0 || elements%shape[0] != 0 {
			return fmt.Errorf("%w: invalid tensor geometry", ErrExperimentNativeReplica)
		}
		rowElements := elements / shape[0]
		if rowElements*width > int64(chunkBytes) {
			for i := int64(0); i < shape[0]; i++ {
				view, err := part.Select(0, i)
				if err != nil {
					return err
				}
				err = errors.Join(visit(view, shape[1:], rowElements), view.Close())
				if err != nil {
					return err
				}
			}
			return nil
		}
		rowsPerChunk := int64(chunkBytes) / (rowElements * width)
		for start := int64(0); start < shape[0]; {
			end := min(shape[0], start+rowsPerChunk)
			view, err := part.Slice(0, start, end, 1)
			if err != nil {
				return err
			}
			err = errors.Join(visit(view, nil, (end-start)*rowElements), view.Close())
			if err != nil {
				return err
			}
			start = end
		}
		return nil
	}
	if err := visit(value, info.Shape, info.Elements); err != nil {
		return "", err
	}
	result = hex.EncodeToString(digest.Sum(nil))
	if expected.SHA256 != "" && result != expected.SHA256 {
		return "", fmt.Errorf("%w: live tensor content differs for %s", ErrExperimentNativeReplica, expected.ReferenceName)
	}
	return result, ctx.Err()
}

func (replica *ExperimentDecoderReplica) registry() (map[string]*torch.Tensor, error) {
	if replica.loaded == nil || replica.loaded.Model == nil || !reflect.DeepEqual(replica.loaded.Summary, replica.summary) {
		return nil, fmt.Errorf("%w: loaded owner identity differs", ErrExperimentNativeReplica)
	}
	m := replica.loaded.Model
	if len(m.Layers) != 32 || len(replica.loaded.Parameters) != 32 || m.Epsilon != 1e-6 {
		return nil, fmt.Errorf("%w: fixed model geometry required", ErrExperimentNativeReplica)
	}
	values := make(map[string]*torch.Tensor, 459)
	add := func(source string, value *torch.Tensor) { values["base_model.model."+source] = value }
	add("model.language_model.embed_tokens.weight", m.Embedding)
	add("model.language_model.norm.weight", m.FinalNorm)
	add("lm_head.weight", m.Head)
	for i, layer := range m.Layers {
		if layer.Device != torch.CUDADevice(i/16) || layer.Config.Epsilon != 1e-6 || layer.Config.MaxInputElements <= 0 {
			return nil, fmt.Errorf("%w: decoder placement or configuration differs", ErrExperimentNativeReplica)
		}
		prefix := fmt.Sprintf("model.language_model.layers.%d.", i)
		w := layer.Weights
		add(prefix+"input_layernorm.weight", w.InputNorm)
		add(prefix+"post_attention_layernorm.weight", w.PostAttentionNorm)
		add(prefix+"mlp.gate_proj.weight", w.Gate)
		add(prefix+"mlp.up_proj.weight", w.Up)
		add(prefix+"mlp.down_proj.weight", w.Down)
		if i%4 == 3 {
			c, a, f := layer.Config.Full, layer.Adapter, w.Full
			if f == nil || w.Linear != nil || a == nil || a.Alpha != 8 || c.Heads != 16 || c.KVHeads != 4 || c.HeadDimension != 256 || c.RotaryDimension != 64 || c.Epsilon != 1e-6 || c.MaxScoreElements <= 0 {
				return nil, fmt.Errorf("%w: full attention differs", ErrExperimentNativeReplica)
			}
			for _, item := range []struct {
				suffix string
				value  *torch.Tensor
			}{
				{"q_proj.base_layer.weight", f.Query}, {"k_proj.weight", f.Key}, {"v_proj.base_layer.weight", f.Value},
				{"o_proj.weight", f.Output}, {"q_norm.weight", f.QueryNorm}, {"k_norm.weight", f.KeyNorm},
				{"q_proj.lora_A.default.weight", a.QueryA}, {"q_proj.lora_B.default.weight", a.QueryB},
				{"v_proj.lora_A.default.weight", a.ValueA}, {"v_proj.lora_B.default.weight", a.ValueB},
			} {
				add(prefix+"self_attn."+item.suffix, item.value)
			}
		} else {
			c, l := layer.Config.Linear, w.Linear
			if l == nil || w.Full != nil || layer.Adapter != nil || c.KeyHeads != 16 || c.ValueHeads != 32 || c.KeyDimension != 128 || c.ValueDimension != 128 || c.Epsilon != 1e-6 || c.MaxWorkingElements <= 0 || c.Sequence.ChunkTokens <= 0 || c.Sequence.MaxTokens <= 0 || c.Sequence.MaxOwnedElements <= 0 {
				return nil, fmt.Errorf("%w: linear attention differs", ErrExperimentNativeReplica)
			}
			for _, item := range []struct {
				suffix string
				value  *torch.Tensor
			}{
				{"in_proj_qkv.weight", l.QKV}, {"in_proj_z.weight", l.Z}, {"in_proj_b.weight", l.Beta}, {"in_proj_a.weight", l.Alpha},
				{"conv1d.weight", l.Convolution}, {"A_log", l.ALog}, {"dt_bias", l.DTBias}, {"norm.weight", l.Norm}, {"out_proj.weight", l.Output},
			} {
				add(prefix+"linear_attn."+item.suffix, item.value)
			}
		}
	}
	if len(values) != len(replica.specs) {
		return nil, fmt.Errorf("%w: live registry count differs", ErrExperimentNativeReplica)
	}
	seen := make(map[torch.Tensor]bool, len(values))
	for _, spec := range replica.specs {
		value := values[spec.ReferenceName]
		if value == nil || seen[*value] {
			return nil, fmt.Errorf("%w: missing or aliased tensor handle", ErrExperimentNativeReplica)
		}
		seen[*value] = true
	}
	layout := ExperimentNativeParameterLayout()
	for i, parameter := range replica.loaded.Parameters {
		if parameter.Name != layout[i].Name || parameter.Value != values[parameter.Name] {
			return nil, fmt.Errorf("%w: adapter registry/model references differ", ErrExperimentNativeReplica)
		}
	}
	return values, nil
}

// The frequency hash authenticates bytes; geometry additionally binds their
// meaning to the actual decoder configuration before loading or calculation.
func validateNativeRotary(experiment *NativeExperiment, plan *decoder.AssemblyPlan) error {
	if experiment == nil || plan == nil {
		return ErrExperimentNativeReplica
	}
	return validateNativeRotaryGeometry(experiment, plan.Geometry())
}

func validateNativeRotaryGeometry(experiment *NativeExperiment, geometry decoder.TextGeometry) error {
	if experiment == nil {
		return ErrExperimentNativeReplica
	}
	rotary := experiment.config.Rotary
	if rotary.Theta != geometry.RoPE.Theta || rotary.Dimension != int(float64(geometry.Dimension)*geometry.RoPE.Partial) {
		return errors.New("native experiment: rotary specification differs from decoder geometry")
	}
	return nil
}
