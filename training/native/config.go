package native

import (
	"encoding/json"
	"errors"
	"io"
	"maps"
	"math"
	"path/filepath"
	"slices"
	"strings"

	"github.com/tayi-ai/arandu-llama/checkpoint"
	decoder "github.com/tayi-ai/arandu-llama/training/decoder"
)

// NativeExperimentConfig binds one numerical recipe to deployment-owned files.
// No model, dataset, initialization or topology is implicitly trusted.
type NativeExperimentConfig struct {
	Assembly             decoder.AssemblyLimits     `json:"assembly"`
	Initializer          decoder.InitialAdapterSpec `json:"initializer"`
	Rotary               decoder.RotarySpec         `json:"rotary"`
	Version              int                        `json:"schema_version"`
	Recipe               ExperimentRecipe           `json:"recipe"`
	ModelRecipe          string                     `json:"model_recipe"`
	BundleName           string                     `json:"bundle_name"`
	ManifestSHA256       string                     `json:"manifest_sha256"`
	LegacyManifestSHA256 string                     `json:"legacy_manifest_sha256,omitempty"`
	BasePath             string                     `json:"base_path"`
	TokenizerSHA256      string                     `json:"tokenizer_sha256"`
	RotarySHA256         string                     `json:"rotary_sha256"`
	IndexSHA256          string                     `json:"index_sha256"`
	ConfigSHA256         string                     `json:"config_sha256"`
	ReferenceSHA256      string                     `json:"reference_sha256"`
}

// NativeExperiment owns a detached immutable recipe. Independent experiments
// can coexist without changing process globals or another experiment's pins.
type NativeExperiment struct {
	config                   NativeExperimentConfig
	sourcePath, sourceSHA256 string
}

func NewNativeExperiment(config NativeExperimentConfig) (*NativeExperiment, error) {
	r := config.Recipe
	if config.Version != 1 || r.Version == "" || config.ModelRecipe == "" || strings.TrimSpace(r.Model.Repository) == "" || strings.TrimSpace(r.Model.Revision) == "" ||
		!filepath.IsAbs(config.BasePath) || filepath.Clean(config.BasePath) != config.BasePath || config.BundleName == "" || filepath.Base(config.BundleName) != config.BundleName || config.BundleName == "." || config.BundleName == ".." {
		return nil, errors.New("native experiment: explicit deployment identity is required")
	}
	for _, digest := range []string{config.ManifestSHA256, config.TokenizerSHA256, config.RotarySHA256, config.IndexSHA256, config.ConfigSHA256, config.ReferenceSHA256, r.Model.ManifestSHA256, r.Data.SHA256, r.LoRA.ExpectedInitialDigest} {
		if !experimentSHA256(digest) {
			return nil, errors.New("native experiment: invalid artifact identity")
		}
	}
	if config.LegacyManifestSHA256 != "" && !experimentSHA256(config.LegacyManifestSHA256) {
		return nil, errors.New("native experiment: invalid legacy identity")
	}
	// These are the capabilities of this numerical kernel, not a default recipe.
	// Other geometries require a separately qualified backend implementation.
	if r.Data.ID == "" || r.Data.Examples != experimentExamples || r.Data.Demonstrations != 4 || r.Data.Epochs != 1 || r.Data.SelectionAllowed || r.Data.SealedAllowed ||
		r.LoRA.Rank != 4 || r.LoRA.Alpha != 8 || r.LoRA.Tensors != 32 || r.LoRA.Parameters != 557056 || r.LoRA.Targets != [2]string{"q_proj", "v_proj"} ||
		r.Optimization.Updates != ExperimentAdamWMaxUpdates || r.Optimization.LearningRate != ExperimentAdamWLearningRate || r.Optimization.Betas != [2]float64{ExperimentAdamWBeta1, ExperimentAdamWBeta2} ||
		r.Optimization.Epsilon != ExperimentAdamWEpsilon || r.Optimization.WeightDecay != ExperimentAdamWWeightDecay || r.Optimization.GradientClip != ExperimentAdamWGradientClip || r.Optimization.BackwardLossScale != ExperimentAdamWLossScale ||
		r.Distribution.Processes != ExperimentAdamWReplicas || r.Distribution.GlobalBatch != ExperimentAdamWGlobalBatch || r.Distribution.GPUsPerProcess != 2 || r.Distribution.GPUs != 2*ExperimentAdamWReplicas {
		return nil, errors.New("native experiment: recipe geometry or numerical mode is not supported by this backend")
	}
	if r.Version != "native-go-v2" || !r.Model.Frozen || r.Model.BasePrecision != "float16" || r.Model.AttentionPrecision != "float32" || r.Data.Protocol != "raw_four_shot_answer_space" || r.LoRA.Dropout != 0 || r.LoRA.Bias != "none" || r.LoRA.Precision != "float32" || !r.LoRA.Fresh || r.Optimization.Optimizer != "AdamW" || r.Optimization.Seed != 83 || r.Optimization.ClipEpsilon != 1e-12 || r.Optimization.SaveSteps != [2]int{9, 18} || r.Optimization.CandidateStep != 18 || r.Optimization.Reduction != "sum_20_rank_local_sums_divide_21_examples" || r.Optimization.UpdateOrder != "scale_each_loss_backward_sum_local_fp32_sum_rank_order_divide_21_unscale_global_norm_clip_adamw" || r.Distribution.LocalBatch != 2 || r.Distribution.LocalBatchMode != "sequential_extra_example_on_rank_update_minus_one" || r.Distribution.LayersPerGPU != [2]int{16, 16} || r.Guards.VRAMFraction != .8 || r.Guards.HostReserveBytes != 6<<30 || !r.Guards.PreloadRAMQualificationRequired || r.Guards.CanarySeconds != 1800 || r.Guards.TrainSeconds != 5400 || r.Guards.TechnicalRetries != 0 || r.Guards.CalibrationRows != 8 || r.Guards.ScoreTolerance != .03 || r.Guards.BaseFileCount != 18 || r.Guards.AutomaticPromotion || r.Backend.ReadyGPU || r.Backend.BitwiseEquivalent || r.Backend.Status != "unqualified" || r.Backend.LossPrecision != "go_float64" {
		return nil, errors.New("native experiment: unqualified numerical or execution capability")
	}
	if len(config.Assembly.PersistentBytes) != 2 || len(config.Assembly.DeviceByLayer) != 32 || config.Assembly.EmbeddingDevice != 0 || config.Assembly.OutputDevice != 1 || config.Assembly.AdapterRank != 4 || config.Assembly.AdapterAlpha != 8 || config.Initializer.ExpectedSHA256 != r.LoRA.ExpectedInitialDigest || len(config.Initializer.Projections) == 0 || config.Rotary.ExpectedSHA256 != config.RotarySHA256 || config.Rotary.MaxTokens < 4096 || config.Rotary.Dimension < 2 || config.Rotary.Theta <= 0 || math.IsNaN(config.Rotary.Theta) || math.IsInf(config.Rotary.Theta, 0) {
		return nil, errors.New("native experiment: explicit assembly, initialization and rotary configuration required")
	}
	if err := validateNativeAssemblyLimits(config.Assembly); err != nil {
		return nil, err
	}
	for layer, device := range config.Assembly.DeviceByLayer {
		if device != layer/16 {
			return nil, errors.New("native experiment: unsupported layer placement")
		}
	}
	for _, size := range config.Assembly.PersistentBytes {
		if size < 1 {
			return nil, errors.New("native experiment: invalid persistent memory budget")
		}
	}
	seen := map[string]bool{}
	for _, id := range r.Distribution.Nodes {
		if !experimentJobID.MatchString(id) || seen[id] {
			return nil, errors.New("native experiment: invalid or duplicate node identity")
		}
		seen[id] = true
	}
	candidates := map[int]bool{}
	for _, id := range r.Data.CandidateVocabularyIDs {
		if id < 0 || id >= 248320 || candidates[id] {
			return nil, errors.New("native experiment: invalid candidate token identity")
		}
		candidates[id] = true
	}
	if len(r.Data.TeacherRules) == 0 {
		return nil, errors.New("native experiment: teacher rules are required")
	}
	for category, rule := range r.Data.TeacherRules {
		if category == "" || rule.Gate == "" || math.IsNaN(rule.Weight) || math.IsInf(rule.Weight, 0) || rule.Weight < 0 || rule.Weight > 1 {
			return nil, errors.New("native experiment: invalid teacher rule")
		}
		total := 0.0
		for id, weight := range rule.Sources {
			if id == "" || math.IsNaN(weight) || math.IsInf(weight, 0) || weight <= 0 {
				return nil, errors.New("native experiment: invalid teacher mixture")
			}
			total += weight
		}
		if (rule.Weight == 0 && len(rule.Sources) != 0) || (rule.Weight > 0 && (len(rule.Sources) == 0 || math.Abs(total-1) > 1e-12)) {
			return nil, errors.New("native experiment: teacher mixture must have unit mass")
		}
	}
	config.Recipe.Data.TeacherRules = maps.Clone(config.Recipe.Data.TeacherRules)
	for key, rule := range config.Recipe.Data.TeacherRules {
		rule.Sources = maps.Clone(rule.Sources)
		config.Recipe.Data.TeacherRules[key] = rule
	}
	config.Recipe.Backend.UnqualifiedCapabilities = slices.Clone(config.Recipe.Backend.UnqualifiedCapabilities)
	body, err := json.Marshal(config)
	if err != nil {
		return nil, err
	}
	var owned NativeExperimentConfig
	if err = json.Unmarshal(body, &owned); err != nil {
		return nil, err
	}
	return &NativeExperiment{config: owned}, nil
}

// Configuration must supply every assembly budget. Smaller parsing limits are
// preserved; larger ones cannot expand the existing loader admission. Required
// tensor/execution payloads are checked against these caps once tokenization
// determines the sequence length, before weight loading or GPU admission.
func validateNativeAssemblyLimits(limits decoder.AssemblyLimits) error {
	h, ceiling := limits.HeaderLimits, checkpoint.DefaultLimits()
	if h.MaxHeaderBytes <= 0 || h.MaxHeaderBytes > ceiling.MaxHeaderBytes || h.MaxTensors <= 0 || h.MaxTensors > ceiling.MaxTensors ||
		h.MaxDimensions < 3 || h.MaxDimensions > ceiling.MaxDimensions || h.MaxMetadataEntries <= 0 || h.MaxMetadataEntries > ceiling.MaxMetadataEntries ||
		h.MaxChunkBytes <= 0 || h.MaxChunkBytes > ceiling.MaxChunkBytes || limits.HashChunkBytes < 4 || limits.HashChunkBytes > 4<<20 ||
		limits.TensorCopyBytes <= 0 || limits.MaxInputElements <= 0 || limits.MaxScoreElements <= 0 || limits.MaxWorkingElements <= 0 ||
		limits.Sequence.ChunkTokens != 8 || limits.Sequence.MaxTokens <= 0 || limits.Sequence.MaxOwnedElements <= 0 {
		return errors.New("native experiment: explicit assembly budgets and supported parsing and sequence limits are required")
	}
	return nil
}

// LoadNativeExperiment reads one hash-bound installation document. Callers pass
// the resulting object explicitly; neither a job body nor a user request owns it.
func LoadNativeExperiment(path, expected string) (*NativeExperiment, error) {
	body, err := readTrainingDocument(path, expected)
	if err != nil {
		return nil, err
	}
	var config NativeExperimentConfig
	if err := experimentDecode(body, &config, true); err != nil {
		return nil, errors.New("native experiment: invalid installation document")
	}
	e, err := NewNativeExperiment(config)
	if err != nil {
		return nil, err
	}
	e.sourcePath = path
	e.sourceSHA256 = expected
	return e, nil
}

// DecodeNativeExperiment permits deterministic fixtures and configuration tools
// to use the same typed validation without an environment or process singleton.
func DecodeNativeExperiment(reader io.Reader) (*NativeExperiment, error) {
	if reader == nil {
		return nil, errors.New("native experiment: missing reader")
	}
	body, err := io.ReadAll(io.LimitReader(reader, 1<<20+1))
	if err != nil || len(body) > 1<<20 {
		return nil, errors.New("native experiment: oversized or unreadable configuration")
	}
	var config NativeExperimentConfig
	if err := experimentDecode(body, &config, true); err != nil {
		return nil, err
	}
	return NewNativeExperiment(config)
}

func (e *NativeExperiment) BundleName() string {
	if e == nil {
		return ""
	}
	return e.config.BundleName
}

func (e *NativeExperiment) RecipeSHA256() string {
	if e == nil {
		return ""
	}
	identity := e.config
	identity.ManifestSHA256 = ""
	identity.LegacyManifestSHA256 = ""
	body, _ := json.Marshal(identity)
	return experimentHash(body)
}
