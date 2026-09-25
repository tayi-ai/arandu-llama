//go:build libtorch && cgo

// Local training of the frozen Arasa base through bounded completion updates.
// Job ownership and queue state remain with the application.
package local

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
	"time"

	"github.com/tayi-ai/arandu-llama/checkpoint"
	"github.com/tayi-ai/arandu-llama/training/ornith"
	"github.com/tayi-ai/arandu-llama/training/sequence"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

const (
	initialDigest = "75185d68fcfdd09bb2dcb6dffe0902a35300f52d9fda716b408189991773aa7c"
	rotaryDigest  = "ec4437c15ead01576c3e6c7412c11daba363d17cee8bfce5c5d1c54e2e09d2d0"
	exampleID     = "arandu-quality-v3/CORE-E12"
)

type fileShard struct {
	*os.File
	size int64
}

func (s *fileShard) Size() int64 { return s.size }

type files struct{ root string }

func (f files) OpenShard(ctx context.Context, name string) (ornith.Shard, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	switch name {
	case "model-00001-of-00004.safetensors", "model-00002-of-00004.safetensors", "model-00003-of-00004.safetensors", "model-00004-of-00004.safetensors":
	default:
		return nil, errors.New("unadmitted shard name")
	}
	path := filepath.Join(f.root, name)
	before, err := os.Lstat(path)
	if err != nil || !before.Mode().IsRegular() {
		return nil, fmt.Errorf("shard is not a regular file: %s: %w", name, err)
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	after, err := file.Stat()
	if err != nil || !os.SameFile(before, after) {
		_ = file.Close()
		return nil, errors.New("shard changed during open")
	}
	return &fileShard{File: file, size: after.Size()}, nil
}

type example struct {
	ID           string  `json:"id"`
	InputIDs     []int64 `json:"input_ids"`
	Labels       []int64 `json:"labels"`
	PromptTokens int     `json:"prompt_tokens"`
}

func selected(path string) (example, error) {
	return selectedID(path, exampleID)
}

func selectedID(path, id string) (example, error) {
	file, err := os.Open(path)
	if err != nil {
		return example{}, err
	}
	defer file.Close()
	decoder := json.NewDecoder(file)
	for {
		var row example
		if err := decoder.Decode(&row); err != nil {
			if errors.Is(err, io.EOF) {
				return example{}, errors.New("selected example absent")
			}
			return example{}, err
		}
		if row.ID != id {
			continue
		}
		if len(row.InputIDs) < 2 || len(row.InputIDs) > 4096 || len(row.Labels) != len(row.InputIDs) || row.PromptTokens < 1 || row.PromptTokens >= len(row.InputIDs) {
			return example{}, errors.New("selected example geometry differs")
		}
		for index, label := range row.Labels {
			if index < row.PromptTokens && label != -100 || index >= row.PromptTokens && label != row.InputIDs[index] {
				return example{}, errors.New("selected example mask differs")
			}
		}
		return row, nil
	}
}

func run(ctx context.Context, bundle, modelDir, data, output, reload, nextID string, curriculumMax, curriculumSteps int) (err error) {
	read := func(name, expected string) ([]byte, error) {
		body, err := os.ReadFile(filepath.Join(bundle, name))
		if err != nil {
			return nil, err
		}
		actual := sha256.Sum256(body)
		if hex.EncodeToString(actual[:]) != expected {
			return nil, fmt.Errorf("%s digest differs", name)
		}
		return body, nil
	}
	index, err := read("model.safetensors.index.json", "d5c7fee99574e9a05f901282aee04fc4fc3dccf094a659df48c6b8e9f39109c3")
	if err != nil {
		return err
	}
	config, err := read("config.json", "1f1b3751c38f16a63340df90a55e870bef0f0b2968d833825a605b7cf930a313")
	if err != nil {
		return err
	}
	reference, err := read("initial-reference.json", "69825f715b8f422e2e155be4bd2ed309e14d77ac9d45a65405a3b5886ca228dc")
	if err != nil {
		return err
	}
	rowID := exampleID
	if nextID != "" {
		rowID = nextID
	}
	row, err := selectedID(data, rowID)
	if err != nil {
		return err
	}
	tokens := int64(len(row.InputIDs))
	if curriculumMax > 0 {
		tokens = int64(curriculumMax)
	}
	limits := ornith.AssemblyLimits{HeaderLimits: checkpoint.DefaultLimits(), TensorCopyBytes: 6_102_712_320,
		PersistentBytes: [2]int64{11_042_374_656, 11_042_382_848}, HashChunkBytes: 4 << 20,
		MaxInputElements: tokens * 4096, MaxScoreElements: 16 * tokens * tokens, MaxWorkingElements: 1 << 30,
		Sequence: sequence.SequenceLimits{ChunkTokens: 8, MaxTokens: tokens, MaxOwnedElements: 1 << 30}}
	plan, err := ornith.PlanLocalMPSAssembly(index, config, reference, ornith.AssemblyIdentity{
		IndexSHA256:          "d5c7fee99574e9a05f901282aee04fc4fc3dccf094a659df48c6b8e9f39109c3",
		ConfigSHA256:         "1f1b3751c38f16a63340df90a55e870bef0f0b2968d833825a605b7cf930a313",
		ReferenceSHA256:      "69825f715b8f422e2e155be4bd2ed309e14d77ac9d45a65405a3b5886ca228dc",
		InitialAdapterSHA256: initialDigest,
	}, limits, 25<<30)
	if err != nil {
		return err
	}
	fmt.Printf("phase=plan tensors=%d local_mps_bytes=%d\n", len(plan.Tensors()), plan.Summary().LocalMPSBytes)
	provider := files{root: modelDir}
	inspection, err := ornith.InspectAssemblySources(ctx, plan, provider)
	if err != nil {
		return err
	}
	fmt.Printf("phase=inspect shards=%d tensors=%d\n", inspection.Shards, inspection.BaseTensors)
	initial, err := ornith.InitializeAdapter(ctx, ornith.InitialAdapterSpec{Seed: 83, PreludeBlocks: 24, ExpectedSHA256: initialDigest})
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, initial.Close()) }()
	start := time.Now()
	loaded, err := ornith.LoadTextAssembly(ctx, plan, provider, initial)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, loaded.Close()) }()
	fmt.Printf("phase=loaded elapsed=%s receipts=%d\n", time.Since(start), len(loaded.Receipts))
	if curriculumMax > 0 {
		return runCurriculum(ctx, loaded, data, reload, output, curriculumMax, curriculumSteps)
	}
	tables, err := ornith.TextRotary(ctx, len(row.InputIDs), torch.MPSDevice(), rotaryDigest)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, tables.Close()) }()
	for index := range loaded.Model.Layers {
		if index%4 == 3 {
			loaded.Model.Layers[index].Cosine, loaded.Model.Layers[index].Sine = tables.Cosine, tables.Sine
		}
	}
	if nextID != "" {
		return resumeNext(ctx, loaded, row, reload, output)
	}
	if reload != "" {
		if err := reloadAndVerifyAdapter(ctx, loaded, row, reload); err != nil {
			return err
		}
		fmt.Printf("phase=reloaded_adapter_verified elapsed=%s\n", time.Since(start))
		return nil
	}
	var before string
	if output != "" {
		before, err = forwardFingerprint(ctx, loaded.Model, row.InputIDs, int64(row.PromptTokens))
		if err != nil {
			return err
		}
		fmt.Printf("phase=base_forward logits_sha256=%s elapsed=%s\n", before, time.Since(start))
	}
	gradient, err := ornith.CompletionGradient(ctx, loaded.Model, row.InputIDs, row.PromptTokens,
		ornith.Limits{MaxTokens: tokens, LogitRows: tokens - int64(row.PromptTokens) + 1, MaxCheckpointBytes: tokens * 4096 * 68}, 1)
	if err != nil {
		return err
	}
	var nonzero int
	for _, item := range gradient.Gradients {
		for _, value := range item.ValuesF32 {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return errors.New("nonfinite gradient")
			}
			if value != 0 {
				nonzero++
			}
		}
	}
	if math.IsNaN(gradient.Loss) || math.IsInf(gradient.Loss, 0) || nonzero == 0 {
		return errors.New("invalid completion result")
	}
	fmt.Printf("phase=gradient example=%s loss=%g tokens=%d gradients=%d nonzero=%d elapsed=%s\n",
		row.ID, gradient.Loss, gradient.Tokens, len(gradient.Gradients), nonzero, time.Since(start))
	if output != "" {
		if err := applyAndVerifyFirstUpdate(ctx, loaded, row, gradient, before, output); err != nil {
			return err
		}
		fmt.Printf("phase=step_1_verified elapsed=%s\n", time.Since(start))
	}
	return nil
}
