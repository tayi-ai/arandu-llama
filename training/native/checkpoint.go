package native

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"

	"github.com/tayi-ai/arandu-llama/checkpoint"
)

// ErrExperimentNativeCheckpoint identifies failed checkpoint admission or persistence.
var ErrExperimentNativeCheckpoint = errors.New("experiment: native checkpoint rejected")

// ExperimentNativeCheckpointFile identifies exact bytes in the published directory.
type ExperimentNativeCheckpointFile struct {
	Name   string `json:"name"`
	SHA256 string `json:"sha256"`
	Bytes  int64  `json:"bytes"`
}

// ExperimentNativeCheckpointManifest records the fixed export contract and verified
// file identities. AdapterSHA256 uses original parameter names; ArtifactSHA256
// hashes the safetensors container with standard PEFT export names.
type ExperimentNativeCheckpointManifest struct {
	SchemaVersion       int                              `json:"schema_version"`
	Format              string                           `json:"format"`
	NumericMode         string                           `json:"numeric_mode"`
	Step                int                              `json:"step"`
	Candidate           bool                             `json:"candidate"`
	AdapterSHA256       string                           `json:"adapter_sha256"`
	ArtifactSHA256      string                           `json:"artifact_sha256"`
	ScheduleSHA256      string                           `json:"schedule_sha256"`
	ModelRepository     string                           `json:"model_repository"`
	ModelRevision       string                           `json:"model_revision"`
	ModelManifestSHA256 string                           `json:"model_manifest_sha256"`
	RecipeVersion       string                           `json:"recipe_version"`
	Tensors             int                              `json:"tensors"`
	Parameters          int                              `json:"parameters"`
	Layout              []ExperimentNativeParameter      `json:"original_parameter_layout"`
	Files               []ExperimentNativeCheckpointFile `json:"files"`
}

type experimentPEFTConfig struct {
	BaseModel       string   `json:"base_model_name_or_path"`
	Revision        string   `json:"revision"`
	PEFTType        string   `json:"peft_type"`
	TaskType        string   `json:"task_type"`
	Rank            int      `json:"r"`
	Alpha           int      `json:"lora_alpha"`
	Dropout         float64  `json:"lora_dropout"`
	Bias            string   `json:"bias"`
	TargetModules   []string `json:"target_modules"`
	InferenceMode   bool     `json:"inference_mode"`
	FanInFanOut     bool     `json:"fan_in_fan_out"`
	InitLoRAWeights bool     `json:"init_lora_weights"`
	UseRSLoRA       bool     `json:"use_rslora"`
	UseDoRA         bool     `json:"use_dora"`
}

// ExperimentNativeFileCheckpointWriter persists only the frozen step-9 and step-18
// artifacts. The owner must exclusively control the existing job directory and
// must not mutate its entries through other APIs while this writer is active.
// A permanent exclusive claim prevents another writer or process from resuming
// the same job directory. No files are overwritten or automatically removed.
// On failure, .step-N.partial remains unpublished for job-scoped inspection and
// cleanup by the caller. A post-rename sync failure can leave step-N present but
// returns no successful receipt; the job must abort without retrying it.
type ExperimentNativeFileCheckpointWriter struct {
	experiment *NativeExperiment

	mu        sync.Mutex
	root      *os.Root
	directory *os.File
	optimizer *ExperimentAdamW
	failed    bool
}

// NewExperimentNativeCheckpointWriter requires an absolute, existing directory with
// no symlink path component. It confines I/O to an opened os.Root, creates and
// syncs an exclusive permanent writer claim, and does not create the job root.
// The optional optimizer is borrowed; each write snapshots it and requires the
// exact checkpoint step and bitwise-identical parameter vector. No resume API is
// provided. The caller closes the writer after rank-zero checkpoint work ends.
func NewExperimentNativeCheckpointWriter(experiment *NativeExperiment, jobDir string, optimizer *ExperimentAdamW) (_ *ExperimentNativeFileCheckpointWriter, err error) {
	if experiment == nil {
		return nil, errors.New("native experiment: installation is not configured")
	}

	if !filepath.IsAbs(jobDir) || filepath.Clean(jobDir) != jobDir || jobDir == string(filepath.Separator) {
		return nil, ErrExperimentNativeCheckpoint
	}
	for path := jobDir; ; path = filepath.Dir(path) {
		info, err := os.Lstat(path)
		if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
			return nil, errors.Join(ErrExperimentNativeCheckpoint, err)
		}
		if filepath.Dir(path) == path {
			break
		}
	}
	before, err := os.Lstat(jobDir)
	if err != nil {
		return nil, errors.Join(ErrExperimentNativeCheckpoint, err)
	}
	root, err := os.OpenRoot(jobDir)
	if err != nil {
		return nil, errors.Join(ErrExperimentNativeCheckpoint, err)
	}
	writer := &ExperimentNativeFileCheckpointWriter{experiment: experiment, root: root, optimizer: optimizer}
	complete := false
	defer func() {
		if !complete {
			err = errors.Join(err, writer.Close())
		}
	}()
	writer.directory, err = root.Open(".")
	if err != nil {
		return nil, errors.Join(ErrExperimentNativeCheckpoint, err)
	}
	after, err := writer.directory.Stat()
	if err != nil || !os.SameFile(before, after) {
		return nil, errors.Join(ErrExperimentNativeCheckpoint, err)
	}
	claim, err := root.OpenFile(".experiment-checkpoint-writer", os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return nil, errors.Join(ErrExperimentNativeCheckpoint, err)
	}
	_, writeErr := claim.WriteString("experiment-native-checkpoint-writer-v1\nno-resume\n")
	err = errors.Join(writeErr, claim.Sync(), claim.Close())
	if err != nil {
		return nil, errors.Join(ErrExperimentNativeCheckpoint, err)
	}
	if err := writer.directory.Sync(); err != nil {
		return nil, errors.Join(ErrExperimentNativeCheckpoint, err)
	}
	complete = true
	return writer, nil
}

// Close releases directory handles, preserving the claim and all artifacts.
// It waits for an active synchronous write; cancellation uses that write's context.
func (writer *ExperimentNativeFileCheckpointWriter) Close() error {
	if writer == nil {
		return nil
	}
	writer.mu.Lock()
	defer writer.mu.Unlock()
	var failures []error
	if writer.directory != nil {
		failures = append(failures, writer.directory.Close())
		writer.directory = nil
	}
	if writer.root != nil {
		failures = append(failures, writer.root.Close())
		writer.root = nil
	}
	return errors.Join(failures...)
}

// WriteCheckpoint validates before creating staging, writes and syncs each file,
// reloads all 32 tensor shapes and FP32 values, verifies config and optimizer
// identities, then syncs and atomically renames the directory within the root.
// A failure consumes this writer; partial files are retained, never retried.
func (writer *ExperimentNativeFileCheckpointWriter) WriteCheckpoint(ctx context.Context, request ExperimentNativeCheckpoint) (receipt ExperimentNativeCheckpointReceipt, err error) {
	if writer == nil {
		return receipt, ErrExperimentNativeCheckpoint
	}
	writer.mu.Lock()
	defer writer.mu.Unlock()
	if writer.root == nil || writer.directory == nil || writer.failed {
		return receipt, ErrExperimentNativeCheckpoint
	}
	defer func() {
		if err != nil {
			writer.failed = true
			receipt = ExperimentNativeCheckpointReceipt{}
		}
	}()
	if ctx == nil {
		return receipt, ErrExperimentNativeCheckpoint
	}
	if err := ctx.Err(); err != nil {
		return receipt, err
	}
	if (request.Step != 9 && request.Step != 18) || request.Candidate != (request.Step == 18) ||
		!experimentNativeHash(request.AdapterSHA256) || !experimentNativeHash(request.ScheduleSHA256) ||
		!reflect.DeepEqual(request.Layout, ExperimentNativeParameterLayout()) {
		return receipt, ErrExperimentNativeCheckpoint
	}
	parameterDigest, err := ExperimentNativeParameterDigest(writer.experiment, request.Parameters)
	if err != nil || parameterDigest != request.AdapterSHA256 {
		return receipt, errors.Join(ErrExperimentNativeCheckpoint, err)
	}
	var optimizerJSON []byte
	if writer.optimizer != nil {
		state := writer.optimizer.Snapshot()
		if err := experimentCheckpointOptimizer(state, request); err != nil {
			return receipt, err
		}
		optimizerJSON, err = experimentCheckpointJSON(state)
		if err != nil {
			return receipt, err
		}
	}
	config := experimentCheckpointConfig(writer.experiment)
	configJSON, err := experimentCheckpointJSON(config)
	if err != nil {
		return receipt, err
	}
	if err := ctx.Err(); err != nil {
		return receipt, err
	}
	final, staging := fmt.Sprintf("step-%d", request.Step), fmt.Sprintf(".step-%d.partial", request.Step)
	if err := experimentCheckpointAbsent(writer.root, final); err != nil {
		return receipt, err
	}
	if err := writer.root.Mkdir(staging, 0700); err != nil {
		return receipt, errors.Join(ErrExperimentNativeCheckpoint, err)
	}
	stage, err := writer.root.OpenRoot(staging)
	if err != nil {
		return receipt, errors.Join(ErrExperimentNativeCheckpoint, err)
	}
	defer stage.Close()
	if err := writer.directory.Sync(); err != nil {
		return receipt, errors.Join(ErrExperimentNativeCheckpoint, err)
	}
	tensors := make([]checkpoint.Float32Tensor, 0, 32)
	position := 0
	for _, item := range request.Layout {
		count := int(item.Shape[0] * item.Shape[1])
		tensors = append(tensors, checkpoint.Float32Tensor{Name: experimentCheckpointExportName(item.Name), Shape: []uint64{uint64(item.Shape[0]), uint64(item.Shape[1])}, Values: request.Parameters[position : position+count]})
		position += count
	}
	limits := experimentCheckpointLimits()
	file, err := stage.OpenFile("adapter_model.safetensors", os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return receipt, errors.Join(ErrExperimentNativeCheckpoint, err)
	}
	written, writeErr := checkpoint.WriteFloat32(ctx, file, tensors, limits)
	if err := errors.Join(writeErr, file.Sync(), file.Close()); err != nil {
		return receipt, errors.Join(ErrExperimentNativeCheckpoint, err)
	}
	files := []ExperimentNativeCheckpointFile{{Name: "adapter_model.safetensors", SHA256: written.SHA256, Bytes: written.Bytes}}
	configFile, err := experimentCheckpointWriteFile(ctx, stage, "adapter_config.json", configJSON)
	if err != nil {
		return receipt, err
	}
	files = append(files, configFile)
	if optimizerJSON != nil {
		optimizerFile, err := experimentCheckpointWriteFile(ctx, stage, "optimizer.json", optimizerJSON)
		if err != nil {
			return receipt, err
		}
		files = append(files, optimizerFile)
	}
	if err := experimentCheckpointReload(writer.experiment, ctx, stage, request, files, configJSON, optimizerJSON); err != nil {
		return receipt, err
	}
	recipe := ExperimentNativeRecipe(writer.experiment)
	manifest := ExperimentNativeCheckpointManifest{SchemaVersion: 1, Format: "peft-safetensors-f32-v1", NumericMode: ExperimentAdamWNumericMode,
		Step: request.Step, Candidate: request.Candidate, AdapterSHA256: request.AdapterSHA256, ArtifactSHA256: written.SHA256,
		ScheduleSHA256: request.ScheduleSHA256, ModelRepository: recipe.Model.Repository, ModelRevision: recipe.Model.Revision,
		ModelManifestSHA256: recipe.Model.ManifestSHA256, RecipeVersion: recipe.Version,
		Tensors: 32, Parameters: 557056, Layout: ExperimentNativeParameterLayout(), Files: files}
	manifestJSON, err := experimentCheckpointJSON(manifest)
	if err != nil {
		return receipt, err
	}
	if _, err := experimentCheckpointWriteFile(ctx, stage, "receipt.json", manifestJSON); err != nil {
		return receipt, err
	}
	reloadedManifest, err := stage.ReadFile("receipt.json")
	if err != nil || !bytes.Equal(reloadedManifest, manifestJSON) {
		return receipt, errors.Join(ErrExperimentNativeCheckpoint, err)
	}
	directory, err := stage.Open(".")
	if err != nil {
		return receipt, errors.Join(ErrExperimentNativeCheckpoint, err)
	}
	if err := errors.Join(directory.Sync(), directory.Close()); err != nil {
		return receipt, errors.Join(ErrExperimentNativeCheckpoint, err)
	}
	if err := stage.Close(); err != nil {
		return receipt, errors.Join(ErrExperimentNativeCheckpoint, err)
	}
	if err := ctx.Err(); err != nil {
		return receipt, err
	}
	// The exclusive permanent claim and private job root exclude another writer
	// between this absence check and rename. Arbitrary external mutations are not
	// permitted by this owner's contract; os.Root additionally prevents escapes.
	if err := experimentCheckpointAbsent(writer.root, final); err != nil {
		return receipt, err
	}
	if err := writer.root.Rename(staging, final); err != nil {
		return receipt, errors.Join(ErrExperimentNativeCheckpoint, err)
	}
	if err := writer.directory.Sync(); err != nil {
		return receipt, errors.Join(ErrExperimentNativeCheckpoint, err)
	}
	return ExperimentNativeCheckpointReceipt{Step: request.Step, Candidate: request.Candidate, AdapterSHA256: request.AdapterSHA256, ArtifactSHA256: written.SHA256}, nil
}

func experimentCheckpointConfig(experiment *NativeExperiment) experimentPEFTConfig {
	recipe := ExperimentNativeRecipe(experiment)
	return experimentPEFTConfig{BaseModel: recipe.Model.Repository, Revision: recipe.Model.Revision, PEFTType: "LORA", TaskType: "CAUSAL_LM",
		Rank: 4, Alpha: 8, Dropout: 0, Bias: "none", TargetModules: []string{"q_proj", "v_proj"}, InferenceMode: true, InitLoRAWeights: true}
}

func experimentCheckpointExportName(name string) string {
	return strings.TrimSuffix(name, ".default.weight") + ".weight"
}

func experimentCheckpointLimits() checkpoint.Limits {
	return checkpoint.Limits{MaxHeaderBytes: 64 << 10, MaxTensors: 32, MaxDimensions: 2, MaxMetadataEntries: 1, MaxChunkBytes: 64 << 10}
}

func experimentCheckpointJSON(value any) ([]byte, error) {
	body, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return nil, errors.Join(ErrExperimentNativeCheckpoint, err)
	}
	return append(body, '\n'), nil
}

func experimentCheckpointAbsent(root *os.Root, name string) error {
	_, err := root.Lstat(name)
	if !errors.Is(err, os.ErrNotExist) {
		return errors.Join(ErrExperimentNativeCheckpoint, err, os.ErrExist)
	}
	return nil
}

func experimentCheckpointWriteFile(ctx context.Context, root *os.Root, name string, body []byte) (ExperimentNativeCheckpointFile, error) {
	if err := ctx.Err(); err != nil {
		return ExperimentNativeCheckpointFile{}, err
	}
	file, err := root.OpenFile(name, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return ExperimentNativeCheckpointFile{}, errors.Join(ErrExperimentNativeCheckpoint, err)
	}
	_, writeErr := io.Copy(file, &experimentCheckpointContextReader{ctx: ctx, reader: bytes.NewReader(body)})
	if err := errors.Join(writeErr, file.Sync(), file.Close()); err != nil {
		return ExperimentNativeCheckpointFile{}, errors.Join(ErrExperimentNativeCheckpoint, err)
	}
	digest := sha256.Sum256(body)
	return ExperimentNativeCheckpointFile{Name: name, SHA256: hex.EncodeToString(digest[:]), Bytes: int64(len(body))}, nil
}

type experimentCheckpointContextReader struct {
	ctx    context.Context
	reader io.Reader
}

func (r *experimentCheckpointContextReader) Read(body []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	return r.reader.Read(body[:min(len(body), 64<<10)])
}

func experimentCheckpointOptimizer(state ExperimentAdamWState, request ExperimentNativeCheckpoint) error {
	if state.SchemaVersion != 1 || state.NumericMode != ExperimentAdamWNumericMode || state.Step != request.Step ||
		len(state.Parameters) != len(request.Parameters) || len(state.FirstMoment) != len(request.Parameters) || len(state.SecondMoment) != len(request.Parameters) {
		return ErrExperimentNativeCheckpoint
	}
	for i, value := range request.Parameters {
		if math.Float32bits(state.Parameters[i]) != math.Float32bits(value) || !experimentFinite32(state.FirstMoment[i]) ||
			!experimentFinite32(state.SecondMoment[i]) || state.SecondMoment[i] < 0 {
			return ErrExperimentNativeCheckpoint
		}
	}
	return nil
}

func experimentCheckpointReload(experiment *NativeExperiment, ctx context.Context, root *os.Root, request ExperimentNativeCheckpoint, files []ExperimentNativeCheckpointFile, configJSON, optimizerJSON []byte) error {
	if experiment == nil {
		return errors.New("native experiment: installation is not configured")
	}

	for _, expected := range files {
		if err := ctx.Err(); err != nil {
			return err
		}
		info, err := root.Lstat(expected.Name)
		if err != nil || !info.Mode().IsRegular() || info.Size() != expected.Bytes {
			return errors.Join(ErrExperimentNativeCheckpoint, err)
		}
		file, err := root.Open(expected.Name)
		if err != nil {
			return errors.Join(ErrExperimentNativeCheckpoint, err)
		}
		hash := sha256.New()
		_, readErr := io.Copy(hash, &experimentCheckpointContextReader{ctx: ctx, reader: file})
		closeErr := file.Close()
		if readErr != nil || closeErr != nil || hex.EncodeToString(hash.Sum(nil)) != expected.SHA256 {
			return errors.Join(ErrExperimentNativeCheckpoint, readErr, closeErr)
		}
	}
	config, err := root.ReadFile("adapter_config.json")
	var decoded experimentPEFTConfig
	if err != nil || !bytes.Equal(config, configJSON) || json.Unmarshal(config, &decoded) != nil || !reflect.DeepEqual(decoded, experimentCheckpointConfig(experiment)) {
		return errors.Join(ErrExperimentNativeCheckpoint, err)
	}
	if optimizerJSON != nil {
		body, err := root.ReadFile("optimizer.json")
		var state ExperimentAdamWState
		if err != nil || !bytes.Equal(body, optimizerJSON) || json.Unmarshal(body, &state) != nil {
			return errors.Join(ErrExperimentNativeCheckpoint, err)
		}
		if err := experimentCheckpointOptimizer(state, request); err != nil {
			return err
		}
	}
	file, err := root.Open("adapter_model.safetensors")
	if err != nil {
		return errors.Join(ErrExperimentNativeCheckpoint, err)
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return errors.Join(ErrExperimentNativeCheckpoint, err)
	}
	index, err := checkpoint.OpenSafetensors(file, info.Size(), experimentCheckpointLimits())
	if err != nil || len(index.Tensors()) != 32 {
		return errors.Join(ErrExperimentNativeCheckpoint, err)
	}
	if !reflect.DeepEqual(index.Metadata(), map[string]string{"format": "pt"}) {
		return ErrExperimentNativeCheckpoint
	}
	position := 0
	aggregate := sha256.New()
	var buffer [64 << 10]byte
	for _, expected := range request.Layout {
		if err := ctx.Err(); err != nil {
			return err
		}
		name := experimentCheckpointExportName(expected.Name)
		tensor, ok := index.Tensor(name)
		shape := []uint64{uint64(expected.Shape[0]), uint64(expected.Shape[1])}
		if !ok || tensor.DType != "F32" || !reflect.DeepEqual(tensor.Shape, shape) || tensor.Size() != expected.Shape[0]*expected.Shape[1]*4 {
			return ErrExperimentNativeCheckpoint
		}
		reader, err := index.TensorReader(name)
		if err != nil {
			return errors.Join(ErrExperimentNativeCheckpoint, err)
		}
		_, _ = aggregate.Write([]byte(expected.Name))
		remaining := tensor.Size()
		for remaining > 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
			part := buffer[:min(int64(len(buffer)), remaining)]
			if _, err := io.ReadFull(reader, part); err != nil {
				return errors.Join(ErrExperimentNativeCheckpoint, err)
			}
			for i := 0; i < len(part); i += 4 {
				if binary.LittleEndian.Uint32(part[i:]) != math.Float32bits(request.Parameters[position]) {
					return ErrExperimentNativeCheckpoint
				}
				position++
			}
			_, _ = aggregate.Write(part)
			remaining -= int64(len(part))
		}
	}
	if position != len(request.Parameters) || hex.EncodeToString(aggregate.Sum(nil)) != request.AdapterSHA256 {
		return ErrExperimentNativeCheckpoint
	}
	return ctx.Err()
}

var _ ExperimentNativeCheckpointWriter = (*ExperimentNativeFileCheckpointWriter)(nil)
