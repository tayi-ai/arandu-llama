package native_test

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/tayi-ai/arandu-llama/checkpoint"
	services "github.com/tayi-ai/arandu-llama/training/native"
)

func checkpointRoot(t *testing.T) string {
	t.Helper()
	// macOS's temporary directory can have a /var -> /private/var ancestor.
	// Pass the real path; the production constructor intentionally rejects links.
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	return root
}

func checkpointRequest(t *testing.T, step int) services.ExperimentNativeCheckpoint {
	t.Helper()
	values := make([]float32, 557056)
	for i := range values {
		values[i] = float32((i%97)-48) / 4096
	}
	values[0], values[1], values[2] = math.Float32frombits(0x80000000), math.SmallestNonzeroFloat32, math.MaxFloat32
	digest, err := services.ExperimentNativeParameterDigest(testNativeExperiment(), values)
	if err != nil {
		t.Fatal(err)
	}
	return services.ExperimentNativeCheckpoint{Step: step, Candidate: step == 18, AdapterSHA256: digest,
		ScheduleSHA256: strings.Repeat("c", 64), Layout: services.ExperimentNativeParameterLayout(), Parameters: values}
}

func checkpointWriter(t *testing.T, root string, optimizer *services.ExperimentAdamW) *services.ExperimentNativeFileCheckpointWriter {
	t.Helper()
	writer, err := services.NewExperimentNativeCheckpointWriter(testNativeExperiment(), root, optimizer)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = writer.Close() })
	return writer
}

func checkpointRead(t *testing.T, path string) []byte {
	t.Helper()
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return body
}

func checkpointHash(body []byte) string {
	sum := sha256.Sum256(body)
	return hex.EncodeToString(sum[:])
}

func assertCheckpoint(t *testing.T, root string, request services.ExperimentNativeCheckpoint, receipt services.ExperimentNativeCheckpointReceipt, optimizer bool) {
	t.Helper()
	directory := filepath.Join(root, "step-"+map[int]string{9: "9", 18: "18"}[request.Step])
	body := checkpointRead(t, filepath.Join(directory, "adapter_model.safetensors"))
	if receipt.ArtifactSHA256 != checkpointHash(body) || receipt.ArtifactSHA256 == receipt.AdapterSHA256 || receipt.AdapterSHA256 != request.AdapterSHA256 || receipt.Step != request.Step || receipt.Candidate != request.Candidate {
		t.Fatal("receipt identities differ")
	}
	file, err := os.Open(filepath.Join(directory, "adapter_model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	index, err := checkpoint.OpenSafetensors(file, int64(len(body)), checkpoint.DefaultLimits())
	if err != nil {
		t.Fatal(err)
	}
	if len(index.Tensors()) != 32 || !reflect.DeepEqual(index.Metadata(), map[string]string{"format": "pt"}) {
		t.Fatal("container metadata differs")
	}
	position := 0
	for _, expected := range request.Layout {
		name := strings.Replace(expected.Name, ".default.weight", ".weight", 1)
		descriptor, ok := index.Tensor(name)
		if !ok || descriptor.DType != "F32" || !reflect.DeepEqual(descriptor.Shape, []uint64{uint64(expected.Shape[0]), uint64(expected.Shape[1])}) {
			t.Fatalf("tensor shape/name differs: %s", name)
		}
		if _, found := index.Tensor(expected.Name); found {
			t.Fatal("default adapter name leaked into export key")
		}
		reader, err := index.TensorReader(name)
		if err != nil {
			t.Fatal(err)
		}
		content, err := io.ReadAll(reader)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < len(content); i += 4 {
			if binary.LittleEndian.Uint32(content[i:]) != math.Float32bits(request.Parameters[position]) {
				t.Fatalf("tensor value changed at %d", position)
			}
			position++
		}
	}
	if position != 557056 {
		t.Fatal("parameter count differs")
	}
	var config struct {
		Base     string   `json:"base_model_name_or_path"`
		Revision string   `json:"revision"`
		Type     string   `json:"peft_type"`
		Task     string   `json:"task_type"`
		Rank     int      `json:"r"`
		Alpha    int      `json:"lora_alpha"`
		Dropout  float64  `json:"lora_dropout"`
		Bias     string   `json:"bias"`
		Targets  []string `json:"target_modules"`
	}
	if err := json.Unmarshal(checkpointRead(t, filepath.Join(directory, "adapter_config.json")), &config); err != nil {
		t.Fatal(err)
	}
	if config.Base != "fixture/student" || config.Revision != "fixture-revision" || config.Type != "LORA" || config.Task != "CAUSAL_LM" || config.Rank != 4 || config.Alpha != 8 || config.Dropout != 0 || config.Bias != "none" || !slices.Equal(config.Targets, []string{"q_proj", "v_proj"}) {
		t.Fatal("PEFT config differs")
	}
	var manifest services.ExperimentNativeCheckpointManifest
	if err := json.Unmarshal(checkpointRead(t, filepath.Join(directory, "receipt.json")), &manifest); err != nil {
		t.Fatal(err)
	}
	wantFiles := 2
	if optimizer {
		wantFiles++
	}
	if manifest.SchemaVersion != 1 || manifest.NumericMode != services.ExperimentAdamWNumericMode || manifest.ScheduleSHA256 != request.ScheduleSHA256 || manifest.ArtifactSHA256 != receipt.ArtifactSHA256 || manifest.AdapterSHA256 != request.AdapterSHA256 || manifest.Step != request.Step || manifest.Candidate != request.Candidate || manifest.Tensors != 32 || manifest.Parameters != 557056 || !reflect.DeepEqual(manifest.Layout, request.Layout) || len(manifest.Files) != wantFiles || manifest.ModelManifestSHA256 != services.ExperimentNativeRecipe(testNativeExperiment()).Model.ManifestSHA256 || manifest.RecipeVersion != services.ExperimentNativeRecipe(testNativeExperiment()).Version {
		t.Fatal("manifest identity differs")
	}
	for _, expected := range manifest.Files {
		content := checkpointRead(t, filepath.Join(directory, expected.Name))
		if int64(len(content)) != expected.Bytes || checkpointHash(content) != expected.SHA256 {
			t.Fatalf("file receipt differs: %s", expected.Name)
		}
	}
	if !optimizer {
		if _, err := os.Lstat(filepath.Join(directory, "optimizer.json")); !errors.Is(err, os.ErrNotExist) {
			t.Fatal("unexpected optimizer snapshot")
		}
	}
	if _, err := os.Lstat(filepath.Join(root, ".step-"+map[int]string{9: "9", 18: "18"}[request.Step]+".partial")); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("successful staging was not renamed")
	}
}

func TestNativeCheckpointWritesOnlyNineAndEighteenAndRoundTrips(t *testing.T) {
	root := checkpointRoot(t)
	writer := checkpointWriter(t, root, nil)
	var last services.ExperimentNativeCheckpoint
	for _, step := range []int{9, 18} {
		request := checkpointRequest(t, step)
		receipt, err := writer.WriteCheckpoint(context.Background(), request)
		if err != nil {
			t.Fatal(err)
		}
		assertCheckpoint(t, root, request, receipt, false)
		last = request
	}
	before := checkpointRead(t, filepath.Join(root, "step-18", "adapter_model.safetensors"))
	if receipt, err := writer.WriteCheckpoint(context.Background(), last); err == nil || receipt != (services.ExperimentNativeCheckpointReceipt{}) {
		t.Fatal("duplicate checkpoint accepted")
	}
	if checkpointHash(checkpointRead(t, filepath.Join(root, "step-18", "adapter_model.safetensors"))) != checkpointHash(before) {
		t.Fatal("duplicate changed existing checkpoint")
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := services.NewExperimentNativeCheckpointWriter(testNativeExperiment(), root, nil); err == nil {
		t.Fatal("job checkpoint writer resumed")
	}
}

func TestNativeCheckpointDeterministicBytesAcrossRoots(t *testing.T) {
	request := checkpointRequest(t, 18)
	first, second := checkpointRoot(t), checkpointRoot(t)
	for _, root := range []string{first, second} {
		if _, err := checkpointWriter(t, root, nil).WriteCheckpoint(context.Background(), request); err != nil {
			t.Fatal(err)
		}
	}
	for _, name := range []string{"adapter_model.safetensors", "adapter_config.json", "receipt.json"} {
		if checkpointHash(checkpointRead(t, filepath.Join(first, "step-18", name))) != checkpointHash(checkpointRead(t, filepath.Join(second, "step-18", name))) {
			t.Fatalf("nondeterministic %s", name)
		}
	}
}

func TestNativeCheckpointOptimizerStateIsExactAndReloadable(t *testing.T) {
	request := checkpointRequest(t, 9)
	state := services.ExperimentAdamWState{SchemaVersion: 1, NumericMode: services.ExperimentAdamWNumericMode, Step: 9,
		Parameters: slices.Clone(request.Parameters), FirstMoment: make([]float32, 557056), SecondMoment: make([]float32, 557056)}
	state.FirstMoment[0], state.SecondMoment[0] = -.25, .5
	optimizer, err := services.RestoreExperimentAdamW(state)
	if err != nil {
		t.Fatal(err)
	}
	root := checkpointRoot(t)
	receipt, err := checkpointWriter(t, root, optimizer).WriteCheckpoint(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	assertCheckpoint(t, root, request, receipt, true)
	var reloaded services.ExperimentAdamWState
	if err := json.Unmarshal(checkpointRead(t, filepath.Join(root, "step-9", "optimizer.json")), &reloaded); err != nil {
		t.Fatal(err)
	}
	restored, err := services.RestoreExperimentAdamW(reloaded)
	if err != nil {
		t.Fatal(err)
	}
	observed := restored.Snapshot()
	for i := range state.Parameters {
		if math.Float32bits(observed.Parameters[i]) != math.Float32bits(state.Parameters[i]) || math.Float32bits(observed.FirstMoment[i]) != math.Float32bits(state.FirstMoment[i]) || math.Float32bits(observed.SecondMoment[i]) != math.Float32bits(state.SecondMoment[i]) {
			t.Fatalf("optimizer round-trip mismatch at %d", i)
		}
	}
	if observed.Step != 9 {
		t.Fatal("optimizer step changed")
	}
	for _, mode := range []string{"step", "values"} {
		t.Run(mode, func(t *testing.T) {
			bad := state
			if mode == "step" {
				bad.Step = 8
			} else {
				bad.Parameters = slices.Clone(state.Parameters)
				bad.Parameters[1] = .2
			}
			optimizer, err := services.RestoreExperimentAdamW(bad)
			if err != nil {
				t.Fatal(err)
			}
			root := checkpointRoot(t)
			if _, err := checkpointWriter(t, root, optimizer).WriteCheckpoint(context.Background(), request); err == nil {
				t.Fatal("mismatched optimizer accepted")
			}
			if _, err := os.Lstat(filepath.Join(root, ".step-9.partial")); !errors.Is(err, os.ErrNotExist) {
				t.Fatal("invalid optimizer created staging")
			}
		})
	}
}

func TestNativeCheckpointRejectsLayoutValuesAndStepBeforeStaging(t *testing.T) {
	for _, mode := range []string{"step", "candidate", "digest", "schedule", "duplicate name", "shape", "order", "short", "NaN", "infinity"} {
		t.Run(mode, func(t *testing.T) {
			root := checkpointRoot(t)
			writer := checkpointWriter(t, root, nil)
			request := checkpointRequest(t, 9)
			switch mode {
			case "step":
				request.Step = 10
			case "candidate":
				request.Candidate = true
			case "digest":
				request.AdapterSHA256 = strings.Repeat("d", 64)
			case "schedule":
				request.ScheduleSHA256 = "invalid"
			case "duplicate name":
				request.Layout[1].Name = request.Layout[0].Name
			case "shape":
				request.Layout[0].Shape = []int64{4096, 4}
			case "order":
				request.Layout[0], request.Layout[2] = request.Layout[2], request.Layout[0]
			case "short":
				request.Parameters = request.Parameters[:10]
			case "NaN":
				request.Parameters[len(request.Parameters)-1] = float32(math.NaN())
			case "infinity":
				request.Parameters[len(request.Parameters)-1] = float32(math.Inf(-1))
			}
			if receipt, err := writer.WriteCheckpoint(context.Background(), request); err == nil || receipt != (services.ExperimentNativeCheckpointReceipt{}) {
				t.Fatal("invalid checkpoint admitted")
			}
			entries, err := os.ReadDir(root)
			if err != nil || len(entries) != 1 || entries[0].Name() != ".experiment-checkpoint-writer" {
				t.Fatalf("admission wrote staging: %v %v", entries, err)
			}
		})
	}
}

func TestNativeCheckpointRejectsSymlinksAndExistingDestinations(t *testing.T) {
	parent, outside := checkpointRoot(t), checkpointRoot(t)
	link := filepath.Join(parent, "job-link")
	if err := os.Symlink(outside, link); err != nil {
		t.Fatal(err)
	}
	if _, err := services.NewExperimentNativeCheckpointWriter(testNativeExperiment(), link, nil); err == nil {
		t.Fatal("root symlink accepted")
	}
	if err := os.Mkdir(filepath.Join(outside, "child"), 0700); err != nil {
		t.Fatal(err)
	}
	if _, err := services.NewExperimentNativeCheckpointWriter(testNativeExperiment(), filepath.Join(link, "child"), nil); err == nil {
		t.Fatal("ancestor symlink accepted")
	}
	if _, err := services.NewExperimentNativeCheckpointWriter(testNativeExperiment(), filepath.Join(parent, "missing"), nil); err == nil {
		t.Fatal("missing root created")
	}
	for _, mode := range []string{"final symlink", "stage symlink", "empty final directory", "writer symlink", "second writer"} {
		t.Run(mode, func(t *testing.T) {
			root := checkpointRoot(t)
			if mode == "writer symlink" {
				if err := os.Symlink(filepath.Join(outside, "claim-target"), filepath.Join(root, ".experiment-checkpoint-writer")); err != nil {
					t.Fatal(err)
				}
				if _, err := services.NewExperimentNativeCheckpointWriter(testNativeExperiment(), root, nil); err == nil {
					t.Fatal("writer claim symlink accepted")
				}
				return
			}
			writer := checkpointWriter(t, root, nil)
			if mode == "second writer" {
				if _, err := services.NewExperimentNativeCheckpointWriter(testNativeExperiment(), root, nil); err == nil {
					t.Fatal("second writer admitted")
				}
				return
			}
			if mode == "empty final directory" {
				if err := os.Mkdir(filepath.Join(root, "step-9"), 0700); err != nil {
					t.Fatal(err)
				}
			} else {
				name := "step-9"
				if mode == "stage symlink" {
					name = ".step-9.partial"
				}
				if err := os.Symlink(outside, filepath.Join(root, name)); err != nil {
					t.Fatal(err)
				}
			}
			if _, err := writer.WriteCheckpoint(context.Background(), checkpointRequest(t, 9)); err == nil {
				t.Fatal("preexisting destination accepted")
			}
		})
	}
	entries, err := os.ReadDir(outside)
	if err != nil || len(entries) != 1 || entries[0].Name() != "child" {
		t.Fatal("symlink target was changed")
	}
}

type checkpointCancelContext struct {
	context.Context
	cancel    context.CancelFunc
	calls     atomic.Int32
	threshold int32
}

func (c *checkpointCancelContext) Err() error {
	if c.calls.Add(1) >= c.threshold {
		c.cancel()
	}
	return c.Context.Err()
}

func TestNativeCheckpointCancellationPreservesUnpublishedPartial(t *testing.T) {
	for _, mode := range []string{"before", "during"} {
		t.Run(mode, func(t *testing.T) {
			root := checkpointRoot(t)
			writer := checkpointWriter(t, root, nil)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			var supplied context.Context = ctx
			if mode == "before" {
				cancel()
			} else {
				supplied = &checkpointCancelContext{Context: ctx, cancel: cancel, threshold: 300}
			}
			if receipt, err := writer.WriteCheckpoint(supplied, checkpointRequest(t, 9)); !errors.Is(err, context.Canceled) || receipt != (services.ExperimentNativeCheckpointReceipt{}) {
				t.Fatalf("cancellation: %v", err)
			}
			if _, err := os.Lstat(filepath.Join(root, "step-9")); !errors.Is(err, os.ErrNotExist) {
				t.Fatal("cancelled checkpoint published")
			}
			partial := filepath.Join(root, ".step-9.partial")
			if mode == "during" {
				if info, err := os.Lstat(partial); err != nil || !info.IsDir() {
					t.Fatalf("partial not preserved: %v", err)
				}
				if entries, err := os.ReadDir(partial); err != nil || len(entries) == 0 {
					t.Fatal("no partial evidence retained")
				}
			} else if _, err := os.Lstat(partial); !errors.Is(err, os.ErrNotExist) {
				t.Fatal("pre-canceled call created staging")
			}
			if _, err := writer.WriteCheckpoint(context.Background(), checkpointRequest(t, 9)); err == nil {
				t.Fatal("failed writer retried")
			}
		})
	}
}

type checkpointTamperContext struct {
	context.Context
	root     string
	tampered bool
	failure  error
}

func (c *checkpointTamperContext) Err() error {
	// Inject corruption after both export files exist, before read-back. This
	// intentionally violates the private-root contract to exercise verification.
	if !c.tampered && c.failure == nil {
		config, err := os.Stat(filepath.Join(c.root, ".step-9.partial", "adapter_config.json"))
		if err == nil && config.Size() > 0 {
			file, err := os.OpenFile(filepath.Join(c.root, ".step-9.partial", "adapter_model.safetensors"), os.O_RDWR, 0)
			if err != nil {
				c.failure = err
				return c.Context.Err()
			}
			info, err := file.Stat()
			if err != nil {
				c.failure = err
				_ = file.Close()
				return c.Context.Err()
			}
			var value [1]byte
			_, readErr := file.ReadAt(value[:], info.Size()-1)
			value[0] ^= 1
			_, writeErr := file.WriteAt(value[:], info.Size()-1)
			c.failure = errors.Join(readErr, writeErr, file.Close())
			c.tampered = true
		}
	}
	return c.Context.Err()
}

func TestNativeCheckpointRejectsCorruptedStagingBeforePublication(t *testing.T) {
	root := checkpointRoot(t)
	writer := checkpointWriter(t, root, nil)
	ctx := &checkpointTamperContext{Context: context.Background(), root: root}
	receipt, err := writer.WriteCheckpoint(ctx, checkpointRequest(t, 9))
	if !ctx.tampered || ctx.failure != nil {
		t.Fatalf("tamper fixture failed: %v", ctx.failure)
	}
	if !errors.Is(err, services.ErrExperimentNativeCheckpoint) || receipt != (services.ExperimentNativeCheckpointReceipt{}) {
		t.Fatalf("corrupted file admitted: %v", err)
	}
	if _, err := os.Lstat(filepath.Join(root, "step-9")); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("corrupted checkpoint published")
	}
	if info, err := os.Lstat(filepath.Join(root, ".step-9.partial")); err != nil || !info.IsDir() {
		t.Fatal("corrupted staging evidence discarded")
	}
}
