//go:build libtorch && cgo

package decoder_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/tayi-ai/arandu-llama/checkpoint"
	"github.com/tayi-ai/arandu-llama/training/decoder"
	"github.com/tayi-ai/arandu-llama/training/layers"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

func restoreFixture(t *testing.T) (*decoder.LoadedTextModel, []checkpoint.Float32Tensor) {
	t.Helper()
	model := &decoder.LoadedTextModel{Model: &decoder.TextModel{Layers: make([]decoder.Layer, 11)}}
	var tensors []checkpoint.Float32Tensor
	// Registry order 2,10 deliberately differs from safetensors' lexical order.
	for _, index := range []int{2, 10} {
		prefix := fmt.Sprintf("base_model.model.model.language_model.layers.%d.self_attn.", index)
		var leaves []*torch.Tensor
		for _, spec := range []struct {
			name  string
			shape []int64
		}{{"q_proj.lora_A.default.weight", []int64{2, 3}}, {"q_proj.lora_B.default.weight", []int64{6, 2}},
			{"v_proj.lora_A.default.weight", []int64{2, 3}}, {"v_proj.lora_B.default.weight", []int64{3, 2}}} {
			data := make([]float32, spec.shape[0]*spec.shape[1])
			for i := range data {
				data[i] = float32(i+1) / 16
			}
			leaf := tensor(t, data, spec.shape, true)
			leaves = append(leaves, leaf)
			name := prefix + spec.name
			model.Parameters = append(model.Parameters, decoder.InitialParameter{Name: name, Value: leaf})
			for i := range data {
				data[i] += float32(index) / 8
			}
			data[0] = math.Float32frombits(0x80000000)
			tensors = append(tensors, checkpoint.Float32Tensor{Name: name, Shape: []uint64{uint64(spec.shape[0]), uint64(spec.shape[1])}, Values: data})
		}
		model.Model.Layers[index] = decoder.Layer{Device: torch.CPUDevice(), Adapter: &layers.AttentionLoRA{
			QueryA: leaves[0], QueryB: leaves[1], ValueA: leaves[2], ValueB: leaves[3], Alpha: 4}}
	}
	t.Cleanup(func() { _ = model.Close() })
	return model, tensors
}

func restoreParameterDigest(tensors []checkpoint.Float32Tensor) string {
	h := sha256.New()
	var scalar [4]byte
	for _, tensor := range tensors {
		_, _ = h.Write([]byte(tensor.Name))
		for _, value := range tensor.Values {
			binary.LittleEndian.PutUint32(scalar[:], math.Float32bits(value))
			_, _ = h.Write(scalar[:])
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}

func restoreFile(t *testing.T, tensors []checkpoint.Float32Tensor) (decoder.AdapterCheckpoint, []byte) {
	t.Helper()
	var body bytes.Buffer
	limits := checkpoint.Limits{MaxHeaderBytes: 64 << 10, MaxTensors: 8, MaxDimensions: 2, MaxMetadataEntries: 1, MaxChunkBytes: 32}
	receipt, err := checkpoint.WriteFloat32(context.Background(), &body, tensors, limits)
	if err != nil {
		t.Fatal(err)
	}
	source := decoder.AdapterCheckpoint{Path: filepath.Join(t.TempDir(), "adapter.safetensors"), FileSHA256: receipt.SHA256,
		ParametersSHA256: restoreParameterDigest(tensors), MaxBytes: receipt.Bytes, MaxWorkingBytes: 32 << 10, Limits: limits}
	if err := os.WriteFile(source.Path, body.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	return source, body.Bytes()
}

func restoredValues(t *testing.T, model *decoder.LoadedTextModel) [][]float32 {
	t.Helper()
	var result [][]float32
	for _, parameter := range model.Parameters {
		result = append(result, read(t, parameter.Value))
	}
	return result
}

func TestRestoreAdapterInstallsExactRegistryAndOwnsNewLeaves(t *testing.T) {
	model, tensors := restoreFixture(t)
	source, _ := restoreFile(t, tensors)
	old := slices.Clone(model.Parameters)
	digest, err := model.RestoreAdapter(context.Background(), source)
	if err != nil || digest != source.ParametersSHA256 {
		t.Fatalf("restoration failed: %s %v", digest, err)
	}
	for i, parameter := range model.Parameters {
		if parameter.Name != tensors[i].Name || parameter.Value == old[i].Value {
			t.Fatal("registry order or atomic replacement differs")
		}
		values := read(t, parameter.Value)
		for j, value := range values {
			if math.Float32bits(value) != math.Float32bits(tensors[i].Values[j]) {
				t.Fatal("restored value bits differ, including signed zero")
			}
		}
		if _, err := old[i].Value.Info(); err == nil {
			t.Fatal("committed restoration retained old leaf")
		}
	}
	// Repeating the exact artifact remains valid after ownership transfers.
	if digest, err := model.RestoreAdapter(context.Background(), source); err != nil || digest != source.ParametersSHA256 {
		t.Fatalf("repeat restoration: %s %v", digest, err)
	}
}

func TestRestoreAdapterRejectsChangedIdentityGeometryAndBudgetsAtomically(t *testing.T) {
	for _, name := range []string{"file-sha", "parameter-sha", "mutated-file", "shape", "name", "missing", "dtype", "nan", "file-budget", "working-budget", "header-budget", "registry"} {
		t.Run(name, func(t *testing.T) {
			model, tensors := restoreFixture(t)
			switch name {
			case "shape":
				tensors[0].Shape = []uint64{3, 2}
			case "name":
				tensors[0].Name += ".wrong"
			case "missing":
				tensors = tensors[:len(tensors)-1]
			}
			source, body := restoreFile(t, tensors)
			switch name {
			case "file-sha":
				source.FileSHA256 = strings.Repeat("0", 64)
			case "parameter-sha":
				source.ParametersSHA256 = strings.Repeat("0", 64)
			case "file-budget":
				source.MaxBytes--
			case "working-budget":
				source.MaxWorkingBytes = 1
			case "header-budget":
				source.Limits.MaxHeaderBytes = 8
			case "registry":
				model.Parameters[0].Name += ".wrong"
			case "mutated-file":
				body[len(body)-1] ^= 1
			case "dtype":
				body = bytes.Replace(body, []byte("\"F32\""), []byte("\"I32\""), 1)
			case "nan":
				payload := 8 + binary.LittleEndian.Uint64(body[:8])
				binary.LittleEndian.PutUint32(body[payload:], math.Float32bits(float32(math.NaN())))
			}
			if name == "nan" || name == "dtype" {
				hash := sha256.Sum256(body)
				source.FileSHA256 = hex.EncodeToString(hash[:])
			}
			if err := os.WriteFile(source.Path, body, 0600); err != nil {
				t.Fatal(err)
			}
			old, values := slices.Clone(model.Parameters), restoredValues(t, model)
			digest, err := model.RestoreAdapter(context.Background(), source)
			if digest != "" || !errors.Is(err, decoder.ErrAdapterCheckpoint) {
				t.Fatalf("invalid artifact admitted: %s %v", digest, err)
			}
			if !reflect.DeepEqual(old, model.Parameters) || !reflect.DeepEqual(values, restoredValues(t, model)) {
				t.Fatal("refused artifact altered prior parameters")
			}
		})
	}
}

type restoreCheckingContext struct {
	context.Context
	cancel    context.CancelFunc
	calls     atomic.Int32
	threshold int32
}

func (c *restoreCheckingContext) Err() error {
	if count := c.calls.Add(1); c.threshold > 0 && count >= c.threshold {
		c.cancel()
	}
	return c.Context.Err()
}

func TestRestoreAdapterCancellationPreservesEveryOldHandle(t *testing.T) {
	model, tensors := restoreFixture(t)
	source, _ := restoreFile(t, tensors)
	counting := &restoreCheckingContext{Context: context.Background()}
	if _, err := model.RestoreAdapter(counting, source); err != nil {
		t.Fatal(err)
	}
	checks := counting.calls.Load()
	if checks < 30 {
		t.Fatal("restoration lacked bounded cancellation points")
	}
	// The final check follows complete native preparation immediately before commit.
	for _, threshold := range []int32{1, checks / 3, checks - 3, checks} {
		ctx, cancel := context.WithCancel(context.Background())
		checking := &restoreCheckingContext{Context: ctx, cancel: cancel, threshold: threshold}
		old, values := slices.Clone(model.Parameters), restoredValues(t, model)
		digest, err := model.RestoreAdapter(checking, source)
		cancel()
		if digest != "" || !errors.Is(err, context.Canceled) || checking.calls.Load() != threshold {
			t.Fatalf("cancelled restoration committed: threshold=%d digest=%s error=%v", threshold, digest, err)
		}
		if !reflect.DeepEqual(old, model.Parameters) || !reflect.DeepEqual(values, restoredValues(t, model)) {
			t.Fatal("cancelled restoration changed old parameter identity or bytes")
		}
	}
}
