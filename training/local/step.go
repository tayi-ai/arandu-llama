//go:build libtorch && cgo

package local

import (
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

	"github.com/tayi-ai/arandu-llama/checkpoint"
	"github.com/tayi-ai/arandu-llama/training/decoder"
)

func forwardFingerprint(ctx context.Context, model *decoder.TextModel, ids []int64, promptTokens int64, maxCheckpointBytes int64) (string, error) {
	tokens := int64(len(ids))
	snapshot, err := model.Forward(ctx, ids, decoder.Limits{MaxTokens: tokens, LogitRows: tokens - promptTokens + 1, MaxCheckpointBytes: maxCheckpointBytes})
	if err != nil {
		return "", err
	}
	data, err := snapshot.Logits.Bytes()
	closeErr := snapshot.Close()
	if err != nil || closeErr != nil {
		return "", errors.Join(err, closeErr)
	}
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:]), nil
}

func tensorDigest(values []float32) string {
	h := sha256.New()
	var scratch [4]byte
	for _, value := range values {
		binary.LittleEndian.PutUint32(scratch[:], math.Float32bits(value))
		_, _ = h.Write(scratch[:])
	}
	return hex.EncodeToString(h.Sum(nil))
}

func writeAndReadback(ctx context.Context, path string, tensors []checkpoint.Float32Tensor) (checkpoint.WriteReceipt, error) {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return checkpoint.WriteReceipt{}, err
	}
	receipt, writeErr := checkpoint.WriteFloat32(ctx, file, tensors, checkpoint.DefaultLimits())
	err = errors.Join(writeErr, file.Sync(), file.Close())
	if err != nil {
		return checkpoint.WriteReceipt{}, err
	}
	file, err = os.Open(path)
	if err != nil {
		return checkpoint.WriteReceipt{}, err
	}
	defer file.Close()
	stat, err := file.Stat()
	if err != nil || stat.Size() != receipt.Bytes {
		return checkpoint.WriteReceipt{}, errors.New("checkpoint size differs after write")
	}
	index, err := checkpoint.OpenSafetensors(file, stat.Size(), checkpoint.DefaultLimits())
	if err != nil {
		return checkpoint.WriteReceipt{}, err
	}
	if len(index.Tensors()) != len(tensors) {
		return checkpoint.WriteReceipt{}, errors.New("checkpoint tensor count differs after write")
	}
	buffer := make([]byte, 4<<20)
	for _, item := range tensors {
		actual, err := index.HashTensor(ctx, item.Name, buffer)
		if err != nil || actual != tensorDigest(item.Values) {
			return checkpoint.WriteReceipt{}, fmt.Errorf("checkpoint tensor readback differs: %s: %w", item.Name, err)
		}
	}
	return receipt, nil
}

func applyAndVerifyFirstUpdate(ctx context.Context, loaded *decoder.LoadedTextModel, row example, gradient decoder.CompletionGradientResult, before, output string, recipe Recipe) error {
	if len(loaded.Parameters) == 0 || len(gradient.Gradients) != len(loaded.Parameters) || gradient.Tokens != len(row.InputIDs)-row.PromptTokens {
		return errors.New("incomplete first update inputs")
	}
	learningRate := recipe.Optimizer.LearningRate
	beta1, beta2, epsilon := recipe.Optimizer.Beta1, recipe.Optimizer.Beta2, recipe.Optimizer.Epsilon
	values := make([]float32, 0, loaded.Summary.AdapterElements)
	grads := make([]float32, 0, loaded.Summary.AdapterElements)
	for i, item := range loaded.Parameters {
		if item.Name != gradient.Gradients[i].Name {
			return errors.New("gradient order differs from adapter")
		}
		current, err := item.Value.Float32Values()
		if err != nil || len(current) != len(gradient.Gradients[i].ValuesF32) {
			return errors.New("gradient shape differs from adapter")
		}
		values = append(values, current...)
		grads = append(grads, gradient.Gradients[i].ValuesF32...)
	}
	if int64(len(values)) != loaded.Summary.AdapterElements || len(grads) != len(values) {
		return errors.New("first update parameter count differs")
	}
	var normSquared float64
	for _, value := range grads {
		normSquared += float64(value) * float64(value)
	}
	norm := math.Sqrt(normSquared)
	if math.IsNaN(norm) || math.IsInf(norm, 0) || norm == 0 {
		return errors.New("invalid first update gradient norm")
	}
	clip := math.Min(1, recipe.Optimizer.MaxGradientNorm/(norm+1e-12))
	first, second := make([]float32, len(values)), make([]float32, len(values))
	updated := make([]float32, len(values))
	changed := 0
	for i, g := range grads {
		g = float32(float64(g) * clip)
		first[i] = float32((1 - beta1) * float64(g))
		second[i] = float32((1 - beta2) * float64(g*g))
		mhat := float64(first[i]) / (1 - beta1)
		vhat := float64(second[i]) / (1 - beta2)
		updated[i] = float32(float64(values[i])*(1-learningRate*recipe.Optimizer.WeightDecay) - learningRate*mhat/(math.Sqrt(vhat)+epsilon))
		if math.IsNaN(float64(updated[i])) || math.IsInf(float64(updated[i]), 0) {
			return errors.New("nonfinite first update")
		}
		if math.Float32bits(updated[i]) != math.Float32bits(values[i]) {
			changed++
		}
	}
	if changed == 0 {
		return errors.New("first update did not change any adapter parameter")
	}
	if err := os.Mkdir(output, 0700); err != nil {
		return err
	}
	adapter, moments := make([]checkpoint.Float32Tensor, 0, len(loaded.Parameters)), make([]checkpoint.Float32Tensor, 0, 2*len(loaded.Parameters))
	position := 0
	for _, parameter := range loaded.Parameters {
		info, err := parameter.Value.Info()
		if err != nil || len(info.Shape) != 2 {
			return errors.New("adapter tensor metadata differs")
		}
		count := int(info.Elements)
		shape := []uint64{uint64(info.Shape[0]), uint64(info.Shape[1])}
		adapter = append(adapter, checkpoint.Float32Tensor{Name: parameter.Name, Shape: shape, Values: updated[position : position+count]})
		moments = append(moments, checkpoint.Float32Tensor{Name: "m." + parameter.Name, Shape: shape, Values: first[position : position+count]},
			checkpoint.Float32Tensor{Name: "v." + parameter.Name, Shape: shape, Values: second[position : position+count]})
		position += count
	}
	if position != len(updated) {
		return errors.New("adapter segmentation differs")
	}
	adapterReceipt, err := writeAndReadback(ctx, filepath.Join(output, "adapter_model.safetensors"), adapter)
	if err != nil {
		return err
	}
	momentsReceipt, err := writeAndReadback(ctx, filepath.Join(output, "optimizer_moments.safetensors"), moments)
	if err != nil {
		return err
	}
	digest, err := loaded.ReplaceParameters(ctx, updated)
	if err != nil || digest == recipe.Initializer.ExpectedSHA256 {
		return errors.Join(errors.New("first update was not installed"), err)
	}
	after, err := forwardFingerprint(ctx, loaded.Model, row.InputIDs, int64(row.PromptTokens), recipe.MaxCheckpointBytes)
	if err != nil {
		return err
	}
	manifest := map[string]any{
		"recipe_sha256": recipe.Digest(), "schema_version": 1, "mode": recipe.Method, "step": 1, "batch": 1,
		"example_id": row.ID, "supervised_tokens": gradient.Tokens, "loss_before": gradient.Loss,
		"learning_rate": learningRate, "betas": [2]float64{beta1, beta2}, "epsilon": epsilon,
		"weight_decay": recipe.Optimizer.WeightDecay, "gradient_clip": recipe.Optimizer.MaxGradientNorm, "gradient_norm": norm, "clip_coefficient": clip,
		"changed_parameters": changed, "initial_adapter_sha256": recipe.Initializer.ExpectedSHA256,
		"updated_adapter_sha256": digest, "adapter_file_sha256": adapterReceipt.SHA256,
		"optimizer_file_sha256": momentsReceipt.SHA256,
		"logits_before_sha256":  before, "logits_after_sha256": after, "inference_changed": before != after,
		"base_revision": recipe.BaseRevision,
	}
	body, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return err
	}
	body = append(body, '\n')
	manifestPath := filepath.Join(output, "manifest.json")
	if err := os.WriteFile(manifestPath, body, 0600); err != nil {
		return err
	}
	fmt.Printf("phase=updated adapter_sha256=%s changed_parameters=%d gradient_norm=%g inference_changed=%v\n", digest, changed, norm, before != after)
	return nil
}

func reloadAndVerifyAdapter(ctx context.Context, loaded *decoder.LoadedTextModel, row example, directory string, recipe Recipe) error {
	body, err := os.ReadFile(filepath.Join(directory, "manifest.json"))
	if err != nil {
		return err
	}
	var manifest struct {
		Step              int    `json:"step"`
		AdapterSHA256     string `json:"updated_adapter_sha256"`
		AdapterFileSHA256 string `json:"adapter_file_sha256"`
		LogitsAfterSHA256 string `json:"logits_after_sha256"`
		BaseRevision      string `json:"base_revision"`
	}
	if err := json.Unmarshal(body, &manifest); err != nil {
		return err
	}
	if manifest.Step != 1 || manifest.BaseRevision != recipe.BaseRevision {
		return errors.New("checkpoint manifest identity differs")
	}
	file, err := os.Open(filepath.Join(directory, "adapter_model.safetensors"))
	if err != nil {
		return err
	}
	defer file.Close()
	stat, err := file.Stat()
	if err != nil {
		return err
	}
	h := sha256.New()
	if _, err := io.Copy(h, file); err != nil || hex.EncodeToString(h.Sum(nil)) != manifest.AdapterFileSHA256 {
		return errors.New("saved adapter container hash differs")
	}
	index, err := checkpoint.OpenSafetensors(file, stat.Size(), checkpoint.DefaultLimits())
	if err != nil || len(index.Tensors()) != len(loaded.Parameters) {
		return errors.New("saved adapter tensor index differs")
	}
	values := make([]float32, 0, loaded.Summary.AdapterElements)
	for _, parameter := range loaded.Parameters {
		tensor, ok := index.Tensor(parameter.Name)
		if !ok || tensor.DType != "F32" {
			return fmt.Errorf("saved adapter tensor missing or wrong dtype: %s", parameter.Name)
		}
		info, err := parameter.Value.Info()
		if err != nil || len(tensor.Shape) != 2 || tensor.Shape[0] != uint64(info.Shape[0]) || tensor.Shape[1] != uint64(info.Shape[1]) || tensor.Size() != info.Elements*4 {
			return fmt.Errorf("saved adapter shape differs: %s", parameter.Name)
		}
		reader, err := index.TensorReader(parameter.Name)
		if err != nil {
			return err
		}
		data, err := io.ReadAll(reader)
		if err != nil || len(data) != int(tensor.Size()) {
			return errors.New("saved adapter tensor payload incomplete")
		}
		for offset := 0; offset < len(data); offset += 4 {
			values = append(values, math.Float32frombits(binary.LittleEndian.Uint32(data[offset:])))
		}
	}
	digest, err := loaded.ReplaceParameters(ctx, values)
	if err != nil || digest != manifest.AdapterSHA256 {
		return errors.New("reloaded adapter identity differs")
	}
	after, err := forwardFingerprint(ctx, loaded.Model, row.InputIDs, int64(row.PromptTokens), recipe.MaxCheckpointBytes)
	if err != nil || after != manifest.LogitsAfterSHA256 {
		return errors.New("reloaded adapter inference differs")
	}
	fmt.Printf("phase=reload adapter_sha256=%s logits_sha256=%s\n", digest, after)
	return nil
}
