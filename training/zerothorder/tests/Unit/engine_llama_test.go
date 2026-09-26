//go:build llama

package unit_test

import (
	"github.com/tayi-ai/arandu-llama/training/adapter"

	"context"
	"math"
	"os"
	"path/filepath"
	"testing"

	services "github.com/tayi-ai/arandu-llama/training/zerothorder"
)

// The binding against llama.cpp itself, on a node: the candidate the probe
// applied is the candidate the fold commits, the base model keeps its bytes,
// and a reopened policy scores what was committed. It is behind the llama tag
// because it links the engine, and behind two environment variables because it
// needs a base model and an F32 adapter converted for it. Without them it
// skips, and what it proves is declared not measured.

func TestTheBindingAppliesWhatItFolds(t *testing.T) {
	model := os.Getenv("TAYI_TEST_MODEL_GGUF")
	adapterPath := os.Getenv("TAYI_ADAPTER_F32_GGUF")
	if model == "" || adapterPath == "" {
		t.Skip("set TAYI_TEST_MODEL_GGUF and TAYI_ADAPTER_F32_GGUF to run the binding against llama.cpp")
	}
	dir := t.TempDir()
	policy := filepath.Join(dir, "policy.gguf")
	body, err := os.ReadFile(adapterPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(policy, body, 0o644); err != nil {
		t.Fatal(err)
	}
	baseBefore := bindingFileDigest(t, model)

	engine, err := services.OpenLlama(services.LlamaBindingConfig{
		ModelPath: model, PolicyPath: policy, DirectionDir: dir, ContextTokens: 512, GPULayers: 0,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer engine.Close()
	ctx := context.Background()
	example := services.Example{ID: "binding", Prompt: "The capital of France is", Completion: " Paris."}
	d := adapter.Direction{Seed: 1, Scale: 1e-3}

	// (1) The unperturbed reading is stable across a step.
	before, err := engine.Score(ctx, example)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := services.Step(ctx, engine, example, d); err != nil {
		t.Fatal(err)
	}
	after, err := engine.Score(ctx, example)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(after.Loss-before.Loss) > 1e-9 {
		t.Fatalf("the unperturbed loss read %v before the step and %v after it", before.Loss, after.Loss)
	}
	if bindingFileDigest(t, policy) != engine.Digest() {
		t.Fatal("a step changed the policy on disk")
	}

	// (2) The probe at +c and the fold at c apply the same candidate.
	if err := engine.Perturb(ctx, d, 1); err != nil {
		t.Fatal(err)
	}
	probed, err := engine.Score(ctx, example)
	if err != nil {
		t.Fatal(err)
	}
	if err := engine.Perturb(ctx, d, 0); err != nil {
		t.Fatal(err)
	}
	if _, err := engine.Fold(d, d.Scale); err != nil {
		t.Fatal(err)
	}
	folded, err := engine.Score(ctx, example)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(folded.Loss-probed.Loss) > 1e-6 {
		t.Fatalf("the folded policy scores %v where the probe at the same coefficient scored %v", folded.Loss, probed.Loss)
	}

	// (3) The base is untouched and the policy on disk is what the engine holds.
	if bindingFileDigest(t, model) != baseBefore {
		t.Fatal("the base model's bytes changed")
	}
	if bindingFileDigest(t, policy) != engine.Digest() {
		t.Fatal("the policy on disk is not the snapshot the engine holds after the fold")
	}

	// (4) Reopening the folded policy reads the same number.
	engine.Close()
	reopened, err := services.OpenLlama(services.LlamaBindingConfig{
		ModelPath: model, PolicyPath: policy, DirectionDir: dir, ContextTokens: 512, GPULayers: 0,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	reloaded, err := reopened.Score(ctx, example)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(reloaded.Loss-folded.Loss) > 1e-6 {
		t.Fatalf("the reopened policy scores %v where the folded one scored %v", reloaded.Loss, folded.Loss)
	}
}

func bindingFileDigest(t *testing.T, path string) string {
	t.Helper()
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return adapter.Digest(body)
}
