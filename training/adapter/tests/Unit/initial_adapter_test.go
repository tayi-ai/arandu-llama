package unit_test

import (
	"os"
	"path/filepath"
	"testing"

	services "github.com/tayi-ai/arandu-llama/training/adapter"
)

func TestInitialLoRAIsDeterministicNoOpWithTrainableFactorGeometry(t *testing.T) {
	dir := t.TempDir()
	first := filepath.Join(dir, "first.gguf")
	second := filepath.Join(dir, "second.gguf")
	targets := []services.LoRATarget{{Name: "output.weight", Input: 5, Output: 7}}
	digest, err := services.WriteInitialLoRA(first, "fixture-decoder", 4, 8, 59, targets)
	if err != nil {
		t.Fatal(err)
	}
	digestAgain, err := services.WriteInitialLoRA(second, "fixture-decoder", 4, 8, 59, targets)
	if err != nil {
		t.Fatal(err)
	}
	if digest != digestAgain {
		t.Fatalf("same manifest produced %s and %s", digest, digestAgain)
	}
	left, _ := os.ReadFile(first)
	right, _ := os.ReadFile(second)
	if string(left) != string(right) {
		t.Fatal("same initial adapter manifest produced different bytes")
	}
	container, err := services.ReadGGUF(first)
	if err != nil {
		t.Fatal(err)
	}
	if len(container.Tensors) != 2 {
		t.Fatalf("initial adapter has %d tensors", len(container.Tensors))
	}
	a, err := services.ReadTensorValues(first, "output.weight.lora_a")
	if err != nil {
		t.Fatal(err)
	}
	b, err := services.ReadTensorValues(first, "output.weight.lora_b")
	if err != nil {
		t.Fatal(err)
	}
	if len(a) != 20 || len(b) != 28 {
		t.Fatalf("factor geometry is A=%d B=%d", len(a), len(b))
	}
	allAZero := true
	for _, value := range a {
		allAZero = allAZero && value == 0
	}
	if allAZero {
		t.Fatal("factor A was initialized to zero, leaving the factor-space derivative degenerate")
	}
	for i, value := range b {
		if value != 0 {
			t.Fatalf("factor B[%d] = %v; initial adapter is not an exact no-op", i, value)
		}
	}
	if alpha, ok := services.AlphaOf(left); !ok || alpha != 8 {
		t.Fatalf("adapter alpha = %v, present=%t", alpha, ok)
	}
}
