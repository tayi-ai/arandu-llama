package unit_test

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	services "github.com/tayi-ai/arandu-llama/training/adapter"
	tests "github.com/tayi-ai/arandu-llama/training/adapter/tests/Fixtures"
)

// The fixtures the positive control is made of, proven here with no card.
//
// The controls that leave B at zero prove only that a contribution of zero is
// applied as zero; an engine computing 100*s*B*A passes all of them. The
// positive control needs a contribution that is not zero and whose value is
// known, which means owning every number in the adapter. These tests prove that
// WriteStructured and WriteAlpha produce exactly the file the control assumes,
// so that a failure on a node is a failure of the engine and not of the fixture.
//
// The template is composed rather than borrowed: it has to be small enough to
// read back element by element, and the real adapter is 5,640,192 numbers.

// structuredTemplate is one module of a rank-2 adapter, shaped the way
// llama-adapter.cpp validates: lora_a is [n_in, r] and lora_b is [r, n_out], so
// a.ne[1] == b.ne[0] is the rank.
func structuredTemplate(t *testing.T, dir string, alpha float32) string {
	t.Helper()
	path := filepath.Join(dir, "template.gguf")
	tests.WriteLoRAGGUF(t, path, "fixture-decoder", alpha, []tests.LoRATensor{
		{
			Name: "blk.0.ffn_down.weight.lora_a", Dims: []uint64{3, 2}, Type: 0,
			Data: []float32{1, 2, 3, 4, 5, 6},
		},
		{
			Name: "blk.0.ffn_down.weight.lora_b", Dims: []uint64{2, 4}, Type: 0,
			Data: []float32{7, 8, 9, 10, 11, 12, 13, 14},
		},
	})
	return path
}

// readTensor decodes one tensor of an adapter, independently of the writer.
func readTensor(t *testing.T, path, name string) []float32 {
	t.Helper()
	container, err := services.ReadGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, tensor := range container.Tensors {
		if tensor.Name != name {
			continue
		}
		values := make([]float32, tensor.Elements)
		start := container.DataOffset + tensor.Offset
		for i := range values {
			values[i] = math.Float32frombits(binary.LittleEndian.Uint32(body[start+int64(i)*4:]))
		}
		return values
	}
	t.Fatalf("%s holds no tensor named %s", filepath.Base(path), name)
	return nil
}

func TestAStructuredFixtureZeroesEverythingItDoesNotName(t *testing.T) {
	dir := t.TempDir()
	template := structuredTemplate(t, dir, 8)
	out := filepath.Join(dir, "structured.gguf")

	// Index 0 of lora_a is (i=0, r=0), because ne[0] varies fastest; index 0 of
	// lora_b is (r=0, j=0). Together they are the single rank component the
	// control switches on.
	if err := services.WriteStructured(template, out, []services.StructuredCell{
		{Tensor: "blk.0.ffn_down.weight.lora_a", Index: 0, Value: 1},
		{Tensor: "blk.0.ffn_down.weight.lora_b", Index: 0, Value: 0.5},
	}); err != nil {
		t.Fatal(err)
	}

	a := readTensor(t, out, "blk.0.ffn_down.weight.lora_a")
	b := readTensor(t, out, "blk.0.ffn_down.weight.lora_b")
	if len(a) != 6 || len(b) != 8 {
		t.Fatalf("the fixture changed the shapes: lora_a holds %d and lora_b holds %d", len(a), len(b))
	}
	want := []float32{1, 0, 0, 0, 0, 0}
	for i, v := range a {
		if v != want[i] {
			t.Fatalf("lora_a[%d] is %v where the fixture asked for %v", i, v, want[i])
		}
	}
	if b[0] != 0.5 {
		t.Fatalf("lora_b[0] is %v where the fixture asked for 0.5", b[0])
	}
	for i := 1; i < len(b); i++ {
		if b[i] != 0 {
			t.Fatalf("lora_b[%d] is %v; a structured fixture zeroes what it does not name", i, b[i])
		}
	}
}

func TestAStructuredFixtureKeepsTheContainer(t *testing.T) {
	dir := t.TempDir()
	template := structuredTemplate(t, dir, 8)
	out := filepath.Join(dir, "structured.gguf")
	if err := services.WriteStructured(template, out, nil); err != nil {
		t.Fatal(err)
	}
	before, err := services.ReadGGUF(template)
	if err != nil {
		t.Fatal(err)
	}
	after, err := services.ReadGGUF(out)
	if err != nil {
		t.Fatal(err)
	}
	// The engine matches an adapter to the base model by name and shape. A
	// fixture that moved either would be applied to different weights, and the
	// control would be measuring something else.
	if after.DataOffset != before.DataOffset || len(after.Tensors) != len(before.Tensors) {
		t.Fatalf("the container moved: %d tensors at offset %d became %d at %d",
			len(before.Tensors), before.DataOffset, len(after.Tensors), after.DataOffset)
	}
	for i, tensor := range before.Tensors {
		got := after.Tensors[i]
		if got.Name != tensor.Name || got.Offset != tensor.Offset || got.Elements != tensor.Elements || got.Type != tensor.Type {
			t.Fatalf("tensor %d became %+v where the template had %+v", i, got, tensor)
		}
	}
}

func TestAStructuredFixtureRefusesWhatItCannotWrite(t *testing.T) {
	dir := t.TempDir()
	template := structuredTemplate(t, dir, 8)
	out := filepath.Join(dir, "structured.gguf")

	// A cell that named nothing, or that fell off the end, would leave the
	// fixture at zero everywhere -- and a contribution of zero passes the very
	// controls this one exists to go beyond.
	if err := services.WriteStructured(template, out, []services.StructuredCell{
		{Tensor: "blk.0.ffn_gate.weight.lora_a", Index: 0, Value: 1},
	}); err == nil {
		t.Fatal("a cell naming a tensor the template does not have was accepted")
	}
	if err := services.WriteStructured(template, out, []services.StructuredCell{
		{Tensor: "blk.0.ffn_down.weight.lora_a", Index: 6, Value: 1},
	}); err == nil {
		t.Fatal("a cell one past the end of the tensor was accepted")
	}
	if err := services.WriteStructured(template, out, []services.StructuredCell{
		{Tensor: "blk.0.ffn_down.weight.lora_a", Index: -1, Value: 1},
	}); err == nil {
		t.Fatal("a negative index was accepted")
	}

	half := filepath.Join(dir, "half.gguf")
	tests.WriteLoRAGGUF(t, half, "fixture-decoder", 8, []tests.LoRATensor{
		{Name: "blk.0.ffn_down.weight.lora_a", Dims: []uint64{3, 2}, Type: 1, Data: make([]float32, 6)},
		{Name: "blk.0.ffn_down.weight.lora_b", Dims: []uint64{2, 4}, Type: 1, Data: make([]float32, 8)},
	})
	if err := services.WriteStructured(half, out, nil); err == nil {
		t.Fatal("a half precision template was accepted; the cell values would be rounded and the expectation with them")
	}
}

func TestRewritingAlphaLeavesEveryNumberAlone(t *testing.T) {
	dir := t.TempDir()
	template := structuredTemplate(t, dir, 8)
	out := filepath.Join(dir, "alpha.gguf")
	if err := services.WriteAlpha(template, out, 4); err != nil {
		t.Fatal(err)
	}

	// The whole point of the pair is that only the scale differs: llama.cpp
	// applies adapter_scale * alpha / rank, so two adapters with compensating
	// alpha and factors must produce the same output. If this rewrote a number
	// as well, that equality would be testing nothing.
	for _, name := range []string{"blk.0.ffn_down.weight.lora_a", "blk.0.ffn_down.weight.lora_b"} {
		before := readTensor(t, template, name)
		after := readTensor(t, out, name)
		if len(before) != len(after) {
			t.Fatalf("%s went from %d elements to %d", name, len(before), len(after))
		}
		for i := range before {
			if before[i] != after[i] {
				t.Fatalf("%s[%d] went from %v to %v", name, i, before[i], after[i])
			}
		}
	}

	if same := fileDigest(t, template); same == fileDigest(t, out) {
		t.Fatal("rewriting alpha produced the same bytes; the key was not found")
	}
	body, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	if !containsAlpha(body, 4) {
		t.Fatal("the rewritten file does not carry alpha 4")
	}
	if containsAlpha(body, 8) {
		t.Fatal("the rewritten file still carries the template's alpha 8")
	}
}

func TestRewritingAlphaRefusesATemplateWithout(t *testing.T) {
	dir := t.TempDir()
	// A template whose alpha is absent is the silent failure Adapter.Meta exists
	// to catch: llama.cpp would apply adapter_scale alone, at rank over alpha the
	// intended magnitude, and every number would be off by a constant factor.
	plain := filepath.Join(dir, "plain.gguf")
	if err := os.WriteFile(plain, []byte("GGUF not really"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := services.WriteAlpha(plain, filepath.Join(dir, "out.gguf"), 4); err == nil {
		t.Fatal("a template declaring no adapter.lora.alpha was accepted")
	}
}

// containsAlpha reports whether the key is stored with this value, found the
// same way WriteAlpha finds it.
func containsAlpha(body []byte, alpha float32) bool {
	const key = "adapter.lora.alpha"
	var prefix []byte
	prefix = binary.LittleEndian.AppendUint64(prefix, uint64(len(key)))
	prefix = append(prefix, key...)
	for i := 0; i+len(prefix)+8 <= len(body); i++ {
		if string(body[i:i+len(prefix)]) != string(prefix) {
			continue
		}
		if binary.LittleEndian.Uint32(body[i+len(prefix):]) != 6 {
			return false
		}
		return math.Float32frombits(binary.LittleEndian.Uint32(body[i+len(prefix)+4:])) == alpha
	}
	return false
}
