package fixtures

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"testing"

	services "github.com/tayi-ai/arandu-llama/training/adapter"
)

// A LoRA adapter composed from nothing, for tests only.
//
// The application never composes a GGUF: Adapter.go copies a real adapter and
// replaces its numbers, because production code that owns an opinion about
// tensor names and shapes is a direction the engine loads and applies to the
// wrong weights. A test is the one place where owning the layout is the point,
// because the oracle needs factors it can name -- A = 2, B = 3 -- and no real
// adapter carries those.
//
// The layout is what llama-adapter.cpp validates when it loads a LoRA: a pair
// of tensors per module named <module>.lora_a and <module>.lora_b, lora_a with
// ne = [n_in, r] and lora_b with ne = [r, n_out], so that a.ne[1] == b.ne[0] is
// the rank. The effective scale is adapter.lora.alpha over that rank, which is
// why alpha is written as metadata and not assumed.

// LoRATensor is one tensor of a synthetic adapter. Dims are in ggml order:
// ne[0] varies fastest, which is how the GGUF header stores them, so a tensor
// with dims [3, 2] holds two contiguous rows of three.
type LoRATensor struct {
	Name string
	Dims []uint64
	// Type is the GGUF type tag: 0 for F32, 1 for F16. Anything else is written
	// as raw F32 payload under the declared tag, which is how a test asks for a
	// tensor the reader has to refuse.
	Type uint32
	Data []float32
}

// WriteLoRAGGUF writes a GGUF version 3 adapter with the metadata
// convert_lora_to_gguf.py emits and the alignment ReadGGUF defaults to.
func WriteLoRAGGUF(t *testing.T, path, arch string, alpha float32, tensors []LoRATensor) {
	t.Helper()
	const alignment = 32
	var head bytes.Buffer
	u32 := func(v uint32) { _ = binary.Write(&head, binary.LittleEndian, v) }
	u64 := func(v uint64) { _ = binary.Write(&head, binary.LittleEndian, v) }
	str := func(s string) { u64(uint64(len(s))); head.WriteString(s) }

	u32(0x46554747) // "GGUF"
	u32(3)
	u64(uint64(len(tensors)))
	u64(5)
	str("general.type")
	u32(8)
	str("adapter")
	str("general.architecture")
	u32(8)
	str(arch)
	str("adapter.type")
	u32(8)
	str("lora")
	str("adapter.lora.alpha")
	u32(6)
	u32(math.Float32bits(alpha))
	str("general.alignment")
	u32(4)
	u32(alignment)

	var data bytes.Buffer
	for _, tensor := range tensors {
		str(tensor.Name)
		u32(uint32(len(tensor.Dims)))
		for _, d := range tensor.Dims {
			u64(d)
		}
		u32(tensor.Type)
		u64(uint64(data.Len()))
		for _, v := range tensor.Data {
			if tensor.Type == 1 {
				_ = binary.Write(&data, binary.LittleEndian, services.Float16BitsForTest(v))
			} else {
				_ = binary.Write(&data, binary.LittleEndian, math.Float32bits(v))
			}
		}
		for data.Len()%alignment != 0 {
			data.WriteByte(0)
		}
	}
	for head.Len()%alignment != 0 {
		head.WriteByte(0)
	}
	head.Write(data.Bytes())
	if err := os.WriteFile(path, head.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}
