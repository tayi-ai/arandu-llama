package unit_test

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	services "github.com/tayi-ai/arandu-llama/training/adapter"
	"github.com/tayi-ai/arandu-llama/training/adapter/tests/Fixtures"
)

func templateGGUF(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "fixture.gguf")
	fixtures.WriteLoRAGGUF(t, path, "fixture-decoder", 8, []fixtures.LoRATensor{
		{Name: "fixture.weight.lora_a", Dims: []uint64{64, 64}, Type: 1, Data: make([]float32, 4096)},
		{Name: "fixture.weight.lora_b", Dims: []uint64{64, 64}, Type: 1, Data: make([]float32, 4096)},
	})
	return path
}

func TestTheReaderFindsEveryTensorOfTheAdapter(t *testing.T) {
	container, err := services.ReadGGUF(templateGGUF(t))
	if err != nil {
		t.Fatal(err)
	}
	if len(container.Tensors) != 2 {
		t.Fatalf("the reader found %d tensors, want 2", len(container.Tensors))
	}
	elements := int64(0)
	for _, tensor := range container.Tensors {
		if tensor.Elements <= 0 {
			t.Fatalf("tensor %s has %d elements", tensor.Name, tensor.Elements)
		}
		elements += tensor.Elements
	}
	if elements != 8192 {
		t.Fatalf("the tensors hold %d elements, want 8192", elements)
	}
	if container.DataOffset <= 0 {
		t.Fatalf("the data section starts at %d", container.DataOffset)
	}
	size, err := os.Stat(templateGGUF(t))
	if err != nil {
		t.Fatal(err)
	}
	// Half precision: two bytes an element, plus the header. If the file were
	// bigger than the header allows, the reader located the data section wrongly.
	if want := container.DataOffset + elements*2; want != size.Size() {
		t.Fatalf("header plus payload is %d bytes, the file is %d", want, size.Size())
	}
}

func TestTheDirectionCountMatchesTheAdapterItPerturbs(t *testing.T) {
	got, err := services.DirectionElements(templateGGUF(t))
	if err != nil {
		t.Fatal(err)
	}
	if got != 8192 {
		t.Fatalf("a direction over this adapter would carry %d elements, want 8192", got)
	}
}

func TestADirectionKeepsTheContainerAndReplacesOnlyTheNumbers(t *testing.T) {
	template := templateGGUF(t)
	out := filepath.Join(t.TempDir(), "direction.gguf")
	if err := services.WriteDirection(template, out, services.Direction{Seed: 7, Scale: 0.01}); err != nil {
		t.Fatal(err)
	}

	original, err := os.ReadFile(template)
	if err != nil {
		t.Fatal(err)
	}
	written, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	if len(original) != len(written) {
		t.Fatalf("the direction is %d bytes, the template is %d", len(written), len(original))
	}

	container, err := services.ReadGGUF(template)
	if err != nil {
		t.Fatal(err)
	}
	// The header has to survive byte for byte: the engine matches an adapter to
	// the base model by the tensor names and shapes recorded there, and a
	// direction that disagreed would be applied to different weights.
	header := int(container.DataOffset)
	for i := 0; i < header; i++ {
		if original[i] != written[i] {
			t.Fatalf("the header changed at byte %d of %d", i, header)
		}
	}
	// And the payload has to have changed, or the direction is the adapter and
	// the perturbation moves nothing.
	same := 0
	for i := header; i < len(original); i++ {
		if original[i] == written[i] {
			same++
		}
	}
	if same == len(original)-header {
		t.Fatal("the payload is identical to the template; nothing was perturbed")
	}
}

func TestTheSameSeedWritesTheSameDirection(t *testing.T) {
	template := templateGGUF(t)
	dir := t.TempDir()
	first := filepath.Join(dir, "a.gguf")
	second := filepath.Join(dir, "b.gguf")
	third := filepath.Join(dir, "c.gguf")
	for path, seed := range map[string]uint64{first: 11, second: 11, third: 12} {
		if err := services.WriteDirection(template, path, services.Direction{Seed: seed, Scale: 0.01}); err != nil {
			t.Fatal(err)
		}
	}
	a, _ := os.ReadFile(first)
	b, _ := os.ReadFile(second)
	c, _ := os.ReadFile(third)
	// Every node writes the same direction from the seed.
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("the same seed produced different files at byte %d", i)
		}
	}
	if len(a) == len(c) {
		identical := true
		for i := range a {
			if a[i] != c[i] {
				identical = false
				break
			}
		}
		if identical {
			t.Fatal("two seeds produced the same direction; they are not independent")
		}
	}
}

func TestTheReaderRefusesWhatIsNotAGGUF(t *testing.T) {
	dir := t.TempDir()
	cases := map[string][]byte{
		"empty":       {},
		"wrong magic": []byte("NOPE\x03\x00\x00\x00"),
		"truncated":   append([]byte("GGUF"), 3, 0, 0, 0),
	}
	for name, body := range cases {
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, body, 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := services.ReadGGUF(path); err == nil {
			t.Errorf("%s was accepted", name)
		}
	}
	if _, err := services.ReadGGUF(filepath.Join(dir, "absent")); err == nil {
		t.Error("a missing file was accepted")
	}
}

// TestHalfPrecisionRoundTripsThroughTheEngineFormat checks the conversion the
// cards force on us: the T10 is compute capability 7.5 and has no bfloat16, so
// a direction is written in IEEE half and read back as half.
func TestHalfPrecisionRoundTripsThroughTheEngineFormat(t *testing.T) {
	for _, value := range []float32{0, 1, -1, 0.5, -0.5, 2.5, -3.75, 1e-4, -1e-4} {
		bits := services.Float16BitsForTest(value)
		back := halfToFloat(bits)
		if math.Abs(float64(back-value)) > math.Abs(float64(value))*1e-3+1e-7 {
			t.Errorf("%v came back as %v", value, back)
		}
	}
	// A magnitude beyond half precision saturates to the largest finite value
	// rather than to infinity. An infinity in a direction becomes a nan loss, and
	// a nan estimate poisons every parameter the step touches.
	if got := services.Float16BitsForTest(1e30); got != 0x7bff {
		t.Errorf("an overflow gave %#04x, want 0x7bff", got)
	}
	if got := services.Float16BitsForTest(-1e30); got != 0xfbff {
		t.Errorf("a negative overflow gave %#04x, want 0xfbff", got)
	}
	// And a magnitude below what half resolves becomes zero: a coordinate the
	// step does not move, which is the honest result at that scale.
	if got := services.Float16BitsForTest(1e-12); got != 0 {
		t.Errorf("an underflow gave %#04x, want 0", got)
	}
}

// halfToFloat is the inverse, written here rather than in the service because
// nothing in the application reads half precision back.
func halfToFloat(bits uint16) float32 {
	sign := uint32(bits&0x8000) << 16
	exponent := uint32(bits>>10) & 0x1f
	mantissa := uint32(bits & 0x3ff)
	switch {
	case exponent == 0:
		if mantissa == 0 {
			return math.Float32frombits(sign)
		}
		exponent = 127 - 14
		for mantissa&0x400 == 0 {
			mantissa <<= 1
			exponent--
		}
		mantissa &= 0x3ff
		return math.Float32frombits(sign | exponent<<23 | mantissa<<13)
	case exponent == 0x1f:
		return math.Float32frombits(sign | 0xff<<23 | mantissa<<13)
	default:
		return math.Float32frombits(sign | (exponent+127-15)<<23 | mantissa<<13)
	}
}

// TestTheDirectionIsDrawnFromTheDistributionTheEstimatorAssumes reads the
// numbers back out of a written direction.
//
// The simultaneous perturbation estimator assumes a standard normal. A direction
// whose spread is not one is a scaling error that presents as a learning rate
// problem, which is the hardest kind to find.
func TestTheDirectionIsDrawnFromTheDistributionTheEstimatorAssumes(t *testing.T) {
	template := templateGGUF(t)
	out := filepath.Join(t.TempDir(), "direction.gguf")
	if err := services.WriteDirection(template, out, services.Direction{Seed: 3, Scale: 0.01}); err != nil {
		t.Fatal(err)
	}
	container, err := services.ReadGGUF(out)
	if err != nil {
		t.Fatal(err)
	}
	body, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	sum, sumSquares, n := 0.0, 0.0, 0
	// One tensor is enough to characterise the distribution.
	tensor := container.Tensors[0]
	start := container.DataOffset + tensor.Offset
	for i := int64(0); i < tensor.Elements; i++ {
		value := float64(halfToFloat(binary.LittleEndian.Uint16(body[start+i*2:])))
		sum += value
		sumSquares += value * value
		n++
	}
	mean := sum / float64(n)
	variance := sumSquares/float64(n) - mean*mean
	if math.Abs(mean) > 0.1 {
		t.Errorf("the direction has mean %v over %d elements, want about 0", mean, n)
	}
	if math.Abs(variance-1) > 0.15 {
		t.Errorf("the direction has variance %v over %d elements, want about 1", variance, n)
	}
}
