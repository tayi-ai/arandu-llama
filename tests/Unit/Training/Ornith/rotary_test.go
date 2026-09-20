//go:build libtorch && cgo

package ornith_test

import (
	"context"
	"errors"
	"math"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/ornith"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

const referenceFrequencySHA256 = "ec4437c15ead01576c3e6c7412c11daba363d17cee8bfce5c5d1c54e2e09d2d0"

func TestTextRotaryAdmitsExactReferenceFrequencyHash(t *testing.T) {
	rotary, err := ornith.TextRotary(context.Background(), 3, torch.CPUDevice(), referenceFrequencySHA256)
	if err != nil {
		t.Fatal(err)
	}
	defer rotary.Close()
	for _, value := range []*torch.Tensor{rotary.Cosine, rotary.Sine} {
		info, err := value.Info()
		if err != nil {
			t.Fatal(err)
		}
		if info.DType != torch.Float32 || info.Device != torch.CPUDevice() || info.RequiresGrad || len(info.Shape) != 2 || info.Shape[0] != 3 || info.Shape[1] != 32 {
			t.Fatalf("unexpected rotary metadata: %+v", info)
		}
		finite, err := value.AllFinite()
		if err != nil || !finite {
			t.Fatalf("nonfinite rotary output: %v", err)
		}
	}
}

// Independent scalar IEEE binary16 rounding for finite trigonometric values.
// This deliberately does not call the tensor library's conversion operation.
func rotaryHalfStep(value float32) float64 {
	abs := math.Abs(float64(value))
	if abs < math.Ldexp(1, -14) {
		return math.Ldexp(1, -24)
	}
	_, exponent := math.Frexp(abs)
	return math.Ldexp(1, exponent-11)
}

func rotaryHalfRound(value float32) float32 {
	step := rotaryHalfStep(value)
	return float32(math.Copysign(math.RoundToEven(math.Abs(float64(value))/step)*step, float64(value)))
}

func rotaryRead(t *testing.T, value *torch.Tensor) []float32 {
	t.Helper()
	result, err := value.Float32Values()
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func TestTextRotaryMatchesIndependentScalarMathAndHalfRounding(t *testing.T) {
	const tokens = 9
	rotary, err := ornith.TextRotary(context.Background(), tokens, torch.CPUDevice(), referenceFrequencySHA256)
	if err != nil {
		t.Fatal(err)
	}
	defer rotary.Close()
	cosine, sine := rotaryRead(t, rotary.Cosine), rotaryRead(t, rotary.Sine)
	var maximumHalfULP float64
	roundedEntries := 0
	for position := 0; position < tokens; position++ {
		for column := 0; column < 32; column++ {
			// HF uses FP32 powers followed by an FP32 reciprocal. Go's math.Pow
			// supplies an independent scalar implementation; native pow/trig
			// may differ by a small ULP before rounding to the storage dtype.
			power := float32(math.Pow(10000000, float64(column)/32))
			frequency := float32(1) / power
			angle := float32(position) * frequency
			for _, entry := range []struct {
				got, raw float32
			}{
				{cosine[position*32+column], float32(math.Cos(float64(angle)))},
				{sine[position*32+column], float32(math.Sin(float64(angle)))},
			} {
				expected := rotaryHalfRound(entry.raw)
				if rotaryHalfRound(entry.got) != entry.got {
					t.Fatalf("position%d frequency%d skipped the FP16 storage cast: %.9g", position, column, entry.got)
				}
				if expected != entry.raw {
					roundedEntries++
				}
				ulp := math.Abs(float64(entry.got)-float64(expected)) / rotaryHalfStep(expected)
				maximumHalfULP = math.Max(maximumHalfULP, ulp)
				if math.IsNaN(float64(entry.got)) || math.IsInf(float64(entry.got), 0) || ulp > 1 {
					t.Fatalf("position%d frequency%d got%.9g expected%.9g error%.4g half ULP", position, column, entry.got, expected, ulp)
				}
			}
		}
	}
	if roundedEntries == 0 {
		t.Fatal("fixture never distinguished FP32 from FP16 storage")
	}
	for column := 0; column < 32; column++ {
		if cosine[column] != 1 || math.Float32bits(sine[column]) != 0 {
			t.Fatal("position zero must be cosine one and positive-zero sine")
		}
	}
	t.Logf("576 scalar comparisons; max error %.4g half ULP; %d entries exercise FP16 rounding", maximumHalfULP, roundedEntries)
}

func TestTextRotaryMaximumTokensHaveSequentialPositionsAndBoundedTables(t *testing.T) {
	rotary, err := ornith.TextRotary(context.Background(), 4096, torch.CPUDevice(), referenceFrequencySHA256)
	if err != nil {
		t.Fatal(err)
	}
	defer rotary.Close()
	for _, value := range []*torch.Tensor{rotary.Cosine, rotary.Sine} {
		info, err := value.Info()
		if err != nil || len(info.Shape) != 2 || info.Shape[0] != 4096 || info.Shape[1] != 32 || info.Elements != 4096*32 {
			t.Fatalf("maximum-length table geometry differs: %+v %v", info, err)
		}
		finite, err := value.AllFinite()
		if err != nil || !finite {
			t.Fatalf("nonfinite maximum-length table: %v", err)
		}
	}
	cosine, sine := rotaryRead(t, rotary.Cosine), rotaryRead(t, rotary.Sine)
	for position := 0; position < 4096; position++ {
		// The first inverse frequency is exactly one, making this a direct
		// check of every text position and the table axes, without pow error.
		for _, entry := range []struct{ got, expected float32 }{
			{cosine[position*32], rotaryHalfRound(float32(math.Cos(float64(position))))},
			{sine[position*32], rotaryHalfRound(float32(math.Sin(float64(position))))},
		} {
			if math.Abs(float64(entry.got)-float64(entry.expected)) > rotaryHalfStep(entry.expected) {
				t.Fatalf("position%d is not the expected sequential text position", position)
			}
		}
	}
}

func TestTextRotaryRejectsInvalidLimitsIdentityAndDevice(t *testing.T) {
	for _, tokens := range []int{-1, 0, 4097, int(^uint(0) >> 1)} {
		result, err := ornith.TextRotary(context.Background(), tokens, torch.CPUDevice(), referenceFrequencySHA256)
		if result != nil || !errors.Is(err, ornith.ErrRotarySpec) {
			_ = result.Close()
			t.Fatalf("invalid tokens%d: %v", tokens, err)
		}
	}
	for _, digest := range []string{"", referenceFrequencySHA256[:63], strings.Repeat("g", 64), strings.ToUpper(referenceFrequencySHA256)} {
		result, err := ornith.TextRotary(context.Background(), 3, torch.CPUDevice(), digest)
		if result != nil || !errors.Is(err, ornith.ErrRotarySpec) {
			_ = result.Close()
			t.Fatalf("invalid digest accepted: %v", err)
		}
	}
	// Invalid target placement would produce a different error if reached:
	// the frequency identity must be checked before target-device operations.
	invalidDevice := torch.Device{Kind: "invalid", Index: 0}
	result, err := ornith.TextRotary(context.Background(), 3, invalidDevice, strings.Repeat("0", 64))
	if result != nil || !errors.Is(err, ornith.ErrRotaryIdentity) {
		_ = result.Close()
		t.Fatalf("frequency guard did not precede target placement: %v", err)
	}
	result, err = ornith.TextRotary(context.Background(), 3, invalidDevice, referenceFrequencySHA256)
	if result != nil || err == nil || errors.Is(err, ornith.ErrRotaryIdentity) {
		_ = result.Close()
		t.Fatalf("invalid device admission: %v", err)
	}
	result, err = ornith.TextRotary(nil, 3, torch.CPUDevice(), referenceFrequencySHA256)
	if result != nil || !errors.Is(err, ornith.ErrRotarySpec) {
		_ = result.Close()
		t.Fatalf("nil context admission: %v", err)
	}
}

type rotaryCheckingContext struct {
	context.Context
	cancel    context.CancelFunc
	calls     atomic.Int32
	threshold int32
}

func (c *rotaryCheckingContext) Err() error {
	if c.calls.Add(1) >= c.threshold {
		c.cancel()
	}
	return c.Context.Err()
}

func TestTextRotaryCancellationAndOwnedOutputLifetime(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if result, err := ornith.TextRotary(ctx, 3, torch.CPUDevice(), referenceFrequencySHA256); result != nil || !errors.Is(err, context.Canceled) {
		_ = result.Close()
		t.Fatalf("pre-canceled rotary: %v", err)
	}
	// Exercise cleanup during CPU frequency construction, table operations and
	// after the first output has escaped the temporary tensor arena.
	for _, threshold := range []int32{5, 19, 27, 29} {
		ctx, cancel := context.WithCancel(context.Background())
		checking := &rotaryCheckingContext{Context: ctx, cancel: cancel, threshold: threshold}
		result, err := ornith.TextRotary(checking, 3, torch.CPUDevice(), referenceFrequencySHA256)
		cancel()
		if result != nil || !errors.Is(err, context.Canceled) {
			_ = result.Close()
			t.Fatalf("mid-operation cancellation at%d: %v", threshold, err)
		}
	}
	first, err := ornith.TextRotary(context.Background(), 1, torch.CPUDevice(), referenceFrequencySHA256)
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()
	second, err := ornith.TextRotary(context.Background(), 1, torch.CPUDevice(), referenceFrequencySHA256)
	if err != nil {
		t.Fatal(err)
	}
	defer second.Close()
	cosine, sine := first.Cosine, first.Sine
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}
	for _, value := range []*torch.Tensor{cosine, sine} {
		if _, err := value.Info(); !errors.Is(err, torch.ErrClosed) {
			t.Fatalf("closed output is still alive: %v", err)
		}
	}
	for index, value := range rotaryRead(t, second.Cosine) {
		if value != 1 {
			t.Fatalf("independent output%d changed after closing first", index)
		}
	}
	var absent *ornith.Rotary
	if err := absent.Close(); err != nil {
		t.Fatal(err)
	}
}
