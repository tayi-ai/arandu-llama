//go:build libtorch && cgo

package decoder_test

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/decoder"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

var referenceFrequencySHA256 = func() string {
	var raw [16]byte
	for i, v := range []float32{1, .25, .0625, .015625} {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(v))
	}
	return fmt.Sprintf("%x", sha256.Sum256(raw[:]))
}()

func rotarySpec(digest string) decoder.RotarySpec {
	return decoder.RotarySpec{Theta: 256, Dimension: 8, MaxTokens: 4096, HalfPrecision: true, ExpectedSHA256: digest}
}

func TestTextRotaryAdmitsExactReferenceFrequencyHash(t *testing.T) {
	rotary, err := decoder.TextRotary(context.Background(), 3, torch.CPUDevice(), rotarySpec(referenceFrequencySHA256))
	if err != nil {
		t.Fatal(err)
	}
	defer rotary.Close()
	for _, value := range []*torch.Tensor{rotary.Cosine, rotary.Sine} {
		info, err := value.Info()
		if err != nil {
			t.Fatal(err)
		}
		if info.DType != torch.Float32 || info.Device != torch.CPUDevice() || info.RequiresGrad || len(info.Shape) != 2 || info.Shape[0] != 3 || info.Shape[1] != 4 {
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
	rotary, err := decoder.TextRotary(context.Background(), tokens, torch.CPUDevice(), rotarySpec(referenceFrequencySHA256))
	if err != nil {
		t.Fatal(err)
	}
	defer rotary.Close()
	cosine, sine := rotaryRead(t, rotary.Cosine), rotaryRead(t, rotary.Sine)
	var maximumHalfULP float64
	roundedEntries := 0
	for position := 0; position < tokens; position++ {
		for column := 0; column < 4; column++ {
			// HF uses FP32 powers followed by an FP32 reciprocal. Go's math.Pow
			// supplies an independent scalar implementation; native pow/trig
			// may differ by a small ULP before rounding to the storage dtype.
			power := float32(math.Pow(256, float64(column)/4))
			frequency := float32(1) / power
			angle := float32(position) * frequency
			for _, entry := range []struct {
				got, raw float32
			}{
				{cosine[position*4+column], float32(math.Cos(float64(angle)))},
				{sine[position*4+column], float32(math.Sin(float64(angle)))},
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
	for column := 0; column < 4; column++ {
		if cosine[column] != 1 || math.Float32bits(sine[column]) != 0 {
			t.Fatal("position zero must be cosine one and positive-zero sine")
		}
	}
	t.Logf("72 scalar comparisons; max error %.4g half ULP; %d entries exercise FP16 rounding", maximumHalfULP, roundedEntries)
}

func TestTextRotaryMaximumTokensHaveSequentialPositionsAndBoundedTables(t *testing.T) {
	rotary, err := decoder.TextRotary(context.Background(), 4096, torch.CPUDevice(), rotarySpec(referenceFrequencySHA256))
	if err != nil {
		t.Fatal(err)
	}
	defer rotary.Close()
	for _, value := range []*torch.Tensor{rotary.Cosine, rotary.Sine} {
		info, err := value.Info()
		if err != nil || len(info.Shape) != 2 || info.Shape[0] != 4096 || info.Shape[1] != 4 || info.Elements != 4096*4 {
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
			{cosine[position*4], rotaryHalfRound(float32(math.Cos(float64(position))))},
			{sine[position*4], rotaryHalfRound(float32(math.Sin(float64(position))))},
		} {
			if math.Abs(float64(entry.got)-float64(entry.expected)) > rotaryHalfStep(entry.expected) {
				t.Fatalf("position%d is not the expected sequential text position", position)
			}
		}
	}
}

func TestTextRotaryRejectsInvalidLimitsIdentityAndDevice(t *testing.T) {
	for _, tokens := range []int{-1, 0, 4097, int(^uint(0) >> 1)} {
		result, err := decoder.TextRotary(context.Background(), tokens, torch.CPUDevice(), rotarySpec(referenceFrequencySHA256))
		if result != nil || !errors.Is(err, decoder.ErrRotarySpec) {
			_ = result.Close()
			t.Fatalf("invalid tokens%d: %v", tokens, err)
		}
	}
	for _, digest := range []string{"", referenceFrequencySHA256[:63], strings.Repeat("g", 64), strings.ToUpper(referenceFrequencySHA256)} {
		result, err := decoder.TextRotary(context.Background(), 3, torch.CPUDevice(), rotarySpec(digest))
		if result != nil || !errors.Is(err, decoder.ErrRotarySpec) {
			_ = result.Close()
			t.Fatalf("invalid digest accepted: %v", err)
		}
	}
	// Invalid target placement would produce a different error if reached:
	// the frequency identity must be checked before target-device operations.
	invalidDevice := torch.Device{Kind: "invalid", Index: 0}
	result, err := decoder.TextRotary(context.Background(), 3, invalidDevice, rotarySpec(strings.Repeat("0", 64)))
	if result != nil || !errors.Is(err, decoder.ErrRotaryIdentity) {
		_ = result.Close()
		t.Fatalf("frequency guard did not precede target placement: %v", err)
	}
	result, err = decoder.TextRotary(context.Background(), 3, invalidDevice, rotarySpec(referenceFrequencySHA256))
	if result != nil || err == nil || errors.Is(err, decoder.ErrRotaryIdentity) {
		_ = result.Close()
		t.Fatalf("invalid device admission: %v", err)
	}
	result, err = decoder.TextRotary(nil, 3, torch.CPUDevice(), rotarySpec(referenceFrequencySHA256))
	if result != nil || !errors.Is(err, decoder.ErrRotarySpec) {
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
	if result, err := decoder.TextRotary(ctx, 3, torch.CPUDevice(), rotarySpec(referenceFrequencySHA256)); result != nil || !errors.Is(err, context.Canceled) {
		_ = result.Close()
		t.Fatalf("pre-canceled rotary: %v", err)
	}
	// Exercise cleanup during CPU frequency construction, table operations and
	// after the first output has escaped the temporary tensor arena.
	for _, threshold := range []int32{5, 19, 27, 29} {
		ctx, cancel := context.WithCancel(context.Background())
		checking := &rotaryCheckingContext{Context: ctx, cancel: cancel, threshold: threshold}
		result, err := decoder.TextRotary(checking, 3, torch.CPUDevice(), rotarySpec(referenceFrequencySHA256))
		cancel()
		if result != nil || !errors.Is(err, context.Canceled) {
			_ = result.Close()
			t.Fatalf("mid-operation cancellation at%d: %v", threshold, err)
		}
	}
	first, err := decoder.TextRotary(context.Background(), 1, torch.CPUDevice(), rotarySpec(referenceFrequencySHA256))
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()
	second, err := decoder.TextRotary(context.Background(), 1, torch.CPUDevice(), rotarySpec(referenceFrequencySHA256))
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
	var absent *decoder.Rotary
	if err := absent.Close(); err != nil {
		t.Fatal(err)
	}
}
