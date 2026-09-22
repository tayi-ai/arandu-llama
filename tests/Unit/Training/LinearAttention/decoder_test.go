//go:build libtorch && cgo

package linearattention_test

import (
	"context"
	"math"
	"reflect"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/layers"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

func TestFP16DecoderNormalizesAttentionBeforeStorageRounding(t *testing.T) {
	f := newFixture()
	for row := 0; row < f.batch*f.tokens; row++ {
		copy(f.x[row*f.hidden:(row+1)*f.hidden], []float32{1, 0, 0})
	}
	shrink := func(values []float32) []float32 {
		result := append([]float32(nil), values...)
		for index := range result {
			result[index] *= 1e-4
		}
		return result
	}
	f.qkv, f.z, f.beta, f.alpha, f.conv, f.out = shrink(f.qkv), shrink(f.z), shrink(f.beta), shrink(f.alpha), shrink(f.conv), shrink(f.out)
	_, attention := f.tensors(t, false)
	half := func(values []float32, shape []int64, requiresGrad bool) *torch.Tensor {
		base := tensor(t, values, shape, requiresGrad)
		return own(t, base.To(torch.CPUDevice(), torch.Float16))
	}
	d := int64(f.hidden)
	x := half(f.x, []int64{int64(f.batch), int64(f.tokens), d}, true)
	inputNorm := half([]float32{40000, 0, 0}, []int64{d}, false)
	weights := layers.DecoderWeights{
		InputNorm:         inputNorm,
		PostAttentionNorm: half(data(f.hidden, 0.8, 0.1), []int64{d}, false),
		Gate:              half(data(5*f.hidden, 0.3, 0.04), []int64{5, d}, false),
		Up:                half(data(5*f.hidden, 0.7, 0.04), []int64{5, d}, false),
		Down:              half(data(f.hidden*5, 0.5, 0.04), []int64{d, 5}, false),
		Linear:            &attention,
	}
	x32 := own(t, x.To(torch.CPUDevice(), torch.Float32))
	normalized := own(t, layers.RMSNorm(x32, inputNorm, 1e-4))
	peak := float32(0)
	for _, value := range read(t, normalized) {
		peak = max(peak, float32(math.Abs(float64(value))))
	}
	if peak <= 65504 {
		t.Fatalf("test did not exceed Float16 range: %g", peak)
	}
	config := layers.DecoderConfig{Epsilon: 1e-4, MaxInputElements: int64(len(f.x)), Linear: f.config}
	output := own(t, layers.DecoderForward(context.Background(), x, weights, nil, nil, nil, config))
	finite, err := output.AllFinite()
	if err != nil || !finite {
		t.Fatalf("attention normalization rounded through Float16: %v", err)
	}
	info, err := output.Info()
	if err != nil || info.DType != torch.Float16 {
		t.Fatalf("qualified residual storage changed: %+v, %v", info, err)
	}
}

// A scalar decoder oracle exercises both residual paths, both normalizations,
// and the recurrent input gradient after the independent MLP graph is consumed.
func TestRecurrentDecoderVJPAgainstIndependentScalarOracle(t *testing.T) {
	f := newFixture()
	x, attention := f.tensors(t, true)
	const intermediate = 5
	inputNorm := data(f.hidden, 0.2, 0.1)
	postNorm := data(f.hidden, 0.8, 0.1)
	gate := data(intermediate*f.hidden, 0.3, 0.4)
	up := data(intermediate*f.hidden, 0.7, 0.4)
	down := data(f.hidden*intermediate, 0.5, 0.4)
	weights := layers.DecoderWeights{
		InputNorm:         tensor(t, inputNorm, []int64{int64(f.hidden)}, false),
		PostAttentionNorm: tensor(t, postNorm, []int64{int64(f.hidden)}, false),
		Gate:              tensor(t, gate, []int64{intermediate, int64(f.hidden)}, false),
		Up:                tensor(t, up, []int64{intermediate, int64(f.hidden)}, false),
		Down:              tensor(t, down, []int64{int64(f.hidden), intermediate}, false),
		Linear:            &attention,
	}
	config := layers.DecoderConfig{Epsilon: 1e-4, MaxInputElements: int64(len(f.x)), Linear: f.config}
	normalize := func(input []float64, weight []float32) []float64 {
		result := make([]float64, len(input))
		for row := 0; row < f.batch*f.tokens; row++ {
			variance := config.Epsilon
			for column := 0; column < f.hidden; column++ {
				value := input[row*f.hidden+column]
				variance += value * value / float64(f.hidden)
			}
			for column := 0; column < f.hidden; column++ {
				index := row*f.hidden + column
				result[index] = input[index] / math.Sqrt(variance) * (1 + float64(weight[column]))
			}
		}
		return result
	}
	oracle := func(input []float64) []float64 {
		attended := f.oracle(normalize(input, inputNorm))
		residual := make([]float64, len(input))
		for index := range input {
			residual[index] = input[index] + attended[index]
		}
		normalized := normalize(residual, postNorm)
		result := append([]float64(nil), residual...)
		for row := 0; row < f.batch*f.tokens; row++ {
			for feature := 0; feature < intermediate; feature++ {
				var gated, lifted float64
				for column := 0; column < f.hidden; column++ {
					value := normalized[row*f.hidden+column]
					gated += value * float64(gate[feature*f.hidden+column])
					lifted += value * float64(up[feature*f.hidden+column])
				}
				activated := gated / (1 + math.Exp(-gated)) * lifted
				for column := 0; column < f.hidden; column++ {
					result[row*f.hidden+column] += activated * float64(down[column*intermediate+feature])
				}
			}
		}
		return result
	}
	dy := data(len(f.x), 0.3, 0.7)
	seed := tensor(t, dy, []int64{int64(f.batch), int64(f.tokens), int64(f.hidden)}, false)
	forward, err := layers.DecoderForward(context.Background(), x, weights, nil, nil, nil, config)
	forward = own(t, forward, err)
	assertDetached(t, forward)
	forwardError := near(t, read(t, forward), oracle(doubles(f.x)), 4e-6)
	finiteDifferences := make([]float64, len(f.x))
	for index := range finiteDifferences {
		plus, minus := doubles(f.x), doubles(f.x)
		const step = 1e-5
		plus[index] += step
		minus[index] -= step
		yp, ym := oracle(plus), oracle(minus)
		for output := range yp {
			finiteDifferences[index] += float64(dy[output]) * (yp[output] - ym[output]) / (2 * step)
		}
	}
	var first []float32
	for iteration := 0; iteration < 2; iteration++ {
		gradients, err := layers.DecoderVJP(context.Background(), x, weights, nil, nil, nil, seed, config)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = gradients.Close() })
		assertDetached(t, gradients.Input)
		got := read(t, gradients.Input)
		gradientError := near(t, got, finiteDifferences, 4e-5)
		if iteration == 0 {
			first = got
			t.Logf("scalar decoder oracle: forward max error %.9g; %d input finite differences max error %.9g", forwardError, len(got), gradientError)
		} else {
			near(t, got, doubles(first), 1e-6)
		}
		if gradients.QueryA != nil || gradients.QueryB != nil || gradients.ValueA != nil || gradients.ValueB != nil {
			t.Fatal("recurrent decoder unexpectedly returned adapter gradients")
		}
	}
	if !reflect.DeepEqual(read(t, x), f.x) || !reflect.DeepEqual(read(t, weights.Gate), gate) || !reflect.DeepEqual(read(t, seed), dy) {
		t.Fatal("borrowed decoder input, weights, or cotangent changed")
	}
	info, err := x.Info()
	if err != nil || !info.RequiresGrad {
		t.Fatalf("caller gradient flag changed: %v", err)
	}
}
