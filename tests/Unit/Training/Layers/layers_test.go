//go:build libtorch && cgo

package layers_test

import (
	"math"
	"reflect"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/layers"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

func tensor(t *testing.T, data []float64, shape []int64, grad bool) *torch.Tensor {
	t.Helper()
	value, err := torch.FromFloat64(data, shape, torch.CPUDevice(), grad)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = value.Close() })
	return value
}

func own(t *testing.T, value *torch.Tensor, err error) *torch.Tensor {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = value.Close() })
	return value
}

func values(t *testing.T, value *torch.Tensor) []float64 {
	t.Helper()
	result, err := value.Float64Values()
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func closeGradients(t *testing.T, gradients []*torch.Tensor) {
	for _, gradient := range gradients {
		t.Cleanup(func() { _ = gradient.Close() })
	}
}

func near(t *testing.T, got, expected []float64, tolerance float64) {
	t.Helper()
	if len(got) != len(expected) {
		t.Fatalf("length: got %d, expected %d", len(got), len(expected))
	}
	for index := range got {
		if math.IsNaN(got[index]) || math.IsInf(got[index], 0) || math.Abs(got[index]-expected[index]) > tolerance {
			t.Fatalf("element %d: got %.12g, expected %.12g, tolerance %.3g", index, got[index], expected[index], tolerance)
		}
	}
}

func TestLoRALinearPropagatesAcrossFrozenWeights(t *testing.T) {
	for _, zeroB := range []bool{false, true} {
		t.Run(map[bool]string{false: "nonzero_B", true: "fresh_zero_B"}[zeroB], func(t *testing.T) {
			x := []float64{0.1, -0.4, 0.9, 0.8, 0.2, -0.3}
			w := []float64{0.2, -0.5, 0.7, 0.1, 0.4, -0.2}
			a := []float64{0.2, -0.1, 0.4, -0.5, 0.3, 0.1, 0.7, -0.2, 0.5, 0.1, 0.2, -0.3}
			b := []float64{0.2, -0.1, 0.3, 0.4, -0.2, 0.4, 0.1, -0.3}
			if zeroB {
				clear(b)
			}
			dy := []float64{0.7, -0.6, -0.3, 0.2}
			xt, wt := tensor(t, x, []int64{2, 3}, true), tensor(t, w, []int64{2, 3}, false)
			at, bt := tensor(t, a, []int64{4, 3}, true), tensor(t, b, []int64{2, 4}, true)
			result, err := layers.LoRALinear(xt, wt, at, bt, 8)
			result = own(t, result, err)
			gradients, err := torch.Grad([]*torch.Tensor{result}, []*torch.Tensor{xt, at, bt}, []*torch.Tensor{tensor(t, dy, []int64{2, 2}, false)}, false, false)
			if err != nil {
				t.Fatal(err)
			}
			closeGradients(t, gradients)
			y, dx, da, db := make([]float64, 4), make([]float64, 6), make([]float64, 12), make([]float64, 8)
			for row := range 2 {
				for output := range 2 {
					for input := range 3 {
						y[row*2+output] += x[row*3+input] * w[output*3+input]
						dx[row*3+input] += dy[row*2+output] * w[output*3+input]
						for rank := range 4 {
							y[row*2+output] += 2 * x[row*3+input] * a[rank*3+input] * b[output*4+rank]
							dx[row*3+input] += 2 * dy[row*2+output] * a[rank*3+input] * b[output*4+rank]
							da[rank*3+input] += 2 * dy[row*2+output] * x[row*3+input] * b[output*4+rank]
							db[output*4+rank] += 2 * dy[row*2+output] * x[row*3+input] * a[rank*3+input]
						}
					}
				}
			}
			near(t, values(t, result), y, 1e-12)
			for index, expected := range [][]float64{dx, da, db} {
				near(t, values(t, gradients[index]), expected, 1e-12)
			}
			if !reflect.DeepEqual(values(t, wt), w) {
				t.Fatal("frozen base changed")
			}
		})
	}
}

func TestNormalizationUsesRawCheckpointConventions(t *testing.T) {
	x, weight, gate := []float64{0.2, -0.5, 0.7, 0.9, -0.3, 0.1}, []float64{0.4, -0.1, 0.2}, []float64{-0.2, 0.3, 0.7, 0.2, -0.8, 0.4}
	for _, gated := range []bool{false, true} {
		t.Run(map[bool]string{false: "offset_weight", true: "direct_weight_with_gate"}[gated], func(t *testing.T) {
			xt, wt := tensor(t, x, []int64{2, 3}, true), tensor(t, weight, []int64{3}, false)
			gt := tensor(t, gate, []int64{2, 3}, true)
			var result *torch.Tensor
			var err error
			if gated {
				xt, wt, gt = tensor32(t, x, []int64{2, 3}, true), tensor32(t, weight, []int64{3}, false), tensor32(t, gate, []int64{2, 3}, true)
				result, err = layers.GatedRMSNorm(xt, wt, gt, 1e-6)
			} else {
				result, err = layers.RMSNorm(xt, wt, 1e-6)
			}
			result = own(t, result, err)
			var reference func([]float64, []float64) []float64
			reference = func(input, g []float64) []float64 {
				output := make([]float64, len(input))
				for row := range 2 {
					variance := 1e-6
					for column := range 3 {
						variance += input[row*3+column] * input[row*3+column] / 3
					}
					for column := range 3 {
						i := row*3 + column
						scale := weight[column] + 1
						if gated {
							scale = weight[column] * g[i] / (1 + math.Exp(-g[i]))
						}
						output[i] = input[i] / math.Sqrt(variance) * scale
					}
				}
				return output
			}
			near(t, values(t, result), reference(x, gate), 2e-6)
			dy := []float64{0.7, -0.6, -0.3, 0.2, 0.4, -0.1}
			inputs := []*torch.Tensor{xt}
			if gated {
				inputs = append(inputs, gt)
			}
			seed := tensor(t, dy, []int64{2, 3}, false)
			if gated {
				seed = tensor32(t, dy, []int64{2, 3}, false)
			}
			gradients, err := torch.Grad([]*torch.Tensor{result}, inputs, []*torch.Tensor{seed}, false, false)
			if err != nil {
				t.Fatal(err)
			}
			closeGradients(t, gradients)
			for which, gradient := range gradients {
				finiteDifference := make([]float64, len(x))
				for i := range x {
					plus, minus := append([]float64(nil), x...), append([]float64(nil), x...)
					gp, gm := append([]float64(nil), gate...), append([]float64(nil), gate...)
					const step = 1e-4
					if which == 0 {
						plus[i], minus[i] = plus[i]+step, minus[i]-step
					} else {
						gp[i], gm[i] = gp[i]+step, gm[i]-step
					}
					yp, ym := reference(plus, gp), reference(minus, gm)
					for j := range yp {
						finiteDifference[i] += dy[j] * (yp[j] - ym[j]) / (2 * step)
					}
				}
				near(t, values(t, gradient), finiteDifference, 2e-6)
			}
		})
	}
}

func TestLoRARefusesTrainableBaseAndBadDimensions(t *testing.T) {
	x := tensor(t, []float64{1, 2}, []int64{1, 2}, true)
	w := tensor(t, []float64{1, 2}, []int64{1, 2}, true)
	a := tensor(t, []float64{1, 2}, []int64{1, 2}, true)
	b := tensor(t, []float64{1}, []int64{1, 1}, true)
	if result, err := layers.GatedRMSNorm(x, a, x, 1e-6); err == nil || result != nil {
		t.Fatal("gated normalization accepted an unqualified dtype")
	}
	if result, err := layers.LoRALinear(x, w, a, b, 1); err == nil || result != nil {
		t.Fatal("trainable base accepted")
	}
	w = tensor(t, []float64{1, 2}, []int64{1, 2}, false)
	for _, alpha := range []float64{0, -1, math.Inf(1), math.NaN()} {
		if result, err := layers.LoRALinear(x, w, a, b, alpha); err == nil || result != nil {
			t.Fatal("invalid alpha accepted")
		}
	}
	for _, epsilon := range []float64{0, -1, math.Inf(1), math.NaN(), math.SmallestNonzeroFloat64} {
		if result, err := layers.RMSNorm(x, w, epsilon); err == nil || result != nil {
			t.Fatal("invalid epsilon accepted")
		}
	}
}

func TestFrozenFeedForwardStillPropagatesInputGradient(t *testing.T) {
	x, gate, up, down := []float64{0.3, -0.7}, []float64{0.2, -0.1, 0.7, 0.4}, []float64{-0.3, 0.9, 0.5, 0.2}, []float64{0.1, -0.4, 0.7, 0.3}
	xt := tensor(t, x, []int64{1, 2}, true)
	result, err := layers.FeedForward(xt, tensor(t, gate, []int64{2, 2}, false), tensor(t, up, []int64{2, 2}, false), tensor(t, down, []int64{2, 2}, false))
	result = own(t, result, err)
	dy := []float64{0.8, -0.4}
	gradients, err := torch.Grad([]*torch.Tensor{result}, []*torch.Tensor{xt}, []*torch.Tensor{tensor(t, dy, []int64{1, 2}, false)}, false, false)
	if err != nil {
		t.Fatal(err)
	}
	closeGradients(t, gradients)
	expected, derivative := make([]float64, 2), make([]float64, 2)
	for hidden := range 2 {
		g := gate[hidden*2]*x[0] + gate[hidden*2+1]*x[1]
		u := up[hidden*2]*x[0] + up[hidden*2+1]*x[1]
		sigmoid := 1 / (1 + math.Exp(-g))
		for output := range 2 {
			expected[output] += down[output*2+hidden] * g * sigmoid * u
			for input := range 2 {
				derivative[input] += dy[output] * down[output*2+hidden] *
					((sigmoid+g*sigmoid*(1-sigmoid))*gate[hidden*2+input]*u + g*sigmoid*up[hidden*2+input])
			}
		}
	}
	near(t, values(t, result), expected, 1e-12)
	near(t, values(t, gradients[0]), derivative, 1e-12)
}
