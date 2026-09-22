package torch_test

import (
	"math"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/torch"
)

func TestCUDAAlternatingDevicesPreservesMatMulGradientAndBFloat16Boundary(t *testing.T) {
	if !torch.CUDAEnabled() {
		t.Skip("CUDA bridge is not enabled")
	}
	count, err := torch.CUDADeviceCount()
	if err != nil {
		t.Fatal(err)
	}
	if count < 2 {
		t.Skip("two CUDA devices are required")
	}
	for iteration := range 8 {
		for _, device := range []int{0, 1, 1, 0} {
			values := []float32{float32(device + 1), 2, 3, float32(iteration + 1)}
			x, err := torch.FromFloat32(values, []int64{2, 2}, torch.CUDADevice(device), true)
			if err != nil {
				t.Fatal(err)
			}
			y, err := x.MatMul(x)
			if err != nil {
				_ = x.Close()
				t.Fatal(err)
			}
			loss, err := y.Sum(nil, false)
			if err != nil {
				_ = y.Close()
				_ = x.Close()
				t.Fatal(err)
			}
			seed, err := torch.FromFloat32([]float32{1}, nil, torch.CUDADevice(device), false)
			if err != nil {
				_ = loss.Close()
				_ = y.Close()
				_ = x.Close()
				t.Fatal(err)
			}
			gradients, err := torch.Grad([]*torch.Tensor{loss}, []*torch.Tensor{x}, []*torch.Tensor{seed}, false, false)
			if err != nil {
				_ = seed.Close()
				_ = loss.Close()
				_ = y.Close()
				_ = x.Close()
				t.Fatal(err)
			}
			actual, err := y.Float32Values()
			gradient, gradientErr := gradients[0].Float32Values()
			closeErr := gradients[0].Close()
			_ = seed.Close()
			_ = loss.Close()
			_ = y.Close()
			_ = x.Close()
			if err != nil || gradientErr != nil || closeErr != nil {
				t.Fatal(err, gradientErr, closeErr)
			}
			a, b, c, d := values[0], values[1], values[2], values[3]
			expected := []float32{a*a + b*c, a*b + b*d, c*a + d*c, c*b + d*d}
			expectedGradient := []float32{2*a + b + c, a + 2*c + d, a + 2*b + d, b + c + 2*d}
			for index := range expected {
				if actual[index] != expected[index] || gradient[index] != expectedGradient[index] {
					t.Fatalf("iteration %d device %d index %d: product=%g want=%g gradient=%g want=%g", iteration, device, index, actual[index], expected[index], gradient[index], expectedGradient[index])
				}
			}
		}
	}

	const elements = 879 * 4096
	values := make([]float32, elements)
	for index := range values {
		values[index] = float32(index%991-247) / 16
	}
	large, err := torch.FromFloat32(values, []int64{1, 879, 4096}, torch.CUDADevice(1), false)
	if err != nil {
		t.Fatal(err)
	}
	rounded, err := large.RoundBFloat16()
	_ = large.Close()
	if err != nil {
		t.Fatal(err)
	}
	actual, err := rounded.Float32Values()
	closeErr := rounded.Close()
	if err != nil || closeErr != nil {
		t.Fatal(err, closeErr)
	}
	for index, value := range actual {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || math.Float32bits(value)&0xffff != 0 {
			t.Fatalf("production boundary index %d is not finite bfloat16-grid Float32: %08x", index, math.Float32bits(value))
		}
	}
}
