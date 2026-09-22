//go:build libtorch && cgo

package torch_test

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/torch"
)

func scope(t *testing.T) func(*torch.Tensor, error) *torch.Tensor {
	t.Helper()
	return func(tensor *torch.Tensor, err error) *torch.Tensor {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := tensor.Close(); err != nil {
				t.Error(err)
			}
		})
		return tensor
	}
}

func values(t *testing.T, tensor *torch.Tensor) []float64 {
	t.Helper()
	result, err := tensor.Float64Values()
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func near(t *testing.T, actual, expected []float64, tolerance float64) {
	t.Helper()
	if len(actual) != len(expected) {
		t.Fatalf("length: got%d want%d", len(actual), len(expected))
	}
	for i, value := range actual {
		if math.IsNaN(value) || math.Abs(value-expected[i]) > tolerance {
			t.Fatalf("index%d: got%.17g want%.17g tolerance%g", i, value, expected[i], tolerance)
		}
	}
}

func TestNativeLoRAVJPMatchesAnalyticAndFiniteDifference(t *testing.T) {
	for _, zeroB := range []bool{true, false} {
		t.Run(map[bool]string{true: "fresh_zero_B", false: "nonzero_B"}[zeroB], func(t *testing.T) {
			keep := scope(t)
			x := []float64{.3, -.2, .5}
			w := []float64{.2, .1, -.3, -.1, .4, .2, .5, -.2, .1, .3, .2, .4}
			a := []float64{.1, .2, .3, -.1, .3, .2, .2, -.2, .1, .4, .1, -.3}
			b := make([]float64, 16)
			if !zeroB {
				for i := range b {
					b[i] = float64(i-7) / 50
				}
			}
			xTensor := keep(torch.FromFloat64(x, []int64{3}, torch.CPUDevice(), true))
			wTensor := keep(torch.FromFloat64(w, []int64{4, 3}, torch.CPUDevice(), false))
			aTensor := keep(torch.FromFloat64(a, []int64{4, 3}, torch.CPUDevice(), true))
			bTensor := keep(torch.FromFloat64(b, []int64{4, 4}, torch.CPUDevice(), true))
			base := keep(wTensor.MatMul(xTensor))
			ax := keep(aTensor.MatMul(xTensor))
			ba := keep(bTensor.MatMul(ax))
			delta := keep(ba.Scale(2))
			output := keep(base.Add(delta))
			logits := linearReference(w, a, b, x)
			near(t, values(t, output), logits, 1e-14)
			target := []float64{.1, .2, .3, .4}
			_, g := lossReference(logits, target)
			for i := range g {
				g[i] /= 1024
			}
			seed := keep(torch.FromFloat64(g, []int64{4}, torch.CPUDevice(), false))
			// Saved graph ownership must outlive closed intermediate Go handles.
			for _, intermediate := range []*torch.Tensor{base, ax, ba, delta} {
				if err := intermediate.Close(); err != nil {
					t.Fatal(err)
				}
			}
			gradients, err := torch.Grad([]*torch.Tensor{output}, []*torch.Tensor{aTensor, bTensor, xTensor}, []*torch.Tensor{seed}, false, false)
			if err != nil {
				t.Fatal(err)
			}
			for _, gradient := range gradients {
				keep(gradient, nil)
			}
			expectedA, expectedB, expectedX := analyticVJP(w, a, b, x, g)
			for i, expected := range [][]float64{expectedA, expectedB, expectedX} {
				near(t, values(t, gradients[i]), expected, 1e-14)
			}
			for group, parameter := range [][]float64{a, b, x} {
				finiteDifference := make([]float64, len(parameter))
				for i := range parameter {
					original := parameter[i]
					parameter[i] = original + 1e-6
					plus, _ := lossReference(linearReference(w, a, b, x), target)
					parameter[i] = original - 1e-6
					minus, _ := lossReference(linearReference(w, a, b, x), target)
					parameter[i] = original
					finiteDifference[i] = (plus - minus) / (2e-6 * 1024)
				}
				near(t, values(t, gradients[group]), finiteDifference, 3e-12)
			}
			near(t, values(t, wTensor), w, 0)
			info, err := wTensor.Info()
			if err != nil || info.RequiresGrad {
				t.Fatalf("frozen base metadata: %+v %v", info, err)
			}
			if zeroB {
				for _, v := range values(t, gradients[0]) {
					if v != 0 {
						t.Fatal("fresh B must yield zero A gradient")
					}
				}
			}
		})
	}
}

func linearReference(w, a, b, x []float64) []float64 {
	output := make([]float64, 4)
	ax := make([]float64, 4)
	for j := 0; j < 4; j++ {
		for k := 0; k < 3; k++ {
			ax[j] += a[j*3+k] * x[k]
		}
	}
	for i := 0; i < 4; i++ {
		for k := 0; k < 3; k++ {
			output[i] += w[i*3+k] * x[k]
		}
		for j := 0; j < 4; j++ {
			output[i] += 2 * b[i*4+j] * ax[j]
		}
	}
	return output
}

func TestNativeFloat32LoRAVJP(t *testing.T) {
	keep := scope(t)
	x := []float64{.5, -.25, .125}
	w, a, b := make([]float64, 12), make([]float64, 12), make([]float64, 16)
	for i := range w {
		w[i] = float64(i-5) / 16
		a[i] = float64(i-3) / 32
	}
	for i := range b {
		b[i] = float64(i-7) / 64
	}
	makeTensor := func(data []float64, shape []int64, grad bool) *torch.Tensor {
		input := make([]float32, len(data))
		for i := range input {
			input[i] = float32(data[i])
		}
		return keep(torch.FromFloat32(input, shape, torch.CPUDevice(), grad))
	}
	wt, at, bt, xt := makeTensor(w, []int64{4, 3}, false), makeTensor(a, []int64{4, 3}, true), makeTensor(b, []int64{4, 4}, true), makeTensor(x, []int64{3}, true)
	output := keep(keep(wt.MatMul(xt)).Add(keep(keep(bt.MatMul(keep(at.MatMul(xt)))).Scale(2))))
	g := []float64{1.0 / 1024, 2.0 / 1024, -1.0 / 1024, 3.0 / 1024}
	seed := makeTensor(g, []int64{4}, false)
	gradients, err := torch.Grad([]*torch.Tensor{output}, []*torch.Tensor{at, bt, xt}, []*torch.Tensor{seed}, false, false)
	if err != nil {
		t.Fatal(err)
	}
	ga, gb, gx := analyticVJP(w, a, b, x, g)
	for i, expected := range [][]float64{ga, gb, gx} {
		gradient := keep(gradients[i], nil)
		info, err := gradient.Info()
		if err != nil || info.DType != torch.Float32 {
			t.Fatalf("FP32 gradient: %+v %v", info, err)
		}
		near(t, values(t, gradient), expected, 1e-9)
	}
}

func lossReference(logits, target []float64) (float64, []float64) {
	maximum := logits[0]
	for _, value := range logits {
		maximum = math.Max(maximum, value)
	}
	normalizer := 0.0
	for _, value := range logits {
		normalizer += math.Exp(value - maximum)
	}
	loss := 0.0
	gradient := make([]float64, len(logits))
	for i, value := range logits {
		loss -= target[i] * (value - maximum - math.Log(normalizer))
		gradient[i] = math.Exp(value-maximum)/normalizer - target[i]
	}
	return loss, gradient
}

func analyticVJP(w, a, b, x, g []float64) ([]float64, []float64, []float64) {
	ga, gb, gx := make([]float64, 12), make([]float64, 16), make([]float64, 3)
	for j := 0; j < 4; j++ {
		ax := 0.0
		for k := 0; k < 3; k++ {
			ax += a[j*3+k] * x[k]
		}
		for i := 0; i < 4; i++ {
			gb[i*4+j] = 2 * g[i] * ax
			for k := 0; k < 3; k++ {
				ga[j*3+k] += 2 * b[i*4+j] * g[i] * x[k]
				gx[k] += 2 * a[j*3+k] * b[i*4+j] * g[i]
			}
		}
	}
	for k := 0; k < 3; k++ {
		for i := 0; i < 4; i++ {
			gx[k] += w[i*3+k] * g[i]
		}
	}
	return ga, gb, gx
}

func TestNativeTensorOwnershipDTypesAndErrors(t *testing.T) {
	keep := scope(t)
	if !torch.Enabled() {
		t.Fatal("native build unavailable")
	}
	version, err := torch.Version()
	if err != nil || version == "" {
		t.Fatalf("version=%q err=%v", version, err)
	}
	source := []float32{1, 2, 3, 4}
	x := keep(torch.FromFloat32(source, []int64{2, 2}, torch.CPUDevice(), false))
	source[0] = 99
	near(t, values(t, x), []float64{1, 2, 3, 4}, 0)
	info, err := x.Info()
	if err != nil || info.DType != torch.Float32 || info.Device != torch.CPUDevice() || !reflect.DeepEqual(info.Shape, []int64{2, 2}) {
		t.Fatalf("metadata: %+v %v", info, err)
	}
	info.Shape[0] = 9
	again, _ := x.Info()
	if again.Shape[0] != 2 {
		t.Fatal("metadata aliases state")
	}
	for _, dtype := range []torch.DType{torch.Float16, torch.BFloat16, torch.Float32, torch.Float64, torch.Int64, torch.Bool} {
		converted := keep(x.To(torch.CPUDevice(), dtype))
		data, err := converted.Bytes()
		if err != nil {
			t.Fatal(err)
		}
		copied := keep(torch.FromBytes(data, []int64{2, 2}, dtype, torch.CPUDevice(), false))
		roundtrip, err := copied.Bytes()
		if err != nil || !bytes.Equal(data, roundtrip) {
			t.Fatalf("dtype%d raw copy failed: %v", dtype, err)
		}
	}
	floats, err := x.Float32Values()
	if err != nil || !reflect.DeepEqual(floats, []float32{1, 2, 3, 4}) {
		t.Fatalf("float32 values: %v %v", floats, err)
	}
	view := keep(x.Transpose(0, 1))
	copiedHandle := *x
	if err := x.Close(); err != nil {
		t.Fatal(err)
	}
	if err := copiedHandle.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := copiedHandle.Info(); !errors.Is(err, torch.ErrClosed) {
		t.Fatalf("copied closed handle: %v", err)
	}
	near(t, values(t, view), []float64{1, 3, 2, 4}, 0)
	for _, test := range []struct {
		shape  []int64
		dtype  torch.DType
		device torch.Device
		grad   bool
	}{
		{[]int64{-1}, torch.Float32, torch.CPUDevice(), false},
		{[]int64{math.MaxInt64, 2}, torch.Float32, torch.CPUDevice(), false},
		{[]int64{1}, torch.Int64, torch.CPUDevice(), true},
		{[]int64{4}, 99, torch.CPUDevice(), false},
		{[]int64{4}, torch.Float32, torch.CUDADevice(-1), false},
	} {
		if tensor, err := torch.FromBytes(make([]byte, 16), test.shape, test.dtype, test.device, test.grad); err == nil {
			tensor.Close()
			t.Fatal("invalid constructor accepted")
		}
	}
	empty := keep(torch.FromFloat64(nil, []int64{0, 3}, torch.CPUDevice(), false))
	if len(values(t, empty)) != 0 {
		t.Fatal("nonempty empty tensor")
	}
	halfBits := []byte{0x01, 0x7e, 0x00, 0x7c, 0x00, 0xfc}
	half := keep(torch.FromBytes(halfBits, []int64{3}, torch.Float16, torch.CPUDevice(), false))
	halfCopy, err := half.Bytes()
	if err != nil || !bytes.Equal(halfCopy, halfBits) {
		t.Fatalf("float16 payload bits changed: %x %v", halfCopy, err)
	}
	if _, err := view.Reshape([]int64{5}); err == nil {
		t.Fatal("reshape mismatch accepted")
	}
	if _, err := view.MatMul(empty); err == nil {
		t.Fatal("matmul mismatch accepted")
	}
	if _, err := view.ClampMin(math.NaN()); err == nil {
		t.Fatal("NaN clamp minimum accepted")
	}
	near(t, values(t, view), []float64{1, 3, 2, 4}, 0)
}

func TestNativePrimitiveCompositionAndFiniteChecks(t *testing.T) {
	keep := scope(t)
	x := keep(torch.FromFloat64([]float64{1, 2, 3, 4}, []int64{2, 2}, torch.CPUDevice(), false))
	one := keep(torch.FromFloat64([]float64{1}, nil, torch.CPUDevice(), false))
	near(t, values(t, keep(x.Add(one))), []float64{2, 3, 4, 5}, 0)
	near(t, values(t, keep(x.Sub(one))), []float64{0, 1, 2, 3}, 0)
	near(t, values(t, keep(x.Mul(x))), []float64{1, 4, 9, 16}, 0)
	near(t, values(t, keep(x.Sum([]int64{1}, true))), []float64{3, 7}, 0)
	near(t, values(t, keep(x.Mean(nil, false))), []float64{2.5}, 0)
	negative := keep(torch.FromFloat64([]float64{-4, 2, -3, 1}, []int64{2, 2}, torch.CPUDevice(), false))
	near(t, values(t, keep(negative.Abs())), []float64{4, 2, 3, 1}, 0)
	near(t, values(t, keep(keep(negative.Abs()).AMax([]int64{1}, true))), []float64{4, 3}, 0)
	near(t, values(t, keep(negative.ClampMin(-1))), []float64{-1, 2, -1, 1}, 0)
	near(t, values(t, keep(keep(x.Exp()).Log())), []float64{1, 2, 3, 4}, 1e-14)
	near(t, values(t, keep(x.RSqrt())), []float64{1, 1 / math.Sqrt(2), 1 / math.Sqrt(3), .5}, 1e-14)
	sigmoid := make([]float64, 4)
	silu := make([]float64, 4)
	softplus := make([]float64, 4)
	for i := range sigmoid {
		v := float64(i + 1)
		sigmoid[i] = 1 / (1 + math.Exp(-v))
		silu[i] = v * sigmoid[i]
		softplus[i] = math.Log1p(math.Exp(v))
	}
	near(t, values(t, keep(x.Sigmoid())), sigmoid, 1e-14)
	near(t, values(t, keep(x.SiLU())), silu, 1e-14)
	near(t, values(t, keep(x.Softplus())), softplus, 1e-14)
	softmax := keep(x.Softmax(1))
	near(t, values(t, keep(softmax.Sum([]int64{1}, false))), []float64{1, 1}, 1e-14)
	near(t, values(t, keep(x.Tril(0))), []float64{1, 0, 3, 4}, 0)
	mask := keep(torch.FromBytes([]byte{0, 1, 0, 0}, []int64{2, 2}, torch.Bool, torch.CPUDevice(), false))
	masked := keep(x.MaskedFill(mask, math.Inf(-1)))
	if finite, err := masked.AllFinite(); err != nil || finite {
		t.Fatalf("masked finite: %v %v", finite, err)
	}
	if finite, err := x.AllFinite(); err != nil || !finite {
		t.Fatalf("finite: %v %v", finite, err)
	}
	nan := keep(torch.FromFloat64([]float64{math.NaN()}, nil, torch.CPUDevice(), false))
	if finite, err := nan.AllFinite(); err != nil || finite {
		t.Fatalf("NaN finite: %v %v", finite, err)
	}
	first := keep(x.Select(0, 0))
	last := keep(x.Slice(0, 1, 2, 1))
	near(t, values(t, keep(torch.Cat([]*torch.Tensor{keep(first.Unsqueeze(0)), last}, 0))), []float64{1, 2, 3, 4}, 0)
	near(t, values(t, keep(torch.Stack([]*torch.Tensor{first, first}, 0))), []float64{1, 2, 1, 2}, 0)
	near(t, values(t, keep(keep(first.Unsqueeze(0)).Squeeze(0))), []float64{1, 2}, 0)
	leaf := keep(keep(x.Detach()).SetRequiresGrad(true))
	if info, _ := x.Info(); info.RequiresGrad {
		t.Fatal("detached leaf changed input requires_grad")
	}
	clone := keep(leaf.Clone())
	seed := keep(torch.FromFloat64([]float64{1, 1, 1, 1}, []int64{2, 2}, torch.CPUDevice(), false))
	grad, err := torch.Grad([]*torch.Tensor{clone}, []*torch.Tensor{leaf}, []*torch.Tensor{seed}, false, false)
	if err != nil {
		t.Fatal(err)
	}
	near(t, values(t, keep(grad[0], nil)), []float64{1, 1, 1, 1}, 0)
}

func TestNativeAutogradHigherOrderAndFailures(t *testing.T) {
	keep := scope(t)
	x := keep(torch.FromFloat64([]float64{2}, nil, torch.CPUDevice(), true))
	y := keep(x.Mul(x))
	if changed, err := y.SetRequiresGrad(false); err == nil {
		changed.Close()
		t.Fatal("non-leaf gradient flag change accepted")
	}
	one := keep(torch.FromFloat64([]float64{1}, nil, torch.CPUDevice(), false))
	first, err := torch.Grad([]*torch.Tensor{y}, []*torch.Tensor{x}, []*torch.Tensor{one}, false, true)
	if err != nil {
		t.Fatal(err)
	}
	keep(first[0], nil)
	second, err := torch.Grad(first, []*torch.Tensor{x}, []*torch.Tensor{one}, false, false)
	if err != nil {
		t.Fatal(err)
	}
	near(t, values(t, keep(second[0], nil)), []float64{2}, 0)
	unused := keep(torch.FromFloat64([]float64{3}, nil, torch.CPUDevice(), true))
	if _, err := torch.Grad([]*torch.Tensor{x}, []*torch.Tensor{unused}, []*torch.Tensor{one}, false, false); err == nil {
		t.Fatal("unused input accepted")
	}
	if _, err := torch.Grad([]*torch.Tensor{x}, []*torch.Tensor{one}, []*torch.Tensor{one}, false, false); err == nil {
		t.Fatal("frozen input accepted")
	}
	var workers sync.WaitGroup
	for range 8 {
		workers.Go(func() {
			if _, err := x.Unsqueeze(99); err == nil || !strings.Contains(err.Error(), "Dimension") && !strings.Contains(err.Error(), "dimension") {
				t.Errorf("per-call native error: %v", err)
			}
			if info, err := x.Info(); err != nil || info.Elements != 1 {
				t.Errorf("error poisoned following call: %+v %v", info, err)
			}
		})
	}
	workers.Wait()
}

func TestBFloat16RoundingReturnsFloat32AndPreservesGradient(t *testing.T) {
	keep := scope(t)
	source := []float32{1.00390625, -2.01171875, 65536, 1e-30}
	x := keep(torch.FromFloat32(source, []int64{4}, torch.CPUDevice(), true))
	rounded := keep(x.RoundBFloat16())
	info, err := rounded.Info()
	if err != nil {
		t.Fatal(err)
	}
	if info.DType != torch.Float32 || info.Device != torch.CPUDevice() || !info.RequiresGrad {
		t.Fatalf("rounded metadata: %+v", info)
	}
	actual, err := rounded.Float32Values()
	if err != nil {
		t.Fatal(err)
	}
	for index, value := range actual {
		if !isFinite32(value) || math.Float32bits(value)&0xffff != 0 {
			t.Fatalf("value %d was not rounded to the bfloat16 grid: %08x", index, math.Float32bits(value))
		}
	}
	seed := keep(torch.FromFloat32([]float32{1, 1, 1, 1}, []int64{4}, torch.CPUDevice(), false))
	gradients, err := torch.Grad([]*torch.Tensor{rounded}, []*torch.Tensor{x}, []*torch.Tensor{seed}, false, false)
	if err != nil {
		t.Fatal(err)
	}
	near(t, values(t, keep(gradients[0], nil)), []float64{1, 1, 1, 1}, 0)

	half := keep(rounded.To(torch.CPUDevice(), torch.Float16))
	if invalid, err := half.RoundBFloat16(); err == nil {
		_ = invalid.Close()
		t.Fatal("non-Float32 rounding input accepted")
	}
}

func isFinite32(value float32) bool {
	return !float32IsNaN(value) && !float32IsInf(value)
}

func float32IsNaN(value float32) bool {
	bits := math.Float32bits(value)
	return bits&0x7f800000 == 0x7f800000 && bits&0x007fffff != 0
}

func float32IsInf(value float32) bool {
	return math.Float32bits(value)&0x7fffffff == 0x7f800000
}

func TestNativeConcurrentCloseAndRead(t *testing.T) {
	keep := scope(t)
	x := keep(torch.FromFloat32([]float32{1, 2}, []int64{2}, torch.CPUDevice(), false))
	start := make(chan struct{})
	var workers sync.WaitGroup
	for range 8 {
		workers.Go(func() {
			<-start
			for range 20 {
				actual, err := x.Float32Values()
				if err != nil && !errors.Is(err, torch.ErrClosed) {
					t.Errorf("concurrent copy: %v", err)
				}
				if err == nil && !reflect.DeepEqual(actual, []float32{1, 2}) {
					t.Errorf("concurrent values: %v", actual)
				}
			}
		})
	}
	workers.Go(func() {
		<-start
		if err := x.Close(); err != nil {
			t.Error(err)
		}
	})
	close(start)
	workers.Wait()
}

func TestNativeIndexSelectPreservesRepeatedIndexGradients(t *testing.T) {
	keep := scope(t)
	x := keep(torch.FromFloat32([]float32{1, 2, 3, 4, 5, 6}, []int64{3, 2}, torch.CPUDevice(), true))
	indexBytes := make([]byte, 24)
	for i, index := range []uint64{2, 0, 2} {
		binary.LittleEndian.PutUint64(indexBytes[i*8:], index)
	}
	indices := keep(torch.FromBytes(indexBytes, []int64{3}, torch.Int64, torch.CPUDevice(), false))
	selected := keep(x.IndexSelect(0, indices))
	near(t, values(t, selected), []float64{5, 6, 1, 2, 5, 6}, 0)
	seed := keep(torch.FromFloat32([]float32{1, 1, 1, 1, 1, 1}, []int64{3, 2}, torch.CPUDevice(), false))
	gradients, err := torch.Grad([]*torch.Tensor{selected}, []*torch.Tensor{x}, []*torch.Tensor{seed}, false, false)
	if err != nil {
		t.Fatal(err)
	}
	near(t, values(t, keep(gradients[0], nil)), []float64{1, 1, 0, 0, 2, 2}, 0)
	badIndices := keep(torch.FromFloat32([]float32{1}, []int64{1}, torch.CPUDevice(), false))
	if selected, err := x.IndexSelect(0, badIndices); err == nil {
		selected.Close()
		t.Fatal("floating point indices accepted")
	}
	binary.LittleEndian.PutUint64(indexBytes[:8], 3)
	outOfRange := keep(torch.FromBytes(indexBytes[:8], []int64{1}, torch.Int64, torch.CPUDevice(), false))
	if selected, err := x.IndexSelect(0, outOfRange); err == nil {
		selected.Close()
		t.Fatal("out-of-range index accepted")
	}
}
