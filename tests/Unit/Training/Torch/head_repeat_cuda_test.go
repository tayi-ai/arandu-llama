//go:build libtorch && cgo

package torch_test

import (
	"encoding/binary"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/torch"
)

func TestCUDAIndexSelectPreservesRepeatedHeadValuesAndGradients(t *testing.T) {
	if !torch.CUDAEnabled() {
		t.Skip("CUDA bridge is not enabled")
	}
	count, err := torch.CUDADeviceCount()
	if err != nil {
		t.Fatal(err)
	}
	if count < 1 {
		t.Skip("one CUDA device is required")
	}

	keep := scope(t)
	device := torch.CUDADevice(0)
	input := keep(torch.FromFloat32(
		[]float32{1, 2, 3, 4},
		[]int64{1, 1, 2, 2},
		device,
		true,
	))

	indexBytes := make([]byte, 4*8)
	for i, index := range []uint64{0, 0, 1, 1} {
		binary.LittleEndian.PutUint64(indexBytes[i*8:], index)
	}
	indices := keep(torch.FromBytes(indexBytes, []int64{4}, torch.Int64, device, false))
	repeated := keep(input.IndexSelect(2, indices))
	if finite, err := repeated.AllFinite(); err != nil || !finite {
		t.Fatalf("repeated heads finite=%t err=%v", finite, err)
	}
	near(t, values(t, repeated), []float64{1, 2, 1, 2, 3, 4, 3, 4}, 0)

	seedValues := make([]float32, 8)
	for i := range seedValues {
		seedValues[i] = 1
	}
	seed := keep(torch.FromFloat32(seedValues, []int64{1, 1, 4, 2}, device, false))
	gradients, err := torch.Grad(
		[]*torch.Tensor{repeated},
		[]*torch.Tensor{input},
		[]*torch.Tensor{seed},
		false,
		false,
	)
	if err != nil {
		t.Fatal(err)
	}
	gradient := keep(gradients[0], nil)
	if finite, err := gradient.AllFinite(); err != nil || !finite {
		t.Fatalf("repeated head gradient finite=%t err=%v", finite, err)
	}
	near(t, values(t, gradient), []float64{2, 2, 2, 2}, 0)
}
