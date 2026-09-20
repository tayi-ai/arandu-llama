//go:build libtorch && cgo

package torch_test

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"fmt"
	"math"
	"sync"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/torch"
)

func privateGenerator(t *testing.T, seed uint64) *torch.Generator {
	t.Helper()
	generator, err := torch.NewCPUGenerator(seed)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = generator.Close() })
	return generator
}

func uniformBytes(t *testing.T, generator *torch.Generator, shape []int64, low, high float64, dtype torch.DType) []byte {
	t.Helper()
	tensor, err := generator.Uniform(shape, low, high, dtype)
	if err != nil {
		t.Fatal(err)
	}
	defer tensor.Close()
	info, err := tensor.Info()
	if err != nil || info.Device != torch.CPUDevice() || info.DType != dtype || info.RequiresGrad {
		t.Fatalf("uniform metadata: %+v %v", info, err)
	}
	data, err := tensor.Bytes()
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func TestNativePrivateCPUGeneratorReproducibilityAndIndependentStreams(t *testing.T) {
	first := privateGenerator(t, 83)
	replay := privateGenerator(t, 83)
	other := privateGenerator(t, 84)
	shape := []int64{4, 4096}
	firstA := uniformBytes(t, first, shape, -1.0/64, 1.0/64, torch.Float32)
	otherA := uniformBytes(t, other, shape, -1.0/64, 1.0/64, torch.Float32)
	replayA := uniformBytes(t, replay, shape, -1.0/64, 1.0/64, torch.Float32)
	if !bytes.Equal(firstA, replayA) || bytes.Equal(firstA, otherA) {
		t.Fatal("private seeds do not create independent reproducible streams")
	}
	firstB := uniformBytes(t, first, shape, -1.0/64, 1.0/64, torch.Float32)
	_ = uniformBytes(t, other, []int64{31}, -7, 3, torch.Float64)
	replayB := uniformBytes(t, replay, shape, -1.0/64, 1.0/64, torch.Float32)
	if bytes.Equal(firstA, firstB) || !bytes.Equal(firstB, replayB) {
		t.Fatal("interleaving another stream changed sequence advancement")
	}
	digest := fmt.Sprintf("%x", sha256.Sum256(firstA))
	version, err := torch.HeaderVersion()
	if err != nil {
		t.Fatal(err)
	}
	// This is a CPU regression fixture, not a CPU/CUDA or cross-version promise.
	const golden214 = "6a5a47d0a71e196837323de2041a31d990d64ffafb766c8501c759ce89c920aa"
	if version == "2.14.0" && digest != golden214 {
		t.Fatalf("CPU 2.14 uniform fixture changed: got %s, want %s", digest, golden214)
	}
	t.Logf("seed83_float32_4x4096_sha256=%s", digest)
}

func TestNativePrivateCPUGeneratorFloatingDTypesAndSeedRange(t *testing.T) {
	for _, dtype := range []torch.DType{torch.Float32, torch.Float64, torch.Float16, torch.BFloat16} {
		t.Run(fmt.Sprint(dtype), func(t *testing.T) {
			first := privateGenerator(t, math.MaxUint64)
			replay := privateGenerator(t, math.MaxUint64)
			value, err := first.Uniform([]int64{257}, -.125, .125, dtype)
			if err != nil {
				t.Fatal(err)
			}
			defer value.Close()
			data, err := value.Bytes()
			if err != nil || !bytes.Equal(data, uniformBytes(t, replay, []int64{257}, -.125, .125, dtype)) {
				t.Fatalf("dtype/full seed reproducibility: %v", err)
			}
			values, err := value.Float64Values()
			if err != nil {
				t.Fatal(err)
			}
			for _, v := range values {
				if math.IsNaN(v) || v < -.125 || v >= .125 {
					t.Fatalf("invalid uniform value: %g", v)
				}
			}
		})
	}
	first, replay := privateGenerator(t, 83), privateGenerator(t, 83)
	if data := uniformBytes(t, first, []int64{0, 4}, -1, 1, torch.Float32); len(data) != 0 {
		t.Fatal("empty tensor has data")
	}
	if !bytes.Equal(uniformBytes(t, first, nil, -1, 1, torch.Float32), uniformBytes(t, replay, nil, -1, 1, torch.Float32)) {
		t.Fatal("empty draw consumed random stream")
	}
	constant, err := first.Uniform([]int64{8}, .5, .5, torch.Float32)
	if err != nil {
		t.Fatal(err)
	}
	defer constant.Close()
	values, err := constant.Float32Values()
	if err != nil {
		t.Fatal(err)
	}
	for _, value := range values {
		if value != .5 {
			t.Fatal("equal bounds did not produce the constant")
		}
	}
}

func TestNativePrivateCPUGeneratorRejectsInvalidDrawsWithoutAdvancing(t *testing.T) {
	generator, replay := privateGenerator(t, 83), privateGenerator(t, 83)
	for _, invalid := range []struct {
		shape     []int64
		low, high float64
		dtype     torch.DType
	}{
		{[]int64{-1}, 0, 1, torch.Float32},
		{[]int64{math.MaxInt64}, 0, 1, torch.Float32},
		{[]int64{math.MaxInt64, 2}, 0, 1, torch.Float32},
		{make([]int64, 33), 0, 1, torch.Float32},
		{[]int64{1}, math.NaN(), 1, torch.Float32},
		{[]int64{1}, 0, math.Inf(1), torch.Float32},
		{[]int64{1}, 1, 0, torch.Float32},
		{[]int64{1}, 0, 1, torch.Int64},
		{[]int64{1}, 0, 1, torch.Bool},
		{[]int64{1}, 0, 1, torch.DType(99)},
		{[]int64{1}, -1e100, 1e100, torch.Float16},
	} {
		value, err := generator.Uniform(invalid.shape, invalid.low, invalid.high, invalid.dtype)
		if err == nil || value != nil {
			if value != nil {
				_ = value.Close()
			}
			t.Fatalf("invalid draw accepted: %+v", invalid)
		}
	}
	if !bytes.Equal(uniformBytes(t, generator, []int64{8}, -1, 1, torch.Float32), uniformBytes(t, replay, []int64{8}, -1, 1, torch.Float32)) {
		t.Fatal("rejected draw advanced the stream")
	}
}

func TestNativePrivateCPUGeneratorOwnershipAndConcurrentClose(t *testing.T) {
	generator := privateGenerator(t, 83)
	value, err := generator.Uniform([]int64{8}, -1, 1, torch.Float32)
	if err != nil {
		t.Fatal(err)
	}
	defer value.Close()
	copied := *generator
	var workers sync.WaitGroup
	for i := 0; i < 16; i++ {
		workers.Add(1)
		go func() {
			defer workers.Done()
			draw, err := copied.Uniform([]int64{16}, -1, 1, torch.Float32)
			if err == nil {
				_ = draw.Close()
			} else if !errors.Is(err, torch.ErrGeneratorClosed) {
				t.Errorf("concurrent draw: %v", err)
			}
		}()
	}
	if err := generator.Close(); err != nil {
		t.Fatal(err)
	}
	workers.Wait()
	if err := copied.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := copied.Uniform(nil, 0, 1, torch.Float32); !errors.Is(err, torch.ErrGeneratorClosed) {
		t.Fatalf("copied closed stream: %v", err)
	}
	if _, err := value.Bytes(); err != nil {
		t.Fatalf("closing generator invalidated tensor: %v", err)
	}
	var zero torch.Generator
	if _, err := zero.Uniform(nil, 0, 1, torch.Float32); !errors.Is(err, torch.ErrGeneratorClosed) {
		t.Fatalf("zero generator: %v", err)
	}
	if err := zero.Close(); err != nil {
		t.Fatal(err)
	}
}
