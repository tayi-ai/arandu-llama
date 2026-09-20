//go:build libtorch && cgo

package torch_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"runtime"
	"testing"
	"time"

	"github.com/tayi-ai/arandu-llama/training/torch"
)

type lifetimeCall struct {
	run   func() error
	reply chan error
}

// Both goroutines retain different OS threads for the whole test, including
// while awaiting work. Channels transfer ownership and establish happens-before
// edges; no tensor is intentionally used concurrently with its Close.
func lifetimeThread(ctx context.Context, calls <-chan lifetimeCall, ready chan<- struct{}, done chan<- struct{}) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	defer func() { done <- struct{}{} }()
	select {
	case ready <- struct{}{}:
	case <-ctx.Done():
		return
	}
	for {
		select {
		case <-ctx.Done():
			return
		case call := <-calls:
			// The buffered reply cannot strand a worker if its caller times out.
			if err := ctx.Err(); err != nil {
				call.reply <- err
				return
			}
			call.reply <- call.run()
		}
	}
}

func TestNativeHalfOwnershipAcrossLockedOSThreadsAndGC(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 15*time.Second)
	defer cancel()
	a, b := make(chan lifetimeCall), make(chan lifetimeCall)
	ready, done := make(chan struct{}, 2), make(chan struct{}, 2)
	joined := false
	go lifetimeThread(ctx, a, ready, done)
	go lifetimeThread(ctx, b, ready, done)
	// Cancel before waiting, including on any failed assertion. The deadline
	// bounds Go coordination; the package's go test timeout bounds native calls.
	defer func() {
		cancel()
		timer := time.NewTimer(time.Second)
		defer timer.Stop()
		for range 2 {
			select {
			case <-done:
			case <-timer.C:
				t.Error("locked native worker did not exit after cancellation")
				return
			}
		}
		joined = true
	}()
	for range 2 {
		select {
		case <-ready:
		case <-ctx.Done():
			t.Fatal("native workers did not become ready before deadline")
		}
	}
	run := func(thread chan lifetimeCall, operation func() error) {
		t.Helper()
		call := lifetimeCall{run: operation, reply: make(chan error, 1)}
		select {
		case thread <- call:
		case <-ctx.Done():
			t.Fatal("native worker admission exceeded deadline")
		}
		select {
		case err := <-call.reply:
			if err != nil {
				t.Fatal(err)
			}
		case <-ctx.Done():
			t.Fatal("native worker result exceeded deadline")
		}
	}
	var owned []*torch.Tensor
	// Every native operation finishes before this test-owned cleanup executes.
	// On a native hang the test timeout is the ultimate bound, not a fake success.
	t.Cleanup(func() {
		// A timed-out native call may still own or append handles. Do not race
		// it or block cleanup on the native-call mutex after the bounded join.
		if !joined {
			return
		}
		for _, value := range owned {
			if err := value.Close(); err != nil {
				t.Error(err)
			}
		}
	})
	keep := func(value *torch.Tensor, err error) (*torch.Tensor, error) {
		if value != nil {
			owned = append(owned, value)
		}
		return value, err
	}
	check := func(value *torch.Tensor, expected []float32) error {
		actual, err := value.Float32Values()
		if err != nil {
			return err
		}
		if len(actual) != len(expected) {
			return fmt.Errorf("native copy length: got %d want %d", len(actual), len(expected))
		}
		for i := range actual {
			if math.Float32bits(actual[i]) != math.Float32bits(expected[i]) {
				return fmt.Errorf("native value %d: got %g want exactly %g", i, actual[i], expected[i])
			}
		}
		return nil
	}
	for iteration := range 24 {
		// Independent IEEE binary16 fixture: [[1,-2],[0.5,3]]. Alternating
		// signs prevents accidentally reusing a previous iteration's result.
		bits := []uint16{0x3c00, 0xc000, 0x3800, 0x4200}
		sign := float32(1)
		if iteration%2 != 0 {
			sign = -1
			for i := range bits {
				bits[i] ^= 0x8000
			}
		}
		input := make([]byte, len(bits)*2)
		for i, value := range bits {
			binary.LittleEndian.PutUint16(input[2*i:], value)
		}
		original := bytes.Clone(input)
		var parent, view, copied, cloned, converted, halfProduct, floatProduct *torch.Tensor
		var exported []byte
		run(a, func() error {
			var err error
			parent, err = keep(torch.FromBytes(input, []int64{2, 2}, torch.Float16, torch.CPUDevice(), false))
			if err != nil {
				return err
			}
			exported, err = parent.Bytes()
			if err != nil {
				return err
			}
			if !bytes.Equal(exported, original) {
				return fmt.Errorf("FP16 source bytes changed on creation")
			}
			for i := range input {
				input[i] = 0xff
			}
			view, err = keep(parent.Transpose(0, 1))
			if err != nil {
				return err
			}
			if err = parent.Close(); err != nil {
				return err
			}
			runtime.GC()
			return nil
		})
		run(b, func() error {
			if err := check(view, []float32{sign, .5 * sign, -2 * sign, 3 * sign}); err != nil {
				return err
			}
			var err error
			copied, err = keep(view.To(torch.CPUDevice(), torch.Float16))
			if err != nil {
				return err
			}
			cloned, err = keep(view.Clone())
			if err != nil {
				return err
			}
			converted, err = keep(view.To(torch.CPUDevice(), torch.Float32))
			if err != nil {
				return err
			}
			if err = view.Close(); err != nil {
				return err
			}
			runtime.GC()
			// Plain scalar oracle below covers X^T * [[2,-1],[1,4]]. All
			// values and results are exactly representable in binary16/32.
			rhs, err := keep(torch.FromFloat32([]float32{2, -1, 1, 4}, []int64{2, 2}, torch.CPUDevice(), false))
			if err != nil {
				return err
			}
			halfRHS, err := keep(rhs.To(torch.CPUDevice(), torch.Float16))
			if err != nil {
				return err
			}
			halfProduct, err = keep(cloned.MatMul(halfRHS))
			if err != nil {
				return err
			}
			floatProduct, err = keep(converted.MatMul(rhs))
			if err != nil {
				return err
			}
			for _, value := range []*torch.Tensor{cloned, converted, halfRHS, rhs} {
				if err := value.Close(); err != nil {
					return err
				}
			}
			runtime.GC()
			return nil
		})
		run(a, func() error {
			left, right := []float32{sign, .5 * sign, -2 * sign, 3 * sign}, []float32{2, -1, 1, 4}
			want := make([]float32, 4)
			for row := range 2 {
				for column := range 2 {
					for k := range 2 {
						want[row*2+column] += left[row*2+k] * right[k*2+column]
					}
				}
			}
			if err := check(halfProduct, want); err != nil {
				return err
			}
			if err := check(floatProduct, want); err != nil {
				return err
			}
			if err := check(copied, left); err != nil {
				return err
			}
			// Bytes owns its Go allocation even after its native parent died.
			if !bytes.Equal(exported, original) {
				return fmt.Errorf("Go-owned exported bytes changed after parent Close/GC")
			}
			rebuilt, err := keep(torch.FromBytes(exported, []int64{2, 2}, torch.Float16, torch.CPUDevice(), false))
			if err != nil {
				return err
			}
			clear(exported)
			runtime.GC()
			if err := check(rebuilt, []float32{sign, -2 * sign, .5 * sign, 3 * sign}); err != nil {
				return err
			}
			for _, value := range []*torch.Tensor{halfProduct, floatProduct, copied, rebuilt} {
				if err := value.Close(); err != nil {
					return err
				}
			}
			return nil
		})
	}
	// Normal completion also exercises cancellation of both workers while they
	// are blocked waiting for a channel request, via the deferred bounded join.
	t.Log("24 exact FP16/FP32 transfers between two locked OS threads passed, with parent Close and GC")
}
