//go:build !libtorch || !cgo

package torch_test

import (
	"errors"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/torch"
)

func TestUnavailableBuildRefusesNativeExecution(t *testing.T) {
	if torch.Enabled() {
		t.Fatal("unlinked bridge advertised availability")
	}
	if _, err := torch.Version(); !errors.Is(err, torch.ErrUnavailable) {
		t.Fatalf("version: %v", err)
	}
	if _, err := torch.FromFloat64([]float64{1}, []int64{1}, torch.CPUDevice(), true); !errors.Is(err, torch.ErrUnavailable) {
		t.Fatalf("creation: %v", err)
	}
	if _, err := torch.Grad(nil, nil, nil, false, false); !errors.Is(err, torch.ErrUnavailable) {
		t.Fatalf("grad: %v", err)
	}
	var tensor torch.Tensor
	if _, err := tensor.Info(); !errors.Is(err, torch.ErrClosed) {
		t.Fatalf("metadata: %v", err)
	}
	if err := tensor.Close(); err != nil {
		t.Fatal(err)
	}
}
