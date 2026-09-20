//go:build !libtorch || !cgo

package loading_test

import (
	"context"
	"errors"
	"testing"

	"github.com/tayi-ai/arandu-llama/checkpoint"
	"github.com/tayi-ai/arandu-llama/training/loading"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

type forbiddenReader struct{ called bool }

func (reader *forbiddenReader) ReadAt([]byte, int64) (int, error) {
	reader.called = true
	return 0, errors.New("unexpected read")
}

func TestUnavailableLoaderRefusesBeforeReading(t *testing.T) {
	reader := &forbiddenReader{}
	options := loading.Options{HeaderLimits: checkpoint.DefaultLimits(), BudgetBytes: 1024, DType: torch.Float32, Device: torch.CPUDevice()}
	tensor, receipt, err := loading.LoadTensor(context.Background(), reader, 1024, "weight", options)
	if !errors.Is(err, torch.ErrUnavailable) || tensor != nil || receipt.Loaded || reader.called {
		t.Fatalf("unavailable native path: %+v %v read%v", receipt, err, reader.called)
	}
}
