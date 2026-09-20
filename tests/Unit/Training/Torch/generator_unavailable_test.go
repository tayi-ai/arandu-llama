//go:build !libtorch || !cgo

package torch_test

import (
	"errors"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/torch"
)

func TestPrivateCPUGeneratorRequiresNativeBackend(t *testing.T) {
	if generator, err := torch.NewCPUGenerator(83); generator != nil || !errors.Is(err, torch.ErrUnavailable) {
		t.Fatalf("unexpected unavailable generator: %v %v", generator, err)
	}
	var generator torch.Generator
	if value, err := generator.Uniform([]int64{1}, 0, 1, torch.Float32); value != nil || !errors.Is(err, torch.ErrUnavailable) {
		t.Fatalf("unexpected unavailable draw: %v %v", value, err)
	}
	if err := generator.Close(); err != nil {
		t.Fatal(err)
	}
}
