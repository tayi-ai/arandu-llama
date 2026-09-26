package local_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/tayi-ai/arandu-llama/training/local"
	"github.com/tayi-ai/arandu-llama/training/pipeline"
)

func TestUninitializedStageRefusesWithoutWaitingOnNilGate(t *testing.T) {
	for _, stage := range []*local.Stage{nil, {}} {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		commit := func(context.Context, pipeline.StageResult) error {
			t.Fatal("uninitialized stage committed")
			return nil
		}
		if err := stage.Run(ctx, pipeline.StageContext{}, commit); !errors.Is(err, local.ErrStage) {
			t.Fatalf("Run did not refuse wiring immediately: %v", err)
		}
		if err := stage.Reconcile(ctx, pipeline.StageContext{}, commit); !errors.Is(err, local.ErrStage) {
			t.Fatalf("Reconcile did not refuse wiring immediately: %v", err)
		}
		cancel()
	}
}
