//go:build !libtorch || !cgo

package step_test

import (
	"context"
	"errors"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/ornith"
)

func TestStepRejectsInvalidRequestWithoutNativeBackend(t *testing.T) {
	limits := ornith.Limits{MaxTokens: 2, LogitRows: 1, MaxCheckpointBytes: 1}
	if values, err := ornith.ReadCandidateLogits(context.Background(), nil, []int64{4, 0}, [4]int64{0, 1, 2, 3}, limits); !errors.Is(err, ornith.ErrCandidateStep) || values != [4]float64{} {
		t.Fatalf("invalid scoring accepted:%v", err)
	}
	if result, err := ornith.CandidateGradient(context.Background(), nil, nil, [4]int64{}, limits, nil); !errors.Is(err, ornith.ErrCandidateStep) || len(result.Gradients) != 0 {
		t.Fatalf("invalid derivative accepted:%v", err)
	}
}
