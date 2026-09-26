//go:build libtorch && cgo

package decoder

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/torch"
)

type captureCancelContext struct {
	context.Context
	cancel context.CancelFunc
	checks int
}

func (c *captureCancelContext) Err() error {
	c.checks--
	if c.checks == 0 {
		c.cancel()
	}
	return c.Context.Err()
}

func TestDecoderOutputCaptureClosesOwnedSnapshotOnEveryExit(t *testing.T) {
	for _, name := range []string{"success", "geometry", "nonfinite", "cancel-before-copy", "cancel-during-copy"} {
		t.Run(name, func(t *testing.T) {
			var owned []*torch.Tensor
			makeTensor := func(values []float32, shape []int64) *torch.Tensor {
				value, err := torch.FromFloat32(values, shape, torch.CPUDevice(), false)
				if err != nil {
					t.Fatal(err)
				}
				owned = append(owned, value)
				t.Cleanup(func() { _ = value.Close() })
				return value
			}
			values := []float32{1, 2, 3, 4}
			if name == "nonfinite" {
				values[3] = float32(math.Inf(1))
			}
			snapshot := &Snapshot{states: []*torch.Tensor{makeTensor([]float32{0, 0, 0, 0}, []int64{1, 1, 4}), makeTensor(values, []int64{1, 1, 4})},
				Logits: makeTensor([]float32{0, 1}, []int64{1, 1, 2})}
			var ctx context.Context = context.Background()
			var want error
			width := int64(4)
			switch name {
			case "geometry":
				width, want = 3, ErrDecoderOutputCapture
			case "nonfinite":
				want = ErrDecoderOutputCapture
			case "cancel-before-copy":
				cancelled, cancel := context.WithCancel(ctx)
				cancel()
				ctx, want = cancelled, context.Canceled
			case "cancel-during-copy":
				cancelled, cancel := context.WithCancel(ctx)
				defer cancel()
				ctx, want = &captureCancelContext{Context: cancelled, cancel: cancel, checks: 5}, context.Canceled
			}
			capture, err := captureDecoderOutputs(ctx, snapshot, 1, width, torch.Float32,
				[]DecoderOutputPosition{{0, 0}}, DecoderOutputLimits{1, 32, 4})
			if want == nil {
				if err != nil || len(capture.Rows) != 1 || !reflect.DeepEqual(capture.Rows[0].Values, []float64{1, 2, 3, 4}) {
					t.Fatalf("capture differs: %+v %v", capture, err)
				}
			} else if !errors.Is(err, want) || !reflect.DeepEqual(capture, DecoderOutputCapture{}) {
				t.Fatalf("partial capture escaped refusal: %+v %v", capture, err)
			}
			for _, value := range owned {
				if _, err := value.Info(); err == nil {
					t.Fatal("owned snapshot tensor remains live after capture")
				}
			}
		})
	}
}
