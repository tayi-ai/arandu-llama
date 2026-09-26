//go:build libtorch && cgo

package decoder_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/decoder"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

func capturePositions(layers ...int) []decoder.DecoderOutputPosition {
	var result []decoder.DecoderOutputPosition
	for _, layer := range layers {
		for position := range int64(3) {
			result = append(result, decoder.DecoderOutputPosition{Layer: layer, Position: position})
		}
	}
	return result
}

func TestDecoderOutputCaptureMatchesExactForwardRowsAndPreservesModel(t *testing.T) {
	for _, dtype := range []torch.DType{torch.Float16, torch.Float32} {
		t.Run(map[torch.DType]string{torch.Float16: "fp16", torch.Float32: "fp32"}[dtype], func(t *testing.T) {
			f := newFixture(t, dtype)
			base := baseHash(t, f)
			hashes := map[int]string{}
			snapshot, err := f.model.ForwardObserved(context.Background(), f.tokens, f.limits, func(_ context.Context, item decoder.ForwardObservation) error {
				if item.Stage == decoder.StageAfterDecoder {
					hashes[item.Layer] = item.SHA256
				}
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
			wantLogits := read(t, snapshot.Logits)
			if err := snapshot.Close(); err != nil {
				t.Fatal(err)
			}
			positions := capturePositions(0, 3, 31)
			bounds := decoder.DecoderOutputLimits{MaxRows: len(positions), MaxOutputBytes: int64(len(positions)) * 4 * 8, MaxCopyBytes: 4}
			capture, err := f.model.CaptureDecoderOutputs(context.Background(), f.tokens, f.limits, positions, bounds)
			if err != nil || capture.Version != decoder.DecoderOutputCaptureVersion || capture.Tensor != "decoder_output" || capture.SourceDType != dtype || capture.Width != 4 || len(capture.Rows) != len(positions) {
				t.Fatalf("capture contract differs: %+v %v", capture, err)
			}
			for offset := 0; offset < len(positions); offset += 3 {
				var values []float32
				for index := offset; index < offset+3; index++ {
					row := capture.Rows[index]
					if row.Layer != positions[index].Layer || row.Position != positions[index].Position {
						t.Fatal("captured position order changed")
					}
					for _, value := range row.Values {
						values = append(values, float32(value))
					}
				}
				value := tensor(t, values, []int64{1, 3, 4}, false)
				converted, err := value.To(torch.CPUDevice(), dtype)
				converted = own(t, converted, err)
				body, err := converted.Bytes()
				if err != nil {
					t.Fatal(err)
				}
				digest := sha256.Sum256(body)
				if hex.EncodeToString(digest[:]) != hashes[positions[offset].Layer] {
					t.Fatal("captured vectors differ from exact decoder output bytes")
				}
			}
			again, err := f.model.CaptureDecoderOutputs(context.Background(), f.tokens, f.limits, positions, bounds)
			if err != nil || !reflect.DeepEqual(again, capture) {
				t.Fatalf("repeated capture differs: %v", err)
			}
			capture.Rows[0].Values[0] = 1000
			if baseHash(t, f) != base || !reflect.DeepEqual(read(t, forward(t, f).Logits), wantLogits) || again.Rows[0].Values[0] == 1000 {
				t.Fatal("capture retained model storage or changed forward values")
			}
		})
	}
}

func TestDecoderOutputCapturePreservesGoldPrefixCausality(t *testing.T) {
	f := newFixture(t, torch.Float32)
	positions := []decoder.DecoderOutputPosition{{Layer: 3, Position: 0}, {Layer: 3, Position: 1}, {Layer: 31, Position: 1}}
	bounds := decoder.DecoderOutputLimits{MaxRows: 3, MaxOutputBytes: 96, MaxCopyBytes: 8}
	full, err := f.model.CaptureDecoderOutputs(context.Background(), f.tokens, f.limits, positions, bounds)
	if err != nil {
		t.Fatal(err)
	}
	changed, err := f.model.CaptureDecoderOutputs(context.Background(), []int64{1, 4, 5}, f.limits, positions, bounds)
	if err != nil {
		t.Fatal(err)
	}
	for index := range f.model.Layers {
		layer := &f.model.Layers[index]
		if layer.Cosine != nil {
			cosine, err := layer.Cosine.Slice(0, 0, 2, 1)
			layer.Cosine = own(t, cosine, err)
			sine, err := layer.Sine.Slice(0, 0, 2, 1)
			layer.Sine = own(t, sine, err)
		}
	}
	prefix, err := f.model.CaptureDecoderOutputs(context.Background(), f.tokens[:2], f.limits, positions, bounds)
	if err != nil {
		t.Fatal(err)
	}
	// Full-row and shorter-prefix native kernels may round differently. The CPU
	// fixture's absolute tolerance is fixed at 1e-6 before observing its values.
	for row := range full.Rows {
		for column, want := range full.Rows[row].Values {
			if math.Abs(changed.Rows[row].Values[column]-want) > 1e-6 || math.Abs(prefix.Rows[row].Values[column]-want) > 1e-6 {
				t.Fatalf("future suffix changed causal feature at row %d column %d", row, column)
			}
		}
	}
}

func TestDecoderOutputCaptureRefusesBeforeForward(t *testing.T) {
	f := newFixture(t, torch.Float32)
	// A decoder would fail if reached. Capture admission must reject first.
	f.model.Layers[0].Weights.InputNorm = nil
	positions := []decoder.DecoderOutputPosition{{Layer: 0, Position: 1}}
	for _, bounds := range []decoder.DecoderOutputLimits{{1, 31, 4}, {1, 32, 3}} {
		capture, err := f.model.CaptureDecoderOutputs(context.Background(), f.tokens, f.limits, positions, bounds)
		if !errors.Is(err, decoder.ErrDecoderOutputCapture) || !reflect.DeepEqual(capture, decoder.DecoderOutputCapture{}) {
			t.Fatalf("invalid bounds reached forward: %+v %v", capture, err)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := f.model.CaptureDecoderOutputs(ctx, f.tokens, f.limits, positions, decoder.DecoderOutputLimits{1, 32, 4}); !errors.Is(err, context.Canceled) {
		t.Fatalf("pre-canceled capture: %v", err)
	}
}
