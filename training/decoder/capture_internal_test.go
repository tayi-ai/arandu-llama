package decoder

import (
	"errors"
	"math"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/torch"
)

func TestDecoderOutputAdmissionRejectsUnboundedOrAmbiguousRequests(t *testing.T) {
	positions := []DecoderOutputPosition{{0, 0}, {0, 2}, {1, 1}}
	bounds := DecoderOutputLimits{MaxRows: 3, MaxOutputBytes: 96, MaxCopyBytes: 4}
	if err := validateDecoderOutputPositions(2, 3, 4, torch.Float32, positions, bounds); err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name      string
		positions []DecoderOutputPosition
		bounds    DecoderOutputLimits
		width     int64
		dtype     torch.DType
	}{
		{"empty", nil, bounds, 4, torch.Float32},
		{"duplicate", []DecoderOutputPosition{{0, 1}, {0, 1}}, bounds, 4, torch.Float32},
		{"unordered-layer", []DecoderOutputPosition{{1, 1}, {0, 2}}, bounds, 4, torch.Float32},
		{"unordered-position", []DecoderOutputPosition{{0, 2}, {0, 1}}, bounds, 4, torch.Float32},
		{"negative-layer", []DecoderOutputPosition{{-1, 0}}, bounds, 4, torch.Float32},
		{"unknown-layer", []DecoderOutputPosition{{2, 0}}, bounds, 4, torch.Float32},
		{"negative-position", []DecoderOutputPosition{{0, -1}}, bounds, 4, torch.Float32},
		{"unknown-position", []DecoderOutputPosition{{0, 3}}, bounds, 4, torch.Float32},
		{"row-budget", positions, DecoderOutputLimits{2, 96, 4}, 4, torch.Float32},
		{"payload-budget", positions, DecoderOutputLimits{3, 95, 4}, 4, torch.Float32},
		{"copy-budget", positions, DecoderOutputLimits{3, 96, 3}, 4, torch.Float32},
		{"copy-cap", positions, DecoderOutputLimits{3, 96, ObservationChunkBytes + 1}, 4, torch.Float32},
		{"overflow", positions, DecoderOutputLimits{3, math.MaxInt64, 4}, math.MaxInt64, torch.Float32},
		{"zero-width", positions, bounds, 0, torch.Float32},
		{"integer-source", positions, bounds, 4, torch.Int64},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if err := validateDecoderOutputPositions(2, 3, test.width, test.dtype, test.positions, test.bounds); !errors.Is(err, ErrDecoderOutputCapture) {
				t.Fatalf("invalid admission returned %v", err)
			}
		})
	}
	if err := validateDecoderOutputPositions(2, 3, 4, torch.Float16, positions, DecoderOutputLimits{3, 96, 2}); err != nil {
		t.Fatalf("one half element per copy refused: %v", err)
	}
}
