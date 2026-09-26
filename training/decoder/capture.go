package decoder

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"

	"github.com/tayi-ai/arandu-llama/training/torch"
)

// DecoderOutputCaptureVersion identifies complete decoder outputs before final
// normalization, copied without pooling or changing the forward's precision.
const DecoderOutputCaptureVersion = "arandu-llama.decoder-output.v1"

// ErrDecoderOutputCapture identifies invalid capture geometry, bounds or values.
var ErrDecoderOutputCapture = errors.New("decoder: output capture rejected")

// DecoderOutputPosition selects a zero-based decoder and absolute input-token
// position. The output at position p has consumed token p, not token p+1.
type DecoderOutputPosition struct {
	Layer    int
	Position int64
}

// DecoderOutputLimits bounds copied output separately from Forward's Limits.
// MaxRows bounds row metadata; MaxOutputBytes bounds returned float64 payload.
// MaxCopyBytes bounds each raw native-to-host copy and cannot exceed
// ObservationChunkBytes. Native staging and the Go copy are separate buffers.
// These bounds exclude model weights, forward activations and allocator overhead.
type DecoderOutputLimits struct {
	MaxRows        int
	MaxOutputBytes int64
	MaxCopyBytes   int64
}

// DecoderOutputRow owns one finite hidden vector. Values exactly represent the
// source Float16 or Float32 elements as float64, ready for projection fitting.
type DecoderOutputRow struct {
	Layer    int
	Position int64
	Values   []float64
}

// DecoderOutputCapture owns only Go values, in requested layer/position order.
// Tensor is decoder_output; SourceDType is the forward's storage precision.
// The caller binds model, adapter, tokens and dataset identities independently;
// neither Version nor a successful capture qualifies those identities.
type DecoderOutputCapture struct {
	Version     string
	Tensor      string
	SourceDType torch.DType
	Width       int64
	Rows        []DecoderOutputRow
}

// CaptureDecoderOutputs runs the unchanged full Forward and copies selected
// decoder outputs before final normalization. Positions must be strictly ordered
// by layer, then token; duplicates are refused. All rows still participate in the
// forward. Selection changes only host copies, never decoder computation.
//
// Capture bounds are checked before Forward or output allocation. Its snapshot
// and temporary tensor views are closed on success, failure and cancellation;
// failures return no partial capture. Borrowed weights, rotary tables, tokens and
// positions must remain unchanged during the call. The caller owns placement,
// rotary preparation and serialization, as with Forward.
func (m *TextModel) CaptureDecoderOutputs(ctx context.Context, tokenIDs []int64, limits Limits, positions []DecoderOutputPosition, bounds DecoderOutputLimits) (DecoderOutputCapture, error) {
	info, err := m.validate(ctx, tokenIDs, limits)
	if err != nil {
		return DecoderOutputCapture{}, err
	}
	if err := validateDecoderOutputPositions(len(m.Layers), int64(len(tokenIDs)), info.Shape[1], info.DType, positions, bounds); err != nil {
		return DecoderOutputCapture{}, err
	}
	snapshot, err := m.Forward(ctx, tokenIDs, limits)
	if err != nil {
		return DecoderOutputCapture{}, err
	}
	return captureDecoderOutputs(ctx, snapshot, int64(len(tokenIDs)), info.Shape[1], info.DType, positions, bounds)
}

func validateDecoderOutputPositions(layers int, tokens, width int64, dtype torch.DType, positions []DecoderOutputPosition, bounds DecoderOutputLimits) error {
	storageBytes := int64(2)
	if dtype == torch.Float32 {
		storageBytes = 4
	} else if dtype != torch.Float16 {
		return fmt.Errorf("%w: unsupported source dtype", ErrDecoderOutputCapture)
	}
	if layers <= 0 || tokens <= 0 || width <= 0 || bounds.MaxRows <= 0 || len(positions) == 0 || len(positions) > bounds.MaxRows ||
		bounds.MaxOutputBytes <= 0 || bounds.MaxCopyBytes < storageBytes || bounds.MaxCopyBytes > ObservationChunkBytes ||
		width > bounds.MaxOutputBytes/8/int64(len(positions)) || width > int64(int(^uint(0)>>1))/8 {
		return fmt.Errorf("%w: invalid geometry or output budget", ErrDecoderOutputCapture)
	}
	for index, position := range positions {
		if position.Layer < 0 || position.Layer >= layers || position.Position < 0 || position.Position >= tokens {
			return fmt.Errorf("%w: layer or token position outside model", ErrDecoderOutputCapture)
		}
		if index > 0 {
			previous := positions[index-1]
			if position.Layer < previous.Layer || position.Layer == previous.Layer && position.Position <= previous.Position {
				return fmt.Errorf("%w: positions must strictly increase by layer and token", ErrDecoderOutputCapture)
			}
		}
	}
	return nil
}

// captureDecoderOutputs takes ownership of the complete forward snapshot.
func captureDecoderOutputs(ctx context.Context, snapshot *Snapshot, tokens, width int64, dtype torch.DType, positions []DecoderOutputPosition, bounds DecoderOutputLimits) (result DecoderOutputCapture, err error) {
	defer func() {
		err = errors.Join(err, snapshot.Close())
		if err != nil {
			result = DecoderOutputCapture{}
		}
	}()
	// Check every selected boundary before allocating any returned vectors.
	for _, position := range positions {
		if err := ctx.Err(); err != nil {
			return DecoderOutputCapture{}, err
		}
		info, err := snapshot.states[position.Layer+1].Info()
		if err != nil || info.RequiresGrad || info.DType != dtype || !sameShape(info.Shape, []int64{1, tokens, width}) {
			return DecoderOutputCapture{}, errors.Join(fmt.Errorf("%w: decoder output metadata differs", ErrDecoderOutputCapture), err)
		}
	}
	result = DecoderOutputCapture{Version: DecoderOutputCaptureVersion, Tensor: "decoder_output", SourceDType: dtype, Width: width,
		Rows: make([]DecoderOutputRow, 0, len(positions))}
	for _, position := range positions {
		values, err := copyDecoderOutputRow(ctx, snapshot.states[position.Layer+1], position.Position, width, dtype, bounds.MaxCopyBytes)
		if err != nil {
			return DecoderOutputCapture{}, err
		}
		result.Rows = append(result.Rows, DecoderOutputRow{Layer: position.Layer, Position: position.Position, Values: values})
	}
	return result, ctx.Err()
}

func copyDecoderOutputRow(ctx context.Context, state *torch.Tensor, position, width int64, dtype torch.DType, maxCopyBytes int64) (_ []float64, err error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	batch, err := state.Select(0, 0)
	if err != nil {
		return nil, err
	}
	defer func() { err = errors.Join(err, batch.Close()) }()
	row, err := batch.Select(0, position)
	if err != nil {
		return nil, err
	}
	defer func() { err = errors.Join(err, row.Close()) }()
	storageBytes := int64(2)
	if dtype == torch.Float32 {
		storageBytes = 4
	}
	values := make([]float64, int(width))
	for start := int64(0); start < width; {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		end := min(width, start+maxCopyBytes/storageBytes)
		part, err := row.Slice(0, start, end, 1)
		if err != nil {
			return nil, err
		}
		body, copyErr := part.Bytes()
		if err := errors.Join(copyErr, part.Close()); err != nil {
			return nil, err
		}
		if int64(len(body)) != (end-start)*storageBytes {
			return nil, fmt.Errorf("%w: native copy size differs", ErrDecoderOutputCapture)
		}
		for index := start; index < end; index++ {
			offset := (index - start) * storageBytes
			var value float64
			if dtype == torch.Float32 {
				value = float64(math.Float32frombits(binary.LittleEndian.Uint32(body[offset:])))
			} else {
				value = observationHalf(binary.LittleEndian.Uint16(body[offset:]))
			}
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return nil, fmt.Errorf("%w: nonfinite decoder output", ErrDecoderOutputCapture)
			}
			values[index] = value
		}
		start = end
	}
	return values, ctx.Err()
}
