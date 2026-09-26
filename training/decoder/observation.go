package decoder

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/tayi-ai/arandu-llama/training/torch"
)

// ObservationChunkBytes limits each raw native-to-host copy to four MiB.
// Native contiguous-copy storage and the Go byte buffer are separate allocations.
const ObservationChunkBytes = 4 << 20

// ForwardStage identifies the activation observed in a model forward.
type ForwardStage string

const (
	// StageEmbedding is the token embedding after reshaping to [batch,tokens,hidden].
	StageEmbedding ForwardStage = "embedding"
	// StageBeforePlacement is the residual immediately before a decoder's device copy.
	StageBeforePlacement ForwardStage = "before_placement"
	// StageAfterPlacement is the copied residual on that decoder's device.
	StageAfterPlacement ForwardStage = "after_placement"
	// StageAfterDecoder is the detached output of a complete decoder block.
	StageAfterDecoder ForwardStage = "after_decoder"
	// StageFinalNorm includes the learned scale, before selecting rows for the head.
	StageFinalNorm ForwardStage = "final_norm"
	// StageLogits is the retained vocabulary rows in Float32, before softmax.
	StageLogits ForwardStage = "logits"
)

// ForwardObservation owns copied metadata and statistics, never native handles.
// SHA256 covers raw little-endian values in logical row-major order, without
// dtype conversion or metadata. NonzeroElements counts finite nonzero values;
// nonfinite values are counted separately and excluded from Minimum/Maximum.
// Both extrema are nil when there is no finite value. Layer is zero-based, or
// -1 for embedding, final normalization and logits. ChunkBytes is the largest
// copy actually performed. An observation is not an independent model receipt.
type ForwardObservation struct {
	Stage             ForwardStage `json:"stage"`
	Layer             int          `json:"layer"`
	Shape             []int64      `json:"shape"`
	DType             torch.DType  `json:"dtype"`
	Device            torch.Device `json:"device"`
	Elements          int64        `json:"elements"`
	Bytes             int64        `json:"bytes"`
	SHA256            string       `json:"sha256"`
	AllFinite         bool         `json:"all_finite"`
	NonzeroElements   int64        `json:"nonzero_elements"`
	NonfiniteElements int64        `json:"nonfinite_elements"`
	Minimum           *float64     `json:"minimum"`
	Maximum           *float64     `json:"maximum"`
	ChunkBytes        int          `json:"chunk_bytes"`
}

// ForwardObserver consumes one complete copied observation synchronously.
// It may retain or modify the value and must respect the supplied context.
// Returning an error stops the forward; no later observations are delivered.
// It must not call back into or mutate the model whose forward is in progress.
type ForwardObserver func(context.Context, ForwardObservation) error

func observeForward(ctx context.Context, observer ForwardObserver, stage ForwardStage, layer int, value *torch.Tensor) error {
	if observer == nil {
		return nil
	}
	observation, err := scanForward(ctx, stage, layer, value)
	if err != nil {
		return fmt.Errorf("decoder: observe %s layer %d: %w", stage, layer, err)
	}
	if err = observer(ctx, observation); err != nil {
		return fmt.Errorf("decoder: observer %s layer %d: %w", stage, layer, err)
	}
	return ctx.Err()
}

func scanForward(ctx context.Context, stage ForwardStage, layer int, value *torch.Tensor) (ForwardObservation, error) {
	var result ForwardObservation
	if ctx == nil || value == nil {
		return result, errors.New("observation requires context and tensor")
	}
	if err := ctx.Err(); err != nil {
		return result, err
	}
	info, err := value.Info()
	if err != nil {
		return result, err
	}
	width := int64(2)
	if info.DType == torch.Float32 {
		width = 4
	} else if info.DType != torch.Float16 {
		return result, errors.New("observation requires Float16 or Float32")
	}
	if info.Elements < 0 || info.Elements > math.MaxInt64/width {
		return result, errors.New("observation byte count overflows")
	}
	result = ForwardObservation{Stage: stage, Layer: layer, Shape: slices.Clone(info.Shape), DType: info.DType,
		Device: info.Device, Elements: info.Elements, Bytes: info.Elements * width, AllFinite: true}
	digest := sha256.New()
	var readBytes int64
	var visit func(*torch.Tensor, []int64, int64) error
	visit = func(part *torch.Tensor, shape []int64, elements int64) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if elements*width <= ObservationChunkBytes {
			body, err := part.Bytes()
			if err != nil {
				return err
			}
			if int64(len(body)) != elements*width {
				return errors.New("observation native byte count differs")
			}
			_, _ = digest.Write(body)
			readBytes += int64(len(body))
			result.ChunkBytes = max(result.ChunkBytes, len(body))
			for offset := 0; offset < len(body); offset += int(width) {
				if offset%(64<<10) == 0 {
					if err := ctx.Err(); err != nil {
						return err
					}
				}
				var number float64
				if width == 4 {
					number = float64(math.Float32frombits(binary.LittleEndian.Uint32(body[offset:])))
				} else {
					number = observationHalf(binary.LittleEndian.Uint16(body[offset:]))
				}
				if math.IsNaN(number) || math.IsInf(number, 0) {
					result.AllFinite = false
					result.NonfiniteElements++
					continue
				}
				if number != 0 {
					result.NonzeroElements++
				}
				if result.Minimum == nil {
					minimum, maximum := number, number
					result.Minimum, result.Maximum = &minimum, &maximum
				} else {
					*result.Minimum = math.Min(*result.Minimum, number)
					*result.Maximum = math.Max(*result.Maximum, number)
				}
			}
			return ctx.Err()
		}
		if len(shape) == 0 || shape[0] <= 0 || elements%shape[0] != 0 {
			return errors.New("observation tensor geometry is invalid")
		}
		rowElements := elements / shape[0]
		if rowElements*width > ObservationChunkBytes {
			for index := int64(0); index < shape[0]; index++ {
				view, err := part.Select(0, index)
				if err != nil {
					return err
				}
				if err = errors.Join(visit(view, shape[1:], rowElements), view.Close()); err != nil {
					return err
				}
			}
			return ctx.Err()
		}
		rowsPerChunk := int64(ObservationChunkBytes) / (rowElements * width)
		for start := int64(0); start < shape[0]; {
			end := min(shape[0], start+rowsPerChunk)
			view, err := part.Slice(0, start, end, 1)
			if err != nil {
				return err
			}
			if err = errors.Join(visit(view, nil, (end-start)*rowElements), view.Close()); err != nil {
				return err
			}
			start = end
		}
		return ctx.Err()
	}
	if err = visit(value, info.Shape, info.Elements); err != nil {
		return ForwardObservation{}, err
	}
	if readBytes != result.Bytes {
		return ForwardObservation{}, errors.New("observation did not cover every tensor byte")
	}
	result.SHA256 = hex.EncodeToString(digest.Sum(nil))
	return result, ctx.Err()
}

func observationHalf(bits uint16) float64 {
	exponent, fraction := (bits>>10)&31, bits&1023
	var value float64
	switch exponent {
	case 0:
		value = math.Ldexp(float64(fraction), -24)
	case 31:
		value = math.Inf(1)
		if fraction != 0 {
			value = math.NaN()
		}
	default:
		value = math.Ldexp(float64(1024+fraction), int(exponent)-25)
	}
	if bits&0x8000 != 0 {
		value = -value
	}
	return value
}
