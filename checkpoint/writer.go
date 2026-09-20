package checkpoint

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"sort"
	"strconv"
	"unicode/utf8"
)

// Float32Tensor describes caller-owned, row-major values. An empty Shape is a
// scalar; a zero dimension describes an empty tensor. Names are preserved.
type Float32Tensor struct {
	Name   string
	Shape  []uint64
	Values []float32
}

// WriteReceipt describes one complete safetensors stream accepted by a writer.
// SHA256 covers the prefix, padded header and payload. HeaderBytes excludes the
// eight-byte prefix. On error only Bytes, the accepted byte count, is populated.
// A receipt does not establish that an underlying file was synced or published.
type WriteReceipt struct {
	SHA256       string
	Bytes        int64
	HeaderBytes  int64
	PayloadBytes int64
	Tensors      int
}

// WriteFloat32 writes a deterministic F32 safetensors container, sorting tensors
// by name and including metadata format=pt. It validates all names, dimensions,
// values and configured limits before the first write. Values must be finite;
// signed zero and all other finite bit patterns are preserved.
//
// The caller must keep tensors unchanged until return and owns destination,
// flushing, atomic publication and any read-back verification. Limits bounds
// the header and individual writes, not the caller's payload size. Payload
// serialization uses at most min(MaxChunkBytes, 64 KiB) scratch plus one scalar;
// the bounded header is assembled separately. No
// full payload copy is made. Cancellation is checked between validation chunks
// and writes, but cannot interrupt a destination already blocked in Write.
func WriteFloat32(ctx context.Context, destination io.Writer, tensors []Float32Tensor, limits Limits) (WriteReceipt, error) {
	if ctx == nil || destination == nil {
		return WriteReceipt{}, errors.New("safetensors: writer requires context and destination")
	}
	if limits.MaxHeaderBytes <= 0 || limits.MaxHeaderBytes > formatHeaderLimit ||
		limits.MaxTensors <= 0 || limits.MaxDimensions <= 0 ||
		limits.MaxMetadataEntries <= 0 || limits.MaxChunkBytes <= 0 {
		return WriteReceipt{}, errors.New("safetensors: invalid limits")
	}
	if err := ctx.Err(); err != nil {
		return WriteReceipt{}, err
	}
	if len(tensors) > limits.MaxTensors {
		return WriteReceipt{}, errors.New("safetensors: tensor count limit exceeded")
	}
	ordered := append([]Float32Tensor(nil), tensors...)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i].Name < ordered[j].Name })
	header, payloadBytes, err := float32Header(ctx, ordered, limits)
	if err != nil {
		return WriteReceipt{}, err
	}
	if payloadBytes > math.MaxInt64-8-int64(len(header)) {
		return WriteReceipt{}, errors.New("safetensors: file length overflows")
	}
	fileBytes := 8 + int64(len(header)) + payloadBytes
	digest := sha256.New()
	written := int64(0)
	write := func(data []byte) error {
		for len(data) > 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
			chunk := data[:min(len(data), limits.MaxChunkBytes)]
			n, err := destination.Write(chunk)
			if n < 0 || n > len(chunk) {
				return errors.New("safetensors: writer returned an invalid byte count")
			}
			written += int64(n)
			_, _ = digest.Write(chunk[:n])
			if err != nil {
				return fmt.Errorf("safetensors: write container: %w", err)
			}
			if n != len(chunk) {
				return io.ErrShortWrite
			}
			data = data[n:]
		}
		return nil
	}
	failure := func(err error) (WriteReceipt, error) { return WriteReceipt{Bytes: written}, err }
	var prefix [8]byte
	binary.LittleEndian.PutUint64(prefix[:], uint64(len(header)))
	if err := write(prefix[:]); err != nil {
		return failure(err)
	}
	if err := write(header); err != nil {
		return failure(err)
	}
	buffer := make([]byte, int(min(payloadBytes, int64(limits.MaxChunkBytes), 64<<10)))
	for _, tensor := range ordered {
		values := tensor.Values
		for len(values) > 0 {
			if err := ctx.Err(); err != nil {
				return failure(err)
			}
			if len(buffer) < 4 {
				var scalar [4]byte
				binary.LittleEndian.PutUint32(scalar[:], math.Float32bits(values[0]))
				if err := write(scalar[:]); err != nil {
					return failure(err)
				}
				values = values[1:]
				continue
			}
			count := min(len(values), len(buffer)/4)
			for index, value := range values[:count] {
				binary.LittleEndian.PutUint32(buffer[index*4:], math.Float32bits(value))
			}
			if err := write(buffer[:count*4]); err != nil {
				return failure(err)
			}
			values = values[count:]
		}
	}
	if err := ctx.Err(); err != nil {
		return failure(err)
	}
	if written != fileBytes {
		return failure(errors.New("safetensors: incomplete container"))
	}
	return WriteReceipt{SHA256: hex.EncodeToString(digest.Sum(nil)), Bytes: written,
		HeaderBytes: int64(len(header)), PayloadBytes: payloadBytes, Tensors: len(ordered)}, nil
}

func float32Header(ctx context.Context, tensors []Float32Tensor, limits Limits) ([]byte, int64, error) {
	header := make([]byte, 0)
	appendText := func(text string) error {
		if int64(len(text)) > limits.MaxHeaderBytes-int64(len(header)) {
			return errors.New("safetensors: header limit exceeded")
		}
		header = append(header, text...)
		return nil
	}
	if err := appendText(`{"__metadata__":{"format":"pt"}`); err != nil {
		return nil, 0, err
	}
	cursor := int64(0)
	for index, tensor := range tensors {
		if err := ctx.Err(); err != nil {
			return nil, 0, err
		}
		if tensor.Name == "" || tensor.Name == "__metadata__" || !utf8.ValidString(tensor.Name) {
			return nil, 0, errors.New("safetensors: tensor name is empty, reserved or invalid UTF-8")
		}
		if index > 0 && tensor.Name == tensors[index-1].Name {
			return nil, 0, errors.New("safetensors: duplicate tensor name")
		}
		if len(tensor.Shape) > limits.MaxDimensions {
			return nil, 0, errors.New("safetensors: dimension limit exceeded")
		}
		elements := uint64(1)
		for dimensionIndex, dimension := range tensor.Shape {
			if dimensionIndex%4096 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, 0, err
				}
			}
			if dimension != 0 && elements > math.MaxUint64/dimension {
				return nil, 0, errors.New("safetensors: shape product overflows")
			}
			elements *= dimension
		}
		// Match the reader's bit-length check as well as int64 byte offsets.
		if elements > math.MaxUint64/32 || elements > uint64(math.MaxInt64-cursor)/4 {
			return nil, 0, errors.New("safetensors: tensor storage length overflows")
		}
		if elements != uint64(len(tensor.Values)) {
			return nil, 0, errors.New("safetensors: shape does not match value count")
		}
		for valueIndex, value := range tensor.Values {
			if valueIndex%4096 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, 0, err
				}
			}
			if math.Float32bits(value)&0x7f800000 == 0x7f800000 {
				return nil, 0, errors.New("safetensors: non-finite float32 value")
			}
		}
		if !float32NameFits(tensor.Name, limits.MaxHeaderBytes-int64(len(header))) {
			return nil, 0, errors.New("safetensors: tensor name exceeds header limit")
		}
		name, err := json.Marshal(tensor.Name)
		if err != nil {
			return nil, 0, err
		}
		for _, part := range []string{",", string(name), `:{"dtype":"F32","shape":[`} {
			if err := appendText(part); err != nil {
				return nil, 0, err
			}
		}
		for dimensionIndex, dimension := range tensor.Shape {
			if dimensionIndex > 0 {
				if err := appendText(","); err != nil {
					return nil, 0, err
				}
			}
			if err := appendText(strconv.FormatUint(dimension, 10)); err != nil {
				return nil, 0, err
			}
		}
		end := cursor + int64(elements*4)
		if err := appendText(`],"data_offsets":[` + strconv.FormatInt(cursor, 10) + "," + strconv.FormatInt(end, 10) + "]}"); err != nil {
			return nil, 0, err
		}
		cursor = end
	}
	if err := appendText("}"); err != nil {
		return nil, 0, err
	}
	for len(header)%8 != 0 {
		if err := appendText(" "); err != nil {
			return nil, 0, err
		}
	}
	return header, cursor, ctx.Err()
}

// Count encoding/json's escaped string length before asking it to allocate.
// This keeps an oversized escaped name from bypassing the header bound.
func float32NameFits(name string, remaining int64) bool {
	remaining -= 2 // Quotes.
	for _, character := range name {
		switch character {
		case '"', '\\', '\b', '\f', '\n', '\r', '\t':
			remaining -= 2
		case '<', '>', '&', '\u2028', '\u2029':
			remaining -= 6
		default:
			if character < 0x20 {
				remaining -= 6
			} else {
				remaining -= int64(utf8.RuneLen(character))
			}
		}
		if remaining < 0 {
			return false
		}
	}
	return remaining >= 0
}
