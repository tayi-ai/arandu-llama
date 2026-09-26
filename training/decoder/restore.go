package decoder

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"math"
	"os"

	"github.com/tayi-ai/arandu-llama/checkpoint"
)

// ErrAdapterCheckpoint reports refused adapter bytes, identity or resource bounds.
var ErrAdapterCheckpoint = errors.New("decoder: adapter checkpoint rejected")

// AdapterCheckpoint admits one immutable F32 safetensors file. FileSHA256 covers
// its complete bytes. ParametersSHA256 uses ReplaceParameters' registry order:
// each UTF-8 parameter name followed by its little-endian FP32 values.
//
// MaxBytes bounds the complete file. MaxWorkingBytes bounds bulk live payload:
// file snapshot, bounded header, three parameter vectors (decoded Go values,
// replacement encoding and prepared native leaves), and two largest-tensor copy
// buffers. It excludes existing model storage, parser/registry metadata, native
// allocator overhead and Go GC retention; those need independent admission.
// Limits additionally bounds header parsing, metadata, tensor count and reads.
type AdapterCheckpoint struct {
	Path             string
	FileSHA256       string
	ParametersSHA256 string
	MaxBytes         int64
	MaxWorkingBytes  int64
	Limits           checkpoint.Limits
}

// RestoreAdapter installs only the admitted adapter; it never reads or creates
// optimizer moments. It verifies the regular file, both external hashes, exact
// registry names/shapes, F32 dtype and finite values before native installation.
// All reads and parsing use one bounded, hash-verified Go-owned file snapshot,
// so later file mutation cannot change the installed vector.
//
// The caller must serialize model access as required by ReplaceParameters.
// Validation, reading, preparation and cancellation failures return an empty
// digest and preserve every old parameter. ErrParameterRelease has the existing
// post-commit meaning: a nonempty digest identifies active new parameters whose
// old handles failed to release. Abort instead of retrying that committed update.
// Cancellation is checked between bounded reads and native operations, but cannot
// interrupt an already blocked filesystem or native call.
func (m *LoadedTextModel) RestoreAdapter(ctx context.Context, source AdapterCheckpoint) (string, error) {
	if ctx == nil || m == nil || m.Model == nil || len(m.Parameters) == 0 || source.Path == "" ||
		!validAssemblyHash(source.FileSHA256) || !validAssemblyHash(source.ParametersSHA256) || source.MaxBytes <= 0 || source.MaxWorkingBytes <= 0 ||
		source.Limits.MaxHeaderBytes <= 0 || source.Limits.MaxHeaderBytes > 100_000_000 || source.Limits.MaxTensors <= 0 ||
		source.Limits.MaxDimensions <= 0 || source.Limits.MaxMetadataEntries <= 0 || source.Limits.MaxChunkBytes <= 0 {
		return "", ErrAdapterCheckpoint
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	registry, err := candidateParameters(m.Model)
	if err != nil || len(registry) != len(m.Parameters) || len(registry) > source.Limits.MaxTensors {
		return "", errors.Join(ErrAdapterCheckpoint, err)
	}
	expected := make([]AssemblyTensor, len(registry))
	var payload, largest int64
	for i, parameter := range registry {
		if parameter.info.Elements > math.MaxInt64/4 || parameter.info.Elements > int64(int(^uint(0)>>1))/4 ||
			len(parameter.info.Shape) > source.Limits.MaxDimensions {
			return "", fmt.Errorf("%w: registry geometry exceeds bounds", ErrAdapterCheckpoint)
		}
		size := parameter.info.Elements * 4
		if size > source.MaxWorkingBytes-payload {
			return "", fmt.Errorf("%w: parameter payload exceeds working budget", ErrAdapterCheckpoint)
		}
		payload += size
		largest = max(largest, size)
		expected[i] = AssemblyTensor{ReferenceName: parameter.name, Shape: parameter.info.Shape, Bytes: size}
	}
	if _, _, err := m.validateParameterRegistry(ctx, expected); err != nil {
		return "", errors.Join(ErrAdapterCheckpoint, err)
	}
	body, err := readAdapterSnapshot(ctx, source, payload, largest)
	if err != nil {
		return "", err
	}
	values, err := decodeAdapterSnapshot(ctx, body, expected, payload, source)
	if err != nil {
		return "", err
	}
	return m.ReplaceParameters(ctx, values)
}

func adapterWorkingBudget(fileBytes, payload, largest int64, source AdapterCheckpoint) error {
	if fileBytes < 8 || fileBytes > source.MaxBytes || fileBytes > int64(int(^uint(0)>>1)) ||
		payload <= 0 || payload > int64(int(^uint(0)>>1)) || largest <= 0 || largest > payload {
		return fmt.Errorf("%w: file or parameter size exceeds bounds", ErrAdapterCheckpoint)
	}
	remaining := source.MaxWorkingBytes
	for _, size := range []int64{fileBytes, min(source.Limits.MaxHeaderBytes, fileBytes-8), payload, payload, payload, largest, largest} {
		if size < 0 || size > remaining {
			return fmt.Errorf("%w: working payload exceeds budget", ErrAdapterCheckpoint)
		}
		remaining -= size
	}
	return nil
}

func readAdapterSnapshot(ctx context.Context, source AdapterCheckpoint, payload, largest int64) (_ []byte, err error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	before, err := os.Lstat(source.Path)
	if err != nil {
		return nil, err
	}
	if !before.Mode().IsRegular() {
		return nil, fmt.Errorf("%w: regular file required", ErrAdapterCheckpoint)
	}
	if err := adapterWorkingBudget(before.Size(), payload, largest, source); err != nil {
		return nil, err
	}
	file, err := os.Open(source.Path)
	if err != nil {
		return nil, err
	}
	defer func() { err = errors.Join(err, file.Close()) }()
	opened, err := file.Stat()
	if err != nil || !opened.Mode().IsRegular() || !os.SameFile(before, opened) || opened.Size() != before.Size() || !opened.ModTime().Equal(before.ModTime()) {
		return nil, errors.Join(fmt.Errorf("%w: file changed before reading", ErrAdapterCheckpoint), err)
	}
	body := make([]byte, int(opened.Size()))
	digest := sha256.New()
	chunk := min(source.Limits.MaxChunkBytes, ObservationChunkBytes)
	for start := 0; start < len(body); {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		end := start + min(chunk, len(body)-start)
		if _, err := io.ReadFull(file, body[start:end]); err != nil {
			return nil, err
		}
		_, _ = digest.Write(body[start:end])
		start = end
	}
	after, err := file.Stat()
	if err != nil || after.Size() != opened.Size() || !after.ModTime().Equal(opened.ModTime()) ||
		hex.EncodeToString(digest.Sum(nil)) != source.FileSHA256 {
		return nil, errors.Join(fmt.Errorf("%w: file identity differs", ErrAdapterCheckpoint), err)
	}
	return body, ctx.Err()
}

func decodeAdapterSnapshot(ctx context.Context, body []byte, expected []AssemblyTensor, payload int64, source AdapterCheckpoint) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	index, err := checkpoint.OpenSafetensors(bytes.NewReader(body), int64(len(body)), source.Limits)
	if err != nil || len(index.Tensors()) != len(expected) {
		return nil, errors.Join(fmt.Errorf("%w: tensor index differs", ErrAdapterCheckpoint), err)
	}
	// Admit every descriptor before allocating the complete incoming vector.
	for _, item := range expected {
		tensor, found := index.Tensor(item.ReferenceName)
		if !found || tensor.DType != "F32" || len(tensor.Shape) != len(item.Shape) || tensor.Size() != item.Bytes {
			return nil, fmt.Errorf("%w: tensor identity or dtype differs", ErrAdapterCheckpoint)
		}
		for axis, dimension := range item.Shape {
			if dimension <= 0 || tensor.Shape[axis] != uint64(dimension) {
				return nil, fmt.Errorf("%w: tensor shape differs", ErrAdapterCheckpoint)
			}
		}
	}
	values := make([]float32, int(payload/4))
	aggregate := sha256.New()
	write := 0
	for _, item := range expected {
		reader, err := index.TensorReader(item.ReferenceName)
		if err != nil {
			return nil, err
		}
		_, _ = aggregate.Write([]byte(item.ReferenceName))
		_, offset, size := reader.Outer()
		content := body[offset : offset+size]
		for start := 0; start < len(content); {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			end := start + min(64<<10, len(content)-start)
			_, _ = aggregate.Write(content[start:end])
			for position := start; position < end; position += 4 {
				value := math.Float32frombits(binary.LittleEndian.Uint32(content[position:]))
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
					return nil, fmt.Errorf("%w: nonfinite tensor value", ErrAdapterCheckpoint)
				}
				values[write] = value
				write++
			}
			start = end
		}
	}
	if write != len(values) || hex.EncodeToString(aggregate.Sum(nil)) != source.ParametersSHA256 {
		return nil, fmt.Errorf("%w: parameter digest differs", ErrAdapterCheckpoint)
	}
	return values, ctx.Err()
}
