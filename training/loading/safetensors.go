// Package loading admits individual checkpoint tensors into native storage.
// Callers own files, shard mapping, model identity and backend resource admission.
package loading

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"math"
	"slices"
	"strings"

	"github.com/tayi-ai/arandu-llama/checkpoint"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

// ErrBudget indicates that explicit copy storage exceeds the caller's admission.
var ErrBudget = errors.New("loading: tensor copy budget is insufficient or invalid")

// ErrExpectation indicates that required tensor metadata or content differs.
var ErrExpectation = errors.New("loading: tensor does not match the expected identity")

// Expectation contains optional checks over the source tensor. A nil Shape skips
// its check; a non-nil empty Shape requires a scalar. DType uses Safetensors names.
// An empty DType or SHA256 skips that check; a supplied digest must contain 64 hex digits.
type Expectation struct {
	Shape  []int64
	DType  string
	SHA256 string
}

// Options explicitly selects parsing limits, copy admission and native placement.
// BudgetBytes must admit three times the larger source/destination storage size.
// This accounts conservatively for the Go payload, native CPU copy and converted
// destination. Header parsing, allocator overhead, transfer workspaces and other
// live tensors are separate caller-owned budgets; this is not a total RSS cap.
// Device selection does not establish permission or qualify GPU memory capacity.
type Options struct {
	HeaderLimits checkpoint.Limits
	BudgetBytes  int64
	DType        torch.DType
	Device       torch.Device
	RequiresGrad bool
	Expected     *Expectation
}

// Receipt records source identity and explicit storage admission. SHA256 is set
// only after a complete payload read, even when an expected digest then differs.
// Loaded is true only when the returned native tensor passed all requested checks.
type Receipt struct {
	Name              string       `json:"name"`
	SourceDType       string       `json:"source_dtype"`
	TargetDType       torch.DType  `json:"target_dtype"`
	Device            torch.Device `json:"device"`
	Shape             []int64      `json:"shape"`
	Elements          int64        `json:"elements"`
	SourceBytes       int64        `json:"source_bytes"`
	TargetBytes       int64        `json:"target_bytes"`
	AdmittedCopyBytes int64        `json:"admitted_copy_bytes"`
	BudgetBytes       int64        `json:"budget_bytes"`
	BytesRead         int64        `json:"bytes_read"`
	ReadChunks        int64        `json:"read_chunks"`
	MaxChunkBytes     int          `json:"max_chunk_bytes"`
	SHA256            string       `json:"sha256"`
	Loaded            bool         `json:"loaded"`
}

// LoadTensor validates a Safetensors header, admits one tensor, hashes its bytes
// in bounded chunks, and copies it to the explicitly requested dtype and device.
// It never writes to or closes source. The caller must keep source immutable and
// open for the duration of this call. This does not verify other tensor payloads
// or a complete model manifest. The returned tensor is an owned autograd leaf.
// Cancellation is checked between reads and native phases; a blocked ReaderAt or
// native operation cannot be interrupted by this function. The caller must Close
// a successful result; every native handle is closed on failure.
func LoadTensor(ctx context.Context, source io.ReaderAt, fileSize int64, name string, options Options) (*torch.Tensor, Receipt, error) {
	receipt := Receipt{Name: name, TargetDType: options.DType, Device: options.Device,
		BudgetBytes: options.BudgetBytes, MaxChunkBytes: options.HeaderLimits.MaxChunkBytes}
	if ctx == nil {
		return nil, receipt, errors.New("loading: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, receipt, err
	}
	if err := validateOptions(options); err != nil {
		return nil, receipt, err
	}
	if !torch.Enabled() {
		return nil, receipt, torch.ErrUnavailable
	}
	if source == nil {
		return nil, receipt, errors.New("loading: nil source")
	}
	index, err := checkpoint.OpenSafetensors(contextReader{ctx: ctx, source: source}, fileSize, options.HeaderLimits)
	if err != nil {
		return nil, receipt, err
	}
	descriptor, found := index.Tensor(name)
	if !found {
		return nil, receipt, fmt.Errorf("loading: tensor %q is absent", name)
	}
	receipt.SourceDType, receipt.SourceBytes = descriptor.DType, descriptor.Size()
	sourceDType, err := storageDType(descriptor.DType)
	if err != nil {
		return nil, receipt, err
	}
	shape, elements, err := nativeShape(descriptor.Shape)
	if err != nil {
		return nil, receipt, err
	}
	receipt.Shape, receipt.Elements = shape, elements
	if options.Expected != nil {
		expected := options.Expected
		if expected.Shape != nil && !slices.Equal(expected.Shape, shape) {
			return nil, receipt, fmt.Errorf("%w: shape got %v, want %v", ErrExpectation, shape, expected.Shape)
		}
		if expected.DType != "" && expected.DType != descriptor.DType {
			return nil, receipt, fmt.Errorf("%w: dtype got %q, want %q", ErrExpectation, descriptor.DType, expected.DType)
		}
	}
	width, _ := dtypeBytes(options.DType)
	if elements > math.MaxInt64/width {
		return nil, receipt, fmt.Errorf("%w: destination storage overflows", ErrBudget)
	}
	receipt.TargetBytes = elements * width
	largest := max(receipt.SourceBytes, receipt.TargetBytes)
	if largest > math.MaxInt64/3 || receipt.SourceBytes > int64(int(^uint(0)>>1)) {
		return nil, receipt, fmt.Errorf("%w: tensor exceeds the addressable copy size", ErrBudget)
	}
	receipt.AdmittedCopyBytes = 3 * largest
	if receipt.AdmittedCopyBytes > options.BudgetBytes {
		return nil, receipt, fmt.Errorf("%w: need %d bytes, admitted %d", ErrBudget, receipt.AdmittedCopyBytes, options.BudgetBytes)
	}
	if err := ctx.Err(); err != nil {
		return nil, receipt, err
	}
	reader, err := index.TensorReader(name)
	if err != nil {
		return nil, receipt, err
	}
	// Allocate only after metadata, dtype, size and the three-buffer admission pass.
	// Each chunk is a view of the final payload buffer; no extra scratch is needed.
	payload := make([]byte, int(receipt.SourceBytes))
	digest := sha256.New()
	for offset := int64(0); offset < receipt.SourceBytes; {
		if err := ctx.Err(); err != nil {
			return nil, receipt, err
		}
		length := min(receipt.SourceBytes-offset, int64(options.HeaderLimits.MaxChunkBytes))
		chunk := payload[int(offset):int(offset+length)]
		n, err := io.ReadFull(reader, chunk)
		receipt.BytesRead += int64(n)
		receipt.ReadChunks++
		if err != nil {
			return nil, receipt, fmt.Errorf("loading: read tensor %q: %w", name, err)
		}
		_, _ = digest.Write(chunk)
		offset += length
	}
	receipt.SHA256 = hex.EncodeToString(digest.Sum(nil))
	if options.Expected != nil && options.Expected.SHA256 != "" && !strings.EqualFold(options.Expected.SHA256, receipt.SHA256) {
		return nil, receipt, fmt.Errorf("%w: SHA-256 got %s", ErrExpectation, receipt.SHA256)
	}
	if err := ctx.Err(); err != nil {
		return nil, receipt, err
	}
	cpu, err := torch.FromBytes(payload, shape, sourceDType, torch.CPUDevice(), false)
	if err != nil {
		return nil, receipt, err
	}
	defer cpu.Close()
	if err := ctx.Err(); err != nil {
		return nil, receipt, err
	}
	converted, err := cpu.To(options.Device, options.DType)
	if err != nil {
		return nil, receipt, err
	}
	keep := false
	defer func() {
		if !keep {
			_ = converted.Close()
		}
	}()
	if err := ctx.Err(); err != nil {
		return nil, receipt, err
	}
	if options.RequiresGrad {
		leaf, err := converted.SetRequiresGrad(true)
		if err != nil {
			return nil, receipt, err
		}
		_ = converted.Close()
		converted = leaf
	}
	if err := ctx.Err(); err != nil {
		return nil, receipt, err
	}
	info, err := converted.Info()
	if err != nil || info.DType != options.DType || info.Device != options.Device ||
		info.RequiresGrad != options.RequiresGrad || !slices.Equal(info.Shape, shape) {
		return nil, receipt, fmt.Errorf("loading: native tensor metadata mismatch: %v", err)
	}
	keep, receipt.Loaded = true, true
	return converted, receipt, nil
}

func validateOptions(options Options) error {
	if options.BudgetBytes <= 0 {
		return ErrBudget
	}
	if _, err := dtypeBytes(options.DType); err != nil {
		return err
	}
	if options.Device != torch.CPUDevice() && options.Device != torch.MPSDevice() &&
		(options.Device.Kind != "cuda" || options.Device.Index < 0 || options.Device.Index > 127) {
		return errors.New("loading: explicit CPU, MPS, or indexed CUDA device is required")
	}
	if options.RequiresGrad && (options.DType == torch.Int64 || options.DType == torch.Bool) {
		return errors.New("loading: integer and boolean destination tensors cannot require gradients")
	}
	if options.Expected != nil && options.Expected.SHA256 != "" {
		digest, err := hex.DecodeString(options.Expected.SHA256)
		if err != nil || len(digest) != sha256.Size {
			return fmt.Errorf("%w: SHA-256 must contain 64 hex digits", ErrExpectation)
		}
	}
	return nil
}

func storageDType(name string) (torch.DType, error) {
	switch name {
	case "F16":
		return torch.Float16, nil
	case "BF16":
		return torch.BFloat16, nil
	case "F32":
		return torch.Float32, nil
	case "F64":
		return torch.Float64, nil
	case "I64":
		return torch.Int64, nil
	case "BOOL":
		return torch.Bool, nil
	default:
		return 0, fmt.Errorf("loading: source dtype %q is unsupported by the native bridge", name)
	}
}

func dtypeBytes(dtype torch.DType) (int64, error) {
	switch dtype {
	case torch.Float16, torch.BFloat16:
		return 2, nil
	case torch.Float32:
		return 4, nil
	case torch.Float64, torch.Int64:
		return 8, nil
	case torch.Bool:
		return 1, nil
	default:
		return 0, errors.New("loading: unsupported destination dtype")
	}
}

func nativeShape(source []uint64) ([]int64, int64, error) {
	if len(source) > 32 {
		return nil, 0, errors.New("loading: native tensor rank exceeds 32")
	}
	shape, elements := make([]int64, len(source)), int64(1)
	for i, dimension := range source {
		if dimension > math.MaxInt64 || dimension > 0 && elements > math.MaxInt64/int64(dimension) {
			return nil, 0, errors.New("loading: source shape exceeds the native addressable range")
		}
		shape[i] = int64(dimension)
		elements *= int64(dimension)
	}
	return shape, elements, nil
}

type contextReader struct {
	ctx    context.Context
	source io.ReaderAt
}

func (reader contextReader) ReadAt(destination []byte, offset int64) (int, error) {
	if err := reader.ctx.Err(); err != nil {
		return 0, err
	}
	return reader.source.ReadAt(destination, offset)
}
