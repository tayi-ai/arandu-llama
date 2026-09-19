// Package checkpoint reads checkpoint containers without converting tensor bytes.
package checkpoint

import (
	"bytes"
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
	"unicode/utf8"
)

const formatHeaderLimit = 100_000_000

// Limits bounds header parsing and the scratch buffer used to stream a tensor.
// All fields must be positive. MaxHeaderBytes cannot exceed 100,000,000 bytes.
type Limits struct {
	MaxHeaderBytes     int64
	MaxTensors         int
	MaxDimensions      int
	MaxMetadataEntries int
	MaxChunkBytes      int
}

// DefaultLimits permits a 16 MiB header, 65,536 tensors and metadata entries,
// 32 dimensions per tensor, and a streaming buffer of at most 4 MiB.
func DefaultLimits() Limits {
	return Limits{16 << 20, 65536, 32, 65536, 4 << 20}
}

// Tensor describes raw little-endian, row-major storage. DataOffsets are relative
// to the payload, not the file. Shape is empty for a scalar.
type Tensor struct {
	Name        string
	DType       string
	Shape       []uint64
	DataOffsets [2]int64
}

// Size returns the number of stored bytes, including packed sub-byte values.
func (t Tensor) Size() int64 { return t.DataOffsets[1] - t.DataOffsets[0] }

// Safetensors is a validated index over a caller-owned ReaderAt. The caller must
// keep the source open and unchanged while using the index and its readers.
// Opening validates structure, not payload values, provenance, or model identity.
type Safetensors struct {
	source        io.ReaderAt
	payloadOffset int64
	limits        Limits
	tensors       map[string]Tensor
	metadata      map[string]string
}

// OpenSafetensors reads only the eight-byte prefix and the bounded JSON header.
// Every payload byte must belong to exactly one tensor; empty tensors are valid
// at storage boundaries. Unknown dtypes and tensor fields are rejected.
// The wire format and dtype widths follow huggingface/safetensors v0.8.0.
func OpenSafetensors(source io.ReaderAt, size int64, limits Limits) (*Safetensors, error) {
	if source == nil {
		return nil, errors.New("safetensors: nil source")
	}
	if limits.MaxHeaderBytes <= 0 || limits.MaxHeaderBytes > formatHeaderLimit ||
		limits.MaxTensors <= 0 || limits.MaxDimensions <= 0 ||
		limits.MaxMetadataEntries <= 0 || limits.MaxChunkBytes <= 0 {
		return nil, errors.New("safetensors: invalid limits")
	}
	if size < 8 {
		return nil, errors.New("safetensors: missing header length")
	}
	var prefix [8]byte
	if err := readExactAt(source, prefix[:], 0); err != nil {
		return nil, fmt.Errorf("safetensors: read header length: %w", err)
	}
	headerLength := binary.LittleEndian.Uint64(prefix[:])
	if headerLength == 0 || headerLength > uint64(limits.MaxHeaderBytes) || headerLength > uint64(size-8) {
		return nil, errors.New("safetensors: header length exceeds file or configured limit")
	}
	header := make([]byte, int(headerLength))
	if err := readExactAt(source, header, 8); err != nil {
		return nil, fmt.Errorf("safetensors: read header: %w", err)
	}
	if header[0] != '{' || !utf8.Valid(header) || !validSurrogates(header) {
		return nil, errors.New("safetensors: header must start with an object and contain valid Unicode")
	}
	index := &Safetensors{
		source: source, payloadOffset: 8 + int64(headerLength), limits: limits,
		tensors: make(map[string]Tensor), metadata: make(map[string]string),
	}
	if err := parseObject(header, func(name string, value json.RawMessage) error {
		if name == "__metadata__" {
			return parseObject(value, func(key string, raw json.RawMessage) error {
				if len(index.metadata) >= limits.MaxMetadataEntries {
					return errors.New("metadata entry limit exceeded")
				}
				var text string
				if len(raw) == 0 || raw[0] != '"' || json.Unmarshal(raw, &text) != nil {
					return errors.New("metadata values must be strings")
				}
				index.metadata[key] = text
				return nil
			})
		}
		if len(index.tensors) >= limits.MaxTensors {
			return errors.New("tensor count limit exceeded")
		}
		tensor, err := parseTensor(name, value, limits.MaxDimensions)
		if err != nil {
			return fmt.Errorf("tensor %q: %w", name, err)
		}
		index.tensors[name] = tensor
		return nil
	}); err != nil {
		return nil, fmt.Errorf("safetensors: invalid header: %w", err)
	}
	ordered := index.Tensors()
	sort.Slice(ordered, func(i, j int) bool {
		if ordered[i].DataOffsets[0] != ordered[j].DataOffsets[0] {
			return ordered[i].DataOffsets[0] < ordered[j].DataOffsets[0]
		}
		return ordered[i].DataOffsets[1] < ordered[j].DataOffsets[1]
	})
	cursor := int64(0)
	payloadLength := size - index.payloadOffset
	for _, tensor := range ordered {
		if tensor.DataOffsets[0] != cursor || tensor.DataOffsets[1] > payloadLength {
			return nil, fmt.Errorf("safetensors: tensor %q leaves a gap, overlaps, or exceeds the payload", tensor.Name)
		}
		cursor = tensor.DataOffsets[1]
	}
	if cursor != payloadLength {
		return nil, errors.New("safetensors: payload contains unindexed bytes")
	}
	return index, nil
}

// Tensors returns detached descriptors sorted by name, independently of header order.
func (s *Safetensors) Tensors() []Tensor {
	tensors := make([]Tensor, 0, len(s.tensors))
	for _, tensor := range s.tensors {
		tensors = append(tensors, cloneTensor(tensor))
	}
	sort.Slice(tensors, func(i, j int) bool { return tensors[i].Name < tensors[j].Name })
	return tensors
}

// Tensor returns a detached descriptor and whether the name exists.
func (s *Safetensors) Tensor(name string) (Tensor, bool) {
	tensor, found := s.tensors[name]
	return cloneTensor(tensor), found
}

// Metadata returns a copy of the optional string metadata.
func (s *Safetensors) Metadata() map[string]string {
	metadata := make(map[string]string, len(s.metadata))
	for key, value := range s.metadata {
		metadata[key] = value
	}
	return metadata
}

// TensorReader returns an independent reader restricted to the tensor's bytes.
// It performs no read and does not close or own the underlying source.
func (s *Safetensors) TensorReader(name string) (*io.SectionReader, error) {
	tensor, found := s.tensors[name]
	if !found {
		return nil, fmt.Errorf("safetensors: unknown tensor %q", name)
	}
	return io.NewSectionReader(s.source, s.payloadOffset+tensor.DataOffsets[0], tensor.Size()), nil
}

// CopyTensor streams unchanged bytes and returns their SHA-256 hex digest.
// The caller supplies a nonempty buffer no larger than MaxChunkBytes. Errors
// return the number of bytes written and an empty digest. Cancellation is checked
// between chunks; it cannot interrupt a blocked source or destination operation.
func (s *Safetensors) CopyTensor(ctx context.Context, name string, destination io.Writer, buffer []byte) (int64, string, error) {
	if ctx == nil || destination == nil || len(buffer) == 0 || len(buffer) > s.limits.MaxChunkBytes {
		return 0, "", errors.New("safetensors: invalid streaming arguments")
	}
	reader, err := s.TensorReader(name)
	if err != nil {
		return 0, "", err
	}
	digest := sha256.New()
	written := int64(0)
	for written < reader.Size() {
		if err := ctx.Err(); err != nil {
			return written, "", err
		}
		length := min(int64(len(buffer)), reader.Size()-written)
		chunk := buffer[:int(length)]
		if _, err := io.ReadFull(reader, chunk); err != nil {
			return written, "", fmt.Errorf("safetensors: read tensor %q: %w", name, err)
		}
		n, err := destination.Write(chunk)
		if n < 0 || n > len(chunk) {
			return written, "", errors.New("safetensors: writer returned an invalid byte count")
		}
		written += int64(n)
		if err != nil {
			return written, "", fmt.Errorf("safetensors: write tensor %q: %w", name, err)
		}
		if n != len(chunk) {
			return written, "", io.ErrShortWrite
		}
		_, _ = digest.Write(chunk)
	}
	if err := ctx.Err(); err != nil {
		return written, "", err
	}
	return written, hex.EncodeToString(digest.Sum(nil)), nil
}

// HashTensor computes SHA-256 over raw tensor bytes using the caller's bounded buffer.
func (s *Safetensors) HashTensor(ctx context.Context, name string, buffer []byte) (string, error) {
	_, digest, err := s.CopyTensor(ctx, name, io.Discard, buffer)
	return digest, err
}

func cloneTensor(tensor Tensor) Tensor {
	tensor.Shape = append([]uint64{}, tensor.Shape...)
	return tensor
}

func readExactAt(source io.ReaderAt, data []byte, offset int64) error {
	n, err := source.ReadAt(data, offset)
	if n != len(data) {
		if err == nil {
			err = io.ErrUnexpectedEOF
		}
		return err
	}
	if err != nil && err != io.EOF {
		return err
	}
	return nil
}

func parseObject(raw []byte, field func(string, json.RawMessage) error) error {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	opening, err := decoder.Token()
	if err != nil || opening != json.Delim('{') {
		return errors.New("expected an object")
	}
	seen := make(map[string]struct{})
	for decoder.More() {
		key, err := decoder.Token()
		if err != nil {
			return err
		}
		name, ok := key.(string)
		if !ok {
			return errors.New("expected an object key")
		}
		if _, duplicate := seen[name]; duplicate {
			return fmt.Errorf("duplicate key %q", name)
		}
		seen[name] = struct{}{}
		var value json.RawMessage
		if err := decoder.Decode(&value); err != nil {
			return err
		}
		if err := field(name, value); err != nil {
			return err
		}
	}
	if ending, err := decoder.Token(); err != nil || ending != json.Delim('}') {
		return errors.New("unterminated object")
	}
	for _, padding := range raw[decoder.InputOffset():] {
		if padding != ' ' {
			return errors.New("only space padding may follow an object")
		}
	}
	return nil
}

func parseTensor(name string, raw []byte, dimensionsLimit int) (Tensor, error) {
	tensor := Tensor{Name: name}
	fields := 0
	err := parseObject(raw, func(key string, value json.RawMessage) error {
		switch key {
		case "dtype":
			if len(value) == 0 || value[0] != '"' || json.Unmarshal(value, &tensor.DType) != nil {
				return errors.New("dtype must be a string")
			}
		case "shape":
			shape, err := parseUnsignedArray(value, dimensionsLimit)
			if err != nil {
				return fmt.Errorf("invalid shape: %w", err)
			}
			tensor.Shape = shape
		case "data_offsets":
			offsets, err := parseUnsignedArray(value, 2)
			if err != nil || len(offsets) != 2 {
				return errors.New("data_offsets must contain exactly two unsigned integers")
			}
			if offsets[0] > math.MaxInt64 || offsets[1] > math.MaxInt64 || offsets[1] < offsets[0] {
				return errors.New("data_offsets exceed the addressable range or are reversed")
			}
			tensor.DataOffsets = [2]int64{int64(offsets[0]), int64(offsets[1])}
		default:
			return fmt.Errorf("unknown tensor field %q", key)
		}
		fields++
		return nil
	})
	if err != nil {
		return Tensor{}, err
	}
	if fields != 3 {
		return Tensor{}, errors.New("dtype, shape and data_offsets are required")
	}
	width, ok := dtypeBits[tensor.DType]
	if !ok {
		return Tensor{}, fmt.Errorf("unsupported dtype %q", tensor.DType)
	}
	elements := uint64(1)
	for _, dimension := range tensor.Shape {
		if dimension != 0 && elements > math.MaxUint64/dimension {
			return Tensor{}, errors.New("shape product overflows")
		}
		elements *= dimension
	}
	if elements > math.MaxUint64/width {
		return Tensor{}, errors.New("tensor bit length overflows")
	}
	bits := elements * width
	if bits%8 != 0 || bits/8 != uint64(tensor.Size()) {
		return Tensor{}, errors.New("dtype and shape do not match the byte-aligned storage length")
	}
	return tensor, nil
}

func parseUnsignedArray(raw []byte, limit int) ([]uint64, error) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	if opening, err := decoder.Token(); err != nil || opening != json.Delim('[') {
		return nil, errors.New("expected an array")
	}
	values := make([]uint64, 0)
	for decoder.More() {
		if len(values) >= limit {
			return nil, errors.New("array length limit exceeded")
		}
		var rawValue json.RawMessage
		if err := decoder.Decode(&rawValue); err != nil {
			return nil, err
		}
		if len(rawValue) == 0 || rawValue[0] < '0' || rawValue[0] > '9' {
			return nil, errors.New("expected an unsigned integer")
		}
		var value uint64
		if err := json.Unmarshal(rawValue, &value); err != nil {
			return nil, errors.New("expected an unsigned 64-bit integer")
		}
		values = append(values, value)
	}
	if ending, err := decoder.Token(); err != nil || ending != json.Delim(']') {
		return nil, errors.New("unterminated array")
	}
	if _, err := decoder.Token(); err != io.EOF {
		return nil, errors.New("unexpected data after array")
	}
	return values, nil
}

// Go's JSON decoder replaces unmatched UTF-16 escapes. Reject them instead of
// changing tensor names or metadata while parsing an otherwise UTF-8 header.
func validSurrogates(raw []byte) bool {
	inString := false
	for i := 0; i < len(raw); i++ {
		if raw[i] == '"' {
			inString = !inString
			continue
		}
		if !inString || raw[i] != '\\' {
			continue
		}
		i++
		if i >= len(raw) {
			return false
		}
		if raw[i] != 'u' {
			continue
		}
		value, ok := hexQuad(raw[i+1:])
		if !ok || value >= 0xDC00 && value <= 0xDFFF {
			return false
		}
		i += 4
		if value < 0xD800 || value > 0xDBFF {
			continue
		}
		if len(raw)-i < 7 || raw[i+1] != '\\' || raw[i+2] != 'u' {
			return false
		}
		low, ok := hexQuad(raw[i+3:])
		if !ok || low < 0xDC00 || low > 0xDFFF {
			return false
		}
		i += 6
	}
	return true
}

func hexQuad(raw []byte) (uint16, bool) {
	if len(raw) < 4 {
		return 0, false
	}
	var value uint16
	for _, character := range raw[:4] {
		value <<= 4
		switch {
		case character >= '0' && character <= '9':
			value += uint16(character - '0')
		case character >= 'a' && character <= 'f':
			value += uint16(character-'a') + 10
		case character >= 'A' && character <= 'F':
			value += uint16(character-'A') + 10
		default:
			return 0, false
		}
	}
	return value, true
}

var dtypeBits = map[string]uint64{
	"BOOL": 8, "F4": 4, "F6_E2M3": 6, "F6_E3M2": 6,
	"U8": 8, "I8": 8, "F8_E5M2": 8, "F8_E4M3": 8, "F8_E8M0": 8,
	"F8_E4M3FNUZ": 8, "F8_E5M2FNUZ": 8,
	"I16": 16, "U16": 16, "F16": 16, "BF16": 16,
	"I32": 32, "U32": 32, "F32": 32,
	"C64": 64, "F64": 64, "I64": 64, "U64": 64,
}
