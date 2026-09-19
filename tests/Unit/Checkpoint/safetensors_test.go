package checkpoint_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"testing"

	"github.com/tayi-ai/arandu-llama/checkpoint"
)

func container(header string, payload []byte) []byte {
	data := make([]byte, 8+len(header)+len(payload))
	binary.LittleEndian.PutUint64(data, uint64(len(header)))
	copy(data[8:], header)
	copy(data[8+len(header):], payload)
	return data
}

func open(t *testing.T, header string, payload []byte) *checkpoint.Safetensors {
	t.Helper()
	data := container(header, payload)
	index, err := checkpoint.OpenSafetensors(bytes.NewReader(data), int64(len(data)), checkpoint.DefaultLimits())
	if err != nil {
		t.Fatal(err)
	}
	return index
}

func TestSafetensorsGoldenPreservesRawBitsAndBounds(t *testing.T) {
	// Payload includes float16 NaN and infinity, bfloat16 and float32. The reader
	// must preserve their bits without decoding, normalizing or rejecting values.
	header := `{"z":{"dtype":"F32","shape":[],"data_offsets":[8,12]},"empty":{"dtype":"F16","shape":[0,7],"data_offsets":[8,8]},"half":{"dtype":"F16","shape":[2],"data_offsets":[0,4]},"__metadata__":{"format":"pt","label":"\uD83C\uDF31"},"brain":{"dtype":"BF16","shape":[2],"data_offsets":[4,8]}}   `
	payload := []byte{0x01, 0x7e, 0, 0x7c, 0x80, 0x3f, 0xc0, 0x7f, 0, 0, 0x80, 0x3f}
	index := open(t, header, payload)
	tensors := index.Tensors()
	var names []string
	for _, tensor := range tensors {
		names = append(names, tensor.Name)
	}
	if strings.Join(names, ",") != "brain,empty,half,z" || index.Metadata()["label"] != "🌱" {
		t.Fatalf("unexpected index: %v %v", names, index.Metadata())
	}
	for name, expected := range map[string][]byte{
		"half": payload[:4], "brain": payload[4:8], "z": payload[8:], "empty": {},
	} {
		t.Run(name, func(t *testing.T) {
			var destination bytes.Buffer
			written, digest, err := index.CopyTensor(context.Background(), name, &destination, make([]byte, 3))
			expectedDigest := sha256.Sum256(expected)
			if err != nil || written != int64(len(expected)) || !bytes.Equal(destination.Bytes(), expected) || digest != hex.EncodeToString(expectedDigest[:]) {
				t.Fatalf("copy: n=%d digest=%q bytes=%x err=%v", written, digest, destination.Bytes(), err)
			}
			actualHash, err := index.HashTensor(context.Background(), name, make([]byte, 2))
			if err != nil || actualHash != digest {
				t.Fatalf("hash: %q %v", actualHash, err)
			}
		})
	}
	reader, err := index.TensorReader("half")
	if err != nil {
		t.Fatal(err)
	}
	buffer := make([]byte, 9)
	if n, err := reader.ReadAt(buffer, 2); n != 2 || err != io.EOF || !bytes.Equal(buffer[:2], payload[2:4]) {
		t.Fatalf("section escaped boundary: n=%d err=%v bytes=%x", n, err, buffer)
	}
	if _, err := reader.Seek(1, io.SeekStart); err != nil {
		t.Fatal(err)
	}
	if n, err := reader.Read(buffer); n != 3 || err != nil || !bytes.Equal(buffer[:3], payload[1:4]) {
		t.Fatalf("section seek: n=%d err=%v", n, err)
	}
	if n, err := reader.Read(buffer); n != 0 || err != io.EOF {
		t.Fatalf("section end: n=%d err=%v", n, err)
	}
	// Mutating public descriptors must not change a later lookup or reader.
	tensors[0].Shape[0] = 999
	tensors[0].DataOffsets[1] = 999
	metadata := index.Metadata()
	metadata["format"] = "changed"
	brain, _ := index.Tensor("brain")
	if brain.Shape[0] != 2 || brain.Size() != 4 || index.Metadata()["format"] != "pt" {
		t.Fatal("index exposed mutable state")
	}
	brain.Shape[0] = 500
	if again, _ := index.Tensor("brain"); again.Shape[0] != 2 {
		t.Fatal("lookup exposed mutable shape")
	}
	if _, exists := index.Tensor("missing"); exists {
		t.Fatal("missing tensor exists")
	}
	if _, err := index.TensorReader("missing"); err == nil {
		t.Fatal("missing tensor accepted")
	}
}

func TestSafetensorsDTypeWidths(t *testing.T) {
	cases := []struct {
		dtype    string
		elements int
		bytes    int
	}{
		{"BOOL", 1, 1}, {"F4", 2, 1}, {"F6_E2M3", 4, 3}, {"F6_E3M2", 4, 3},
		{"U8", 1, 1}, {"I8", 1, 1}, {"F8_E5M2", 1, 1}, {"F8_E4M3", 1, 1},
		{"F8_E8M0", 1, 1}, {"F8_E4M3FNUZ", 1, 1}, {"F8_E5M2FNUZ", 1, 1},
		{"I16", 1, 2}, {"U16", 1, 2}, {"F16", 1, 2}, {"BF16", 1, 2},
		{"I32", 1, 4}, {"U32", 1, 4}, {"F32", 1, 4},
		{"C64", 1, 8}, {"F64", 1, 8}, {"I64", 1, 8}, {"U64", 1, 8},
	}
	for _, test := range cases {
		t.Run(test.dtype, func(t *testing.T) {
			header := fmt.Sprintf(`{"x":{"dtype":%q,"shape":[%d],"data_offsets":[0,%d]}}`, test.dtype, test.elements, test.bytes)
			index := open(t, header, make([]byte, test.bytes))
			tensor, _ := index.Tensor("x")
			if tensor.Size() != int64(test.bytes) || tensor.DType != test.dtype {
				t.Fatalf("unexpected dtype descriptor: %+v", tensor)
			}
		})
	}
}

func TestSafetensorsRejectsMalformedHeadersAndLayouts(t *testing.T) {
	validTensor := `{"dtype":"U8","shape":[1],"data_offsets":[0,1]}`
	cases := []struct {
		name    string
		header  string
		payload int
	}{
		{"empty", "", 0}, {"not_object", `[]`, 0}, {"leading_space", ` {}`, 0},
		{"invalid_utf8", "{\"\xff\":{}}", 0}, {"trailing_document", `{} {}`, 0},
		{"newline_padding", "{}\n", 0}, {"nul_padding", "{}\x00", 0}, {"incomplete", `{"x":`, 0},
		{"duplicate_name", `{"x":` + validTensor + `,"x":` + validTensor + `}`, 1},
		{"escaped_duplicate_name", `{"x":` + validTensor + `,"\u0078":` + validTensor + `}`, 1},
		{"duplicate_field", `{"x":{"dtype":"U8","dtype":"U8","shape":[1],"data_offsets":[0,1]}}`, 1},
		{"duplicate_metadata_key", `{"__metadata__":{"a":"b","\u0061":"c"}}`, 0},
		{"duplicate_metadata", `{"__metadata__":{},"__metadata__":{}}`, 0},
		{"metadata_number", `{"__metadata__":{"a":1}}`, 0},
		{"metadata_null_value", `{"__metadata__":{"a":null}}`, 0},
		{"metadata_null", `{"__metadata__":null}`, 0}, {"metadata_array", `{"__metadata__":[]}`, 0},
		{"metadata_nested", `{"__metadata__":{"a":{"b":"c"}}}`, 0},
		{"surrogate_high", `{"__metadata__":{"a":"\uD800"}}`, 0},
		{"surrogate_low", `{"__metadata__":{"a":"\uDC00"}}`, 0},
		{"surrogate_pair_invalid", `{"__metadata__":{"a":"\uD800\u1234"}}`, 0},
		{"unknown_field", `{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1],"extra":0}}`, 1},
		{"missing_dtype", `{"x":{"shape":[1],"data_offsets":[0,1]}}`, 1},
		{"missing_shape", `{"x":{"dtype":"U8","data_offsets":[0,1]}}`, 1},
		{"missing_offsets", `{"x":{"dtype":"U8","shape":[1]}}`, 1},
		{"unknown_dtype", `{"x":{"dtype":"F128","shape":[1],"data_offsets":[0,1]}}`, 1},
		{"null_dtype", `{"x":{"dtype":null,"shape":[1],"data_offsets":[0,1]}}`, 1},
		{"null_shape", `{"x":{"dtype":"U8","shape":null,"data_offsets":[0,1]}}`, 1},
		{"negative_shape", `{"x":{"dtype":"U8","shape":[-1],"data_offsets":[0,1]}}`, 1},
		{"negative_zero_shape", `{"x":{"dtype":"U8","shape":[-0],"data_offsets":[0,0]}}`, 0},
		{"fraction_shape", `{"x":{"dtype":"U8","shape":[1.0],"data_offsets":[0,1]}}`, 1},
		{"exponent_shape", `{"x":{"dtype":"U8","shape":[1e0],"data_offsets":[0,1]}}`, 1},
		{"null_dimension", `{"x":{"dtype":"U8","shape":[null],"data_offsets":[0,0]}}`, 0},
		{"shape_overflow", `{"x":{"dtype":"U8","shape":[18446744073709551616],"data_offsets":[0,0]}}`, 0},
		{"product_overflow", `{"x":{"dtype":"U8","shape":[9223372036854775808,2],"data_offsets":[0,0]}}`, 0},
		{"bitlength_overflow", `{"x":{"dtype":"F64","shape":[288230376151711744],"data_offsets":[0,0]}}`, 0},
		{"offset_overflow", `{"x":{"dtype":"U8","shape":[0],"data_offsets":[9223372036854775808,9223372036854775808]}}`, 0},
		{"offset_negative", `{"x":{"dtype":"U8","shape":[1],"data_offsets":[-1,0]}}`, 0},
		{"offset_null", `{"x":{"dtype":"U8","shape":[0],"data_offsets":[null,0]}}`, 0},
		{"offset_extra", `{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1,2]}}`, 1},
		{"offset_reversed", `{"x":{"dtype":"U8","shape":[1],"data_offsets":[1,0]}}`, 1},
		{"shape_mismatch", `{"x":{"dtype":"U8","shape":[2],"data_offsets":[0,1]}}`, 1},
		{"subbyte_misaligned", `{"x":{"dtype":"F4","shape":[1],"data_offsets":[0,1]}}`, 1},
		{"sixbit_misaligned", `{"x":{"dtype":"F6_E2M3","shape":[2],"data_offsets":[0,1]}}`, 1},
		{"scalar_mismatch", `{"x":{"dtype":"F32","shape":[],"data_offsets":[0,0]}}`, 0},
		{"gap", `{"x":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}`, 2},
		{"overlap", `{"x":` + validTensor + `,"y":` + validTensor + `}`, 1},
		{"empty_inside_tensor", `{"x":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"y":{"dtype":"U8","shape":[0],"data_offsets":[1,1]}}`, 2},
		{"past_payload", `{"x":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}`, 1},
		{"trailing_payload", `{"x":` + validTensor + `}`, 2}, {"unindexed_payload", `{}`, 1},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			data := container(test.header, make([]byte, test.payload))
			if _, err := checkpoint.OpenSafetensors(bytes.NewReader(data), int64(len(data)), checkpoint.DefaultLimits()); err == nil {
				t.Fatalf("accepted malformed header %q", test.header)
			}
		})
	}
}

func TestSafetensorsEmptyTensorsAndEscapedStrings(t *testing.T) {
	open(t, `{}`, nil)
	open(t, `{"__metadata__":{"literal":"\\uD800","quote":"\"","empty":""}}`, nil)
	open(t, `{"b":{"dtype":"F64","shape":[0],"data_offsets":[0,0]},"a":{"dtype":"U8","shape":[0],"data_offsets":[0,0]}}`, nil)
	open(t, `{"a":{"dtype":"U8","shape":[0],"data_offsets":[0,0]},"b":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"c":{"dtype":"U8","shape":[0],"data_offsets":[1,1]}}`, []byte{7})
}

func TestSafetensorsLimitsAndTruncatedSources(t *testing.T) {
	header := `{"a":{"dtype":"U8","shape":[1,1],"data_offsets":[0,1]},"b":{"dtype":"U8","shape":[1],"data_offsets":[1,2]},"__metadata__":{"a":"b","c":"d"}}`
	data := container(header, []byte{1, 2})
	cases := []struct {
		name   string
		change func(*checkpoint.Limits)
	}{
		{"header", func(l *checkpoint.Limits) { l.MaxHeaderBytes = int64(len(header) - 1) }},
		{"tensors", func(l *checkpoint.Limits) { l.MaxTensors = 1 }},
		{"dimensions", func(l *checkpoint.Limits) { l.MaxDimensions = 1 }},
		{"metadata", func(l *checkpoint.Limits) { l.MaxMetadataEntries = 1 }},
		{"zero_header", func(l *checkpoint.Limits) { l.MaxHeaderBytes = 0 }},
		{"huge_header_limit", func(l *checkpoint.Limits) { l.MaxHeaderBytes = 100_000_001 }},
		{"zero_tensors", func(l *checkpoint.Limits) { l.MaxTensors = 0 }},
		{"negative_dimensions", func(l *checkpoint.Limits) { l.MaxDimensions = -1 }},
		{"zero_metadata", func(l *checkpoint.Limits) { l.MaxMetadataEntries = 0 }},
		{"zero_chunk", func(l *checkpoint.Limits) { l.MaxChunkBytes = 0 }},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			limits := checkpoint.DefaultLimits()
			test.change(&limits)
			if _, err := checkpoint.OpenSafetensors(bytes.NewReader(data), int64(len(data)), limits); err == nil {
				t.Fatal("limit ignored")
			}
		})
	}
	for _, declaredSize := range []int64{-1, 0, 7, int64(len(data) - 10)} {
		if _, err := checkpoint.OpenSafetensors(bytes.NewReader(data), declaredSize, checkpoint.DefaultLimits()); err == nil {
			t.Fatalf("invalid declared size accepted: %d", declaredSize)
		}
	}
	for _, actualSize := range []int{0, 7, 10, len(data) - 3} {
		if _, err := checkpoint.OpenSafetensors(bytes.NewReader(data[:actualSize]), int64(len(data)), checkpoint.DefaultLimits()); err == nil {
			t.Fatalf("truncated header accepted: %d", actualSize)
		}
	}
	if _, err := checkpoint.OpenSafetensors(nil, int64(len(data)), checkpoint.DefaultLimits()); err == nil {
		t.Fatal("nil source accepted")
	}
	for _, headerLength := range []uint64{0, 100_000_001, 1 << 63, ^uint64(0)} {
		prefix := make([]byte, 8)
		binary.LittleEndian.PutUint64(prefix, headerLength)
		if _, err := checkpoint.OpenSafetensors(bytes.NewReader(prefix), 1<<62, checkpoint.DefaultLimits()); err == nil {
			t.Fatalf("hostile header length accepted: %d", headerLength)
		}
	}
	limits := checkpoint.DefaultLimits()
	limits.MaxHeaderBytes = int64(len(header))
	limits.MaxTensors, limits.MaxDimensions, limits.MaxMetadataEntries = 2, 2, 2
	if _, err := checkpoint.OpenSafetensors(bytes.NewReader(data), int64(len(data)), limits); err != nil {
		t.Fatalf("exact limits rejected: %v", err)
	}
}

type virtualShard struct {
	header       []byte
	size         int64
	payloadReads int
	requests     []int
	bytesRead    int64
	allowPayload bool
	maxPayload   int
}

func (v *virtualShard) ReadAt(destination []byte, offset int64) (int, error) {
	v.requests = append(v.requests, len(destination))
	if offset < 0 || offset >= v.size {
		return 0, io.EOF
	}
	if offset < int64(len(v.header)) {
		n := copy(destination, v.header[offset:])
		v.bytesRead += int64(n)
		if n != len(destination) {
			return n, io.EOF
		}
		return n, nil
	}
	v.payloadReads++
	if !v.allowPayload || len(destination) > v.maxPayload {
		return 0, errors.New("unexpected payload read")
	}
	n := min(int64(len(destination)), v.size-offset)
	for i := int64(0); i < n; i++ {
		destination[i] = byte((offset - int64(len(v.header)) + i) % 251)
	}
	v.bytesRead += n
	if n != int64(len(destination)) {
		return int(n), io.EOF
	}
	return int(n), nil
}

func TestSafetensorsOpensVirtualMultiGiBShardWithoutReadingPayload(t *testing.T) {
	const payloadSize = int64(5<<30) + 19
	header := fmt.Sprintf(`{"large":{"dtype":"U8","shape":[%d],"data_offsets":[0,%d]},"tail":{"dtype":"U8","shape":[19],"data_offsets":[%d,%d]}}`, payloadSize-19, payloadSize-19, payloadSize-19, payloadSize)
	prefix := container(header, nil)
	source := &virtualShard{header: prefix, size: int64(len(prefix)) + payloadSize}
	index, err := checkpoint.OpenSafetensors(source, source.size, checkpoint.DefaultLimits())
	if err != nil {
		t.Fatal(err)
	}
	if source.payloadReads != 0 || source.bytesRead != int64(len(prefix)) || len(source.requests) != 2 || source.requests[0] != 8 || source.requests[1] != len(header) {
		t.Fatalf("opening read outside header: %+v", source)
	}
	source.allowPayload, source.maxPayload = true, 7
	var destination bytes.Buffer
	written, digest, err := index.CopyTensor(context.Background(), "tail", &destination, make([]byte, 7))
	expected := make([]byte, 19)
	for i := range expected {
		expected[i] = byte((payloadSize - 19 + int64(i)) % 251)
	}
	expectedHash := sha256.Sum256(expected)
	if err != nil || written != 19 || !bytes.Equal(destination.Bytes(), expected) || digest != hex.EncodeToString(expectedHash[:]) || source.payloadReads != 3 {
		t.Fatalf("large-offset streaming: written=%d bytes=%x digest=%s reads=%d err=%v", written, destination.Bytes(), digest, source.payloadReads, err)
	}
}

type writerFunc func([]byte) (int, error)

func (f writerFunc) Write(data []byte) (int, error) { return f(data) }

func TestSafetensorsStreamingFailuresAndCancellation(t *testing.T) {
	index := open(t, `{"x":{"dtype":"U8","shape":[6],"data_offsets":[0,6]}}`, []byte{0, 1, 2, 3, 4, 5})
	cases := []struct {
		name   string
		writer io.Writer
		count  int64
	}{
		{"short_write", writerFunc(func(p []byte) (int, error) { return len(p) - 1, nil }), 1},
		{"writer_error", writerFunc(func([]byte) (int, error) { return 1, io.ErrClosedPipe }), 1},
		{"negative_count", writerFunc(func([]byte) (int, error) { return -1, nil }), 0},
		{"excess_count", writerFunc(func(p []byte) (int, error) { return len(p) + 1, nil }), 0},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			written, digest, err := index.CopyTensor(context.Background(), "x", test.writer, make([]byte, 2))
			if err == nil || digest != "" || written != test.count {
				t.Fatalf("failure receipt: written=%d digest=%q err=%v", written, digest, err)
			}
		})
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if written, digest, err := index.CopyTensor(ctx, "x", io.Discard, make([]byte, 2)); written != 0 || digest != "" || !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled copy: %d %q %v", written, digest, err)
	}
	ctx, cancel = context.WithCancel(context.Background())
	writer := writerFunc(func(p []byte) (int, error) { cancel(); return len(p), nil })
	if written, digest, err := index.CopyTensor(ctx, "x", writer, make([]byte, 2)); written != 2 || digest != "" || !errors.Is(err, context.Canceled) {
		t.Fatalf("midstream cancellation: %d %q %v", written, digest, err)
	}
	for _, arguments := range []struct {
		ctx    context.Context
		writer io.Writer
		buffer []byte
	}{
		{nil, io.Discard, make([]byte, 2)}, {context.Background(), nil, make([]byte, 2)},
		{context.Background(), io.Discard, nil},
		{context.Background(), io.Discard, make([]byte, checkpoint.DefaultLimits().MaxChunkBytes+1)},
	} {
		if _, _, err := index.CopyTensor(arguments.ctx, "x", arguments.writer, arguments.buffer); err == nil {
			t.Fatal("invalid streaming arguments accepted")
		}
	}
	if _, err := index.HashTensor(context.Background(), "missing", make([]byte, 1)); err == nil {
		t.Fatal("missing tensor hashed")
	}
	// Opening cannot validate unread payload availability. Streaming must surface
	// a truncated source and must not publish a digest for the partial tensor.
	header := `{"x":{"dtype":"U8","shape":[6],"data_offsets":[0,6]}}`
	data := container(header, []byte{0, 1, 2})
	truncated, err := checkpoint.OpenSafetensors(bytes.NewReader(data), int64(len(data)+3), checkpoint.DefaultLimits())
	if err != nil {
		t.Fatal(err)
	}
	if written, digest, err := truncated.CopyTensor(context.Background(), "x", io.Discard, make([]byte, 2)); written != 2 || digest != "" || !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("truncated stream: %d %q %v", written, digest, err)
	}
}

func TestSafetensorsConcurrentIndependentReaders(t *testing.T) {
	index := open(t, `{"x":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}`, []byte{1, 2, 3, 4})
	expected, err := index.HashTensor(context.Background(), "x", make([]byte, 3))
	if err != nil {
		t.Fatal(err)
	}
	var workers sync.WaitGroup
	for range 8 {
		workers.Go(func() {
			for range 8 {
				actual, err := index.HashTensor(context.Background(), "x", make([]byte, 1))
				if err != nil || actual != expected {
					t.Errorf("concurrent hash: %q %v", actual, err)
				}
				index.Metadata()["unshared"] = "value"
				index.Tensors()[0].Shape[0] = 999
			}
		})
	}
	workers.Wait()
}
