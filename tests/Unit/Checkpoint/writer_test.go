package checkpoint_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"io"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/tayi-ai/arandu-llama/checkpoint"
)

func float32Fixture() []checkpoint.Float32Tensor {
	return []checkpoint.Float32Tensor{
		{Name: "zeta", Values: []float32{math.Float32frombits(0x80000000)}},
		{Name: "empty", Shape: []uint64{0, 7}},
		{Name: "alpha", Shape: []uint64{2, 2}, Values: []float32{0, math.Float32frombits(0x80000000), math.SmallestNonzeroFloat32, math.MaxFloat32}},
	}
}

func float32Bytes(values []float32) []byte {
	data := make([]byte, 4*len(values))
	for index, value := range values {
		binary.LittleEndian.PutUint32(data[index*4:], math.Float32bits(value))
	}
	return data
}

func writeFloat32(t *testing.T, tensors []checkpoint.Float32Tensor, limits checkpoint.Limits) ([]byte, checkpoint.WriteReceipt) {
	t.Helper()
	var destination bytes.Buffer
	receipt, err := checkpoint.WriteFloat32(context.Background(), &destination, tensors, limits)
	if err != nil {
		t.Fatal(err)
	}
	data := destination.Bytes()
	digest := sha256.Sum256(data)
	if receipt.SHA256 != hex.EncodeToString(digest[:]) || receipt.Bytes != int64(len(data)) || receipt.Tensors != len(tensors) ||
		receipt.HeaderBytes%8 != 0 || receipt.Bytes != 8+receipt.HeaderBytes+receipt.PayloadBytes {
		t.Fatalf("invalid complete receipt: %+v", receipt)
	}
	return data, receipt
}

func TestFloat32WriterGoldenRoundTripPreservesBitsAndInputOrder(t *testing.T) {
	tensors := float32Fixture()
	data, receipt := writeFloat32(t, tensors, checkpoint.DefaultLimits())
	if receipt.PayloadBytes != 20 {
		t.Fatalf("payload size: %d", receipt.PayloadBytes)
	}
	header := `{"__metadata__":{"format":"pt"},"alpha":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"empty":{"dtype":"F32","shape":[0,7],"data_offsets":[16,16]},"zeta":{"dtype":"F32","shape":[],"data_offsets":[16,20]}}`
	header += strings.Repeat(" ", (8-len(header)%8)%8)
	if binary.LittleEndian.Uint64(data[:8]) != uint64(len(header)) || string(data[8:8+len(header)]) != header {
		t.Fatalf("noncanonical header: %q", data[8:8+receipt.HeaderBytes])
	}
	index, err := checkpoint.OpenSafetensors(bytes.NewReader(data), int64(len(data)), checkpoint.DefaultLimits())
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(index.Metadata(), map[string]string{"format": "pt"}) {
		t.Fatalf("metadata: %v", index.Metadata())
	}
	for _, expected := range tensors {
		tensor, found := index.Tensor(expected.Name)
		if !found || tensor.DType != "F32" || !reflect.DeepEqual(tensor.Shape, append([]uint64{}, expected.Shape...)) {
			t.Fatalf("descriptor mismatch: %+v", tensor)
		}
		var restored bytes.Buffer
		if _, _, err := index.CopyTensor(context.Background(), expected.Name, &restored, make([]byte, 3)); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(restored.Bytes(), float32Bytes(expected.Values)) {
			t.Fatalf("changed raw bits for %q: %x", expected.Name, restored.Bytes())
		}
	}
	if !reflect.DeepEqual(tensors, float32Fixture()) {
		t.Fatal("writer changed caller names, shapes, values or tensor order")
	}
	reordered := []checkpoint.Float32Tensor{tensors[2], tensors[0], tensors[1]}
	again, secondReceipt := writeFloat32(t, reordered, checkpoint.DefaultLimits())
	if !bytes.Equal(again, data) || secondReceipt != receipt {
		t.Fatal("input permutation changed the container or receipt")
	}
}

func TestFloat32WriterEmptyContainerUnicodeAndEscapedNames(t *testing.T) {
	for _, tensors := range [][]checkpoint.Float32Tensor{
		nil,
		{
			{Name: "<&>\u2028\u2029🌱\x00\n\"\\", Shape: []uint64{0, math.MaxUint64}},
			{Name: "é", Shape: []uint64{}, Values: []float32{1}},
			{Name: "e\u0301", Shape: []uint64{0}},
		},
	} {
		data, _ := writeFloat32(t, tensors, checkpoint.DefaultLimits())
		index, err := checkpoint.OpenSafetensors(bytes.NewReader(data), int64(len(data)), checkpoint.DefaultLimits())
		if err != nil {
			t.Fatal(err)
		}
		if len(index.Tensors()) != len(tensors) {
			t.Fatal("tensor names were normalized or lost")
		}
		for _, tensor := range tensors {
			if _, found := index.Tensor(tensor.Name); !found {
				t.Fatalf("name was changed: %q", tensor.Name)
			}
		}
	}
}

func TestFloat32WriterRejectsInvalidInputsBeforeAnyWrite(t *testing.T) {
	valid := checkpoint.Float32Tensor{Name: "valid", Shape: []uint64{1}, Values: []float32{1}}
	cases := []struct {
		name   string
		tensor checkpoint.Float32Tensor
	}{
		{"empty_name", checkpoint.Float32Tensor{Values: []float32{1}}},
		{"reserved_name", checkpoint.Float32Tensor{Name: "__metadata__", Values: []float32{1}}},
		{"invalid_utf8", checkpoint.Float32Tensor{Name: "\xff", Values: []float32{1}}},
		{"duplicate", valid},
		{"missing_scalar", checkpoint.Float32Tensor{Name: "z"}},
		{"shape_mismatch", checkpoint.Float32Tensor{Name: "z", Shape: []uint64{2}, Values: []float32{1}}},
		{"nonempty_zero_shape", checkpoint.Float32Tensor{Name: "z", Shape: []uint64{0}, Values: []float32{1}}},
		{"shape_overflow", checkpoint.Float32Tensor{Name: "z", Shape: []uint64{math.MaxUint64, 2}}},
		{"bit_length_overflow", checkpoint.Float32Tensor{Name: "z", Shape: []uint64{math.MaxUint64/32 + 1}}},
		{"positive_infinity", checkpoint.Float32Tensor{Name: "z", Values: []float32{float32(math.Inf(1))}}},
		{"negative_infinity", checkpoint.Float32Tensor{Name: "z", Values: []float32{float32(math.Inf(-1))}}},
		{"quiet_nan", checkpoint.Float32Tensor{Name: "z", Values: []float32{math.Float32frombits(0x7fc00001)}}},
		{"signaling_nan", checkpoint.Float32Tensor{Name: "z", Values: []float32{math.Float32frombits(0x7f800001)}}},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			var destination bytes.Buffer
			receipt, err := checkpoint.WriteFloat32(context.Background(), &destination, []checkpoint.Float32Tensor{valid, test.tensor}, checkpoint.DefaultLimits())
			if err == nil || receipt != (checkpoint.WriteReceipt{}) || destination.Len() != 0 {
				t.Fatalf("invalid tensor wrote bytes or claimed success: %+v %v", receipt, err)
			}
		})
	}
	if receipt, err := checkpoint.WriteFloat32(nil, io.Discard, nil, checkpoint.DefaultLimits()); err == nil || receipt != (checkpoint.WriteReceipt{}) {
		t.Fatalf("nil context accepted: %+v %v", receipt, err)
	}
	if receipt, err := checkpoint.WriteFloat32(context.Background(), nil, nil, checkpoint.DefaultLimits()); err == nil || receipt != (checkpoint.WriteReceipt{}) {
		t.Fatalf("nil destination accepted: %+v %v", receipt, err)
	}
}

func TestFloat32WriterLimitsAndExactPaddedHeaderBoundary(t *testing.T) {
	_, original := writeFloat32(t, float32Fixture(), checkpoint.DefaultLimits())
	for name, change := range map[string]func(*checkpoint.Limits){
		"header":          func(l *checkpoint.Limits) { l.MaxHeaderBytes = original.HeaderBytes - 1 },
		"tensors":         func(l *checkpoint.Limits) { l.MaxTensors = 2 },
		"dimensions":      func(l *checkpoint.Limits) { l.MaxDimensions = 1 },
		"zero_header":     func(l *checkpoint.Limits) { l.MaxHeaderBytes = 0 },
		"huge_header":     func(l *checkpoint.Limits) { l.MaxHeaderBytes = 100_000_001 },
		"zero_tensors":    func(l *checkpoint.Limits) { l.MaxTensors = 0 },
		"zero_dimensions": func(l *checkpoint.Limits) { l.MaxDimensions = 0 },
		"zero_metadata":   func(l *checkpoint.Limits) { l.MaxMetadataEntries = 0 },
		"zero_chunk":      func(l *checkpoint.Limits) { l.MaxChunkBytes = 0 },
	} {
		t.Run(name, func(t *testing.T) {
			limits := checkpoint.DefaultLimits()
			change(&limits)
			var destination bytes.Buffer
			if receipt, err := checkpoint.WriteFloat32(context.Background(), &destination, float32Fixture(), limits); err == nil || destination.Len() != 0 || receipt != (checkpoint.WriteReceipt{}) {
				t.Fatalf("budget not enforced before writing: %+v %v", receipt, err)
			}
		})
	}
	limits := checkpoint.DefaultLimits()
	limits.MaxHeaderBytes, limits.MaxTensors, limits.MaxDimensions, limits.MaxMetadataEntries = original.HeaderBytes, 3, 2, 1
	writeFloat32(t, float32Fixture(), limits)
	limits.MaxHeaderBytes = 80
	var destination bytes.Buffer
	tensors := []checkpoint.Float32Tensor{{Name: strings.Repeat("\x00", 20), Values: []float32{1}}}
	if receipt, err := checkpoint.WriteFloat32(context.Background(), &destination, tensors, limits); err == nil || destination.Len() != 0 || receipt != (checkpoint.WriteReceipt{}) {
		t.Fatalf("escaped-name expansion bypassed header budget: %+v %v", receipt, err)
	}
}

func TestFloat32WriterBoundsEveryWriteAndStreamsLargePayload(t *testing.T) {
	for _, chunk := range []int{1, 2, 3, 4, 5, 7, 8, 17, 1 << 20} {
		limits := checkpoint.DefaultLimits()
		limits.MaxChunkBytes = chunk
		tensors := float32Fixture()
		if chunk == 1<<20 {
			values := make([]float32, 32769)
			for index := range values {
				values[index] = math.Float32frombits(uint32(index + 1))
			}
			tensors = []checkpoint.Float32Tensor{{Name: "large", Shape: []uint64{uint64(len(values))}, Values: values}}
		}
		var destination bytes.Buffer
		calls, largest := 0, 0
		writer := writerFunc(func(data []byte) (int, error) {
			calls++
			largest = max(largest, len(data))
			return destination.Write(data)
		})
		receipt, err := checkpoint.WriteFloat32(context.Background(), writer, tensors, limits)
		if err != nil || largest > min(chunk, 64<<10) || calls < 3 {
			t.Fatalf("chunk %d: calls=%d largest=%d receipt=%+v err=%v", chunk, calls, largest, receipt, err)
		}
		expected, _ := writeFloat32(t, tensors, checkpoint.DefaultLimits())
		if !bytes.Equal(destination.Bytes(), expected) {
			t.Fatalf("chunk size %d changed the file", chunk)
		}
	}
}

func TestFloat32WriterFailuresNeverReturnACompleteReceipt(t *testing.T) {
	for name, test := range map[string]struct {
		writer io.Writer
		bytes  int64
		error  error
	}{
		"short":          {writerFunc(func(data []byte) (int, error) { return len(data) - 1, nil }), 7, io.ErrShortWrite},
		"zero":           {writerFunc(func([]byte) (int, error) { return 0, nil }), 0, io.ErrShortWrite},
		"partial_error":  {writerFunc(func([]byte) (int, error) { return 2, io.ErrClosedPipe }), 2, io.ErrClosedPipe},
		"complete_error": {writerFunc(func(data []byte) (int, error) { return len(data), io.ErrClosedPipe }), 8, io.ErrClosedPipe},
		"negative_count": {writerFunc(func([]byte) (int, error) { return -1, nil }), 0, nil},
		"excess_count":   {writerFunc(func(data []byte) (int, error) { return len(data) + 1, nil }), 0, nil},
	} {
		t.Run(name, func(t *testing.T) {
			receipt, err := checkpoint.WriteFloat32(context.Background(), test.writer, float32Fixture(), checkpoint.DefaultLimits())
			if err == nil || receipt != (checkpoint.WriteReceipt{Bytes: test.bytes}) || test.error != nil && !errors.Is(err, test.error) {
				t.Fatalf("invalid failure receipt: %+v %v", receipt, err)
			}
		})
	}
	_, complete := writeFloat32(t, float32Fixture(), checkpoint.DefaultLimits())
	accepted := int64(0)
	writer := writerFunc(func(data []byte) (int, error) {
		if accepted >= 8+complete.HeaderBytes {
			accepted += 2
			return 2, io.ErrClosedPipe
		}
		accepted += int64(len(data))
		return len(data), nil
	})
	receipt, err := checkpoint.WriteFloat32(context.Background(), writer, float32Fixture(), checkpoint.DefaultLimits())
	if !errors.Is(err, io.ErrClosedPipe) || accepted != 8+complete.HeaderBytes+2 || receipt != (checkpoint.WriteReceipt{Bytes: accepted}) {
		t.Fatalf("partial payload receipt: %+v accepted=%d err=%v", receipt, accepted, err)
	}
}

func TestFloat32WriterCancellationBeforeDuringAndAfterLastWrite(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	var destination bytes.Buffer
	if receipt, err := checkpoint.WriteFloat32(ctx, &destination, float32Fixture(), checkpoint.DefaultLimits()); !errors.Is(err, context.Canceled) || receipt != (checkpoint.WriteReceipt{}) || destination.Len() != 0 {
		t.Fatalf("preflight cancellation: %+v %v", receipt, err)
	}
	_, complete := writeFloat32(t, float32Fixture(), checkpoint.DefaultLimits())
	for _, cancelAt := range []int64{8, 8 + complete.HeaderBytes + 4, complete.Bytes} {
		ctx, cancel := context.WithCancel(context.Background())
		accepted := int64(0)
		writer := writerFunc(func(data []byte) (int, error) {
			accepted += int64(len(data))
			if accepted >= cancelAt {
				cancel()
			}
			return len(data), nil
		})
		receipt, err := checkpoint.WriteFloat32(ctx, writer, float32Fixture(), checkpoint.DefaultLimits())
		cancel()
		if !errors.Is(err, context.Canceled) || accepted < cancelAt || receipt != (checkpoint.WriteReceipt{Bytes: accepted}) {
			t.Fatalf("cancel at %d: receipt=%+v accepted=%d err=%v", cancelAt, receipt, accepted, err)
		}
	}
}
