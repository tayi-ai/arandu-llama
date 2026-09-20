//go:build libtorch && cgo

package loading_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"math"
	"reflect"
	"testing"

	"github.com/tayi-ai/arandu-llama/checkpoint"
	"github.com/tayi-ai/arandu-llama/training/loading"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

const halfDigest = "2e1d6038edb9c7c82987074a9cb292c656a23deef19334b2c79dda64d49bf4fe"
const brainDigest = "8c2c6a65c7c224cc1521e4272dd00b0b47a70a11dc5819e2e193ed41207fef58"

func fixture(t *testing.T, dtype string) ([]byte, int64) {
	t.Helper()
	hexPayload := "003c00c000380000"
	if dtype == "BF16" {
		hexPayload = "803f00c0003f0000"
	}
	payload, err := hex.DecodeString(hexPayload)
	if err != nil {
		t.Fatal(err)
	}
	header := fmt.Sprintf(`{"weight":{"dtype":%q,"shape":[2,2],"data_offsets":[0,8]},"unused":{"dtype":"F16","shape":[2],"data_offsets":[8,12]}}`, dtype)
	return container(header, append(payload, 1, 2, 3, 4)), int64(8 + len(header))
}

func container(header string, payload []byte) []byte {
	data := make([]byte, 8+len(header)+len(payload))
	binary.LittleEndian.PutUint64(data, uint64(len(header)))
	copy(data[8:], header)
	copy(data[8+len(header):], payload)
	return data
}

func options(dtype, digest string) loading.Options {
	limits := checkpoint.DefaultLimits()
	limits.MaxChunkBytes = 3
	return loading.Options{HeaderLimits: limits, BudgetBytes: 48, DType: torch.Float32, Device: torch.CPUDevice(),
		Expected: &loading.Expectation{Shape: []int64{2, 2}, DType: dtype, SHA256: digest}}
}

type observedReader struct {
	source            io.ReaderAt
	payloadOffset     int64
	reads             int
	payloadReads      int
	payloadBytes      int
	maxPayloadRequest int
	highestEnd        int64
	onRead            func(int64)
}

func (reader *observedReader) ReadAt(destination []byte, offset int64) (int, error) {
	reader.reads++
	n, err := reader.source.ReadAt(destination, offset)
	if offset >= reader.payloadOffset {
		reader.payloadReads++
		reader.payloadBytes += n
		reader.maxPayloadRequest = max(reader.maxPayloadRequest, len(destination))
	}
	reader.highestEnd = max(reader.highestEnd, offset+int64(n))
	if reader.onRead != nil {
		reader.onRead(offset)
	}
	return n, err
}

func TestLoadConvertsKnownHalfAndBrainTensorsWithBoundedReads(t *testing.T) {
	for _, test := range []struct{ dtype, digest string }{{"F16", halfDigest}, {"BF16", brainDigest}} {
		t.Run(test.dtype, func(t *testing.T) {
			data, payloadOffset := fixture(t, test.dtype)
			before := bytes.Clone(data)
			reader := &observedReader{source: bytes.NewReader(data), payloadOffset: payloadOffset}
			tensor, receipt, err := loading.LoadTensor(context.Background(), reader, int64(len(data)), "weight", options(test.dtype, test.digest))
			if err != nil {
				t.Fatal(err)
			}
			defer tensor.Close()
			values, err := tensor.Float32Values()
			if err != nil || !reflect.DeepEqual(values, []float32{1, -2, .5, 0}) {
				t.Fatalf("values:%v err:%v", values, err)
			}
			if !receipt.Loaded || receipt.SourceBytes != 8 || receipt.TargetBytes != 16 || receipt.AdmittedCopyBytes != 48 || receipt.BytesRead != 8 || receipt.ReadChunks != 3 || receipt.SHA256 != test.digest {
				t.Fatalf("receipt:%+v", receipt)
			}
			if reader.reads != 5 || reader.payloadReads != 3 || reader.maxPayloadRequest != 3 || reader.highestEnd != payloadOffset+8 {
				t.Fatalf("unbounded or unrelated payload read:%+v", reader)
			}
			if !bytes.Equal(before, data) {
				t.Fatal("checkpoint bytes changed")
			}
			info, err := tensor.Info()
			if err != nil || info.DType != torch.Float32 || info.Device != torch.CPUDevice() || info.RequiresGrad || !reflect.DeepEqual(info.Shape, []int64{2, 2}) {
				t.Fatalf("native placement:%+v %v", info, err)
			}
			// The native tensor must own its bytes independently of the caller's source.
			data[payloadOffset] = 255
			values, err = tensor.Float32Values()
			if err != nil || values[0] != 1 {
				t.Fatalf("native tensor aliases source:%v %v", values, err)
			}
		})
	}
}

func TestLoadMetadataAndBudgetRejectionsNeverReadPayload(t *testing.T) {
	cases := []struct {
		name     string
		change   func(*loading.Options)
		sentinel error
	}{
		{"budget", func(o *loading.Options) { o.BudgetBytes = 47 }, loading.ErrBudget},
		{"zero_budget", func(o *loading.Options) { o.BudgetBytes = 0 }, loading.ErrBudget},
		{"shape", func(o *loading.Options) { o.Expected.Shape = []int64{4} }, loading.ErrExpectation},
		{"scalar_shape", func(o *loading.Options) { o.Expected.Shape = []int64{} }, loading.ErrExpectation},
		{"dtype", func(o *loading.Options) { o.Expected.DType = "BF16" }, loading.ErrExpectation},
		{"malformed_digest", func(o *loading.Options) { o.Expected.SHA256 = "invalid" }, loading.ErrExpectation},
		{"missing_device", func(o *loading.Options) { o.Device = torch.Device{} }, nil},
		{"invalid_device", func(o *loading.Options) { o.Device = torch.CUDADevice(-1) }, nil},
		{"unknown_target_dtype", func(o *loading.Options) { o.DType = 99 }, nil},
		{"integer_grad", func(o *loading.Options) { o.DType = torch.Int64; o.RequiresGrad = true }, nil},
		{"invalid_header_limits", func(o *loading.Options) { o.HeaderLimits.MaxChunkBytes = 0 }, nil},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			data, offset := fixture(t, "F16")
			reader := &observedReader{source: bytes.NewReader(data), payloadOffset: offset}
			option := options("F16", halfDigest)
			test.change(&option)
			tensor, receipt, err := loading.LoadTensor(context.Background(), reader, int64(len(data)), "weight", option)
			if err == nil || tensor != nil || receipt.Loaded || reader.payloadReads != 0 {
				t.Fatalf("invalid admission reached payload: tensor%v receipt%+v reader%+v err%v", tensor, receipt, reader, err)
			}
			if test.sentinel != nil && !errors.Is(err, test.sentinel) {
				t.Fatalf("wrong rejection: %v", err)
			}
		})
	}
}

func TestLoadRejectsTamperedPayloadAndTruncation(t *testing.T) {
	data, offset := fixture(t, "F16")
	data[offset] ^= 1
	tensor, receipt, err := loading.LoadTensor(context.Background(), bytes.NewReader(data), int64(len(data)), "weight", options("F16", halfDigest))
	if !errors.Is(err, loading.ErrExpectation) || tensor != nil || receipt.Loaded || receipt.SHA256 == "" || receipt.SHA256 == halfDigest {
		t.Fatalf("tamper accepted: %+v %v", receipt, err)
	}
	data, offset = fixture(t, "F16")
	tensor, receipt, err = loading.LoadTensor(context.Background(), bytes.NewReader(data[:offset+7]), int64(len(data)), "weight", options("F16", halfDigest))
	if !errors.Is(err, io.ErrUnexpectedEOF) || tensor != nil || receipt.Loaded || receipt.SHA256 != "" || receipt.BytesRead != 7 {
		t.Fatalf("truncation: %+v %v", receipt, err)
	}
	// A valid different header interpretation must still fail an explicit dtype check.
	brain, _ := fixture(t, "BF16")
	tensor, receipt, err = loading.LoadTensor(context.Background(), bytes.NewReader(brain), int64(len(brain)), "weight", options("F16", brainDigest))
	if !errors.Is(err, loading.ErrExpectation) || tensor != nil || receipt.BytesRead != 0 {
		t.Fatalf("dtype tamper: %+v %v", receipt, err)
	}
}

func TestLoadCancellationBeforeAndBetweenChunks(t *testing.T) {
	for _, phase := range []string{"before_header", "after_header", "after_first_chunk"} {
		t.Run(phase, func(t *testing.T) {
			data, offset := fixture(t, "F16")
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			reader := &observedReader{source: bytes.NewReader(data), payloadOffset: offset}
			if phase == "before_header" {
				cancel()
			}
			reader.onRead = func(position int64) {
				if phase == "after_header" && position == 8 || phase == "after_first_chunk" && position == offset {
					cancel()
				}
			}
			tensor, receipt, err := loading.LoadTensor(ctx, reader, int64(len(data)), "weight", options("F16", halfDigest))
			if !errors.Is(err, context.Canceled) || tensor != nil || receipt.Loaded {
				t.Fatalf("cancellation: %+v %v", receipt, err)
			}
			if phase == "before_header" && reader.reads != 0 {
				t.Fatal("canceled call read header")
			}
			if phase == "after_header" && reader.payloadReads != 0 {
				t.Fatal("canceled header reached payload")
			}
			if phase == "after_first_chunk" && (reader.payloadReads != 1 || receipt.BytesRead != 3 || receipt.SHA256 != "") {
				t.Fatalf("canceled chunks: %+v %+v", reader, receipt)
			}
		})
	}
}

func TestLoadRequiresGradCreatesIndependentLeaf(t *testing.T) {
	data, _ := fixture(t, "F16")
	option := options("F16", halfDigest)
	option.RequiresGrad = true
	tensor, _, err := loading.LoadTensor(context.Background(), bytes.NewReader(data), int64(len(data)), "weight", option)
	if err != nil {
		t.Fatal(err)
	}
	defer tensor.Close()
	info, err := tensor.Info()
	if err != nil || !info.RequiresGrad {
		t.Fatalf("missing leaf flag: %+v %v", info, err)
	}
	squared, err := tensor.Mul(tensor)
	if err != nil {
		t.Fatal(err)
	}
	defer squared.Close()
	seed, err := torch.FromFloat32([]float32{1, 1, 1, 1}, []int64{2, 2}, torch.CPUDevice(), false)
	if err != nil {
		t.Fatal(err)
	}
	defer seed.Close()
	gradients, err := torch.Grad([]*torch.Tensor{squared}, []*torch.Tensor{tensor}, []*torch.Tensor{seed}, false, false)
	if err != nil {
		t.Fatal(err)
	}
	defer gradients[0].Close()
	values, err := gradients[0].Float32Values()
	if err != nil || !reflect.DeepEqual(values, []float32{2, -4, 1, 0}) {
		t.Fatalf("loaded leaf gradient:%v %v", values, err)
	}
	leaf, err := tensor.SetRequiresGrad(false)
	if err != nil {
		t.Fatalf("loaded tensor is not a leaf:%v", err)
	}
	defer leaf.Close()
}

func TestLoadOptionalExpectationsScalarsAndEmptyTensors(t *testing.T) {
	data, _ := fixture(t, "F16")
	option := options("F16", halfDigest)
	option.Expected = nil
	tensor, receipt, err := loading.LoadTensor(context.Background(), bytes.NewReader(data), int64(len(data)), "weight", option)
	if err != nil {
		t.Fatal(err)
	}
	tensor.Close()
	if receipt.SHA256 != halfDigest {
		t.Fatal("optional expectations skipped content hash")
	}
	for _, test := range []struct {
		header  string
		payload []byte
		shape   []int64
		bytes   int64
	}{
		{`{"weight":{"dtype":"F16","shape":[],"data_offsets":[0,2]}}`, []byte{0, 0x3c}, []int64{}, 12},
		{`{"weight":{"dtype":"F16","shape":[0,3],"data_offsets":[0,0]}}`, nil, []int64{0, 3}, 1},
	} {
		data := container(test.header, test.payload)
		option.Expected = &loading.Expectation{Shape: test.shape, DType: "F16"}
		option.BudgetBytes = test.bytes
		tensor, receipt, err := loading.LoadTensor(context.Background(), bytes.NewReader(data), int64(len(data)), "weight", option)
		if err != nil {
			t.Fatal(err)
		}
		tensor.Close()
		if !receipt.Loaded || !reflect.DeepEqual(receipt.Shape, test.shape) {
			t.Fatalf("scalar/empty receipt:%+v", receipt)
		}
	}
}

func TestLoadRejectsUnaddressableCopyBeforeReadingPayload(t *testing.T) {
	for _, test := range []struct {
		dtype                 string
		elements, sourceBytes int64
		target                torch.DType
	}{
		{"F16", 1_000_000_000_000_000_000, 2_000_000_000_000_000_000, torch.Float64},
		{"BOOL", 1_300_000_000_000_000_000, 1_300_000_000_000_000_000, torch.Float64},
	} {
		header := fmt.Sprintf(`{"weight":{"dtype":%q,"shape":[%d],"data_offsets":[0,%d]}}`, test.dtype, test.elements, test.sourceBytes)
		prefix := container(header, nil)
		reader := &observedReader{source: bytes.NewReader(prefix), payloadOffset: int64(len(prefix))}
		option := options("", "")
		option.Expected = nil
		option.BudgetBytes = math.MaxInt64
		option.DType = test.target
		tensor, receipt, err := loading.LoadTensor(context.Background(), reader, int64(len(prefix))+test.sourceBytes, "weight", option)
		if !errors.Is(err, loading.ErrBudget) || tensor != nil || receipt.Loaded || reader.payloadReads != 0 {
			t.Fatalf("overflow not rejected:%+v %v", receipt, err)
		}
	}
}

func TestLoadRejectsMissingAndUnsupportedTensors(t *testing.T) {
	data, _ := fixture(t, "F16")
	if tensor, _, err := loading.LoadTensor(context.Background(), bytes.NewReader(data), int64(len(data)), "missing", options("F16", halfDigest)); err == nil || tensor != nil {
		t.Fatal("missing tensor accepted")
	}
	data = container(`{"weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}`, []byte{1})
	option := options("", "")
	option.Expected = nil
	if tensor, receipt, err := loading.LoadTensor(context.Background(), bytes.NewReader(data), int64(len(data)), "weight", option); err == nil || tensor != nil || receipt.BytesRead != 0 {
		t.Fatalf("unsupported native dtype:%+v %v", receipt, err)
	}
}
