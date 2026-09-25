//go:build libtorch && cgo && darwin

package loading_test

import (
	"bytes"
	"context"
	"reflect"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/loading"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

func TestLoadSafetensorsToMPSWithSourceDigest(t *testing.T) {
	if !torch.MPSAvailable() {
		t.Skip("MPS device is unavailable")
	}
	data, _ := fixture(t, "F16")
	admission := options("F16", halfDigest)
	admission.Device = torch.MPSDevice()
	value, receipt, err := loading.LoadTensor(context.Background(), bytes.NewReader(data), int64(len(data)), "weight", admission)
	if err != nil {
		t.Fatal(err)
	}
	defer value.Close()
	info, err := value.Info()
	if err != nil || info.Device != torch.MPSDevice() || info.DType != torch.Float32 || !receipt.Loaded || receipt.SHA256 != halfDigest {
		t.Fatalf("MPS tensor or receipt differs: %+v %+v %v", info, receipt, err)
	}
	actual, err := value.Float32Values()
	if err != nil || !reflect.DeepEqual(actual, []float32{1, -2, .5, 0}) {
		t.Fatalf("MPS values differ: %v %v", actual, err)
	}
	admission.Device = torch.Device{Kind: "mps", Index: 1}
	if result, _, err := loading.LoadTensor(context.Background(), bytes.NewReader(data), int64(len(data)), "weight", admission); result != nil || err == nil {
		t.Fatal("indexed MPS device was accepted")
	}
}
