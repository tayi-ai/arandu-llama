//go:build libtorch && cgo && !libtorch_cuda

package assembly_test

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/ornith"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

func TestCompleteCPUInitialSetIsBorrowedAndGPUBackendMustBeExplicit(t *testing.T) {
	initial, err := ornith.InitializeAdapter(context.Background(), ornith.InitialAdapterSpec{Seed: 83, PreludeBlocks: 24, ExpectedSHA256: initialDigest})
	if err != nil {
		t.Fatal(err)
	}
	defer initial.Close()
	d := fixture()
	content := map[string]string{}
	for _, parameter := range initial.Parameters {
		raw, err := parameter.Value.Bytes()
		if err != nil {
			t.Fatal(err)
		}
		content[parameter.Name] = fmt.Sprintf("%x", sha256.Sum256(raw))
	}
	for i := range d.Reference {
		if hash, found := content[d.Reference[i].Name]; found {
			d.Reference[i].Hash = hash
		}
	}
	p := plan(t, d)
	source := provider(t, p, nil)
	model, err := ornith.LoadTextAssembly(context.Background(), p, source, initial)
	if model != nil || !errors.Is(err, torch.ErrCUDAUnavailable) || source.opens != 0 {
		t.Fatalf("CPU build reached GPU placement: %v", err)
	}
	for _, parameter := range initial.Parameters {
		info, err := parameter.Value.Info()
		if err != nil || info.Device != torch.CPUDevice() || !info.RequiresGrad {
			t.Fatalf("borrowed initial tensor changed: %+v %v", info, err)
		}
		raw, err := parameter.Value.Bytes()
		if err != nil || fmt.Sprintf("%x", sha256.Sum256(raw)) != content[parameter.Name] {
			t.Fatal("borrowed initial content changed")
		}
	}
	// Same geometry and aggregate label cannot override a required per-tensor hash.
	for i := range d.Reference {
		if d.Reference[i].Grad {
			d.Reference[i].Hash = fmt.Sprintf("%064x", 1)
			break
		}
	}
	p = plan(t, d)
	source = provider(t, p, nil)
	if model, err := ornith.LoadTextAssembly(context.Background(), p, source, initial); model != nil || !errors.Is(err, ornith.ErrAssembly) || source.opens != 0 {
		t.Fatalf("adapter hash bypass:%v", err)
	}
}
