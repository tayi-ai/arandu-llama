//go:build libtorch && cgo

package initialadapter_test

import (
	"context"
	"errors"
	fixture "github.com/tayi-ai/arandu-llama/tests/Unit/Training/Fixture"
	"github.com/tayi-ai/arandu-llama/training/decoder"
	"github.com/tayi-ai/arandu-llama/training/torch"
	"strings"
	"testing"
)

func smallSpec(t *testing.T) decoder.InitialAdapterSpec {
	s := decoder.InitialAdapterSpec{Seed: 7, PreludeBlocks: 2, PreludeWidth: 3, PreludeLow: -.25, PreludeHigh: .5, Projections: []decoder.InitialProjection{{Name: "synthetic.layer.0.query", Input: 6, Output: 8, Rank: 2}, {Name: "synthetic.layer.1.value", Input: 8, Output: 4, Rank: 3}}}
	s.ExpectedSHA256 = fixture.Digest(t, s)
	return s
}

func TestArbitraryProjectionGeometryMatchesIndependentCPUReference(t *testing.T) {
	for _, s := range []decoder.InitialAdapterSpec{smallSpec(t), fixture.Spec(t)} {
		a, err := decoder.InitializeAdapter(context.Background(), s)
		if err != nil {
			t.Fatal(err)
		}
		if a.SHA256 != s.ExpectedSHA256 || len(a.Parameters) != 2*len(s.Projections) {
			t.Fatal("identity or count differs")
		}
		for i, p := range a.Parameters {
			info, err := p.Value.Info()
			if err != nil || !info.RequiresGrad || info.DType != torch.Float32 || info.Device != torch.CPUDevice() {
				t.Fatalf("trainable CPU leaf differs: %v", err)
			}
			b, err := p.Value.Bytes()
			if err != nil {
				t.Fatal(err)
			}
			if i%2 == 1 {
				for _, v := range b {
					if v != 0 {
						t.Fatal("B must be exact positive zero")
					}
				}
			}
		}
		handles := append([]decoder.InitialParameter(nil), a.Parameters...)
		if err := a.Close(); err != nil {
			t.Fatal(err)
		}
		if err := a.Close(); err != nil {
			t.Fatal(err)
		}
		for _, p := range handles {
			if _, err := p.Value.Info(); !errors.Is(err, torch.ErrClosed) {
				t.Fatal("owned leaf remained alive")
			}
		}
	}
}

func TestInitializerRejectsChangedScheduleAndInvalidGeometry(t *testing.T) {
	for _, change := range []func(*decoder.InitialAdapterSpec){func(s *decoder.InitialAdapterSpec) { s.Seed++ }, func(s *decoder.InitialAdapterSpec) { s.PreludeBlocks = 0 }, func(s *decoder.InitialAdapterSpec) { s.Projections[0].Rank++ }, func(s *decoder.InitialAdapterSpec) { s.ExpectedSHA256 = strings.Repeat("0", 64) }} {
		s := smallSpec(t)
		change(&s)
		a, err := decoder.InitializeAdapter(context.Background(), s)
		if a != nil || !errors.Is(err, decoder.ErrInitialAdapterIdentity) {
			t.Fatalf("unqualified initialization: %v", err)
		}
	}
	for _, change := range []func(*decoder.InitialAdapterSpec){func(s *decoder.InitialAdapterSpec) { s.Projections = nil }, func(s *decoder.InitialAdapterSpec) { s.Projections[0].Input = 0 }, func(s *decoder.InitialAdapterSpec) { s.Projections[1].Name = s.Projections[0].Name }, func(s *decoder.InitialAdapterSpec) { s.ExpectedSHA256 = "" }, func(s *decoder.InitialAdapterSpec) { s.PreludeBlocks = 4097 }} {
		s := smallSpec(t)
		change(&s)
		a, err := decoder.InitializeAdapter(context.Background(), s)
		if a != nil || !errors.Is(err, decoder.ErrInitialAdapterSpec) {
			t.Fatalf("invalid specification: %v", err)
		}
	}
}

func TestCancellationDoesNotPoisonIndependentGenerator(t *testing.T) {
	s := smallSpec(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if a, err := decoder.InitializeAdapter(ctx, s); a != nil || !errors.Is(err, context.Canceled) {
		t.Fatal("canceled initialization returned tensors")
	}
	first, err := decoder.InitializeAdapter(context.Background(), s)
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()
	second, err := decoder.InitializeAdapter(context.Background(), s)
	if err != nil {
		t.Fatal(err)
	}
	defer second.Close()
	if first.SHA256 != second.SHA256 {
		t.Fatal("independent streams diverged")
	}
	for i, p := range first.Parameters {
		if p.Value == second.Parameters[i].Value {
			t.Fatal("independent results share handles")
		}
	}
}
