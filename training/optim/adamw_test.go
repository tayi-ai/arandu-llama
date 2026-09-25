package optim_test

import (
	"math"
	"reflect"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/optim"
)

func TestAdamWResumeMatchesUninterruptedUpdate(t *testing.T) {
	config := optim.AdamWConfig{LearningRate: .01, Beta1: .9, Beta2: .999, Epsilon: 1e-8, WeightDecay: .1, MaxGradientNorm: 1}
	start, err := optim.NewAdamWState([]float32{1, -2, 0})
	if err != nil {
		t.Fatal(err)
	}
	first, receipt, err := optim.UpdateAdamW(start, []float32{3, 4, -1}, config)
	if err != nil {
		t.Fatal(err)
	}
	if receipt.Step != 1 || receipt.ChangedParameter != 3 || math.Abs(receipt.GradientNorm-math.Sqrt(26)) > 1e-12 {
		t.Fatalf("unexpected first receipt: %+v", receipt)
	}
	if !reflect.DeepEqual(start.Parameters, []float32{1, -2, 0}) || start.Step != 0 {
		t.Fatal("accepted update changed input state")
	}
	restored := optim.AdamWState{Step: first.Step, Parameters: append([]float32(nil), first.Parameters...), First: append([]float32(nil), first.First...), Second: append([]float32(nil), first.Second...)}
	second, _, err := optim.UpdateAdamW(first, []float32{-2, 1, 4}, config)
	if err != nil {
		t.Fatal(err)
	}
	resumed, _, err := optim.UpdateAdamW(restored, []float32{-2, 1, 4}, config)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(second, resumed) {
		t.Fatal("resumed update differs from uninterrupted update")
	}
}

func TestAdamWRejectsCorruptResumeAndNonfiniteGradient(t *testing.T) {
	config := optim.AdamWConfig{LearningRate: .01, Beta1: .9, Beta2: .999, Epsilon: 1e-8, MaxGradientNorm: 1}
	state, err := optim.NewAdamWState([]float32{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	for _, candidate := range []optim.AdamWState{
		{Step: 2, Parameters: []float32{1, 2}, First: []float32{0}, Second: []float32{0, 0}},
		{Step: 2, Parameters: []float32{1, 2}, First: []float32{0, 0}, Second: []float32{-1, 0}},
	} {
		if err := optim.ValidateAdamWState(candidate); err == nil {
			t.Fatal("corrupt state was accepted")
		}
	}
	if _, _, err := optim.UpdateAdamW(state, []float32{1, float32(math.NaN())}, config); err == nil {
		t.Fatal("nonfinite gradient was accepted")
	}
	if _, _, err := optim.UpdateAdamW(state, []float32{0, 0}, config); err == nil {
		t.Fatal("zero gradient was accepted")
	}
	if !reflect.DeepEqual(state.Parameters, []float32{1, 2}) {
		t.Fatal("rejected update changed input state")
	}
}
