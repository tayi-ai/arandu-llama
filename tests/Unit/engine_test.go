package unit_test

import (
	"context"
	"errors"
	"testing"

	llama "github.com/tayi-ai/arandu-llama"
)

// What taking a reading has to refuse before a card is involved.
//
// The wrapper this replaced declared an Engine interface and held none, so these
// could be exercised against a stub. This package is the engine now, so the only
// thing testable without weights is the refusal -- and the refusal is the part
// that matters, because the alternative to refusing is recording a loss of zero,
// which reads downstream as a perfect prediction rather than as an absent
// measurement.

func TestTakingAReadingWithNoContextIsRefused(t *testing.T) {
	_, err := llama.Take(context.Background(), nil, administrator(), nil, "n", "s", []int32{1, 2}, 0)
	if !errors.Is(err, llama.ErrNoContext) {
		t.Fatalf("taking a reading with no context returned %v, want ErrNoContext", err)
	}
}

func TestAnAdapterCarriesWhatLabelsAReading(t *testing.T) {
	// A reading that cannot say which adapter produced it cannot be compared with
	// any other, and comparison is the only thing a reading is for. The digest is
	// read from the file at load, because two exports of the same training run
	// have different bytes only if they are different training runs.
	var a *llama.Adapter
	if a != nil {
		_ = a.Digest()
		_ = a.Path()
	}
	// The accessors exist and are exported; a compile of this file is the
	// assertion. A digest that stopped being reachable would break the label
	// silently, and a silent label failure is a table where the base model and an
	// unrecorded adapter are indistinguishable.
}
