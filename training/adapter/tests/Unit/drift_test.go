package unit_test

import (
	"encoding/binary"
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"

	services "github.com/tayi-ai/arandu-llama/training/adapter"
)

// idleHost is an AdapterHost with no card behind it: it hashes what it is
// asked to load and applies nothing. It is what lets the fold be measured on
// the real adapter without a model.
type idleHost struct{}

type idleHandle struct{ digest string }

func (h idleHandle) Digest() string { return h.digest }
func (h idleHandle) Close() error   { return nil }

func (idleHost) LoadAdapter(path string) (services.AdapterHandle, error) {
	body, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return idleHandle{digest: services.Digest(body)}, nil
}

func (idleHost) SetAdapters([]services.AdapterHandle, []float32) error { return nil }

// TestFoldingAndUnfoldingDoesNotDriftTheAdapter checks reversible updates on a
// synthetic F32 adapter. It does not qualify a production model or dataset.
func TestFoldingAndUnfoldingDoesNotDriftTheAdapter(t *testing.T) {
	dir := t.TempDir()
	working := filepath.Join(dir, "policy.gguf")
	if _, err := services.WriteInitialLoRA(working, "fixture-decoder", 2, 4, 19, []services.LoRATarget{{Name: "fixture.weight", Input: 4, Output: 4}}); err != nil {
		t.Fatal(err)
	}
	body, err := os.ReadFile(working)
	if err != nil {
		t.Fatal(err)
	}
	original := append([]byte(nil), body...)

	s, err := services.OpenSnapshot(idleHost{}, working, dir)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	container := s.Container()

	// A step folds -eta*g along z. The inverse folds +eta*g along the same z.
	// Together they are the identity, in exact arithmetic.
	const coefficient = 1e-4
	const cycles = 1000
	for i := 0; i < cycles; i++ {
		d := services.Direction{Seed: uint64(1000 + i), Scale: 1}
		if _, err := s.Fold(d, coefficient); err != nil {
			t.Fatal(err)
		}
		if _, err := s.Fold(d, -coefficient); err != nil {
			t.Fatal(err)
		}
	}

	after, err := os.ReadFile(working)
	if err != nil {
		t.Fatal(err)
	}
	if services.Digest(after) != s.Digest() {
		t.Fatal("the file on disk is not the snapshot the folds left in memory")
	}
	var maxAbs, sumAbs, sumSigned float64
	moved := 0
	for _, tensor := range container.Tensors {
		start := container.DataOffset + tensor.Offset
		for k := int64(0); k < tensor.Elements; k++ {
			at := start + k*4
			was := math.Float32frombits(binary.LittleEndian.Uint32(original[at:]))
			is := math.Float32frombits(binary.LittleEndian.Uint32(after[at:]))
			delta := float64(is - was)
			if delta != 0 {
				moved++
				sumSigned += delta
				sumAbs += math.Abs(delta)
				if math.Abs(delta) > maxAbs {
					maxAbs = math.Abs(delta)
				}
			}
		}
	}
	total := 0
	for _, tensor := range container.Tensors {
		total += int(tensor.Elements)
	}
	t.Logf("after %d fold/unfold cycles at coefficient %g:", cycles, coefficient)
	t.Logf("  elements changed : %d of %d (%.4f%%)", moved, total, 100*float64(moved)/float64(total))
	t.Logf("  largest drift    : %g", maxAbs)
	t.Logf("  mean |drift|     : %g", sumAbs/float64(total))
	t.Logf("  signed sum       : %g  (a biased rounding shows here)", sumSigned)

	// Drift larger than the update would measure rounding instead of the step.
	if maxAbs > coefficient {
		t.Errorf("a single element drifted by %g, larger than the update of %g: folding is not reversible enough to train with", maxAbs, coefficient)
	}
}

// TestFoldingIntoAHalfPrecisionAdapterIsRefused is the other half of the gate.
//
// The refusal is what keeps the measurement from coming back. An adapter
// converted with the default --outtype f16 looks identical to a good one: same
// tensor names, same shapes, same count. It differs only in that a thousand
// updates folded into it move the weights by more than the updates did, and the
// run that results reads as a method that does not work. It is refused at
// open, so a run never starts on it.
func TestFoldingIntoAHalfPrecisionAdapterIsRefused(t *testing.T) {
	half := templateGGUF(t)
	dir := t.TempDir()
	working := filepath.Join(dir, "half.gguf")
	body, err := os.ReadFile(half)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(working, body, 0o644); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(working)
	if err != nil {
		t.Fatal(err)
	}
	_, err = services.OpenSnapshot(idleHost{}, working, dir)
	if !errors.Is(err, services.ErrHalfPrecisionPolicy) {
		t.Fatalf("opening a half-precision adapter returned %v; it has to be refused", err)
	}
	// And refused before anything was written: a partial fold would leave an
	// adapter that is neither the one measured before nor the one intended after.
	after, err := os.ReadFile(working)
	if err != nil {
		t.Fatal(err)
	}
	if len(before) != len(after) {
		t.Fatal("the refused open changed the file size")
	}
	for i := range before {
		if before[i] != after[i] {
			t.Fatalf("the refused open still wrote at byte %d", i)
		}
	}
}
