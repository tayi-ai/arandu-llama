//go:build libtorch && cgo

package decoder

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/pipeline"
)

func recoveryFixture(t *testing.T) (*RecoveryBackend, *LoadedTextModel, ConstrainedFactoryConfig, RecoveryBatchDocument, RecoveryExpectation, RecoveryLimits, string) {
	t.Helper()
	c, _, loader, _ := sessionFixture(t)
	d := RecoveryBatchDocument{Version: 1, Dataset: c.Recipe.Recipe.Recovery, Student: c.Recipe.Admission.Student, Aggregation: RecoveryTokenMean, SupervisedTokens: 3,
		Examples: []pipeline.Example{{ID: "long", DatasetDigest: c.Recipe.Recipe.Recovery.SHA256, Tokens: []int64{1, 2, 3}, PromptTokens: 1}, {ID: "short", DatasetDigest: c.Recipe.Recipe.Recovery.SHA256, Tokens: []int64{1, 4}, PromptTokens: 1}}}
	l := RecoveryLimits{MaxBytes: 1 << 20, MaxExamples: 4, MaxTotalTokens: 12, MaxTokens: 3, WorkingBytes: 4 << 20}
	targets, err := d.TargetsDigest(l)
	if err != nil {
		t.Fatal(err)
	}
	e := RecoveryExpectation{Dataset: d.Dataset, Student: d.Student, Aggregation: d.Aggregation, TargetsSHA256: targets, Examples: 2, SupervisedTokens: 3}
	body, err := json.Marshal(d)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "recovery.json")
	if err := os.WriteFile(path, body, 0600); err != nil {
		t.Fatal(err)
	}
	batch, err := ReadRecoveryBatch(context.Background(), path, assemblyHash(body), int64(len(body)), e, l)
	if err != nil {
		t.Fatal(err)
	}
	initial, err := InitializeAdapter(context.Background(), c.Recipe.Initializer)
	if err != nil {
		t.Fatal(err)
	}
	loaded, err := loader(context.Background(), c.Assembly, c.Shards, initial)
	if closeErr := initial.Close(); err != nil || closeErr != nil {
		t.Fatal(errors.Join(err, closeErr))
	}
	if _, err := loaded.RestoreAdapter(context.Background(), c.Recipe.Initial); err != nil {
		t.Fatal(err)
	}
	prepare := func(ctx context.Context, n int) (func() error, error) {
		return constrainedRotary(ctx, loaded.Model, n, c.Recipe.Rotary)
	}
	backend, err := NewRecoveryBackend(loaded, batch, c.Recipe.Limits, prepare, c.Recipe.Admission)
	if err != nil {
		t.Fatal(err)
	}
	return backend, loaded, c, d, e, l, path
}

func TestNativeRecoveryWeightsByReturnedSupervisedTokens(t *testing.T) {
	m, loaded, _, d, _, _, _ := recoveryFixture(t)
	ctx := context.Background()
	before, err := loaded.Model.Head.Float32Values()
	if err != nil {
		t.Fatal(err)
	}
	var want pipeline.Objective
	var total int
	var exampleMean float64
	for _, row := range d.Examples {
		limits, close, err := m.state.prepare(ctx, row)
		if err != nil {
			t.Fatal(err)
		}
		gradient, gradErr := CompletionGradient(ctx, loaded.Model, row.Tokens, row.PromptTokens, limits, 1)
		if err := errors.Join(gradErr, close()); err != nil {
			t.Fatal(err)
		}
		flat, err := m.state.flatten(gradient.Gradients, 1)
		if err != nil {
			t.Fatal(err)
		}
		if want.Gradient == nil {
			want.Gradient = make([]float64, len(flat))
		}
		want.Loss += gradient.Loss * float64(gradient.Tokens)
		exampleMean += gradient.Loss
		total += gradient.Tokens
		for i, v := range flat {
			want.Gradient[i] += v * float64(gradient.Tokens)
		}
	}
	want.Loss /= float64(total)
	want.HardLoss = want.Loss
	for i := range want.Gradient {
		want.Gradient[i] /= float64(total)
	}
	got, err := m.Objective(ctx)
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("token-weighted objective differs: %v", err)
	}
	if math.Abs(got.Loss-exampleMean/float64(len(d.Examples))) < 1e-6 {
		t.Fatal("fixture does not distinguish token mean from example mean")
	}
	again, err := m.Objective(ctx)
	if err != nil || !reflect.DeepEqual(got, again) {
		t.Fatalf("repeated objective differs: %v", err)
	}
	after, err := loaded.Model.Head.Float32Values()
	if err != nil || !reflect.DeepEqual(before, after) || loaded.Model.Layers[0].Cosine != nil || loaded.Model.Layers[0].Sine != nil {
		t.Fatal("objective mutated frozen weights or retained rotary tables")
	}
}

func TestNativeRecoveryRunsProtectedStepsAndRestoresRejectedCandidate(t *testing.T) {
	m, loaded, c, _, _, _, _ := recoveryFixture(t)
	ctx := context.Background()
	cfg := c.Recipe.Protocol.Updates[0].Config
	cfg.Lambda = 10
	var previousLoss float64
	for step := 0; step < 3; step++ {
		prior, err := m.Parameters(ctx)
		if err != nil {
			t.Fatal(err)
		}
		receipt, err := pipeline.Step(ctx, m, cfg)
		if err != nil || !receipt.Evaluation.Accepted || receipt.Objective.FeatureLoss != 0 || receipt.CandidateObjective.Loss >= receipt.Objective.Loss || reflect.DeepEqual(prior, receipt.Parameters) {
			t.Fatalf("causal protected step %d failed: %v", step, err)
		}
		if step > 0 && receipt.Objective.Loss != previousLoss {
			t.Fatal("next step did not read the accepted adapter")
		}
		previousLoss = receipt.CandidateObjective.Loss
	}
	prior, err := m.Parameters(ctx)
	if err != nil {
		t.Fatal(err)
	}
	cfg.MinimumGain = 1000
	if _, err := pipeline.Step(ctx, m, cfg); err == nil {
		t.Fatal("candidate escaped impossible frozen gain requirement")
	}
	after, err := m.Parameters(ctx)
	if err != nil || !reflect.DeepEqual(prior, after) {
		t.Fatal("rejected candidate left changed parameters")
	}
	if err := loaded.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Objective(ctx); err == nil {
		t.Fatal("closed model remained usable")
	}
}

func TestNativeRecoveryCancellationClosesBetweenRows(t *testing.T) {
	m, loaded, c, _, _, _, _ := recoveryFixture(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	calls := 0
	m.state.prepareSequence = func(ctx context.Context, n int) (func() error, error) {
		calls++
		close, err := constrainedRotary(ctx, loaded.Model, n, c.Recipe.Rotary)
		return func() error {
			err := close()
			cancel()
			return err
		}, err
	}
	result, err := m.Objective(ctx)
	if !errors.Is(err, context.Canceled) || calls != 1 || len(result.Gradient) != 0 || loaded.Model.Layers[0].Cosine != nil || loaded.Model.Layers[0].Sine != nil {
		t.Fatalf("canceled batch retained partial result or rotary: %v", err)
	}
}

func TestNativeRecoveryRefusesUnverifiedBatchAndAdmission(t *testing.T) {
	m, loaded, c, _, _, _, _ := recoveryFixture(t)
	for _, batch := range []*RecoveryBatch{nil, {}} {
		if _, err := NewRecoveryBackend(loaded, batch, c.Recipe.Limits, nil, c.Recipe.Admission); err == nil {
			t.Fatal("unverified batch admitted")
		}
	}
	wrong := c.Recipe.Admission
	wrong.Student.TemplateSHA256 = assemblyHash([]byte("different template"))
	if _, err := NewRecoveryBackend(loaded, m.batch, c.Recipe.Limits, nil, wrong); err == nil {
		t.Fatal("different template admitted")
	}
	limits := c.Recipe.Limits
	limits.LogitRows = 2
	if _, err := NewRecoveryBackend(loaded, m.batch, limits, nil, c.Recipe.Admission); err == nil {
		t.Fatal("insufficient logits budget admitted")
	}
}
