package pipeline_test

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/tayi-ai/arandu-llama/checkpoint"
	"github.com/tayi-ai/arandu-llama/training/pipeline"
)

func constrainedHash(v any) string {
	body, _ := json.Marshal(v)
	h := sha256.Sum256(body)
	return hex.EncodeToString(h[:])
}
func constrainedBodyHash(body []byte) string {
	h := sha256.Sum256(body)
	return hex.EncodeToString(h[:])
}
func constrainedScalarHash(value float32) string {
	h := sha256.New()
	_, _ = h.Write([]byte("adapter"))
	var scalar [4]byte
	binary.LittleEndian.PutUint32(scalar[:], math.Float32bits(value))
	_, _ = h.Write(scalar[:])
	return hex.EncodeToString(h.Sum(nil))
}
func constrainedReceipt(r pipeline.StageReceipt) pipeline.StageReceipt {
	r.SHA256 = ""
	r.SHA256 = constrainedHash(r)
	return r
}

type constrainedTracker struct {
	opens, closes atomic.Int32
	before        func(pipeline.ConstrainedRequest) error
	last          *scalarModel
}

func constrainedFixture(t *testing.T, steps int) (pipeline.ConstrainedConfig, pipeline.StageContext, *constrainedTracker) {
	t.Helper()
	_, execution, _ := durableFixture(t.TempDir())
	root, sources := t.TempDir(), t.TempDir()
	body := []byte("qualified deterministic objective fixture")
	if err := os.WriteFile(filepath.Join(sources, "source.bin"), body, 0600); err != nil {
		t.Fatal(err)
	}
	old := execution.Recipe.Training
	execution.Recipe.Training.SHA256 = constrainedBodyHash(body)
	for i := range execution.Recipe.Stages {
		for j := range execution.Recipe.Stages[i].Inputs {
			if execution.Recipe.Stages[i].Inputs[j] == old {
				execution.Recipe.Stages[i].Inputs[j] = execution.Recipe.Training
			}
		}
	}
	execution.Recipe.Stages[3].MaxSteps = steps
	r := execution.Recipe
	stage := r.Stages[3]
	rsha, err := r.Digest()
	if err != nil {
		t.Fatal(err)
	}
	id := pipeline.ReceiptIdentity{RunID: execution.RunID, TenantID: execution.TenantID, RecipeSHA256: rsha, TargetSHA256: execution.TargetSHA256, PlacementSHA256: constrainedHash(execution.Placement), Generation: execution.Generation}
	parent := constrainedReceipt(pipeline.StageReceipt{Version: 1, Identity: id, StageID: stage.ParentStage, Result: pipeline.StageResult{Steps: 1, Complete: true}})
	c := pipeline.StageContext{Execution: execution, Stage: stage, ArtifactDirectory: root, Parent: &parent, Completed: []pipeline.StageReceipt{parent}}
	x := stepConfig()
	x.Lambda = 4
	x.MinimumGain = 0.0001
	x.Protection[0].Floor = 0.1
	x.Protection[0].Positive.DatasetDigest, x.Protection[0].Negative.DatasetDigest = r.Protection.SHA256, r.Protection.SHA256
	p := pipeline.ConstrainedProtocol{Version: 1, RecipeSHA256: rsha, PlacementSHA256: id.PlacementSHA256, FactoryQualificationSHA256: execution.Placement.QualificationSHA256,
		StageID: stage.ID, InitialParametersSHA256: constrainedScalarHash(1), Layout: []pipeline.ConstrainedTensor{{Name: "adapter", Shape: []uint64{1}}},
		Limits: pipeline.ConstrainedLimits{MaxProtocolBytes: 1 << 20, MaxReceiptBytes: 1 << 20, MaxCheckpointBytes: 1 << 20, MaxTotalBytes: int64(steps) * 3 << 20, MaxSourceBytes: 1 << 20, MaxParameters: 1, MaxSteps: steps,
			Checkpoint: checkpoint.Limits{MaxHeaderBytes: 4096, MaxTensors: 1, MaxDimensions: 1, MaxMetadataEntries: 1, MaxChunkBytes: 32}}}
	for i := 0; i < steps; i++ {
		p.Updates = append(p.Updates, pipeline.ConstrainedUpdate{Config: x, Sources: []pipeline.ConstrainedSource{{Input: r.Training, Artifact: pipeline.StageArtifact{Path: "source.bin", SHA256: r.Training.SHA256, Bytes: int64(len(body))}}}})
	}
	tracker := &constrainedTracker{}
	config := pipeline.ConstrainedConfig{Recipe: r, Placement: execution.Placement, Protocol: p, SourceDirectory: sources,
		Factory: func(_ context.Context, request pipeline.ConstrainedRequest) (pipeline.ConstrainedSession, error) {
			tracker.opens.Add(1)
			if tracker.before != nil {
				if err := tracker.before(request); err != nil {
					return pipeline.ConstrainedSession{}, err
				}
			}
			tracker.last = &scalarModel{values: []float32{1}, quadratic: true}
			return pipeline.ConstrainedSession{Model: tracker.last, Close: func() error { tracker.closes.Add(1); return nil }}, nil
		}}
	constrainedPin(t, &config)
	return config, c, tracker
}
func constrainedPin(t *testing.T, config *pipeline.ConstrainedConfig) {
	t.Helper()
	hash, err := config.Protocol.Digest()
	if err != nil {
		t.Fatal(err)
	}
	config.ProtocolSHA256 = hash
}
func constrainedHandler(t *testing.T, c pipeline.ConstrainedConfig) *pipeline.ConstrainedStage {
	t.Helper()
	h, err := pipeline.NewConstrainedStage(c)
	if err != nil {
		t.Fatal(err)
	}
	return h
}
func constrainedCommit(t *testing.T, h *pipeline.ConstrainedStage, c *pipeline.StageContext, after func(pipeline.StageResult) error) func(context.Context, pipeline.StageResult) error {
	t.Helper()
	return func(ctx context.Context, result pipeline.StageResult) error {
		previous := c.Parent.SHA256
		if c.Previous != nil {
			previous = c.Previous.SHA256
		}
		id := c.Parent.Identity
		id.Generation = c.Execution.Generation
		r := constrainedReceipt(pipeline.StageReceipt{Version: 1, Identity: id, StageID: c.Stage.ID, Result: result, PreviousSHA256: previous})
		if err := h.Verify(ctx, *c, r); err != nil {
			return err
		}
		c.Previous = &r
		if after != nil {
			return after(result)
		}
		return nil
	}
}

func TestConstrainedStageExecutesThreeRealStepsAndVerifiesPersistedEvidence(t *testing.T) {
	config, c, tracker := constrainedFixture(t, 3)
	h := constrainedHandler(t, config)
	var losses []float64
	err := h.Run(context.Background(), c, constrainedCommit(t, h, &c, func(result pipeline.StageResult) error {
		body, err := os.ReadFile(filepath.Join(c.ArtifactDirectory, result.Artifacts[1].Path))
		if err != nil {
			return err
		}
		var e pipeline.ConstrainedEvidence
		if err := json.Unmarshal(body, &e); err != nil {
			return err
		}
		losses = append(losses, e.Receipt.CandidateObjective.Loss)
		if !e.Receipt.Evaluation.Accepted || e.Receipt.Evaluation.Margins[0].Value < .1 || len(result.Artifacts) != 3 {
			t.Fatal("invalid scientific receipt")
		}
		return nil
	}))
	if err != nil || tracker.opens.Load() != 3 || tracker.closes.Load() != 3 || c.Previous == nil || !c.Previous.Result.Complete {
		t.Fatalf("run: %v %+v", err, tracker)
	}
	wanted := []float64{.25, .0625, .015625}
	for i := range wanted {
		if losses[i] != wanted[i] {
			t.Fatal(losses)
		}
	}
	if err := h.Run(context.Background(), c, func(context.Context, pipeline.StageResult) error { t.Fatal("completed step replayed"); return nil }); err != nil {
		t.Fatal(err)
	}
	if tracker.opens.Load() != 3 {
		t.Fatal("model reopened after completion")
	}
}

func TestConstrainedStageRecoversCommitFailureWithoutRecomputing(t *testing.T) {
	config, c, tracker := constrainedFixture(t, 2)
	h := constrainedHandler(t, config)
	interrupted := errors.New("application commit interrupted")
	if err := h.Run(context.Background(), c, func(context.Context, pipeline.StageResult) error { return interrupted }); !errors.Is(err, interrupted) {
		t.Fatal(err)
	}
	if tracker.opens.Load() != 1 || tracker.closes.Load() != 1 {
		t.Fatal("first attempt did not stop")
	}
	h = constrainedHandler(t, config)
	c.Execution.Generation++
	if err := h.Reconcile(context.Background(), c, constrainedCommit(t, h, &c, nil)); err != nil {
		t.Fatal(err)
	}
	if tracker.opens.Load() != 1 || c.Previous.Result.Steps != 1 {
		t.Fatal("reconciliation computed or missed durable step")
	}
	if err := h.Run(context.Background(), c, constrainedCommit(t, h, &c, nil)); err != nil {
		t.Fatal(err)
	}
	if tracker.opens.Load() != 2 || !c.Previous.Result.Complete {
		t.Fatal("resume cursor differs")
	}
}

func TestConstrainedStageRejectsCandidateAndBlocksRedelivery(t *testing.T) {
	config, c, tracker := constrainedFixture(t, 1)
	config.Protocol.Updates[0].Config.Lambda = .1
	config.Protocol.Updates[0].Config.Protection[0].Floor = -100
	constrainedPin(t, &config)
	h := constrainedHandler(t, config)
	commit := func(context.Context, pipeline.StageResult) error { t.Fatal("rejected candidate advanced"); return nil }
	if err := h.Run(context.Background(), c, commit); err == nil {
		t.Fatal("objective degradation admitted")
	}
	if tracker.last.values[0] != 1 || tracker.closes.Load() != 1 {
		t.Fatal("rejected model not restored and closed")
	}
	h = constrainedHandler(t, config)
	if err := h.Run(context.Background(), c, commit); !errors.Is(err, pipeline.ErrConstrainedAttempt) {
		t.Fatal(err)
	}
	if err := h.Reconcile(context.Background(), c, commit); !errors.Is(err, pipeline.ErrConstrainedAttempt) {
		t.Fatal(err)
	}
	if tracker.opens.Load() != 1 {
		t.Fatal("rejected attempt repeated")
	}
}

func TestConstrainedStageInterruptedIntentCannotReopenModel(t *testing.T) {
	config, c, tracker := constrainedFixture(t, 1)
	tracker.before = func(pipeline.ConstrainedRequest) error { return io.ErrUnexpectedEOF }
	h := constrainedHandler(t, config)
	commit := func(context.Context, pipeline.StageResult) error { t.Fatal("interrupted attempt advanced"); return nil }
	if err := h.Run(context.Background(), c, commit); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatal(err)
	}
	tracker.before = nil
	h = constrainedHandler(t, config)
	if err := h.Run(context.Background(), c, commit); !errors.Is(err, pipeline.ErrConstrainedAttempt) {
		t.Fatal(err)
	}
	if tracker.opens.Load() != 1 {
		t.Fatal("uncertain computation repeated")
	}
}

func TestConstrainedStageCorruptionAndOrphansNeverAdvance(t *testing.T) {
	for _, mode := range []string{"parameters", "evidence", "symlink", "orphan"} {
		t.Run(mode, func(t *testing.T) {
			config, c, tracker := constrainedFixture(t, 1)
			h := constrainedHandler(t, config)
			var output pipeline.StageResult
			if err := h.Run(context.Background(), c, func(_ context.Context, r pipeline.StageResult) error { output = r; return io.ErrClosedPipe }); !errors.Is(err, io.ErrClosedPipe) {
				t.Fatal(err)
			}
			path := filepath.Join(c.ArtifactDirectory, output.Artifacts[0].Path)
			switch mode {
			case "parameters":
				body, _ := os.ReadFile(path)
				body[len(body)-1] ^= 1
				if err := os.WriteFile(path, body, 0600); err != nil {
					t.Fatal(err)
				}
			case "evidence":
				path = filepath.Join(c.ArtifactDirectory, output.Artifacts[1].Path)
				if err := os.WriteFile(path, []byte("{}\n"), 0600); err != nil {
					t.Fatal(err)
				}
			case "symlink":
				original := filepath.Join(t.TempDir(), "file")
				if err := os.Rename(path, original); err != nil {
					t.Fatal(err)
				}
				if err := os.Symlink(original, path); err != nil {
					t.Fatal(err)
				}
			case "orphan":
				if err := os.Remove(filepath.Join(c.ArtifactDirectory, output.Artifacts[1].Path)); err != nil {
					t.Fatal(err)
				}
			}
			h = constrainedHandler(t, config)
			if err := h.Reconcile(context.Background(), c, func(context.Context, pipeline.StageResult) error { t.Fatal("corrupt checkpoint advanced"); return nil }); err == nil {
				t.Fatal("corruption admitted")
			}
			if tracker.opens.Load() != 1 {
				t.Fatal("corruption triggered compute")
			}
		})
	}
}

func TestConstrainedStageRefusesUnpinnedSourcesAndBudgets(t *testing.T) {
	for _, mode := range []string{"configuration-hash", "total-budget", "coefficients", "source-reference", "source-bytes", "source-symlink", "initial-parameters"} {
		t.Run(mode, func(t *testing.T) {
			config, c, tracker := constrainedFixture(t, 1)
			switch mode {
			case "configuration-hash":
				config.ProtocolSHA256 = strings.Repeat("0", 64)
			case "total-budget":
				config.Protocol.Limits.MaxTotalBytes--
			case "coefficients":
				config.Protocol.Updates[0].Config.Solver.MaxCoefficients = 0
			case "source-reference":
				config.Protocol.Updates[0].Sources[0].Input.ID = "unregistered"
				constrainedPin(t, &config)
			case "source-bytes":
				if err := os.WriteFile(filepath.Join(config.SourceDirectory, "source.bin"), []byte("changed"), 0600); err != nil {
					t.Fatal(err)
				}
			case "source-symlink":
				original := filepath.Join(config.SourceDirectory, "source.bin")
				if err := os.Rename(original, original+".actual"); err != nil {
					t.Fatal(err)
				}
				if err := os.Symlink(original+".actual", original); err != nil {
					t.Fatal(err)
				}
			case "initial-parameters":
				config.Protocol.InitialParametersSHA256 = constrainedScalarHash(2)
				constrainedPin(t, &config)
			}
			h, err := pipeline.NewConstrainedStage(config)
			if err == nil {
				err = h.Run(context.Background(), c, func(context.Context, pipeline.StageResult) error { t.Fatal("invalid admission advanced"); return nil })
			}
			if err == nil {
				t.Fatal("invalid contract admitted")
			}
			if mode != "initial-parameters" && tracker.opens.Load() != 0 {
				t.Fatal("refused source reached model")
			}
		})
	}
}

func TestConstrainedStageSerialGateWaitIsCancelable(t *testing.T) {
	config, c, tracker := constrainedFixture(t, 1)
	entered, release := make(chan struct{}), make(chan struct{})
	tracker.before = func(pipeline.ConstrainedRequest) error { close(entered); <-release; return nil }
	h := constrainedHandler(t, config)
	done := make(chan error, 1)
	go func() {
		done <- h.Run(context.Background(), c, func(context.Context, pipeline.StageResult) error { return nil })
	}()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("factory never opened")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := h.Run(ctx, c, func(context.Context, pipeline.StageResult) error { return nil }); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if tracker.opens.Load() != 1 || tracker.closes.Load() != 1 {
		t.Fatal("gate admitted concurrent model")
	}
}
