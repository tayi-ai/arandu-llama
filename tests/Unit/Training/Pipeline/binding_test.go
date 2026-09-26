package pipeline_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/tayi-ai/arandu-llama/checkpoint"
	"github.com/tayi-ai/arandu-llama/training/pipeline"
)

func boundConstrainedFixture(t *testing.T, steps int) (pipeline.ConstrainedConfig, pipeline.StageContext, *constrainedTracker) {
	t.Helper()
	config, c, tracker := constrainedFixture(t, steps)
	config.Protocol.Version = 2
	config.Protocol.InitialParametersSHA256 = ""
	config.Protocol.InitialSource = pipeline.ArtifactBinding{Kind: pipeline.BindingPredecessor, Predecessor: pipeline.PredecessorOutput{StageID: c.Execution.Recipe.Stages[0].ID, Path: "initial.safetensors", Schema: "adapter-safetensors-v1", MaxBytes: 4096}}
	var initial bytes.Buffer
	written, err := checkpoint.WriteFloat32(context.Background(), &initial, []checkpoint.Float32Tensor{{Name: "adapter", Shape: []uint64{1}, Values: []float32{2}}}, config.Protocol.Limits.Checkpoint)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(c.ArtifactDirectory, "initial.safetensors"), initial.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	origin := constrainedReceipt(pipeline.StageReceipt{Version: 1, Identity: c.Parent.Identity, StageID: config.Protocol.InitialSource.Predecessor.StageID, Result: pipeline.StageResult{Complete: true, Steps: 1, Artifacts: []pipeline.StageArtifact{{Path: "initial.safetensors", SHA256: written.SHA256, Bytes: written.Bytes}}}})
	c.Completed = append([]pipeline.StageReceipt{origin}, c.Completed...)
	// Derived cache bytes differ from the admitted underlying training corpus.
	body := []byte("derived objective cache from the frozen corpus")
	if err := os.WriteFile(filepath.Join(config.SourceDirectory, "source.bin"), body, 0600); err != nil {
		t.Fatal(err)
	}
	for i := range config.Protocol.Updates {
		config.Protocol.Updates[i].Sources = []pipeline.ConstrainedSource{{Binding: pipeline.ArtifactBinding{Kind: pipeline.BindingDerivedInput, Input: config.Recipe.Training, Artifact: pipeline.StageArtifact{Path: "source.bin", SHA256: constrainedBodyHash(body), Bytes: int64(len(body))}}}}
	}
	config.Factory = func(ctx context.Context, r pipeline.ConstrainedRequest) (pipeline.ConstrainedSession, error) {
		tracker.opens.Add(1)
		if tracker.before != nil {
			if err := tracker.before(r); err != nil {
				return pipeline.ConstrainedSession{}, err
			}
		}
		f, err := os.Open(r.Initial.File.Path)
		if err != nil {
			return pipeline.ConstrainedSession{}, err
		}
		defer f.Close()
		s, err := checkpoint.OpenSafetensors(f, r.Initial.File.Artifact.Bytes, config.Protocol.Limits.Checkpoint)
		if err != nil {
			return pipeline.ConstrainedSession{}, err
		}
		rd, err := s.TensorReader("adapter")
		if err != nil {
			return pipeline.ConstrainedSession{}, err
		}
		var b [4]byte
		if _, err = io.ReadFull(rd, b[:]); err != nil {
			return pipeline.ConstrainedSession{}, err
		}
		tracker.last = &scalarModel{values: []float32{math.Float32frombits(binary.LittleEndian.Uint32(b[:]))}, quadratic: true}
		return pipeline.ConstrainedSession{Model: tracker.last, Close: func() error { tracker.closes.Add(1); return nil }}, nil
	}
	constrainedPin(t, &config)
	return config, c, tracker
}

func TestConstrainedV2ConstructsBeforeOutputsAndConsumesInitialCheckpoint(t *testing.T) {
	config, c, tracker := boundConstrainedFixture(t, 1)
	completed := c.Completed
	c.Completed = nil
	initialPath := filepath.Join(c.ArtifactDirectory, "initial.safetensors")
	body, err := os.ReadFile(initialPath)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.Remove(initialPath); err != nil {
		t.Fatal(err)
	}
	h := constrainedHandler(t, config)
	if err = h.Run(context.Background(), c, func(context.Context, pipeline.StageResult) error { t.Fatal("advanced missing source"); return nil }); err == nil || tracker.opens.Load() != 0 {
		t.Fatal("missing predecessor opened model", err)
	}
	if err = os.WriteFile(initialPath, body, 0600); err != nil {
		t.Fatal(err)
	}
	c.Completed = completed
	tracker.before = func(r pipeline.ConstrainedRequest) error {
		if r.Initial.ParametersSHA256 != constrainedScalarHash(2) || r.Initial.File.Binding.StageID != config.Recipe.Stages[0].ID || r.Sources[0].Artifact.SHA256 == config.Recipe.Training.SHA256 {
			t.Fatal("wrong bound inputs", r.Initial)
		}
		intents, err := filepath.Glob(filepath.Join(c.ArtifactDirectory, "constrained-*", "attempt-*.json"))
		if err != nil {
			t.Fatal(err)
		}
		// Inspect the exact persisted bound input rather than relying on a callback flag.
		if len(intents) != 1 {
			t.Fatalf("intent absent before numerical factory: %v", intents)
		}
		intent, err := os.ReadFile(intents[0])
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Contains(intent, []byte(r.Initial.File.Binding.ReceiptSHA256)) || !bytes.Contains(intent, []byte(r.Sources[0].Artifact.SHA256)) {
			t.Fatal("intent did not pin actual inputs")
		}
		return nil
	}
	if err = h.Run(context.Background(), c, constrainedCommit(t, h, &c, nil)); err != nil {
		t.Fatal(err)
	}
	if tracker.opens.Load() != 1 || tracker.last.values[0] != 1 || !c.Previous.Result.Complete {
		t.Fatal("initial checkpoint was not consumed by real update", tracker.last.values)
	}
}

func TestConstrainedV2RecoversEvidenceWithoutReopeningAndRefusesChangedBinding(t *testing.T) {
	config, c, tracker := boundConstrainedFixture(t, 1)
	h := constrainedHandler(t, config)
	lost := errors.New("lost application commit")
	if err := h.Run(context.Background(), c, func(context.Context, pipeline.StageResult) error { return lost }); !errors.Is(err, lost) {
		t.Fatal(err)
	}
	h = constrainedHandler(t, config)
	c.Execution.Generation++
	if err := h.Reconcile(context.Background(), c, constrainedCommit(t, h, &c, nil)); err != nil {
		t.Fatal(err)
	}
	if tracker.opens.Load() != 1 || !c.Previous.Result.Complete {
		t.Fatal("replayed model")
	}
	old := c.Completed[0]
	c.Completed[0].Identity.Generation++
	c.Completed[0] = constrainedReceipt(c.Completed[0])
	if err := h.Verify(context.Background(), c, *c.Previous); err == nil {
		t.Fatal("changed producer receipt accepted")
	}
	c.Completed[0] = old
	if err := os.WriteFile(filepath.Join(c.ArtifactDirectory, "initial.safetensors"), []byte("corrupt"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := h.Verify(context.Background(), c, *c.Previous); err == nil {
		t.Fatal("corrupt initial accepted")
	}
}

func TestConstrainedV2IgnoredInitialAndUncertainAttemptRefused(t *testing.T) {
	config, c, tracker := boundConstrainedFixture(t, 1)
	config.Factory = func(context.Context, pipeline.ConstrainedRequest) (pipeline.ConstrainedSession, error) {
		tracker.opens.Add(1)
		return pipeline.ConstrainedSession{Model: &scalarModel{values: []float32{1}, quadratic: true}, Close: func() error { return nil }}, nil
	}
	h := constrainedHandler(t, config)
	if err := h.Run(context.Background(), c, func(context.Context, pipeline.StageResult) error { t.Fatal("wrong initial committed"); return nil }); err == nil {
		t.Fatal("ignored initial accepted")
	}
	if err := h.Run(context.Background(), c, func(context.Context, pipeline.StageResult) error { return nil }); !errors.Is(err, pipeline.ErrConstrainedAttempt) {
		t.Fatal(err)
	}
	if tracker.opens.Load() != 1 {
		t.Fatal("uncertain computation replayed")
	}
}

func TestConstrainedV2InitialPhaseAndDerivedRoleAreExplicit(t *testing.T) {
	for _, change := range []func(*pipeline.ConstrainedConfig){
		func(c *pipeline.ConstrainedConfig) {
			c.Protocol.InitialSource.Predecessor.StageID = c.Recipe.Stages[2].ID
		},
		func(c *pipeline.ConstrainedConfig) {
			c.Protocol.Updates[0].Sources[0].Binding.Input = c.Recipe.Protection
		},
		func(c *pipeline.ConstrainedConfig) {
			c.Protocol.InitialSource.Predecessor.Schema = "arbitrary-container"
		},
		func(c *pipeline.ConstrainedConfig) { c.Protocol.InitialParametersSHA256 = constrainedScalarHash(2) },
	} {
		config, _, _ := boundConstrainedFixture(t, 1)
		change(&config)
		config.ProtocolSHA256, _ = config.Protocol.Digest()
		if _, err := pipeline.NewConstrainedStage(config); err == nil {
			t.Fatal("invalid source admitted")
		}
	}
}

func TestArtifactBindingRequiresUniqueCurrentCompletedReceiptAndVerifiedBytes(t *testing.T) {
	config, c, _ := boundConstrainedFixture(t, 1)
	b := config.Protocol.InitialSource
	resolved, err := pipeline.ResolveArtifact(context.Background(), c, config.SourceDirectory, b, 4096)
	if err != nil {
		t.Fatal(err)
	}
	if resolved.Binding.Artifact != c.Completed[0].Result.Artifacts[0] {
		t.Fatal("resolved wrong checkpoint")
	}
	for name, change := range map[string]func(*pipeline.StageContext){
		"missing":   func(c *pipeline.StageContext) { c.Completed = c.Completed[1:] },
		"duplicate": func(c *pipeline.StageContext) { c.Completed = append(c.Completed, c.Completed[0]) },
		"future-generation": func(c *pipeline.StageContext) {
			c.Completed[0].Identity.Generation++
			c.Completed[0] = constrainedReceipt(c.Completed[0])
		},
		"wrong-run": func(c *pipeline.StageContext) {
			c.Completed[0].Identity.RunID = "another-run"
			c.Completed[0] = constrainedReceipt(c.Completed[0])
		},
		"unfinished": func(c *pipeline.StageContext) {
			c.Completed[0].Result.Complete = false
			c.Completed[0] = constrainedReceipt(c.Completed[0])
		},
	} {
		t.Run(name, func(t *testing.T) {
			copy := c
			copy.Completed = append([]pipeline.StageReceipt(nil), c.Completed...)
			change(&copy)
			if _, err := pipeline.ResolveArtifact(context.Background(), copy, config.SourceDirectory, b, 4096); err == nil {
				t.Fatal("invalid completed receipt accepted")
			}
		})
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err = pipeline.ResolveArtifact(ctx, c, config.SourceDirectory, b, 4096); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	changed := b
	changed.Predecessor.MaxBytes = 1
	if _, err = pipeline.ResolveArtifact(context.Background(), c, config.SourceDirectory, changed, 4096); err == nil {
		t.Fatal("oversized artifact accepted")
	}
	if err = os.Remove(resolved.Path); err != nil {
		t.Fatal(err)
	}
	outside := filepath.Join(t.TempDir(), "initial")
	if err = os.WriteFile(outside, []byte("outside"), 0600); err != nil {
		t.Fatal(err)
	}
	if err = os.Symlink(outside, resolved.Path); err != nil {
		t.Fatal(err)
	}
	if _, err = pipeline.ResolveArtifact(context.Background(), c, config.SourceDirectory, b, 4096); err == nil {
		t.Fatal("symlink accepted")
	}
}

func TestConstrainedV2CancelledBeforeBindingHasNoIntent(t *testing.T) {
	config, c, tracker := boundConstrainedFixture(t, 1)
	h := constrainedHandler(t, config)
	before, err := os.ReadDir(c.ArtifactDirectory)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err = h.Run(ctx, c, func(context.Context, pipeline.StageResult) error { return nil }); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	after, err := os.ReadDir(c.ArtifactDirectory)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(before, after) || tracker.opens.Load() != 0 {
		t.Fatal("cancelled stage created an intent")
	}
}
