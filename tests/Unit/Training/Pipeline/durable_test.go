package pipeline_test

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tayi-ai/arandu-llama/training/pipeline"
)

type fixturePhase struct {
	mu        sync.Mutex
	calls     map[string]int
	admit     func(context.Context, pipeline.Recipe, pipeline.Stage, pipeline.Placement) error
	verify    func(context.Context, pipeline.StageContext, pipeline.StageReceipt) error
	run       func(context.Context, pipeline.StageContext, func(context.Context, pipeline.StageResult) error) error
	reconcile func(context.Context, pipeline.StageContext, func(context.Context, pipeline.StageResult) error) error
}

func (h *fixturePhase) Admit(ctx context.Context, r pipeline.Recipe, s pipeline.Stage, p pipeline.Placement) error {
	if h.admit != nil {
		return h.admit(ctx, r, s, p)
	}
	return ctx.Err()
}
func (h *fixturePhase) Verify(ctx context.Context, c pipeline.StageContext, r pipeline.StageReceipt) error {
	if h.verify != nil {
		return h.verify(ctx, c, r)
	}
	for _, a := range r.Result.Artifacts {
		body, err := os.ReadFile(filepath.Join(c.ArtifactDirectory, a.Path))
		if err != nil {
			return err
		}
		if string(body) != fmt.Sprintf("verified %s step %d", c.Stage.ID, r.Result.Steps) {
			return errors.New("scientific evidence differs")
		}
	}
	if c.Stage.ParentStage != "" && (c.Parent == nil || c.Parent.StageID != c.Stage.ParentStage || !c.Parent.Result.Complete) {
		return errors.New("parent evidence missing")
	}
	return ctx.Err()
}
func (h *fixturePhase) Reconcile(ctx context.Context, c pipeline.StageContext, commit func(context.Context, pipeline.StageResult) error) error {
	if h.reconcile != nil {
		return h.reconcile(ctx, c, commit)
	}
	return ctx.Err()
}
func (h *fixturePhase) Run(ctx context.Context, c pipeline.StageContext, commit func(context.Context, pipeline.StageResult) error) error {
	h.mu.Lock()
	h.calls[c.Stage.ID]++
	h.mu.Unlock()
	if h.run != nil {
		return h.run(ctx, c, commit)
	}
	step := int64(0)
	if c.Previous != nil {
		step = c.Previous.Result.Steps
	}
	for step < int64(c.Stage.MaxSteps) {
		step++
		r, err := writeFixtureResult(c, step, step == int64(c.Stage.MaxSteps))
		if err != nil {
			return err
		}
		if err := commit(ctx, r); err != nil {
			return err
		}
	}
	return nil
}
func writeFixtureResult(c pipeline.StageContext, step int64, complete bool) (pipeline.StageResult, error) {
	body := []byte(fmt.Sprintf("verified %s step %d", c.Stage.ID, step))
	name := fmt.Sprintf("%s-%d.bin", c.Stage.ID, step)
	file, err := os.OpenFile(filepath.Join(c.ArtifactDirectory, name), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return pipeline.StageResult{}, err
	}
	_, e := file.Write(body)
	err = errors.Join(e, file.Sync(), file.Close())
	if err != nil {
		return pipeline.StageResult{}, err
	}
	dir, err := os.Open(c.ArtifactDirectory)
	if err != nil {
		return pipeline.StageResult{}, err
	}
	if err = errors.Join(dir.Sync(), dir.Close()); err != nil {
		return pipeline.StageResult{}, err
	}
	sum := sha256.Sum256(body)
	return pipeline.StageResult{Steps: step, Complete: complete, Artifacts: []pipeline.StageArtifact{{Path: name, SHA256: hex.EncodeToString(sum[:]), Bytes: int64(len(body))}}}, nil
}
func durableFixture(root string) (pipeline.DurableConfig, pipeline.Execution, *fixturePhase) {
	h := &fixturePhase{calls: map[string]int{}}
	c := pipeline.DurableConfig{Root: root, Backend: "fixture", RuntimeSHA256: ref("runtime").SHA256, QualificationSHA256: ref("qualification").SHA256, Limits: pipeline.DurableLimits{MaxStateBytes: 1 << 20, MaxReceipts: 100, MaxArtifactsPerReceipt: 4, MaxArtifactBytes: 1 << 16, MaxTotalArtifactBytes: 1 << 20}, Handlers: map[pipeline.Phase]pipeline.StageHandler{}}
	r := recipe()
	for _, s := range r.Stages {
		c.Handlers[s.Phase] = h
	}
	x := pipeline.Execution{RunID: "run-1", TenantID: "tenant-1", Generation: 1, Recipe: r, TargetSHA256: ref("target").SHA256, Placement: pipeline.Placement{Backend: c.Backend, MemoryBytes: 1 << 20, MaxTokens: 128, RuntimeSHA256: c.RuntimeSHA256, QualificationSHA256: c.QualificationSHA256}}
	return c, x, h
}
func runtimeFixture(t *testing.T, c pipeline.DurableConfig) *pipeline.DurableRuntime {
	t.Helper()
	r, e := pipeline.NewDurableRuntime(c)
	if e != nil {
		t.Fatal(e)
	}
	return r
}
func noReport(context.Context, pipeline.Progress) error { return nil }
func statePath(t *testing.T, root string) string {
	t.Helper()
	p, e := filepath.Glob(filepath.Join(root, "*", "state.json"))
	if e != nil || len(p) != 1 {
		t.Fatalf("state files: %v %v", p, e)
	}
	return p[0]
}

func TestDurableRuntimeCompletesOnlyVerifiedPhasesAndResumesReportFailure(t *testing.T) {
	c, x, h := durableFixture(t.TempDir())
	r := runtimeFixture(t, c)
	interrupted := errors.New("database report interrupted")
	if err := r.Run(context.Background(), x, func(context.Context, pipeline.Progress) error { return interrupted }); !errors.Is(err, interrupted) {
		t.Fatal(err)
	}
	if h.calls["stage-0"] != 1 || len(h.calls) != 1 {
		t.Fatal(h.calls)
	}
	x.Generation = 2
	r = runtimeFixture(t, c)
	p, err := r.Reconcile(context.Background(), x)
	if err != nil || p.CompletedStages != 1 || p.CompletedSteps != 1 || p.Checkpoint == "" {
		t.Fatal(p, err)
	}
	if err := r.Run(context.Background(), x, noReport); err != nil {
		t.Fatal(err)
	}
	p, err = r.Reconcile(context.Background(), x)
	if err != nil || p.CompletedStages != 13 || p.CompletedSteps != 13 {
		t.Fatal(p, err)
	}
	for _, s := range x.Recipe.Stages {
		if h.calls[s.ID] != 1 {
			t.Fatalf("stage replayed: %s %d", s.ID, h.calls[s.ID])
		}
	}
	if err := r.Run(context.Background(), x, noReport); err != nil {
		t.Fatal(err)
	}
	for _, n := range h.calls {
		if n != 1 {
			t.Fatal("completed phase replayed")
		}
	}
}
func TestDurableRuntimeResumesWithinStage(t *testing.T) {
	c, x, h := durableFixture(t.TempDir())
	x.Recipe.Stages[0].MaxSteps = 3
	r := runtimeFixture(t, c)
	if err := r.Run(context.Background(), x, func(context.Context, pipeline.Progress) error { return io.ErrClosedPipe }); !errors.Is(err, io.ErrClosedPipe) {
		t.Fatal(err)
	}
	x.Generation++
	p, err := r.Reconcile(context.Background(), x)
	if err != nil || p.StageSteps != 1 || p.CompletedStages != 0 {
		t.Fatal(p, err)
	}
	if err := r.Run(context.Background(), x, noReport); err != nil {
		t.Fatal(err)
	}
	p, err = r.Reconcile(context.Background(), x)
	if err != nil || p.CompletedSteps != 15 || p.CompletedStages != 13 || h.calls["stage-0"] != 2 {
		t.Fatal(p, err, h.calls)
	}
}
func TestDurableRuntimeRejectsIdentityChangesAndOldGeneration(t *testing.T) {
	c, x, _ := durableFixture(t.TempDir())
	r := runtimeFixture(t, c)
	x.Generation = 2
	if _, err := r.Reconcile(context.Background(), x); err != nil {
		t.Fatal(err)
	}
	old := x
	old.Generation = 1
	if _, err := r.Reconcile(context.Background(), old); !errors.Is(err, pipeline.ErrDurableFence) {
		t.Fatal(err)
	}
	for _, kind := range []string{"recipe", "target", "placement"} {
		t.Run(kind, func(t *testing.T) {
			y := x
			switch kind {
			case "recipe":
				y.Recipe.ID = "different"
			case "target":
				y.TargetSHA256 = ref("different-target").SHA256
			case "placement":
				y.Placement.MemoryBytes++
			}
			if _, err := r.Reconcile(context.Background(), y); !errors.Is(err, pipeline.ErrDurableIdentity) {
				t.Fatal(err)
			}
		})
	}
	other := x
	other.TenantID = "tenant-2"
	other.Generation = 1
	if _, err := r.Reconcile(context.Background(), other); err != nil {
		t.Fatal("tenants not isolated", err)
	}
}
func TestDurableRuntimeAdmissionRequiresAllPhasesAndSnapshotsWiring(t *testing.T) {
	c, x, h := durableFixture(t.TempDir())
	r := runtimeFixture(t, c)
	delete(c.Handlers, pipeline.PhaseMaster)
	if err := r.Admit(context.Background(), x.Recipe, x.Placement); err != nil {
		t.Fatal(err)
	}
	missing := runtimeFixture(t, c)
	if err := missing.Admit(context.Background(), x.Recipe, x.Placement); err == nil {
		t.Fatal("missing phase admitted")
	}
	deny := errors.New("format not qualified")
	h.admit = func(_ context.Context, r pipeline.Recipe, s pipeline.Stage, _ pipeline.Placement) error {
		r.Stages[0].ID = "mutated"
		if s.Format == "Q4_K_M" {
			return deny
		}
		return nil
	}
	if err := r.Admit(context.Background(), x.Recipe, x.Placement); !errors.Is(err, deny) {
		t.Fatal(err)
	}
	if x.Recipe.Stages[0].ID != "stage-0" {
		t.Fatal("handler mutated caller recipe")
	}
	if files, _ := os.ReadDir(c.Root); len(files) != 0 {
		t.Fatal("admission wrote files")
	}
}
func TestDurableRuntimeRefusesEmptyOrUnverifiedResults(t *testing.T) {
	for _, kind := range []string{"empty", "no-artifacts", "zero-steps", "semantic", "ignored-error", "oversized", "traversal", "symlink"} {
		t.Run(kind, func(t *testing.T) {
			c, x, h := durableFixture(t.TempDir())
			r := runtimeFixture(t, c)
			h.run = func(ctx context.Context, s pipeline.StageContext, commit func(context.Context, pipeline.StageResult) error) error {
				if kind == "empty" {
					return nil
				}
				if kind == "no-artifacts" {
					return commit(ctx, pipeline.StageResult{Steps: 1, Complete: true})
				}
				if kind == "ignored-error" {
					_ = commit(ctx, pipeline.StageResult{Steps: 1, Complete: true})
					return nil
				}
				v, err := writeFixtureResult(s, 1, true)
				if err != nil {
					return err
				}
				switch kind {
				case "zero-steps":
					v.Steps = 0
				case "oversized":
					v.Artifacts[0].Bytes = c.Limits.MaxArtifactBytes + 1
				case "traversal":
					v.Artifacts[0].Path = "../escape"
				case "symlink":
					target := v.Artifacts[0].Path
					v.Artifacts[0].Path = "link"
					if err := os.Symlink(target, filepath.Join(s.ArtifactDirectory, "link")); err != nil {
						return err
					}
				}
				return commit(ctx, v)
			}
			if kind == "semantic" {
				h.verify = func(context.Context, pipeline.StageContext, pipeline.StageReceipt) error {
					return errors.New("threshold failed")
				}
			}
			if err := r.Run(context.Background(), x, noReport); err == nil {
				t.Fatal("unverified phase accepted")
			}
			p, err := r.Reconcile(context.Background(), x)
			if err != nil || p.CompletedStages != 0 || p.CompletedSteps != 0 {
				t.Fatal(p, err)
			}
		})
	}
}
func TestDurableRuntimeRefusesCorruptedArtifactsAndReceipts(t *testing.T) {
	for _, kind := range []string{"artifact", "digest", "unknown", "truncated", "bound"} {
		t.Run(kind, func(t *testing.T) {
			c, x, _ := durableFixture(t.TempDir())
			r := runtimeFixture(t, c)
			if err := r.Run(context.Background(), x, func(context.Context, pipeline.Progress) error { return io.ErrClosedPipe }); err == nil {
				t.Fatal("expected interruption")
			}
			path := statePath(t, c.Root)
			switch kind {
			case "artifact":
				if err := os.WriteFile(filepath.Join(filepath.Dir(path), "artifacts", "stage-0-1.bin"), []byte("corrupt"), 0600); err != nil {
					t.Fatal(err)
				}
			case "digest":
				body, _ := os.ReadFile(path)
				body = []byte(strings.Replace(string(body), `"previous_sha256":""`, `"previous_sha256":"bad"`, 1))
				if err := os.WriteFile(path, body, 0600); err != nil {
					t.Fatal(err)
				}
			case "unknown":
				body, _ := os.ReadFile(path)
				body = append([]byte(`{"unknown":1,`), body[1:]...)
				if err := os.WriteFile(path, body, 0600); err != nil {
					t.Fatal(err)
				}
			case "truncated":
				if err := os.WriteFile(path, []byte(`{"version":1`), 0600); err != nil {
					t.Fatal(err)
				}
			case "bound":
				if err := os.WriteFile(path, make([]byte, 1025), 0600); err != nil {
					t.Fatal(err)
				}
				c.Limits.MaxStateBytes = 1024
				r = runtimeFixture(t, c)
			}
			x.Generation++
			if _, err := r.Reconcile(context.Background(), x); err == nil {
				t.Fatal("corrupt evidence accepted")
			}
		})
	}
}
func TestDurableRuntimeCancellationWaitsForHandlerAndKeepsUnfinishedState(t *testing.T) {
	c, x, h := durableFixture(t.TempDir())
	r := runtimeFixture(t, c)
	started := make(chan struct{})
	stopped := make(chan struct{})
	h.run = func(ctx context.Context, _ pipeline.StageContext, _ func(context.Context, pipeline.StageResult) error) error {
		close(started)
		<-ctx.Done()
		close(stopped)
		return ctx.Err()
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- r.Run(ctx, x, noReport) }()
	<-started
	if _, err := r.Reconcile(context.Background(), x); !errors.Is(err, pipeline.ErrDurableBusy) {
		t.Fatal(err)
	}
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("cancellation did not finish")
	}
	select {
	case <-stopped:
	default:
		t.Fatal("runtime detached handler")
	}
	x.Generation++
	p, err := r.Reconcile(context.Background(), x)
	if err != nil || p.CompletedStages != 0 {
		t.Fatal(p, err)
	}
}
func TestDurableRuntimeRecoversOutputBeforeReceiptCommit(t *testing.T) {
	c, x, h := durableFixture(t.TempDir())
	r := runtimeFixture(t, c)
	var orphan pipeline.StageResult
	h.run = func(_ context.Context, s pipeline.StageContext, _ func(context.Context, pipeline.StageResult) error) error {
		var e error
		orphan, e = writeFixtureResult(s, 1, true)
		return errors.Join(e, io.ErrClosedPipe)
	}
	if err := r.Run(context.Background(), x, noReport); !errors.Is(err, io.ErrClosedPipe) {
		t.Fatal(err)
	}
	h.run = nil
	h.reconcile = func(ctx context.Context, s pipeline.StageContext, commit func(context.Context, pipeline.StageResult) error) error {
		if s.Stage.ID == "stage-0" && s.Previous == nil {
			return commit(ctx, orphan)
		}
		return nil
	}
	x.Generation++
	p, err := r.Reconcile(context.Background(), x)
	if err != nil || p.CompletedStages != 1 {
		t.Fatal(p, err)
	}
	if err := r.Run(context.Background(), x, noReport); err != nil {
		t.Fatal(err)
	}
	if h.calls["stage-0"] != 1 {
		t.Fatal("recovered computation replayed")
	}
}
func TestDurableRuntimeProcessExclusionAndCrashReleasesLock(t *testing.T) {
	root := t.TempDir()
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestDurableRuntimeProcessHelper$")
	cmd.Env = append(os.Environ(), "DURABLE_RUNTIME_HELPER="+root)
	input, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	defer input.Close()
	output, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	defer cmd.Process.Kill()
	line, err := bufio.NewReader(output).ReadString('\n')
	if err != nil || line != "locked\n" {
		t.Fatal(line, err)
	}
	c, x, _ := durableFixture(root)
	r := runtimeFixture(t, c)
	x.Generation = 2
	if _, err := r.Reconcile(context.Background(), x); !errors.Is(err, pipeline.ErrDurableBusy) {
		t.Fatal(err)
	}
	if err := cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	if err := cmd.Wait(); err == nil {
		t.Fatal("helper unexpectedly exited successfully")
	}
	if _, err := r.Reconcile(context.Background(), x); err != nil {
		t.Fatal(err)
	}
	x.Generation = 1
	if _, err := r.Reconcile(context.Background(), x); !errors.Is(err, pipeline.ErrDurableFence) {
		t.Fatal(err)
	}
}
func TestDurableRuntimeProcessHelper(t *testing.T) {
	root := os.Getenv("DURABLE_RUNTIME_HELPER")
	if root == "" {
		return
	}
	c, x, h := durableFixture(root)
	r := runtimeFixture(t, c)
	h.run = func(context.Context, pipeline.StageContext, func(context.Context, pipeline.StageResult) error) error {
		fmt.Println("locked")
		_, err := io.Copy(io.Discard, os.Stdin)
		return errors.Join(err, io.ErrClosedPipe)
	}
	_ = r.Run(context.Background(), x, noReport)
}
func TestDurableRuntimeRejectsDuplicateJSONKeys(t *testing.T) {
	c, x, _ := durableFixture(t.TempDir())
	r := runtimeFixture(t, c)
	if _, err := r.Reconcile(context.Background(), x); err != nil {
		t.Fatal(err)
	}
	path := statePath(t, c.Root)
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var document map[string]json.RawMessage
	if err := json.Unmarshal(body, &document); err != nil {
		t.Fatal(err)
	}
	body = append([]byte(`{"version":1,`), body[1:]...)
	if err := os.WriteFile(path, body, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Reconcile(context.Background(), x); err == nil {
		t.Fatal("duplicate keys accepted")
	}
}

func TestDurableRuntimeReconcileAdmitsBeforeWritingOrCallingHandler(t *testing.T) {
	c, x, h := durableFixture(t.TempDir())
	r := runtimeFixture(t, c)
	denied := errors.New("qualification denied")
	h.admit = func(context.Context, pipeline.Recipe, pipeline.Stage, pipeline.Placement) error { return denied }
	h.reconcile = func(context.Context, pipeline.StageContext, func(context.Context, pipeline.StageResult) error) error {
		t.Fatal("unadmitted reconciliation")
		return nil
	}
	if _, err := r.Reconcile(context.Background(), x); !errors.Is(err, denied) {
		t.Fatal(err)
	}
	if files, err := os.ReadDir(c.Root); err != nil || len(files) != 0 {
		t.Fatal("admission created files", err)
	}
}

func TestDurableRuntimeResumesAfterProcessDiesBeforeReport(t *testing.T) {
	root := t.TempDir()
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestDurableRuntimeCommitCrashHelper$")
	cmd.Env = append(os.Environ(), "DURABLE_RUNTIME_COMMIT_CRASH="+root)
	body, err := cmd.CombinedOutput()
	var exit *exec.ExitError
	if !errors.As(err, &exit) || exit.ExitCode() != 17 {
		t.Fatalf("unexpected helper exit: %s %v", body, err)
	}
	c, x, h := durableFixture(root)
	x.Generation = 2
	r := runtimeFixture(t, c)
	p, err := r.Reconcile(context.Background(), x)
	if err != nil || p.CompletedStages != 1 || p.Checkpoint == "" {
		t.Fatal(p, err)
	}
	if err := r.Run(context.Background(), x, noReport); err != nil {
		t.Fatal(err)
	}
	if h.calls["stage-0"] != 0 {
		t.Fatal("durable computation replayed after process crash")
	}
}

func TestDurableRuntimeCommitCrashHelper(t *testing.T) {
	root := os.Getenv("DURABLE_RUNTIME_COMMIT_CRASH")
	if root == "" {
		return
	}
	c, x, _ := durableFixture(root)
	r := runtimeFixture(t, c)
	if err := r.Run(context.Background(), x, func(context.Context, pipeline.Progress) error { os.Exit(17); return nil }); err != nil {
		t.Fatal(err)
	}
	t.Fatal("helper reached end before exiting")
}
