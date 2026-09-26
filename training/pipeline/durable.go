package pipeline

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"reflect"
	"slices"
	"sync"
	"time"
)

var (
	// ErrDurableConfiguration means the runtime lacks explicit installation bounds.
	ErrDurableConfiguration = errors.New("pipeline runtime: invalid configuration")
	// ErrDurableBusy means another process owns this run's filesystem lock.
	ErrDurableBusy = errors.New("pipeline runtime: run is locked")
	// ErrDurableFence means the request is older than the persisted generation.
	ErrDurableFence = errors.New("pipeline runtime: stale generation")
	// ErrDurableIdentity means the run was previously bound to another execution.
	ErrDurableIdentity = errors.New("pipeline runtime: execution identity differs")
	// ErrDurableReceipt means persisted evidence is malformed or cannot be verified.
	ErrDurableReceipt = errors.New("pipeline runtime: invalid receipt")
	// ErrDurableIncomplete means a phase returned without a verified completion.
	ErrDurableIncomplete = errors.New("pipeline runtime: phase did not complete")
)

// DurableLimits bounds metadata and the immutable artifacts in one run.
type DurableLimits struct {
	MaxStateBytes          int64
	MaxReceipts            int
	MaxArtifactsPerReceipt int
	MaxArtifactBytes       int64
	MaxTotalArtifactBytes  int64
}

// DurableConfig binds an installed backend and explicitly injected phase handlers.
// Root is one capacity domain: every runtime and worker using that capacity must
// share this private local directory. Independent nodes or capacity domains may
// use different roots only when their capacity owner explicitly assigns them.
// The root lock serializes all handler effects across runs, tenants and backends;
// it cannot protect against external processes that do not share this root.
// External writers must not replace its lock files or alter committed artifacts.
// Network filesystems are not supported. No field has an implicit default.
type DurableConfig struct {
	Root                string
	Backend             string
	RuntimeSHA256       string
	QualificationSHA256 string
	Limits              DurableLimits
	Handlers            map[Phase]StageHandler
}

// StageArtifact is a nonempty immutable file relative to ArtifactDirectory.
type StageArtifact struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
	Bytes  int64  `json:"bytes"`
}

// StageResult describes a materialized checkpoint, not a proposed update.
// Complete is accepted only after the handler verifies the scientific criteria.
type StageResult struct {
	Steps     int64           `json:"steps"`
	Complete  bool            `json:"complete"`
	Artifacts []StageArtifact `json:"artifacts"`
}

// ReceiptIdentity pins a receipt to one immutable execution and publishing fence.
type ReceiptIdentity struct {
	RunID           string `json:"run_id"`
	TenantID        string `json:"tenant_id"`
	RecipeSHA256    string `json:"recipe_sha256"`
	TargetSHA256    string `json:"target_sha256"`
	PlacementSHA256 string `json:"placement_sha256"`
	Generation      uint64 `json:"generation"`
}

// StageReceipt binds a checkpoint to its ordered predecessor and execution.
// SHA256 hashes every field except SHA256 itself; Checkpoint in Progress is this hash.
type StageReceipt struct {
	Version        int             `json:"version"`
	Identity       ReceiptIdentity `json:"identity"`
	StageID        string          `json:"stage_id"`
	Result         StageResult     `json:"result"`
	PreviousSHA256 string          `json:"previous_sha256"`
	SHA256         string          `json:"sha256"`
}

// StageContext gives a handler detached copies of the execution and prior evidence.
// Parent follows the recipe lineage, including independent variants from Master.
// Previous is the last checkpoint within this stage. Completed contains earlier
// stage completion receipts. Only ArtifactDirectory is available for new files.
type StageContext struct {
	Execution         Execution
	Stage             Stage
	ArtifactDirectory string
	Parent            *StageReceipt
	Previous          *StageReceipt
	Completed         []StageReceipt
}

// StageHandler owns the scientific work and qualification of one phase.
// Admit must check the exact recipe, format, placement and independent qualification
// without launching work. Verify must inspect artifact contents and enforce phase
// acceptance criteria; a counter or hash alone is not scientific completion.
// Reconcile may recover existing output after an interrupted commit, but must not
// compute new steps. Run resumes from Previous. Both methods must stop all compute
// before returning, honor cancellation, and stop immediately when commit fails.
// Artifact files are immutable, fsynced before commit, and confined to the supplied
// directory. A handler must never detach work or overwrite a previous checkpoint.
type StageHandler interface {
	Admit(context.Context, Recipe, Stage, Placement) error
	Verify(context.Context, StageContext, StageReceipt) error
	Reconcile(context.Context, StageContext, func(context.Context, StageResult) error) error
	Run(context.Context, StageContext, func(context.Context, StageResult) error) error
}

// DurableRuntime coordinates verified phase implementations. It provides no
// scientific implementation and cannot admit a recipe with a missing handler.
type DurableRuntime struct{ config DurableConfig }

var _ Runtime = (*DurableRuntime)(nil)

// NewDurableRuntime snapshots explicit wiring without creating files or launching work.
func NewDurableRuntime(config DurableConfig) (*DurableRuntime, error) {
	l := config.Limits
	if !durableLockSupported || !filepath.IsAbs(config.Root) || !identifier(config.Backend) || !digest(config.RuntimeSHA256) || !digest(config.QualificationSHA256) || l.MaxStateBytes < 1024 || l.MaxStateBytes > 64<<20 || l.MaxReceipts < 13 || l.MaxReceipts > 1_000_000 || l.MaxArtifactsPerReceipt < 1 || l.MaxArtifactsPerReceipt > 4096 || l.MaxArtifactBytes < 1 || l.MaxArtifactBytes > 1<<60 || l.MaxTotalArtifactBytes < l.MaxArtifactBytes || len(config.Handlers) == 0 {
		return nil, ErrDurableConfiguration
	}
	config.Root = filepath.Clean(config.Root)
	copyHandlers := make(map[Phase]StageHandler, len(config.Handlers))
	for phase, handler := range config.Handlers {
		if !knownPhase(phase) || nilHandler(handler) {
			return nil, ErrDurableConfiguration
		}
		copyHandlers[phase] = handler
	}
	config.Handlers = copyHandlers
	return &DurableRuntime{config}, nil
}

// Admit requires every phase implementation and validates its exact qualification.
func (r *DurableRuntime) Admit(ctx context.Context, recipe Recipe, placement Placement) error {
	if err := r.validate(ctx, recipe, placement); err != nil {
		return err
	}
	for _, stage := range recipe.Stages {
		handler := r.config.Handlers[stage.Phase]
		if nilHandler(handler) {
			return fmt.Errorf("%w: missing phase %s", ErrDurableConfiguration, stage.Phase)
		}
		if err := handler.Admit(ctx, cloneRecipe(recipe), cloneStage(stage), clonePlacement(placement)); err != nil {
			return fmt.Errorf("pipeline runtime: admit stage %s: %w", stage.ID, err)
		}
		if err := ctx.Err(); err != nil {
			return err
		}
	}
	return nil
}

// Reconcile waits cancelably for shared capacity, fences older generations and
// verifies all stored artifacts and receipts while retaining both OS locks.
// It also lets the next phase recover an output whose receipt commit was interrupted.
func (r *DurableRuntime) Reconcile(ctx context.Context, execution Execution) (Progress, error) {
	if err := r.Admit(ctx, execution.Recipe, execution.Placement); err != nil {
		return Progress{}, err
	}
	s, err := r.openExecution(ctx, execution)
	if err != nil {
		return Progress{}, err
	}
	defer s.close()
	if err := s.verify(ctx); err != nil {
		return Progress{}, err
	}
	if index := s.nextStage(); index < len(s.execution.Recipe.Stages) {
		if err := s.invoke(ctx, index, true, nil); err != nil {
			return Progress{}, err
		}
	}
	return s.progress(), nil
}

// Run holds the run and shared capacity locks until all handlers stop. Contention
// for this run fails immediately; contention for capacity waits until cancellation
// or release, allowing the caller's lease monitor to remain active.
// Each report follows a verified, fsynced receipt, so a report failure resumes
// without replaying the step.
func (r *DurableRuntime) Run(ctx context.Context, execution Execution, report func(context.Context, Progress) error) error {
	if report == nil {
		return ErrDurableConfiguration
	}
	if err := r.Admit(ctx, execution.Recipe, execution.Placement); err != nil {
		return err
	}
	s, err := r.openExecution(ctx, execution)
	if err != nil {
		return err
	}
	defer s.close()
	if err := s.verify(ctx); err != nil {
		return err
	}
	if len(s.state.Receipts) > 0 {
		if err := report(ctx, s.progress()); err != nil {
			return err
		}
	}
	for index := s.nextStage(); index < len(s.execution.Recipe.Stages); index = s.nextStage() {
		if err := s.invoke(ctx, index, true, report); err != nil {
			return err
		}
		if s.nextStage() != index {
			continue
		}
		if err := s.invoke(ctx, index, false, report); err != nil {
			return err
		}
		if s.nextStage() == index {
			return ErrDurableIncomplete
		}
	}
	return s.verify(ctx)
}

func (r *DurableRuntime) validate(ctx context.Context, recipe Recipe, p Placement) error {
	if r == nil || ctx == nil {
		return ErrDurableConfiguration
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := recipe.Validate(); err != nil {
		return err
	}
	if p.Backend != r.config.Backend || p.RuntimeSHA256 != r.config.RuntimeSHA256 || p.QualificationSHA256 != r.config.QualificationSHA256 || p.MemoryBytes <= 0 || p.MaxTokens < 2 || p.MaxTokens > 1<<20 || len(p.Nodes) > 1024 {
		return ErrDurableConfiguration
	}
	seen := map[string]bool{}
	for _, node := range p.Nodes {
		if !identifier(node) || seen[node] {
			return ErrDurableConfiguration
		}
		seen[node] = true
	}
	for _, stage := range recipe.Stages {
		if stage.MaxTokens > p.MaxTokens || nilHandler(r.config.Handlers[stage.Phase]) {
			return ErrDurableConfiguration
		}
	}
	return nil
}

func (s *durableSession) invoke(ctx context.Context, index int, reconcile bool, report func(context.Context, Progress) error) error {
	stage := s.execution.Recipe.Stages[index]
	bounded, cancel := context.WithTimeout(ctx, time.Duration(stage.TimeoutSeconds)*time.Second)
	defer cancel()
	handler := s.runtime.config.Handlers[stage.Phase]
	var mu sync.Mutex
	var commitError error
	active := true
	commit := func(call context.Context, result StageResult) error {
		mu.Lock()
		defer mu.Unlock()
		if !active {
			return ErrDurableFence
		}
		if commitError != nil {
			return commitError
		}
		commitError = func() error {
			if err := bounded.Err(); err != nil {
				return err
			}
			if call == nil {
				return ErrDurableConfiguration
			}
			if err := call.Err(); err != nil {
				return err
			}
			if err := s.commit(bounded, index, result); err != nil {
				return err
			}
			if err := bounded.Err(); err != nil {
				return err
			}
			if report != nil {
				return report(bounded, s.progress())
			}
			return nil
		}()
		return commitError
	}
	var runError error
	if reconcile {
		runError = handler.Reconcile(bounded, s.stageContext(index, s.state.Receipts), commit)
	} else {
		runError = handler.Run(bounded, s.stageContext(index, s.state.Receipts), commit)
	}
	mu.Lock()
	active = false
	err := errors.Join(runError, commitError, bounded.Err())
	mu.Unlock()
	return err
}

func (s *durableSession) stageContext(index int, receipts []StageReceipt) StageContext {
	stage := s.execution.Recipe.Stages[index]
	result := StageContext{Execution: cloneExecution(s.execution), Stage: cloneStage(stage), ArtifactDirectory: filepath.Join(s.directory, "artifacts")}
	for _, receipt := range receipts {
		copy := cloneReceipt(receipt)
		if receipt.StageID == stage.ID {
			result.Previous = &copy
		} else if receipt.Result.Complete {
			result.Completed = append(result.Completed, copy)
			if receipt.StageID == stage.ParentStage {
				parent := cloneReceipt(receipt)
				result.Parent = &parent
			}
		}
	}
	return result
}

func (s *durableSession) nextStage() int {
	completed := 0
	for _, receipt := range s.state.Receipts {
		if receipt.Result.Complete {
			completed++
		}
	}
	return completed
}

func (s *durableSession) progress() Progress {
	p := Progress{StageID: s.execution.Recipe.Stages[0].ID}
	for _, receipt := range s.state.Receipts {
		if receipt.StageID != p.StageID {
			p.StageSteps = 0
		}
		p.CompletedSteps += receipt.Result.Steps - p.StageSteps
		p.StageID, p.StageSteps, p.Checkpoint = receipt.StageID, receipt.Result.Steps, receipt.SHA256
		if receipt.Result.Complete {
			p.CompletedStages++
		}
	}
	return p
}

func knownPhase(p Phase) bool {
	return slices.Contains([]Phase{PhaseSFT, PhaseTeacherCache, PhaseAlignment, PhaseFusion, PhaseRecovery, PhaseMaster, PhaseCalibration, PhaseQuantize, PhaseVariantRecovery}, p)
}
func nilHandler(h StageHandler) bool {
	if h == nil {
		return true
	}
	v := reflect.ValueOf(h)
	switch v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return v.IsNil()
	}
	return false
}
func cloneRecipe(r Recipe) Recipe {
	r.Teachers = slices.Clone(r.Teachers)
	r.Stages = slices.Clone(r.Stages)
	for i := range r.Stages {
		r.Stages[i] = cloneStage(r.Stages[i])
	}
	return r
}
func cloneStage(s Stage) Stage             { s.Inputs = slices.Clone(s.Inputs); return s }
func clonePlacement(p Placement) Placement { p.Nodes = slices.Clone(p.Nodes); return p }
func cloneExecution(e Execution) Execution {
	e.Recipe = cloneRecipe(e.Recipe)
	e.Placement = clonePlacement(e.Placement)
	return e
}
func cloneReceipt(r StageReceipt) StageReceipt {
	r.Result.Artifacts = slices.Clone(r.Result.Artifacts)
	return r
}
func jsonSHA256(v any) string {
	body, _ := json.Marshal(v)
	sum := sha256.Sum256(body)
	return hex.EncodeToString(sum[:])
}
func receiptSHA256(r StageReceipt) string { r.SHA256 = ""; return jsonSHA256(r) }
