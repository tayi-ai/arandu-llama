package pipeline

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"reflect"
)

// ErrStageDispatch identifies missing wiring or a divergent stage/receipt context.
var ErrStageDispatch = errors.New("pipeline: stage dispatch refused")

// StageDispatch selects explicitly installed handlers for one phase by stage ID.
// It snapshots the map, not the handler instances, which remain caller-owned.
// DurableRuntime still owns execution locks, receipt chains and artifact checks;
// each selected handler still owns scientific admission and verification.
type StageDispatch struct {
	phase    Phase
	handlers map[string]StageHandler
}

var _ StageHandler = (*StageDispatch)(nil)

// NewStageDispatch binds one phase to explicit stage IDs without discovery or
// defaults. Construction performs no admission, I/O or handler execution.
func NewStageDispatch(phase Phase, handlers map[string]StageHandler) (*StageDispatch, error) {
	if !knownPhase(phase) || len(handlers) == 0 || len(handlers) > 13 {
		return nil, ErrStageDispatch
	}
	d := &StageDispatch{phase: phase, handlers: make(map[string]StageHandler, len(handlers))}
	for id, handler := range handlers {
		if !identifier(id) || nilHandler(handler) {
			return nil, ErrStageDispatch
		}
		d.handlers[id] = handler
	}
	return d, nil
}

func (d *StageDispatch) selectHandler(ctx context.Context, recipe Recipe, stage Stage) (StageHandler, error) {
	if d == nil || ctx == nil || !knownPhase(d.phase) || len(d.handlers) == 0 {
		return nil, ErrStageDispatch
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := recipe.Validate(); err != nil {
		return nil, errors.Join(ErrStageDispatch, err)
	}
	if stage.Phase != d.phase {
		return nil, ErrStageDispatch
	}
	var selected StageHandler
	for _, exact := range recipe.Stages {
		if exact.Phase != d.phase {
			continue
		}
		handler := d.handlers[exact.ID]
		if nilHandler(handler) {
			return nil, fmt.Errorf("%w: missing stage %s", ErrStageDispatch, exact.ID)
		}
		if exact.ID == stage.ID {
			if !reflect.DeepEqual(exact, stage) {
				return nil, ErrStageDispatch
			}
			selected = handler
		}
	}
	if nilHandler(selected) {
		return nil, ErrStageDispatch
	}
	return selected, nil
}

// Admit checks every stage of this phase in recipe order before any effect.
// The full runtime must additionally admit all other phases of the recipe.
func (d *StageDispatch) Admit(ctx context.Context, recipe Recipe, stage Stage, placement Placement) error {
	if _, err := d.selectHandler(ctx, recipe, stage); err != nil {
		return err
	}
	for _, exact := range recipe.Stages {
		if exact.Phase != d.phase {
			continue
		}
		if err := d.handlers[exact.ID].Admit(ctx, cloneRecipe(recipe), cloneStage(exact), clonePlacement(placement)); err != nil {
			return fmt.Errorf("pipeline dispatch: admit stage %s: %w", exact.ID, err)
		}
		if err := ctx.Err(); err != nil {
			return err
		}
	}
	return nil
}

func (d *StageDispatch) admittedHandler(ctx context.Context, c StageContext) (StageHandler, error) {
	handler, err := d.selectHandler(ctx, c.Execution.Recipe, c.Stage)
	if err != nil {
		return nil, err
	}
	x := c.Execution
	if !identifier(x.RunID) || !identifier(x.TenantID) || x.Generation == 0 || !digest(x.TargetSHA256) || !filepath.IsAbs(c.ArtifactDirectory) {
		return nil, ErrStageDispatch
	}
	if err := d.Admit(ctx, x.Recipe, c.Stage, x.Placement); err != nil {
		return nil, err
	}
	return handler, nil
}

// Verify refuses another stage or execution's receipt before delegating content
// verification. Historical publishing generations remain valid during replay.
func (d *StageDispatch) Verify(ctx context.Context, c StageContext, receipt StageReceipt) error {
	handler, err := d.admittedHandler(ctx, c)
	if err != nil {
		return err
	}
	x := c.Execution
	recipeSHA, _ := x.Recipe.Digest()
	identity := ReceiptIdentity{RunID: x.RunID, TenantID: x.TenantID, RecipeSHA256: recipeSHA, TargetSHA256: x.TargetSHA256, PlacementSHA256: jsonSHA256(x.Placement), Generation: x.Generation}
	if receipt.Version != 1 || receipt.StageID != c.Stage.ID || !sameIdentity(receipt.Identity, identity) || receipt.Identity.Generation == 0 || receipt.Identity.Generation > x.Generation || receipt.SHA256 != receiptSHA256(receipt) || receipt.Result.Steps < 1 || receipt.Result.Steps > int64(c.Stage.MaxSteps) {
		return ErrStageDispatch
	}
	return handler.Verify(ctx, cloneDispatchContext(c), cloneReceipt(receipt))
}

// Reconcile forwards only to the exact admitted stage's recovery implementation.
func (d *StageDispatch) Reconcile(ctx context.Context, c StageContext, commit func(context.Context, StageResult) error) error {
	if commit == nil {
		return ErrStageDispatch
	}
	handler, err := d.admittedHandler(ctx, c)
	if err != nil {
		return err
	}
	return handler.Reconcile(ctx, cloneDispatchContext(c), commit)
}

// Run forwards only to the exact admitted stage's scientific implementation.
func (d *StageDispatch) Run(ctx context.Context, c StageContext, commit func(context.Context, StageResult) error) error {
	if commit == nil {
		return ErrStageDispatch
	}
	handler, err := d.admittedHandler(ctx, c)
	if err != nil {
		return err
	}
	return handler.Run(ctx, cloneDispatchContext(c), commit)
}

func cloneDispatchContext(c StageContext) StageContext {
	c.Execution, c.Stage = cloneExecution(c.Execution), cloneStage(c.Stage)
	if c.Parent != nil {
		parent := cloneReceipt(*c.Parent)
		c.Parent = &parent
	}
	if c.Previous != nil {
		previous := cloneReceipt(*c.Previous)
		c.Previous = &previous
	}
	if c.Completed != nil {
		completed := make([]StageReceipt, len(c.Completed))
		for i, receipt := range c.Completed {
			completed[i] = cloneReceipt(receipt)
		}
		c.Completed = completed
	}
	return c
}
