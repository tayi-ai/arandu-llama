package pipeline

import (
	"context"
	"errors"
	"reflect"
	"slices"
)

// Run restores the verified preceding vector, executes real constrained Steps,
// closes each model session, and publishes each immutable unit before commit.
// A commit error returns immediately; Reconcile reports that unit without
// opening another session or recalculating its accepted update.
func (h *ConstrainedStage) Run(ctx context.Context, c StageContext, commit func(context.Context, StageResult) error) error {
	if err := h.enter(ctx); err != nil {
		return err
	}
	defer func() { <-h.gate }()
	if commit == nil {
		return ErrConstrained
	}
	c = constrainedContext(c)
	state, err := h.scan(ctx, c)
	if err != nil {
		return err
	}
	previous, err := h.previous(c, state)
	if err != nil {
		return err
	}
	if len(state.results) > previous {
		if err := h.syncRecovered(ctx, c, state.results[previous]); err != nil {
			return err
		}
		if err := commit(ctx, constrainedResult(state.results[previous])); err != nil {
			return err
		}
	}
	identity, err := h.stageIdentity(c)
	if err != nil {
		return err
	}
	for step := len(state.results) + 1; step <= len(h.config.Protocol.Updates); step++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		files, initial, bindings, err := h.resolvedInputs(ctx, c, step)
		if err != nil {
			return err
		}
		priorSHA := h.config.Protocol.InitialParametersSHA256
		if bindings != nil {
			priorSHA = bindings.InitialParametersSHA256
		}
		if step > 1 {
			priorSHA = state.last.ParametersSHA256
		}
		intent, err := h.beginAttempt(ctx, c, step, state.lastSHA, priorSHA, bindings)
		if err != nil {
			return err
		}
		prior, receipt, err := h.compute(ctx, c, step, state.parameters, priorSHA, files, initial)
		if err != nil {
			return err
		}
		// Detect changed inputs before accepting any durable model output. The
		// qualified Factory additionally verifies bytes during its own reads.
		_, _, afterBindings, err := h.resolvedInputs(ctx, c, step)
		if err != nil || !reflect.DeepEqual(bindings, afterBindings) {
			return errors.Join(ErrConstrained, err)
		}
		parametersSHA, err := h.parameterSHA(receipt.Parameters)
		if err != nil {
			return err
		}
		parameters := receipt.Parameters
		receipt.Parameters = nil
		evidence := ConstrainedEvidence{Version: h.config.Protocol.Version, Identity: identity, StageID: h.stage.ID, Step: step,
			ProtocolSHA256: h.config.ProtocolSHA256, UpdateSHA256: jsonSHA256(h.config.Protocol.Updates[step-1]), ParentSHA256: c.Parent.SHA256,
			PreviousSHA256: state.lastSHA, PriorParametersSHA256: priorSHA, ParametersSHA256: parametersSHA, Intent: intent, Prior: prior, Receipt: receipt, Bindings: bindings}
		result, err := h.persist(ctx, c, evidence, parameters)
		if err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := commit(ctx, constrainedResult(result)); err != nil {
			return err
		}
		state.results = append(state.results, result)
		state.last, state.lastSHA, state.parameters = evidence, result.Artifacts[1].SHA256, parameters
	}
	return ctx.Err()
}

// Reconcile only verifies and republishes a single fully persisted, unreported
// unit. It never loads a model, advances a rejected step or repairs partial data.
func (h *ConstrainedStage) Reconcile(ctx context.Context, c StageContext, commit func(context.Context, StageResult) error) error {
	if err := h.enter(ctx); err != nil {
		return err
	}
	defer func() { <-h.gate }()
	if commit == nil {
		return ErrConstrained
	}
	c = constrainedContext(c)
	state, err := h.scan(ctx, c)
	if err != nil {
		return err
	}
	previous, err := h.previous(c, state)
	if err != nil {
		return err
	}
	if len(state.results) > previous {
		if err := h.syncRecovered(ctx, c, state.results[previous]); err != nil {
			return err
		}
		return commit(ctx, constrainedResult(state.results[previous]))
	}
	return ctx.Err()
}

// Verify validates the recorded numerical acceptance, complete parameter bytes
// and predecessor chain. The stored KKT certificate and real-margin readings
// remain evidence from Factory's qualified backend, not a fresh model evaluation.
// Verify is read-only and may be called synchronously from the commit callback.
func (h *ConstrainedStage) Verify(ctx context.Context, c StageContext, receipt StageReceipt) error {
	if h == nil || ctx == nil {
		return ErrConstrained
	}
	c = constrainedContext(c)
	state, err := h.scan(ctx, c)
	if err != nil {
		return err
	}
	identity, err := h.stageIdentity(c)
	if err != nil || receipt.Version != 1 || receipt.StageID != h.stage.ID || receipt.SHA256 != receiptSHA256(receipt) ||
		!sameIdentity(receipt.Identity, identity) || receipt.Identity.Generation == 0 || receipt.Identity.Generation > identity.Generation || receipt.Result.Steps < 1 || receipt.Result.Steps > int64(len(state.results)) {
		return ErrConstrained
	}
	if !constrainedSameResult(receipt.Result, state.results[receipt.Result.Steps-1]) {
		return ErrConstrained
	}
	return ctx.Err()
}

func (h *ConstrainedStage) previous(c StageContext, state constrainedState) (int, error) {
	previous := 0
	if c.Previous != nil {
		r := c.Previous
		identity, err := h.stageIdentity(c)
		if err != nil || r.Version != 1 || r.StageID != h.stage.ID || r.SHA256 != receiptSHA256(*r) || !sameIdentity(r.Identity, identity) || r.Identity.Generation == 0 || r.Identity.Generation > identity.Generation ||
			r.Result.Steps < 1 || r.Result.Steps > int64(len(state.results)) || !constrainedSameResult(r.Result, state.results[r.Result.Steps-1]) {
			return 0, ErrConstrained
		}
		previous = int(r.Result.Steps)
	}
	if len(state.results) < previous || len(state.results) > previous+1 {
		return 0, ErrConstrained
	}
	return previous, nil
}

func (h *ConstrainedStage) compute(ctx context.Context, c StageContext, step int, parameters []float32, expected string, sources []ConstrainedSourceFile, initial ConstrainedInitialCheckpoint) (prior []float32, receipt StepReceipt, err error) {
	session, factoryErr := h.config.Factory(ctx, ConstrainedRequest{Context: constrainedContext(c), Step: step, Parameters: slices.Clone(parameters), Sources: slices.Clone(sources), Initial: initial})
	if session.Close != nil {
		defer func() { err = errors.Join(err, session.Close()) }()
	}
	if factoryErr != nil {
		return nil, receipt, factoryErr
	}
	if session.Close == nil || session.Model == nil {
		return nil, receipt, ErrConstrained
	}
	value := reflect.ValueOf(session.Model)
	if (value.Kind() == reflect.Pointer || value.Kind() == reflect.Interface) && value.IsNil() {
		return nil, receipt, ErrConstrained
	}
	if err := ctx.Err(); err != nil {
		return nil, receipt, err
	}
	if parameters != nil {
		if err := session.Model.Install(ctx, slices.Clone(parameters)); err != nil {
			return nil, receipt, err
		}
	}
	prior, err = session.Model.Parameters(ctx)
	if err != nil {
		return nil, receipt, err
	}
	prior = slices.Clone(prior)
	actual, err := h.parameterSHA(prior)
	if err != nil || actual != expected {
		return nil, receipt, ErrConstrained
	}
	config := h.config.Protocol.Updates[step-1].Config
	receipt, err = Step(ctx, session.Model, config)
	if err != nil {
		return prior, receipt, err
	}
	installed, err := session.Model.Parameters(ctx)
	if err != nil {
		return prior, receipt, err
	}
	actual, err = h.parameterSHA(installed)
	wanted, wantErr := h.parameterSHA(receipt.Parameters)
	if err != nil || wantErr != nil || actual != wanted {
		return prior, receipt, ErrConstrained
	}
	return prior, receipt, ctx.Err()
}

func constrainedContext(c StageContext) StageContext {
	c.Execution, c.Stage = cloneExecution(c.Execution), cloneStage(c.Stage)
	if c.Parent != nil {
		r := cloneReceipt(*c.Parent)
		c.Parent = &r
	}
	if c.Previous != nil {
		r := cloneReceipt(*c.Previous)
		c.Previous = &r
	}
	c.Completed = slices.Clone(c.Completed)
	for i := range c.Completed {
		c.Completed[i] = cloneReceipt(c.Completed[i])
	}
	return c
}
func constrainedResult(r StageResult) StageResult { r.Artifacts = slices.Clone(r.Artifacts); return r }
