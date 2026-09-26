package protection

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// ErrCandidate identifies an invalid candidate, floor set or callback contract.
var ErrCandidate = errors.New("protection: invalid candidate evaluation")

// ErrRejected identifies a real margin below its anchored floor.
var ErrRejected = errors.New("protection: real candidate violates protection floor")

// ErrRollback means restoration failed and the caller must treat model state as
// unknown. The original rejection and restoration error remain discoverable.
var ErrRollback = errors.New("protection: candidate rollback failed")

// ErrCallbackPanic identifies a callback panic converted to a transactional error.
var ErrCallbackPanic = errors.New("protection: candidate callback panicked")

// Floor identifies one immutable, pre-update protection threshold.
type Floor struct {
	ID    string
	Value float64
}

// Margin identifies one real model measurement under the installed candidate.
type Margin struct {
	ID    string
	Value float64
}

// CandidateCallbacks operates on the same serialized model parameter vector.
// Apply materializes the supplied candidate; Measure executes every protection
// example with fresh model state and returns exactly the anchored IDs; Restore
// reinstalls the supplied prior parameters even after a partially failed Apply.
// Callbacks must honor context deadlines. Restore receives a fresh 30-second
// deadline independent of cancellation of the candidate operation. A blocked
// callback cannot be forcibly interrupted by Go or safely run in the background.
//
// Callers must serialize the entire evaluation with all model access and must
// supply the actual prior parameters. Callbacks may not mutate optimizer moments,
// counters or unrelated state: those are outside this parameter transaction.
// The helper owns no optimizer and never advances one. Each parameter argument
// is a private copy; callbacks must not retain it for later mutation.
type CandidateCallbacks struct {
	Apply   func(context.Context, []float32) error
	Measure func(context.Context) ([]Margin, error)
	Restore func(context.Context, []float32) error
}

// Evaluation reports measured margins in floor order, acceptance and restoration.
// Restored is true only if Restore returned successfully after a rejected trial.
type Evaluation struct {
	Accepted bool
	Restored bool
	Margins  []Margin
}

// EvaluateCandidate installs a candidate and checks all real margins against the
// complete floor set with no numerical allowance below a floor. It snapshots the
// prior parameters, candidate and floors before any callback. Errors, panics,
// cancellation, missing/extra/duplicate IDs, nonfinite margins and floor failures
// after Apply begins always trigger restoration of the immutable prior snapshot.
// The returned error joins the original cause with ErrRollback if restoring fails.
// Input validation and already-cancelled calls have no external effect.
func EvaluateCandidate(ctx context.Context, prior, candidate []float32, floors []Floor, callbacks CandidateCallbacks) (evaluation Evaluation, err error) {
	if ctx == nil || callbacks.Apply == nil || callbacks.Measure == nil || callbacks.Restore == nil || len(prior) == 0 || len(prior) != len(candidate) || len(prior) > maxParameters || len(floors) == 0 || len(floors) > maxConstraints {
		return evaluation, ErrCandidate
	}
	if err := ctx.Err(); err != nil {
		return evaluation, err
	}
	before := append([]float32(nil), prior...)
	proposed := append([]float32(nil), candidate...)
	anchored := append([]Floor(nil), floors...)
	for i := range before {
		if err := checkContext(ctx, i); err != nil {
			return evaluation, err
		}
		if !finite(float64(before[i])) || !finite(float64(proposed[i])) {
			return evaluation, ErrCandidate
		}
	}
	ids := make(map[string]int, len(anchored))
	for i, floor := range anchored {
		if _, duplicate := ids[floor.ID]; duplicate || floor.ID == "" || len(floor.ID) > 1024 || !finite(floor.Value) {
			return evaluation, ErrCandidate
		}
		ids[floor.ID] = i
	}
	if err := ctx.Err(); err != nil {
		return evaluation, err
	}
	defer func() {
		if evaluation.Accepted {
			return
		}
		restoreContext, cancel := context.WithTimeout(context.WithoutCancel(ctx), 30*time.Second)
		defer cancel()
		if restoreErr := callParameters(callbacks.Restore, restoreContext, before); restoreErr != nil {
			err = errors.Join(err, ErrRollback, restoreErr)
		} else {
			evaluation.Restored = true
		}
	}()
	if err := callParameters(callbacks.Apply, ctx, proposed); err != nil {
		return evaluation, err
	}
	if err := ctx.Err(); err != nil {
		return evaluation, err
	}
	measured, err := callMeasure(callbacks.Measure, ctx)
	if err != nil {
		return evaluation, err
	}
	if err := ctx.Err(); err != nil {
		return evaluation, err
	}
	if len(measured) != len(anchored) {
		return evaluation, fmt.Errorf("%w: protection margin count differs", ErrCandidate)
	}
	evaluation.Margins = make([]Margin, len(anchored))
	seen := make([]bool, len(anchored))
	// Validate the entire ID set and every finite reading before any floor check.
	for _, margin := range measured {
		index, exists := ids[margin.ID]
		if !exists || seen[index] || !finite(margin.Value) {
			return evaluation, fmt.Errorf("%w: protection margin IDs or values differ", ErrCandidate)
		}
		seen[index] = true
		evaluation.Margins[index] = margin
	}
	for i, floor := range anchored {
		if evaluation.Margins[i].Value < floor.Value {
			return evaluation, fmt.Errorf("%w: %s", ErrRejected, floor.ID)
		}
	}
	if err := ctx.Err(); err != nil {
		return evaluation, err
	}
	evaluation.Accepted = true
	return evaluation, nil
}

func callParameters(callback func(context.Context, []float32) error, ctx context.Context, values []float32) (err error) {
	defer func() {
		if recover() != nil {
			err = ErrCallbackPanic
		}
	}()
	return callback(ctx, values)
}

func callMeasure(callback func(context.Context) ([]Margin, error), ctx context.Context) (margins []Margin, err error) {
	defer func() {
		if recover() != nil {
			margins, err = nil, ErrCallbackPanic
		}
	}()
	return callback(ctx)
}
