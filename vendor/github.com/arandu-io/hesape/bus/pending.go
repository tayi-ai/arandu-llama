package bus

import (
	"context"
	"fmt"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/database"
)

// PendingBatch is a batch being described, before anything is written.
//
// It collects the first error it makes rather than returning one per call --
// Add cannot both chain and return an error, and
// a builder whose every step has to be checked is a builder nobody uses.
// Dispatch is where the error surfaces, and Dispatch is the call that could act
// on it anyway.
type PendingBatch struct {
	name    string
	jobs    []Step
	options BatchOptions
	events  EventRecorder
	err     error
}

// NewBatch starts describing a batch.
//
// Dispatcher.Batch takes the jobs and Name sets the name; this is the
// constructor for the common case of naming it first. The name is for people:
// it is what the batch is called on a dashboard, and nothing looks a batch up
// by it.
func NewBatch(name string) *PendingBatch {
	return &PendingBatch{name: name}
}

// Add puts a job in the batch. The payload is serialized here, so a value that
// cannot be is refused at the call site that built it rather than in a worker.
func (p *PendingBatch) Add(name string, payload any) *PendingBatch {
	s, err := step("", name, payload)
	if err != nil {
		p.fail(err)
		return p
	}
	p.jobs = append(p.jobs, s)
	return p
}

// AddStep puts an already-built job in the batch.
//
// It is Add for the caller that has a Step in hand -- Dispatcher.Batch and
// ChainedBatch both do -- and it is the only place in this package where a job
// enters a batch without being serialized again.
func (p *PendingBatch) AddStep(jobs ...Step) *PendingBatch {
	for _, s := range jobs {
		if !s.declared() {
			p.fail(ErrNoName)
			return p
		}
		p.jobs = append(p.jobs, s)
	}
	return p
}

// Jobs is the jobs described so far.
func (p *PendingBatch) Jobs() []Step { return append([]Step(nil), p.jobs...) }

// Name sets what the batch is called.
func (p *PendingBatch) Name(name string) *PendingBatch {
	p.name = name
	return p
}

// Before names the job dispatched once the batch has been stored and before its
// own jobs are pushed.
//
// It is the hook for "write down that the import started": it runs with the
// batch id in hand and with nothing yet in flight.
func (p *PendingBatch) Before(name string, payload any) *PendingBatch {
	return p.callback(&p.options.Before, name, payload)
}

// BeforeCallbacks is the job Before named.
//
// A moment names one job, so the slice is empty or has one in it.
func (p *PendingBatch) BeforeCallbacks() []Step { return declaredSteps(p.options.Before) }

// Progress names the job dispatched after every job in the batch reports.
func (p *PendingBatch) Progress(name string, payload any) *PendingBatch {
	return p.callback(&p.options.Progress, name, payload)
}

// ProgressCallbacks is the job Progress named.
func (p *PendingBatch) ProgressCallbacks() []Step { return declaredSteps(p.options.Progress) }

// Then names the job dispatched when every job in the batch succeeded.
func (p *PendingBatch) Then(name string, payload any) *PendingBatch {
	return p.callback(&p.options.Then, name, payload)
}

// ThenCallbacks is the job Then named.
func (p *PendingBatch) ThenCallbacks() []Step { return declaredSteps(p.options.Then) }

// Catch names the job dispatched on the first failure, and only the first.
//
// It fires whether or not failures are allowed: AllowFailures decides what
// happens to the other jobs, not whether anybody is told.
func (p *PendingBatch) Catch(name string, payload any) *PendingBatch {
	return p.callback(&p.options.Catch, name, payload)
}

// CatchCallbacks is the job Catch named.
func (p *PendingBatch) CatchCallbacks() []Step { return declaredSteps(p.options.Catch) }

// Failure names the job dispatched on every failure of a batch that allows
// them.
//
// Catch is the first failure; Failure is each of them. A batch that does not
// allow failures never reaches a second one, which is why this one runs only
// when AllowFailures is on.
func (p *PendingBatch) Failure(name string, payload any) *PendingBatch {
	return p.callback(&p.options.Failure, name, payload)
}

// FailureCallbacks is the job Failure named.
func (p *PendingBatch) FailureCallbacks() []Step { return declaredSteps(p.options.Failure) }

// Finally names the job dispatched when every job in the batch has reported,
// whichever way each of them went.
func (p *PendingBatch) Finally(name string, payload any) *PendingBatch {
	return p.callback(&p.options.Finally, name, payload)
}

// FinallyCallbacks is the job Finally named.
func (p *PendingBatch) FinallyCallbacks() []Step { return declaredSteps(p.options.Finally) }

// AllowFailures lets the rest of the batch continue after a job fails.
//
// Without it the first failure cancels the batch: the jobs already queued are
// delivered, ask Batchable.Batching, and skip their work. That is the default
// because a batch is usually one operation split up, and finishing half of an
// operation is worse than finishing none of it. An import that is genuinely
// row-by-row independent is the case this exists for.
//
// It takes no argument: naming the failure callback is [PendingBatch.Failure],
// because one method that means two things is one method too many.
func (p *PendingBatch) AllowFailures() *PendingBatch {
	p.options.AllowFailures = true
	return p
}

// AllowsFailures reports whether a failing job would leave the rest running.
func (p *PendingBatch) AllowsFailures() bool { return p.options.AllowFailures }

// OnQueue is which queue the jobs and the callbacks go on.
func (p *PendingBatch) OnQueue(queue string) *PendingBatch {
	p.options.Queue = queue
	return p
}

// Queue is the queue OnQueue named.
func (p *PendingBatch) Queue() string { return p.options.Queue }

// OnConnection is which queue connection the jobs go to.
//
// It is carried and stored rather than acted on: the Queue this package is
// given is already a connection, and the name is what a dashboard shows and
// what a worker configuration is checked against.
func (p *PendingBatch) OnConnection(connection string) *PendingBatch {
	p.options.Connection = connection
	return p
}

// Connection is the connection OnConnection named.
func (p *PendingBatch) Connection() string { return p.options.Connection }

// WithOption hangs an application's own bookkeeping off the batch: the id of
// the upload the import came from, the user who pressed the button.
//
// The value is a string because the options are written to one column and read
// back by every replica: a value that only this binary can decode is a value
// the next release cannot read.
func (p *PendingBatch) WithOption(key, value string) *PendingBatch {
	if p.options.Extra == nil {
		p.options.Extra = make(map[string]string)
	}
	p.options.Extra[key] = value
	return p
}

// WithEvents records batch.dispatched, batch.cancelled and batch.finished into
// r.
//
// Without it the batch records nothing, which is the right default for a
// one-off script and the wrong one for an application: "when did the import
// stop" is answered by these rows and by nothing else.
func (p *PendingBatch) WithEvents(r EventRecorder) *PendingBatch {
	p.events = r
	return p
}

// Options is the batch's settings and callbacks as they stand.
func (p *PendingBatch) Options() BatchOptions { return p.options }

// Dispatch writes the batch and pushes its jobs.
//
// The batch row is written before the first job, because a job that arrives
// before its batch exists has nothing to report to -- and with a worker already
// running, "before" is measured in microseconds.
//
// Inside database.Transaction, with the table queue, the row and the jobs
// commit together and a rollback takes all of it. With a queue that cannot join
// the transaction, a push that fails partway leaves jobs already queued against
// a batch whose remaining jobs will never arrive; that batch is cancelled before
// the error is returned, so the ones already queued skip their work instead of
// doing half an import.
func (p *PendingBatch) Dispatch(ctx context.Context, g auth.Grant, r BatchRepository, q Queue) (Batch, error) {
	if p.err != nil {
		return Batch{}, p.err
	}
	if len(p.jobs) == 0 {
		return Batch{}, ErrEmptyBatch
	}
	if r == nil || q == nil {
		return Batch{}, fmt.Errorf("bus: dispatching the batch %q needs a repository and a queue", p.name)
	}
	if _, err := tenantOf(g); err != nil {
		return Batch{}, err
	}

	b, err := r.Store(ctx, g, p)
	if err != nil {
		return Batch{}, err
	}

	if b.HasBeforeCallbacks() {
		if err := push(ctx, g, q, b, b.Options.Before); err != nil {
			return Batch{}, p.abandon(ctx, g, r, b.ID, err)
		}
	}

	for _, j := range p.jobs {
		// Every job carries an id of its own, because the failed list is keyed
		// by it: a job that failed, was retried and then succeeded has to stop
		// counting as a failure, and "which job" is not a question the batch id
		// can answer.
		jobID, err := database.NewID()
		if err != nil {
			return Batch{}, p.abandon(ctx, g, r, b.ID, err)
		}
		payload, err := wrap(envelope{Bus: formatVersion, Batch: b.ID, Job: jobID, Body: j.Payload})
		if err != nil {
			return Batch{}, p.abandon(ctx, g, r, b.ID, err)
		}
		if err := q.Push(ctx, g, b.queueFor(j), j.Name, payload); err != nil {
			return Batch{}, p.abandon(ctx, g, r, b.ID,
				fmt.Errorf("bus: pushing %s of batch %s: %w", j.Name, b.ID, err))
		}
	}

	return b, nil
}

// DispatchIf dispatches the batch when the condition holds, and reports the
// zero Batch and no error when it does not.
//
// The condition is the last argument because the context comes first and a
// Batch has to be dispatched somewhere.
func (p *PendingBatch) DispatchIf(ctx context.Context, g auth.Grant, r BatchRepository, q Queue, ok bool) (Batch, error) {
	if !ok {
		return Batch{}, nil
	}
	return p.Dispatch(ctx, g, r, q)
}

// DispatchUnless dispatches the batch when the condition does not hold.
func (p *PendingBatch) DispatchUnless(ctx context.Context, g auth.Grant, r BatchRepository, q Queue, ok bool) (Batch, error) {
	return p.DispatchIf(ctx, g, r, q, !ok)
}

// DispatchAfterResponse stores the batch now and pushes its jobs when the
// Dispatcher is told the response has gone out.
//
// The row exists as soon as the request is handled, so the id can be returned
// to the browser, and the work starts once nobody is waiting on it.
// Dispatcher.Terminating is what runs it.
func (p *PendingBatch) DispatchAfterResponse(ctx context.Context, g auth.Grant, d *Dispatcher) (Batch, error) {
	if d == nil || d.repository == nil || d.queue == nil {
		return Batch{}, fmt.Errorf("bus: dispatching the batch %q after the response needs a dispatcher with a repository and a queue", p.name)
	}
	if p.err != nil {
		return Batch{}, p.err
	}
	if len(p.jobs) == 0 {
		return Batch{}, ErrEmptyBatch
	}
	if _, err := tenantOf(g); err != nil {
		return Batch{}, err
	}

	b, err := d.repository.Store(ctx, g, p)
	if err != nil {
		return Batch{}, err
	}
	d.afterResponse(func(ctx context.Context) error {
		_, err := b.Add(ctx, g, d.repository, d.queue, p.jobs...)
		if err != nil {
			return b.Delete(ctx, g, d.repository)
		}
		return nil
	})
	return b, nil
}

// abandon cancels a batch that could not be fully dispatched, and keeps the
// original error whatever the cancel does. A cancel that also fails leaves a
// batch that never finishes, and saying so is more useful than replacing the
// error that caused it.
func (p *PendingBatch) abandon(ctx context.Context, g auth.Grant, r BatchRepository, id string, cause error) error {
	if err := r.Cancel(ctx, g, id); err != nil {
		return fmt.Errorf("%w (and cancelling the batch failed too: %v)", cause, err)
	}
	return cause
}

// callback fills one of the six callback slots.
func (p *PendingBatch) callback(slot *Step, name string, payload any) *PendingBatch {
	s, err := step("", name, payload)
	if err != nil {
		p.fail(err)
		return p
	}
	*slot = s
	return p
}

// fail keeps the first error. The first is the one with the cause in it; the
// ones after it are usually the same value refusing to serialize again.
func (p *PendingBatch) fail(err error) {
	if p.err == nil {
		p.err = err
	}
}

// declaredSteps is a callback slot as a slice: empty, or the one job it names.
func declaredSteps(s Step) []Step {
	if !s.declared() {
		return nil
	}
	return []Step{s}
}
