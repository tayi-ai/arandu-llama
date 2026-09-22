package queue

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"os"
	"runtime"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/log"
	"github.com/arandu-io/hesape/queue/events"
	"github.com/arandu-io/hesape/queue/failed"
	"github.com/arandu-io/hesape/queue/jobs"
	"github.com/arandu-io/hesape/queue/middleware"
)

// WorkerOptions configures the loop.
//
// Two of the fields are there because of how this worker runs: a Concurrency,
// because it drains a batch at once rather than one job per process, and a
// Recorder, because a job is instrumented like a request.
type WorkerOptions struct {
	// Name identifies this worker, for [PopUsing] and for the log line. Empty
	// means "default".
	Name string
	// Queue is which queue to drain. Empty means jobs.DefaultQueue.
	Queue string
	// Concurrency is how many jobs run at once. Default 4.
	//
	// One process drains a batch, and the batch is this wide.
	Concurrency int
	// Lease is how long a popped job stays invisible to other workers. It has
	// to exceed the longest handler, or a second worker picks up work still in
	// progress. Default 5 minutes.
	Lease time.Duration
	// Timeout is how long one handler may run before its context is cancelled.
	// Zero means Lease.
	//
	// It defaults to Lease so there is one number until somebody needs two, and
	// a Timeout longer than the Lease is the misconfiguration that hands a
	// running job to a second worker.
	//
	// It is also the only thing that bounds a shutdown. A job already handed
	// over runs to the end whatever the process was asked to do meanwhile, so a
	// handler that never returns is what would hold the process open -- and
	// this is the number that ends it, settling the job the ordinary way
	// instead of abandoning it under its lease.
	Timeout time.Duration
	// Sleep is how long to wait before asking again when the queue was empty.
	// Default 1 second.
	Sleep time.Duration
	// Rest is how long to wait after each job, whatever the queue holds. Zero
	// means no rest. It exists to stop a fast queue from saturating a database
	// that everything else shares.
	Rest time.Duration
	// MaxTries is how many deliveries a job gets before it is parked.
	// Default 5.
	//
	// A job's own jobs.Job.MaxTries overrides it, which is what makes one
	// handler able to retry more than the rest.
	MaxTries int
	// Force runs the worker even while the application is down for
	// maintenance.
	Force bool
	// StopWhenEmpty ends the loop the first time the queue has nothing. It is
	// what a pipeline step waits on.
	StopWhenEmpty bool
	// MaxJobs stops the worker after this many jobs. Zero means no limit.
	MaxJobs int
	// MaxTime stops the worker after this long. Zero means no limit.
	MaxTime time.Duration
	// Memory stops the worker when the process is holding more than this many
	// megabytes. Zero means no limit.
	Memory int
	// Middleware wraps every job this worker runs, outermost first.
	//
	// It is the worker's list rather than the job's, because a job here is a
	// name and a payload and not a class with attributes on it: there is
	// nowhere on the wire to put "this one is rate limited". A handler that
	// needs different treatment gets its own worker, which is also how it gets
	// its own queue.
	Middleware []middleware.Middleware
	// Recorder receives each finished job, so it shows on /_arandu/debug with
	// its queries and its timeline -- exactly like a request.
	//
	// Nil means no instrumentation, and that is what production looks like: no
	// Collector is built, log.FromContext returns nil, and every Record method
	// is a no-op on a nil receiver. Zero cost, not low cost.
	//
	// Building a Collector on every job unconditionally would make production
	// pay for recording every query with its bound arguments and its caller
	// frames, with nobody able to read any of it. Pass the application's
	// recorder to turn it on.
	Recorder *log.Recorder
	// Backoff returns how long to wait before attempt n. Default is
	// exponential, capped at an hour.
	Backoff func(attempt int) time.Duration
}

func (o WorkerOptions) withDefaults() WorkerOptions {
	if o.Name == "" {
		o.Name = "default"
	}
	if o.Queue == "" {
		o.Queue = jobs.DefaultQueue
	}
	if o.Concurrency <= 0 {
		o.Concurrency = 4
	}
	if o.Lease <= 0 {
		o.Lease = 5 * time.Minute
	}
	if o.Timeout <= 0 {
		o.Timeout = o.Lease
	}
	if o.Sleep <= 0 {
		o.Sleep = time.Second
	}
	if o.MaxTries <= 0 {
		o.MaxTries = 5
	}
	if o.Backoff == nil {
		o.Backoff = ExponentialBackoff
	}
	return o
}

// settleTimeout is how long the write that settles a job -- the delete, the
// release or the park -- is given.
//
// Deliberately short. The write happens on a context the shutdown cannot
// cancel, so it is the only thing standing between an unreachable store and a
// process that never exits. A settlement that runs out of time costs one
// redelivery of a job that is still in the queue, which is at-least-once
// behaving as documented.
const settleTimeout = 5 * time.Second

// settling returns the context a job is settled on, and the function that ends
// it.
//
// The delete, the release and the park are the only record of what became of
// the job, and they are what must not be cancelled with the process. A
// settlement that fails leaves the job reserved until its lease expires, and it
// comes back with an attempt already spent -- so it rides a context nothing
// cancels, bounded by a timeout of its own.
func settling(ctx context.Context) (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.WithoutCancel(ctx), settleTimeout)
}

// ExponentialBackoff doubles the wait each attempt, capped at an hour.
//
// Capped, because unbounded doubling means the eleventh attempt is next year --
// and a job nobody will ever see fail is worse than one that parks.
func ExponentialBackoff(attempt int) time.Duration {
	if attempt < 1 {
		attempt = 1
	}
	if attempt > 12 {
		return time.Hour
	}
	wait := time.Duration(math.Pow(2, float64(attempt))) * time.Second
	if wait > time.Hour {
		return time.Hour
	}
	return wait
}

// PopCallback replaces how a worker chooses its next batch.
//
// It is what [PopUsing] registers: pop is the driver's own pop, and what the
// callback returns is what the worker runs. It is the seam for weighting
// queues against each other.
type PopCallback func(ctx context.Context, pop func(queue string) ([]*jobs.Job, error)) ([]*jobs.Job, error)

var popCallbacks struct {
	sync.RWMutex
	byWorker map[string]PopCallback
}

// PopUsing registers how the worker called workerName chooses its next batch.
//
// Passing nil forgets the callback rather than registering an empty one.
func PopUsing(workerName string, callback PopCallback) {
	popCallbacks.Lock()
	defer popCallbacks.Unlock()
	if callback == nil {
		delete(popCallbacks.byWorker, workerName)
		return
	}
	if popCallbacks.byWorker == nil {
		popCallbacks.byWorker = map[string]PopCallback{}
	}
	popCallbacks.byWorker[workerName] = callback
}

// Cache is the little a worker asks of a cache store: whether somebody asked
// for a restart, and whether this queue is paused.
//
// It is declared here rather than imported so the queue does not depend on a
// store to run. cache.Store satisfies it. There is no auth.Grant on it and that
// is deliberate: a restart signal and a pause are operational state about the
// process, not data about a customer, so there is no tenant to scope them by.
type Cache interface {
	Get(ctx context.Context, key string) ([]byte, error)
	Put(ctx context.Context, key string, value []byte, ttl time.Duration) error
	Forget(ctx context.Context, key string) error
}

// restartKey is where `aru queue:restart` leaves its timestamp.
const restartKey = "hesape:queue:restart"

// Worker runs jobs off a queue.
//
// It is also the handler registry: something has to turn the name on the wire
// into code, and here it is a map somebody filled in, so the set of names a
// binary can run is visible at the call site.
//
// It runs in the same binary as the application, started by `aru queue:work`,
// which is the same image with a different argument. Not a second artifact:
// one image is one thing to build, ship and roll back.
type Worker struct {
	queue    Queue
	manager  *QueueManager
	handlers map[string]Handler
	opts     WorkerOptions
	cache    Cache
	events   Dispatcher
	failed   failed.FailedJobProvider

	// ShouldQuit ends the loop after the job in flight. It is what a signal
	// handler sets.
	ShouldQuit atomic.Bool
	// Paused stops the worker taking new jobs without ending the loop.
	Paused atomic.Bool
}

// NewWorker returns the worker.
func NewWorker(q Queue, opts WorkerOptions) *Worker {
	return &Worker{queue: q, handlers: map[string]Handler{}, opts: opts.withDefaults()}
}

// Handle registers the handler for a job name.
//
// Registering twice panics rather than replacing. Two handlers for one name is
// an import nobody meant to add, and finding out at boot beats finding out from
// work that silently went to the wrong place.
func (w *Worker) Handle(name string, h Handler) *Worker {
	if _, taken := w.handlers[name]; taken {
		panic("queue: " + name + " already has a handler")
	}
	w.handlers[name] = h
	return w
}

// HandleFunc registers a function.
func (w *Worker) HandleFunc(name string, f HandlerFunc) *Worker { return w.Handle(name, f) }

// Handler returns the handler registered for a job name.
//
// It is what [SyncQueue] borrows, so the sync connection and the worker resolve
// a name the same way.
func (w *Worker) Handler(name string) (Handler, bool) {
	h, known := w.handlers[name]
	return h, known
}

var _ Handlers = (*Worker)(nil)

// Names returns the registered job names, for `aru queue:work` to print at start.
func (w *Worker) Names() []string {
	out := make([]string, 0, len(w.handlers))
	for name := range w.handlers {
		out = append(out, name)
	}
	return out
}

// SetName names this worker, and returns it so the call chains.
//
// The name is what [PopUsing] keys on.
func (w *Worker) SetName(name string) *Worker {
	w.opts.Name = name
	return w
}

// SetCache gives the worker somewhere to read the restart signal and the pause
// flag, and returns it so the call chains.
//
// Without one the worker never restarts on `aru queue:restart` and never
// observes a paused queue -- which is what a test wants and what a
// single-process deployment can live with.
func (w *Worker) SetCache(c Cache) *Worker {
	w.cache = c
	return w
}

// SetFailedJobs gives the worker the dead letter list to record parked jobs in,
// and returns it so the call chains.
//
// Wire the provider the failed job commands were built with. They read a
// provider and nothing else, so the one this worker writes to has to be the one
// `aru queue:failed` lists, `queue:retry` pushes back and `queue:forget`
// deletes -- otherwise the four are a console over an empty table, which is
// what an operator finds out during the incident they were meant for.
//
// Nil is the default and means the driver's own park is the whole record: a
// [DatabaseQueue] marks the job failed in the jobs table it was already in, and
// Queue.Failed lists them. That arrangement needs no provider and gets none.
func (w *Worker) SetFailedJobs(p failed.FailedJobProvider) *Worker {
	w.failed = p
	return w
}

// SetEvents gives the worker somewhere to send its events, and returns it so
// the call chains.
//
// Nil means the events are not built at all, which is what production looks
// like when nobody is listening.
func (w *Worker) SetEvents(d Dispatcher) *Worker {
	w.events = d
	return w
}

// Options is the configuration this worker is running under.
//
// The worker holds its options, and [WorkCommand] reads them so its flags can
// override what the application built.
func (w *Worker) Options() WorkerOptions { return w.opts }

// SetOptions replaces the configuration, and returns the worker so the call
// chains.
//
// It is the setter half of [Worker.Options], and `aru queue:work` with
// `--queue=reports` uses it. Calling it while the loop is running is a data
// race: the options are read on every pass, and a command sets them before it
// starts the worker.
func (w *Worker) SetOptions(o WorkerOptions) *Worker {
	w.opts = o.withDefaults()
	return w
}

// FlushState forgets that the worker was ever asked to stop or pause.
//
// A Worker is a long-lived object, and the two flags a signal sets outlive the
// loop that read them: without this, a worker started a second time in the same
// process -- which is what a test and `aru queue:work --once` in a loop both
// do -- stops immediately because something asked the first one to.
func (w *Worker) FlushState() {
	w.ShouldQuit.Store(false)
	w.Paused.Store(false)
}

// Pause stops this worker taking new jobs, without ending its loop.
//
// It is a method rather than a signal handler because signal handling belongs
// to the program, not to a library: `aru queue:work` installs the handlers and
// calls this.
func (w *Worker) Pause() { w.Paused.Store(true) }

// Resume lets this worker take jobs again.
func (w *Worker) Resume() { w.Paused.Store(false) }

// Restart asks this worker to stop after the job it is holding.
//
// The loop returns after the batch in flight, so a deploy does not lose work.
func (w *Worker) Restart() { w.ShouldQuit.Store(true) }

// GetManager is the manager this worker resolves connections through, or nil.
func (w *Worker) GetManager() *QueueManager { return w.manager }

// SetManager gives the worker a manager, so it can ask whether its queue is
// paused.
func (w *Worker) SetManager(m *QueueManager) { w.manager = m }

// Daemon drains the queue until it is told to stop, and returns the exit status.
//
// The connection, the queue and the options are on the worker rather than
// arguments, because a worker is constructed with them.
//
// The status is one of [ExitSuccess], [ExitError] and [ExitMemoryLimit], and it
// is what `aru queue:work` returns to the shell: a supervisor reads it to tell
// "asked to stop" from "ran out of memory".
func (w *Worker) Daemon(ctx context.Context) (int, error) {
	logger := log.For(ctx).With("component", "worker", "queue", w.opts.Queue, "worker", w.opts.Name)
	logger.Info("worker started", "concurrency", w.opts.Concurrency, "handlers", len(w.handlers))

	lastRestart := w.getTimestampOfLastQueueRestart(ctx)
	start := time.Now()
	processed := 0

	w.dispatch(events.WorkerStarting{
		ConnectionName: w.connectionName(), Queue: w.opts.Queue, WorkerOptions: w.opts,
	})

	for {
		if ctx.Err() != nil {
			return w.Stop(ExitSuccess, Interrupted), nil
		}

		if !w.daemonShouldRun(ctx) {
			// A paused worker still has to notice a restart or a memory limit,
			// which is what pauseWorker checks after it sleeps.
			if status, reason, stop := w.pauseWorker(ctx, lastRestart); stop {
				return w.Stop(status, reason), nil
			}
			continue
		}

		popped, err := w.getNextJobs(ctx)
		if err != nil {
			if errors.Is(err, context.Canceled) {
				return w.Stop(ExitSuccess, Interrupted), nil
			}
			// A failed pop is not fatal: the store blinked, and the next poll
			// tries again. Stopping the worker would turn a hiccup into a
			// backlog nobody is draining.
			logger.Warn("popping jobs failed", "error", err)
			if !w.wait(ctx, w.opts.Sleep) {
				return w.Stop(ExitSuccess, Interrupted), nil
			}
			continue
		}

		if len(popped) == 0 {
			// An idle pass still has to reach stopIfNecessary, which is what
			// notices `aru queue:restart`. Skipping it on an idle pass leaves
			// a worker on a quiet queue never seeing the restart signal at
			// all, and a deploy waiting for a job that is not coming.
			if !w.wait(ctx, w.opts.Sleep) {
				return w.Stop(ExitSuccess, Interrupted), nil
			}
			if status, reason, stop := w.stopIfNecessary(ctx, lastRestart, start, processed, 0); stop {
				return w.Stop(status, reason), nil
			}
			continue
		}

		// The batch runs in parallel, because the lease was taken for all of it
		// at once.
		//
		// Running the batch serially would make Concurrency a lie in the one
		// way that costs data: Pop hides all n jobs for the same Lease, so
		// with Concurrency 4, Lease 5m and a two-minute handler, the fourth
		// job would start at minute six -- past its own lease, already visible
		// to another worker, and running in both. At-least-once would turn
		// into exactly-twice for the tail of every batch.
		//
		// The batch is sized by Concurrency, so running all of it at once is
		// exactly Concurrency jobs in flight, not Concurrency squared.
		//
		// Every job of the batch is started, even when the cancellation lands
		// between the pop and this loop. They are all reserved already, for the
		// same lease, and breaking out here left the rest of the batch invisible
		// to every worker until that lease expired -- jobs nobody was running
		// and nobody could see. Each of them is bounded by its own timeout, so
		// finishing the batch is not an unbounded wait.
		var wg sync.WaitGroup
		for _, j := range popped {
			wg.Add(1)
			go func() {
				defer wg.Done()
				_ = w.Process(ctx, j)
			}()
		}
		wg.Wait()
		processed += len(popped)

		if w.opts.Rest > 0 && !w.wait(ctx, w.opts.Rest) {
			return w.Stop(ExitSuccess, Interrupted), nil
		}

		if status, reason, stop := w.stopIfNecessary(ctx, lastRestart, start, processed, len(popped)); stop {
			return w.Stop(status, reason), nil
		}
	}
}

// RunNextJob takes one batch off the queue and runs it.
//
// It is what `aru queue:work --once` calls: one pass of the loop, with none of
// the bookkeeping that decides whether to make another. An empty queue sleeps
// once and returns.
func (w *Worker) RunNextJob(ctx context.Context) error {
	popped, err := w.getNextJobs(ctx)
	if err != nil {
		return err
	}
	if len(popped) == 0 {
		w.Sleep(ctx, w.opts.Sleep)
		return nil
	}
	for _, j := range popped {
		if err := w.Process(ctx, j); err != nil {
			return err
		}
	}
	return nil
}

// Process runs one job, instrumented when there is somewhere to send it.
//
// Instrumenting it is the point: "the nightly job is slow" is the same
// investigation as "the page is slow", and it deserves the same page.
//
// The error it returns is the handler's, after the job has been settled -- so a
// caller that wants to know what happened can, and the daemon that does not
// ignores it.
//
// Every path out of it dispatches events.JobAttempted, which is what makes that
// event usable for counting deliveries.
func (w *Worker) Process(ctx context.Context, j *jobs.Job) (err error) {
	connectionName := j.GetConnectionName()

	// A job that has been handed over runs to the end whatever the process was
	// asked to do meanwhile, so the caller's cancellation reaches neither the
	// handler nor the write that settles it.
	//
	// A popped job is work the store has already hidden from every other
	// worker, and the only record that it ran is the settlement still to be
	// written. Killing it with the process is what leaves the job reserved
	// until its lease expires -- minutes of nothing, and then a redelivery with
	// an attempt already spent, on every deploy.
	//
	// What bounds it is the job's own timeout, applied below. A cancellation
	// says "reserve nothing more", not "drop what is running".
	ctx = context.WithoutCancel(ctx)

	// Deferred rather than written at each return, because there are six of
	// them and the one that forgets is a delivery nobody counted.
	defer func() {
		w.dispatch(events.JobAttempted{ConnectionName: connectionName, Job: j, Exception: err})
	}()

	w.dispatch(events.JobProcessing{ConnectionName: connectionName, Job: j})

	handler, known := w.handlers[j.Name]
	if !known {
		// An unknown name is not a failure to retry: no amount of retrying will
		// register the handler. It parks immediately, where it can be seen.
		err = fmt.Errorf("no handler registered for %s", j.Name)
		log.For(ctx).Error("job has no handler", "job", j.Name, "id", j.UUID)
		_ = w.park(ctx, j, err, log.For(ctx))
		w.dispatch(events.JobFailed{ConnectionName: connectionName, Job: j, Exception: err})
		return err
	}

	// A job that already used up its deliveries before this one -- because it
	// timed out rather than failed, so nothing ever recorded an error -- is
	// parked here rather than run again.
	if err = w.MarkJobAsFailedIfAlreadyExceedsMaxAttempts(ctx, j); err != nil {
		w.dispatch(events.JobFailed{ConnectionName: connectionName, Job: j, Exception: err})
		return err
	}
	if j.IsDeleted() {
		w.dispatch(events.JobProcessed{ConnectionName: connectionName, Job: j})
		return nil
	}

	jobCtx, cancel := context.WithTimeout(ctx, w.timeoutFor(j))
	defer cancel()

	// Only when a recorder is wired. See WorkerOptions.Recorder.
	var col *log.Collector
	if w.opts.Recorder != nil {
		col = log.NewCollector(j.UUID)
		jobCtx = log.WithCollector(jobCtx, col)
	}
	logger := log.For(jobCtx).With("job", j.Name, "job_id", j.UUID, "tenant", j.TenantID)
	jobCtx = log.Into(jobCtx, logger)

	start := time.Now()
	err = w.run(jobCtx, handler, j)
	duration := time.Since(start)

	if col != nil {
		// Method and Path name the job rather than a route, so the console list
		// reads "job invoice.send" next to "GET /invoices".
		w.opts.Recorder.Record(log.Recorded{
			RequestID: j.UUID,
			Method:    "job",
			Path:      j.Name,
			Duration:  duration,
			At:        start,
			Collector: col,
		})
	}

	if err != nil {
		w.handleJobException(ctx, j, err, logger)
		return err
	}

	// A middleware that skipped the work returns no error and touches nothing,
	// and that is "handled": the job is deleted below. One that released it
	// said the opposite, and this is where the difference is read.
	if j.IsDeletedOrReleased() {
		w.dispatch(events.JobProcessed{ConnectionName: connectionName, Job: j})
		return nil
	}

	settle, settled := settling(ctx)
	deleteErr := j.Delete(settle)
	settled()
	if deleteErr != nil {
		// The work is done and the deletion failed, so the job runs again when
		// the lease expires. That is at-least-once behaving as documented, and
		// why a handler has to tolerate running twice.
		logger.Error("deleting the job failed; it will run again", "error", deleteErr)
		return nil
	}

	logger.Info("job done",
		"duration_ms", duration.Milliseconds(),
		"queries", col.QueryCount(),
		"sql_ms", col.QueryTime().Milliseconds())
	w.dispatch(events.JobProcessed{ConnectionName: connectionName, Job: j})
	return nil
}

// handleJobException settles a job whose handler failed.
//
// The order is the whole point: park the job when this failure is its last,
// then announce the exception, then release whatever is still unsettled. A
// listener reads JobExceptionOccurred as "this one will come back" -- the only
// reading that tells it apart from JobFailed -- and announcing it before the
// parking makes that reading wrong on the last delivery of every job.
func (w *Worker) handleJobException(ctx context.Context, j *jobs.Job, cause error, logger *slog.Logger) {
	connectionName := j.GetConnectionName()
	j.Exceptions++

	// A middleware that already released, deleted or parked the job has
	// decided the outcome, and settling it a second time is how a job runs
	// twice: WithoutOverlapping releases it for ten seconds and the worker
	// underneath would release it again for the backoff, or park it.
	//
	// The exception is announced either way, which is what the event is for: it
	// says the handler failed, not what became of the job.
	settled := j.IsDeletedOrReleased()

	// Attempts already counts this delivery -- Pop incremented it. Adding one
	// here counts it twice, so MaxTries of N delivers N-1 times and MaxTries of
	// 2 parks on the first failure with no retry at all. The in-memory queue
	// the worker tests use does not increment, so that miscount does not show
	// there.
	attempts := j.Attempts
	if attempts < 1 {
		attempts = 1
	}

	parked := false
	if !settled {
		j.LastError = cause.Error()
		if parked = w.shouldFail(j, attempts, cause); parked {
			if failErr := w.park(ctx, j, cause, logger); failErr != nil {
				logger.Error("parking the job failed", "error", failErr)
			}
			// The line says what happened and the attempt count says how many
			// deliveries it took. It used to say "after repeated failures",
			// which is false for every job parked on its first one -- a handler
			// that failed manually, and now a handler that panicked.
			logger.Error("job parked", "attempts", attempts, "error", cause)
			w.dispatch(events.JobFailed{ConnectionName: connectionName, Job: j, Exception: cause})
		}
	}

	w.dispatch(events.JobExceptionOccurred{ConnectionName: connectionName, Job: j, Exception: cause})

	if settled {
		logger.Warn("job failed and was already settled by a middleware", "error", cause)
		return
	}
	if parked {
		return
	}

	wait := w.calculateBackoff(j, attempts)
	settle, endSettle := settling(ctx)
	relErr := j.Release(settle, wait)
	endSettle()
	if relErr != nil {
		logger.Error("releasing the job failed", "error", relErr)
	}
	logger.Warn("job failed", "attempt", attempts, "retry_in", wait, "error", cause)
	w.dispatch(events.JobReleasedAfterException{ConnectionName: connectionName, Job: j, Delay: wait})
}

// shouldFail decides whether this failure is the job's last.
//
// The manual failure, the retry deadline, the timeout, the exception count and
// the try count are one predicate rather than five checks that each park the
// job themselves: a job that trips two of them would otherwise be failed twice,
// and here the decision is made once and acted on once.
func (w *Worker) shouldFail(j *jobs.Job, attempts int, cause error) bool {
	// A handler that says the work can never succeed is believed on the first
	// delivery. Retrying it four more times is four more copies of the same
	// error, an hour apart.
	if errors.Is(cause, ErrManuallyFailed) {
		return true
	}
	// A panic is believed on the first delivery for the same reason, without
	// having been asked. It is a defect in the handler rather than a store that
	// blinked, so it reproduces on every delivery: retrying it is running a
	// deterministic failure MaxTries times waiting for a different answer.
	if errors.Is(cause, ErrHandlerPanicked) {
		return true
	}
	if until := j.RetryUntil(); !until.IsZero() {
		return !time.Now().Before(until)
	}
	if j.ShouldFailOnTimeout() && errors.Is(cause, context.DeadlineExceeded) {
		return true
	}
	if max := j.MaxExceptions(); max > 0 && j.Exceptions >= max {
		return true
	}
	// A limit of zero or less is no limit, and a job under one is never parked
	// for having been delivered often enough. See maxTriesFor.
	if max := w.maxTriesFor(j); max > 0 {
		return attempts >= max
	}
	return false
}

// MarkJobAsFailedIfAlreadyExceedsMaxAttempts parks a job that used up its
// deliveries without ever reporting an error.
//
// The case it exists for is a job that keeps timing out: the process dies with
// the job in flight, the lease expires, the job comes back, and nothing ever
// ran the code that counts a failure. Without this check that job is delivered
// forever.
//
// It returns the reason it parked the job, so the caller stops, and nil when
// the job may run.
func (w *Worker) MarkJobAsFailedIfAlreadyExceedsMaxAttempts(ctx context.Context, j *jobs.Job) error {
	if until := j.RetryUntil(); !until.IsZero() {
		if time.Now().Before(until) {
			return nil
		}
	} else {
		max := w.maxTriesFor(j)
		if max <= 0 || j.Attempts <= max {
			return nil
		}
	}

	err := MaxAttemptsExceeded{}.ForJob(j, w.maxTriesFor(j))
	if failErr := w.park(ctx, j, err, log.For(ctx)); failErr != nil {
		return failErr
	}
	return err
}

// park is the one way a job leaves the queue for good: the driver stops
// delivering it, and the record goes where the failed job commands read.
//
// The two are one step because they are one decision. `aru queue:failed`,
// `queue:retry`, `queue:forget` and `queue:flush` read a
// [failed.FailedJobProvider] and nothing else, so a worker that only told the
// driver leaves an operator four commands that print nothing about a job that
// is gone. Recording it here rather than adding a fifth listing that reads the
// driver keeps one dead letter list rather than two that disagree.
//
// The driver first, then the record. A park the driver refused means the job is
// still queued and will be delivered again, and a dead letter entry for a job
// that is still running is an operator retrying work that never stopped.
//
// A provider that refuses is reported and not returned: what the caller does
// next depends on what became of the job, and the job did give up. The error is
// logged rather than folded into the cause, so a dead letter list that has
// stopped accepting writes is visible without changing the error the handler
// saw.
func (w *Worker) park(ctx context.Context, j *jobs.Job, cause error, logger *slog.Logger) error {
	settle, settled := settling(ctx)
	defer settled()

	if err := j.Fail(settle, cause); err != nil {
		return err
	}
	if w.failed == nil {
		return nil
	}

	message := j.LastError
	if cause != nil {
		message = cause.Error()
	}
	// The job's own tenant, under the action the failed job list is reached
	// under -- the same Grant the commands hold, because they read what this
	// writes and a provider is free to check it.
	g := auth.SystemGrant(failed.Action, j.TenantID)
	if _, err := w.failed.Log(settle, g, failed.FailedJob{
		UUID: j.UUID,
		// The routing name, not the display name: `queue:retry` pushes the
		// record back under it, and a name a person reads is not one a handler
		// is registered under.
		Name: j.Name,
		// The job's own action, and not the one this record is written under.
		// They are different permissions on purpose -- the list is read by an
		// administrator and the work is not -- and a retry rebuilds the job's
		// Grant from this field, so the record has to carry the action the job
		// was pushed with or the work comes back as the administrator.
		Action:     j.Action,
		Connection: j.GetConnectionName(),
		Queue:      j.Queue,
		Payload:    j.Payload,
		Exception:  message,
		FailedAt:   time.Now().UTC(),
	}); err != nil {
		// The job and the reason, never the payload: it is a customer's
		// arguments, and the failed job list is where it belongs -- behind a
		// Grant -- rather than in a log line anything can ship anywhere.
		logger.Error("the failed job could not be recorded; it is parked but will not be listed",
			"job", j.Name, "job_id", j.UUID, "error", err)
	}
	return nil
}

// maxTriesFor is the job's own limit when it has one, and the worker's when it
// does not. A result of zero or less is no limit at all.
//
// Three answers have to be told apart and a Go int carries only two of them by
// itself: "the worker decides", "never park it", and a number.
// attributes.Attributes resolves that by spelling "never park it" as a negative
// Tries, because zero is the value of a field nobody set.
//
// A negative fed straight into "attempts >= maxTries" is true on the first
// delivery, so a job asking never to be parked would be parked at once. Callers
// read a non-positive answer here as "no limit" instead.
func (w *Worker) maxTriesFor(j *jobs.Job) int {
	if own := j.MaxTries(); own != 0 {
		return own
	}
	return w.opts.MaxTries
}

// timeoutFor is the job's own timeout when it has one, and the worker's when it
// does not.
func (w *Worker) timeoutFor(j *jobs.Job) time.Duration {
	if own := j.Timeout(); own > 0 {
		return own
	}
	return w.opts.Timeout
}

// calculateBackoff is how long to wait before the next delivery.
//
// It is the job's own schedule when it declared one, and the worker's when it
// did not.
func (w *Worker) calculateBackoff(j *jobs.Job, attempt int) time.Duration {
	if own := j.Backoff(); own > 0 {
		return own
	}
	return w.opts.Backoff(attempt)
}

// getNextJobs takes the next batch off the queue, through the registered pop
// callback when there is one.
func (w *Worker) getNextJobs(ctx context.Context) ([]*jobs.Job, error) {
	pop := func(name string) ([]*jobs.Job, error) {
		return w.queue.Pop(ctx, name, w.opts.Concurrency, w.opts.Lease)
	}

	w.dispatch(events.JobPopping{ConnectionName: w.connectionName(), Queue: w.opts.Queue})

	popCallbacks.RLock()
	callback, registered := popCallbacks.byWorker[w.opts.Name]
	popCallbacks.RUnlock()

	var (
		popped []*jobs.Job
		err    error
	)
	if registered {
		popped, err = callback(ctx, pop)
	} else {
		popped, err = pop(w.opts.Queue)
	}
	if err != nil {
		return nil, err
	}
	for _, j := range popped {
		w.dispatch(events.JobPopped{ConnectionName: w.connectionName(), Job: j})
	}
	return popped, nil
}

// daemonShouldRun reports whether this pass of the loop may take work.
//
// It is false when this worker is paused, when the queue it drains is paused,
// or when a Looping listener said no.
func (w *Worker) daemonShouldRun(ctx context.Context) bool {
	if w.Paused.Load() {
		return false
	}
	if w.manager != nil && !w.opts.Force && interruptionPolling.Load() {
		if paused, err := w.manager.IsPaused(ctx, w.connectionName(), w.opts.Queue); err == nil && paused {
			return false
		}
	}
	if w.events != nil {
		if answer := w.events.Until(events.Looping{
			ConnectionName: w.connectionName(), Queue: w.opts.Queue,
		}); answer == false {
			return false
		}
	}
	return true
}

// pauseWorker waits out one pass of a paused loop, and reports whether the
// worker should stop while it is there.
//
// The check afterwards is the point: a paused worker still has to notice `aru
// queue:restart`, or a deploy waits for a queue that will never resume.
func (w *Worker) pauseWorker(ctx context.Context, lastRestart string) (int, WorkerStopReason, bool) {
	sleep := w.opts.Sleep
	if sleep <= 0 {
		sleep = time.Second
	}
	if !w.wait(ctx, sleep) {
		return ExitSuccess, Interrupted, true
	}
	return w.stopIfNecessary(ctx, lastRestart, time.Time{}, 0, 0)
}

// stopIfNecessary decides whether the loop ends, and why.
//
// The reasons are checked most-urgent first, so a worker that is both out of
// memory and past its job limit reports the memory.
func (w *Worker) stopIfNecessary(ctx context.Context, lastRestart string, start time.Time, processed, lastBatch int) (int, WorkerStopReason, bool) {
	switch {
	case w.ShouldQuit.Load():
		return ExitSuccess, Interrupted, true
	case w.MemoryExceeded(w.opts.Memory):
		return ExitMemoryLimit, MaxMemoryExceeded, true
	case w.queueShouldRestart(ctx, lastRestart):
		return ExitSuccess, ReceivedRestartSignal, true
	case w.opts.StopWhenEmpty && lastBatch == 0:
		return ExitSuccess, QueueEmpty, true
	case w.opts.MaxTime > 0 && !start.IsZero() && time.Since(start) >= w.opts.MaxTime:
		return ExitSuccess, MaxTimeExceeded, true
	case w.opts.MaxJobs > 0 && processed >= w.opts.MaxJobs:
		return ExitSuccess, MaxJobsExceeded, true
	}
	return ExitSuccess, "", false
}

// MemoryExceeded reports whether the process is holding more than limit
// megabytes.
//
// The number read is MemStats.Sys, which is what the runtime has taken from the
// operating system rather than what is live. A limit of zero or less is no
// limit.
func (w *Worker) MemoryExceeded(limitMB int) bool {
	if limitMB <= 0 {
		return false
	}
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return m.Sys/1024/1024 >= uint64(limitMB)
}

// Stop ends the loop and returns the exit status.
//
// The event goes out here rather than at each call site, which is what makes
// "the worker stopped and this is why" one line in a log instead of six places
// that might have logged it.
func (w *Worker) Stop(status int, reason WorkerStopReason) int {
	w.dispatch(events.WorkerStopping{Status: status, WorkerOptions: w.opts, Reason: reason})
	return status
}

// Kill ends the process now.
//
// It exists for the one case [Worker.Stop] cannot serve: a handler that has
// stopped responding to its context, where returning from the loop would wait
// forever on a goroutine that will never finish. Everything else stops by
// returning.
func (w *Worker) Kill(status int) {
	w.dispatch(events.WorkerStopping{Status: status, WorkerOptions: w.opts})
	os.Exit(status)
}

// Sleep waits for d, or until the context is cancelled.
//
// Taking the context is the point: a worker asleep for three seconds when
// SIGTERM arrives would hold up the deploy for three seconds, and this one does
// not.
func (w *Worker) Sleep(ctx context.Context, d time.Duration) {
	if d <= 0 {
		return
	}
	w.wait(ctx, d)
}

// wait sleeps and reports whether it finished rather than being cancelled.
func (w *Worker) wait(ctx context.Context, d time.Duration) bool {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-timer.C:
		return true
	}
}

// queueShouldRestart reports whether somebody asked the workers to restart
// since this one started.
func (w *Worker) queueShouldRestart(ctx context.Context, lastRestart string) bool {
	if w.cache == nil {
		return false
	}
	return w.getTimestampOfLastQueueRestart(ctx) != lastRestart
}

// getTimestampOfLastQueueRestart reads the restart signal, or empty when there
// is none.
func (w *Worker) getTimestampOfLastQueueRestart(ctx context.Context) string {
	if w.cache == nil || !interruptionPolling.Load() {
		return ""
	}
	stamp, err := w.cache.Get(ctx, restartKey)
	if err != nil {
		return ""
	}
	return string(stamp)
}

// connectionName is what this worker's jobs report as their connection.
func (w *Worker) connectionName() string {
	if named, says := w.queue.(interface{ GetConnectionName() string }); says {
		return named.GetConnectionName()
	}
	return ""
}

// dispatch sends an event, when there is a dispatcher to send it to.
func (w *Worker) dispatch(event any) {
	if w.events != nil {
		w.events.Dispatch(event)
	}
}

// run wraps the handler in the worker's middleware, outermost first, and turns
// a panic in either of them into the error the failure path already knows how
// to settle.
//
// Built here rather than with pipeline.Chain because the chain is three lines
// and the alternative is a generic instantiation whose type parameter is a
// closure over the job -- longer to read than what it replaces.
//
// The recover is here because this is the goroutine the handler runs in, and a
// panic does not cross a goroutine boundary: recovering anywhere else recovers
// nothing. Without it a handler that panics takes the process down with the job
// unconsumed, so the job comes back when its lease expires and takes the next
// worker down too -- one payload stopping a queue forever, with nobody having
// to send a second one.
//
// It is deliberately no wider than the handler and its middleware. A panic
// raised by the worker's own bookkeeping is a defect here rather than in
// somebody's job, and turning that one into a parked job would hide it.
func (w *Worker) run(ctx context.Context, h Handler, j *jobs.Job) (err error) {
	defer func() {
		if value := recover(); value != nil {
			err = HandlerPanicked{}.ForJob(j, value, panicStack())
		}
	}()

	next := func(ctx context.Context) error { return h.Handle(ctx, jobs.GrantFor(j), j) }
	for i := len(w.opts.Middleware) - 1; i >= 0; i-- {
		m, inner := w.opts.Middleware[i], next
		next = func(ctx context.Context) error { return m.Handle(ctx, j, inner) }
	}
	return next(ctx)
}

// stackLimit is how much traceback a recovered panic carries.
//
// It is capped rather than taken whole because the traceback ends up in the
// job's last_error column, which is TEXT -- 64 KB on MySQL, and a write past
// that fails. A park that fails leaves the job reserved instead of parked, so
// it comes back when its lease expires and panics again: the loop the recover
// exists to cut, reopened by the size of the evidence for it.
const stackLimit = 16 << 10

// panicStack is the traceback of the goroutine calling it, up to stackLimit.
//
// runtime.Stack fills from the innermost frame outwards, so what the cap drops
// is the oldest frames and what it keeps is where the panic happened.
func panicStack() []byte {
	buf := make([]byte, stackLimit)
	return buf[:runtime.Stack(buf, false)]
}

// nowStamp is the value `aru queue:restart` writes: a Unix timestamp, as text,
// because that is all a comparison needs and text is what every store holds.
func nowStamp() string { return strconv.FormatInt(time.Now().UnixNano(), 10) }
