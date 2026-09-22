package queue

import (
	"context"
	"fmt"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/queue/jobs"
	"github.com/arandu-io/hesape/queue/middleware"
)

// CallQueuedHandler turns a job on the wire into a call to the code that runs
// it.
//
// The name on the wire is a string and the registry is the [Worker]. What this
// does is run the handler through the job's middleware and settle the job
// afterwards, the same way whoever called it would have.
//
//	h := queue.NewCallQueuedHandler(w).Through(middleware.Skip{}.When(readOnly))
//	err := h.Call(ctx, j)
//
// [Worker.Process] is the caller that matters, and it does this inline. What
// this type is for is everything else that has a job and needs it run --
// [SyncQueue], a test, a command that replays one job by hand -- so the
// settling is written once.
type CallQueuedHandler struct {
	handlers   Handlers
	middleware []middleware.Middleware
}

// NewCallQueuedHandler returns the handler over a registry.
func NewCallQueuedHandler(h Handlers) *CallQueuedHandler {
	return &CallQueuedHandler{handlers: h}
}

// Through wraps every call in middleware, outermost first, and returns the
// handler so the call chains.
//
// A job here is a name and a payload, with nowhere on it to hang a middleware
// list, so the list comes off whoever is running it.
func (h *CallQueuedHandler) Through(m ...middleware.Middleware) *CallQueuedHandler {
	h.middleware = append(h.middleware, m...)
	return h
}

// Call runs the job and settles it.
//
// The settling is the part worth reading, and the order is:
//
//   - a job the middleware already released or deleted is left alone, because
//     settling it twice is how a job runs twice
//   - a job whose handler failed is left for the caller to release or park,
//     because only the caller knows how many tries are left
//   - anything else is deleted, which is what "it worked" means to a queue
func (h *CallQueuedHandler) Call(ctx context.Context, j *jobs.Job) error {
	handler, known := h.handlers.Handler(j.Name)
	if !known {
		err := fmt.Errorf("queue: no handler registered for %s", j.Name)
		_ = j.Fail(ctx, err)
		return err
	}

	next := func(ctx context.Context) error { return handler.Handle(ctx, jobs.GrantFor(j), j) }
	for i := len(h.middleware) - 1; i >= 0; i-- {
		m, inner := h.middleware[i], next
		next = func(ctx context.Context) error { return m.Handle(ctx, j, inner) }
	}

	if err := next(ctx); err != nil {
		return err
	}
	if j.IsDeletedOrReleased() {
		return nil
	}
	return j.Delete(ctx)
}

// Failed runs whatever the job wants done when it gives up for good.
//
// It is called after the job has been parked, not instead of parking it: the
// dead letter row is what an operator sees, and this is the compensating action
// -- refund the hold, mark the import abandoned, tell the customer.
func (h *CallQueuedHandler) Failed(ctx context.Context, j *jobs.Job, cause error) {
	handler, known := h.handlers.Handler(j.Name)
	if !known {
		return
	}
	if onFailure, has := handler.(HandlesFailure); has {
		onFailure.Failed(ctx, jobs.GrantFor(j), j, cause)
	}
}

// HandlesFailure is a handler that wants to know when its job gave up.
//
// A handler that does not implement it is not called.
type HandlesFailure interface {
	// Failed is called once, after the job has been parked.
	Failed(ctx context.Context, g auth.Grant, j *jobs.Job, cause error)
}

// CallQueuedClosure is a job whose work is a function rather than a registered
// name.
//
// Go cannot serialize a function, so the closure stays in the binary and what
// travels is the name it was registered under.
//
//	c := queue.NewCallQueuedClosure(func(ctx context.Context, g auth.Grant, j *jobs.Job) error {
//		return warmTheCache(ctx, g)
//	}).Name("cache.warm").OnFailure(func(err error) { alert(err) })
//
//	w.Handle(c.JobName(), c)
//	q.Push(ctx, g, c.Job(g))
//
// The registration is the price of not having serializable closures, and it is
// the honest one: a closure that is not in the binary the worker is running
// cannot be called by it.
type CallQueuedClosure struct {
	closure          HandlerFunc
	name             string
	failureCallbacks []func(error)
}

// NewCallQueuedClosure returns a job for a function.
func NewCallQueuedClosure(closure HandlerFunc) *CallQueuedClosure {
	return &CallQueuedClosure{closure: closure}
}

var (
	_ Handler        = (*CallQueuedClosure)(nil)
	_ HandlesFailure = (*CallQueuedClosure)(nil)
	_ HasDisplayName = (*CallQueuedClosure)(nil)
)

// Name assigns a name to the job, and returns it so the call chains.
func (c *CallQueuedClosure) Name(name string) *CallQueuedClosure {
	c.name = name
	return c
}

// JobName is the name this closure is registered and pushed under.
//
// A closure has no name of its own, so an unnamed one gets a name that says so
// rather than an empty string that would collide with every other unnamed
// closure.
func (c *CallQueuedClosure) JobName() string {
	if c.name == "" {
		return "closure"
	}
	return c.name
}

// DisplayName is what a console and the failed job list show.
//
// It is the name the caller gave, not the one the runtime has:
// runtime.FuncForPC would answer pkg.glob..func3, which tells a person less.
func (c *CallQueuedClosure) DisplayName() string {
	if c.name == "" {
		return "Closure"
	}
	return c.name + " - Closure"
}

// OnFailure adds a callback for when the job gives up, and returns the job so
// the call chains.
func (c *CallQueuedClosure) OnFailure(callback func(error)) *CallQueuedClosure {
	c.failureCallbacks = append(c.failureCallbacks, callback)
	return c
}

// Handle runs the closure.
func (c *CallQueuedClosure) Handle(ctx context.Context, g auth.Grant, j *jobs.Job) error {
	if c.closure == nil {
		return fmt.Errorf("queue: the closure job %s has no function to run", c.JobName())
	}
	return c.closure(ctx, g, j)
}

// Failed calls every failure callback with the cause.
func (c *CallQueuedClosure) Failed(_ context.Context, _ auth.Grant, _ *jobs.Job, cause error) {
	for _, callback := range c.failureCallbacks {
		callback(cause)
	}
}

// Job is the record to push for this closure.
//
// The closure and the record are separate: the closure is registered with a
// worker once, and this is what goes on the queue each time.
func (c *CallQueuedClosure) Job(g auth.Grant) (jobs.Job, error) {
	j, err := jobs.New(g, "", c.JobName(), nil)
	if err != nil {
		return jobs.Job{}, err
	}
	j.DisplayName = c.DisplayName()
	return j, nil
}
