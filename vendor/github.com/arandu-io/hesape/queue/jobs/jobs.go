package jobs

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/database"
	"github.com/arandu-io/hesape/queue/attributes"
)

// DefaultQueue is where a job goes when nobody said otherwise.
const DefaultQueue = "default"

// MaxPayload is the largest payload a job may carry, in bytes.
//
// One number for every driver rather than one per store. A queue whose drivers
// accepted different jobs would be several queues: moving an application from
// one connection to another would be a migration instead of a line in its
// wiring, and the connection a developer runs locally would take work the
// connection in production refuses.
//
// So it is the narrowest thing a payload has to fit through, and there are
// three. The payload column is TEXT, and the narrowest engine with a connector
// holds 65,535 bytes in one of those -- past that the insert is either refused
// or, on a server that is not in strict mode, silently shortened, and a
// shortened payload is a handler decoding arguments the pusher never wrote. The
// dead letter table writes the payload a second time into a column of the same
// kind, so a job that gave up has to still fit at the moment its record is the
// only thing left of it. And a driver that runs the job in another process
// hands it over on an argument list, where one entry is capped far below what
// any store would take.
//
// Half the column rather than all of it, because the payload is counted in
// bytes here and the column is counted in bytes there, and nothing should
// depend on the two agreeing to the last one.
//
// It is also what a payload is for. 32 KiB of JSON is thousands of ids; more
// than that is a document, and a document belongs in storage with the job
// carrying its key.
const MaxPayload = 32 << 10

// Job is one unit of work.
//
// It is both the record a driver stores and the handle the worker settles. A
// job built by [New] carries no driver and can only be pushed; a job returned
// by a queue's Pop carries the queue it came off, and [Job.Release],
// [Job.Delete] and [Job.Fail] are how it settles itself.
//
// A job popped from a queue is used through a pointer, because releasing it and
// deleting it are decisions the worker and the middleware make about the same
// job and a copy would lose them.
type Job struct {
	// UUID is the deduplication key. It is stable across retries, which is what
	// makes a handler able to recognize work it already did. It is also the
	// job's id in the store, because the id is minted by the application rather
	// than handed back by the store (see database.NewID).
	UUID string
	// Queue separates work by urgency: a password reset email and a monthly
	// report should not wait behind each other.
	Queue string
	// Name routes the job to its handler: "invoice.send", "report.monthly".
	Name string
	// DisplayName is what a person reads on a console, a log line or the failed
	// job list, and empty means Name. The only thing that sets it is a job that
	// wraps another -- a queued closure, a queued listener -- where the name
	// that routes and the name that explains are different strings.
	DisplayName string
	// TenantID is who the work belongs to. It comes from the Grant at Push, and
	// the worker rebuilds a Grant from it -- a job with no tenant cannot be
	// scoped, and everything downstream of it reads across customers.
	TenantID string
	// Payload is the arguments, as JSON. Keep it to facts and ids: a payload
	// that says "look it up" is a payload that reads a row which has already
	// changed.
	Payload []byte
	// AuthorizedBy and Action record the Grant that pushed it, which is the
	// audit trail and what the worker reissues the work under.
	AuthorizedBy string
	Action       string
	// RunAt is when it becomes eligible. Zero means now.
	RunAt time.Time
	// Attempts counts the deliveries INCLUDING the current one: a job being
	// handled for the first time has Attempts == 1. LastError is why the most
	// recent one failed -- stored rather than logged, because the thing anyone
	// needs at 3am is "this failed twelve times with this message".
	Attempts  int
	LastError string
	// Exceptions counts the deliveries that ended in an error, which is not the
	// same number as Attempts: a middleware that releases the job -- rate
	// limited, overlapping -- spends an attempt without the handler ever having
	// thrown. It is what Attributes.MaxExceptions is compared against.
	Exceptions int
	// Attributes are the per-job settings: tries, backoff, timeout, the queue
	// and the connection. They travel with the job, so the worker reads back
	// exactly what the push declared.
	Attributes attributes.Attributes

	// driver is the queue this job came off. It is nil on a job that was built
	// to be pushed and never popped, and that is what ErrDetached reports.
	driver Driver
	// connection names the queue connection, for the log line and for a handler
	// that pushes a follow-up onto the same one.
	connection string
	// released, deleted and failed record that the job has been settled. The
	// worker reads them to find out whether a middleware already settled it,
	// because a job released by WithoutOverlapping must not then be deleted by
	// the success path.
	released bool
	deleted  bool
	failed   bool
}

// Driver is the three things a running job asks of the queue it came off.
//
// A queue implements it and hands itself to [Popped], so the job can settle
// itself without the worker having to know which store it lives in.
type Driver interface {
	// ReleaseJob puts the job back on its queue, eligible again after delay.
	ReleaseJob(ctx context.Context, j *Job, delay time.Duration) error
	// DeleteJob removes a finished job.
	DeleteJob(ctx context.Context, j *Job) error
	// FailJob parks the job: it stops being delivered and starts being
	// something a person can list and retry.
	FailJob(ctx context.Context, j *Job, cause error) error
}

// Popped attaches a job to the queue it came off, and to that connection's
// name.
//
// Every driver's Pop ends with it.
func Popped(d Driver, connection string, j Job) *Job {
	j.driver = d
	j.connection = connection
	return &j
}

// GetConnectionName is the name of the queue connection this job came off.
func (j *Job) GetConnectionName() string { return j.connection }

// SetConnectionName names the connection this job belongs to.
//
// The driver sets it when it pops.
func (j *Job) SetConnectionName(name string) { j.connection = name }

// GetQueue is the name of the queue the job belongs to.
func (j *Job) GetQueue() string { return j.Queue }

// GetName is the name of the queued job.
//
// It is the name a handler is registered under: "invoice.send".
func (j *Job) GetName() string { return j.Name }

// GetJobID is the identifier of the job in its store.
//
// It is the same value as UUID, because the id is minted by the application
// rather than handed back by the store: see database.NewID.
func (j *Job) GetJobID() string { return j.UUID }

// GetRawBody is the job's payload as it was stored.
//
// The envelope is the record's own columns, and this is the arguments, which is
// the half a handler decodes.
func (j *Job) GetRawBody() []byte { return j.Payload }

// MaxTries is how many deliveries this job gets before it is parked, or zero
// when the worker's own limit decides.
func (j *Job) MaxTries() int { return j.Attributes.Tries }

// MaxExceptions is how many failures this job gets before it is parked, or zero
// when only MaxTries decides.
func (j *Job) MaxExceptions() int { return j.Attributes.MaxExceptions }

// Backoff is how long to wait before the next delivery, or zero when the
// worker's own schedule decides.
//
// The attempt counts from one, and the last entry of the list repeats once the
// list runs out.
func (j *Job) Backoff() time.Duration { return j.Attributes.BackoffFor(j.Attempts) }

// Timeout is how long the handler may run, or zero when the worker's lease
// decides.
func (j *Job) Timeout() time.Duration { return j.Attributes.Timeout }

// ShouldFailOnTimeout reports whether running past the timeout parks the job
// rather than releasing it.
func (j *Job) ShouldFailOnTimeout() bool { return j.Attributes.FailOnTimeout }

// RetryUntil is the deadline after which this job stops being retried, whatever
// MaxTries says, or the zero time when there is none.
func (j *Job) RetryUntil() time.Time { return j.Attributes.RetryUntil }

// ResolveName is the name to show for this job.
func (j *Job) ResolveName() string { return JobName{}.Resolve(j.Name, j) }

// ResolveQueuedJobClass is the name of the work behind a wrapper job.
//
// A queued closure and a queued listener arrive under the wrapper's name and
// record what they wrap; a job that wraps nothing answers with its own name.
func (j *Job) ResolveQueuedJobClass() string { return JobName{}.ResolveClassName(j.Name, j) }

// Decode unmarshals the payload into v.
func (j *Job) Decode(v any) error {
	if err := json.Unmarshal(j.Payload, v); err != nil {
		return fmt.Errorf("queue: decoding %s: %w", j.Name, err)
	}
	return nil
}

// Release puts the job back on its queue, eligible again after delay.
//
// A middleware that decides the work should not run now calls it --
// WithoutOverlapping when another worker holds the lock, RateLimited when the
// budget is spent -- and so does the worker after a failure that has attempts
// left.
func (j *Job) Release(ctx context.Context, delay time.Duration) error {
	if j.driver == nil {
		return ErrDetached
	}
	j.released = true
	return j.driver.ReleaseJob(ctx, j, delay)
}

// Delete removes the job.
func (j *Job) Delete(ctx context.Context) error {
	if j.driver == nil {
		return ErrDetached
	}
	j.deleted = true
	return j.driver.DeleteJob(ctx, j)
}

// Fail parks the job.
//
// Parked rather than retried: the job stops being delivered and starts being
// something a person can see, which is what a dead letter queue is for. It
// marks the job deleted too, so nothing downstream settles it a second time.
func (j *Job) Fail(ctx context.Context, cause error) error {
	if j.driver == nil {
		return ErrDetached
	}
	if cause != nil {
		j.LastError = cause.Error()
	}
	j.MarkAsFailed()
	j.deleted = true
	return j.driver.FailJob(ctx, j, cause)
}

// IsReleased reports whether the job was put back on its queue.
func (j *Job) IsReleased() bool { return j.released }

// IsDeleted reports whether the job was removed.
func (j *Job) IsDeleted() bool { return j.deleted }

// HasFailed reports whether the job was parked.
func (j *Job) HasFailed() bool { return j.failed }

// MarkAsFailed records that the job failed without telling the store.
//
// It is the half of [Job.Fail] that does not park the job: a handler that has
// already written its own failure somewhere calls this so the worker does not
// release the job for another attempt.
func (j *Job) MarkAsFailed() { j.failed = true }

// IsDeletedOrReleased reports whether the job has already been settled.
//
// It is what the worker asks before it acts on the outcome of a handler: a
// middleware may have released the job before the handler ever ran, and
// settling it twice is how a job runs twice.
func (j *Job) IsDeletedOrReleased() bool { return j.deleted || j.released }

// ErrNoTenant is returned when a Grant carries no tenant.
//
// It is an error rather than a default: a job with no tenant cannot be scoped,
// and everything the handler touches would read across customers.
var ErrNoTenant = errors.New("queue: the Grant carries no tenant, and a job without one cannot be scoped")

// ErrNoName is returned when a job has no name to route by.
var ErrNoName = errors.New("queue: a job with no name cannot be routed to a handler")

// ErrForged is returned when a job claims an action or a tenant the Grant
// pushing it does not carry.
var ErrForged = errors.New("queue: the job does not match the Grant pushing it")

// ErrDetached is returned when a job that was never popped is asked to settle
// itself.
var ErrDetached = errors.New("queue: this job was built to be pushed and never came off a queue, so there is nothing to release, delete or fail")

// ErrPayloadTooLarge is returned when a job's payload is over [MaxPayload].
//
// A sentinel rather than a message, so a caller can tell a payload it can do
// something about -- write the document to storage, push its key -- from a
// store that is down. It is not the same refusal as arguments that cannot be
// encoded at all: those are a defect in the value, and this is a value that is
// merely too big to travel as one.
var ErrPayloadTooLarge = errors.New("queue: the job's payload is larger than a queue stores")

// New builds a job from a Grant and a payload.
//
// This is the only constructor, so every job in the system carries a tenant, an
// id and the Grant that authorized it -- there is no shape of Job that skipped
// any of the three.
func New(g auth.Grant, queue, name string, payload any) (Job, error) {
	if name == "" {
		return Job{}, ErrNoName
	}
	tenant := auth.Tenant(g)
	if tenant == "" {
		return Job{}, ErrNoTenant
	}
	if queue == "" {
		queue = DefaultQueue
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return Job{}, fmt.Errorf("queue: serializing %s: %w", name, err)
	}
	id, err := database.NewID()
	if err != nil {
		return Job{}, err
	}

	return Job{
		UUID:         id,
		Queue:        queue,
		Name:         name,
		TenantID:     tenant,
		Payload:      body,
		AuthorizedBy: g.Subject().ID,
		Action:       string(g.Action()),
	}, nil
}

// GrantFor rebuilds the Grant a job runs under.
//
// The action and the tenant come from the record, so the worker reissues
// exactly what the push authorized -- not more. A worker that invented its own
// Grant would be a way to reach the database with permissions nobody granted.
func GrantFor(j *Job) auth.Grant {
	return auth.SystemGrant(auth.Action(j.Action), j.TenantID)
}

// Authorized reports whether a job may be pushed under this Grant, and whether
// it is small enough for a queue to store.
//
// Every driver calls it at the top of Push, and it closes an escalation the
// contract otherwise allows. [New] builds a job from the Grant, so what it
// produces always matches -- but Push takes a Job, and a Job is a struct
// anybody can fill in:
//
//	j := jobs.Job{UUID: id, Name: "invoice.send", Action: "invoice.delete", TenantID: other}
//	q.Push(ctx, viewGrant, j)
//
// The worker rebuilds the Grant from the record -- [GrantFor] gives
// auth.SystemGrant(j.Action, j.TenantID) -- so the handler would run with an
// action nobody authorized, in a tenant nobody authorized, and every Policy
// downstream would say yes because the Grant looks legitimate. The queue would
// be the one way past the authorization the whole collection exists to enforce.
//
// Checked here rather than in each driver, because a driver that forgets is a
// driver that reopens it.
//
// [MaxPayload] is checked here as well, and last. It is not about the Grant,
// and it is in this function for the reason the name is: this is the one call
// every Push in every driver already makes and already returns the error of, so
// a ceiling that lives here cannot be the ceiling one driver has and the next
// one does not. Last, because a job that is both forged and oversized is a
// forged job first, whatever its size.
//
// A job over the limit is refused before anything is written. The caller
// holding the payload is the only one who can do something about it -- put the
// document in storage and push its key -- and it is holding it now; refused at
// the pop instead, the job has already been stored and the news arrives in a
// worker's log with the code that built the payload long since returned.
func Authorized(g auth.Grant, j Job) error {
	tenant := auth.Tenant(g)
	if tenant == "" {
		return ErrNoTenant
	}
	if j.Name == "" {
		return ErrNoName
	}
	if j.TenantID != "" && j.TenantID != tenant {
		return fmt.Errorf("%w: the job says %q and the Grant says %q",
			ErrForged, j.TenantID, tenant)
	}
	if j.Action != "" && j.Action != string(g.Action()) {
		return fmt.Errorf("%w: the job says %q and the Grant authorizes %q. Build it with jobs.New, which takes both from the Grant",
			ErrForged, j.Action, g.Action())
	}
	if len(j.Payload) > MaxPayload {
		return fmt.Errorf("%w: %s carries %d bytes, and the limit is %d",
			ErrPayloadTooLarge, j.Name, len(j.Payload), MaxPayload)
	}
	return nil
}

// Prepare fills in the defaults a driver would otherwise each have to remember:
// the queue, the run time and the tenant.
//
// It runs after [Authorized] and returns the job a driver stores. Three drivers
// deciding independently what an empty Queue means is three chances to disagree.
func Prepare(g auth.Grant, j Job) Job {
	j.TenantID = auth.Tenant(g)
	if j.Queue == "" {
		j.Queue = DefaultQueue
	}
	// UTC, always. Every other timestamp a driver writes is UTC and every
	// comparison it makes is against time.Now().UTC(), but a RunAt supplied by
	// the caller kept whatever zone it arrived in -- so a job scheduled from a
	// UTC-3 machine was stored three hours in the past on an engine that
	// compares timestamps as text, and ran immediately.
	if j.RunAt.IsZero() {
		j.RunAt = time.Now().UTC()
	} else {
		j.RunAt = j.RunAt.UTC()
	}
	return j
}
