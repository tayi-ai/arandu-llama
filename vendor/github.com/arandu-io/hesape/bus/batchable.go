package bus

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/arandu-io/hesape/auth"
)

// formatVersion marks a payload as one this package wrote.
//
// It is what tells a batch member apart from an ordinary job whose arguments
// happen to be a JSON object, and it is a number rather than a bool because the
// day the envelope changes shape, a worker running the old binary has to be
// able to say so instead of silently reading the new one as the old.
const formatVersion = 1

// envelope is what bus puts around a job's arguments.
//
// The remaining links of a chain travel in it, which is why a chain needs no
// table: the chain is exactly as durable as the queue, and an abandoned one
// leaves nothing behind to clean up.
//
// Catch carries no omitempty because encoding/json has none for a struct: a
// zero Step encodes as a name nobody set, which is exactly what "this chain has
// no Catch" means, and declared() is what reads it back.
type envelope struct {
	Bus   int             `json:"bus"`
	Batch string          `json:"batch,omitempty"`
	Job   string          `json:"job,omitempty"`
	Chain []Step          `json:"chain,omitempty"`
	Catch Step            `json:"chain_catch"`
	Body  json.RawMessage `json:"body,omitempty"`
}

// wrap serializes an envelope.
func wrap(e envelope) ([]byte, error) {
	raw, err := json.Marshal(e)
	if err != nil {
		return nil, fmt.Errorf("bus: serializing the envelope: %w", err)
	}
	return raw, nil
}

// Queueable is the dispatch settings a job carries: where it goes, when a
// worker may take it, and what runs after it.
//
// It is a struct a caller fills in, and what travels to the worker is the
// envelope.
type Queueable struct {
	// Connection and Queue are where the job goes. ChainConnection and
	// ChainQueue are where the rest of its chain goes, which is not always the
	// same place: the work can be spread over a fast queue and its report
	// filed on a slow one.
	Connection      string `json:"connection,omitempty"`
	Queue           string `json:"queue,omitempty"`
	ChainConnection string `json:"chain_connection,omitempty"`
	ChainQueue      string `json:"chain_queue,omitempty"`
	// MessageGroup and Deduplicator are carried for the transports that have
	// them. Nothing in this package reads them: they are what a queue driver
	// puts on the message, and a driver that has no such thing ignores them.
	MessageGroup string `json:"message_group,omitempty"`
	Deduplicator string `json:"deduplicator,omitempty"`
	// DelaySeconds is how long the job waits before a worker may take it.
	//
	// Go has one namespace for the fields and the methods of a type, so the
	// field and its setter cannot both be called Delay: the setter keeps the
	// short name, because Delay is what somebody types, and the field says what
	// it holds.
	DelaySeconds int `json:"delay,omitempty"`
	// DispatchAfterCommit says whether the push waits for the surrounding
	// database transaction. Nil is "whatever the driver is configured to do".
	// It is named apart from its setter for the reason DelaySeconds is.
	DispatchAfterCommit *bool `json:"after_commit,omitempty"`
	// Middleware is the names of the wrappers the handler runs inside.
	Middleware []string `json:"middleware,omitempty"`
	// Chained is the links that have not run yet, and ChainCatch is the job
	// that reports a chain that stopped.
	Chained    []Step `json:"chained,omitempty"`
	ChainCatch Step   `json:"chain_catch"`
}

// OnConnection sets which queue connection the job goes to.
func (q Queueable) OnConnection(connection string) Queueable {
	q.Connection = connection
	return q
}

// OnQueue sets which queue the job goes on.
func (q Queueable) OnQueue(queue string) Queueable {
	q.Queue = queue
	return q
}

// OnGroup sets the message group, for the transports that order by one.
func (q Queueable) OnGroup(group string) Queueable {
	q.MessageGroup = group
	return q
}

// WithDeduplicator sets the deduplication id, for the transports that have one.
//
// It takes the id rather than a function that computes one, because a closure
// cannot cross the gap between the process that dispatched the job and the one
// that runs it (see the package doc).
func (q Queueable) WithDeduplicator(id string) Queueable {
	q.Deduplicator = id
	return q
}

// AllOnConnection sets the connection for this job and for the rest of its
// chain.
func (q Queueable) AllOnConnection(connection string) Queueable {
	q.ChainConnection = connection
	q.Connection = connection
	return q
}

// AllOnQueue sets the queue for this job and for the rest of its chain.
func (q Queueable) AllOnQueue(queue string) Queueable {
	q.ChainQueue = queue
	q.Queue = queue
	return q
}

// Delay sets how many seconds the job waits before a worker may take it.
//
// It writes DelaySeconds -- see there for why the field and the setter could
// not keep the same spelling.
func (q Queueable) Delay(seconds int) Queueable {
	q.DelaySeconds = seconds
	return q
}

// WithoutDelay sets the delay to zero.
func (q Queueable) WithoutDelay() Queueable {
	q.DelaySeconds = 0
	return q
}

// AfterCommit makes the push wait for the surrounding database transaction.
//
// It is the fix for the oldest race in a queued application: a worker picking
// up a job about a row the transaction has not committed yet, and failing
// because the row is not there.
func (q Queueable) AfterCommit() Queueable {
	yes := true
	q.DispatchAfterCommit = &yes
	return q
}

// BeforeCommit pushes without waiting for the transaction.
func (q Queueable) BeforeCommit() Queueable {
	no := false
	q.DispatchAfterCommit = &no
	return q
}

// Through sets the middleware the handler runs inside, replacing whatever was
// there.
func (q Queueable) Through(middleware ...string) Queueable {
	q.Middleware = append([]string(nil), middleware...)
	return q
}

// Chain sets the jobs that run after this one succeeds, replacing whatever was
// there.
func (q Queueable) Chain(jobs ...Step) Queueable {
	q.Chained = append([]Step(nil), jobs...)
	return q
}

// PrependToChain puts jobs at the front of the chain, so they run immediately
// after the one in hand.
func (q Queueable) PrependToChain(jobs ...Step) Queueable {
	q.Chained = append(append([]Step(nil), jobs...), q.Chained...)
	return q
}

// AppendToChain puts jobs at the end of the chain.
func (q Queueable) AppendToChain(jobs ...Step) Queueable {
	q.Chained = append(append([]Step(nil), q.Chained...), jobs...)
	return q
}

// Batchable is what a job belongs to: a batch, a chain, both, or neither.
//
// The batch id and the job id are read back out of the envelope by Batched, and
// the repository the batch is then loaded through is an argument.
//
// The zero value is a job that was pushed on its own, and Handled on it does
// nothing. That is deliberate: a handler can call Batched and Handled
// unconditionally, and a job that was never part of anything costs one failed
// type assertion for the privilege of not having two versions of every handler.
type Batchable struct {
	// Queueable is where this job and the rest of its chain go.
	Queueable
	// BatchID is the batch the job belongs to, empty when it belongs to none.
	BatchID string
	// JobID identifies this delivery inside the batch. It is what
	// DecrementPendingJobs and IncrementFailedJobs record, so that a job which
	// failed and was then retried into success stops counting as a failure.
	JobID string
	// fake is what WithFakeBatch put there: a batch that exists only for this
	// value, so a handler's batch branch can be exercised without a repository.
	fake *Batch
}

// Batched reports what a job belongs to and decodes its arguments into v.
//
// It is the first of the two calls a handler makes:
//
//	var row Invoice
//	m, err := bus.Batched(j.Payload, &row)
//
// What arrives at a worker is bytes, so the envelope has to be read back before
// anything in it can be asked a question.
//
// A payload this package did not write is returned as it is, with a zero
// Batchable, so a handler written for batches still works on a job pushed
// directly. Pass nil for v when the arguments are not wanted.
func Batched(payload []byte, v any) (Batchable, error) {
	var e envelope
	if err := json.Unmarshal(payload, &e); err != nil || e.Bus == 0 {
		// Not ours. The payload is the arguments.
		return Batchable{}, decodeInto(payload, v)
	}
	if e.Bus != formatVersion {
		return Batchable{}, fmt.Errorf("bus: this payload was written in envelope format %d and this binary reads %d", e.Bus, formatVersion)
	}
	m := Batchable{BatchID: e.Batch, JobID: e.Job}
	m.Chained = e.Chain
	m.ChainCatch = e.Catch
	return m, decodeInto(e.Body, v)
}

// WithBatchId puts the job in a batch.
//
// The d is lowercase, unlike [Batchable.BatchID]: the field is a Go identifier
// and takes the Go spelling of an initialism, and the method keeps the spelling
// somebody greps for.
func (m Batchable) WithBatchId(batchID string) Batchable {
	m.BatchID = batchID
	return m
}

// WithFakeBatch attaches a batch that exists only in this value.
//
// It is for the handler test that has to reach the batch branch without a
// repository, a table and a dispatch behind it. The Batch is returned as well
// as attached, so the test can assert on the same value the handler will see.
func (m Batchable) WithFakeBatch(b Batch) (Batchable, Batch) {
	if b.ID == "" {
		b.ID = "fake-batch"
	}
	m.BatchID = b.ID
	m.fake = &b
	return m, b
}

// Batch is the batch the job belongs to.
//
// It reports database.ErrNotFound wrapped when the job belongs to no batch, so
// a caller that forgot to check gets an error rather than a nil dereference on
// the next line.
func (m Batchable) Batch(ctx context.Context, g auth.Grant, r BatchRepository) (Batch, error) {
	if m.fake != nil {
		return *m.fake, nil
	}
	if m.BatchID == "" {
		return Batch{}, notFound("")
	}
	if r == nil {
		return Batch{}, fmt.Errorf("bus: reading batch %s needs a repository", m.BatchID)
	}
	return r.Find(ctx, g, m.BatchID)
}

// Batching reports whether the batch is still worth working for.
//
// A handler asks before it does anything expensive. It is how cancelling a
// batch means something: the jobs already in the queue cannot be recalled, so
// they are delivered, they ask, and they skip. A batch that stopped on a
// failure answers false for the same reason -- cancelling is what the first
// failure of a batch that does not allow them does.
//
// A job in no batch is always worth working for, and the repository is not
// consulted.
func (m Batchable) Batching(ctx context.Context, g auth.Grant, r BatchRepository) (bool, error) {
	if m.BatchID == "" {
		return true, nil
	}
	b, err := m.Batch(ctx, g, r)
	if err != nil {
		return false, err
	}
	return !b.Cancelled(), nil
}

// DispatchNextJobInChain pushes the link that follows this one, if there is
// one.
//
// It is what makes a chain a chain: there is no moment at which two links are
// in the queue at once, so the order is real rather than hoped for.
func (m Batchable) DispatchNextJobInChain(ctx context.Context, g auth.Grant, q Queue) error {
	if len(m.Chained) == 0 {
		return nil
	}
	if q == nil {
		return errors.New("bus: pushing the next link of a chain needs a queue")
	}

	next, rest := m.Chained[0], m.Chained[1:]
	payload, err := wrap(envelope{Bus: formatVersion, Chain: rest, Catch: m.ChainCatch, Body: next.Payload})
	if err != nil {
		return err
	}
	if err := q.Push(ctx, g, m.queueForChain(next), next.Name, payload); err != nil {
		return fmt.Errorf("bus: pushing the next link %s: %w", next.Name, err)
	}
	return nil
}

// InvokeChainCatchCallbacks pushes the job that reports a chain that stopped.
//
// What is registered is a job rather than a function, so what happens here is a
// push (see the package doc).
func (m Batchable) InvokeChainCatchCallbacks(ctx context.Context, g auth.Grant, q Queue, _ error) error {
	if !m.ChainCatch.declared() {
		return nil
	}
	if q == nil {
		return errors.New("bus: reporting a chain that stopped needs a queue")
	}
	return q.Push(ctx, g, m.queueForChain(m.ChainCatch), m.ChainCatch.Name, m.ChainCatch.Payload)
}

// AssertHasChain reports the error a test fails on when the remaining chain is
// not the expected list of job names.
//
// A library that imported testing to fail the test itself would put the test
// framework in every binary, so this returns the failure and the test calls
// t.Fatal on it -- which is also how it composes with a table-driven test.
func (m Batchable) AssertHasChain(expected ...string) error {
	if len(expected) == 0 {
		return errors.New("bus: the expected chain cannot be empty")
	}
	got := make([]string, 0, len(m.Chained))
	for _, s := range m.Chained {
		got = append(got, s.Name)
	}
	if len(got) != len(expected) {
		return fmt.Errorf("bus: the job chains %v, want %v", got, expected)
	}
	for i := range got {
		if got[i] != expected[i] {
			return fmt.Errorf("bus: the job chains %v, want %v", got, expected)
		}
	}
	return nil
}

// AssertDoesntHaveChain reports the error a test fails on when the job still
// has links after it.
func (m Batchable) AssertDoesntHaveChain() error {
	if len(m.Chained) == 0 {
		return nil
	}
	return fmt.Errorf("bus: the job still chains %d job(s)", len(m.Chained))
}

// queueForChain is where a link goes: its own queue, or the chain's.
func (q Queueable) queueForChain(s Step) string {
	if s.Queue != "" {
		return s.Queue
	}
	return q.ChainQueue
}

// decodeInto unmarshals raw into v, tolerating both being absent.
func decodeInto(raw []byte, v any) error {
	if v == nil || len(raw) == 0 {
		return nil
	}
	if err := json.Unmarshal(raw, v); err != nil {
		return fmt.Errorf("bus: decoding the job payload: %w", err)
	}
	return nil
}

// Handled reports one job's outcome and does whatever that outcome triggers.
//
// It is the second of the two calls a handler makes, and the one that has to
// happen on every path out of it -- including the failing one, which is why it
// takes the error rather than being called only on success:
//
//	err := doTheWork(ctx, g, row)
//	return errors.Join(err, bus.Handled(ctx, g, repo, queue, m, err))
//
// It is Batch.RecordSuccessfulJob or Batch.RecordFailedJob, plus
// DispatchNextJobInChain or InvokeChainCatchCallbacks, in one call. Those four
// are where the behaviour lives; this is the one line a handler writes instead
// of the four branches.
//
// A job that belongs to neither a batch nor a chain does nothing and returns
// nil.
func Handled(ctx context.Context, g auth.Grant, r BatchRepository, q Queue, m Batchable, cause error) error {
	if m.BatchID == "" && len(m.Chained) == 0 && !m.ChainCatch.declared() {
		return nil
	}
	if q == nil {
		return errors.New("bus: reporting a job needs a queue to push what the outcome triggers")
	}

	var errs []error
	if m.BatchID != "" {
		if r == nil {
			errs = append(errs, fmt.Errorf("bus: reporting a job of batch %s needs a repository", m.BatchID))
		} else {
			b, err := r.Find(ctx, g, m.BatchID)
			switch {
			case err != nil:
				errs = append(errs, err)
			case cause != nil:
				errs = append(errs, b.RecordFailedJob(ctx, g, r, q, m.JobID, cause))
			default:
				errs = append(errs, b.RecordSuccessfulJob(ctx, g, r, q, m.JobID))
			}
		}
	}

	// A chain stops at its first failure and does not resume: the point of a
	// chain is that link N assumes link N-1 happened.
	if cause != nil {
		errs = append(errs, m.InvokeChainCatchCallbacks(ctx, g, q, cause))
	} else {
		errs = append(errs, m.DispatchNextJobInChain(ctx, g, q))
	}
	return errors.Join(errs...)
}

// push sends a callback as an ordinary job.
//
// A callback carries no envelope: it is not part of the batch it closes, and a
// callback that reported to its own batch would decrement a counter that has
// already reached zero.
func push(ctx context.Context, g auth.Grant, q Queue, b Batch, s Step) error {
	if q == nil {
		return fmt.Errorf("bus: pushing the callback %s of batch %s needs a queue", s.Name, b.ID)
	}
	if err := q.Push(ctx, g, b.queueFor(s), s.Name, s.Payload); err != nil {
		return fmt.Errorf("bus: pushing the callback %s of batch %s: %w", s.Name, b.ID, err)
	}
	return nil
}
