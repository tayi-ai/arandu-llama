package attributes

import "time"

// Attributes are the per-job settings that travel with the job.
//
// One setting per field, and the decision is made once here so nobody has to
// make it again: Go has no runtime metadata on a type, and the two candidates
// were struct tags read by reflection or a struct of options passed by value.
// Struct tags lose the compiler -- a typo in `queue:"tires=5"` is a silent zero
// -- and they only attach to a struct, while a job here is a name and a
// payload. A struct of options is checked at the call site, and it is the same
// value the driver stores and the worker reads back.
//
// The zero value means "the worker decides": Tries of 0 falls back to
// WorkerOptions.MaxTries, Timeout of 0 to WorkerOptions.Lease.
type Attributes struct {
	// Backoff is how long to wait before each retry. One duration per attempt,
	// and the last one repeats once the list runs out.
	//
	// Empty means WorkerOptions.Backoff decides.
	Backoff []time.Duration `json:"backoff,omitempty"`

	// Connection is the queue connection this job belongs on, by the name it
	// was registered under in QueueManager. Empty means the default.
	Connection string `json:"connection,omitempty"`

	// DeleteWhenMissingModels deletes the job instead of failing it when the
	// row its payload names is gone.
	//
	// A job that refers to a deleted invoice cannot succeed on any retry, and
	// failing it puts a row in the dead letter list that a person has to read
	// and dismiss. This is what says "that is expected, drop it".
	DeleteWhenMissingModels bool `json:"deleteWhenMissingModels,omitempty"`

	// FailOnTimeout parks the job when the handler runs past Timeout, instead
	// of releasing it to be tried again.
	FailOnTimeout bool `json:"failOnTimeout,omitempty"`

	// MaxExceptions is how many failures the job gets before it is parked, when
	// that number is smaller than Tries.
	//
	// A job may be released many times by a middleware -- rate limited,
	// overlapping -- without ever having thrown, and those releases spend
	// Tries. MaxExceptions counts only the deliveries that ended in an
	// error.
	MaxExceptions int `json:"maxExceptions,omitempty"`

	// Queue is the queue this job goes on. Empty means jobs.DefaultQueue.
	Queue string `json:"queue,omitempty"`

	// RetryUntil is the deadline after which the job stops being retried,
	// whatever Tries says. Zero means no deadline.
	//
	// It is here because it is the fourth thing the worker reads before
	// deciding to release a job, and it belongs next to the other three.
	RetryUntil time.Time `json:"retryUntil,omitzero"`

	// Timeout is how long the handler may run. Zero means WorkerOptions.Lease.
	Timeout time.Duration `json:"timeout,omitempty"`

	// Tries is how many deliveries the job gets before it is parked. Zero means
	// WorkerOptions.MaxTries, and a negative number means forever.
	Tries int `json:"maxTries,omitempty"`

	// UniqueFor is how long the uniqueness lock on this job is held, for a job
	// that must not be queued twice.
	UniqueFor time.Duration `json:"uniqueFor,omitempty"`

	// WithoutRelations strips the loaded relations from the models in the
	// payload before it is written.
	//
	// It is advice to whatever builds the payload rather than something the
	// queue can enforce, because the payload is JSON the caller marshalled: a
	// job that serializes an object graph it did not mean to send is a job with
	// a large payload and a stale read, and this is where that intent is
	// recorded.
	WithoutRelations bool `json:"withoutRelations,omitempty"`
}

// ReadsQueueAttributes is what a job type implements to carry its own settings.
//
// A type that declares its settings implements it, and [Of] is what reads them
// back.
//
//	type SendInvoice struct{ InvoiceID string }
//
//	func (SendInvoice) QueueAttributes() attributes.Attributes {
//		return attributes.Attributes{Tries: 5, Timeout: 30 * time.Second}
//	}
type ReadsQueueAttributes interface {
	QueueAttributes() Attributes
}

// Of returns the attributes a value declares, or the zero value when it
// declares none.
//
// A type either implements [ReadsQueueAttributes] or it does not, and an
// embedded struct that implements it answers for the type that embeds it.
func Of(v any) Attributes {
	if reader, declares := v.(ReadsQueueAttributes); declares {
		return reader.QueueAttributes()
	}
	return Attributes{}
}

// BackoffFor is how long to wait before attempt n, counting from one.
//
// The last entry repeats once the list runs out. An empty list returns zero,
// and the worker's own schedule takes over.
func (a Attributes) BackoffFor(attempt int) time.Duration {
	if len(a.Backoff) == 0 {
		return 0
	}
	if attempt < 1 {
		attempt = 1
	}
	if attempt > len(a.Backoff) {
		return a.Backoff[len(a.Backoff)-1]
	}
	return a.Backoff[attempt-1]
}
