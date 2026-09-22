package queue

import (
	"sync"
	"time"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/queue/attributes"
	"github.com/arandu-io/hesape/queue/jobs"
)

// PayloadHook may add to a job on its way onto a queue.
//
// It is what [CreatePayloadUsing] registers, and it is how an observability
// package adds a trace id to every job it never wrote. It is given the
// connection, the queue and the job as it stands; what it changes, it changes
// on the job.
//
// It takes a pointer rather than returning something to be merged over the
// record, because merging untyped maps is how a hook silently overwrites
// maxTries.
type PayloadHook func(connection, queue string, j *jobs.Job)

var payloadHooks struct {
	sync.RWMutex
	hooks []PayloadHook
}

// CreatePayloadUsing registers a hook that runs while a job's record is built.
//
// Passing nil clears the registered hooks rather than adding a nil one: a test
// that registers a hook has no other way to take it back.
//
// It is process-wide. A hook is registered once at boot -- tracing, an audit
// field -- and never per request; anything that varies per job belongs on the
// job.
func CreatePayloadUsing(hook PayloadHook) {
	payloadHooks.Lock()
	defer payloadHooks.Unlock()
	if hook == nil {
		payloadHooks.hooks = nil
		return
	}
	payloadHooks.hooks = append(payloadHooks.hooks, hook)
}

// CreatePayload builds the record a driver stores.
//
// It takes what the caller pushed and returns the thing that goes on the wire,
// with the identity, the tenant and the settings filled in. The return is a
// [jobs.Job] rather than an encoded envelope, because the record has columns
// and the arguments are one of them.
//
// delay is folded into RunAt rather than kept beside it: a delay and an
// availability instant carried side by side can disagree, and one field
// cannot.
func CreatePayload(g auth.Grant, connection, queue, name string, data any, delay time.Duration) (jobs.Job, error) {
	j, err := jobs.New(g, queue, name, data)
	if err != nil {
		return jobs.Job{}, err
	}

	j.Attributes = attributes.Of(data)
	if j.Attributes.Queue != "" && queue == "" {
		j.Queue = j.Attributes.Queue
	}
	j.DisplayName = GetDisplayName(data)
	if delay > 0 {
		j.RunAt = time.Now().UTC().Add(delay)
	}

	payloadHooks.RLock()
	hooks := payloadHooks.hooks
	payloadHooks.RUnlock()
	for _, hook := range hooks {
		hook(connection, j.Queue, &j)
	}
	return j, nil
}

// HasDisplayName is a job value that names itself for a person.
//
// A value that does not implement it is named by the job name it was pushed
// under.
type HasDisplayName interface {
	// DisplayName is what a console, a log line and the failed job list show.
	DisplayName() string
}

// GetDisplayName is what to show for a job's arguments, or empty when the value
// has nothing to say.
func GetDisplayName(data any) string {
	if named, says := data.(HasDisplayName); says {
		return named.DisplayName()
	}
	return ""
}

// GetJobTries is how many deliveries a job value asks for, or zero when it asks
// for nothing and the worker's own limit decides.
//
// It reads attributes.Attributes.Tries, which is the one place a value says
// it.
func GetJobTries(data any) int { return attributes.Of(data).Tries }

// GetJobBackoff is how long a job value asks to wait between deliveries, or nil
// when the worker's own schedule decides.
//
// It is the slice attributes.Attributes.Backoff holds, not an encoding of
// it.
func GetJobBackoff(data any) []time.Duration { return attributes.Of(data).Backoff }

// GetJobExpiration is the deadline after which a job value stops being retried,
// or the zero time when there is none.
//
// It is a time.Time rather than a Unix timestamp: the same instant with its
// unit attached.
func GetJobExpiration(data any) time.Time { return attributes.Of(data).RetryUntil }
