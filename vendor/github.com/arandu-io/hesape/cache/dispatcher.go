package cache

import (
	"time"

	"github.com/arandu-io/hesape/cache/events"
)

// Dispatcher is the one method the cache needs of an event dispatcher.
//
// It is one method, and it is declared here rather than imported because a
// cache that had to import the event bus to fire a CacheHit
// would drag the bus into every binary that caches anything -- and because
// hesape/events is the transactional outbox, which is a different thing with a
// different guarantee.
//
// A dispatcher runs on the calling goroutine, so a listener that blocks blocks
// the cache. Anything slow belongs on a queue, which is where the listener
// should put it.
type Dispatcher interface {
	// Dispatch delivers one event to whoever is listening.
	//
	// It returns nothing: a listener that fails must not fail the cache
	// operation that fired it, and a Repository that had to decide what to do
	// with a listener's error would have to decide it eight times.
	Dispatch(event any)
}

// GetEventDispatcher returns the dispatcher this repository fires into, or nil.
func (r *Repository) GetEventDispatcher() Dispatcher { return r.events }

// SetEventDispatcher returns a repository that fires its events into d.
//
// It derives and returns a new repository rather than mutating this one, for
// the reason SetStore and SetDefaultCacheTime derive: a repository handed to
// two modules must not change underneath one of them.
func (r *Repository) SetEventDispatcher(d Dispatcher) *Repository {
	out := *r
	out.events = d
	return &out
}

// event fires one event, if anybody is listening.
func (r *Repository) event(e any) {
	if r.events == nil {
		return
	}
	r.events.Dispatch(e)
}

// tagNames is the tag set carried into every event, and nil on an untagged
// repository.
//
// One repository serves both the tagged and the untagged case, so the nil is
// what tells them apart.
func (r *Repository) tagNames() []string {
	if r.tags == nil {
		return nil
	}
	return r.tags.GetNames()
}

// seconds is a ttl as the events carry it.
//
// The events hold an int, because a listener that formats one should not have
// to convert; everything else in this package holds a time.Duration, because
// that is what a caller writes.
func seconds(ttl time.Duration) int { return int(ttl / time.Second) }

// eventRetrieving fires RetrievingKey.
func (r *Repository) eventRetrieving(key string) {
	if r.events == nil {
		return
	}
	r.event(events.NewRetrievingKey(r.GetName(), key, r.tagNames()))
}

// eventHit fires CacheHit.
func (r *Repository) eventHit(key string, value any) {
	if r.events == nil {
		return
	}
	r.event(events.NewCacheHit(r.GetName(), key, value, r.tagNames()))
}

// eventMissed fires CacheMissed.
func (r *Repository) eventMissed(key string) {
	if r.events == nil {
		return
	}
	r.event(events.NewCacheMissed(r.GetName(), key, r.tagNames()))
}

// eventWriting fires WritingKey.
func (r *Repository) eventWriting(key string, value any, ttl time.Duration) {
	if r.events == nil {
		return
	}
	r.event(events.NewWritingKey(r.GetName(), key, value, seconds(ttl), r.tagNames()))
}

// eventWritten fires KeyWritten on success and KeyWriteFailed on failure.
func (r *Repository) eventWritten(key string, value any, ttl time.Duration, err error) {
	if r.events == nil {
		return
	}
	if err != nil {
		r.event(events.NewKeyWriteFailed(r.GetName(), key, value, seconds(ttl), r.tagNames()))
		return
	}
	r.event(events.NewKeyWritten(r.GetName(), key, value, seconds(ttl), r.tagNames()))
}

// eventForgetting fires ForgettingKey.
func (r *Repository) eventForgetting(key string) {
	if r.events == nil {
		return
	}
	r.event(events.NewForgettingKey(r.GetName(), key, r.tagNames()))
}

// eventForgotten fires KeyForgotten on success and KeyForgetFailed on failure.
func (r *Repository) eventForgotten(key string, err error) {
	if r.events == nil {
		return
	}
	if err != nil {
		r.event(events.NewKeyForgetFailed(r.GetName(), key, r.tagNames()))
		return
	}
	r.event(events.NewKeyForgotten(r.GetName(), key, r.tagNames()))
}
