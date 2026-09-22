package events

// CacheEvent is what every keyed cache event carries: which store, which key,
// and under which tags.
//
// It is a struct the other events embed, so a listener written against CacheHit
// reads e.Key and e.StoreName directly.
//
// It is not an interface. A listener that wants "any cache event" switches on
// the concrete type, and a listener that wants the common part reads the
// embedded CacheEvent.
type CacheEvent struct {
	// StoreName is the name of the cache store the event came from. It is empty
	// when the repository was built without one.
	StoreName string

	// Key is the key the event is about.
	Key string

	// Tags are the tags assigned to the key, and nil on an untagged repository.
	Tags []string
}

// SetTags sets the tags for the cache event.
//
// It is on a pointer receiver and returns the event, so assigning tags and
// dispatching read as one line.
func (e *CacheEvent) SetTags(tags []string) *CacheEvent {
	e.Tags = tags
	return e
}

// RetrievingKey is fired before a key is read.
//
// It is the "about to look" event, and it fires whether or not anything is
// there -- CacheHit or CacheMissed follows it.
type RetrievingKey struct{ CacheEvent }

// NewRetrievingKey returns the event.
func NewRetrievingKey(storeName, key string, tags []string) *RetrievingKey {
	return &RetrievingKey{CacheEvent{StoreName: storeName, Key: key, Tags: tags}}
}

// RetrievingManyKeys is fired before several keys are read at once.
type RetrievingManyKeys struct {
	CacheEvent

	// Keys are the keys being retrieved.
	Keys []string
}

// NewRetrievingManyKeys returns the event.
//
// Key is the first of keys, and the empty string when there are none.
func NewRetrievingManyKeys(storeName string, keys []string, tags []string) *RetrievingManyKeys {
	first := ""
	if len(keys) > 0 {
		first = keys[0]
	}
	return &RetrievingManyKeys{
		CacheEvent: CacheEvent{StoreName: storeName, Key: first, Tags: tags},
		Keys:       keys,
	}
}

// CacheHit is fired when a key was found.
type CacheHit struct {
	CacheEvent

	// Value is the value that was retrieved.
	Value any
}

// NewCacheHit returns the event.
func NewCacheHit(storeName, key string, value any, tags []string) *CacheHit {
	return &CacheHit{
		CacheEvent: CacheEvent{StoreName: storeName, Key: key, Tags: tags},
		Value:      value,
	}
}

// CacheMissed is fired when a key was not found.
//
// It is the event a cache hit rate is computed from, together with CacheHit.
type CacheMissed struct{ CacheEvent }

// NewCacheMissed returns the event.
func NewCacheMissed(storeName, key string, tags []string) *CacheMissed {
	return &CacheMissed{CacheEvent{StoreName: storeName, Key: key, Tags: tags}}
}

// WritingKey is fired before a key is written. KeyWritten or KeyWriteFailed
// follows it.
type WritingKey struct {
	CacheEvent

	// Value is the value that will be written.
	Value any

	// Seconds is how long the key should be valid for. It is a whole number of
	// seconds rather than a time.Duration, so a listener that formats or logs
	// it does not have to convert.
	Seconds int
}

// NewWritingKey returns the event.
func NewWritingKey(storeName, key string, value any, seconds int, tags []string) *WritingKey {
	return &WritingKey{
		CacheEvent: CacheEvent{StoreName: storeName, Key: key, Tags: tags},
		Value:      value,
		Seconds:    seconds,
	}
}

// KeyWritten is fired after a key was written.
type KeyWritten struct {
	CacheEvent

	// Value is the value that was written.
	Value any

	// Seconds is how long the key is valid for.
	Seconds int
}

// NewKeyWritten returns the event.
func NewKeyWritten(storeName, key string, value any, seconds int, tags []string) *KeyWritten {
	return &KeyWritten{
		CacheEvent: CacheEvent{StoreName: storeName, Key: key, Tags: tags},
		Value:      value,
		Seconds:    seconds,
	}
}

// KeyWriteFailed is fired when a write did not happen.
//
// This is the event worth listening to: a cache that has quietly stopped
// accepting writes looks exactly like a cache with a very low hit rate, and
// only this tells the two apart.
type KeyWriteFailed struct {
	CacheEvent

	// Value is the value that would have been written.
	Value any

	// Seconds is how long the key should have been valid for.
	Seconds int
}

// NewKeyWriteFailed returns the event.
func NewKeyWriteFailed(storeName, key string, value any, seconds int, tags []string) *KeyWriteFailed {
	return &KeyWriteFailed{
		CacheEvent: CacheEvent{StoreName: storeName, Key: key, Tags: tags},
		Value:      value,
		Seconds:    seconds,
	}
}

// WritingManyKeys is fired before several keys are written at once.
type WritingManyKeys struct {
	CacheEvent

	// Keys are the keys being written.
	Keys []string

	// Values are the values being written, in the order of Keys.
	Values []any

	// Seconds is how long the keys should be valid for.
	Seconds int
}

// NewWritingManyKeys returns the event.
//
// Key is the first of keys, and the empty string when there are none.
func NewWritingManyKeys(storeName string, keys []string, values []any, seconds int, tags []string) *WritingManyKeys {
	first := ""
	if len(keys) > 0 {
		first = keys[0]
	}
	return &WritingManyKeys{
		CacheEvent: CacheEvent{StoreName: storeName, Key: first, Tags: tags},
		Keys:       keys,
		Values:     values,
		Seconds:    seconds,
	}
}

// ForgettingKey is fired before a key is removed. KeyForgotten or
// KeyForgetFailed follows it.
type ForgettingKey struct{ CacheEvent }

// NewForgettingKey returns the event.
func NewForgettingKey(storeName, key string, tags []string) *ForgettingKey {
	return &ForgettingKey{CacheEvent{StoreName: storeName, Key: key, Tags: tags}}
}

// KeyForgotten is fired after a key was removed.
type KeyForgotten struct{ CacheEvent }

// NewKeyForgotten returns the event.
func NewKeyForgotten(storeName, key string, tags []string) *KeyForgotten {
	return &KeyForgotten{CacheEvent{StoreName: storeName, Key: key, Tags: tags}}
}

// KeyForgetFailed is fired when a key could not be removed.
//
// It is the one that matters after a deploy: an invalidation that failed leaves
// the old value being served, and nothing else says so.
type KeyForgetFailed struct{ CacheEvent }

// NewKeyForgetFailed returns the event.
func NewKeyForgetFailed(storeName, key string, tags []string) *KeyForgetFailed {
	return &KeyForgetFailed{CacheEvent{StoreName: storeName, Key: key, Tags: tags}}
}

// CacheFlushing is fired before a store is flushed.
//
// It carries no key, because a flush is about all of them.
type CacheFlushing struct {
	// StoreName is the name of the cache store.
	StoreName string

	// Tags are the tags being flushed, and nil when the whole namespace is.
	Tags []string
}

// NewCacheFlushing returns the event.
func NewCacheFlushing(storeName string, tags []string) *CacheFlushing {
	return &CacheFlushing{StoreName: storeName, Tags: tags}
}

// SetTags sets the tags for the event and returns it.
func (e *CacheFlushing) SetTags(tags []string) *CacheFlushing {
	e.Tags = tags
	return e
}

// CacheFlushed is fired after a store was flushed.
type CacheFlushed struct {
	// StoreName is the name of the cache store.
	StoreName string

	// Tags are the tags that were flushed.
	Tags []string
}

// NewCacheFlushed returns the event.
func NewCacheFlushed(storeName string, tags []string) *CacheFlushed {
	return &CacheFlushed{StoreName: storeName, Tags: tags}
}

// SetTags sets the tags for the event and returns it.
func (e *CacheFlushed) SetTags(tags []string) *CacheFlushed {
	e.Tags = tags
	return e
}

// CacheFlushFailed is fired when a flush did not happen.
type CacheFlushFailed struct {
	// StoreName is the name of the cache store.
	StoreName string

	// Tags are the tags that were being flushed.
	Tags []string
}

// NewCacheFlushFailed returns the event.
func NewCacheFlushFailed(storeName string, tags []string) *CacheFlushFailed {
	return &CacheFlushFailed{StoreName: storeName, Tags: tags}
}

// SetTags sets the tags for the event and returns it.
func (e *CacheFlushFailed) SetTags(tags []string) *CacheFlushFailed {
	e.Tags = tags
	return e
}

// CacheLocksFlushing is fired before every lock in a store is released.
type CacheLocksFlushing struct {
	// StoreName is the name of the cache store.
	StoreName string
}

// NewCacheLocksFlushing returns the event.
func NewCacheLocksFlushing(storeName string) *CacheLocksFlushing {
	return &CacheLocksFlushing{StoreName: storeName}
}

// CacheLocksFlushed is fired after every lock in a store was released.
type CacheLocksFlushed struct {
	// StoreName is the name of the cache store.
	StoreName string
}

// NewCacheLocksFlushed returns the event.
func NewCacheLocksFlushed(storeName string) *CacheLocksFlushed {
	return &CacheLocksFlushed{StoreName: storeName}
}

// CacheLocksFlushFailed is fired when the locks could not be released.
type CacheLocksFlushFailed struct {
	// StoreName is the name of the cache store.
	StoreName string
}

// NewCacheLocksFlushFailed returns the event.
func NewCacheLocksFlushFailed(storeName string) *CacheLocksFlushFailed {
	return &CacheLocksFlushFailed{StoreName: storeName}
}

// CacheFailedOver is fired when one store of a failover set refused an
// operation and the next one was tried.
//
// It fires once per store per failure, and only when that store was not already
// failing -- so a cache that has been down for an hour produces one event, not
// one per request.
type CacheFailedOver struct {
	// StoreName is the name of the cache store that failed.
	StoreName string

	// Err is what it failed with.
	Err error
}

// NewCacheFailedOver returns the event.
func NewCacheFailedOver(storeName string, err error) *CacheFailedOver {
	return &CacheFailedOver{StoreName: storeName, Err: err}
}
