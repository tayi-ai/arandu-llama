package deferpkg

import (
	"crypto/rand"
	"encoding/hex"
	"sync"
)

// DeferredCallback is work put off until the response has been sent, carrying
// a name so it can be called off before it runs.
//
// [DeferredCallback.Name] and [DeferredCallback.Always] write the two
// settings; [DeferredCallback.GetName], [DeferredCallback.GetAlways] and
// [DeferredCallback.GetCallback] read them back, because a field and a method
// cannot share one name in Go.
type DeferredCallback struct {
	callback func()
	name     string
	always   bool
}

// NewDeferredCallback builds a deferred callback. An empty name is filled with
// a random one, so the callback is not deduplicated against another.
func NewDeferredCallback(callback func(), name string, always bool) *DeferredCallback {
	if name == "" {
		name = randomName()
	}
	return &DeferredCallback{callback: callback, name: name, always: always}
}

// randomName returns a version 4 UUID, used as a name no other callback will
// have. An unreadable random source yields the empty string.
func randomName() string {
	raw := make([]byte, 16)
	if _, err := rand.Read(raw); err != nil {
		return ""
	}
	raw[6] = (raw[6] & 0x0f) | 0x40
	raw[8] = (raw[8] & 0x3f) | 0x80
	return hex.EncodeToString(raw[0:4]) + "-" + hex.EncodeToString(raw[4:6]) + "-" +
		hex.EncodeToString(raw[6:8]) + "-" + hex.EncodeToString(raw[8:10]) + "-" +
		hex.EncodeToString(raw[10:16])
}

// Name sets the name the callback can be called off by, and returns the
// callback.
func (d *DeferredCallback) Name(name string) *DeferredCallback {
	d.name = name
	return d
}

// Always marks the callback to run even when the request or the job failed,
// and returns the callback. The variadic argument defaults to true.
func (d *DeferredCallback) Always(always ...bool) *DeferredCallback {
	d.always = true
	if len(always) > 0 {
		d.always = always[0]
	}
	return d
}

// GetName returns the name the callback was registered under.
func (d *DeferredCallback) GetName() string { return d.name }

// GetAlways reports whether the callback runs even when the request or the job
// failed.
func (d *DeferredCallback) GetAlways() bool { return d.always }

// GetCallback returns the func the callback will run.
func (d *DeferredCallback) GetCallback() func() { return d.callback }

// Invoke runs the callback. A nil receiver, or a callback with no func, does
// nothing.
func (d *DeferredCallback) Invoke() {
	if d == nil || d.callback == nil {
		return
	}
	d.callback()
}

// DeferredCallbackCollection holds every callback put off during one request,
// in the order they were deferred. It is safe for concurrent use.
type DeferredCallbackCollection struct {
	mu        sync.Mutex
	callbacks []*DeferredCallback
}

// NewDeferredCallbackCollection returns an empty collection.
func NewDeferredCallbackCollection() *DeferredCallbackCollection {
	return &DeferredCallbackCollection{}
}

// First returns the callback deferred earliest, or nil when the collection is
// empty.
func (c *DeferredCallbackCollection) First() *DeferredCallback {
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.callbacks) == 0 {
		return nil
	}
	return c.callbacks[0]
}

// Invoke runs every callback and empties the collection.
func (c *DeferredCallbackCollection) Invoke() { c.InvokeWhen(nil) }

// InvokeWhen empties the collection and runs the callbacks the test accepts. A
// nil test accepts all of them.
//
// Duplicates are dropped first, so of two callbacks deferred under one name
// only the later one runs. A callback that panics is recovered and dropped:
// one deferred callback must not take the others down.
func (c *DeferredCallbackCollection) InvokeWhen(when func(callback *DeferredCallback) bool) {
	if when == nil {
		when = func(*DeferredCallback) bool { return true }
	}

	c.mu.Lock()
	c.forgetDuplicates()
	pending := c.callbacks
	c.callbacks = nil
	c.mu.Unlock()

	for _, callback := range pending {
		if when(callback) {
			rescue(callback)
		}
	}
}

// rescue runs the callback and swallows whatever it panics with.
func rescue(callback *DeferredCallback) {
	defer func() { _ = recover() }()
	callback.Invoke()
}

// Forget drops every callback deferred under the given name.
func (c *DeferredCallbackCollection) Forget(name string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	kept := make([]*DeferredCallback, 0, len(c.callbacks))
	for _, callback := range c.callbacks {
		if callback.name != name {
			kept = append(kept, callback)
		}
	}
	c.callbacks = kept
}

// forgetDuplicates keeps, of two callbacks deferred under one name, only the
// later one, and leaves the survivors in the order they were deferred.
//
// The caller holds the lock.
func (c *DeferredCallbackCollection) forgetDuplicates() {
	seen := map[string]struct{}{}
	kept := make([]*DeferredCallback, 0, len(c.callbacks))
	for i := len(c.callbacks) - 1; i >= 0; i-- {
		callback := c.callbacks[i]
		if _, held := seen[callback.name]; held {
			continue
		}
		seen[callback.name] = struct{}{}
		kept = append(kept, callback)
	}
	for i, j := 0, len(kept)-1; i < j; i, j = i+1, j-1 {
		kept[i], kept[j] = kept[j], kept[i]
	}
	c.callbacks = kept
}

// OffsetSet writes the callback at the given offset. A negative offset, or one
// past the end, appends instead.
func (c *DeferredCallbackCollection) OffsetSet(offset int, callback *DeferredCallback) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if offset < 0 || offset >= len(c.callbacks) {
		c.callbacks = append(c.callbacks, callback)
		return
	}
	c.callbacks[offset] = callback
}

// OffsetExists reports whether the collection holds the given offset, once
// duplicates have been dropped.
func (c *DeferredCallbackCollection) OffsetExists(offset int) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.forgetDuplicates()
	return offset >= 0 && offset < len(c.callbacks)
}

// OffsetGet returns the callback at the given offset, or nil when the
// collection does not hold it.
func (c *DeferredCallbackCollection) OffsetGet(offset int) *DeferredCallback {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.forgetDuplicates()
	if offset < 0 || offset >= len(c.callbacks) {
		return nil
	}
	return c.callbacks[offset]
}

// OffsetUnset drops the callback at the given offset. An offset the collection
// does not hold is a no-op.
func (c *DeferredCallbackCollection) OffsetUnset(offset int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.forgetDuplicates()
	if offset < 0 || offset >= len(c.callbacks) {
		return
	}
	c.callbacks = append(c.callbacks[:offset], c.callbacks[offset+1:]...)
}

// Count returns how many callbacks the collection holds, once duplicates have
// been dropped.
func (c *DeferredCallbackCollection) Count() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.forgetDuplicates()
	return len(c.callbacks)
}
