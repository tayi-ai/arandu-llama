package model

import "slices"

// Event is one of the model events HasEvents fires.
//
// A callback is registered on the model that will fire it, and NewInstance
// carries the registrations onto every instance made from that model.
type Event string

// The events Model fires, in the order it fires them.
//
// There is no booting event: a Go value has no class initialisation to hook.
const (
	Retrieved     Event = "retrieved"
	Creating      Event = "creating"
	Created       Event = "created"
	Updating      Event = "updating"
	Updated       Event = "updated"
	Saving        Event = "saving"
	Saved         Event = "saved"
	Deleting      Event = "deleting"
	Deleted       Event = "deleted"
	Trashed       Event = "trashed"
	Restoring     Event = "restoring"
	Restored      Event = "restored"
	ForceDeleting Event = "forceDeleting"
	ForceDeleted  Event = "forceDeleted"
	Replicating   Event = "replicating"
)

// RegisterModelEvent registers callback to run when event fires.
//
// A callback that returns an error stops the operation, and the error says
// why. A saving callback that fails means nothing was written.
func (m *Model[T]) RegisterModelEvent(event Event, callback func(*Model[T]) error) *Model[T] {
	if m.events == nil {
		m.events = map[Event][]func(*Model[T]) error{}
	}
	m.events[event] = append(m.events[event], callback)
	return m
}

// WithoutEvents runs callback with model events muted on this model, and
// restores the previous setting however it ends.
//
// It mutes the model it is called on, and the instances it makes while
// muted, because they are made from it.
func (m *Model[T]) WithoutEvents(callback func() error) error {
	previous := m.muted
	m.muted = true
	defer func() { m.muted = previous }()
	return callback()
}

// fireModelEvent runs every callback registered for event, in registration
// order, stopping at the first one that returns an error.
func (m *Model[T]) fireModelEvent(event Event) error {
	if m.muted {
		return nil
	}
	for _, callback := range m.events[event] {
		if err := callback(m); err != nil {
			return err
		}
	}
	return nil
}

func cloneEvents[T any](in map[Event][]func(*Model[T]) error) map[Event][]func(*Model[T]) error {
	if in == nil {
		return nil
	}
	out := make(map[Event][]func(*Model[T]) error, len(in))
	for event, callbacks := range in {
		out[event] = slices.Clone(callbacks)
	}
	return out
}
