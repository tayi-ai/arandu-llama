package events

// Repository is the slice of the log context repository that these two events
// carry.
//
// It is an interface rather than the concrete type for one reason: log/context
// dispatches these events, so an event that named *context.Repository would
// import the package that imports it. A listener that wants the whole repository
// asserts back to it, which is what Repository.Dehydrating and
// Repository.Hydrated do for you.
type Repository interface {
	// All returns the visible half of the context.
	All() map[string]any

	// AllHidden returns the hidden half of the context.
	AllHidden() map[string]any
}

// ContextDehydrating is dispatched while the context is being written down to
// travel with a queued job, before the values are serialised, so a listener may
// still add to it or take something out.
//
// The repository it carries is a copy taken for the dehydration, not the live
// one.
type ContextDehydrating struct {
	// Context is the context instance.
	Context Repository
}

// ContextHydrated is dispatched after a dehydrated context was read back, on the
// other side of the queue, with the values already restored.
type ContextHydrated struct {
	// Context is the context instance.
	Context Repository
}
