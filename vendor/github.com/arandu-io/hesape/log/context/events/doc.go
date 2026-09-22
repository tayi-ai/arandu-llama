// Package events is the two events the log context fires: ContextDehydrating,
// when the context is about to be written down for a queued job, and
// ContextHydrated, once it has been read back.
//
// Both carry the repository, and both carry it as the Repository interface
// declared here rather than as the concrete log/context.Repository -- the
// concrete one dispatches them, and naming it would close an import loop. The
// type doc says so on the interface itself.
package events
