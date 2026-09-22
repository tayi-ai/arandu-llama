// Package middleware wraps the handling of a job.
//
// A middleware sits between the worker and the handler and can decide the work
// should not happen now -- another worker holds the lock, the budget for this
// minute is spent, the batch was cancelled -- by putting the job back on its
// queue instead of running it.
//
// [Middleware.Handle] gets the context, the job and the rest of the chain, and
// what it does with the job is release it, delete it, fail it, or hand it on.
//
// [RateLimited] and [ThrottlesExceptions] count in a cache.Store, so which
// store they count in is wiring rather than a second type: the RESP-backed
// version of RateLimited is RateLimited over a RESP-backed store.
//
// # Every key is scoped by tenant
//
// The lock a job takes and the counter a job spends are named after the tenant
// the job belongs to. Without it, one customer's slow import would rate limit
// every other customer's.
package middleware
