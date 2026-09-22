// Package events is everything the queue announces about itself.
//
// It is one struct per moment worth hearing about: a job going onto a queue,
// coming off one, running, failing, being released or being parked, and a
// worker starting and stopping.
//
// They are values, and nothing here dispatches: queue.Dispatcher is the two
// methods the worker needs, and what is on the other side of it is the
// application's business. A worker with no dispatcher builds no events at all.
//
// [WorkerStopReason] is here rather than in the queue package because
// [WorkerStopping] carries it and the queue package is what dispatches these --
// the queue package re-exports it.
package events
