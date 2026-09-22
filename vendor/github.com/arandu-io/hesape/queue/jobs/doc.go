// Package jobs is the job itself: what Push takes, and what the worker holds
// while the work runs.
//
// There is one [Job] and one [Driver] interface, rather than a job type per
// store. The three things a running job does about itself -- release, delete
// and fail -- differ only in which store they write to, and a type per store
// would be a second way to say the same thing. The queue that popped the job
// supplies the Driver, and [DatabaseJob] and [SyncJob] are aliases of [Job].
//
// The split between this package and [github.com/arandu-io/hesape/queue] is
// the job on one side and the thing that holds jobs on the other. It is what
// keeps the import graph one-directional -- queue imports jobs, jobs imports
// nothing of queue -- so a driver in its own module can build a Job without
// depending on the drivers that ship in the collection.
//
// # A field is an accessor
//
// The values a job carries are fields, not accessors: they are columns of the
// record, and nothing has to be decoded to read them. The methods that remain
// are the ones that compute something: [Job.Backoff] picks the entry for this
// attempt, [Job.ResolveName] prefers the display name,
// [Job.IsDeletedOrReleased] reads two flags.
package jobs
