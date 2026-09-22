// Package bus groups jobs: a batch that knows when all of it is done, and a
// chain that runs one job after another.
//
// It is the shape of every CSV import and every bulk export. Ten thousand rows
// are ten thousand jobs, because one job that loops over ten thousand rows is a
// job that dies on row nine thousand and starts again from one. Splitting them
// is easy; knowing when the last one finished is not, and that is what a batch
// is: a counter that survives the workers, plus the callbacks fired the moment
// it reaches zero.
//
// A batch is dispatched, tracked and finished:
//
//	b, err := bus.NewBatch("import-invoices").
//	    Add("invoice.import", row1).
//	    Add("invoice.import", row2).
//	    Then("import.done", report).
//	    Catch("import.failed", report).
//	    OnQueue("imports").
//	    Dispatch(ctx, g, repository, queue)
//
// The worker side is two calls, whatever the job:
//
//	m, err := bus.Batched(j.Payload, &row)
//	...
//	err = bus.Handled(ctx, g, repository, queue, m, doTheWork(ctx, g, row))
//
// Handled is Batch.RecordSuccessfulJob or Batch.RecordFailedJob plus
// Batchable.DispatchNextJobInChain or Batchable.InvokeChainCatchCallbacks, in
// one call: it moves the counter, fires Then, Catch, Failure, Progress and
// Finally exactly once each, and pushes the next link of a chain. A handler
// that never calls it leaves its batch pending forever, which is the one
// failure mode worth remembering.
//
// # The counters
//
// PendingJobs falls by one for every job that *succeeded*; FailedJobs rises for
// every job that failed and leaves PendingJobs alone. So "every job succeeded"
// is PendingJobs reaching zero -- which is what finishes a batch and fires Then
// -- and "every job has reported" is PendingJobs minus FailedJobs reaching
// zero, which is UpdatedBatchJobCounts.AllJobsHaveRanExactlyOnce and what fires
// Finally.
//
// # No closures
//
// Then, Catch and Finally name a job; they do not take a function. A callback
// runs in a different process from the one that dispatched the batch -- often a
// different build -- and Go cannot serialize a closure across that gap. A named
// job is the same thing with the indirection made visible.
//
// It is also why each moment names one job rather than a list: a list of
// closures is cheap to register and a list of pushes is not, and one job that
// fans out is the same thing with a name somebody can grep.
//
// # No stored chain
//
// A chain needs no table. The remaining links travel inside the payload of the
// link currently running, so a chain is exactly as durable as the queue is and
// nothing has to be cleaned up when one is abandoned. A batch does need a table,
// because a counter shared by N workers cannot live in any one of their
// payloads.
//
// [Chain] is a chain being described, and [Dispatcher.DispatchChain] builds one
// and dispatches it in the same expression.
//
// # One store
//
// Batches live in the application's own database, through
// [DatabaseBatchRepository] over the table [Migrations] creates. That table is
// job_batches, named nowhere else, because a table the framework reads and
// writes is not one an application renames -- and it is created by `aru
// migrate` as a pipeline step, never by the process that uses it.
//
// There is one store and there will not be a second: a store with a different
// consistency model is a second set of rules for when a callback fires.
// [Memory] is the implementation that needs no table, for a test.
//
// # Testing
//
// There is no global dispatcher to swap: a package-level one would be shared
// mutable state that two tests calling t.Parallel would fight over. A test
// builds the dispatcher it wants, in one line, and the handler it maps is the
// recording:
//
//	var ran []string
//	d := bus.NewDispatcher(nil, bus.NewMemory()).Map(map[string]bus.Handler{
//		"invoice.email": func(context.Context, auth.Grant, []byte) error {
//			ran = append(ran, "invoice.email")
//			return nil
//		},
//	})
//
//	importInvoices(ctx, g, d)
//
//	if len(ran) != 1 {
//		t.Fatalf("ran %d jobs, want 1", len(ran))
//	}
//
// A nil queue runs every job in this process, which is what a test usually
// wants: assert what the work did, not that it was scheduled. To assert the
// other half -- that a job was queued and did not run -- hand [NewDispatcher] a
// [Queue] whose one Push method appends to a slice. That interface has exactly
// one method for this reason, and [Memory] is the batch repository that needs
// no table, so a batch is exercised end to end with nothing installed and
// nothing shared between tests.
package bus
