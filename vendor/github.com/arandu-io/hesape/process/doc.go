// Package process runs external programs.
//
//	factory := process.NewFactory()
//	result, err := factory.Run(ctx, []string{"go", "build", "./..."}, nil)
//	if result.Failed() {
//		log.Print(result.ErrorOutput())
//	}
//
// # A failed exit is a result, not an error
//
// This is the one thing to know before writing anything else with it. A program
// that exits non-zero comes back with a nil error and Failed reporting it:
//
//	result, err := factory.Run(ctx, []string{"git", "diff", "--quiet"}, nil)
//	// err is nil. A difference is exit code 1, and it is not a failure.
//	if result.ExitCode() == 1 {
//		// there are changes
//	}
//
// The error is for the program that never ran, ran out of time, or was stray.
// Throw is what turns a failed exit into one, when the caller wants that:
//
//	result, err := factory.Run(ctx, []string{"go", "test", "./..."}, nil)
//	if _, err = result.Throw(nil); err != nil {
//		return err // a *ProcessFailedException, carrying both streams
//	}
//
// # What os/exec does not do
//
//   - An error that repeats what the program said. os/exec reports a failed
//     command as "exit status 1" and throws the explanation away, so a build
//     that stopped because a module could not be fetched answers with the exit
//     status of a program the person did not know was running.
//   - A deadline that also covers the program that never finishes and never
//     prints. A context deadline bounds the total; IdleTimeout bounds the
//     silence, which is the shape a hung download or a stalled compiler has.
//   - A fake. Factory.Fake answers a command pattern with a canned result, so a
//     test that shells out stops shelling out, and PreventStrayProcesses turns
//     "somebody forgot to fake git" from a test that quietly runs git on CI
//     into a test that fails naming the command.
//
// # No shell, and no string form to add
//
// A command is a program and its arguments, and nothing here hands a string to
// sh. Accepting a string would mean exec.Command("sh", "-c", line) with an
// interpolated line, which is command injection with a familiar signature.
//
// # Several at once, and one after another
//
// Pool, InvokedProcessPool, ProcessPoolResults and Pipe run a set of programs
// together, and three methods of the factory reach them:
//
//	results, err := factory.Concurrently(ctx, func(pool *process.Pool) {
//		pool.As("build").Command("go", "build", "./...")
//		pool.Command("go", "vet", "./...")
//	}, nil)
//
//	result, err := factory.Pipe(ctx, func(pipe *process.Pipe) {
//		pipe.Command("cat", "notes.txt")
//		pipe.Command("grep", "-i", "todo")
//	}, nil)
//
// Concurrently starts them all and waits; Factory.Pool is the same set started
// and not waited for, which is what Signal, Running and Wait on the
// InvokedProcessPool are for. A pipe feeds each process the output of the one
// before it and stops at the first that exits non-zero, whose result is what
// comes back.
//
// The key is the string a process is added under -- As names one, and a process
// added without a name takes the next integer key, counted only among the
// unnamed. It is what the pool's output handler is given alongside the chunk,
// so output from four programs at once can be told apart.
//
// # The output is held in memory, unbounded
//
// A program that prints without stopping is held in full until it exits. The
// caller that expects one bounds it, by streaming through the output handler
// rather than reading the result.
package process
