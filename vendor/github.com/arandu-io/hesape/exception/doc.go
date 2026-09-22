// Package exception is where a failed request stops.
//
// # The two ways a request fails, and the one place they meet
//
// A handler either returns an error or panics. Both arrive here: Recover turns
// a panic into a value the Handler answers, and the routing layer hands the
// Handler whatever a controller action returned. One Handler, one decision about
// what the person in front of the browser sees.
//
// Report and Render are two calls and the caller makes both. Render used to
// report on its way in as well, so an application written that way wrote every
// failure to the log twice. Recover is the exception and says so where it does
// it: a panic is news whatever it classifies as, so it logs what it caught
// itself.
//
// # Abort is a returned error, not a call that never comes back
//
// There is no throw here and no global helper, so the equivalent is a value:
//
//	if invoice == nil {
//		return exception.Abort(http.StatusNotFound, "no invoice with that number")
//	}
//
// AbortIf and AbortUnless return nil when the condition does not call for a
// failure, which is what lets the caller write one line instead of the same if
// statement twice.
//
// Abort builds an *HTTPError, StatusOf reads the status back out of an error
// chain, and classify is the closed table that says what the collection's own
// sentinels mean -- auth.ErrForbidden is 403, an expired CSRF token is 419.
// Before this existed, every error leaving a handler became 500, including the
// ones that had already said exactly what they were.
//
// The table is closed on purpose. An application that wants a status says so
// with Abort; it does not get a second mechanism for the same sentence. Map is
// not that second mechanism: it turns somebody else's error -- a driver's, a
// library's -- into one of these, and the answer still comes from the one
// table.
//
// # The pages
//
// Two kinds, and they never overlap. A status the framework recognises gets the
// status page -- 401, 403, 404, 405, 419, 429, 500, 503, and the standard text
// for anything else. An error nobody claimed gets the debug page in development,
// which is Ignition's idea with the Collector behind it: the stack with source
// snippets, the queries with their timing, the dumps, the events, and the
// hints -- the part that names the probable cause instead of only showing the
// data.
//
// The two are the two Displayers: PlainDisplayer and DebugDisplayer.
//
// An application overrides a status page by providing a view named errors/404,
// errors/403 and so on, and wiring it through Config.Views. Nothing is required:
// the built-in pages answer until somebody wants their own.
//
// # The JSON answer
//
// A client that asked for JSON gets a [Problem]: the problem details document
// of RFC 9457, served as [ProblemContentType]. It is one shape, and
// [WriteProblem] is the one function that writes it, so a refusal from a
// middleware and a failure from a handler read the same to whoever parses them.
//
// htmx is not that client. It sends X-Requested-With and swaps HTML, so
// wantsJSON excludes it: a problem document swapped into a div is a JSON
// document on the page.
//
// # What a test writes instead of a global fake
//
// There is no registry to swap a handler in, and a package-level handler a test
// could swap would be shared mutable state that two tests calling t.Parallel
// would fight over.
//
// The [Handler] a test holds is one it built, and the recording is a callback
// on it:
//
//	var reported []error
//	h := exception.NewHandler(exception.Config{})
//	h.Reportable(func(err error) { reported = append(reported, err) }).Stop()
//
//	placeOrder(ctx, g, h)
//
//	if len(reported) != 1 {
//		t.Fatalf("reported %d failures, want 1", len(reported))
//	}
//
// [ReportableHandler.Stop] is what makes it a fake rather than a bystander:
// reporting ends at the callback, so nothing reaches the log and the test
// output stays the failures the test itself printed.
//
// # Absolute rule
//
// Nothing that reveals the inside of the process -- the debug page, a stack, an
// error string that is not an *HTTPError message -- may be reachable when
// Config.Dev is false.
package exception
