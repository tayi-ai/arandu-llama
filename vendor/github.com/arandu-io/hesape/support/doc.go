// Package support holds the small, dependency-free building blocks the rest of
// the framework is written on: time, environment, value holders, URLs and a
// handful of generic helpers.
//
// # Time
//
// [Now] is the one place the current instant comes from, and everything else
// here reads it: [Today], [Tomorrow], [Yesterday], [Parse] and the CreateFrom
// constructors. A test moves it with [Travel], [TravelTo], [FreezeTime] or
// [SetTestNow] and puts it back with [TravelBack]; [Use] and [UseCallable]
// replace the [Clock] outright.
//
// There is no date type of this package's own. time.Time already is the value,
// and a second one would be a second way to hold an instant.
//
// # Values
//
// [Fluent] is a bag of attributes read and written by dotted key, and
// [ValidatedInput] is the input that passed validation; both carry the typed
// readers String, Integer, Float, Boolean, Array and Date. [Optional] reads
// into a value that may be nil without checking at every step. [MessageBag]
// and [ViewErrorBag] collect messages keyed by field. [HtmlString] and [Js]
// carry markup that must not be escaped again.
//
// [Uri] parses and edits a URL: every writer hands back a new instance and
// leaves the old one alone, and [UriQueryString] reads the query as nested
// data.
//
// # Control flow
//
// [Retry] runs a callback again while it fails. [Sleep] builds a duration unit
// by unit. [Once] computes a value the first time a line is reached and gives
// it back after. [Timebox] refuses to return before its time is up, so the
// time a check took says nothing about its result. [Lottery] runs a callback
// on a fraction of the calls. [Defer] puts work off until the response has
// been sent. [Tap], [Transform] and [With] pass a value through a callback.
//
// # Process-wide state
//
// The clock, [Sleep], [Lottery] and [Once] each hold state a test replaces for
// the whole process, so a test that replaces one cannot run in parallel with a
// test that does not. Each has a matching call that puts the default back:
// [TravelBack] and [UseDefault], [Fake], [DetermineResultNormally], [Flush].
package support
