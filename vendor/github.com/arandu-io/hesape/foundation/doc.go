// Package foundation boots the application.
//
// It is the single place where an application is composed. One difference
// matters: what this package builds is built ONCE, at process start, not per
// request, so nothing here may assume request scope.
//
// # What lives here
//
// [Module] and its optional interfaces -- Bootable, Background, Closable,
// Diagnostic, Health, Schedulable, Migratable -- are the whole
// vocabulary of composition. A module declares; the kernel collects; nothing
// runs until something asks. That shape is why [Task] and [Migration] are
// declared here rather than in the packages that execute them: the scheduler
// and the migration runner are consumers, and a module must be able to hand its
// declarations straight to either one.
//
// It also owns what the framework mounts for itself. Everything under
// internalPrefix -- the health probe, the debug console, the development
// reload -- answers to the framework rather than to the application, and
// exceptInternal is the one place that boundary is enforced. [Observe] installs
// the request id, the request logger and, in development or under an authorized
// tracing header, the Collector behind that console. The live reload in
// reload.go follows a restart in development and costs nothing anywhere else.
//
// The boot sequence, the view renderer and the console command belong to the
// layer above and are not declared here. This package keeps only the vocabulary
// a module needs in order to declare itself, so that the adapter packages that
// implement [Module] can compile against it without depending on that layer.
package foundation
