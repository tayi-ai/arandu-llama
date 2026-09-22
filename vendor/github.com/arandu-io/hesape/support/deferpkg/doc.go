// Package deferpkg holds work put off until after the response has been sent.
//
// [DeferredCallback] is one such piece of work, carrying a name so it can be
// called off before it runs. [DeferredCallbackCollection] holds everything
// deferred during one request and runs it in order, keeping only the last
// callback registered under any one name and swallowing a panic from one so it
// does not take the others down.
//
// Callbacks are registered through support.Defer, which reaches the collection
// support.DeferredCallbackCollection returns.
//
// The package is named deferpkg because defer is a Go keyword and cannot be a
// package name.
package deferpkg
