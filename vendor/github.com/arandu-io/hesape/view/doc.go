// Package view is the view layer: compiled markup, HTMX for interaction, one
// embedded script for ephemeral client state, Tailwind for style. It is a
// binary and it is never Node.
//
// A project that uses it still runs with a single command: no node_modules,
// no package.json, no lockfile of JavaScript, nothing installed beyond Go and
// the standalone binaries the tooling fetches. Having a build step is
// allowed; being Node is not.
//
// It lives in the collection and the views do not, and that split is
// deliberate: resources/views/ belongs to the project, because it is edited;
// the rendering machinery belongs here, because it is not. It used to be a
// repository of its own.
//
// The error page deliberately does not use this package. It has to render when
// the rest is broken, including when the view build failed, so it stays as
// html/template inline in hesape/exception.
//
// # What it holds
//
// Register and Registered are the registry a generated view writes itself into
// from init(), the same shape as a database/sql driver. Renderer draws one of
// them to a response, and RenderToString draws one for a message nobody is
// waiting on a socket for. Text, Yield, Include and CSRF are what generated
// code calls; everything decidable at build time already was, so what is left
// is small on purpose. Page is the chrome every screen embeds, carrying
// the messages and the typed input of a rejected attempt without any handler
// having to remember them. The assets -- HTMX, the client behaviours, the
// stylesheet -- are embedded and served content-addressed, because the CSP is
// script-src 'self'.
//
// There is no finder: a view is not looked up on disk at request time, it is
// compiled into the binary and registered by name, which is why a missing
// view is a boot-time panic or a named error rather than a blank page. And
// there is no service provider: Module below registers the one route this
// package owns and hands over the renderer, and nothing is resolved out of a
// container.
//
// The compiler that turns a .kyse.go file into the Go this package runs is
// not in here yet.
package view
