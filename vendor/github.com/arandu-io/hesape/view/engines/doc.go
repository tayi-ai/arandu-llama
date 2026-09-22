// Package engines resolves a view path to the engine that renders it, and
// the compiled-view bookkeeping each engine needs.
//
// Resolver maps an engine name to a constructor, building and caching each
// engine on first use. Engine is the contract every engine satisfies: it
// turns a view path and data into rendered content. Compiler and File are
// the two concrete engines, and Base is the state they share.
package engines
