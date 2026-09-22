// Package toon encodes JSON-shaped Go values as Token-Oriented Object
// Notation (TOON) 4.1, pinned to the specification repository's v4.1.1
// release.
//
// The package is output-only. Encode returns a string and accepts functional
// options for the delimiter and indentation. Marshal preserves the canonical
// comma and two-space profile as bytes. JSON remains the representation for
// request bodies, storage, and general-purpose APIs; TOON is an explicit
// adapter for data sent to language models.
package toon
