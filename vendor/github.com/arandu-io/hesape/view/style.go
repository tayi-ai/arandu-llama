package view

import (
	"crypto/sha256"
	"encoding/hex"
)

// StyleClass is the class name a scoped stylesheet is written under: "k-" and
// the first twelve hex characters of the SHA-256 of its text.
//
// # What it is for
//
// A component may carry a block of CSS of its own. The policy is
// style-src 'self' with no unsafe-inline, so that block cannot travel in the
// page: neither a style attribute nor a style element survives, and both fail
// by being dropped rather than by saying anything. What travels instead is a
// class, and the rules for it are compiled into the stylesheet the origin
// already serves.
//
// The class is the hash of the CSS, so the two sides need no table between
// them. The build reads the text out of the source and writes ".k-xxxxxxxxxxxx
// { ... }" into the stylesheet; the render hashes the same text and writes
// class="k-xxxxxxxxxxxx" onto the element. Neither knows about the other, and
// they agree because they are looking at the same bytes.
//
// # Why the bytes are hashed exactly as they came
//
// Nothing is trimmed, folded or reformatted first. A normalisation is a second
// thing that has to be implemented identically on both sides, and the side that
// got it wrong would emit a rule under one name while the page asked for
// another -- an element with a class nothing styles, which renders, passes
// every test that looks at the markup, and is simply unstyled.
//
// Two blocks that differ only in indentation therefore get two classes and two
// identical rules. That is a few duplicated bytes in a stylesheet, against an
// agreement that cannot drift.
//
// # Why it is exported
//
// One thing outside this repository has to produce the same string. The view
// build extracts the CSS at compile time and has to name the class the render
// will ask for. The CLI is a separate module and cannot import this one, so it
// computes the same three lines -- the same contract across a repository
// boundary that AssetHash carries, and the same reason this has a name and a
// test rather than being an expression inlined at its one call site.
func StyleClass(css string) string {
	sum := sha256.Sum256([]byte(css))
	return "k-" + hex.EncodeToString(sum[:])[:12]
}
