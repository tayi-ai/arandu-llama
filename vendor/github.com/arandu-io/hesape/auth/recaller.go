package auth

import "strings"

// Recaller is the "remember me" cookie, read.
//
// The cookie is three fields joined by a pipe -- the user id, the remember
// token, and an HMAC of the password hash -- and this type is the only thing
// that takes it apart. It validates nothing about the user: a valid recaller is
// a well-formed one, and whether it names anybody is the provider's answer to
// RetrieveByToken.
//
// The value is the cookie's string, decoded no further.
type Recaller struct {
	// recaller is the cookie value.
	recaller string
}

// NewRecaller returns a recaller over the cookie value.
func NewRecaller(recaller string) *Recaller {
	return &Recaller{recaller: recaller}
}

// ID is the user id, the first segment.
//
// A segment that is not there is "". Ask [Recaller.Valid] before believing any
// of the three, which is what SessionGuard does.
func (r *Recaller) ID() string {
	return r.segment(strings.SplitN(r.recaller, "|", 3), 0)
}

// Token is the remember token, the second segment.
func (r *Recaller) Token() string {
	return r.segment(strings.SplitN(r.recaller, "|", 3), 1)
}

// Hash is the HMAC of the password hash, the third segment.
//
// The split here allows a fourth field where the other two methods allow a
// third, so a cookie carrying a stray pipe gives the third field alone rather
// than the rest of the string.
func (r *Recaller) Hash() string {
	return r.segment(strings.SplitN(r.recaller, "|", 4), 2)
}

// Valid reports that the cookie is a pipe-joined string with all of its
// segments.
func (r *Recaller) Valid() bool {
	return r.properString() && r.hasAllSegments()
}

// Segments is every segment of the cookie, however many there are.
func (r *Recaller) Segments() []string {
	return strings.Split(r.recaller, "|")
}

// properString reports that the value is pipe-joined at all.
func (r *Recaller) properString() bool {
	return strings.Contains(r.recaller, "|")
}

// hasAllSegments reports three segments, with an id and a token that are not
// blank. The hash is not checked -- a cookie from before the hash existed still
// names its user.
func (r *Recaller) hasAllSegments() bool {
	segments := r.Segments()

	return len(segments) >= 3 &&
		strings.TrimSpace(segments[0]) != "" &&
		strings.TrimSpace(segments[1]) != ""
}

// segment reads one field of an already split recaller, and answers "" past the
// end.
func (r *Recaller) segment(segments []string, index int) string {
	if index >= len(segments) {
		return ""
	}
	return segments[index]
}
