package auth

import "context"

// subjectKey is the context key the subject travels under. It is an unexported
// empty struct type, so nothing outside this package can write the value a
// policy will later be asked about.
type subjectKey struct{}

// WithSubject returns a copy of ctx carrying the subject.
//
// The middleware that loads the session calls it once, at the edge, and
// everything downstream reads it back with SubjectFrom. It is how a handler
// several calls deep authorizes without every function in between taking a
// Subject parameter it does nothing with.
//
// It is not a place to put a subject together by hand. Whoever calls this is
// stating that a session was loaded and believed, and `aru doctor` reads the
// same rule here that it reads for SystemGrant: a subject assembled from
// anything that arrived with the request is the tenant-from-request finding.
func WithSubject(ctx context.Context, s Subject) context.Context {
	return context.WithValue(ctx, subjectKey{}, s)
}

// SubjectFrom returns the subject carried by ctx.
//
// The second result is false when nothing put one there, and that is a
// different fact from an anonymous reader: a request with no session at all has
// no subject, while a public page has the Subject that Guest built. Callers
// that conflate the two turn a missing session into a visitor, which is the
// mistake Authorize refuses the zero Subject to prevent.
func SubjectFrom(ctx context.Context) (Subject, bool) {
	s, ok := ctx.Value(subjectKey{}).(Subject)
	return s, ok
}

// Check reports whether ctx carries somebody who signed in.
//
// It answers only that question: a declared guest is not signed in, and neither
// is a request that never loaded a session.
// It decides nothing -- a view uses it to choose between the sign-in link and
// the account menu, and a handler that needs a decision calls Authorize or
// Allows, which ask a policy.
func Check(ctx context.Context) bool {
	s, ok := SubjectFrom(ctx)
	return ok && s.ID != "" && !s.guest
}
