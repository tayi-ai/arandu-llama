package auth

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"time"
)

// ErrForbidden is the only authorization error. Handlers translate it to 403.
var ErrForbidden = errors.New("arandu: action not authorized")

// Subject is whoever is acting. It comes from the session, never from the
// request body.
type Subject struct {
	ID     string
	Tenant string

	// Roles are what this subject is: "admin", "member", "owner". A policy
	// asks about one with HasRole.
	//
	// Names an application chose, and this package never validates them --
	// which is exactly why nothing else may be put here. A permission written
	// into Roles is a permission HasRole answers about, and every policy that
	// asks HasRole("admin") starts answering false the moment something fills
	// this list with anything else. Both sides are []string, so the compiler
	// says nothing, and what reaches support is "my access disappeared".
	Roles []string

	// Actions are what this subject may do: "user.view", "invoice.create". A
	// policy asks about one with Can.
	//
	// It is a separate list from Roles because the two are separate questions.
	// An application that decides by role reads Roles and never fills this; one
	// whose permissions are administered -- by arandu-permission or by anything
	// else -- fills this and leaves Roles to mean what it always meant. An
	// application that does both is answered correctly by both.
	//
	// Typed as Action rather than string, so a role slug put here does not
	// compile. That is the whole of the protection, and it is enough: the
	// mistake this exists to prevent was two lists of strings that looked
	// alike.
	Actions []Action

	// Verified says whether the address behind this account was confirmed.
	//
	// It is on the subject and not read from the database per request, because
	// the question is asked by policies -- on every write, sometimes twice --
	// and a database round trip inside an authorization check is a round trip
	// nobody can see from the call site.
	//
	// The cost is that it is as old as the session: somebody who verifies while
	// signed in stays unverified until they sign in again. That is the right
	// trade for the direction it fails in -- an account is created unverified,
	// so a stale session can only be MORE restrictive than the truth, never
	// less.
	Verified bool

	// Remembered says whether this session was started with the remember-me box
	// ticked, and therefore lives for RememberLifetime instead of the store's
	// configured ttl.
	//
	// It is on the record and not recomputed, because the session store rewrites
	// the record when the password is confirmed and has to write back the same
	// lifetime it started with. Without it, confirming a password on a remembered
	// session silently cut it down to the plain ttl -- somebody who ticked the box
	// and then confirmed a payment was signed out that evening.
	//
	// Only the session store's Start and Regenerate set it, from the Remember
	// option, and they overwrite whatever the caller put here: a field set by hand
	// on the way in would be a second way to ask for a longer session, and there
	// is only ever one.
	//
	// A policy may also read it: a session nobody has authenticated for a month
	// is the right moment to ask for the password again before a destructive
	// action.
	Remembered bool

	// PasswordConfirmedAt is when the subject last typed their password again on
	// an already open session.
	//
	// It exists so a sensitive action can demand the password once and then leave
	// the person alone for a while, instead of on every click. Ask through the
	// session store's PasswordConfirmedWithin, never by comparing this field: the
	// zero value has to mean unconfirmed, and a comparison written at the call
	// site is where that gets forgotten.
	//
	// It is carried on the subject for the same reason Verified is, and it fails
	// in the same direction: a session written by an older binary has no stamp, so
	// it reads as never confirmed and the person is asked for their password. The
	// opposite default -- treating an absent stamp as recent -- would let every
	// session that survived a deploy walk past the confirmation screen.
	PasswordConfirmedAt time.Time

	// guest marks a subject that is deliberately anonymous. It is unexported and
	// only Guest sets it, which is the whole point: a Subject nobody filled in
	// is not a guest, it is a session somebody forgot to load, and Authorize
	// tells those two apart.
	guest bool
}

// Guest is a reader with no session, declared on purpose.
//
// It exists because a public page is a real requirement and the alternative was
// worse. Authorize refuses an empty subject before it consults a policy -- which
// is right, because an empty subject is almost always a forgotten session load
// -- and that left no way at all to say "anybody may read a published post".
// The only path was SystemGrant, which skips the policy entirely: a blog served
// with the same instrument a scheduled job uses.
//
// So the refusal stays and the exception is explicit. A zero Subject is still
// refused. This one reaches the policy, and the POLICY decides:
//
//	func (PostPolicy) Can(ctx context.Context, s auth.Subject, a auth.Action, p models.Post) error {
//		if s.IsGuest() {
//			if a == PostView && !p.PublishedAt.IsZero() {
//				return nil
//			}
//			return fmt.Errorf("%s is not public", a)
//		}
//		…
//	}
//
// Nothing is loosened by this. Authorization still happens in one place, the
// Grant is still the only way to a repository, and a policy that says nothing
// about guests denies them -- which is what every generated policy does, so the
// default is closed.
//
// The tenant is required and is the application's, from configuration. A
// visitor cannot choose whose rows they read, and nothing about that is
// suspended because nobody signed in.
func Guest(tenant string) Subject {
	return Subject{Tenant: tenant, guest: true}
}

// IsGuest reports whether this subject is a declared anonymous reader.
//
// A policy that never asks denies them, because it will fall through to its
// final refusal -- HasRole answers false for a guest, and there is no id to
// compare an owner against.
func (s Subject) IsGuest() bool { return s.guest }

// HasRole reports whether the subject carries the given role.
//
// A role is what somebody is. For what they may do, ask Can: the two lists are
// separate, and a permission is never answered here.
func (s Subject) HasRole(r string) bool {
	for _, have := range s.Roles {
		if have == r {
			return true
		}
	}
	return false
}

// Can reports whether the subject carries the given action.
//
// It answers about Actions and nothing else. A subject whose permissions are
// administered elsewhere -- by a module that resolves group membership into a
// flat list, or by an application that computes one itself -- carries them
// there, and a policy that asks this question gets the answer whether or not
// that subject also has roles.
//
// A subject with no actions answers false to everything, which is what an
// application that decides purely by role carries and is the right answer for
// it: such an application asks HasRole, and this is not its question.
func (s Subject) Can(a Action) bool {
	for _, have := range s.Actions {
		if have == a {
			return true
		}
	}
	return false
}

// Action is the intended operation, in "module.verb" form.
type Action string

// Policy decides. One policy per entity, always in the module's
// <entity>.policy.go file -- the CLI generates the skeleton and `aru doctor`
// complains when a repository exists without a matching policy.
type Policy[T any] interface {
	// Can decides whether subject may perform action on resource. resource may
	// be the zero value for collection actions (e.g. "customer.list").
	Can(ctx context.Context, s Subject, a Action, resource T) error
}

// Grant is the proof that an authorization decision happened.
//
// THIS IS THE CENTRAL PIECE OF THE FRAMEWORK. Grant has only unexported fields,
// so it cannot be built by writing a struct literal: every repository signature
// requires one, and reaching the database without a Grant does not compile.
//
// What the compiler does NOT decide is which Grant. Authorize is the mandatory
// path and the only one where a Policy answered; SystemGrant is the named
// escape hatch, and GrantFor in the queue package wraps it for jobs. All three
// are exported, so a handler can construct a Grant nobody authorized. What
// stops that is
// `aru doctor` -- a lint, not the type system -- with system-grant-outside-scope,
// system-grant-without-tenant and tenant-from-request.
//
// This comment used to say "no public constructor other than Authorize", which
// was never true and read as a compile-time guarantee for something a lint
// enforces. It is the difference between the promise and the mechanism, and
// stating it wrong here is worse than anywhere else: this is the doc a reader
// checks the thesis against.
//
// The alternative shape -- authorization as a call the handler remembers to make
// -- is authorization that gets forgotten, and nothing warns you.
type Grant struct {
	subject Subject
	action  Action
	valid   bool
	// reason is why an invalid Grant is invalid, when something knew.
	//
	// The zero Grant carries none, and its message is the right one for it: a
	// caller who never authorized anything is told to. A Grant refused by
	// SystemGrant is a different mistake, and Check says which.
	reason string
}

// Authorize runs the policy and, when allowed, issues the Grant.
func Authorize[T any](ctx context.Context, p Policy[T], s Subject, a Action, resource T) (Grant, error) {
	// An empty subject is refused before the policy is asked, because it is
	// almost always a session that was not loaded -- and a policy asked about
	// nobody answers about nobody.
	//
	// A Guest is the exception, and it is an exception the caller declared: it
	// carries a marker only Guest sets. The policy decides about it like any
	// other subject, and a policy that says nothing about guests refuses them.
	if s.ID == "" && !s.guest {
		return Grant{}, fmt.Errorf("%w: anonymous subject on %s", ErrForbidden, a)
	}
	if err := p.Can(ctx, s, a, resource); err != nil {
		who := s.ID
		if s.guest {
			who = "a guest"
		}
		return Grant{}, fmt.Errorf("%w: %s denied for subject %s: %v", ErrForbidden, a, who, err)
	}
	return Grant{subject: s, action: a, valid: true}, nil
}

// Allows asks the same question as Authorize and answers yes or no.
//
// It exists for the view: a template that draws a delete button has to know
// whether the button would work, and it has no use for the Grant or for the
// sentence explaining the refusal.
//
// It is not a second way to authorize, because it cannot authorize anything --
// it throws the Grant away, and without a Grant no repository is reachable. A
// handler that acts on the answer of Allows still has to call Authorize to do
// the work, and that call is the one that decides.
func Allows[T any](ctx context.Context, p Policy[T], s Subject, a Action, resource T) bool {
	_, err := Authorize(ctx, p, s, a, resource)
	return err == nil
}

// Check is the guard every repository operation must call.
//
// It fails on every Grant that was never issued -- the zero value, and the one
// SystemGrant returns when it refuses -- and when the grant was issued for a
// different action, which catches copy-paste between repository methods.
func (g Grant) Check(expected Action) error {
	if !g.valid {
		// A refused SystemGrant says why it was refused. It used to fall through
		// to the message below, which tells the caller to call Authorize -- and
		// in a job or a scheduled task there is no request to authorize from, so
		// the advice is impossible to follow and points away from the real
		// cause, which is the tenant. Found by audit.
		if g.reason != "" {
			return fmt.Errorf("%w: %s", ErrForbidden, g.reason)
		}
		return fmt.Errorf("%w: missing grant for %s (call auth.Authorize first)", ErrForbidden, expected)
	}
	if g.action != expected {
		return fmt.Errorf("%w: grant issued for %s, used on %s", ErrForbidden, g.action, expected)
	}
	return nil
}

// Subject exposes who was authorized -- used to scope SQL by tenant.
func (g Grant) Subject() Subject { return g.subject }

// Action exposes what was authorized.
func (g Grant) Action() Action { return g.action }

// Tenant returns the tenant from the Grant. Every multi-tenant statement must
// take this value, never a tenant that came in with the request.
//
// It lives here and not next to the SQL because it is one field read off a
// Grant, and the cache, the filesystem and the scheduler all need it: with it
// in the database package, knowing which customer a cache key belongs to meant
// importing the database.
func Tenant(g Grant) string { return g.subject.Tenant }

// tenantName is what a tenant identifier may contain.
//
// Closed on purpose. A tenant is concatenated into a storage path, a cache key,
// a scheduler lock name and a queue key -- so a tenant carrying "/" or ":"
// collides with another tenant's namespace, and one carrying ".." leaves it.
//
// Found by audit: tenant "acme/reports" storing key "q1.pdf" and tenant "acme"
// storing "reports/q1.pdf" resolved to the same object, each holding a
// perfectly valid Grant of its own. No Policy was violated -- the path is built
// after the Policy runs.
//
// One case only, and it is the same class. Measured: tenant "Acme" and tenant
// "acme" are two identifiers here and one directory on a filesystem that folds
// case, which is the default on macOS and on Windows -- so Exists for one of
// them answered true about the other one's file. The separator arrives through
// the key and the case arrives through the filesystem, and both end at two
// tenants sharing a namespace nobody meant to share. Refused rather than folded
// to lowercase, because a tenant that had to be rewritten to be safe is a
// tenant whoever called did not mean.
//
// Lowercase UUIDs, slugs and numeric ids all pass. Anything that could be read
// as a separator does not, and neither does anything a filesystem could read as
// another spelling of the same name.
var tenantName = regexp.MustCompile(`^[a-z0-9][a-z0-9_-]{0,63}$`)

// ValidTenant reports whether a tenant identifier is safe to use as a namespace.
//
// Exported because the adapters build keys from it and a second, slightly
// different definition in each of them is how one of them ends up permissive.
func ValidTenant(tenant string) bool { return tenantName.MatchString(tenant) }

// SystemGrant exists for jobs that run outside a request, and for the login
// path, where there is no subject yet.
//
// The tenant is required and cannot be empty. A system grant without a tenant
// would read across every customer of the system, which in a SaaS is the worst
// bug there is -- so it is not expressible: an empty tenant yields the zero
// Grant, and the zero Grant fails Check.
//
// Every call site is auditable, and `aru doctor` reports the ones outside a
// seeder, a job or a command. `--strict` does not list them -- it turns that
// warning into a failure, which is what CI runs.
func SystemGrant(a Action, tenant string) Grant {
	// An invalid tenant produces the zero Grant, which passes no Check -- the
	// same answer an empty one has always produced, for the same reason: a
	// tenant that cannot be trusted as a namespace cannot scope anything.
	if !ValidTenant(tenant) {
		if tenant == "" {
			return Grant{reason: fmt.Sprintf(
				"a system grant for %s was asked for with no tenant. Nothing can be scoped without one, and a query that is not scoped reads every customer. The tenant comes from the job, the task or the row that caused this work", a)}
		}
		return Grant{reason: fmt.Sprintf(
			"a system grant for %s was asked for with the tenant %q, which cannot be one: a tenant is concatenated into a storage path, a cache key and a lock name, so it is limited to lowercase letters, digits, - and _, up to 64 characters", a, tenant)}
	}
	return Grant{
		subject: Subject{ID: "system", Tenant: tenant, Roles: []string{"system"}},
		action:  a,
		valid:   true,
	}
}
