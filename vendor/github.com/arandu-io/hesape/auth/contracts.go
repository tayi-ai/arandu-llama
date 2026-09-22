package auth

import "context"

// This file holds the authentication interfaces. They live in the package that
// implements them rather than in a contracts package of their own, because Go
// satisfies an interface structurally and nothing has to be declared.
//
// Every method that touches storage takes a context.Context: without it a
// database lookup in a guard cannot be cancelled when the request is, and the
// caller has no way to pass one in later.

// Authenticatable is the account an authenticated request is about.
//
// It is what a guard hands back, and what a user provider retrieves. An
// application's own user type satisfies it by having the seven methods; nothing
// has to be embedded and nothing has to be registered.
type Authenticatable interface {
	// GetAuthIdentifierName is the column holding the id.
	GetAuthIdentifierName() string

	// GetAuthIdentifier is the id itself.
	GetAuthIdentifier() any

	// GetAuthPasswordName is the column holding the hash.
	GetAuthPasswordName() string

	// GetAuthPassword is the hash, never the plain password.
	GetAuthPassword() string

	// GetRememberToken is the token the "remember me" cookie is checked against.
	GetRememberToken() string

	// SetRememberToken replaces that token on the account.
	SetRememberToken(token string)

	// GetRememberTokenName is the column holding it.
	GetRememberTokenName() string
}

// BroadcastsIdentifier is the optional method that names an account on a
// broadcast channel.
//
// It is an interface of its own rather than an eighth method on
// [Authenticatable]: adding one there would break every user type an
// application already wrote, and [AuthIdentifierForBroadcasting] falls back to
// the plain identifier for a type that does not implement it.
//
// What it is for: the id that goes out on a presence channel, which a person
// other than its owner will read. A model whose primary key is a sequential
// integer leaks how many customers there are and how new this one is, so this
// is the seam where a UUID or a hashid is substituted.
type BroadcastsIdentifier interface {
	// GetAuthIdentifierForBroadcasting is the id that is safe to publish.
	GetAuthIdentifierForBroadcasting() any
}

// AuthIdentifierForBroadcasting returns the id to publish on a channel.
//
// A user type that implements [BroadcastsIdentifier] answers with its own
// value, and every other one answers with GetAuthIdentifier. A nil user is nil.
func AuthIdentifierForBroadcasting(user Authenticatable) any {
	if user == nil {
		return nil
	}
	if b, ok := user.(BroadcastsIdentifier); ok {
		return b.GetAuthIdentifierForBroadcasting()
	}
	return user.GetAuthIdentifier()
}

// UserProvider is the seam between a guard and wherever the users are kept.
//
// A guard knows how a session or a token proves identity; a provider knows how
// to find the row and how to check the password.
type UserProvider interface {
	// RetrieveByID finds the account with this identifier.
	RetrieveByID(ctx context.Context, identifier any) (Authenticatable, error)

	// RetrieveByToken finds the account a "remember me" cookie names, by
	// identifier and remember token.
	RetrieveByToken(ctx context.Context, identifier any, token string) (Authenticatable, error)

	// UpdateRememberToken writes a new remember token for this account.
	UpdateRememberToken(ctx context.Context, user Authenticatable, token string) error

	// RetrieveByCredentials finds the account these credentials name. It must not
	// look at the password key -- finding a user BY their password is a lookup
	// nobody can index and a comparison that is not constant time.
	RetrieveByCredentials(ctx context.Context, credentials map[string]any) (Authenticatable, error)

	// ValidateCredentials reports whether these credentials belong to this
	// account.
	ValidateCredentials(ctx context.Context, user Authenticatable, credentials map[string]any) bool

	// RehashPasswordIfRequired upgrades a hash that was made with weaker
	// parameters, on a sign-in that already proved the plain password.
	RehashPasswordIfRequired(ctx context.Context, user Authenticatable, credentials map[string]any, force bool) error
}

// Guard is what a request asks who is signed in.
//
// It answers who is acting. It decides nothing about what they may do -- that
// is the Policy, and the Grant is the proof it ran.
type Guard interface {
	// Check reports that somebody is signed in.
	Check() bool

	// Guest reports that nobody is.
	Guest() bool

	// User is the authenticated user, or nil.
	User() Authenticatable

	// ID is the authenticated user's identifier, or nil.
	ID() any

	// Validate reports whether these credentials are good, without signing
	// anybody in.
	Validate(ctx context.Context, credentials map[string]any) bool

	// HasUser reports that a user is already resolved, without resolving one.
	HasUser() bool

	// SetUser sets the authenticated user.
	SetUser(user Authenticatable)
}

// StatefulGuard is a [Guard] that can also start and end a session.
type StatefulGuard interface {
	Guard

	// Attempt signs in with these credentials, and reports whether it worked.
	Attempt(ctx context.Context, credentials map[string]any, remember bool) bool

	// Once authenticates for this request only, with no session.
	Once(ctx context.Context, credentials map[string]any) bool

	// Login signs this user in.
	Login(ctx context.Context, user Authenticatable, remember bool)

	// LoginUsingID signs in the account with this identifier, and answers nil
	// when there is none.
	LoginUsingID(ctx context.Context, id any, remember bool) Authenticatable

	// OnceUsingID signs that account in for this request only, and answers nil
	// when there is none.
	OnceUsingID(ctx context.Context, id any) Authenticatable

	// ViaRemember reports that this session came from the cookie, not from a
	// password typed in this browser session.
	ViaRemember() bool

	// Logout ends the session.
	Logout(ctx context.Context)
}

// SupportsBasicAuth is the optional half of a guard that can sign a caller in
// straight from the Authorization: Basic header, with no login form and no
// session to start. A guard declares it when a username and a password on every
// request is a way in: the machine calling an endpoint, and the internal tool
// nobody wants to build a sign-in screen for.
//
// Both methods take the column the username is matched against -- the e-mail
// address by default -- and extra conditions the lookup must also satisfy, and
// both answer nil when the request may carry on. Basic starts a session and
// does nothing when somebody is already signed in; OnceBasic authenticates for
// this request only and writes nothing.
//
// It is a separate interface and not more methods on Guard, so that a guard
// with no business accepting a password on every request simply does not have
// them. [github.com/arandu-io/hesape/auth/middleware.AuthenticateWithBasicAuth]
// asserts for it and answers 500 when the assertion fails, because a guard
// wired where basic auth was expected is a configuration mistake and not a bad
// password.
type SupportsBasicAuth interface {
	// Basic signs in from the header and starts a session.
	Basic(ctx context.Context, field string, extraConditions map[string]any) error

	// OnceBasic signs in from the header for this request only.
	OnceBasic(ctx context.Context, field string, extraConditions map[string]any) error
}

// CanResetPassword is what a user type must satisfy to take part in a password
// reset. It answers with the address a reset link is sent to, and it knows how
// to send one. A type without it can sign in and cannot recover an account,
// because there is nowhere to send the link.
//
// The address is asked for rather than read off a field because it is not
// necessarily the sign-in identifier: an application that signs people in by
// username still has to know where to write to. Sending is asked of the user
// rather than done for them because what arrives is an application's own
// message, on its own template, over its own transport.
type CanResetPassword interface {
	// GetEmailForPasswordReset is the address a reset link is sent to.
	GetEmailForPasswordReset() string

	// SendPasswordResetNotification delivers a reset link carrying token.
	SendPasswordResetNotification(ctx context.Context, token string) error
}

// MustVerifyEmail is what a user type must satisfy for an application to hold
// it to a confirmed e-mail address. It reports whether the address has been
// confirmed, stamps it as confirmed, sends the message that asks, and answers
// with the address that message goes to.
//
// It is a type assertion and not a flag on the user. The
// [github.com/arandu-io/hesape/auth/middleware.EnsureEmailIsVerified]
// middleware asks whether the signed-in user implements this interface and lets
// a user type that does not through untouched, so an application with two kinds
// of account only confirms the addresses of the kind that has one.
//
// [MustVerifyEmailTrait] is the ready implementation a user type embeds to
// answer all four methods.
type MustVerifyEmail interface {
	// HasVerifiedEmail reports that the address has been confirmed.
	HasVerifiedEmail() bool

	// MarkEmailAsVerified stamps the address as confirmed.
	MarkEmailAsVerified(ctx context.Context) error

	// SendEmailVerificationNotification delivers the message that asks for the
	// confirmation.
	SendEmailVerificationNotification(ctx context.Context) error

	// GetEmailForVerification is the address that message goes to.
	GetEmailForVerification() string
}

// Hasher is the part of a password hasher that authentication uses. It is
// declared here, in the consuming package, so that the root of auth keeps
// importing nothing but the standard library -- hesape/hashing satisfies it
// without either side importing the other.
type Hasher interface {
	// Make hashes a plain password.
	Make(value string) (string, error)

	// Check reports whether the plain value matches the hash.
	Check(value, hashedValue string) bool

	// NeedsRehash reports that the hash was made with weaker parameters than the
	// ones now configured.
	NeedsRehash(hashedValue string) bool
}
