package hashing

// AuthHasher is a [Hasher] behind the three methods hesape/auth declares as
// auth.Hasher, so that a real hasher can be wired into a guard, a user provider
// and a password broker.
//
// The two Check methods are not the same method. auth.Hasher declares
//
//	Check(value, hashedValue string) bool
//
// and [Hasher] declares
//
//	Check(value, hashedValue string) (bool, error)
//
// so no hasher here satisfies auth.Hasher on its own.
//
// Neither side widens. auth.Hasher is the narrow contract the consuming package
// declares: a guard checks a password and a user provider rehashes one, and
// neither has anywhere to put an error. Widening it would make the root of auth
// name [Params], which means importing this package -- and the root of auth
// imports nothing outside the standard library on purpose, because every package
// that scopes itself by tenant imports it to read auth.Tenant off a Grant. The
// error stays on this side because the two things it reports are worth
// distinguishing here: a wrong password and a column nobody can read.
//
// So this is an adapter and not a second hasher: it hashes nothing of its own
// and holds no parameters of its own. It forwards to the hasher it was given,
// which hashes with the parameters that hasher was built with.
type AuthHasher struct {
	// hasher is the hasher every call is forwarded to.
	hasher Hasher
}

// ForAuth returns h behind the interface hesape/auth consumes.
//
//	provider := users.NewModelUserProvider(hashing.ForAuth(nil), model, newQuery, "acme")
//
// A nil hasher is this package's own path: [Make], which writes argon2id at
// parameters that are compiled in, and [Check], which reads back the argon2i and
// bcrypt an imported users table holds. That is the hasher an application wants,
// and passing nil is how it says so.
//
// It reads argon2id and not bcrypt on purpose. Both were reachable here, and the
// two answers disagreed about what a password column would end up holding, which
// is the one question a hashing default is not allowed to leave open.
func ForAuth(h Hasher) *AuthHasher {
	if h == nil {
		h = defaultHasher{}
	}
	return &AuthHasher{hasher: h}
}

// Make hashes value, and is the Make of auth.Hasher.
//
// The parameters that apply are the ones the hasher behind it was built with.
// There is nothing to pass per call and nowhere to pass it from, which is the
// right shape for the callers it has: a per-call cost factor at a sign-in is a
// caller weakening the hash of the password it is about to store.
func (a *AuthHasher) Make(value string) (string, error) {
	return a.hasher.Make(value)
}

// Check reports whether this password matches this hash, and is the Check of
// auth.Hasher.
//
// An error from the underlying hasher is false. An unreadable hash is the one
// that produces it, which is a corrupt password column and never a right
// password.
//
// The sign-in is refused because there is nowhere to put the error:
// auth.UserProvider.ValidateCredentials answers bool, and so does this contract.
// Refusing is the only reading that is safe in the direction it fails -- a
// column nobody can verify must not authenticate anybody.
func (a *AuthHasher) Check(value, hashedValue string) bool {
	ok, err := a.hasher.Check(value, hashedValue)
	return err == nil && ok
}

// NeedsRehash reports that the hash was written with parameters other than the
// ones in force, so the next sign-in that proves the plain password should
// rewrite it. It is the NeedsRehash of auth.Hasher.
func (a *AuthHasher) NeedsRehash(hashedValue string) bool {
	return a.hasher.NeedsRehash(hashedValue)
}
