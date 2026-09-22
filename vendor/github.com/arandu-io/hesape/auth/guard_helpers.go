package auth

// GuardHelpers is the handful of methods every guard answers the same way, and
// which SessionGuard, TokenGuard and RequestGuard all gain by embedding it.
//
// A promoted method in Go sees the struct it was declared on, never the struct
// it was embedded in, so these methods cannot call the guard's own User. Each
// guard hands its own over at construction, in resolveUser.
//
// Its fields are unexported, because every guard that reads them is in this
// package.
type GuardHelpers struct {
	// user is the currently authenticated user, already resolved. It is nil
	// until something resolves one.
	user Authenticatable

	// provider is where the accounts are looked up.
	provider UserProvider

	// resolveUser is the guard's own User, set at construction. When it is nil
	// -- a GuardHelpers used on its own, which no guard does -- the helpers fall
	// back to the resolved user, so nothing here panics on a zero value.
	resolveUser func() Authenticatable
}

// Authenticate is the user, or the failure.
//
// The error is an *[AuthenticationError], so a caller can read the guards off
// it with errors.As.
func (g *GuardHelpers) Authenticate() (Authenticatable, error) {
	if user := g.currentUser(); user != nil {
		return user, nil
	}
	return nil, NewAuthenticationError("", nil, "")
}

// HasUser reports that a user is already resolved.
//
// It asks about the field, not about the guard's own User, so it never goes to
// the session or the database to answer.
func (g *GuardHelpers) HasUser() bool {
	return g.user != nil
}

// Check reports that somebody is signed in.
func (g *GuardHelpers) Check() bool {
	return g.currentUser() != nil
}

// Guest reports that nobody is.
func (g *GuardHelpers) Guest() bool {
	return !g.Check()
}

// ID is the authenticated user's identifier, or nil.
func (g *GuardHelpers) ID() any {
	if user := g.currentUser(); user != nil {
		return user.GetAuthIdentifier()
	}
	return nil
}

// SetUser sets the authenticated user.
//
// It returns nothing, because the Guard contract's SetUser returns nothing and
// because a promoted method that returned something would hand back the
// embedded struct, not the guard that embedded it.
func (g *GuardHelpers) SetUser(user Authenticatable) {
	g.user = user
}

// ForgetUser drops the resolved user without touching the session or the
// cookie.
//
// It returns nothing, for the reason [GuardHelpers.SetUser] gives.
func (g *GuardHelpers) ForgetUser() {
	g.user = nil
}

// GetProvider is where the accounts are looked up.
func (g *GuardHelpers) GetProvider() UserProvider {
	return g.provider
}

// SetProvider sets where the accounts are looked up.
func (g *GuardHelpers) SetProvider(provider UserProvider) {
	g.provider = provider
}

// currentUser is the guard's own resolution when there is one, and the resolved
// user otherwise.
func (g *GuardHelpers) currentUser() Authenticatable {
	if g.resolveUser != nil {
		return g.resolveUser()
	}
	return g.user
}
