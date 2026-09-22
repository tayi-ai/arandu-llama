package auth

import (
	"context"
	"errors"
)

var (
	// ErrInvalidCredentials reports that credentials did not identify and
	// validate an account. It deliberately does not say which half failed.
	ErrInvalidCredentials = errors.New("auth: invalid credentials")
)

// CredentialVerifier validates credentials and returns their account without
// creating request-local identity, a cookie, or a session.
type CredentialVerifier struct {
	provider        UserProvider
	newTimebox      func() Timebox
	rehashOnSuccess bool
	timeboxDuration int
}

// NewCredentialVerifier returns a verifier backed by provider.
//
// newTimebox is called for every verification because a Timebox keeps the
// mutable state set by ReturnEarly. A nil factory uses [NewTimebox], and a zero
// duration uses 200000 microseconds.
func NewCredentialVerifier(
	provider UserProvider,
	newTimebox func() Timebox,
	rehashOnSuccess bool,
	timeboxDuration int,
) *CredentialVerifier {
	if newTimebox == nil {
		newTimebox = NewTimebox
	}
	if timeboxDuration == 0 {
		timeboxDuration = defaultTimeboxDuration
	}

	return &CredentialVerifier{
		provider:        provider,
		newTimebox:      newTimebox,
		rehashOnSuccess: rehashOnSuccess,
		timeboxDuration: timeboxDuration,
	}
}

// Verify returns the account these credentials identify when they validate.
// Invalid credentials return [ErrInvalidCredentials] after the same timebox
// path whether the account was absent or its secret was wrong.
func (v *CredentialVerifier) Verify(ctx context.Context, credentials map[string]any) (Authenticatable, error) {
	timebox := v.newTimebox()

	verified, err := timebox.Call(func(timebox Timebox) (any, error) {
		user, err := verifyCredentials(ctx, v.provider, credentials)
		if err != nil {
			return nil, err
		}
		if v.rehashOnSuccess {
			_ = v.provider.RehashPasswordIfRequired(ctx, user, credentials, false)
		}

		timebox.ReturnEarly()

		return user, nil
	}, v.timeboxDuration)
	if err != nil {
		return nil, err
	}

	user, _ := verified.(Authenticatable)
	return user, nil
}

// verifyCredentials is the lookup and credential check shared by stateless
// verification and SessionGuard.
func verifyCredentials(ctx context.Context, provider UserProvider, credentials map[string]any) (Authenticatable, error) {
	user, err := provider.RetrieveByCredentials(ctx, credentials)
	if err != nil {
		return nil, err
	}
	if user == nil || !provider.ValidateCredentials(ctx, user, credentials) {
		return user, ErrInvalidCredentials
	}

	return user, nil
}
