package auth

import (
	"context"
	"errors"
	"time"
)

// MustVerifyEmailTrait is the column, the check and the notification behind
// "please confirm your address", ready for a user type to embed.
//
// It carries the Trait suffix because the interface it satisfies is
// [MustVerifyEmail], declared in this same package: Go has one namespace per
// package, and the name was already spoken for.
//
// Two of the four methods reach out of the model -- saving, and notifying --
// and an embedded struct cannot reach the struct it was embedded in. They are
// the Save and Notify fields, and the model sets them where it is built.
type MustVerifyEmailTrait struct {
	// Email is the address the confirmation goes to.
	Email string

	// EmailVerifiedAt is when the address was confirmed. The zero time means it
	// was not.
	EmailVerifiedAt time.Time

	// Save writes the value back, and is what the two mark methods end with. It
	// answers with an error, because the contract does.
	Save func(ctx context.Context) error

	// Notify delivers a notification. It is given
	// [MustVerifyEmailTrait.VerifyEmail].
	Notify func(ctx context.Context, notification any) error

	// VerifyEmail is the notification that asks for the confirmation.
	//
	// It is an any because this package cannot name the type: the root of auth
	// imports nothing outside the standard library (see doc.go), and the message
	// is the application's anyway -- so is the link in it, which has to be signed
	// by whoever holds the application key. The model puts the notification here,
	// and Notify is what knows how to send it.
	VerifyEmail any
}

var _ MustVerifyEmail = (*MustVerifyEmailTrait)(nil)

// HasVerifiedEmail reports that the address has been confirmed.
func (m *MustVerifyEmailTrait) HasVerifiedEmail() bool {
	return !m.EmailVerifiedAt.IsZero()
}

// MarkEmailAsVerified stamps the address as confirmed, at time.Now, and saves.
func (m *MustVerifyEmailTrait) MarkEmailAsVerified(ctx context.Context) error {
	m.EmailVerifiedAt = time.Now()

	return m.save(ctx)
}

// MarkEmailAsUnverified clears that stamp, and saves.
func (m *MustVerifyEmailTrait) MarkEmailAsUnverified(ctx context.Context) error {
	m.EmailVerifiedAt = time.Time{}

	return m.save(ctx)
}

// SendEmailVerificationNotification hands
// [MustVerifyEmailTrait.VerifyEmail] to [MustVerifyEmailTrait.Notify], and
// fails when nothing was wired to deliver it.
func (m *MustVerifyEmailTrait) SendEmailVerificationNotification(ctx context.Context) error {
	if m.Notify == nil {
		return errors.New("auth: MustVerifyEmailTrait.Notify is not set, so the verification notification has nowhere to go")
	}
	return m.Notify(ctx, m.VerifyEmail)
}

// GetEmailForVerification is the address the confirmation goes to.
func (m *MustVerifyEmailTrait) GetEmailForVerification() string {
	return m.Email
}

// save writes the value back, and fails when nothing was wired to write it.
func (m *MustVerifyEmailTrait) save(ctx context.Context) error {
	if m.Save == nil {
		return errors.New("auth: MustVerifyEmailTrait.Save is not set, so the verification stamp was not written")
	}
	return m.Save(ctx)
}
