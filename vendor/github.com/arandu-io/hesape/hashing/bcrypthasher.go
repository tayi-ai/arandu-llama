package hashing

import (
	"errors"
	"fmt"

	"golang.org/x/crypto/bcrypt"
)

// defaultBcryptRounds is the cost [NewBcryptHasher] starts at.
const defaultBcryptRounds = 12

// BcryptHasher writes and reads bcrypt.
//
// It is not how a password is hashed here. [Make] is, and it writes argon2id at
// parameters that are compiled in. This hasher exists for the hashes that were
// written somewhere else -- a users table imported from an application that
// hashed with bcrypt -- so that a row of that shape can be produced and read
// back on this side of the import.
//
// Reading one does not need it: [Check] accepts bcrypt on its own, and
// [NeedsRehash] reports every bcrypt hash as due for a rewrite, so an imported
// account authenticates once and is stored as argon2id from the next sign-in.
type BcryptHasher struct {
	rounds int
}

// NewBcryptHasher returns a bcrypt hasher. An absent option keeps the default of
// 12 rounds.
func NewBcryptHasher(options ...Options) *BcryptHasher {
	h := &BcryptHasher{rounds: defaultBcryptRounds}
	if o := firstOption(options); o.Rounds > 0 {
		h.rounds = o.Rounds
	}
	return h
}

// Make hashes value with bcrypt at this hasher's cost. bcrypt refuses a value
// longer than 72 bytes, and that refusal is the error.
func (h *BcryptHasher) Make(value string) (string, error) {
	hashed, err := bcrypt.GenerateFromPassword([]byte(value), h.rounds)
	if err != nil {
		return "", fmt.Errorf("hashing: bcrypt hashing not supported: %w", err)
	}
	return string(hashed), nil
}

// Check reports whether value hashes to hashedValue. An empty hashed value is
// false rather than an error.
//
// It reads every hash [Check] reads and not only bcrypt, because the table this
// hasher exists for is one part way through a migration: the rows it has not
// reached yet are bcrypt and the rows it has are argon2id, and both have to
// authenticate.
func (h *BcryptHasher) Check(value, hashedValue string) (bool, error) {
	if hashedValue == "" {
		return false, nil
	}
	switch err := Check(value, hashedValue); {
	case err == nil:
		return true, nil
	case errors.Is(err, ErrInvalidPassword):
		return false, nil
	default:
		return false, err
	}
}

// NeedsRehash is true when the hash is not bcrypt, or was written at a cost
// other than this hasher's. A value it cannot read needs a rehash too: it will
// never verify.
func (h *BcryptHasher) NeedsRehash(hashedValue string) bool {
	p, ok := Info(hashedValue)
	if !ok {
		return true
	}
	return p.Algorithm != Bcrypt || p.Cost != h.rounds
}
