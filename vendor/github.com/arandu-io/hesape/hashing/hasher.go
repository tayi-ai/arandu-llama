package hashing

import "errors"

// Hasher writes and reads a password hash. [AuthHasher] forwards to one, and
// [ForAuth] is what puts it behind the narrower interface hesape/auth consumes.
//
// Two types in this package declare it, and they are not two ways to hash a
// password. The one [ForAuth] uses when it is given none is this package's own
// path -- [Make], which writes argon2id with parameters that are compiled in.
// [BcryptHasher] is the other, and it writes the hashes that already exist
// somewhere else.
type Hasher interface {
	// Make hashes value.
	Make(value string) (string, error)
	// Check reports whether value hashes to hashedValue. A wrong password is
	// false with no error; the error is a hash that cannot be read at all,
	// which is a corrupt column and a different event.
	Check(value, hashedValue string) (bool, error)
	// NeedsRehash reports whether hashedValue was written with parameters
	// other than the ones in force now.
	NeedsRehash(hashedValue string) bool
}

// Options are the settings a hasher is constructed with. A zero field means the
// setting was not given, so the hasher keeps its own value.
type Options struct {
	// Rounds is the bcrypt cost factor.
	Rounds int
}

// firstOption reads the single optional Options. Only the first is read.
func firstOption(options []Options) Options {
	if len(options) == 0 {
		return Options{}
	}
	return options[0]
}

// defaultHasher is [Make], [Check] and [NeedsRehash] behind [Hasher], so that
// the argon2id the framework writes is what reaches hesape/auth.
//
// It holds no fields because there are none to hold: the argon2id cost factors
// [Make] uses are compiled in and are not reachable from configuration.
type defaultHasher struct{}

// Make hashes value with argon2id, and is [Make].
func (defaultHasher) Make(value string) (string, error) {
	return Make(value)
}

// Check reports whether value hashes to hashedValue. An empty hashed value is
// false rather than an error: an account with no password set is not a corrupt
// column.
func (defaultHasher) Check(value, hashedValue string) (bool, error) {
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

// NeedsRehash reports whether the hash was written with anything other than the
// current argon2id parameters, and is [NeedsRehash].
func (defaultHasher) NeedsRehash(hashedValue string) bool {
	return NeedsRehash(hashedValue)
}
