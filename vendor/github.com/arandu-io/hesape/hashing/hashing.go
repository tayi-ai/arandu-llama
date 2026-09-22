package hashing

import (
	"crypto/rand"
	"crypto/subtle"
	"encoding/base64"
	"errors"
	"fmt"
	"strings"

	"golang.org/x/crypto/argon2"
	"golang.org/x/crypto/bcrypt"
)

// argon2id parameters. They are deliberately not configurable through the
// environment: an insecure hash configuration is the most common way to break
// authentication without noticing.
const (
	argonTime    = 3
	argonMemory  = 64 * 1024 // 64 MiB
	argonThreads = 4
	argonKeyLen  = 32
	saltLen      = 16

	// MinPasswordLen is the shortest password the framework accepts. Length
	// beats composition rules: it is the only parameter that reliably raises
	// the cost of an offline attack.
	//
	// What it measures is the string [Make] is handed, which is the right
	// measure of a secret a person chose and no measure at all of one a machine
	// drew: ten characters from a thirty-two letter alphabet are fifty bits and
	// are turned away here, while twelve characters of one chosen word are not.
	//
	// There is deliberately no second entry point that waives the floor for the
	// machine-generated case. A waiver a caller can reach for a generated
	// secret is one a caller can reach for a chosen password, and for a chosen
	// password this floor is the whole of the defence -- so the waiver would
	// cost more than it saves, and it would be a second way to hash a
	// credential besides.
	//
	// A caller hashing a generated secret hashes it behind a constant that
	// names what the secret is for, applied identically where it is issued and
	// where it is checked. That constant is what keeps a hash moved between two
	// columns from verifying in the wrong one, so it is there whatever this
	// floor says, and it carries the value past the floor on the way.
	MinPasswordLen = 12

	// MaxPasswordLen is the longest password the framework accepts. A
	// passphrase has to fit, so the ceiling is high; an unbounded one lets an
	// anonymous caller spend the server's memory and CPU one megabyte at a
	// time. Both limits count runes, not bytes.
	MaxPasswordLen = 128
)

// Algorithm names a password hashing function. Make writes Argon2id; Check
// also reads Argon2i and Bcrypt, which is what a database imported from
// another framework contains.
type Algorithm string

// The algorithms this package can read. Only Argon2id is ever written.
const (
	Argon2id Algorithm = "argon2id"
	Argon2i  Algorithm = "argon2i"
	Bcrypt   Algorithm = "bcrypt"
)

// ErrInvalidPassword means the password does not match the stored hash. An
// unreadable hash returns a different error: the column is corrupt, which is
// not the same event as a wrong password and must not be reported as one.
var ErrInvalidPassword = errors.New("hashing: invalid password")

// Params describes how a stored hash was produced. Fields that do not apply to
// the algorithm are zero: Cost is bcrypt only, Memory, Time and Threads are
// argon2 only.
type Params struct {
	// Algorithm is Argon2id, Argon2i or Bcrypt.
	Algorithm Algorithm
	// Memory is the argon2 memory cost in KiB.
	Memory uint32
	// Time is the number of argon2 passes over the memory.
	Time uint32
	// Threads is the argon2 degree of parallelism.
	Threads uint8
	// Cost is the base-2 logarithm of the bcrypt round count.
	Cost int
	// KeyLen is the length in bytes of the derived key, argon2 only.
	KeyLen int
}

// Make returns the hash in PHC string format, which is self-describing and
// therefore allows rehashing when parameters change.
func Make(plain string) (string, error) {
	switch n := len([]rune(plain)); {
	case n < MinPasswordLen:
		return "", fmt.Errorf("hashing: password must be at least %d characters", MinPasswordLen)
	case n > MaxPasswordLen:
		return "", fmt.Errorf("hashing: password must be at most %d characters", MaxPasswordLen)
	}
	salt := make([]byte, saltLen)
	if _, err := rand.Read(salt); err != nil {
		return "", err
	}
	key := argon2.IDKey([]byte(plain), salt, argonTime, argonMemory, argonThreads, argonKeyLen)
	return fmt.Sprintf("$%s$v=%d$m=%d,t=%d,p=%d$%s$%s",
		Argon2id, argon2.Version,
		argonMemory, argonTime, argonThreads,
		base64.RawStdEncoding.EncodeToString(salt),
		base64.RawStdEncoding.EncodeToString(key),
	), nil
}

// Check compares in constant time. It reads the argon2id hashes Make writes,
// and also argon2i and bcrypt, so an imported table authenticates before it is
// rehashed.
func Check(plain, encoded string) error {
	switch {
	case strings.HasPrefix(encoded, bcryptPrefix):
		err := bcrypt.CompareHashAndPassword([]byte(encoded), []byte(plain))
		switch {
		case err == nil:
			return nil
		case errors.Is(err, bcrypt.ErrMismatchedHashAndPassword):
			return ErrInvalidPassword
		default:
			return fmt.Errorf("hashing: unreadable bcrypt hash: %w", err)
		}
	case strings.HasPrefix(encoded, argonPrefix):
		h, err := parseArgon(encoded)
		if err != nil {
			return err
		}
		var got []byte
		if h.params.Algorithm == Argon2id {
			got = argon2.IDKey([]byte(plain), h.salt, h.params.Time, h.params.Memory, h.params.Threads, uint32(h.params.KeyLen))
		} else {
			got = argon2.Key([]byte(plain), h.salt, h.params.Time, h.params.Memory, h.params.Threads, uint32(h.params.KeyLen))
		}
		if subtle.ConstantTimeCompare(got, h.key) != 1 {
			return ErrInvalidPassword
		}
		return nil
	default:
		return fmt.Errorf("hashing: unknown hash format")
	}
}

// Info reports how a stored hash was produced. It returns false when the value
// is not a hash this package can read, which is what IsHashed is built on.
func Info(encoded string) (Params, bool) {
	switch {
	case strings.HasPrefix(encoded, bcryptPrefix):
		cost, err := bcrypt.Cost([]byte(encoded))
		if err != nil {
			return Params{}, false
		}
		return Params{Algorithm: Bcrypt, Cost: cost}, true
	case strings.HasPrefix(encoded, argonPrefix):
		h, err := parseArgon(encoded)
		if err != nil {
			return Params{}, false
		}
		return h.params, true
	default:
		return Params{}, false
	}
}

// IsHashed reports whether the value is already a hash. It answers the one
// question worth asking before writing a password column: whether a plaintext
// leaked into it.
func IsHashed(v string) bool {
	_, ok := Info(v)
	return ok
}

// NeedsRehash reports that the hash was produced with older parameters or
// another algorithm, so the caller should rehash on the next successful login.
// A value it cannot read needs a rehash too: it will never verify.
func NeedsRehash(encoded string) bool {
	p, ok := Info(encoded)
	if !ok {
		return true
	}
	return p.Algorithm != Argon2id ||
		p.Memory != argonMemory ||
		p.Time != argonTime ||
		p.Threads != argonThreads ||
		p.KeyLen != argonKeyLen
}

// The prefixes the PHC strings this package reads begin with. bcrypt writes
// "$2a$", "$2b$" or "$2y$"; the minor version does not change the digest for
// the passwords Make accepts.
const (
	bcryptPrefix = "$2"
	argonPrefix  = "$argon2"
)

// argonHash is a parsed argon2 PHC string: the public half plus the two byte
// slices Check needs and Info must not expose.
type argonHash struct {
	params Params
	salt   []byte
	key    []byte
}

// parseArgon reads a PHC string and refuses every shape that would make the
// argon2 package panic — a zero pass count, a zero degree of parallelism or an
// empty key. Those bytes come from a database column, and a corrupt column
// must fail the login, not the process.
func parseArgon(encoded string) (argonHash, error) {
	parts := strings.Split(encoded, "$")
	if len(parts) != 6 {
		return argonHash{}, fmt.Errorf("hashing: unknown hash format")
	}
	alg := Algorithm(parts[1])
	if alg != Argon2id && alg != Argon2i {
		return argonHash{}, fmt.Errorf("hashing: unknown hash format")
	}
	var version int
	if _, err := fmt.Sscanf(parts[2], "v=%d", &version); err != nil || version != argon2.Version {
		return argonHash{}, fmt.Errorf("hashing: unsupported argon2 version %q", parts[2])
	}
	var m, t uint32
	var p uint8
	if _, err := fmt.Sscanf(parts[3], "m=%d,t=%d,p=%d", &m, &t, &p); err != nil {
		return argonHash{}, fmt.Errorf("hashing: unreadable hash parameters: %w", err)
	}
	if parts[3] != fmt.Sprintf("m=%d,t=%d,p=%d", m, t, p) {
		return argonHash{}, fmt.Errorf("hashing: unreadable hash parameters")
	}
	if t < 1 || p < 1 || m < 8*uint32(p) {
		return argonHash{}, fmt.Errorf("hashing: hash parameters out of range")
	}
	salt, err := base64.RawStdEncoding.DecodeString(parts[4])
	if err != nil {
		return argonHash{}, fmt.Errorf("hashing: unreadable hash salt: %w", err)
	}
	if len(salt) < 8 {
		return argonHash{}, fmt.Errorf("hashing: hash salt is too short")
	}
	key, err := base64.RawStdEncoding.DecodeString(parts[5])
	if err != nil {
		return argonHash{}, fmt.Errorf("hashing: unreadable hash key: %w", err)
	}
	if len(key) < 16 || len(key) > 64 {
		return argonHash{}, fmt.Errorf("hashing: hash key length is out of range")
	}
	return argonHash{
		params: Params{Algorithm: alg, Memory: m, Time: t, Threads: p, KeyLen: len(key)},
		salt:   salt,
		key:    key,
	}, nil
}
