package encryption

import (
	"crypto/rand"
	"encoding/base64"
	"errors"
	"fmt"
	"strings"
)

// KeySize is the length of the application key, in bytes.
//
// It is a constant and not a setting: the key feeds HMAC-SHA256, which has one
// right key length, and a configurable one only lets a project pick a worse
// value than the default. Everything the application signs -- the session
// cookie, the CSRF token, the links Signer issues -- is signed with this one
// key, because an attacker holding it does not need three.
const KeySize = 32

// keyPrefix marks a key written in base64. Random bytes are not printable, so
// the generated form always carries it; a key typed by hand may not.
const keyPrefix = "base64:"

// GenerateKey returns a new application key, already written the way a .env
// file has to hold it: keyPrefix followed by KeySize random bytes in base64.
//
// [Cipher.GenerateKey] is the one that returns raw bytes. The two are one call
// apart: GenerateKey is "base64:" plus base64 of AES256GCM.GenerateKey().
//
// It returns the encoded string rather than the raw bytes on purpose. The
// caller that wants a key wants to print it or write it down, and if it were
// handed bytes it would have to encode them -- which means the encoding would
// be known in two places, and the one that drifts is the one ParseKey does not
// read.
//
// There is no error to return: crypto/rand.Read fills the slice or the process
// dies trying, and a caller cannot recover from an operating system that has
// stopped producing randomness.
func GenerateKey() string {
	key := make([]byte, KeySize)
	// Errors are documented as impossible since Go 1.24: Read either fills the
	// slice entirely or panics.
	_, _ = rand.Read(key)
	return keyPrefix + base64.StdEncoding.EncodeToString(key)
}

// ErrMissingAppKey is what [ParseKey] returns when the configured key is
// empty. Its message names the command that fixes it, because there is no
// renderer between [ParseKey] and the process that is refusing to start.
//
// It is a distinct error and not the length error below because the two are
// different mistakes: a missing key means the application was never keyed, and a
// short one means it was keyed wrongly. Only the first is what a fresh checkout
// hits.
var ErrMissingAppKey = errors.New("encryption: no application encryption key has been specified (run `aru key:generate`)")

// ParseKey reads a configured application key and returns the bytes it names,
// accepting both the base64 form GenerateKey emits and a raw KeySize-byte
// string typed by hand.
//
// The empty key is [ErrMissingAppKey].
//
// The length is checked here rather than by the caller, so that a key that
// parses is a key that works. Refusing it at boot is the whole point: a short
// key is a weaker signature everywhere at once, and it is not visible from any
// request that would go wrong.
func ParseKey(v string) ([]byte, error) {
	if v == "" {
		return nil, ErrMissingAppKey
	}
	key := []byte(v)
	if after, ok := strings.CutPrefix(v, keyPrefix); ok {
		b, err := base64.StdEncoding.DecodeString(after)
		if err != nil {
			return nil, fmt.Errorf("encryption: the application key is not valid base64: %w", err)
		}
		key = b
	}
	if len(key) != KeySize {
		return nil, fmt.Errorf("encryption: the application key must be %d bytes, got %d (run `aru key:generate`)", KeySize, len(key))
	}
	return key, nil
}
