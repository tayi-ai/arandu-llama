package encryption

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// Cipher names an encryption algorithm.
//
// This package supports one, [AES256GCM]. A second cipher would be a second way
// to encrypt, and the alternatives are not equally safe: an unauthenticated
// mode has to carry an HMAC alongside the ciphertext and check it by hand,
// getting that wrong is silent, and the mistake is only visible to whoever is
// forging payloads. AES-256-GCM authenticates as part of decrypting, so there
// is no second thing to check and no way to forget to check it.
//
// The value is compared case-insensitively.
type Cipher string

// AES256GCM is the cipher this package encrypts with.
const AES256GCM Cipher = "aes-256-gcm"

// ivLength is the nonce size of AES-GCM in bytes, which is what crypto/cipher
// produces from NewGCM.
const ivLength = 12

// tagLength is the AEAD tag size in bytes. A payload carrying a shorter or
// longer tag is refused before any key is tried.
const tagLength = 16

// size reports the key length the cipher requires, and whether the cipher is
// one this package has.
func (c Cipher) size() (int, bool) {
	switch Cipher(strings.ToLower(string(c))) {
	case AES256GCM:
		return 32, true
	}
	return 0, false
}

// GenerateKey returns a new random key of the length the cipher requires.
//
// The cipher is the receiver rather than an argument because Go has no
// overloading and the package already exports a [GenerateKey] that returns the
// printable "base64:" form a configuration file holds.
//
// An unrecognised cipher yields 32 bytes rather than an error. There is no
// error to return: crypto/rand fills the slice or the process dies trying.
func (c Cipher) GenerateKey() []byte {
	size, ok := c.size()
	if !ok {
		size = 32
	}
	key := make([]byte, size)
	// Errors are documented as impossible since Go 1.24: Read either fills the
	// slice entirely or panics.
	_, _ = rand.Read(key)
	return key
}

// Supported reports whether the key and cipher combination is usable.
//
// Both halves matter: an unknown cipher fails, and so does a key of the wrong
// length for a known one. The key is measured in bytes.
func Supported(key []byte, cipher Cipher) bool {
	size, ok := cipher.size()
	if !ok {
		return false
	}
	return len(key) == size
}

// ErrUnsupportedCipher is returned by [NewEncrypter] and
// [Encrypter.PreviousKeys] when the key length does not match the cipher.
var ErrUnsupportedCipher = errors.New("encryption: unsupported cipher or incorrect key length; the supported cipher is: aes-256-gcm")

// ErrEncrypt wraps every failure on the way out: the value would not serialise
// to JSON, or the cipher refused the key.
//
// It is not a mistake a caller recovers from by trying something else. Both
// causes are configuration -- a key of the wrong length, or a value holding
// something json.Marshal cannot represent -- so a handler that reaches it
// should report and stop, not retry.
//
// The underlying error is wrapped, so errors.Is finds this one and %v still
// prints what actually went wrong.
var ErrEncrypt = errors.New("encryption: could not encrypt the data")

// ErrDecrypt is the answer to anything that arrives and does not decrypt: a
// payload that was tampered with, one encrypted under a different key, or one
// that is not a payload at all.
//
// The three are deliberately indistinguishable from outside. Telling a caller
// WHICH of them happened tells an attacker whether their forgery got closer,
// which is the oracle that makes padding attacks work. errInvalidPayload wraps
// this one for the same reason -- it is more specific inside the package and
// identical to anyone outside it.
//
// Every failure below wraps it, so a caller answers "this value is not ours"
// once rather than switching on four reasons it is not. The reasons are still
// in the message, because the one reading it is a developer holding a payload,
// not a request.
var ErrDecrypt = errors.New("encryption: could not decrypt the data")

// errInvalidPayload is DecryptException('The payload is invalid.') -- the value
// is not a payload this encrypter wrote, and no key was tried.
var errInvalidPayload = fmt.Errorf("%w: the payload is invalid", ErrDecrypt)

// Encrypter encrypts and decrypts values with the application key.
//
// The payload it writes is base64 of a JSON object with iv, value, mac and tag,
// in that order, each of them base64 itself. The mac is empty, because with an
// AEAD cipher the tag is the MAC; the field stays in place because a reader
// holding a column of mixed formats needs it to tell them apart.
//
// It is safe for concurrent use. Nothing mutates after construction except
// PreviousKeys, which is a boot-time call.
type Encrypter struct {
	key          []byte
	previousKeys [][]byte
	cipher       Cipher
}

// NewEncrypter returns an Encrypter over the given key.
//
// The cipher has no default. Passing [AES256GCM] is the only call that works,
// and requiring it keeps the payload format a thing the caller chose rather
// than a thing it inherited.
//
// The key is copied, so a caller that zeroes or reuses its buffer does not
// silently change what the application encrypts with.
func NewEncrypter(key []byte, cipher Cipher) (*Encrypter, error) {
	if !Supported(key, cipher) {
		return nil, fmt.Errorf("%w: got a %d-byte key for cipher %q", ErrUnsupportedCipher, len(key), cipher)
	}
	return &Encrypter{
		key:    append([]byte(nil), key...),
		cipher: cipher,
	}, nil
}

// payloadOut is the JSON object an encrypted value is. Go marshals struct
// fields in declaration order, so the field order here is the wire order: iv,
// value, mac, tag.
type payloadOut struct {
	IV    string `json:"iv"`
	Value string `json:"value"`
	MAC   string `json:"mac"`
	Tag   string `json:"tag"`
}

// payloadIn is the same object being read back. The fields are pointers so a
// missing key and an empty string stay distinguishable: mac is legitimately ""
// under an AEAD cipher, but a payload with no mac field at all is not one of
// ours. A field holding a non-string makes Unmarshal fail, which is the same
// answer.
type payloadIn struct {
	IV    *string `json:"iv"`
	Value *string `json:"value"`
	MAC   *string `json:"mac"`
	Tag   *string `json:"tag"`
}

// Encrypt encrypts a value of any type, serialising it to JSON first.
//
// It is a function and not a method because it is generic and Go methods
// cannot be. The type parameter is what the caller states the payload holds,
// and the compiler holds it to it.
//
// Serialization is encoding/json. A value written by [Encrypt] is read back by
// [Decrypt] and by nothing else; [Encrypter.EncryptString] is the pair whose
// payload is portable.
func Encrypt[T any](e *Encrypter, value T) (string, error) {
	serialized, err := json.Marshal(value)
	if err != nil {
		return "", fmt.Errorf("%w: %w", ErrEncrypt, err)
	}
	return e.encrypt(serialized)
}

// EncryptString encrypts a string without serializing it.
//
// The payload is the portable one: anything else holding the same key and
// reading this format can decrypt it, and this package can decrypt what such a
// reader wrote. An empty string is a legal input and produces a real payload:
// GCM over no plaintext is a nonce and a tag, and [Encrypter.DecryptString]
// returns "" from it.
func (e *Encrypter) EncryptString(value string) (string, error) {
	return e.encrypt([]byte(value))
}

// encrypt is the body both public forms share, after the serialise decision.
func (e *Encrypter) encrypt(plaintext []byte) (string, error) {
	aead, err := e.aead(e.key)
	if err != nil {
		return "", fmt.Errorf("%w: %w", ErrEncrypt, err)
	}

	iv := make([]byte, aead.NonceSize())
	_, _ = rand.Read(iv)

	// Seal returns ciphertext and tag as one slice; the payload carries them
	// in two fields, so they are split here.
	sealed := aead.Seal(nil, iv, plaintext, nil)
	split := len(sealed) - aead.Overhead()

	body, err := json.Marshal(payloadOut{
		IV:    base64.StdEncoding.EncodeToString(iv),
		Value: base64.StdEncoding.EncodeToString(sealed[:split]),
		// Empty for an AEAD cipher: the tag is the MAC.
		MAC: "",
		Tag: base64.StdEncoding.EncodeToString(sealed[split:]),
	})
	if err != nil {
		return "", fmt.Errorf("%w: %w", ErrEncrypt, err)
	}
	return base64.StdEncoding.EncodeToString(body), nil
}

// Decrypt decrypts a value written by [Encrypt] back into T.
//
// It is a function and not a method for the same reason [Encrypt] is.
func Decrypt[T any](e *Encrypter, payload string) (T, error) {
	var zero T
	plaintext, err := e.decrypt(payload)
	if err != nil {
		return zero, err
	}
	var value T
	if err := json.Unmarshal(plaintext, &value); err != nil {
		// The key was right -- the payload opened. What is inside is not what
		// this call expected, which is a different mistake from a bad key and
		// says so, because the two are fixed in different places.
		return zero, fmt.Errorf("%w: the payload decrypted but does not hold this type; was it written by EncryptString?", ErrDecrypt)
	}
	return value, nil
}

// DecryptString decrypts a string without deserialising it. It is the pair to
// [Encrypter.EncryptString].
func (e *Encrypter) DecryptString(payload string) (string, error) {
	plaintext, err := e.decrypt(payload)
	if err != nil {
		return "", err
	}
	return string(plaintext), nil
}

// decrypt is the body both public forms share, before the deserialise
// decision.
func (e *Encrypter) decrypt(payload string) ([]byte, error) {
	p, err := e.getJSONPayload(payload)
	if err != nil {
		return nil, err
	}

	// Already validated by validPayload, which is why the error is dropped.
	iv, _ := base64.StdEncoding.DecodeString(*p.IV)

	var tag []byte
	if p.Tag != nil && *p.Tag != "" {
		if tag, err = base64.StdEncoding.DecodeString(*p.Tag); err != nil {
			return nil, errInvalidPayload
		}
	}
	if err := e.ensureTagIsValid(tag); err != nil {
		return nil, err
	}

	value, err := base64.StdEncoding.DecodeString(*p.Value)
	if err != nil {
		return nil, ErrDecrypt
	}
	sealed := append(value, tag...)

	// Every key, current first, then the retired ones. This is the whole of key
	// rotation: a deploy sets the new key and moves the old one to
	// PreviousKeys, and everything already encrypted keeps opening. Nothing is
	// ever encrypted with a previous key.
	for _, key := range e.GetAllKeys() {
		aead, err := e.aead(key)
		if err != nil || len(iv) != aead.NonceSize() {
			continue
		}
		// Open verifies the tag before returning any plaintext, so a forged
		// payload fails here rather than downstream.
		if plaintext, err := aead.Open(nil, iv, sealed, nil); err == nil {
			return plaintext, nil
		}
	}
	return nil, ErrDecrypt
}

// aead builds the AEAD for a key. Split out because decrypt builds one per key
// it tries.
func (e *Encrypter) aead(key []byte) (cipher.AEAD, error) {
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}
	return cipher.NewGCM(block)
}

// getJSONPayload reads the payload and refuses anything that is not one.
func (e *Encrypter) getJSONPayload(payload string) (*payloadIn, error) {
	decoded, err := base64.StdEncoding.DecodeString(payload)
	if err != nil {
		return nil, errInvalidPayload
	}
	var p payloadIn
	if err := json.Unmarshal(decoded, &p); err != nil {
		return nil, errInvalidPayload
	}
	if !e.validPayload(&p) {
		return nil, errInvalidPayload
	}
	return &p, nil
}

// validPayload reports whether the decoded object has the shape of a payload.
//
// The iv length check is the load-bearing one, and it is why this runs before
// any key is tried: it separates "not one of ours" from "ours, and it failed",
// and only the second is worth reporting as a decryption failure.
func (e *Encrypter) validPayload(p *payloadIn) bool {
	if p.IV == nil || p.Value == nil || p.MAC == nil {
		return false
	}
	iv, err := base64.StdEncoding.DecodeString(*p.IV)
	if err != nil {
		return false
	}
	return len(iv) == ivLength
}

// ensureTagIsValid refuses a tag of the wrong length.
//
// There is no branch for a payload carrying a tag under a cipher that cannot
// use one: [AES256GCM] is the only cipher here and it is AEAD, so nothing could
// reach it.
func (e *Encrypter) ensureTagIsValid(tag []byte) error {
	if len(tag) != tagLength {
		return ErrDecrypt
	}
	return nil
}

// AppearsEncrypted reports whether a value looks like something this encrypter
// wrote.
//
// It is a guess and says so by returning a bool rather than an error: it reads
// the envelope, never a key. It is what a migration re-encrypting a column uses
// to skip the rows it already did. A true here does not promise [DecryptString]
// will succeed -- only that it is worth calling.
func AppearsEncrypted(value string) bool {
	decoded, err := base64.StdEncoding.DecodeString(value)
	if err != nil {
		return false
	}
	var p payloadIn
	if err := json.Unmarshal(decoded, &p); err != nil {
		return false
	}
	return p.IV != nil && p.Value != nil && p.MAC != nil
}

// GetKey returns the key the encrypter encrypts with.
//
// The copy is not politeness. A []byte is a window onto the encrypter's own
// memory, and a caller that appended to it would be rewriting the application
// key from outside.
func (e *Encrypter) GetKey() []byte {
	return append([]byte(nil), e.key...)
}

// GetAllKeys returns the current key followed by every previous key.
//
// The order is the order decryption tries them, and the current key is first
// because it is the one that will work.
func (e *Encrypter) GetAllKeys() [][]byte {
	keys := make([][]byte, 0, 1+len(e.previousKeys))
	keys = append(keys, e.GetKey())
	return append(keys, e.GetPreviousKeys()...)
}

// GetPreviousKeys returns the retired keys, without the current one.
func (e *Encrypter) GetPreviousKeys() [][]byte {
	keys := make([][]byte, 0, len(e.previousKeys))
	for _, key := range e.previousKeys {
		keys = append(keys, append([]byte(nil), key...))
	}
	return keys
}

// PreviousKeys sets the retired keys that decryption falls back to, and
// returns the encrypter so the call chains.
//
// Every key is checked against the cipher first, and none is stored if any
// fails, so a bad list leaves the encrypter untouched. A key that is the wrong
// length would not be used at decryption time anyway; the point of refusing it
// here is that a typo in
// APP_PREVIOUS_KEYS should stop a deploy, not quietly stop old payloads from
// opening.
func (e *Encrypter) PreviousKeys(keys [][]byte) (*Encrypter, error) {
	for i, key := range keys {
		if !Supported(key, e.cipher) {
			return nil, fmt.Errorf("%w: previous key %d is %d bytes", ErrUnsupportedCipher, i+1, len(key))
		}
	}
	previous := make([][]byte, 0, len(keys))
	for _, key := range keys {
		previous = append(previous, append([]byte(nil), key...))
	}
	e.previousKeys = previous
	return e, nil
}
