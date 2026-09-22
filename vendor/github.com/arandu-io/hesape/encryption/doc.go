// Package encryption encrypts, decrypts and signs values with the application
// key.
//
// [Encrypter] is the type that holds the key. [Encrypt] and [Decrypt] carry an
// arbitrary Go value through JSON; [Encrypter.EncryptString] and
// [Encrypter.DecryptString] carry a string as it is. [Supported] reports
// whether a key and cipher pair can be used at all, and [AppearsEncrypted]
// recognises a payload written by this package without needing the key.
//
// [ParseKey] reads the "base64:" form a configuration file holds and returns
// [ErrMissingAppKey] for the empty key, so an unkeyed application stops at boot
// rather than at the first request that needed the key.
//
// [Signer] signs and verifies URLs and tokens. It lives here because it signs
// with the same application key, and a second place holding that key is a
// second place to get it wrong.
package encryption
