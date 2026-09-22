// Package hashing writes and verifies password hashes.
//
// There is one way to hash a password: [Make], which writes argon2id. Its cost
// factors are compiled in and are not reachable from configuration or the
// environment, because an insecure hash configuration is the most common way to
// break authentication without noticing, and because the answer to "what is in
// the password column" has to be one answer.
//
// [Check] reads back what [Make] writes, and also argon2i and bcrypt, which is
// what a users table imported from an existing application holds. That is a read
// path and not a second way to hash: an imported row authenticates on the first
// sign-in, [NeedsRehash] reports it as due, and the caller rewrites it as
// argon2id from there. [Info] reports the parameters a stored hash was written
// with, and [IsHashed] answers the one question worth asking before writing a
// password column: whether a plaintext leaked into it.
//
// [BcryptHasher] is the exception that proves the rule. It writes bcrypt, which
// nothing here hashes a password with; it exists so that the rows an import will
// contain can be produced and read back on this side. Reading one does not need
// it -- [Check] accepts bcrypt on its own.
//
// [AuthHasher] is an adapter, not a hasher. hesape/auth declares a narrower
// three-method contract that has nowhere to put an error, and [ForAuth] puts a
// [Hasher] behind it. ForAuth(nil) is [Make] and [Check], which is what an
// application wants.
//
// The package knows nothing about users, sessions or requests. It takes a string
// and returns a string, and the only third-party import is golang.org/x/crypto.
package hashing
