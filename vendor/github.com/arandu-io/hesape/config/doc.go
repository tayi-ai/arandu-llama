// Package config reads and validates the settings an application runs on.
//
// It holds: Repository, App, Env, LoadDotenv, String, MustString, Bool, Int,
// Seconds.
//
// # Two readers, and the difference between them
//
// [Repository] is the dotted-key store -- Has, Get, GetMany, String, Integer,
// Float, Boolean, Array, Collection, Set, Prepend, Push, All. Keys are strings,
// so a wrong one is found at runtime.
//
// [App] is the same settings as a typed struct, and it is what the framework
// itself reads: a wrong field is a compile error there, instead of a nil that
// surfaces on the first request that happens to need it. A first-party package
// that looks a setting up by string is doing it wrong.
//
// # The struct is the source of truth
//
// These are not two ways to configure an application. There is one source, and
// it is [App]: [Load] parses the environment into it, validates it, and fails
// the process at boot if it does not hold together. [Repository] is a reader
// laid over the same settings for the case the struct cannot serve -- a key
// whose name is only known at runtime -- and it is a reader and not a second
// store.
//
// Nothing the framework depends on is read through [Repository], so a key set
// there and nowhere else configures nothing. When the two disagree about a
// setting the framework uses, [App] is right and the string key is a typo.
//
// # Load is the one entry point
//
// [Load] reads a .env file from the working directory when there is one --
// filling only what the environment has not already defined -- and then
// validates the result. There is no second loader and no flag that moves the
// file.
//
// It fails at boot, not on the first request. A configuration that cannot be
// served is a process that must not start.
//
// # What this package does not carry
//
// [App] is the identity of the application and the key it signs with, and
// nothing else. The connection belongs to the database package, the store URL
// to cache and session, the TTLs to session, the level and the tracing secret
// to log. Each of those parses what it needs from the environment this package
// has already populated, which is why [LoadDotenv] and the readers below are
// exported: they are the mechanism, and the mechanism is shared.
package config
