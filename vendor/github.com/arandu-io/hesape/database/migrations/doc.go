// Package migrations runs schema changes: the Migration contract, the registry
// they declare themselves in, the migrator that applies them and the repository
// table that records what ran.
//
// # A migration does not run at boot
//
// `aru migrate` is a step of the deployment pipeline, run once, by one process.
// Called from the start-up path of an application with N replicas it becomes N
// migrators racing over one table, and the ones that lose report a duplicate key
// on a table they were in the middle of creating. Nothing in this package makes
// that convenient: there is no MigrateOnBoot, and there will not be one.
//
// The other half belongs to the migration, and no code can check it: every
// migration is compatible with the binary that is still serving traffic while
// the rollout finishes. A new column is nullable or carries a default, because
// the old binary's INSERT does not mention it. Removing a column takes two
// releases -- the first stops writing it, the second drops it -- because the old
// binary's SELECT still names it. A rename is an add, a backfill, and a drop,
// which is three releases.
//
// # Discovery: a registry, not a directory
//
// A package nothing imports is not in the binary, so a scan of a migrations
// directory at run time would find files the compiler never saw, and in a
// deployed container would find nothing at all, because the source is not there.
//
// So a migration registers itself:
//
//	func init() { migrations.Register(CreateUsersTable{}) }
//
// and main.go blank-imports the package they live in. That import is the loading
// step, moved to link time, with the compiler checking that every registered
// migration implements Migration before anything runs. Register carries the
// argument in full.
//
// The alternative considered and rejected was embedding the directory and
// treating a migration as SQL text. It reads well until the first migration that
// has to read rows before it writes -- a backfill, a conditional drop -- and
// then there are two kinds of migration.
//
// Order comes from the name and from nothing else: GetName answers
// "2026_08_11_000000_create_users_table", the registry sorts by that string, and
// two machines therefore apply the same migrations in the same order.
//
// # There is no Grant here
//
// Every path to application rows carries an auth.Grant and filters by
// auth.Tenant(g), on reads as much as on writes. A migration is not such a path:
// it is DDL, run by a pipeline step, in a process with no request and no subject
// -- there is nothing a Grant could be built from, and inventing a subject to
// satisfy a signature would be worse than not having one. The repository table
// is framework metadata, not tenant data, for the same reason.
//
// Neither the migrator nor the creator holds a filesystem: the migrator reads
// the registry rather than a directory, and the creator's stubs are string
// constants rather than files, which is also why `aru make:migration` works from
// any working directory.
package migrations
