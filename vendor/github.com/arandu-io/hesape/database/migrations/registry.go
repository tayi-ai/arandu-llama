package migrations

import (
	"fmt"
	"sort"
	"sync"
)

// DefaultPath is the group a migration lands in when it registers without
// naming one.
//
// It is spelled like a path so that `aru migrate --path=...` takes something a
// person recognises. Nothing opens it: it is a key.
const DefaultPath = "database/migrations"

// registry is where Register puts them, keyed by group and then by name.
//
// The mutex is not decoration even though init() runs single-threaded: Register
// is exported, and a test that registers a migration from a goroutine would
// otherwise race the read side under -race.
var (
	registryMu sync.RWMutex
	registry   = map[string]map[string]Migration{}
)

// Register records a migration so the Migrator can find it.
//
// # Why discovery is a registry and not a directory scan
//
// A package that nothing imports is not in the binary at all, so a directory
// scan at run time would find files the compiler never saw -- and, worse, would
// find nothing at all in a deployed binary, where the source directory does not
// exist. `aru migrate` running against a scratch container with no repository
// checked out is the normal case, not the exotic one.
//
// The two candidates were an embed of the directory and a registry filled from
// init(). Embedding would mean the migration is data -- SQL text -- which reads
// well until the first migration that has to backfill a column by reading rows,
// and then there are two kinds of migration. So:
//
//	func init() { migrations.Register(CreateUsersTable{}) }
//
// and main.go blank-imports the package the migrations live in. That import is
// the loading step, moved to link time -- with the compiler checking, before
// anything runs, that every registered migration actually implements Migration.
//
// Registering the same name twice panics rather than picking one. Two
// migrations under one name is a copied file somebody forgot to rename, and it
// would apply one of them and record the other.
func Register(migration Migration, path ...string) {
	group := DefaultPath
	if len(path) > 0 && path[0] != "" {
		group = path[0]
	}

	name := migration.GetName()
	if name == "" {
		panic("migrations: a migration registered with an empty GetName -- the name carries the order, so it cannot be blank")
	}

	registryMu.Lock()
	defer registryMu.Unlock()

	if registry[group] == nil {
		registry[group] = map[string]Migration{}
	}
	if existing, taken := registry[group][name]; taken && existing != migration {
		panic(fmt.Sprintf("migrations: %q is registered twice under %q -- one of them is a copied file that kept its neighbour's name", name, group))
	}
	registry[group][name] = migration
}

// Registered returns the migrations of the given groups, sorted by name.
//
// It reads the registry rather than a directory, keyed by migration name and
// sorted by that key. No group named means every group.
func Registered(paths ...string) []Migration {
	registryMu.RLock()
	defer registryMu.RUnlock()

	groups := paths
	if len(groups) == 0 {
		for group := range registry {
			groups = append(groups, group)
		}
	}

	seen := map[string]Migration{}
	for _, group := range groups {
		for name, migration := range registry[group] {
			seen[name] = migration
		}
	}

	names := make([]string, 0, len(seen))
	for name := range seen {
		names = append(names, name)
	}
	// One order, taken from the name, on every machine. It is the reason the
	// date prefix exists, and the reason nothing else may decide it.
	sort.Strings(names)

	out := make([]Migration, 0, len(names))
	for _, name := range names {
		out = append(out, seen[name])
	}
	return out
}

// RegisteredPaths answers the groups something has registered under, sorted.
//
// `aru migrate --path=` reads it to say what the choices are, which is the
// difference between a usable error and "no migrations found".
func RegisteredPaths() []string {
	registryMu.RLock()
	defer registryMu.RUnlock()

	out := make([]string, 0, len(registry))
	for group := range registry {
		out = append(out, group)
	}
	sort.Strings(out)
	return out
}

// flushRegistry empties the registry. It exists for this package's own tests,
// which register migrations and must not leak them into the next test.
func flushRegistry() {
	registryMu.Lock()
	defer registryMu.Unlock()
	registry = map[string]map[string]Migration{}
}
