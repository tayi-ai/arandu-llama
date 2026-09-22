package database

import (
	"context"
	"fmt"
	"strings"
)

// Seeder is one unit of seeding.
//
// The shape is a DatabaseSeeder that calls the others, and `aru db:seed <Name>`
// to run one. A seeder is a type satisfying an interface rather than something
// discovered, so one that does not compile is caught at build time and not when
// somebody runs it against production.
//
// D is whatever the project decided a seeder is allowed to touch. It stays in
// the project (arandu/database/seeders) rather than being declared here, because
// "allowed to touch" is the application's answer and not the framework's: a
// seeder that can reach anything is a seeder nobody can review. What lives here
// is only the contract and the naming, which is what every project repeated.
type Seeder[D any] interface {
	// Name is how the seeder is addressed on the command line.
	Name() string
	// Run performs the seeding. It must be safe to run twice: a seeder that
	// fails on the second run cannot be part of a deploy.
	Run(ctx context.Context, d D) error
}

// Seed runs the seeder named in args, or fallback when none is, and returns the
// name of the one that ran.
//
//	aru db:seed                                    the fallback
//	aru db:seed PostSeeder                         one of them
//	aru db:seed UserSeeder -e a@b.com -p secret    one of them, with arguments
//
// The name is positional. A name that is sometimes a flag and sometimes a word
// is two spellings of one thing, so the `--class=` form is refused with the word
// to use instead rather than accepted quietly.
//
// Everything after the name reaches deps, unparsed, which is why deps is a
// function rather than a value: the project's Deps carries an Args field and
// this package does not know the field exists. What a flag means is the
// seeder's business -- this function does not know that UserSeeder has a -p, and
// adding a seeder does not edit this file. Flag and Switch below are what a
// seeder reads them with.
//
// It prints nothing. A library that writes to stdout is a library that cannot
// be called from a test, and the caller already knows which name to report.
func Seed[D any](ctx context.Context, registry []Seeder[D], fallback string, args []string, deps func(rest []string) D) (string, error) {
	name := fallback

	if len(args) > 0 {
		if value, ok := strings.CutPrefix(args[0], "--class="); ok {
			return "", fmt.Errorf("--class= is not how a seeder is named any more.\n\n    aru db:seed %s", value)
		}
		if !strings.HasPrefix(args[0], "-") {
			name, args = args[0], args[1:]
		}
	}

	seeder, err := lookup(registry, name)
	if err != nil {
		return "", err
	}
	if err := seeder.Run(ctx, deps(args)); err != nil {
		return seeder.Name(), fmt.Errorf("%s: %w", seeder.Name(), err)
	}
	return seeder.Name(), nil
}

// Flag reads `-name value` out of the arguments.
//
// A hand-written reader rather than the flag package, because the flag package
// wants to own os.Args, prints its own usage to stderr and calls os.Exit on a
// mistake -- in the middle of a command that has already opened a database.
//
// It accepts `-name value` and `-name=value`. A long form (`--name`) is the same
// flag: nobody should have to remember how many dashes a seeder wanted.
func Flag(args []string, name string) (string, bool) {
	short, long := "-"+name, "--"+name
	for i, a := range args {
		if a == short || a == long {
			if i+1 < len(args) && !strings.HasPrefix(args[i+1], "-") {
				return args[i+1], true
			}
			// Present with nothing after it. Reporting it as absent would turn a
			// typed-but-empty password into the demo fallback.
			return "", true
		}
		for _, prefix := range []string{short + "=", long + "="} {
			if value, ok := strings.CutPrefix(a, prefix); ok {
				return value, true
			}
		}
	}
	return "", false
}

// Switch reports whether a valueless flag is present.
func Switch(args []string, name string) bool {
	for _, a := range args {
		if a == "-"+name || a == "--"+name {
			return true
		}
	}
	return false
}

func lookup[D any](registry []Seeder[D], name string) (Seeder[D], error) {
	for _, s := range registry {
		if strings.EqualFold(s.Name(), name) {
			return s, nil
		}
	}

	available := make([]string, 0, len(registry))
	for _, s := range registry {
		available = append(available, s.Name())
	}
	return nil, fmt.Errorf("unknown seeder %q (available: %s)", name, strings.Join(available, ", "))
}
