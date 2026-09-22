package migrations

import (
	"context"
	"errors"
	"fmt"
)

// IsolationLock is one named lock, held for as long as a migration run takes.
//
// It is declared here rather than imported for the reason Dispatcher is: the
// Migrator takes a lock and never issues one, so what it needs is the idea of a
// lock rather than a lock package.
type IsolationLock interface {
	// Get takes the lock, runs fn if it took it, releases it afterwards, and
	// reports whether it took it.
	//
	// A lock somebody else holds is (false, nil) and not an error: another
	// process is doing the work, which is the expected answer to asking and not
	// a fault.
	Get(ctx context.Context, fn func(context.Context) error) (bool, error)
}

// IsolationLockName is the name of the lock an isolated run against connection
// takes.
//
// The connection is in the name, so two databases migrate at the same time and
// two processes pointed at one database do not. Pass the resolved name rather
// than the string a flag carried, or an unset --database and the name it
// resolves to become two locks over one schema.
//
// The name carries no tenant. A lock per tenant would let N replicas each
// migrate for a different tenant at once, which is the problem and not the
// solution, and a schema belongs to no single tenant anyway.
func IsolationLockName(connection string) string {
	return "framework/migrate-" + connection
}

// IsolateWith sets where RunIsolated gets its lock, and returns the Migrator so
// the call can be chained onto the constructor.
//
// The issuer is handed the lock's name and answers a lock nobody holds yet. How
// long that lock lives is the issuer's to decide, because it is the deadlock
// protection and nothing else: a process that dies mid-migration blocks every
// later run until its lock expires, so the duration is sized above the longest
// migration run there is rather than against the usual one.
//
//	migrator.IsolateWith(func(name string) migrations.IsolationLock {
//		return locks.Lock(name, time.Hour)
//	})
func (m *Migrator) IsolateWith(issue func(name string) IsolationLock) *Migrator {
	m.issueLock = issue
	return m
}

// RunIsolated applies what Run applies, and only while no other process is
// migrating the same connection.
//
// What the process that does not get the lock sees is the whole of the design:
// applied is empty, ran is false, and err is nil. It migrated nothing, and that
// is not a failure -- the schema is being changed by whoever got there first,
// and the caller's job is to carry on and let the application start. Reporting
// it as an error instead would fail every deployment that rolls more than one
// replica, which is the shape of deployment this exists to serve.
//
// So the answer to branch on is ran and never err. A run that took the lock
// answers what Run answered; a run that did not answers nothing, and both are
// success.
//
// A Migrator with no issuer refuses instead of migrating: a run that says it is
// isolated and is not is worse than one that does not start, because the
// failure it hides is two migrators racing over one table.
func (m *Migrator) RunIsolated(ctx context.Context, paths []string, options Options) (applied []string, ran bool, err error) {
	if m.issueLock == nil {
		return nil, false, errors.New("migrations: an isolated run needs a lock issuer and this migrator has none; " +
			"pass one to IsolateWith, or every replica of a rollout would migrate at once")
	}

	// The name is resolved rather than taken from options, so that the process
	// started without a connection and the one started with its name contend
	// over the same lock.
	conn, err := m.ResolveConnection("")
	if err != nil {
		return nil, false, err
	}
	name := IsolationLockName(conn.GetName())

	lock := m.issueLock(name)
	if lock == nil {
		return nil, false, fmt.Errorf("migrations: the lock issuer answered no lock for %q", name)
	}

	ran, err = lock.Get(ctx, func(ctx context.Context) error {
		var runErr error
		applied, runErr = m.Run(ctx, paths, options)
		return runErr
	})
	if err != nil {
		return nil, ran, err
	}
	if !ran {
		m.write(fmt.Sprintf("Another process is migrating %s. Nothing was applied here.", conn.GetName()))
	}
	return applied, ran, nil
}
