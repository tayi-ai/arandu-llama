package migrations

import (
	"context"
	"fmt"
	"io"
	"sort"
	"strings"
	"sync"

	"github.com/arandu-io/hesape/database/events"
)

// Dispatcher is where the Migrator sends the events of a run -- started,
// ended, one migration started, one skipped, nothing to do.
//
// It is one method because that is all the Migrator ever does with an event
// dispatcher: it publishes and never subscribes. Declaring the whole contract
// would make anything that wants to watch a migration implement listener
// registration it does not use, and would make this package depend on a
// dispatcher rather than on the idea of one. A Migrator with none dispatches
// nothing and migrates the same.
type Dispatcher interface {
	// Dispatch fires an event.
	Dispatch(event any)
}

// TransactionalConnection is a Connection that can wrap a migration in a
// transaction.
//
// Two things decide whether a migration is wrapped: the schema grammar's
// support for transactional DDL, asked for here, and the migration's own
// WithinTransaction flag. A connection that does not satisfy this interface
// runs its migrations unwrapped -- MySQL has no transactional DDL, so a
// failed migration there leaves half a schema whatever anybody wants.
type TransactionalConnection interface {
	Connection

	// SupportsSchemaTransactions reports whether the schema grammar supports
	// rolling DDL back.
	SupportsSchemaTransactions() bool

	// Transaction runs callback inside a transaction.
	Transaction(ctx context.Context, callback func() error) error
}

// Options is the set of flags the Migrator's methods take.
//
// Step serves two different purposes depending on direction: a bool on the
// way up (one batch per migration) and an int on the way down (how many to
// undo). Two fields is the same information with the ambiguity removed, and
// the ambiguity is worth removing -- `--step` on migrate and `--step=3` on
// rollback are not the same flag.
type Options struct {
	// Pretend prints the statements a run would execute, and runs none of
	// them.
	Pretend bool

	// Step gives every migration its own batch on the way up, so each can be
	// rolled back on its own.
	Step bool

	// Steps is how many migrations to roll back on the way down.
	Steps int

	// Batch rolls back one named batch.
	Batch int
}

// Migrator runs migrations up and down and keeps the repository in step with the
// schema.
//
// # It does not run at boot, and this is where that is enforced
//
// `aru migrate` is a step of the deployment pipeline, never a call in the
// start-up path of the process. With N replicas rolling, calling Run from
// main means N migrators racing each other over the same table, and the one
// that loses reports a duplicate key on a table it was creating. There is no
// Migrate-on-boot helper here to make that easy, and there will not be one.
//
// A pipeline that cannot promise it runs the step once uses RunIsolated, which
// takes a lock named after the connection and lets the process that does not
// get it finish successfully having migrated nothing.
//
// The other half of the same rule is the migration's own: every migration is
// compatible with the binary that is still running while the rollout finishes.
// A new column is nullable or has a default; removing one takes two releases,
// the first stopping the writes and the second dropping the column.
type Migrator struct {
	// configurationMu keeps configuration changes on this Migrator from racing
	// with a run of the same Migrator. It does not coordinate different
	// Migrators: each instance owns the values it protects.
	configurationMu sync.RWMutex

	// events is where the Migrator dispatches its events.
	events Dispatcher

	// repository is where the Migrator records which migrations have run.
	repository MigrationRepositoryInterface

	// resolver is how the Migrator reaches a connection by name.
	resolver Resolver

	// resolveConnection is an optional instance-specific connection resolver.
	resolveConnection func(resolver Resolver, name string) (Connection, error)

	// skippedMigrations holds names this Migrator leaves pending.
	skippedMigrations []string

	// connection is the default connection name.
	connection string

	// paths is the migration groups the Migrator looks in.
	paths []string

	// output is where progress is written. Nil writes nothing.
	output io.Writer

	// issueLock is where RunIsolated gets its lock, and nil is what makes it
	// refuse rather than migrate unprotected. IsolateWith sets it.
	issueLock func(name string) IsolationLock
}

// NewMigrator creates a Migrator.
//
// There is no filesystem argument: a migration is code, and the registry
// replaced the glob. See Register for the whole of that decision.
func NewMigrator(repository MigrationRepositoryInterface, resolver Resolver, dispatcher Dispatcher) *Migrator {
	return &Migrator{repository: repository, resolver: resolver, events: dispatcher}
}

// Run applies everything that has not been applied yet, and returns the
// names of what it applied.
func (m *Migrator) Run(ctx context.Context, paths []string, options Options) ([]string, error) {
	files := m.GetMigrationFiles(paths)

	ran, err := m.repository.GetRan(ctx)
	if err != nil {
		return nil, err
	}

	pending := m.pendingMigrations(files, ran)

	if err := m.RunPending(ctx, pending, options); err != nil {
		return nil, err
	}

	names := make([]string, 0, len(pending))
	for _, migration := range pending {
		names = append(names, migration.GetName())
	}
	return names, nil
}

// pendingMigrations returns every registered migration that has not run yet
// and is not in the skip list, sorted by name.
func (m *Migrator) pendingMigrations(files map[string]Migration, ran []string) []Migration {
	done := make(map[string]bool, len(ran))
	for _, name := range ran {
		done[name] = true
	}
	for _, name := range m.migrationsToSkip() {
		done[name] = true
	}

	names := make([]string, 0, len(files))
	for name := range files {
		if !done[name] {
			names = append(names, name)
		}
	}
	sort.Strings(names)

	out := make([]Migration, 0, len(names))
	for _, name := range names {
		out = append(out, files[name])
	}
	return out
}

// migrationsToSkip returns the names this Migrator was told to leave pending,
// resolved to their migration names.
func (m *Migrator) migrationsToSkip() []string {
	m.configurationMu.RLock()
	names := append([]string(nil), m.skippedMigrations...)
	m.configurationMu.RUnlock()

	out := make([]string, 0, len(names))
	for _, name := range names {
		out = append(out, m.GetMigrationName(name))
	}
	return out
}

// RunPending applies the given migrations, in the order they arrive.
//
// It stops at the first failure. Applying later migrations over a schema that a
// failed one left half-changed turns one clear error into a database nobody can
// get back.
func (m *Migrator) RunPending(ctx context.Context, migrations []Migration, options Options) error {
	if len(migrations) == 0 {
		m.FireMigrationEvent(events.NewNoPendingMigrations("up"))
		m.write("Nothing to migrate")
		return nil
	}

	batch, err := m.repository.GetNextBatchNumber(ctx)
	if err != nil {
		return err
	}

	m.FireMigrationEvent(events.NewMigrationsStarted("up", optionsMap(options)))
	m.write("Running migrations.")

	for _, migration := range migrations {
		if err := m.runUp(ctx, migration, batch, options.Pretend); err != nil {
			return err
		}
		if options.Step {
			batch++
		}
	}

	m.FireMigrationEvent(events.NewMigrationsEnded("up", optionsMap(options)))
	m.write("")

	return nil
}

// runUp runs one migration's Up, records it in the given batch, and reports
// its result.
func (m *Migrator) runUp(ctx context.Context, migration Migration, batch int, pretend bool) error {
	name := migration.GetName()

	if pretend {
		return m.pretendToRun(ctx, migration, "up")
	}

	if !migration.ShouldRun() {
		m.FireMigrationEvent(events.NewMigrationSkipped(name))
		m.write(fmt.Sprintf("%s %s", name, Skipped))
		return nil
	}

	if err := m.runMigration(ctx, migration, "up"); err != nil {
		m.write(fmt.Sprintf("%s %s", name, Failure))
		return fmt.Errorf("migration %s failed: %w", name, err)
	}

	// Recorded only after it succeeded. A name in the table for a migration
	// that did not finish is the one state a migrator cannot recover from.
	if err := m.repository.Log(ctx, name, batch); err != nil {
		return err
	}

	m.write(fmt.Sprintf("%s %s", name, Success))
	return nil
}

// Rollback undoes the last batch, or the batch or step count options names.
//
// It undoes all of the batch or none of it: if a migration in it declares
// neither Down nor Irreversible, the batch is refused before the first Down
// runs. Migrations declaring Irreversible are named up front and left applied.
//
// The names it answers are the migrations it undid, and nothing else: one left
// applied because it declares Irreversible, and one whose code is no longer
// registered, are both absent from them. The caller prints a line per name, so a
// name here is a claim that the change is gone and its record with it. Under
// Pretend the names are what a real run would undo, which is what pretending
// answers.
func (m *Migrator) Rollback(ctx context.Context, paths []string, options Options) ([]string, error) {
	records, err := m.getMigrationsForRollback(ctx, options)
	if err != nil {
		return nil, err
	}

	if len(records) == 0 {
		m.FireMigrationEvent(events.NewNoPendingMigrations("down"))
		m.write("Nothing to rollback.")
		return []string{}, nil
	}

	rolledBack, err := m.rollbackMigrations(ctx, records, paths, options)
	m.write("")
	return rolledBack, err
}

// getMigrationsForRollback returns the migration records a rollback should
// undo, chosen by options.Steps, options.Batch, or the last batch when
// neither is set.
func (m *Migrator) getMigrationsForRollback(ctx context.Context, options Options) ([]MigrationRecord, error) {
	if options.Steps > 0 {
		return m.repository.GetMigrations(ctx, options.Steps)
	}
	if options.Batch > 0 {
		return m.repository.GetMigrationsByBatch(ctx, options.Batch)
	}
	return m.repository.GetLast(ctx)
}

// rollbackMigrations runs Down for each recorded migration, newest first.
//
// The batch is read before any of it is undone, and a migration that cannot be
// rolled back refuses the whole batch there -- see checkBatchCanRollBack.
//
// A recorded migration whose code is no longer registered is reported and
// skipped rather than stopping the rollback: the alternative is a rollback
// that refuses to start because of one file somebody deleted six releases
// ago.
//
// A Down that fails at run time still stops the loop. That one cannot be read
// ahead of time, so the ones already undone stay undone and their records are
// already gone.
//
// It collects a name only once runDown reports having undone that migration.
// Collecting on reaching one instead put every record in the answer, including
// the ones runDown went on to skip -- and the answer is what the command prints
// its table from, so a migration left applied was announced as reverted on the
// line after the one saying it was skipped.
func (m *Migrator) rollbackMigrations(ctx context.Context, records []MigrationRecord, paths []string, options Options) ([]string, error) {
	var rolledBack []string

	files := m.GetMigrationFiles(paths)

	if err := m.checkBatchCanRollBack(records, files); err != nil {
		return nil, err
	}

	m.FireMigrationEvent(events.NewMigrationsStarted("down", optionsMap(options)))
	m.write("Rolling back migrations.")

	for _, record := range records {
		migration, known := files[record.Migration]
		if !known {
			m.write(fmt.Sprintf("%s Migration not found", record.Migration))
			continue
		}

		undone, err := m.runDown(ctx, migration, record, options.Pretend)
		if undone {
			rolledBack = append(rolledBack, record.Migration)
		}
		if err != nil {
			return rolledBack, err
		}
	}

	m.FireMigrationEvent(events.NewMigrationsEnded("down", optionsMap(options)))

	return rolledBack, nil
}

// Reset rolls every applied migration back, newest first.
//
// Every one that can be: a migration declaring itself irreversible is left
// applied and keeps its record, so a reset is not a guarantee of an empty
// schema. One that declares neither a Down nor Irreversible refuses the reset
// before any of it is undone.
//
// The names it answers are the migrations it undid, on the same terms as
// Rollback: the one left applied is not among them, which is how a caller counts
// what is still there.
func (m *Migrator) Reset(ctx context.Context, paths []string, pretend bool) ([]string, error) {
	ran, err := m.repository.GetRan(ctx)
	if err != nil {
		return nil, err
	}
	if len(ran) == 0 {
		m.write("Nothing to rollback.")
		return []string{}, nil
	}

	records := make([]MigrationRecord, 0, len(ran))
	for _, name := range ran {
		records = append(records, MigrationRecord{Migration: name})
	}
	// GetRan answers oldest first, and a reset undoes them the other way.
	sortRecordsByName(records, true)

	rolledBack, err := m.rollbackMigrations(ctx, records, paths, Options{Pretend: pretend})
	m.write("")
	return rolledBack, err
}

// checkBatchCanRollBack reads the whole batch before any of it is undone, and
// refuses it if a migration in it cannot be rolled back.
//
// Without this the rollback learned of such a migration by reaching it. It
// undoes newest first, so everything applied after it was already undone and
// its records already deleted -- the command stopped in the middle, which is
// the one state a migrator has no way back from. Half a batch is not a version
// anybody deployed. Refusing here costs nothing to act on, because nothing has
// happened yet.
//
// A migration that declares Irreversible does not refuse the batch: it is
// skipped and left applied, which the rollback survives. It is named in what
// this prints, because "this batch will not come back whole" is worth knowing
// before the first Down runs.
//
// A recorded migration whose code is no longer registered is passed over here
// exactly as the loop passes over it, so nothing is refused that the rollback
// would have survived.
func (m *Migrator) checkBatchCanRollBack(records []MigrationRecord, files map[string]Migration) error {
	var leftApplied []string

	for _, record := range records {
		migration, known := files[record.Migration]
		if !known {
			continue
		}

		if err := rollbackRefusal(migration); err != nil {
			return fmt.Errorf("this batch would stop partway, so nothing was rolled back: %w", err)
		}

		if irreversible, is := migration.(IrreversibleMigration); is {
			leftApplied = append(leftApplied,
				fmt.Sprintf("%s will be left applied: %s", migration.GetName(), irreversible.Irreversible()))
		}
	}

	for _, line := range leftApplied {
		m.write(line)
	}

	return nil
}

// rollbackRefusal reports why migration cannot be rolled back, or nil when it
// can.
//
// A migration that reverses nothing has to say which of two things it is. One
// that declares Irreversible can take part in a rollback -- it is skipped and
// left applied, with its record and its reason. One that declares neither
// cannot, and neither can one declaring both, because the two say opposite
// things about the same migration and nothing outside it can tell which its
// author meant.
//
// It is one function because checkBatchCanRollBack and runDown have to agree.
// A second copy of these two cases would drift, and what drift produces here is
// a batch the preflight cleared and runDown then refused halfway through --
// which is the failure the preflight exists to prevent.
func rollbackRefusal(migration Migration) error {
	name := migration.GetName()

	_, isReversible := migration.(ReversibleMigration)
	_, isIrreversible := migration.(IrreversibleMigration)

	switch {
	case isReversible && isIrreversible:
		return fmt.Errorf("migration %s declares both Down and Irreversible, "+
			"and nothing outside it can tell which one is true: keep the one that is", name)

	case !isReversible && !isIrreversible:
		return fmt.Errorf("migration %s cannot be rolled back: it declares neither Down nor Irreversible. "+
			"Write Down to undo what Up applied, or Irreversible to declare that nothing can", name)
	}

	return nil
}

// runDown runs one migration's Down, removes its record, and reports whether
// the migration was undone.
//
// The record is deleted only when the change was actually undone, and that is
// the whole of the rule. A migration with no Down used to reach the delete
// anyway: the schema stayed, the row saying so was removed, the line printed
// said Success, and the next migrate ran the Up again and failed on what was
// already there -- two commands away from the thing that caused it.
//
// The bool is that same rule made readable by the caller. Several paths out of
// here end with the migration still recorded as applied -- a refusal, one
// declaring Irreversible, a Down that failed, a delete that did -- and a nil
// error is not enough to tell them apart from the one that undid it, because a
// skip is not a failure and must not stop the batch. Under pretend it reports
// true: nothing ran, but the migration is one a real run would undo, and that is
// the question pretending is asked.
//
// It refuses the same migrations checkBatchCanRollBack refuses, through the
// same function. That is not a second gate but the same one: every batch is
// read first, and this is what answers for a migration reached any other way.
func (m *Migrator) runDown(ctx context.Context, migration Migration, record MigrationRecord, pretend bool) (bool, error) {
	name := migration.GetName()

	if err := rollbackRefusal(migration); err != nil {
		return false, err
	}

	if irreversible, isIrreversible := migration.(IrreversibleMigration); isIrreversible {
		m.FireMigrationEvent(events.NewMigrationSkipped(name))
		m.write(fmt.Sprintf("%s %s: %s", name, Skipped, irreversible.Irreversible()))
		return false, nil
	}

	if pretend {
		err := m.pretendToRun(ctx, migration, "down")
		return err == nil, err
	}

	if err := m.runMigration(ctx, migration, "down"); err != nil {
		m.write(fmt.Sprintf("%s %s", name, Failure))
		return false, fmt.Errorf("rollback of %s failed: %w", name, err)
	}

	if err := m.repository.Delete(ctx, record); err != nil {
		// The change is undone and the row saying it is applied is still there.
		// Reporting it reverted would put the name in a table the operator reads
		// as "this is gone and forgotten", when the next migrate will not run it
		// again. The error carries what happened; the name does not.
		return false, err
	}

	m.write(fmt.Sprintf("%s %s", name, Success))
	return true, nil
}

// runMigration runs one direction of one migration, inside a transaction
// when the engine and the migration both allow it.
func (m *Migrator) runMigration(ctx context.Context, migration Migration, method string) error {
	conn, err := m.ResolveConnection(migration.GetConnection())
	if err != nil {
		return err
	}

	callback := func() error {
		reversible, isReversible := migration.(ReversibleMigration)
		if method == "down" && !isReversible {
			// runDown decides what a migration with no Down means, and every
			// rollback goes through it. Reaching here is a caller that skipped
			// that decision, so it says so rather than returning nil -- which
			// is the shape that printed Success over a schema nobody touched.
			return fmt.Errorf("migration %s has no Down", migration.GetName())
		}

		m.FireMigrationEvent(events.NewMigrationStarted(migration, method))

		var err error
		if method == "down" {
			err = reversible.Down(ctx, conn)
		} else {
			err = migration.Up(ctx, conn)
		}
		if err != nil {
			return err
		}

		m.FireMigrationEvent(events.NewMigrationEnded(migration, method))
		return nil
	}

	transactional, ok := conn.(TransactionalConnection)
	if ok && transactional.SupportsSchemaTransactions() && migration.WithinTransaction() {
		return transactional.Transaction(ctx, callback)
	}
	return callback()
}

// pretendToRun runs the migration against the connection in a mode that
// collects its statements without executing them, and prints them.
//
// A connection that implements PretendingConnection returns the statements
// without executing them, and one that does not says so rather than
// pretending to pretend.
func (m *Migrator) pretendToRun(ctx context.Context, migration Migration, method string) error {
	name := migration.GetName()
	m.write(name)

	conn, err := m.ResolveConnection(migration.GetConnection())
	if err != nil {
		return err
	}

	pretender, ok := conn.(PretendingConnection)
	if !ok {
		m.write("  this connection cannot pretend, so nothing was printed and nothing ran")
		return nil
	}

	queries, err := pretender.Pretend(ctx, func() error {
		if method == "down" {
			if reversible, isReversible := migration.(ReversibleMigration); isReversible {
				return reversible.Down(ctx, conn)
			}
			return nil
		}
		return migration.Up(ctx, conn)
	})
	if err != nil {
		return err
	}

	for _, query := range queries {
		m.write("  " + query)
	}
	return nil
}

// PretendingConnection is a Connection that can run a callback without letting
// any of its statements reach the server.
//
// Pretend returns the query log rather than the callback's result.
type PretendingConnection interface {
	Connection

	// Pretend runs callback and returns the statements it would have run,
	// without executing them.
	Pretend(ctx context.Context, callback func() error) ([]string, error)
}

// Resolve returns the migration registered under name.
//
// The registry already holds the instance, because a Go migration is
// registered rather than discovered -- so this is a lookup, and an unknown
// name is an error rather than a construction failure.
func (m *Migrator) Resolve(name string) (Migration, error) {
	for _, migration := range Registered(m.paths...) {
		if migration.GetName() == name {
			return migration, nil
		}
	}
	return nil, fmt.Errorf("no migration is registered under %q", name)
}

// GetMigrationFiles answers every migration of the given paths, keyed by name.
//
// There is nothing on disk to glob -- see Register -- so this reads the registry
// and keys by GetName.
func (m *Migrator) GetMigrationFiles(paths []string) map[string]Migration {
	groups := append(append([]string(nil), m.paths...), paths...)

	out := map[string]Migration{}
	for _, migration := range Registered(groups...) {
		out[migration.GetName()] = migration
	}
	return out
}

// RequireFiles has no counterpart, and its absence is the point.
//
// A migration file needs a separate load step only when something has to
// read it off disk to make it exist as code. Go has no such step: the import
// in main.go put every registered migration in the binary before it started.

// GetMigrationName returns the name of a migration, given either the name
// itself or a path that ends in it.
//
// It still takes a path-shaped string because `aru migrate --without=` and the
// squashed-schema paths hand it one, and because a person copying a file name
// out of a log should get the right answer.
func (m *Migrator) GetMigrationName(path string) string {
	name := path
	if i := strings.LastIndexAny(name, `/\`); i >= 0 {
		name = name[i+1:]
	}
	name = strings.TrimSuffix(name, ".go")
	return strings.TrimSuffix(name, ".php")
}

// Path adds a group the Migrator should look in.
func (m *Migrator) Path(path string) {
	for _, existing := range m.paths {
		if existing == path {
			return
		}
	}
	m.paths = append(m.paths, path)
}

// Paths returns the groups the Migrator looks in.
func (m *Migrator) Paths() []string { return m.paths }

// WithoutMigrations sets names this Migrator leaves pending however many times
// it runs.
//
// The names are copied so changing the caller's slice cannot reconfigure a run
// that is already being prepared.
func (m *Migrator) WithoutMigrations(names []string) {
	m.configurationMu.Lock()
	defer m.configurationMu.Unlock()
	m.skippedMigrations = append([]string(nil), names...)
}

// GetConnection returns the default connection name.
func (m *Migrator) GetConnection() string { return m.connection }

// UsingConnection runs callback with a different default connection, and
// puts the old one back afterward.
func (m *Migrator) UsingConnection(name string, callback func() error) error {
	previous := m.resolver.GetDefaultConnection()

	m.SetConnection(name)
	defer m.SetConnection(previous)

	return callback()
}

// SetConnection replaces the default connection name.
func (m *Migrator) SetConnection(name string) {
	if name != "" {
		m.resolver.SetDefaultConnection(name)
	}
	m.repository.SetSource(name)
	m.connection = name
}

// ResolveConnection returns the named connection, or the default connection
// when connection is empty, through this Migrator's resolver callback when one
// was set.
func (m *Migrator) ResolveConnection(connection string) (Connection, error) {
	m.configurationMu.RLock()
	callback := m.resolveConnection
	m.configurationMu.RUnlock()

	name := connection
	if name == "" {
		name = m.connection
	}

	if callback != nil {
		return callback(m.resolver, name)
	}
	return m.resolver.Connection(name)
}

// ResolveConnectionsUsing sets the callback this Migrator's ResolveConnection
// uses to resolve a connection.
func (m *Migrator) ResolveConnectionsUsing(callback func(resolver Resolver, name string) (Connection, error)) {
	m.configurationMu.Lock()
	defer m.configurationMu.Unlock()
	m.resolveConnection = callback
}

// GetRepository returns the repository migrations are recorded in.
func (m *Migrator) GetRepository() MigrationRepositoryInterface { return m.repository }

// RepositoryExists reports whether the migration repository's table exists.
func (m *Migrator) RepositoryExists(ctx context.Context) bool {
	return m.repository.RepositoryExists(ctx)
}

// HasRunAnyMigrations reports whether the repository exists and has at least
// one migration recorded.
func (m *Migrator) HasRunAnyMigrations(ctx context.Context) bool {
	if !m.RepositoryExists(ctx) {
		return false
	}
	ran, err := m.repository.GetRan(ctx)
	return err == nil && len(ran) > 0
}

// DeleteRepository drops the migration repository's table.
func (m *Migrator) DeleteRepository(ctx context.Context) error {
	return m.repository.DeleteRepository(ctx)
}

// SetOutput sets where progress is written.
//
// It is a plain io.Writer rather than something that renders console
// components, because a library that draws a table is a library a test
// cannot read.
func (m *Migrator) SetOutput(output io.Writer) *Migrator {
	m.output = output
	return m
}

// FireMigrationEvent dispatches event, if a dispatcher was given.
func (m *Migrator) FireMigrationEvent(event any) {
	if m.events != nil {
		m.events.Dispatch(event)
	}
}

// write appends line to the output, if one was set.
func (m *Migrator) write(line string) {
	if m.output == nil {
		return
	}
	_, _ = io.WriteString(m.output, line+"\n")
}

// optionsMap shapes Options back into the map the events carry.
func optionsMap(options Options) map[string]any {
	return map[string]any{
		"pretend": options.Pretend,
		"step":    options.Step,
		"steps":   options.Steps,
		"batch":   options.Batch,
	}
}
