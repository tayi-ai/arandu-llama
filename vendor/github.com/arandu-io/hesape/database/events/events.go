package events

// Connection is what an event needs of the connection it carries.
//
// It is narrowed to the one method every event in this package calls on it:
// the name of the connection.
//
// It is declared here rather than imported from the database package because
// database dispatches these events, so naming the concrete type here would
// close an import cycle -- the same reason query.Connection is declared in the
// query package.
type Connection interface {
	// GetName returns the connection's name.
	GetName() string
}

// RawSQLConnection is what QueryExecuted.ToRawSQL asks of its connection.
//
// Reaching the grammar's bindings-substitution method through the
// connection, its query builder and the grammar would be three hops through
// concrete types, which is a cycle here, so the destination is named
// directly.
type RawSQLConnection interface {
	Connection

	// PrepareBindings converts bindings into the values a driver accepts.
	PrepareBindings(bindings []any) []any

	// SubstituteBindingsIntoRawSQL writes bindings directly into sql.
	SubstituteBindingsIntoRawSQL(sql string, bindings []any) string
}

// ConnectionEvent is the state the four connection events share.
//
// It is a struct they embed, so the fields below are reachable on every one of
// them, spelled the same way.
type ConnectionEvent struct {
	// ConnectionName is the name of the connection the event fired on.
	ConnectionName string

	// Connection is the connection the event fired on.
	Connection Connection
}

// NewConnectionEvent creates a ConnectionEvent, reading the name off the
// connection rather than taking it as a separate argument.
func NewConnectionEvent(connection Connection) ConnectionEvent {
	name := ""
	if connection != nil {
		name = connection.GetName()
	}
	return ConnectionEvent{Connection: connection, ConnectionName: name}
}

// ConnectionEstablished is dispatched by the DatabaseManager the first time a
// named connection is built, and again when one is reconnected.
//
// It fires once per connection rather than once per query, which makes it the
// place to configure a connection the moment it exists: set a session variable,
// register a query listener, record that the pool grew. A listener that runs
// per query wants QueryExecuted instead.
type ConnectionEstablished struct{ ConnectionEvent }

// NewConnectionEstablished creates a ConnectionEstablished event.
func NewConnectionEstablished(connection Connection) *ConnectionEstablished {
	return &ConnectionEstablished{NewConnectionEvent(connection)}
}

// TransactionBeginning is dispatched after a transaction level opens on a
// connection.
//
// Every level fires it, not only the outermost: a nested transaction is a
// savepoint, and the savepoint counts. A listener that only wants the real
// transaction has to read the connection's level itself, because the event
// carries nothing but the connection. It fires after the statement succeeded,
// so a listener seeing it knows the transaction is open.
type TransactionBeginning struct{ ConnectionEvent }

// NewTransactionBeginning creates a TransactionBeginning event.
func NewTransactionBeginning(connection Connection) *TransactionBeginning {
	return &TransactionBeginning{NewConnectionEvent(connection)}
}

// TransactionCommitting is dispatched as the outermost commit is about to be
// sent.
//
// It fires before the commit reaches the server, which is the only moment a
// listener can still refuse one.
type TransactionCommitting struct{ ConnectionEvent }

// NewTransactionCommitting creates a TransactionCommitting event.
func NewTransactionCommitting(connection Connection) *TransactionCommitting {
	return &TransactionCommitting{NewConnectionEvent(connection)}
}

// TransactionCommitted is dispatched after a transaction level closes
// successfully.
//
// Every level fires it, so releasing a savepoint fires it as well as the
// outermost commit. Only the outermost one has actually reached the server --
// TransactionCommitting is the event that fires exactly once for that, just
// before it does. Work that must not happen unless the data is durable belongs
// after the outermost commit, which means this event alone is not enough to
// decide it.
type TransactionCommitted struct{ ConnectionEvent }

// NewTransactionCommitted creates a TransactionCommitted event.
func NewTransactionCommitted(connection Connection) *TransactionCommitted {
	return &TransactionCommitted{NewConnectionEvent(connection)}
}

// TransactionRolledBack is dispatched after a rollback has been carried out on
// a connection.
//
// A rollback to a savepoint fires it too, so it does not on its own mean the
// whole transaction was abandoned -- only that the connection is back at some
// earlier level. It is the signal for undoing what was done outside the
// database on the strength of a write that is now gone: a cache entry primed
// early, a counter held in memory.
type TransactionRolledBack struct{ ConnectionEvent }

// NewTransactionRolledBack creates a TransactionRolledBack event.
func NewTransactionRolledBack(connection Connection) *TransactionRolledBack {
	return &TransactionRolledBack{NewConnectionEvent(connection)}
}

// QueryExecuted is dispatched after a statement has run on a connection.
//
// Time is the duration in milliseconds, rounded to two decimals. It is a float rather
// than a time.Duration because the field is read by listeners that compare it
// against a threshold in milliseconds, and changing the unit would make every
// one of those comparisons silently wrong.
type QueryExecuted struct {
	// SQL is the query as it was issued.
	SQL string

	// Bindings are the values that went with it.
	Bindings []any

	// Time is how long it took, in milliseconds.
	Time float64

	// Connection is the connection the query ran on.
	Connection RawSQLConnection

	// ConnectionName is the name of the connection the query ran on.
	ConnectionName string

	// ReadWriteType is "read", "write", or empty.
	ReadWriteType string
}

// NewQueryExecuted creates a QueryExecuted event.
func NewQueryExecuted(sql string, bindings []any, time float64, connection RawSQLConnection, readWriteType string) *QueryExecuted {
	name := ""
	if connection != nil {
		name = connection.GetName()
	}
	return &QueryExecuted{
		SQL:            sql,
		Bindings:       bindings,
		Time:           time,
		Connection:     connection,
		ConnectionName: name,
		ReadWriteType:  readWriteType,
	}
}

// ToRawSQL returns the query with its bindings written directly into it, for
// a log a person reads rather than one a machine parses.
//
// A nil connection returns the query unchanged rather than panicking,
// because an event constructed in a test has nowhere to ask.
func (e *QueryExecuted) ToRawSQL() string {
	if e.Connection == nil {
		return e.SQL
	}
	return e.Connection.SubstituteBindingsIntoRawSQL(e.SQL, e.Connection.PrepareBindings(e.Bindings))
}

// StatementPrepared is dispatched once the driver has prepared a statement.
//
// Statement is the prepared statement the driver handed back. It is typed any
// because database/sql hands back a *sql.Stmt and
// naming it would put the standard library's driver types into an event package
// that has no other reason to know about them.
type StatementPrepared struct {
	// Connection is the connection the statement was prepared on.
	Connection Connection

	// Statement is the prepared statement the driver handed back.
	Statement any
}

// NewStatementPrepared creates a StatementPrepared event.
func NewStatementPrepared(connection Connection, statement any) *StatementPrepared {
	return &StatementPrepared{Connection: connection, Statement: statement}
}

// DatabaseBusy is what `db:monitor` dispatches when a connection is over its
// threshold.
type DatabaseBusy struct {
	// ConnectionName is the name of the connection that is over its
	// threshold.
	ConnectionName string

	// Connections is how many connections are open.
	Connections int
}

// NewDatabaseBusy creates a DatabaseBusy event.
func NewDatabaseBusy(connectionName string, connections int) *DatabaseBusy {
	return &DatabaseBusy{ConnectionName: connectionName, Connections: connections}
}

// DatabaseRefreshed is what `migrate:fresh` and `migrate:refresh` dispatch once
// the schema is back.
type DatabaseRefreshed struct {
	// Database is the name of the database that was refreshed, empty when
	// none was given.
	Database string

	// Seeding is whether the refresh reseeded the database.
	Seeding bool
}

// NewDatabaseRefreshed creates a DatabaseRefreshed event.
func NewDatabaseRefreshed(database string, seeding bool) *DatabaseRefreshed {
	return &DatabaseRefreshed{Database: database, Seeding: seeding}
}

// Migration is what a migration event carries.
//
// It is narrowed to what the events read. The migrations package cannot be
// imported here: the Migrator
// dispatches these, so it imports this package.
type Migration interface {
	// GetConnection returns the name of the connection the migration runs on.
	GetConnection() string
}

// MigrationEvent is the pair of fields every single-migration event carries:
// which migration, and which direction it is being run in.
//
// It is embedded by MigrationStarted and MigrationEnded rather than inherited,
// so both read the same two fields under the same names. Nothing dispatches
// this on its own -- it is only ever the embedded half of one of those.
type MigrationEvent struct {
	// Migration is the migration the event is about.
	Migration Migration

	// Method is "up" or "down".
	Method string
}

// NewMigrationEvent creates a MigrationEvent.
func NewMigrationEvent(migration Migration, method string) MigrationEvent {
	return MigrationEvent{Migration: migration, Method: method}
}

// MigrationStarted is dispatched by the Migrator immediately before one
// migration's Up or Down is called.
//
// It fires inside the transaction the migration runs in, where the engine has
// transactional DDL, so a listener writing to the same connection is writing
// inside that transaction. A migration skipped by ShouldRun does not fire it;
// MigrationSkipped does.
type MigrationStarted struct{ MigrationEvent }

// NewMigrationStarted creates a MigrationStarted event.
func NewMigrationStarted(migration Migration, method string) *MigrationStarted {
	return &MigrationStarted{NewMigrationEvent(migration, method)}
}

// MigrationEnded is dispatched after one migration's Up or Down returned
// without an error.
//
// A migration that failed does not fire it -- the error stops the run before
// this point -- so seeing MigrationStarted with no MigrationEnded for the same
// name is exactly the signature of a migration that broke. It fires before the
// transaction commits, so the schema change is not durable yet.
type MigrationEnded struct{ MigrationEvent }

// NewMigrationEnded creates a MigrationEnded event.
func NewMigrationEnded(migration Migration, method string) *MigrationEnded {
	return &MigrationEnded{NewMigrationEvent(migration, method)}
}

// MigrationSkipped is what the Migrator dispatches when a migration's ShouldRun
// answers false.
type MigrationSkipped struct {
	// MigrationName is the name of the migration that was skipped.
	MigrationName string
}

// NewMigrationSkipped creates a MigrationSkipped event.
func NewMigrationSkipped(migrationName string) *MigrationSkipped {
	return &MigrationSkipped{MigrationName: migrationName}
}

// MigrationsEvent is the pair of fields the whole-run events carry: the
// direction, and the options the run was given.
//
// It is embedded by MigrationsStarted and MigrationsEnded, which bracket a
// whole `aru migrate` or `aru migrate:rollback` -- the plural to
// MigrationEvent's singular. Nothing dispatches this on its own.
type MigrationsEvent struct {
	// Method is "up" or "down".
	Method string

	// Options is the options the run was given.
	Options map[string]any
}

// NewMigrationsEvent creates a MigrationsEvent.
func NewMigrationsEvent(method string, options map[string]any) MigrationsEvent {
	return MigrationsEvent{Method: method, Options: options}
}

// MigrationsStarted is dispatched once at the top of a run that has something
// to do, before the first migration is touched.
//
// A run with nothing pending fires NoPendingMigrations instead and never fires
// this, so it means "the schema is about to change". It is the hook for the
// things that bracket a whole deploy step: put the application into
// maintenance, open a span, take a note of the time.
type MigrationsStarted struct{ MigrationsEvent }

// NewMigrationsStarted creates a MigrationsStarted event.
func NewMigrationsStarted(method string, options map[string]any) *MigrationsStarted {
	return &MigrationsStarted{NewMigrationsEvent(method, options)}
}

// MigrationsEnded is dispatched once at the end of a run in which every
// migration succeeded.
//
// The run stops at the first failure, so a failed run fires MigrationsStarted
// and never this: the pair is the signal that the schema reached the state the
// binary expects. It is where the bracket MigrationsStarted opened is closed.
type MigrationsEnded struct{ MigrationsEvent }

// NewMigrationsEnded creates a MigrationsEnded event.
func NewMigrationsEnded(method string, options map[string]any) *MigrationsEnded {
	return &MigrationsEnded{NewMigrationsEvent(method, options)}
}

// NoPendingMigrations is dispatched when a run found nothing to do: `aru
// migrate` with every migration already applied, or `aru migrate:rollback` with
// nothing left to undo.
//
// It takes the place of the Started/Ended pair rather than joining it, so a
// listener that brackets a run has to treat this as the third possible shape.
// It is not a failure -- a deploy that changes no schema is the normal case.
type NoPendingMigrations struct {
	// Method is "up" or "down".
	Method string
}

// NewNoPendingMigrations creates a NoPendingMigrations event.
func NewNoPendingMigrations(method string) *NoPendingMigrations {
	return &NoPendingMigrations{Method: method}
}

// MigrationsPruned reports that the squashed migration files were deleted after
// a schema dump.
type MigrationsPruned struct {
	// Connection is the connection the migrations were pruned on.
	Connection Connection

	// ConnectionName is the name of the connection the migrations were
	// pruned on.
	ConnectionName string

	// Path is the path the migration files were pruned from.
	Path string
}

// NewMigrationsPruned creates a MigrationsPruned event.
func NewMigrationsPruned(connection Connection, path string) *MigrationsPruned {
	name := ""
	if connection != nil {
		name = connection.GetName()
	}
	return &MigrationsPruned{Connection: connection, ConnectionName: name, Path: path}
}

// SchemaDumped reports that a connection's whole schema was written to a file,
// and where.
//
// The dump is what replaces years of migration files: a single SQL file the
// engine's own tool produced, loaded in one step by a fresh database instead of
// replayed migration by migration. Nothing in this package dispatches the
// event -- SchemaState.Dump does the work and the command that drives it lives
// in the application -- so it exists so that a project wiring `schema:dump` has
// an event to fire, and so that a listener can add the file to a release
// artifact.
type SchemaDumped struct {
	// Connection is the connection whose schema was dumped.
	Connection Connection

	// ConnectionName is the name of the connection whose schema was dumped.
	ConnectionName string

	// Path is the path the schema was written to.
	Path string
}

// NewSchemaDumped creates a SchemaDumped event.
func NewSchemaDumped(connection Connection, path string) *SchemaDumped {
	name := ""
	if connection != nil {
		name = connection.GetName()
	}
	return &SchemaDumped{Connection: connection, ConnectionName: name, Path: path}
}

// SchemaLoaded reports that a schema file was run against a connection, and
// which file.
//
// It is the other half of SchemaDumped: the migrator loads the dump into an
// empty database and then applies only the migrations written since. Nothing
// here dispatches it, for the same reason -- SchemaState.Load does the work,
// and the event is the vocabulary an application uses when it wires the
// command.
type SchemaLoaded struct {
	// Connection is the connection the schema file was loaded into.
	Connection Connection

	// ConnectionName is the name of the connection the schema file was
	// loaded into.
	ConnectionName string

	// Path is the path the schema file was loaded from.
	Path string
}

// NewSchemaLoaded creates a SchemaLoaded event.
func NewSchemaLoaded(connection Connection, path string) *SchemaLoaded {
	name := ""
	if connection != nil {
		name = connection.GetName()
	}
	return &SchemaLoaded{Connection: connection, ConnectionName: name, Path: path}
}

// ModelsPruned reports how many rows one prunable removed.
//
// Nothing here dispatches it: `model:prune` is a command a project writes
// against its own repositories, and this is the event it fires.
type ModelsPruned struct {
	// Model is the type whose rows were pruned.
	Model string

	// Count is how many rows were removed.
	Count int
}

// NewModelsPruned creates a ModelsPruned event.
func NewModelsPruned(model string, count int) *ModelsPruned {
	return &ModelsPruned{Model: model, Count: count}
}

// ModelPruningStarting is dispatched by `aru model:prune` before it prunes
// anything, carrying the names of everything it is about to prune.
//
// It fires once for the whole command, in sorted order, and only when there is
// at least one registered prunable. Because the list is complete before the
// first deletion, a listener can log or refuse on the whole set rather than
// discovering it one ModelsPruned at a time. --pretend still fires it: the
// names are what pretending is for.
type ModelPruningStarting struct {
	// Models is the names of everything about to be pruned.
	Models []string
}

// NewModelPruningStarting creates a ModelPruningStarting event.
func NewModelPruningStarting(models []string) *ModelPruningStarting {
	return &ModelPruningStarting{Models: models}
}

// ModelPruningFinished is dispatched by `aru model:prune` once every registered
// prunable has run, carrying the same list ModelPruningStarting carried.
//
// The command stops at the first prunable that errors, so this not arriving
// after a ModelPruningStarting means one of them failed. How many rows each one
// removed is in the ModelsPruned events between the two, not here.
type ModelPruningFinished struct {
	// Models is the same list ModelPruningStarting carried.
	Models []string
}

// NewModelPruningFinished creates a ModelPruningFinished event.
func NewModelPruningFinished(models []string) *ModelPruningFinished {
	return &ModelPruningFinished{Models: models}
}
