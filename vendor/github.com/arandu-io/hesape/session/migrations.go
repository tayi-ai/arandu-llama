package session

import (
	"context"

	"github.com/arandu-io/hesape/database/migrations"
	"github.com/arandu-io/hesape/database/schema"
)

// Table is the table [DatabaseSessionHandler] reads and writes.
const Table = "sessions"

// CreateSessionsTable creates the table the database session handler uses.
//
// It is code rather than a file in a tree, for the reason
// [migrations.Migration] gives: a migration written as SQL text describes one
// engine, and this collection runs on three.
type CreateSessionsTable struct{ migrations.BaseMigration }

// GetName returns the migration's name.
func (CreateSessionsTable) GetName() string {
	return "2026_08_10_000003_create_sessions_table"
}

// Up creates the sessions table and the two indexes it is read by.
//
// Three of its columns stay null in practice: the handler does not fill
// user_id, ip_address or user_agent, because there is no container to fetch the
// guard and the request out of -- see [DatabaseSessionHandler]. They are in the
// table anyway, because an application that wants them writes them itself and
// adding a column later costs a second migration.
//
// last_activity is indexed because garbage collection deletes on it, and an
// unindexed sweep of the session table is a full scan on the busiest table in
// the schema.
func (CreateSessionsTable) Up(ctx context.Context, conn migrations.Connection) error {
	return conn.Schema().Create(ctx, Table, func(table *schema.Blueprint) {
		table.String("id").Primary()
		// A string and not a bigint: the ids in this collection are uuids
		// (see database.NewID), and an autoincrement column here would be a
		// column no application in it can fill.
		table.String("user_id").Nullable()
		table.String("ip_address").Nullable()
		table.Text("user_agent").Nullable()
		table.Text("payload")
		table.BigInteger("last_activity")

		table.Index([]string{"user_id"}, "sessions_user_id_index")
		table.Index([]string{"last_activity"}, "sessions_last_activity_index")
	})
}

// Down drops the sessions table, and the two indexes with it.
func (CreateSessionsTable) Down(ctx context.Context, conn migrations.Connection) error {
	return conn.Schema().DropIfExists(ctx, Table)
}

// Migrations is the sessions table.
//
// The table belongs to the package that reads it: an application keeping
// sessions in a cookie or in Redis does not create it, and one using
// [DatabaseSessionHandler] adds this to the list it hands the migrator.
func Migrations() []migrations.Migration {
	return []migrations.Migration{CreateSessionsTable{}}
}
