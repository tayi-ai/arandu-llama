package migrations

import (
	"context"

	"github.com/arandu-io/hesape/database/schema"
	"github.com/arandu-io/hesape/database/schema/grammars"
)

// UpStatements returns the statements a migration's Up would run, without
// running any of them.
//
// It is what a generator writes to a file and what a test asserts on: both need
// the text a migration would send, and neither has a server. A migration that
// reads before it writes sees no rows here, because Select returns none, so
// what comes back is the path taken over an empty result.
func UpStatements(ctx context.Context, migration Migration) ([]string, error) {
	conn := &recordingConnection{name: migration.GetConnection()}
	if err := migration.Up(ctx, conn); err != nil {
		return nil, err
	}
	return conn.statements, nil
}

// DownStatements returns the statements a migration's Down would run, without
// running any of them. A migration that is not reversible returns none.
func DownStatements(ctx context.Context, migration Migration) ([]string, error) {
	reversible, isReversible := migration.(ReversibleMigration)
	if !isReversible {
		return nil, nil
	}
	conn := &recordingConnection{name: migration.GetConnection()}
	if err := reversible.Down(ctx, conn); err != nil {
		return nil, err
	}
	return conn.statements, nil
}

// recordingConnection is the Connection UpStatements and DownStatements run
// against: it keeps every statement and sends none.
//
// Bindings are dropped rather than recorded. A schema change is DDL, which
// carries none, and a caller that needs them wants the connection's own
// Pretend rather than this.
type recordingConnection struct {
	name       string
	statements []string
}

// GetName returns the connection name the migration asked for.
func (c *recordingConnection) GetName() string { return c.name }

// Schema answers a builder over the same recorder, so a migration written with
// the Blueprint is captured exactly like one written with Statement.
//
// Without this, UpStatements would answer nothing at all for the migrations
// written the standard way -- and a --pretend that prints nothing reads like a
// migration that does nothing.
func (c *recordingConnection) Schema() *schema.Builder {
	return schema.NewBuilder(&recordingSchemaConnection{recorder: c})
}

// recordingSchemaConnection is what the recorded builder compiles against.
//
// It answers no rows, which is the same answer recordingConnection.Select gives
// and for the same reason: there is no server here. A Blueprint that asks
// whether a table exists is told no, so what comes back is the path taken over
// an empty database -- which is what a caller printing statements is asking for.
type recordingSchemaConnection struct{ recorder *recordingConnection }

func (c *recordingSchemaConnection) GetSchemaGrammar() schema.Grammar {
	return grammars.NewSQLiteGrammar(c)
}

func (c *recordingSchemaConnection) GetConfig(string) string { return "" }
func (c *recordingSchemaConnection) GetTablePrefix() string  { return "" }
func (c *recordingSchemaConnection) GetDriverName() string   { return "sqlite" }
func (c *recordingSchemaConnection) GetServerVersion() string {
	return ""
}
func (c *recordingSchemaConnection) IsMaria() bool                      { return false }
func (c *recordingSchemaConnection) ForeignKeyConstraintsEnabled() bool { return true }

func (c *recordingSchemaConnection) Statement(_ context.Context, statement string) error {
	c.recorder.statements = append(c.recorder.statements, statement)
	return nil
}

func (c *recordingSchemaConnection) Select(context.Context, string) ([]schema.Record, error) {
	return nil, nil
}

func (c *recordingSchemaConnection) Scalar(context.Context, string) (any, error) { return nil, nil }

func (c *recordingSchemaConnection) ProcessTables([]schema.Record) []schema.TableInfo { return nil }
func (c *recordingSchemaConnection) ProcessViews([]schema.Record) []schema.ViewInfo   { return nil }
func (c *recordingSchemaConnection) ProcessColumns([]schema.Record) []schema.ColumnInfo {
	return nil
}
func (c *recordingSchemaConnection) ProcessIndexes([]schema.Record) []schema.IndexInfo { return nil }
func (c *recordingSchemaConnection) ProcessForeignKeys([]schema.Record) []schema.ForeignKeyInfo {
	return nil
}
func (c *recordingSchemaConnection) ProcessSchemas([]schema.Record) []schema.SchemaInfo {
	return nil
}

// Statement records query instead of running it.
func (c *recordingConnection) Statement(_ context.Context, query string, _ []any) (bool, error) {
	c.statements = append(c.statements, query)
	return true, nil
}

// Select returns no rows, because there is no server to ask.
func (c *recordingConnection) Select(_ context.Context, _ string, _ []any) ([]map[string]any, error) {
	return nil, nil
}
