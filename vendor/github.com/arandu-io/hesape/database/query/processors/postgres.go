package processors

import (
	"context"
	"fmt"
	"strings"

	"github.com/arandu-io/hesape/database/query"
)

// PostgresProcessor is the Processor for Postgres: it reads back an inserted
// identifier from the returning clause, and turns the catalogue's one-letter
// codes into words.
//
// Postgres reports an insert's identifier as a row rather than out of band, so
// ProcessInsertGetID runs the statement as a select -- against the write
// connection, because a replica has not seen the row yet. Everything else here
// is the schema introspection: pg_type stores a kind as "e" and a category as
// "n", and ProcessTypes, ProcessColumns, ProcessIndexes and ProcessForeignKeys
// spell those out so `aru db:show` reads the same whatever the engine.
type PostgresProcessor struct {
	Processor
}

var _ query.Processor = (*PostgresProcessor)(nil)

// NewPostgresProcessor creates a PostgresProcessor.
func NewPostgresProcessor() *PostgresProcessor { return &PostgresProcessor{} }

// ProcessInsertGetID runs an insert and returns the identifier the engine
// assigned to the new row.
//
// PostgresGrammar compiles the insert with a returning clause, so the
// identifier comes back as a row and the statement runs as a select. It
// runs against the write connection, because a replica has not seen the
// row yet.
//
// It takes no authorization credential and runs the statement it is handed: a
// caller that reaches this is below the authorization layer, not outside it.
// The credential was checked where the *query.Builder was obtained, and a
// second check here would be a second place to forget it.
func (p *PostgresProcessor) ProcessInsertGetID(ctx context.Context, q *query.Builder, sql string, values []any, sequence string) (int64, error) {
	connection := q.GetConnection()
	if connection == nil {
		return 0, fmt.Errorf("query/processors: the builder has no connection to insert through")
	}

	results, err := connection.Select(ctx, sql, values, false)
	if err != nil {
		return 0, err
	}
	if len(results) == 0 {
		return 0, fmt.Errorf("query/processors: the insert returned no row to read the identifier from")
	}

	if sequence == "" {
		sequence = "id"
	}

	return numericID(results[0][sequence])
}

// ProcessTypes spells out the one letter codes pg_type stores for a
// type's kind and category.
func (p *PostgresProcessor) ProcessTypes(results []query.Record) []query.Record {
	kinds := map[string]any{
		"b": "base", "c": "composite", "d": "domain", "e": "enum",
		"p": "pseudo", "r": "range", "m": "multirange",
	}
	categories := map[string]any{
		"a": "array", "b": "boolean", "c": "composite", "d": "date_time",
		"e": "enum", "g": "geometric", "i": "network_address", "n": "numeric",
		"p": "pseudo", "r": "range", "s": "string", "t": "timespan",
		"u": "user_defined", "v": "bit_string", "x": "unknown", "z": "internal_use",
	}

	out := make([]query.Record, 0, len(results))
	for _, result := range results {
		name := stringOf(result, "name")
		schema := stringOf(result, "schema")

		out = append(out, query.Record{
			"name":                  name,
			"schema":                schema,
			"schema_qualified_name": schema + "." + name,
			"implicit":              boolOf(result, "implicit"),
			"type":                  kinds[strings.ToLower(stringOf(result, "type"))],
			"category":              categories[strings.ToLower(stringOf(result, "category"))],
		})
	}

	return out
}

// ProcessColumns normalises the columns of a column listing, splitting a
// generated column's expression out of its default.
func (p *PostgresProcessor) ProcessColumns(results []query.Record, sql ...string) []query.Record {
	out := make([]query.Record, 0, len(results))

	for _, result := range results {
		def := optional(result, "default")
		autoincrement := def != nil && strings.HasPrefix(stringOf(result, "default"), "nextval(")

		generated := stringOf(result, "generated")

		var (
			value      = def
			generation any
		)
		if generated != "" {
			value = nil

			var kind any
			switch generated {
			case "s":
				kind = "stored"
			case "v":
				kind = "virtual"
			}
			generation = query.Record{"type": kind, "expression": def}
		}

		out = append(out, query.Record{
			"name":           stringOf(result, "name"),
			"type_name":      stringOf(result, "type_name"),
			"type":           stringOf(result, "type"),
			"collation":      optional(result, "collation"),
			"nullable":       boolOf(result, "nullable"),
			"default":        value,
			"auto_increment": autoincrement,
			"comment":        optional(result, "comment"),
			"generation":     generation,
		})
	}

	return out
}

// ProcessIndexes normalises the columns of an index listing.
func (p *PostgresProcessor) ProcessIndexes(results []query.Record) []query.Record {
	out := make([]query.Record, 0, len(results))

	for _, result := range results {
		out = append(out, query.Record{
			"name":    strings.ToLower(stringOf(result, "name")),
			"columns": splitList(stringOf(result, "columns")),
			"type":    strings.ToLower(stringOf(result, "type")),
			"unique":  boolOf(result, "unique"),
			"primary": boolOf(result, "primary"),
		})
	}

	return out
}

// ProcessForeignKeys spells out the one letter referential actions for a
// foreign key's on-update and on-delete behavior.
func (p *PostgresProcessor) ProcessForeignKeys(results []query.Record) []query.Record {
	actions := map[string]any{
		"a": "no action", "r": "restrict", "c": "cascade",
		"n": "set null", "d": "set default",
	}

	out := make([]query.Record, 0, len(results))
	for _, result := range results {
		out = append(out, query.Record{
			"name":            stringOf(result, "name"),
			"columns":         strings.Split(stringOf(result, "columns"), ","),
			"foreign_schema":  stringOf(result, "foreign_schema"),
			"foreign_table":   stringOf(result, "foreign_table"),
			"foreign_columns": strings.Split(stringOf(result, "foreign_columns"), ","),
			"on_update":       actions[strings.ToLower(stringOf(result, "on_update"))],
			"on_delete":       actions[strings.ToLower(stringOf(result, "on_delete"))],
		})
	}

	return out
}
