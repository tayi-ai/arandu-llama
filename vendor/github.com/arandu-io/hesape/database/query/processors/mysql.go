package processors

import (
	"strings"

	"github.com/arandu-io/hesape/database/query"
)

// MySQLProcessor is the Processor for MySQL.
//
// There is one route to an inserted identifier here --
// LastInsertIDConnection -- so the shared ProcessInsertGetID is already the
// right one and nothing is overridden.
type MySQLProcessor struct {
	Processor
}

var _ query.Processor = (*MySQLProcessor)(nil)

// NewMySQLProcessor creates a MySQLProcessor.
func NewMySQLProcessor() *MySQLProcessor { return &MySQLProcessor{} }

// ProcessColumnListing reduces a column listing to the column names.
//
// Deprecated: use ProcessColumns, which reports the same columns along
// with their type, default and the rest of it.
func (p *MySQLProcessor) ProcessColumnListing(results []query.Record) []string {
	out := make([]string, 0, len(results))
	for _, result := range results {
		out = append(out, stringOf(result, "column_name"))
	}
	return out
}

// ProcessColumns normalises the columns of a column listing.
func (p *MySQLProcessor) ProcessColumns(results []query.Record, sql ...string) []query.Record {
	out := make([]query.Record, 0, len(results))

	for _, result := range results {
		extra := stringOf(result, "extra")

		var generation any
		if expression := optional(result, "expression"); expression != nil && stringOf(result, "expression") != "" {
			var kind any
			switch extra {
			case "STORED GENERATED":
				kind = "stored"
			case "VIRTUAL GENERATED":
				kind = "virtual"
			}
			generation = query.Record{"type": kind, "expression": expression}
		}

		comment := optional(result, "comment")
		if stringOf(result, "comment") == "" {
			comment = nil
		}

		out = append(out, query.Record{
			"name":           stringOf(result, "name"),
			"type_name":      stringOf(result, "type_name"),
			"type":           stringOf(result, "type"),
			"collation":      optional(result, "collation"),
			"nullable":       stringOf(result, "nullable") == "YES",
			"default":        optional(result, "default"),
			"auto_increment": extra == "auto_increment",
			"comment":        comment,
			"generation":     generation,
		})
	}

	return out
}

// ProcessIndexes normalises the columns of an index listing.
func (p *MySQLProcessor) ProcessIndexes(results []query.Record) []query.Record {
	out := make([]query.Record, 0, len(results))

	for _, result := range results {
		name := strings.ToLower(stringOf(result, "name"))

		out = append(out, query.Record{
			"name":    name,
			"columns": splitList(stringOf(result, "columns")),
			"type":    strings.ToLower(stringOf(result, "type")),
			"unique":  boolOf(result, "unique"),
			"primary": name == "primary",
		})
	}

	return out
}

// ProcessForeignKeys normalises the columns of a foreign key listing.
func (p *MySQLProcessor) ProcessForeignKeys(results []query.Record) []query.Record {
	out := make([]query.Record, 0, len(results))

	for _, result := range results {
		out = append(out, query.Record{
			"name":            stringOf(result, "name"),
			"columns":         strings.Split(stringOf(result, "columns"), ","),
			"foreign_schema":  stringOf(result, "foreign_schema"),
			"foreign_table":   stringOf(result, "foreign_table"),
			"foreign_columns": strings.Split(stringOf(result, "foreign_columns"), ","),
			"on_update":       strings.ToLower(stringOf(result, "on_update")),
			"on_delete":       strings.ToLower(stringOf(result, "on_delete")),
		})
	}

	return out
}

// splitList splits a comma separated column list, with the empty string
// staying an empty list rather than becoming a list of one empty name.
func splitList(value string) []string {
	if value == "" {
		return []string{}
	}
	return strings.Split(value, ",")
}
