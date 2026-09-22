package processors

import (
	"regexp"
	"strings"

	"github.com/arandu-io/hesape/database/query"
)

// SQLiteProcessor is the Processor for SQLite.
//
// SQLite reports less about a column than the other two do, so this is the one
// processor that reads the CREATE TABLE statement back: the collation and the
// expression of a generated column exist only in the text the table was
// declared with.
type SQLiteProcessor struct {
	Processor
}

var _ query.Processor = (*SQLiteProcessor)(nil)

// NewSQLiteProcessor creates a SQLiteProcessor.
func NewSQLiteProcessor() *SQLiteProcessor { return &SQLiteProcessor{} }

// ProcessColumns normalises the columns of a column listing, reading the
// collation and any generated expression out of the CREATE TABLE
// statement.
//
// The optional argument is that statement; without it the collation and the
// generated expression come back nil.
func (p *SQLiteProcessor) ProcessColumns(results []query.Record, sql ...string) []query.Record {
	statement := ""
	if len(sql) > 0 {
		statement = sql[0]
	}

	// An INTEGER PRIMARY KEY is the rowid, and only then is it the auto
	// incrementing column -- which is why a composite primary key has none.
	primaries := 0
	for _, result := range results {
		if boolOf(result, "primary") {
			primaries++
		}
	}
	hasPrimaryKey := primaries == 1

	out := make([]query.Record, 0, len(results))

	for _, result := range results {
		name := stringOf(result, "name")
		typ := strings.ToLower(stringOf(result, "type"))
		extra := intOf(result, "extra")

		isGenerated := extra == 2 || extra == 3

		var generation any
		if isGenerated {
			var kind any
			switch extra {
			case 3:
				kind = "stored"
			case 2:
				kind = "virtual"
			}
			generation = query.Record{
				"type":       kind,
				"expression": matchColumnExpression(statement, name),
			}
		}

		out = append(out, query.Record{
			"name":           name,
			"type_name":      typeName(typ),
			"type":           typ,
			"collation":      matchColumnCollation(statement, name),
			"nullable":       boolOf(result, "nullable"),
			"default":        optional(result, "default"),
			"auto_increment": hasPrimaryKey && boolOf(result, "primary") && typ == "integer",
			"comment":        nil,
			"generation":     generation,
		})
	}

	return out
}

// ProcessIndexes normalises the columns of an index listing.
//
// SQLite reports the implicit index of a composite primary key alongside
// the key itself, so when more than one index calls itself primary the
// named one is dropped and the real key is kept.
func (p *SQLiteProcessor) ProcessIndexes(results []query.Record) []query.Record {
	primaries := 0

	indexes := make([]query.Record, 0, len(results))
	for _, result := range results {
		isPrimary := boolOf(result, "primary")
		if isPrimary {
			primaries++
		}

		indexes = append(indexes, query.Record{
			"name":    strings.ToLower(stringOf(result, "name")),
			"columns": splitList(stringOf(result, "columns")),
			"type":    nil,
			"unique":  boolOf(result, "unique"),
			"primary": isPrimary,
		})
	}

	if primaries > 1 {
		kept := make([]query.Record, 0, len(indexes))
		for _, index := range indexes {
			if index["name"] == "primary" {
				continue
			}
			kept = append(kept, index)
		}
		return kept
	}

	return indexes
}

// ProcessForeignKeys normalises the columns of a foreign key listing.
// SQLite does not name a foreign key, so the name is nil rather than
// invented.
func (p *SQLiteProcessor) ProcessForeignKeys(results []query.Record) []query.Record {
	out := make([]query.Record, 0, len(results))

	for _, result := range results {
		out = append(out, query.Record{
			"name":            nil,
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

// typeName strips a trailing size or precision: "varchar(255)" is a
// varchar.
func typeName(typ string) string {
	if i := strings.Index(typ, "("); i >= 0 {
		return typ[:i]
	}
	return typ
}

// matchColumnCollation reads the collation of one column out of the CREATE
// TABLE statement, which is the only place SQLite keeps it.
func matchColumnCollation(sql, column string) any {
	if sql == "" {
		return nil
	}

	pattern, err := regexp.Compile(`(?i)\b` + regexp.QuoteMeta(column) +
		`\b[^,(]+(?:\([^()]+\)[^,]*)?(?:(?:default|check|as)\s*(?:\(.*?\))?[^,]*)*collate\s+["'` + "`" + `]?(\w+)`)
	if err != nil {
		return nil
	}

	if matches := pattern.FindStringSubmatch(sql); matches != nil {
		return strings.ToLower(matches[1])
	}

	return nil
}

// matchColumnExpression reads the expression of a generated column out of the
// CREATE TABLE statement.
func matchColumnExpression(sql, column string) any {
	if sql == "" {
		return nil
	}

	pattern, err := regexp.Compile(`(?i)\b` + regexp.QuoteMeta(column) +
		`\b[^,]+\s+as\s+\(((?:[^()]+|\((?:[^()]+|\([^()]*\))*\))*)\)`)
	if err != nil {
		return nil
	}

	if matches := pattern.FindStringSubmatch(sql); matches != nil {
		return matches[1]
	}

	return nil
}
