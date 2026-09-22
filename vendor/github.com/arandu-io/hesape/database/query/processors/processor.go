package processors

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/arandu-io/hesape/database/query"
)

// Processor is the hook a driver takes to adjust results on the way out of the
// connection.
//
// Almost every method is the identity, and that is the point: the processor is
// where an engine that resolves a question differently is made to resolve it
// the same. Postgres hands back an inserted identifier as a row; MySQL reports
// it out of band. Both arrive at the caller as an int64.
//
// # Where authorization is
//
// Nowhere here, and that is deliberate. ProcessInsertGetID does run a
// statement, through the connection the builder is holding, and the Grant that
// allowed it was checked one layer up: a *query.Builder is reachable only from
// a repository that holds an auth.Grant and has already filtered by
// auth.Tenant(g). A processor that took a Grant would be a second place to
// enforce authorization, which is a second place for it to be forgotten.
type Processor struct{}

var _ query.Processor = (*Processor)(nil)

// NewProcessor creates a Processor.
func NewProcessor() *Processor { return &Processor{} }

// LastInsertIDConnection is the part of the connection ProcessInsertGetID needs
// beyond query.connection: the identifier the engine assigned to the row it
// just inserted.
//
// query.connection is narrowed to running statements and has no such
// method, so a connection that can answer implements this and one that
// cannot says so rather than returning a zero that reads like an
// identifier.
type LastInsertIDConnection interface {
	// GetLastInsertID returns the identifier the engine assigned to the last
	// inserted row. The sequence names the generator to ask, for an engine
	// that has more than one.
	GetLastInsertID(sequence string) (int64, error)
}

// ProcessSelect is the identity: select results need no adjustment.
func (p *Processor) ProcessSelect(q *query.Builder, results []query.Record) []query.Record {
	return results
}

// ProcessInsertGetID runs an insert and returns the identifier the engine
// assigned to the new row. Go initialisms are upper case throughout, hence
// ID rather than Id.
//
// It returns an int64: query.Processor declares the signature, and an
// engine whose identifier is not a number is a repository's problem before
// it is a processor's.
//
// It takes no authorization credential and runs the statement it is handed: a
// caller that reaches this is below the authorization layer, not outside it.
// The credential was checked where the *query.Builder was obtained, and a
// second check here would be a second place to forget it.
func (p *Processor) ProcessInsertGetID(ctx context.Context, q *query.Builder, sql string, values []any, sequence string) (int64, error) {
	connection := q.GetConnection()
	if connection == nil {
		return 0, fmt.Errorf("query/processors: the builder has no connection to insert through")
	}

	if _, err := connection.Insert(ctx, sql, values); err != nil {
		return 0, err
	}

	last, ok := connection.(LastInsertIDConnection)
	if !ok {
		return 0, fmt.Errorf("query/processors: this connection cannot report the identifier it assigned; it does not implement GetLastInsertID")
	}

	return last.GetLastInsertID(sequence)
}

// ProcessSchemas normalises the columns of a schema listing.
func (p *Processor) ProcessSchemas(results []query.Record) []query.Record {
	out := make([]query.Record, 0, len(results))
	for _, result := range results {
		out = append(out, query.Record{
			"name":    stringOf(result, "name"),
			"path":    optional(result, "path"), // SQLite only
			"default": boolOf(result, "default"),
		})
	}
	return out
}

// ProcessTables normalises the columns of a table listing.
func (p *Processor) ProcessTables(results []query.Record) []query.Record {
	out := make([]query.Record, 0, len(results))
	for _, result := range results {
		out = append(out, query.Record{
			"name":                  stringOf(result, "name"),
			"schema":                optional(result, "schema"),
			"schema_qualified_name": qualify(result, "name"),
			"size":                  optionalInt(result, "size"),
			"comment":               optional(result, "comment"),   // MySQL and PostgreSQL
			"collation":             optional(result, "collation"), // MySQL only
			"engine":                optional(result, "engine"),    // MySQL only
		})
	}
	return out
}

// ProcessViews normalises the columns of a view listing.
func (p *Processor) ProcessViews(results []query.Record) []query.Record {
	out := make([]query.Record, 0, len(results))
	for _, result := range results {
		out = append(out, query.Record{
			"name":                  stringOf(result, "name"),
			"schema":                optional(result, "schema"),
			"schema_qualified_name": qualify(result, "name"),
			"definition":            stringOf(result, "definition"),
		})
	}
	return out
}

// ProcessTypes is the identity: only Postgres has types to spell out.
func (p *Processor) ProcessTypes(results []query.Record) []query.Record { return results }

// ProcessColumns is the identity here; a driver processor normalises the
// columns of a column listing.
//
// The variadic argument is SQLiteProcessor's second parameter, the CREATE
// TABLE statement it has to read a collation out of. Go has no default
// argument, so every processor accepts it and only that one reads it.
func (p *Processor) ProcessColumns(results []query.Record, sql ...string) []query.Record {
	return results
}

// ProcessIndexes is the identity: only a driver processor normalises the
// columns of an index listing.
func (p *Processor) ProcessIndexes(results []query.Record) []query.Record { return results }

// ProcessForeignKeys is the identity: only a driver processor normalises
// the columns of a foreign key listing.
func (p *Processor) ProcessForeignKeys(results []query.Record) []query.Record { return results }

// qualify joins a schema and a name as "schema.name", the form a statement
// can use from another schema.
func qualify(result query.Record, key string) string {
	name := stringOf(result, key)
	if schema := stringOf(result, "schema"); schema != "" {
		return schema + "." + name
	}
	return name
}

// A driver hands a column back as whatever its wire format is -- a MySQL
// count arrives as []byte, a SQLite flag as int64 -- so the coercions below
// settle the shape a caller reads here rather than at every call site.

func stringOf(result query.Record, key string) string {
	switch value := result[key].(type) {
	case nil:
		return ""
	case string:
		return value
	case []byte:
		return string(value)
	default:
		return fmt.Sprint(value)
	}
}

func boolOf(result query.Record, key string) bool {
	switch value := result[key].(type) {
	case nil:
		return false
	case bool:
		return value
	case int64:
		return value != 0
	case int:
		return value != 0
	case string:
		return value != "" && value != "0" && !strings.EqualFold(value, "false")
	case []byte:
		text := string(value)
		return text != "" && text != "0" && !strings.EqualFold(text, "false")
	default:
		return false
	}
}

func intOf(result query.Record, key string) int64 {
	switch value := result[key].(type) {
	case nil:
		return 0
	case int64:
		return value
	case int:
		return int64(value)
	case float64:
		return int64(value)
	case bool:
		if value {
			return 1
		}
		return 0
	case string:
		number, _ := strconv.ParseInt(value, 10, 64)
		return number
	case []byte:
		number, _ := strconv.ParseInt(string(value), 10, 64)
		return number
	default:
		return 0
	}
}

// optional returns the value at key, or nil if it is absent: absent stays
// absent, which is what tells a caller the engine does not report it at
// all.
func optional(result query.Record, key string) any {
	value, ok := result[key]
	if !ok || value == nil {
		return nil
	}
	if raw, isBytes := value.([]byte); isBytes {
		return string(raw)
	}
	return value
}

func optionalInt(result query.Record, key string) any {
	if optional(result, key) == nil {
		return nil
	}
	return intOf(result, key)
}

// numericID parses value as the int64 query.Processor declares, refusing
// anything that is not a number.
func numericID(value any) (int64, error) {
	switch id := value.(type) {
	case nil:
		return 0, fmt.Errorf("query/processors: the insert returned no identifier")
	case int64:
		return id, nil
	case int:
		return int64(id), nil
	case float64:
		return int64(id), nil
	case string:
		return parseID(id)
	case []byte:
		return parseID(string(id))
	default:
		return 0, fmt.Errorf("query/processors: the insert returned an identifier of type %T, which is not a number", value)
	}
}

func parseID(value string) (int64, error) {
	id, err := strconv.ParseInt(value, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("query/processors: the insert returned %q, which is not a number", value)
	}
	return id, nil
}
