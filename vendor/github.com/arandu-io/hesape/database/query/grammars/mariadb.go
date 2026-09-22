package grammars

import (
	"fmt"

	"github.com/arandu-io/hesape/database/query"
)

// MariaDBGrammar is the grammar for MariaDB.
//
// MariaDB is MySQL's grammar with four differences, and it is a separate
// grammar rather than a flag on MySQL's for exactly that reason: it has no
// lateral join, it reads a JSON value with json_value rather than
// json_unquote(json_extract()), its open connection count lives in a different
// table, and it has had window functions since 10.2, so the group limit never
// needs the session variable form.
type MariaDBGrammar struct {
	*MySQLGrammar
}

var _ query.Grammar = (*MariaDBGrammar)(nil)

// NewMariaDBGrammar creates a MariaDBGrammar.
func NewMariaDBGrammar() *MariaDBGrammar {
	g := &MariaDBGrammar{MySQLGrammar: NewMySQLGrammar()}
	g.Grammar.self = g
	return g
}

// CompileJoinLateral returns an error: MariaDB has no lateral join, and
// MySQL's grammar would otherwise have said it does.
func (g *MariaDBGrammar) CompileJoinLateral(join *query.JoinClause, expression string) (string, error) {
	return "", fmt.Errorf("query/grammars: this database engine does not support lateral joins")
}

// CompileJSONValueCast casts a value expression to JSON, MariaDB's way.
func (g *MariaDBGrammar) CompileJSONValueCast(value string) string {
	return "json_query(" + value + ", '$')"
}

// CompileThreadCount builds the SQL that counts open connections,
// MariaDB's way.
func (g *MariaDBGrammar) CompileThreadCount() string {
	return "select variable_value as `Value` from information_schema.global_status where variable_name = 'THREADS_CONNECTED'"
}

// UseLegacyGroupLimit reports false: MariaDB has had window functions
// since 10.2, so the group limit never needs the legacy form.
func (g *MariaDBGrammar) UseLegacyGroupLimit(q *query.Builder) bool { return false }

// WrapJSONSelector quotes a JSON path selector, MariaDB's way.
func (g *MariaDBGrammar) WrapJSONSelector(value string) string {
	field, path := g.wrapJSONFieldAndPath(value)
	return "json_value(" + field + path + ")"
}
