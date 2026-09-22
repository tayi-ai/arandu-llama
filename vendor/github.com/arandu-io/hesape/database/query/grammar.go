package query

import (
	"fmt"
	"strings"
	"unicode"
)

// Grammar is what a Builder needs of the thing that spells its SQL.
//
// It is an interface, declared in the package that consumes it, because the
// concrete grammars live in query/grammars and naming them here would close an
// import cycle: grammars imports query for *Builder, so query cannot import
// grammars for the type.
//
// A driver grammar embeds BaseGrammar and overrides what its dialect spells
// differently.
type Grammar interface {
	// CompileSelect compiles a select statement for the query.
	CompileSelect(query *Builder) string

	// CompileInsert compiles an insert statement for the given rows.
	CompileInsert(query *Builder, values []map[string]any) string

	// CompileInsertOrIgnore compiles an insert that silently skips rows that
	// would violate a constraint.
	CompileInsertOrIgnore(query *Builder, values []map[string]any) string

	// CompileInsertGetID compiles an insert and reports the SQL that reads
	// back the inserted row's ID from the given sequence.
	CompileInsertGetID(query *Builder, values map[string]any, sequence string) string

	// CompileUpdate compiles an update statement for the given values.
	CompileUpdate(query *Builder, values map[string]any) string

	// CompileUpsert compiles an insert that updates the named columns when a
	// row conflicts on uniqueBy.
	CompileUpsert(query *Builder, values []map[string]any, uniqueBy []string, update []string) string

	// CompileDelete compiles a delete statement for the query.
	CompileDelete(query *Builder) string

	// CompileTruncate compiles a truncate. A truncate is more than one
	// statement on some engines, so the result is a map from each statement
	// to its bindings.
	CompileTruncate(query *Builder) map[string][]any

	// CompileExists compiles a select wrapped so the engine can report only
	// whether a row matches.
	CompileExists(query *Builder) string

	// CompileRandom compiles a random ordering expression, optionally seeded.
	CompileRandom(seed string) string

	// PrepareBindingsForUpdate arranges the where bindings after the update
	// values, in the order the compiled statement consumes them.
	PrepareBindingsForUpdate(bindings map[string][]any, values map[string]any) []any

	// PrepareBindingsForDelete returns the where bindings for a delete
	// statement.
	PrepareBindingsForDelete(bindings map[string][]any) []any

	// SupportsSavepoints reports whether the engine can savepoint within a
	// transaction.
	SupportsSavepoints() bool

	// CompileSavepoint compiles a statement that creates the named savepoint.
	CompileSavepoint(name string) string

	// CompileSavepointRollBack compiles a statement that rolls back to the
	// named savepoint.
	CompileSavepointRollBack(name string) string

	// GetOperators returns the comparison operators the grammar accepts.
	GetOperators() []string

	// GetBitwiseOperators returns the operators treated as bitwise rather
	// than comparison.
	GetBitwiseOperators() []string

	// Wrap quotes an identifier, leaving an Expression alone.
	Wrap(value any) string

	// WrapTable quotes a table name, applying the table prefix and any alias.
	WrapTable(table any) string

	// WrapArray quotes each value in a list of identifiers.
	WrapArray(values []any) []string

	// Columnize quotes a list of columns and joins them with commas.
	Columnize(columns []any) string

	// Parameterize returns the comma-separated placeholders for a list of
	// values.
	Parameterize(values []any) string

	// Parameter returns the placeholder for a value, or the value itself when
	// it is an Expression.
	Parameter(value any) string

	// QuoteString quotes a string literal, or each one in a list.
	QuoteString(value any) string

	// Escape returns a value escaped for inclusion directly in SQL, for a
	// caller building a statement by hand rather than through a placeholder.
	//
	// It returns an error rather than panicking: a grammar with no connection
	// cannot escape safely, and returning the unescaped value in that case
	// would be an injection with a reassuring name.
	Escape(value any, binary bool) (string, error)

	// GetDateFormat returns the layout the engine's date columns are
	// formatted with, in Go's reference-time syntax.
	GetDateFormat() string

	// GetTablePrefix returns the prefix applied to every table name.
	GetTablePrefix() string

	// SetTablePrefix sets the prefix applied to every table name.
	SetTablePrefix(prefix string) Grammar
}

// BaseGrammar is everything a grammar can spell without knowing its dialect.
//
// A driver grammar embeds it and overrides what its dialect spells differently.
// It is exported because query/grammars has to embed it.
type BaseGrammar struct {
	tablePrefix string
}

// operators is the list of comparison operators every grammar accepts. A
// driver grammar that accepts more overrides GetOperators and appends.
var operators = []string{
	"=", "<", ">", "<=", ">=", "<>", "!=", "<=>",
	"like", "like binary", "not like", "ilike",
	"&", "|", "^", "<<", ">>", "&~", "is", "is not",
	"rlike", "not rlike", "regexp", "not regexp",
	"~", "~*", "!~", "!~*", "similar to",
	"not similar to", "not ilike", "~~*", "!~~*",
}

// bitwiseOperators is the list of operators treated as bitwise rather than
// comparison.
var bitwiseOperators = []string{"&", "|", "^", "<<", ">>", "&~"}

// normalizeOperator returns the canonical spelling of an operator under the
// policy of the grammar that will compile the query. It deliberately does not
// trim: surrounding, repeated or control whitespace is malformed input rather
// than an alternate spelling.
//
// compiler is the grammar that spells the SQL. declared is the grammar the
// builder was given when that is a different one, and nil when the builder is
// being compiled by its own. The compiling grammar is the one that decides,
// because a token is read by the engine that receives it: "#" is an operator
// in PostgreSQL and the start of a comment in MySQL, so a query assembled for
// the first would lose everything after it under the second.
//
// A word operator the builder's grammar declares is accepted even when the
// compiling one does not know it, which is what lets a grammar extend a
// dialect it does not otherwise replace. Nothing is given up: a word cannot
// open a comment, end a statement or stand in for a placeholder, so an engine
// that does not know it refuses the statement instead of reading it as
// something else.
func normalizeOperator(compiler, declared Grammar, operator string) (string, error) {
	if !lexicallySafeOperator(operator) {
		return "", &InvalidOperatorError{Operator: operator}
	}
	if canonical, ok := declaredOperator(compiler, operator); ok {
		return canonical, nil
	}
	if declared != nil && wordOperator(operator) {
		if canonical, ok := declaredOperator(declared, operator); ok {
			return canonical, nil
		}
	}
	return "", &InvalidOperatorError{Operator: operator}
}

// declaredOperator reports the grammar's own spelling of an operator it
// declares. A nil grammar declares the operators every dialect shares.
func declaredOperator(grammar Grammar, operator string) (string, bool) {
	allowed := operators
	if grammar != nil {
		allowed = grammar.GetOperators()
	}
	for _, candidate := range allowed {
		if lexicallySafeOperator(candidate) && strings.EqualFold(candidate, operator) {
			return candidate, true
		}
	}
	return "", false
}

// wordOperator reports whether an operator is spelled out of letters, digits,
// underscores and the single spaces between them, and so carries no character
// any dialect reads as punctuation.
func wordOperator(operator string) bool {
	if operator == "" {
		return false
	}
	for _, r := range operator {
		if r == ' ' || r == '_' || unicode.IsLetter(r) || unicode.IsDigit(r) {
			continue
		}
		return false
	}
	return true
}

// normalizeOrderDirection rewrites an order's direction in place, in the one
// spelling the grammar interpolates.
//
// OrderBy screens the direction it is handed, but Order.Direction is an
// exported field that reaches the statement verbatim, so a direction written
// straight into the clause has to pass the same screen before compilation.
// Matching is case-insensitive for the same reason operator matching is: a
// clause assembled by hand is not malformed for spelling a keyword in capitals.
//
// A raw order carries an expression instead of a column, and its direction is
// never interpolated, so it has none to check.
func normalizeOrderDirection(order *Order) error {
	if order.SQL != nil {
		return nil
	}
	switch {
	case strings.EqualFold(order.Direction, "asc"):
		order.Direction = "asc"
	case strings.EqualFold(order.Direction, "desc"):
		order.Direction = "desc"
	default:
		return &InvalidDirectionError{Direction: order.Direction}
	}
	return nil
}

func lexicallySafeOperator(operator string) bool {
	if operator == "" || strings.TrimSpace(operator) != operator ||
		strings.Contains(operator, "--") || strings.Contains(operator, "/*") ||
		strings.Contains(operator, "*/") {
		return false
	}

	compound := strings.ContainsRune(operator, ' ')
	for i, r := range operator {
		if r == ' ' {
			if i == 0 || i == len(operator)-1 || operator[i-1] == ' ' {
				return false
			}
			continue
		}
		if unicode.IsControl(r) || unicode.IsSpace(r) || r > unicode.MaxASCII {
			return false
		}
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' {
			continue
		}
		if !compound &&
			strings.ContainsRune("+-*/<>=~!@#%^&|?", r) {
			continue
		}
		return false
	}
	return true
}

// GetOperators returns the comparison operators the grammar accepts.
func (g *BaseGrammar) GetOperators() []string {
	out := make([]string, len(operators))
	copy(out, operators)
	return out
}

// GetBitwiseOperators returns the operators treated as bitwise rather than
// comparison.
func (g *BaseGrammar) GetBitwiseOperators() []string {
	out := make([]string, len(bitwiseOperators))
	copy(out, bitwiseOperators)
	return out
}

// GetTablePrefix returns the prefix applied to every table name.
func (g *BaseGrammar) GetTablePrefix() string { return g.tablePrefix }

// SetTablePrefix sets the prefix applied to every table name.
func (g *BaseGrammar) SetTablePrefix(prefix string) Grammar {
	g.tablePrefix = prefix
	return nil // a driver grammar overrides this to return itself
}

// Wrap quotes an identifier, splitting an aliased name into its two quoted
// halves.
//
// An Expression passes through untouched, which is the whole reason Expression
// exists: it is how a caller says "this is SQL, not a name".
func (g *BaseGrammar) Wrap(value any) string {
	if IsExpression(value) {
		return stringify(value)
	}
	name := stringify(value)
	// "column as alias" is split into its two halves before quoting each one.
	if i := aliasIndex(name); i >= 0 {
		return g.Wrap(strings.TrimSpace(name[:i])) + " as " +
			g.wrapValue(strings.TrimSpace(name[i+4:]))
	}
	return g.wrapSegments(name)
}

// WrapTable quotes a table name. The table prefix is applied here and nowhere
// else, which is why a raw table name in a where clause is a bug that only
// shows up on a prefixed connection.
func (g *BaseGrammar) WrapTable(table any) string {
	if IsExpression(table) {
		return stringify(table)
	}
	name := stringify(table)
	if i := aliasIndex(name); i >= 0 {
		return g.WrapTable(strings.TrimSpace(name[:i])) + " as " +
			g.wrapValue(g.tablePrefix+strings.TrimSpace(name[i+4:]))
	}
	return g.wrapSegments(g.tablePrefix + name)
}

// WrapArray quotes each value in a list of identifiers.
func (g *BaseGrammar) WrapArray(values []any) []string {
	out := make([]string, len(values))
	for i, v := range values {
		out[i] = g.Wrap(v)
	}
	return out
}

// Columnize quotes a list of columns and joins them with commas.
func (g *BaseGrammar) Columnize(columns []any) string {
	return strings.Join(g.WrapArray(columns), ", ")
}

// Parameterize returns the comma-separated placeholders for a list of values.
func (g *BaseGrammar) Parameterize(values []any) string {
	out := make([]string, len(values))
	for i, v := range values {
		out[i] = g.Parameter(v)
	}
	return strings.Join(out, ", ")
}

// Parameter returns the placeholder for a value, or the value itself when it
// is an Expression.
//
// The placeholder is "?". A grammar for an engine that numbers its
// placeholders overrides this, and overrides it knowing the number comes from
// the position in the binding list rather than from the value.
func (g *BaseGrammar) Parameter(value any) string {
	if IsExpression(value) {
		return stringify(value)
	}
	return "?"
}

// QuoteString quotes a string literal, or each one in a list.
func (g *BaseGrammar) QuoteString(value any) string {
	if values, ok := value.([]any); ok {
		out := make([]string, len(values))
		for i, v := range values {
			out[i] = g.QuoteString(v)
		}
		return strings.Join(out, ", ")
	}
	return "'" + strings.ReplaceAll(stringify(value), "'", "''") + "'"
}

// Escape returns a value escaped for inclusion directly in SQL.
//
// The base grammar has no connection to escape through, so it refuses. A
// driver grammar that can escape overrides it. This is deliberately not a
// best-effort quote: a caller reaching for Escape is building SQL by hand, and
// a value that looks escaped but is not is worse than one that refuses.
func (g *BaseGrammar) Escape(value any, binary bool) (string, error) {
	return "", fmt.Errorf("query: the base grammar has no connection to escape through; use a parameter placeholder")
}

// GetDateFormat returns the layout the engine's date columns are formatted
// with, in Go's reference-time syntax.
func (g *BaseGrammar) GetDateFormat() string { return "2006-01-02 15:04:05" }

// SupportsSavepoints reports whether the engine can savepoint within a
// transaction.
func (g *BaseGrammar) SupportsSavepoints() bool { return true }

// CompileSavepoint compiles a statement that creates the named savepoint.
func (g *BaseGrammar) CompileSavepoint(name string) string { return "SAVEPOINT " + name }

// CompileSavepointRollBack compiles a statement that rolls back to the named
// savepoint.
func (g *BaseGrammar) CompileSavepointRollBack(name string) string {
	return "ROLLBACK TO SAVEPOINT " + name
}

// CompileRandom compiles a random ordering expression, optionally seeded.
func (g *BaseGrammar) CompileRandom(seed string) string { return "RANDOM()" }

// wrapValue quotes one identifier segment. The base grammar uses the SQL
// standard double quote; MySQL overrides it with a backtick.
//
// A quote character inside the identifier is doubled rather than stripped,
// because stripping it would silently rename the column.
func (g *BaseGrammar) wrapValue(value string) string {
	if value == "*" {
		return value
	}
	return `"` + strings.ReplaceAll(value, `"`, `""`) + `"`
}

// wrapSegments quotes each dotted segment of a qualified name.
func (g *BaseGrammar) wrapSegments(name string) string {
	segments := strings.Split(name, ".")
	out := make([]string, len(segments))
	for i, segment := range segments {
		out[i] = g.wrapValue(segment)
	}
	return strings.Join(out, ".")
}

// aliasIndex reports where the " as " of an aliased name starts, or -1. The
// search is case-insensitive.
func aliasIndex(name string) int {
	return strings.Index(lower(name), " as ")
}
