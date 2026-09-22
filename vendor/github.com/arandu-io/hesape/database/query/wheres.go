package query

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/arandu-io/hesape/str"
)

// The rest of the where vocabulary: the clause types the grammar knows how to
// compile and the "or" twin of each one.
//
// The conjunction and the negation are in the method name rather than in a
// trailing argument, because Go has no default argument. Nothing in this file
// reaches the connection: a where clause is state, and the tenant is put on the
// statement at execution, in scoped.

// PrepareValueAndOperator resolves the value and operator for a where clause
// that may have been built from two arguments or three.
//
// When useDefault is true, operator is treated as the value and the
// comparison operator defaults to "="; otherwise value and operator are
// returned as given, after checking the combination.
//
// It returns an error for the combination it refuses: an operator with no
// value, such as where('votes', '>'), which would otherwise compile to a
// comparison against NULL and quietly match nothing.
//
// The fluent methods do not call it; they use the unexported
// prepareValueAndOperator, which reads the same two-or-three-argument
// distinction off the length of a variadic slice. This one is exported
// because a caller assembling an operator and a value from user input has the
// same combination to validate.
//
// It screens and reports; it does not record the refusal on the builder. A
// caller validating a combination is expected to handle the error and fall
// back to something safe, and a builder that had been disabled by the question
// would drop every clause added after it. Nothing is given up by staying
// quiet: the operator returned here is not what reaches SQL. Where, Having and
// On screen what they are given, and the final barrier screens the whole graph
// again against the compiling grammar before a statement is built.
func (b *Builder) PrepareValueAndOperator(value, operator any, useDefault bool) (any, string, error) {
	if useDefault {
		return operator, "=", nil
	}
	canonical, err := normalizeOperator(b.Grammar, nil, stringify(operator))
	if err != nil {
		return nil, "", err
	}
	if invalidOperatorAndValue(canonical, value) {
		return nil, "", fmt.Errorf("query: illegal operator and value combination")
	}
	return value, canonical, nil
}

// invalidOperatorAndValue reports whether operator is one that needs a value,
// with no value given.
func invalidOperatorAndValue(operator string, value any) bool {
	if value != nil {
		return false
	}
	switch operator {
	case "=", "<>", "!=":
		return false
	}
	return containsFold(operators, operator)
}

// containsFold reports whether the list contains value without changing the
// caller's spelling.
func containsFold(list []string, value string) bool {
	for _, item := range list {
		if strings.EqualFold(item, value) {
			return true
		}
	}
	return false
}

// flattenValue returns the first leaf of a value that arrived as a list.
func flattenValue(value any) any {
	values, ok := value.([]any)
	if !ok || len(values) == 0 {
		return value
	}
	return flattenValue(values[0])
}

// refuse records a clause that could not be built as a false one carrying the
// reason, which is what the grammars do with a clause they cannot compile.
//
// A fluent method has no error to return, and the two alternatives are worse:
// dropping the clause widens the query to rows nobody filtered for, and
// panicking takes down a request over a query that was about to be refused
// anyway.
func (b *Builder) refuse(boolean string, err error) *Builder {
	b.Wheres = append(b.Wheres, Where{
		Type:    "Raw",
		SQL:     "1 = 0 /* " + strings.ReplaceAll(err.Error(), "*/", "* /") + " */",
		Boolean: boolean,
	})
	return b
}

// MergeWheres appends another query's where clauses and bindings onto this
// one.
//
// The bindings are appended to the where segment as given. They are the
// caller's to line up with the clauses: the two arguments are the two halves
// of somebody else's query, and nothing here can check that they match.
func (b *Builder) MergeWheres(wheres []Where, bindings []any) *Builder {
	b.Wheres = append(b.Wheres, wheres...)
	b.Bindings["where"] = append(b.Bindings["where"], bindings...)
	return b
}

// OrWhereColumn adds an "or" clause comparing two columns.
func (b *Builder) OrWhereColumn(first any, args ...any) *Builder {
	operator, second := prepareValueAndOperator(args...)
	operator = b.acceptOperator(operator)
	if b.Err() != nil {
		return b
	}
	b.Wheres = append(b.Wheres, Where{
		Type:     "Column",
		First:    first,
		Operator: operator,
		Second:   second,
		Boolean:  "or",
	})
	return b
}

// OrWhereExists adds an "or exists" clause for the subquery the callback
// builds, optionally negated.
func (b *Builder) OrWhereExists(callback func(*Builder), not ...bool) *Builder {
	return b.WhereExists(callback, "or", len(not) > 0 && not[0])
}

// OrWhereNotExists adds an "or not exists" clause for the subquery the
// callback builds.
func (b *Builder) OrWhereNotExists(callback func(*Builder)) *Builder {
	return b.OrWhereExists(callback, true)
}

// WhereLike adds a LIKE clause comparing column against a pattern, optionally
// case sensitive.
//
// The clause carries the value that is bound rather than the pattern that was
// written, because a grammar may rewrite it -- SQLite spells a case sensitive
// like as a glob, whose wildcards are the other way round -- and the tenant
// scoping rebuilds the binding list from the clauses. A clause that did not
// carry its own value would lose it there.
func (b *Builder) WhereLike(column any, value any, caseSensitive bool, boolean string, not bool) *Builder {
	if boolean == "" {
		boolean = "and"
	}
	if grammar, ok := b.Grammar.(LikeBindingGrammar); ok {
		value = grammar.PrepareWhereLikeBinding(stringify(value), caseSensitive)
	}
	b.Wheres = append(b.Wheres, Where{
		Type:          "Like",
		Column:        column,
		Value:         value,
		CaseSensitive: caseSensitive,
		Boolean:       boolean,
		Not:           not,
	})
	b.AddBinding([]any{value}, "where")
	return b
}

// LikeBindingGrammar is the part of a grammar that rewrites a like pattern for
// the engine. A grammar that does not implement it leaves the pattern alone.
type LikeBindingGrammar interface {
	// PrepareWhereLikeBinding rewrites a like pattern into the binding the
	// grammar's engine expects.
	PrepareWhereLikeBinding(value string, caseSensitive bool) string
}

// OrWhereLike adds an "or" LIKE clause.
func (b *Builder) OrWhereLike(column any, value any, caseSensitive bool) *Builder {
	return b.WhereLike(column, value, caseSensitive, "or", false)
}

// WhereNotLike adds a NOT LIKE clause.
func (b *Builder) WhereNotLike(column any, value any, caseSensitive bool, boolean string) *Builder {
	return b.WhereLike(column, value, caseSensitive, boolean, true)
}

// OrWhereNotLike adds an "or" NOT LIKE clause.
func (b *Builder) OrWhereNotLike(column any, value any, caseSensitive bool) *Builder {
	return b.WhereNotLike(column, value, caseSensitive, "or")
}

// WhereIntegerInRaw adds a where-in clause whose values are written directly
// into the statement as integers rather than bound.
//
// Every value is cast to an integer first: an integer has no quoting to
// escape. A value that is not a number at all becomes zero rather than
// reaching the statement as itself.
func (b *Builder) WhereIntegerInRaw(column any, values []any, boolean string, not bool) *Builder {
	if boolean == "" {
		boolean = "and"
	}
	integers := make([]any, 0, len(values))
	for _, value := range values {
		number, _ := castNumber(value)
		integers = append(integers, int64(number))
	}
	b.Wheres = append(b.Wheres, Where{
		Type: "InRaw", Column: column, Values: integers, Boolean: boolean, Not: not,
	})
	return b
}

// OrWhereIntegerInRaw adds an "or" where-in clause with raw integer values.
func (b *Builder) OrWhereIntegerInRaw(column any, values []any) *Builder {
	return b.WhereIntegerInRaw(column, values, "or", false)
}

// WhereIntegerNotInRaw adds a where-not-in clause with raw integer values.
func (b *Builder) WhereIntegerNotInRaw(column any, values []any, boolean string) *Builder {
	return b.WhereIntegerInRaw(column, values, boolean, true)
}

// OrWhereIntegerNotInRaw adds an "or" where-not-in clause with raw integer
// values.
func (b *Builder) OrWhereIntegerNotInRaw(column any, values []any) *Builder {
	return b.WhereIntegerNotInRaw(column, values, "or")
}

// WhereBetweenColumns adds a clause requiring column to be between two other
// columns, so nothing here is bound.
func (b *Builder) WhereBetweenColumns(column any, values []any, boolean string, not bool) *Builder {
	if boolean == "" {
		boolean = "and"
	}
	b.Wheres = append(b.Wheres, Where{
		Type: "betweenColumns", Column: column, Values: values, Boolean: boolean, Not: not,
	})
	return b
}

// OrWhereBetweenColumns adds an "or" between-columns clause.
func (b *Builder) OrWhereBetweenColumns(column any, values []any) *Builder {
	return b.WhereBetweenColumns(column, values, "or", false)
}

// WhereNotBetweenColumns adds a not-between-columns clause.
func (b *Builder) WhereNotBetweenColumns(column any, values []any, boolean string) *Builder {
	return b.WhereBetweenColumns(column, values, boolean, true)
}

// OrWhereNotBetweenColumns adds an "or" not-between-columns clause.
func (b *Builder) OrWhereNotBetweenColumns(column any, values []any) *Builder {
	return b.WhereNotBetweenColumns(column, values, "or")
}

// WhereValueBetween adds a clause requiring value to fall between two
// columns, which is the between comparison the other way round.
func (b *Builder) WhereValueBetween(value any, columns []any, boolean string, not bool) *Builder {
	if boolean == "" {
		boolean = "and"
	}
	b.Wheres = append(b.Wheres, Where{
		Type: "valueBetween", Value: value, Columns: columns, Boolean: boolean, Not: not,
	})
	b.AddBinding([]any{value}, "where")
	return b
}

// OrWhereValueBetween adds an "or" value-between-columns clause.
func (b *Builder) OrWhereValueBetween(value any, columns []any) *Builder {
	return b.WhereValueBetween(value, columns, "or", false)
}

// WhereValueNotBetween adds a value-not-between-columns clause.
func (b *Builder) WhereValueNotBetween(value any, columns []any, boolean string) *Builder {
	return b.WhereValueBetween(value, columns, boolean, true)
}

// OrWhereValueNotBetween adds an "or" value-not-between-columns clause.
func (b *Builder) OrWhereValueNotBetween(value any, columns []any) *Builder {
	return b.WhereValueNotBetween(value, columns, "or")
}

// WhereDate adds a clause comparing the date part of a timestamp column with
// a date.
//
// The operator is optional, as it is on Where: two arguments are a column and
// a value compared with "=". A time.Time value is formatted down to the day.
func (b *Builder) WhereDate(column any, args ...any) *Builder {
	return b.addDateBasedWhere("Date", column, "and", args...)
}

// OrWhereDate adds an "or" date clause.
func (b *Builder) OrWhereDate(column any, args ...any) *Builder {
	return b.addDateBasedWhere("Date", column, "or", args...)
}

// WhereTime adds a clause comparing the time part of a timestamp column with
// a time.
func (b *Builder) WhereTime(column any, args ...any) *Builder {
	return b.addDateBasedWhere("Time", column, "and", args...)
}

// OrWhereTime adds an "or" time clause.
func (b *Builder) OrWhereTime(column any, args ...any) *Builder {
	return b.addDateBasedWhere("Time", column, "or", args...)
}

// WhereDay adds a clause comparing the day part of a timestamp column.
func (b *Builder) WhereDay(column any, args ...any) *Builder {
	return b.addDateBasedWhere("Day", column, "and", args...)
}

// OrWhereDay adds an "or" day clause.
func (b *Builder) OrWhereDay(column any, args ...any) *Builder {
	return b.addDateBasedWhere("Day", column, "or", args...)
}

// WhereMonth adds a clause comparing the month part of a timestamp column.
func (b *Builder) WhereMonth(column any, args ...any) *Builder {
	return b.addDateBasedWhere("Month", column, "and", args...)
}

// OrWhereMonth adds an "or" month clause.
func (b *Builder) OrWhereMonth(column any, args ...any) *Builder {
	return b.addDateBasedWhere("Month", column, "or", args...)
}

// WhereYear adds a clause comparing the year part of a timestamp column.
func (b *Builder) WhereYear(column any, args ...any) *Builder {
	return b.addDateBasedWhere("Year", column, "and", args...)
}

// OrWhereYear adds an "or" year clause.
func (b *Builder) OrWhereYear(column any, args ...any) *Builder {
	return b.addDateBasedWhere("Year", column, "or", args...)
}

// addDateBasedWhere is the shared body of the five date-part clauses: the
// operator shortcut and the formatting of the value, which is the one line
// that differs between them.
func (b *Builder) addDateBasedWhere(typ string, column any, boolean string, args ...any) *Builder {
	operator, value := prepareValueAndOperator(args...)
	operator = b.acceptOperator(operator)
	if b.Err() != nil {
		return b
	}
	value = formatDatePart(typ, flattenValue(value))

	b.Wheres = append(b.Wheres, Where{
		Type: typ, Column: column, Operator: operator, Value: value, Boolean: boolean,
	})
	if !IsExpression(value) {
		b.AddBinding([]any{value}, "where")
	}
	return b
}

// formatDatePart formats value to the part of a timestamp that typ names --
// the date, the time, the day, the month or the year. A day or month that
// arrived as a plain number is zero-padded to two digits.
func formatDatePart(typ string, value any) any {
	if IsExpression(value) {
		return value
	}
	if moment, ok := value.(time.Time); ok {
		switch typ {
		case "Date":
			return moment.Format("2006-01-02")
		case "Time":
			return moment.Format("15:04:05")
		case "Day":
			return moment.Format("02")
		case "Month":
			return moment.Format("01")
		case "Year":
			return moment.Format("2006")
		}
		return value
	}
	if typ != "Day" && typ != "Month" {
		return value
	}
	if number, ok := castNumber(value); ok {
		return fmt.Sprintf("%02d", int64(number))
	}
	return value
}

// WhereRowValues adds a clause comparing a tuple of columns with a tuple of
// values.
func (b *Builder) WhereRowValues(columns []any, operator string, values []any, boolean string) *Builder {
	if boolean == "" {
		boolean = "and"
	}
	if len(columns) != len(values) {
		return b.refuse(boolean, fmt.Errorf(
			"query: the number of columns must match the number of values, got %d and %d",
			len(columns), len(values)))
	}
	operator = b.acceptOperator(operator)
	if b.Err() != nil {
		return b
	}
	b.Wheres = append(b.Wheres, Where{
		Type: "RowValues", Columns: columns, Operator: operator, Values: values, Boolean: boolean,
	})
	b.AddBinding(b.CleanBindings(values), "where")
	return b
}

// OrWhereRowValues adds an "or" row-values clause.
func (b *Builder) OrWhereRowValues(columns []any, operator string, values []any) *Builder {
	return b.WhereRowValues(columns, operator, values, "or")
}

// WhereJSONContains adds a clause requiring the JSON column to contain value.
func (b *Builder) WhereJSONContains(column any, value any, boolean string, not bool) *Builder {
	return b.addWhereJSON("JsonContains", column, value, boolean, not)
}

// OrWhereJSONContains adds an "or" JSON-contains clause.
func (b *Builder) OrWhereJSONContains(column any, value any) *Builder {
	return b.WhereJSONContains(column, value, "or", false)
}

// WhereJSONDoesntContain adds a clause requiring the JSON column not to
// contain value.
func (b *Builder) WhereJSONDoesntContain(column any, value any, boolean string) *Builder {
	return b.WhereJSONContains(column, value, boolean, true)
}

// OrWhereJSONDoesntContain adds an "or" JSON-does-not-contain clause.
func (b *Builder) OrWhereJSONDoesntContain(column any, value any) *Builder {
	return b.WhereJSONDoesntContain(column, value, "or")
}

// WhereJSONOverlaps adds a clause requiring the JSON column to overlap value.
func (b *Builder) WhereJSONOverlaps(column any, value any, boolean string, not bool) *Builder {
	return b.addWhereJSON("JsonOverlaps", column, value, boolean, not)
}

// OrWhereJSONOverlaps adds an "or" JSON-overlaps clause.
func (b *Builder) OrWhereJSONOverlaps(column any, value any) *Builder {
	return b.WhereJSONOverlaps(column, value, "or", false)
}

// WhereJSONDoesntOverlap adds a clause requiring the JSON column not to
// overlap value.
func (b *Builder) WhereJSONDoesntOverlap(column any, value any, boolean string) *Builder {
	return b.WhereJSONOverlaps(column, value, boolean, true)
}

// OrWhereJSONDoesntOverlap adds an "or" JSON-does-not-overlap clause.
func (b *Builder) OrWhereJSONDoesntOverlap(column any, value any) *Builder {
	return b.WhereJSONDoesntOverlap(column, value, "or")
}

// addWhereJSON is the shared body of the contains and overlaps clauses: the
// value is bound as JSON, through the grammar when it has an opinion about how.
func (b *Builder) addWhereJSON(typ string, column any, value any, boolean string, not bool) *Builder {
	if boolean == "" {
		boolean = "and"
	}
	if IsExpression(value) {
		b.Wheres = append(b.Wheres, Where{
			Type: typ, Column: column, Value: value, Boolean: boolean, Not: not,
		})
		return b
	}

	binding, err := prepareBindingForJSONContains(b.Grammar, value)
	if err != nil {
		return b.refuse(boolean, err)
	}
	b.Wheres = append(b.Wheres, Where{
		Type: typ, Column: column, Value: binding, Boolean: boolean, Not: not,
	})
	b.AddBinding([]any{binding}, "where")
	return b
}

// JSONContainsGrammar is the part of a grammar that turns a value into the JSON
// binding its engine expects. It is asked for
// rather than declared on Grammar, for the reason InsertUsingGrammar is.
type JSONContainsGrammar interface {
	// PrepareBindingForJSONContains turns a value into the JSON binding the
	// grammar's engine expects.
	PrepareBindingForJSONContains(binding any) (any, error)
}

// prepareBindingForJSONContains asks the grammar, and encodes the value itself
// when the grammar has nothing to say.
func prepareBindingForJSONContains(grammar Grammar, value any) (any, error) {
	if grammar, ok := grammar.(JSONContainsGrammar); ok {
		return grammar.PrepareBindingForJSONContains(value)
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil, fmt.Errorf("query: the value of a JSON clause cannot be encoded: %w", err)
	}
	return string(encoded), nil
}

// WhereJSONContainsKey adds a clause requiring the JSON column to contain the
// given key path. The key is part of the path, so there is nothing to bind.
func (b *Builder) WhereJSONContainsKey(column any, boolean string, not bool) *Builder {
	if boolean == "" {
		boolean = "and"
	}
	b.Wheres = append(b.Wheres, Where{
		Type: "JsonContainsKey", Column: column, Boolean: boolean, Not: not,
	})
	return b
}

// OrWhereJSONContainsKey adds an "or" JSON-contains-key clause.
func (b *Builder) OrWhereJSONContainsKey(column any) *Builder {
	return b.WhereJSONContainsKey(column, "or", false)
}

// WhereJSONDoesntContainKey adds a clause requiring the JSON column not to
// contain the given key path.
func (b *Builder) WhereJSONDoesntContainKey(column any, boolean string) *Builder {
	return b.WhereJSONContainsKey(column, boolean, true)
}

// OrWhereJSONDoesntContainKey adds an "or" JSON-does-not-contain-key clause.
func (b *Builder) OrWhereJSONDoesntContainKey(column any) *Builder {
	return b.WhereJSONDoesntContainKey(column, "or")
}

// WhereJSONLength adds a clause comparing the length of a JSON column. The
// bound value is an integer: it is compared with a length.
func (b *Builder) WhereJSONLength(column any, args ...any) *Builder {
	return b.addWhereJSONLength(column, "and", args...)
}

// OrWhereJSONLength adds an "or" JSON-length clause.
func (b *Builder) OrWhereJSONLength(column any, args ...any) *Builder {
	return b.addWhereJSONLength(column, "or", args...)
}

func (b *Builder) addWhereJSONLength(column any, boolean string, args ...any) *Builder {
	operator, value := prepareValueAndOperator(args...)
	operator = b.acceptOperator(operator)
	if b.Err() != nil {
		return b
	}

	b.Wheres = append(b.Wheres, Where{
		Type: "JsonLength", Column: column, Operator: operator, Value: value, Boolean: boolean,
	})
	if !IsExpression(value) {
		number, _ := castNumber(flattenValue(value))
		bound := int64(number)
		b.Wheres[len(b.Wheres)-1].Value = bound
		b.AddBinding([]any{bound}, "where")
	}
	return b
}

// WhereFullText adds a full-text search clause over the given columns.
//
// options carries the engine's search modes -- MySQL reads "mode" and
// "expanded", Postgres reads "language" -- and a grammar that has none
// ignores it.
func (b *Builder) WhereFullText(columns []any, value any, options map[string]any, boolean string) *Builder {
	if boolean == "" {
		boolean = "and"
	}
	b.Wheres = append(b.Wheres, Where{
		Type: "Fulltext", Columns: columns, Value: value, Options: options, Boolean: boolean,
	})
	b.AddBinding([]any{value}, "where")
	return b
}

// OrWhereFullText adds an "or" full-text search clause.
func (b *Builder) OrWhereFullText(columns []any, value any, options map[string]any) *Builder {
	return b.WhereFullText(columns, value, options, "or")
}

// WhereAll adds a clause requiring every one of the columns to compare with
// the same value, joined by "and" inside one group.
func (b *Builder) WhereAll(columns []any, args ...any) *Builder {
	return b.addWhereAcross(columns, "and", "and", args...)
}

// OrWhereAll adds an "or" version of WhereAll.
func (b *Builder) OrWhereAll(columns []any, args ...any) *Builder {
	return b.addWhereAcross(columns, "or", "and", args...)
}

// WhereAny adds a clause requiring any one of the columns to compare with the
// same value, joined by "or" inside one group.
func (b *Builder) WhereAny(columns []any, args ...any) *Builder {
	return b.addWhereAcross(columns, "and", "or", args...)
}

// OrWhereAny adds an "or" version of WhereAny.
func (b *Builder) OrWhereAny(columns []any, args ...any) *Builder {
	return b.addWhereAcross(columns, "or", "or", args...)
}

// WhereNone adds a clause requiring none of the columns to compare with the
// same value: WhereAny negated, with the group joined by "and not".
func (b *Builder) WhereNone(columns []any, args ...any) *Builder {
	return b.addWhereAcross(columns, "and not", "or", args...)
}

// OrWhereNone adds an "or" version of WhereNone.
func (b *Builder) OrWhereNone(columns []any, args ...any) *Builder {
	return b.addWhereAcross(columns, "or not", "or", args...)
}

// addWhereAcross is the shared body of WhereAll, WhereAny and WhereNone: one
// nested group holding the same comparison over each column.
//
// The group is what makes the three of them safe next to a tenant filter --
// `tenant_id = ? and (name like ? or email like ?)`.
func (b *Builder) addWhereAcross(columns []any, boolean, inner string, args ...any) *Builder {
	operator, value := prepareValueAndOperator(args...)
	return b.WhereNested(func(query *Builder) {
		for _, column := range columns {
			query.addWhere(inner, column, operator, value)
		}
	}, boolean)
}

// DynamicWhere reads a where clause out of a method name, as in
// "whereNameAndEmail", and binds the values positionally.
//
// The name is an argument rather than something dispatched to, so a caller with
// a column list coming from configuration has one string to turn into clauses.
func (b *Builder) DynamicWhere(method string, parameters []any) *Builder {
	finder := strings.TrimPrefix(method, "where")
	connector := "and"
	index := 0

	for _, segment := range splitDynamicSegments(finder) {
		if segment != "And" && segment != "Or" {
			if index < len(parameters) {
				b.addWhere(lower(connector), str.Snake(segment, "_"), "=", parameters[index])
			}
			index++
			continue
		}
		connector = segment
	}
	return b
}

// splitDynamicSegments splits a dynamic-where method name on "And" and "Or"
// connectors that are followed by an uppercase letter, keeping the
// connectors themselves: the column names and the connectors between them,
// in order.
func splitDynamicSegments(finder string) []string {
	segments := make([]string, 0, 4)
	start := 0
	for i := 0; i < len(finder); i++ {
		for _, connector := range []string{"And", "Or"} {
			if !strings.HasPrefix(finder[i:], connector) {
				continue
			}
			next := i + len(connector)
			if next >= len(finder) || finder[next] < 'A' || finder[next] > 'Z' {
				continue
			}
			if i > start {
				segments = append(segments, finder[start:i])
			}
			segments = append(segments, connector)
			i = next - 1
			start = next
			break
		}
	}
	if start < len(finder) {
		segments = append(segments, finder[start:])
	}
	return segments
}

// WhereVectorDistanceLessThan adds a clause matching rows whose embedding is
// nearer than maxDistance to the given vector.
//
// It takes the vector directly rather than accepting text and computing an
// embedding from it, because a query builder that makes a network call to an
// embedding provider is a query builder that can time out. The caller fetches
// the embedding and passes it in.
//
// Nothing here checks that the connection is Postgres: the operator is
// Postgres's, and a fluent method has nothing to report an error through, so
// an unsupported connection fails when the engine refuses the SQL rather than
// before.
func (b *Builder) WhereVectorDistanceLessThan(column any, vector []float64, maxDistance float64, boolean string) *Builder {
	if boolean == "" {
		boolean = "and"
	}
	encoded, err := json.Marshal(vector)
	if err != nil {
		return b.refuse(boolean, fmt.Errorf("query: the vector cannot be encoded: %w", err))
	}

	sql := "(" + b.wrapColumn(column) + " <=> ?) <= ?"
	bindings := []any{string(encoded), maxDistance}
	b.Wheres = append(b.Wheres, Where{Type: "Raw", SQL: sql, Boolean: boolean, Values: bindings})
	b.AddBinding(bindings, "where")
	return b
}

// OrWhereVectorDistanceLessThan adds an "or" vector-distance clause.
func (b *Builder) OrWhereVectorDistanceLessThan(column any, vector []float64, maxDistance float64) *Builder {
	return b.WhereVectorDistanceLessThan(column, vector, maxDistance, "or")
}

// WhereVectorSimilarTo adds the distance filter and, optionally, the ordering
// that goes with it, given a similarity between 0 and 1.
func (b *Builder) WhereVectorSimilarTo(column any, vector []float64, minSimilarity float64, order bool) *Builder {
	b.WhereVectorDistanceLessThan(column, vector, 1-minSimilarity, "and")
	if order {
		b.OrderByVectorDistance(column, vector)
	}
	return b
}

// wrapColumn quotes a column through the grammar, and leaves it alone when
// there is no grammar to quote it with.
func (b *Builder) wrapColumn(column any) string {
	if b.Grammar == nil {
		return stringify(column)
	}
	return b.Grammar.Wrap(column)
}
