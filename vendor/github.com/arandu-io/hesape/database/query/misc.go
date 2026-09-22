package query

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"

	"github.com/arandu-io/hesape/auth"
)

// The remainder of the query builder: the vector clauses, the insert that
// reports what it wrote, and the dump that prints the statement rather than
// running it.

// SelectVectorDistance adds the distance between a column's embedding and the
// vector given, as a column of the result.
//
// as defaults to the column's name with _distance appended.
//
// Accepting a string as the vector, and turning it into an embedding through a
// model provider, is not supported here, for the reason
// WhereVectorDistanceLessThan gives.
func (b *Builder) SelectVectorDistance(column any, vector []float64, as string) *Builder {
	encoded, err := json.Marshal(vector)
	if err != nil {
		return b.refuse("and", fmt.Errorf("query: the vector cannot be encoded: %w", err))
	}
	if as == "" {
		as = stringify(column) + "_distance"
	}

	b.AddBinding([]any{string(encoded)}, "select")
	return b.AddSelect(Raw("(" + b.wrapColumn(column) + " <=> ?) as " + b.wrapColumn(as)))
}

// OrderByVectorDistance orders by the distance between a column's embedding
// and the vector given, nearest first.
func (b *Builder) OrderByVectorDistance(column any, vector []float64) *Builder {
	encoded, err := json.Marshal(vector)
	if err != nil {
		return b.refuse("and", fmt.Errorf("query: the vector cannot be encoded: %w", err))
	}

	order := Order{SQL: Raw("(" + b.wrapColumn(column) + " <=> ?)"), Direction: "asc"}
	if len(b.Unions) > 0 {
		b.AddBinding([]any{string(encoded)}, "unionOrder")
		b.UnionOrders = append(b.UnionOrders, order)
		return b
	}
	b.AddBinding([]any{string(encoded)}, "order")
	b.Orders = append(b.Orders, order)
	return b
}

// InsertOrIgnoreReturningGrammar is the part of a grammar that compiles an
// insert which skips the rows that would collide and reports the ones it wrote.
// It is asked
// for rather than declared on Grammar, for the reason InsertUsingGrammar is.
type InsertOrIgnoreReturningGrammar interface {
	// CompileInsertOrIgnoreReturning compiles an insert that skips colliding
	// rows and returns the ones it wrote.
	CompileInsertOrIgnoreReturning(query *Builder, values []map[string]any, uniqueBy, returning []string) (string, error)
}

// InsertOrIgnoreReturning inserts rows, letting the engine drop the ones that
// would collide, and returns the rows it kept rather than a count.
//
// The rows carry the tenant of the Grant, as every insert here does, and the
// statement goes to the write connection: the rows it returns were written
// microseconds ago and a replica has not seen them.
func (b *Builder) InsertOrIgnoreReturning(ctx context.Context, g auth.Grant, values []map[string]any, uniqueBy, returning []string) ([]Record, error) {
	if len(values) == 0 {
		return nil, nil
	}
	if len(uniqueBy) == 0 {
		return nil, errors.New("query: the unique columns must not be empty")
	}
	if returning == nil {
		returning = []string{"*"}
	}
	if len(returning) == 0 {
		return nil, errors.New("query: the returning columns must not be empty")
	}

	tenant, err := b.tenantFor(ctx, g)
	if err != nil {
		return nil, err
	}
	rows := stampTenant(values, tenant)

	query := b.Clone()
	query.ApplyBeforeQueryCallbacks()
	if err := query.Err(); err != nil {
		return nil, err
	}

	grammar, ok := query.Grammar.(InsertOrIgnoreReturningGrammar)
	if !ok {
		return nil, errors.New("query: this database engine does not support insert or ignore with returning")
	}
	sql, err := grammar.CompileInsertOrIgnoreReturning(query, rows, uniqueBy, returning)
	if err != nil {
		return nil, err
	}

	if query.connection == nil {
		return nil, errors.New("query: the builder has no connection to run against")
	}
	return query.connection.Select(ctx, sql, query.CleanBindings(flattenRows(rows)), false)
}

// Dump writes the statement and its bindings, and hands the query back so the
// chain continues.
//
// The destination is an argument, as it is on DumpRawSQL.
//
// What it prints is the query as written, without the tenant filter, because
// the filter belongs to the statement and not to the builder. DumpRawSQL takes
// a Grant and prints what would actually run.
func (b *Builder) Dump(w io.Writer, args ...any) *Builder {
	fmt.Fprintln(w, b.ToSQL())
	fmt.Fprintln(w, b.GetBindings()...)
	for _, arg := range args {
		fmt.Fprintln(w, arg)
	}
	return b
}
