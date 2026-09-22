package model

import (
	"context"
	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/database/query"
)

// The where clauses and the aggregates that the base builder had and this one
// did not.
//
// Where, OrWhere, WhereNot and their neighbours were forwarded; WhereIn,
// WhereNull, WhereBetween and the four aggregates were not. The only way to
// reach them was GetQuery(), which hands back the base builder and ends the
// typed chain -- so a query as ordinary as "these three ids" could not be
// written without leaving the API it was written in.
//
// Each of these is the same two lines: hand the clause to the base builder,
// return the typed one so the chain continues.

// WhereIn adds a where-in clause.
func (b *Builder[T]) WhereIn(column any, values []any) *Builder[T] {
	b.query.WhereIn(column, values)
	return b
}

// WhereNotIn adds a where-not-in clause.
func (b *Builder[T]) WhereNotIn(column any, values []any) *Builder[T] {
	b.query.WhereNotIn(column, values)
	return b
}

// WhereNull adds a where-null clause for each column named.
func (b *Builder[T]) WhereNull(columns ...any) *Builder[T] {
	b.query.WhereNull(columns...)
	return b
}

// WhereNotNull adds a where-not-null clause for each column named.
func (b *Builder[T]) WhereNotNull(columns ...any) *Builder[T] {
	b.query.WhereNotNull(columns...)
	return b
}

// WhereBetween adds a where-between clause.
func (b *Builder[T]) WhereBetween(column any, from, to any) *Builder[T] {
	b.query.WhereBetween(column, from, to)
	return b
}

// WhereExists adds a where-exists clause built by callback.
//
// The callback takes the base builder rather than this one, and that is not an
// oversight: the subquery of an exists names another table, so a builder typed
// on T would be the wrong type for it. Has and WhereHas are the typed way to
// ask the same question about a relation, and they are what most callers want.
func (b *Builder[T]) WhereExists(callback func(*query.Builder)) *Builder[T] {
	b.query.WhereExists(callback, "and", false)
	return b
}

// WhereNotExists adds a where-not-exists clause built by callback.
func (b *Builder[T]) WhereNotExists(callback func(*query.Builder)) *Builder[T] {
	b.query.WhereExists(callback, "and", true)
	return b
}

// DoesntExist reports whether the query matches no rows.
//
// It is Exists read the other way round, and it exists as its own method for
// the reason the base builder has one: `if !exists` after a call that also
// returns an error reads as a mistake even when it is not.
func (b *Builder[T]) DoesntExist(ctx context.Context, g auth.Grant) (bool, error) {
	exists, err := b.Exists(ctx, g)
	if err != nil {
		return false, err
	}
	return !exists, nil
}

// Sum returns the sum of column over the matching rows.
func (b *Builder[T]) Sum(ctx context.Context, g auth.Grant, column any) (any, error) {
	return b.Aggregate(ctx, g, "sum", column)
}

// Avg returns the average of column over the matching rows.
func (b *Builder[T]) Avg(ctx context.Context, g auth.Grant, column any) (any, error) {
	return b.Aggregate(ctx, g, "avg", column)
}

// Min returns the smallest value of column over the matching rows.
func (b *Builder[T]) Min(ctx context.Context, g auth.Grant, column any) (any, error) {
	return b.Aggregate(ctx, g, "min", column)
}

// Max returns the largest value of column over the matching rows.
func (b *Builder[T]) Max(ctx context.Context, g auth.Grant, column any) (any, error) {
	return b.Aggregate(ctx, g, "max", column)
}

// WhereColumn compares two columns.
func (b *Builder[T]) WhereColumn(first any, args ...any) *Builder[T] {
	b.query.WhereColumn(first, args...)
	return b
}

// Join adds an inner join.
func (b *Builder[T]) Join(table any, first any, args ...any) *Builder[T] {
	b.query.Join(table, first, args...)
	return b
}

// GroupBy groups the rows.
func (b *Builder[T]) GroupBy(groups ...any) *Builder[T] {
	b.query.GroupBy(groups...)
	return b
}
