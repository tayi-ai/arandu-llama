package model

import (
	"context"
	"fmt"
	"strings"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/database/query"
	"github.com/arandu-io/hesape/pagination"
)

// Paginate runs the query for one page and returns a length-aware
// paginator.
//
// The page number is an argument: no request is reachable from here, and the
// caller reads it with pagination.ResolveCurrentPage.
//
// perPage of zero means the model's own.
func (b *Builder[T]) Paginate(ctx context.Context, g auth.Grant, perPage, page int, opts pagination.Options, columns ...any) (*pagination.LengthAwarePaginator[*T], error) {
	return paginateAs(b, ctx, g, perPage, page, opts, entityOf[T], columns...)
}

// paginateAs is Paginate with the item type left open.
//
// It exists because the same page has to be answered as *T to the caller who
// typed the query and as a relation's Model to the relation seam, and a Go
// method cannot introduce a type parameter of its own. Duplicating the body was
// the alternative, and a paginator that counts differently in two places is the
// bug that would follow.
func paginateAs[T, R any](b *Builder[T], ctx context.Context, g auth.Grant, perPage, page int, opts pagination.Options, convert func(*Model[T]) R, columns ...any) (*pagination.LengthAwarePaginator[R], error) {
	if perPage <= 0 {
		perPage = b.model.GetPerPage()
	}
	if page < 1 {
		page = 1
	}

	total, err := b.GetCountForPagination(ctx, g)
	if err != nil {
		return nil, err
	}

	items := models[T]{}
	if total > 0 {
		items, err = b.clone().ForPage(page, perPage).get(ctx, g, columns...)
		if err != nil {
			return nil, err
		}
	}
	return pagination.Paginate(convertAll(items, convert), int(total), perPage, page, opts), nil
}

// convertAll maps a page of models through the conversion its caller asked for.
func convertAll[T, R any](items models[T], convert func(*Model[T]) R) []R {
	out := make([]R, 0, len(items))
	for _, item := range items {
		out = append(out, convert(item))
	}
	return out
}

// SimplePaginate returns one page and whether there is another, without the
// count.
func (b *Builder[T]) SimplePaginate(ctx context.Context, g auth.Grant, perPage, page int, opts pagination.Options, columns ...any) (*pagination.Paginator[*T], error) {
	return simplePaginateAs(b, ctx, g, perPage, page, opts, entityOf[T], columns...)
}

// simplePaginateAs is SimplePaginate with the item type left open. See
// paginateAs.
func simplePaginateAs[T, R any](b *Builder[T], ctx context.Context, g auth.Grant, perPage, page int, opts pagination.Options, convert func(*Model[T]) R, columns ...any) (*pagination.Paginator[R], error) {
	if perPage <= 0 {
		perPage = b.model.GetPerPage()
	}
	if page < 1 {
		page = 1
	}

	items, err := b.clone().
		Offset((page-1)*perPage).
		Limit(perPage+1).
		get(ctx, g, columns...)
	if err != nil {
		return nil, err
	}
	return pagination.SimplePaginate(convertAll(items, convert), perPage, page, opts), nil
}

// GetCountForPagination returns the row count of the query, ignoring its
// order, limit and offset.
//
// The orders, the limit and the offset come off before the count, because a
// count with a limit on it counts the page rather than the result set.
func (b *Builder[T]) GetCountForPagination(ctx context.Context, g auth.Grant) (int64, error) {
	counted := b.clone()
	counted.query = counted.query.CloneWithout("columns", "orders", "limit", "offset")
	return counted.Count(ctx, g)
}

// CursorPaginate returns the page after (or before) a boundary named by
// cursor rather than by offset.
//
// It is the paginator to reach for past the first few pages: ForPage makes the
// engine count and discard every row it skips, and a row inserted between two
// requests shifts the boundary so that a row is never shown.
//
// cursor is nil for the first page. The columns the query orders by are the
// cursor's parameters, so every one of them has to be selected -- the cursor is
// built out of the rows that come back.
func (b *Builder[T]) CursorPaginate(ctx context.Context, g auth.Grant, perPage int, cursor *pagination.Cursor, opts pagination.Options, columns ...any) (*pagination.CursorPaginator[*T], error) {
	return cursorPaginateAs(b, ctx, g, perPage, cursor, opts, entityOf[T], columns...)
}

// cursorPaginateAs is CursorPaginate with the item type left open. See
// paginateAs.
func cursorPaginateAs[T any, R comparable](b *Builder[T], ctx context.Context, g auth.Grant, perPage int, cursor *pagination.Cursor, opts pagination.Options, convert func(*Model[T]) R, columns ...any) (*pagination.CursorPaginator[R], error) {
	if perPage <= 0 {
		perPage = b.model.GetPerPage()
	}
	if len(b.query.Unions) > 0 {
		return nil, fmt.Errorf("model: cursor pagination over a union is not supported: the boundary conditions have to be repeated inside every branch, and this builder has no way to reach them")
	}

	paginated := b.clone()
	orders := paginated.ensureOrderForCursorPagination(cursor != nil && cursor.PointsToPreviousItems())
	if len(orders) == 0 {
		return nil, fmt.Errorf("model: cursor pagination needs an order it can compare against")
	}

	if cursor != nil {
		if err := addCursorConditions(paginated.query, *cursor, orders, 0); err != nil {
			return nil, err
		}
	}

	items, err := paginated.Limit(perPage+1).get(ctx, g, columns...)
	if err != nil {
		return nil, err
	}

	parameters := make([]string, 0, len(orders))
	for _, order := range orders {
		parameters = append(parameters, order.column)
	}

	// The cursor is built here, off the models, and the paginator reads it back
	// by the item it holds. It is not read off the item itself because the item
	// is whatever R the caller converted to: a row of a T that does not embed
	// Model[T] has fields and no GetAttribute, and there is no constraint to
	// write that would say otherwise for a T the caller chooses.
	//
	// That is the whole of the constraint on R: the paginator asks this map a
	// question about an item it was handed, so the item has to be a key.
	converted := make([]R, 0, len(items))
	cursors := make(map[R]map[string]string, len(items))
	for _, item := range items {
		row := convert(item)
		converted = append(converted, row)

		values := make(map[string]string, len(parameters))
		for _, parameter := range parameters {
			values[parameter] = fmt.Sprint(item.GetAttribute(afterLastDot(parameter)))
		}
		cursors[row] = values
	}
	key := func(item R) map[string]string { return cursors[item] }
	return pagination.CursorPaginate(converted, perPage, cursor, key, opts), nil
}

// cursorOrder is one entry of what ensureOrderForCursorPagination returns: a
// column the cursor compares against, and the direction it is read in.
type cursorOrder struct {
	column    string
	direction string
}

// ensureOrderForCursorPagination returns the columns to compare the cursor
// against, reversing their direction first when walking backward.
func (b *Builder[T]) ensureOrderForCursorPagination(shouldReverse bool) []cursorOrder {
	b.enforceOrderBy()

	if shouldReverse {
		for i := range b.query.Orders {
			b.query.Orders[i].Direction = flipDirection(b.query.Orders[i].Direction)
		}
		for i := range b.query.UnionOrders {
			b.query.UnionOrders[i].Direction = flipDirection(b.query.UnionOrders[i].Direction)
		}
	}

	orders := b.query.Orders
	if len(b.query.UnionOrders) > 0 {
		orders = b.query.UnionOrders
	}

	out := make([]cursorOrder, 0, len(orders))
	for _, order := range orders {
		if order.Direction == "" || order.Column == nil {
			// An orderByRaw has no direction to compare against, so it is
			// filtered out before the conditions are built.
			continue
		}
		out = append(out, cursorOrder{column: fmt.Sprint(order.Column), direction: order.Direction})
	}
	return out
}

func flipDirection(direction string) string {
	if direction == "asc" {
		return "desc"
	}
	return "asc"
}

// addCursorConditions adds the where clauses that skip every row up to and
// including the cursor's position.
//
// It reads: past the first ordering column, every earlier one has to be
// equal, and this one has to be past the boundary -- or, if there is
// another column after it, equal here and past the boundary there. That
// nesting is what makes a compound cursor skip exactly the rows already
// seen.
func addCursorConditions(q *query.Builder, cursor pagination.Cursor, orders []cursorOrder, i int) error {
	if i > 0 {
		previous := orders[i-1].column
		value, err := cursor.Parameter(previous)
		if err != nil {
			return err
		}
		q.Where(cursorColumn(previous), "=", value)
	}

	order := orders[i]
	value, err := cursor.Parameter(order.column)
	if err != nil {
		return err
	}

	operator := ">"
	if order.direction != "asc" {
		operator = "<"
	}

	var inner error
	q.Where(func(nested *query.Builder) {
		nested.Where(cursorColumn(order.column), operator, value)
		if i < len(orders)-1 {
			nested.OrWhere(func(deeper *query.Builder) {
				inner = addCursorConditions(deeper, cursor, orders, i+1)
			})
		}
	})
	return inner
}

// cursorColumn returns column unchanged when it is a plain name, or wrapped
// as a raw expression when it looks like one, so that ordering by a
// function still compares against the same function.
func cursorColumn(column string) any {
	if strings.ContainsAny(column, "()") {
		return query.Raw(column)
	}
	return column
}
