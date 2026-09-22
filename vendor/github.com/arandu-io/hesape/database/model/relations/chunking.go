package relations

import (
	"context"
	"errors"
	"fmt"
	"iter"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/pagination"
)

// errChunkSize is a chunk of nothing, which would loop forever asking for zero
// rows. PHP divides by the count and dies on the division; this says so.
var errChunkSize = errors.New("relations: a chunk size has to be at least 1")

// errChunkKeyMissing is the id-paged walk being unable to find the column it
// pages by in the rows that came back.
//
// It is fatal rather than ignorable: without the last id there is no boundary
// for the next page, and carrying on would either loop over page one forever or
// silently fall back to an offset, which is the failure mode chunkById exists
// to avoid.
func errChunkKeyMissing(alias string) error {
	return fmt.Errorf("relations: the rows of this chunk carry no %q, so there is no id to page after -- name the column and its alias explicitly", alias)
}

// This file holds the walking half of BelongsToMany and HasOneOrManyThrough:
// Chunk, ChunkByID, ChunkByIDDesc, OrderedChunkByID, Each, EachByID, Lazy,
// LazyByID, LazyByIDDesc, Cursor, Paginate, SimplePaginate and CursorPaginate.
//
// The two relations want the same thirteen methods, differing in one line:
// BelongsToMany hydrates the pivot on every page and the through relation does
// not. So the bodies live here once, take the prepared query and a per-page
// hook, and each relation's method is the two lines that supply those.
//
// Every one of them takes ctx and an auth.Grant and reaches the database
// through the relation's scoped query. A chunked read is a read, and a walk that
// authorized the first page and not the rest would leak on page two, where
// nobody is looking.

// chunker is a relation's prepared query plus what it does to each page.
//
// prepare answers the query with the relation's select applied, already
// narrowed to the Grant's tenant. hydrate is BelongsToMany's pivot hydration,
// and is nil for a relation that has none.
type chunker struct {
	prepare func(g auth.Grant) (Builder, error)
	hydrate func(models []Model)
	keyName func() string
}

func (c chunker) page(models []Model) []Model {
	if c.hydrate != nil {
		c.hydrate(models)
	}
	return models
}

// chunk answers BuildsQueries::chunk as the two relations reach it: pages of
// count rows, walked until the callback answers false or a short page ends it.
//
// The order is the caller's problem in PHP, which throws when there is none.
// Here the query is ordered by the related key when nothing else ordered it,
// because a chunked read of an unordered result set is allowed by every engine
// to return a row twice and skip another -- and it does so under load, which is
// where it is hardest to reproduce.
func (c chunker) chunk(ctx context.Context, g auth.Grant, count int, callback func(results []Model, page int) bool) (bool, error) {
	if count < 1 {
		return false, errChunkSize
	}

	for page := 1; ; page++ {
		q, err := c.prepare(g)
		if err != nil {
			return false, err
		}

		results, err := q.OrderBy(c.keyName()).Offset((page-1)*count).Limit(count).Get(ctx, g)
		if err != nil {
			return false, err
		}
		if len(results) == 0 {
			return true, nil
		}

		if !callback(c.page(results), page) {
			return false, nil
		}
		if len(results) < count {
			return true, nil
		}
	}
}

// orderedChunkByID answers BuildsQueries::orderedChunkById: pages taken by
// comparing against the last id seen rather than by an offset.
//
// It is the one to reach for when the walk writes to the rows it is walking. An
// offset-paged chunk that deletes or reorders rows skips the row that slid into
// the offset it already passed; comparing against the last id cannot, because
// the boundary moves with the data.
func (c chunker) orderedChunkByID(ctx context.Context, g auth.Grant, count int, callback func(results []Model, page int) bool, column, alias string, descending bool) (bool, error) {
	if count < 1 {
		return false, errChunkSize
	}
	if column == "" {
		column = c.keyName()
	}
	if alias == "" {
		alias = lastSegment(column)
	}

	var lastID any
	for page := 1; ; page++ {
		q, err := c.prepare(g)
		if err != nil {
			return false, err
		}

		if lastID != nil {
			if descending {
				q = q.Where(column, "<", lastID)
			} else {
				q = q.Where(column, ">", lastID)
			}
		}

		direction := "asc"
		if descending {
			direction = "desc"
		}

		results, err := q.OrderBy(column, direction).Limit(count).Get(ctx, g)
		if err != nil {
			return false, err
		}
		if len(results) == 0 {
			return true, nil
		}

		hydrated := c.page(results)
		lastID = hydrated[len(hydrated)-1].GetAttribute(alias)
		if lastID == nil {
			return false, errChunkKeyMissing(alias)
		}

		if !callback(hydrated, page) {
			return false, nil
		}
		if len(results) < count {
			return true, nil
		}
	}
}

// each answers BuildsQueries::each: chunk, and hand the callback one row at a
// time with its index across the whole walk rather than within the page.
func (c chunker) each(ctx context.Context, g auth.Grant, callback func(value Model, key int) bool, count int) (bool, error) {
	return c.chunk(ctx, g, count, func(results []Model, page int) bool {
		for index, value := range results {
			if !callback(value, (page-1)*count+index) {
				return false
			}
		}
		return true
	})
}

// lazy answers BuildsQueries::lazy: the same walk as chunk, handed back as a
// sequence so the caller writes a range loop instead of a callback.
//
// The error travels beside the value because a walk can fail on page four,
// after three pages have already been handed over and acted on.
func (c chunker) lazy(ctx context.Context, g auth.Grant, chunkSize int, byID bool, column, alias string, descending bool) iter.Seq2[Model, error] {
	return func(yield func(Model, error) bool) {
		stopped := false

		emit := func(results []Model, _ int) bool {
			for _, model := range results {
				if !yield(model, nil) {
					stopped = true
					return false
				}
			}
			return true
		}

		var err error
		if byID {
			_, err = c.orderedChunkByID(ctx, g, chunkSize, emit, column, alias, descending)
		} else {
			_, err = c.chunk(ctx, g, chunkSize, emit)
		}

		if err != nil && !stopped {
			yield(nil, err)
		}
	}
}

// cursor answers Builder::cursor for a relation: the rows streamed, with the
// relation's select applied and the pivot hydrated one row at a time.
//
// Unlike lazy it does not page. One statement stays open and the driver hands
// rows over as they arrive, which is what makes it the cheaper of the two on a
// read that runs to the end -- and the more expensive one to abandon halfway.
func (c chunker) cursor(ctx context.Context, g auth.Grant) iter.Seq2[Model, error] {
	return func(yield func(Model, error) bool) {
		q, err := c.prepare(g)
		if err != nil {
			yield(nil, err)
			return
		}

		for model, err := range q.Cursor(ctx, g) {
			if err != nil {
				yield(nil, err)
				return
			}
			if !yield(c.page([]Model{model})[0], nil) {
				return
			}
		}
	}
}

// paginate answers Builder::paginate for a relation.
func (c chunker) paginate(ctx context.Context, g auth.Grant, perPage, page int, opts pagination.Options, columns ...any) (*pagination.LengthAwarePaginator[Model], error) {
	q, err := c.prepare(g)
	if err != nil {
		return nil, err
	}

	paginator, err := q.Paginate(ctx, g, perPage, page, opts, columns...)
	if err != nil {
		return nil, err
	}
	c.page(paginator.Items())
	return paginator, nil
}

// simplePaginate answers Builder::simplePaginate for a relation.
func (c chunker) simplePaginate(ctx context.Context, g auth.Grant, perPage, page int, opts pagination.Options, columns ...any) (*pagination.Paginator[Model], error) {
	q, err := c.prepare(g)
	if err != nil {
		return nil, err
	}

	paginator, err := q.SimplePaginate(ctx, g, perPage, page, opts, columns...)
	if err != nil {
		return nil, err
	}
	c.page(paginator.Items())
	return paginator, nil
}

// cursorPaginate answers Builder::cursorPaginate for a relation.
func (c chunker) cursorPaginate(ctx context.Context, g auth.Grant, perPage int, cursor *pagination.Cursor, opts pagination.Options, columns ...any) (*pagination.CursorPaginator[Model], error) {
	q, err := c.prepare(g)
	if err != nil {
		return nil, err
	}

	paginator, err := q.CursorPaginate(ctx, g, perPage, cursor, opts, columns...)
	if err != nil {
		return nil, err
	}
	c.page(paginator.Items())
	return paginator, nil
}

// lastSegment answers the column name without its table qualifier, which is the
// key the row comes back under and therefore the alias to read the last id from.
func lastSegment(column string) string {
	for i := len(column) - 1; i >= 0; i-- {
		if column[i] == '.' {
			return column[i+1:]
		}
	}
	return column
}
