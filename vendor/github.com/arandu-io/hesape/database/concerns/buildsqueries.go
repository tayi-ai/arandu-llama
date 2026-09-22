package concerns

import (
	"context"
	"errors"
	"fmt"

	"github.com/arandu-io/hesape/auth"
)

// Chunkable is what the chunking functions ask of the query they walk.
//
// It is an interface rather than a concrete builder, because a function that
// took *query.Builder would only work for that one builder.
//
// T is the row type. A repository knows what it selects, so the chunk arrives
// typed.
//
// Get takes an auth.Grant because it executes. Authorization does not bend for a
// read: List, Find, First, Get, Paginate, an export and a report all pass
// through a Policy and filter by auth.Tenant(g). The Grant travels from the
// caller of Chunk down to every page it fetches, so a chunked read is
// authorized once and stays authorized.
type Chunkable[T any] interface {
	// GetOffset returns the query's offset, or nil when none is set.
	GetOffset() *int

	// GetLimit returns the query's limit, or nil when none is set.
	GetLimit() *int

	// Offset sets the query's offset.
	Offset(value int)

	// Limit sets the query's limit.
	Limit(value int)

	// ForPage sets the offset and limit for one page of results.
	ForPage(page, perPage int)

	// Get runs the query.
	Get(ctx context.Context, g auth.Grant) ([]T, error)

	// EnforceOrderBy fails when the query has no order. Chunking an
	// unordered result set returns rows twice and skips others, and the
	// database is within its rights.
	EnforceOrderBy() error
}

// KeyChunkable is what ChunkByID adds to Chunkable: paging by a comparison
// against the last id seen rather than by an offset.
//
// It is a separate interface because the id-based methods are the ones a
// builder can only offer when it knows its key, and because Chunk works
// without them.
type KeyChunkable[T any] interface {
	Chunkable[T]

	// Clone returns a copy of the query, so the where clause added for the
	// last id does not accumulate across pages.
	Clone() KeyChunkable[T]

	// DefaultKeyName returns the query's own key column.
	DefaultKeyName() string

	// ForPageAfterID adds a where clause that reaches perPage rows after
	// lastID, ordered by column.
	ForPageAfterID(perPage int, lastID any, column string)

	// ForPageBeforeID is ForPageAfterID in reverse order.
	ForPageBeforeID(perPage int, lastID any, column string)

	// ValueOf reads the value named by alias off the last row of a chunk. A
	// row here is a T rather than an untyped map, so the builder that
	// produced it is the only thing that can say which field the alias
	// names.
	ValueOf(row T, alias string) (any, bool)
}

// ErrRecordNotFound is what FirstOrFail returns when nothing matched.
//
// It is declared here rather than in the database package because that package
// imports this one, and Go refuses the cycle. The database package re-exports
// it, so database.ErrRecordNotFound and concerns.ErrRecordNotFound are one
// value under two names.
var ErrRecordNotFound = errors.New("No record found for the given query.")

// ErrRecordsNotFound is what Sole returns when the query matched no row at all.
//
// Sole is the query that asserts "there is exactly one of these", so both ways
// of being wrong are errors: none is this one, more than one is
// MultipleRecordsFoundError. A caller that would rather have a zero value than
// an error wants First, not Sole.
var ErrRecordsNotFound = errors.New("records not found")

// MultipleRecordsFoundError is what Sole returns when the query matched more
// than one row, and it carries how many it saw.
//
// Sole reads two rows and stops, so Count is the number it actually fetched
// rather than the size of the whole result set -- enough to say "this is not
// unique", which is the only thing the caller can act on. The uniqueness the
// query assumed is missing, which is usually a missing unique index or a
// filter that lost a column.
type MultipleRecordsFoundError struct {
	// Count is how many rows the query matched.
	Count int
}

// NewMultipleRecordsFoundError wraps the count of rows a Sole query matched.
func NewMultipleRecordsFoundError(count int) *MultipleRecordsFoundError {
	return &MultipleRecordsFoundError{Count: count}
}

// Error reports how many records were found.
func (e *MultipleRecordsFoundError) Error() string {
	return fmt.Sprintf("%d records were found.", e.Count)
}

// GetCount returns how many rows the query matched.
func (e *MultipleRecordsFoundError) GetCount() int { return e.Count }

// Chunk walks the result set a page at a time so a large table does not have
// to fit in memory.
//
// The callback returns false to stop, and Chunk returns false when it was
// stopped. An unordered query returns the EnforceOrderBy error instead of
// running, because a chunk that silently repeats rows is the failure the
// check exists to prevent.
//
// An existing limit and offset on the query are honoured: the offset shifts
// the first page and the limit bounds the total.
func Chunk[T any](ctx context.Context, g auth.Grant, query Chunkable[T], count int, callback func(results []T, page int) bool) (bool, error) {
	if err := query.EnforceOrderBy(); err != nil {
		return false, err
	}

	skip := 0
	if o := query.GetOffset(); o != nil {
		skip = *o
	}
	remaining := query.GetLimit()

	page := 1
	for {
		offset := (page-1)*count + skip

		limit := count
		if remaining != nil {
			limit = min(count, *remaining)
		}
		if limit == 0 {
			break
		}

		query.Offset(offset)
		query.Limit(limit)

		results, err := query.Get(ctx, g)
		if err != nil {
			return false, err
		}

		countResults := len(results)
		if countResults == 0 {
			break
		}

		if remaining != nil {
			left := max(*remaining-countResults, 0)
			remaining = &left
		}

		if !callback(results, page) {
			return false, nil
		}

		page++

		if countResults != count {
			break
		}
	}

	return true, nil
}

// ChunkMap chunks the query, and collects what the callback returns for
// every row.
//
// Go has no default arguments, so the count is always given -- pass
// DefaultChunkSize for the conventional default.
func ChunkMap[T, R any](ctx context.Context, g auth.Grant, query Chunkable[T], count int, callback func(T) R) ([]R, error) {
	var out []R
	_, err := Chunk(ctx, g, query, count, func(items []T, _ int) bool {
		for _, item := range items {
			out = append(out, callback(item))
		}
		return true
	})
	return out, err
}

// DefaultChunkSize is the conventional chunk size for ChunkMap, Each, Lazy
// and LazyByID.
const DefaultChunkSize = 1000

// Each chunks the query, and hands the callback one row at a time with its
// index.
//
// The index is the position within the chunk, resetting to zero on every
// page. EachByID numbers across chunks instead, which is a deliberate
// difference between the two, kept.
func Each[T any](ctx context.Context, g auth.Grant, query Chunkable[T], count int, callback func(value T, key int) bool) (bool, error) {
	return Chunk(ctx, g, query, count, func(results []T, _ int) bool {
		for key, value := range results {
			if !callback(value, key) {
				return false
			}
		}
		return true
	})
}

// ChunkByID chunks the query by comparing ids rather than by offset.
//
// It is the one to reach for when the rows being walked are also being written
// to. An offset-based chunk over a table that is having rows inserted into it
// skips rows, because page two of a list that grew by one row starts where page
// one already ended.
//
// column empty takes the query's own key; alias empty takes column.
func ChunkByID[T any](ctx context.Context, g auth.Grant, query KeyChunkable[T], count int, callback func(results []T, page int) bool, column, alias string) (bool, error) {
	return OrderedChunkByID(ctx, g, query, count, callback, column, alias, false)
}

// ChunkByIDDesc is ChunkByID in descending order.
func ChunkByIDDesc[T any](ctx context.Context, g auth.Grant, query KeyChunkable[T], count int, callback func(results []T, page int) bool, column, alias string) (bool, error) {
	return OrderedChunkByID(ctx, g, query, count, callback, column, alias, true)
}

// OrderedChunkByID is the body ChunkByID and ChunkByIDDesc share.
//
// It returns an error when the aliased column is not in the result. It is
// worth the words it costs: a chunk whose id column was not selected loops
// on page one forever otherwise.
func OrderedChunkByID[T any](ctx context.Context, g auth.Grant, query KeyChunkable[T], count int, callback func(results []T, page int) bool, column, alias string, descending bool) (bool, error) {
	if column == "" {
		column = query.DefaultKeyName()
	}
	if alias == "" {
		alias = column
	}

	var lastID any
	skip := 0
	if o := query.GetOffset(); o != nil {
		skip = *o
	}
	remaining := query.GetLimit()

	page := 1
	for {
		clone := query.Clone()

		if skip != 0 && page > 1 {
			clone.Offset(0)
		}

		limit := count
		if remaining != nil {
			limit = min(count, *remaining)
		}
		if limit == 0 {
			break
		}

		if descending {
			clone.ForPageBeforeID(limit, lastID, column)
		} else {
			clone.ForPageAfterID(limit, lastID, column)
		}

		results, err := clone.Get(ctx, g)
		if err != nil {
			return false, err
		}

		countResults := len(results)
		if countResults == 0 {
			break
		}

		if remaining != nil {
			left := max(*remaining-countResults, 0)
			remaining = &left
		}

		if !callback(results, page) {
			return false, nil
		}

		value, ok := clone.ValueOf(results[countResults-1], alias)
		if !ok || value == nil {
			return false, fmt.Errorf("The chunkById operation was aborted because the [%s] column is not present in the query result.", alias)
		}
		lastID = value

		page++

		if countResults != count {
			break
		}
	}

	return true, nil
}

// EachByID chunks the query by id, and hands the callback one row at a time.
//
// The index runs across chunks -- ((page - 1) * count) + key -- which is why
// it differs from Each.
func EachByID[T any](ctx context.Context, g auth.Grant, query KeyChunkable[T], callback func(value T, key int) bool, count int, column, alias string) (bool, error) {
	return ChunkByID(ctx, g, query, count, func(results []T, page int) bool {
		for key, value := range results {
			if !callback(value, (page-1)*count+key) {
				return false
			}
		}
		return true
	}, column, alias)
}

// Lazy returns an iterator over the whole result set, fetched a chunk at a
// time.
//
// Go 1.23 has range-over-func, so this returns an iter.Seq2-shaped function:
// the loop reads
//
//	for row, err := range concerns.Lazy(ctx, g, q, 1000) {
//
// and an error ends the iteration after being yielded once, so a caller that
// forgets to check it still stops rather than looping on a broken query.
func Lazy[T any](ctx context.Context, g auth.Grant, query Chunkable[T], chunkSize int) func(yield func(T, error) bool) {
	return func(yield func(T, error) bool) {
		var zero T

		if chunkSize < 1 {
			yield(zero, errors.New("The chunk size should be at least 1"))
			return
		}
		if err := query.EnforceOrderBy(); err != nil {
			yield(zero, err)
			return
		}

		page := 1
		for {
			query.ForPage(page, chunkSize)
			page++

			results, err := query.Get(ctx, g)
			if err != nil {
				yield(zero, err)
				return
			}

			for _, result := range results {
				if !yield(result, nil) {
					return
				}
			}

			if len(results) < chunkSize {
				return
			}
		}
	}
}

// LazyByID is Lazy, chunked by comparing ids rather than by offset.
func LazyByID[T any](ctx context.Context, g auth.Grant, query KeyChunkable[T], chunkSize int, column, alias string) func(yield func(T, error) bool) {
	return orderedLazyByID(ctx, g, query, chunkSize, column, alias, false)
}

// LazyByIDDesc is LazyByID in descending order.
func LazyByIDDesc[T any](ctx context.Context, g auth.Grant, query KeyChunkable[T], chunkSize int, column, alias string) func(yield func(T, error) bool) {
	return orderedLazyByID(ctx, g, query, chunkSize, column, alias, true)
}

// orderedLazyByID is the body LazyByID and LazyByIDDesc share.
func orderedLazyByID[T any](ctx context.Context, g auth.Grant, query KeyChunkable[T], chunkSize int, column, alias string, descending bool) func(yield func(T, error) bool) {
	return func(yield func(T, error) bool) {
		var zero T

		if chunkSize < 1 {
			yield(zero, errors.New("The chunk size should be at least 1"))
			return
		}
		if column == "" {
			column = query.DefaultKeyName()
		}
		if alias == "" {
			alias = column
		}

		var lastID any
		for {
			clone := query.Clone()

			if descending {
				clone.ForPageBeforeID(chunkSize, lastID, column)
			} else {
				clone.ForPageAfterID(chunkSize, lastID, column)
			}

			results, err := clone.Get(ctx, g)
			if err != nil {
				yield(zero, err)
				return
			}

			for _, result := range results {
				if !yield(result, nil) {
					return
				}
			}

			if len(results) < chunkSize {
				return
			}

			value, ok := clone.ValueOf(results[len(results)-1], alias)
			if !ok || value == nil {
				yield(zero, fmt.Errorf("The lazyById operation was aborted because the [%s] column is not present in the query result.", alias))
				return
			}
			lastID = value
		}
	}
}

// First limits the query to one row and returns it.
//
// It returns (zero, false) for an empty result, because a zero-valued struct
// is indistinguishable from a row of zeroes and a caller that cannot tell
// them apart writes a bug that only shows up on real data.
func First[T any](ctx context.Context, g auth.Grant, query Chunkable[T]) (T, bool, error) {
	var zero T

	query.Limit(1)

	results, err := query.Get(ctx, g)
	if err != nil {
		return zero, false, err
	}
	if len(results) == 0 {
		return zero, false, nil
	}
	return results[0], true, nil
}

// FirstOrFail is First, failing instead of returning false when no row
// matched.
//
// message empty takes a default sentence. The error wraps ErrRecordNotFound,
// so errors.Is keeps working when a caller adds context to it and the
// exception classifier still reports 404.
func FirstOrFail[T any](ctx context.Context, g auth.Grant, query Chunkable[T], message string) (T, error) {
	var zero T

	result, found, err := First(ctx, g, query)
	if err != nil {
		return zero, err
	}
	if found {
		return result, nil
	}
	if message == "" {
		return zero, ErrRecordNotFound
	}
	return zero, fmt.Errorf("%s: %w", message, ErrRecordNotFound)
}

// Sole returns the one row that matched, and an error when there was none or
// more than one.
//
// It fetches two rows, not one, which is the whole point: a query for "the
// invoice for this reference" has to be able to say that two of them exist.
func Sole[T any](ctx context.Context, g auth.Grant, query Chunkable[T]) (T, error) {
	var zero T

	query.Limit(2)

	results, err := query.Get(ctx, g)
	if err != nil {
		return zero, err
	}
	switch {
	case len(results) == 0:
		return zero, ErrRecordsNotFound
	case len(results) > 1:
		return zero, NewMultipleRecordsFoundError(len(results))
	}
	return results[0], nil
}

// Tap hands the query to callback, and gives the query back.
func Tap[Q any](query Q, callback func(Q)) Q {
	callback(query)
	return query
}

// Pipe hands the query to callback, and gives back what callback returned.
//
// There is no fallback to the original query when callback returns a zero
// value: a function returning R returns an R, and Go has no null to fall
// back from. It is not missed -- Tap already covers wanting the query back.
func Pipe[Q, R any](query Q, callback func(Q) R) R {
	return callback(query)
}
