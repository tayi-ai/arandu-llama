package concerns

import (
	"context"
	"sort"

	"github.com/arandu-io/hesape/database/query"
)

// The four functions below run a statement built with the base query builder.
//
// They exist because the pivot table is reached with the base query builder
// rather than a typed one, and the base builder compiles SQL without
// executing it.
//
// They are unexported and they stay that way. They reach the connection through
// GetConnection, which is the one declared way past the builder's own terminals
// -- the field behind it is not exported, so this is the only door and it is a
// door with a name a lint can be written against.
//
// What they must not become is a second way to run a statement for anyone else.
// The rows they run are already tenant-scoped, by ScopeTenantQuery at
// NewPivotQuery, which is why they do not go through the builder's terminals:
// those would scope again.

// insert runs the insert the builder compiled.
//
// The values are key-sorted before they are flattened, and that is not tidiness:
// the grammar writes the column list from the same records, and bindings
// flattened in a different order than the columns were written is every value
// landing in the wrong column -- a corrupt row rather than an error.
func insert(ctx context.Context, q *query.Builder, values []map[string]any) error {
	if len(values) == 0 {
		return nil
	}

	q.ApplyBeforeQueryCallbacks()
	if err := q.Err(); err != nil {
		return err
	}

	bindings := make([]any, 0, len(values)*len(values[0]))
	for _, record := range values {
		for _, column := range sortedKeys(record) {
			if value := record[column]; !query.IsExpression(value) {
				bindings = append(bindings, value)
			}
		}
	}

	_, err := q.GetConnection().Insert(ctx, q.Grammar.CompileInsert(q, values), bindings)
	return err
}

// update runs the update the builder compiled.
func update(ctx context.Context, q *query.Builder, values map[string]any) (int64, error) {
	q.ApplyBeforeQueryCallbacks()
	if err := q.Err(); err != nil {
		return 0, err
	}

	sql := q.Grammar.CompileUpdate(q, values)
	return q.GetConnection().Update(ctx, sql, q.Grammar.PrepareBindingsForUpdate(q.GetRawBindings(), values))
}

// deleteFrom runs the delete the builder compiled.
//
// It is not called delete: that is the builtin removing a key from a map, and
// shadowing it inside this package would be a trap for the next reader.
func deleteFrom(ctx context.Context, q *query.Builder) (int64, error) {
	q.ApplyBeforeQueryCallbacks()
	if err := q.Err(); err != nil {
		return 0, err
	}

	sql := q.Grammar.CompileDelete(q)
	return q.GetConnection().Delete(ctx, sql, q.Grammar.PrepareBindingsForDelete(q.GetRawBindings()))
}

// selectRows runs the select the builder compiled: the rows, unhydrated.
func selectRows(ctx context.Context, q *query.Builder) ([]query.Record, error) {
	sql := q.ToSQL()
	if err := q.Err(); err != nil {
		return nil, err
	}
	rows, err := q.GetConnection().Select(ctx, sql, q.GetBindings(), false)
	if err != nil {
		return nil, err
	}
	if q.Processor != nil {
		rows = q.Processor.ProcessSelect(q, rows)
	}
	return rows, nil
}

func sortedKeys(record map[string]any) []string {
	keys := make([]string, 0, len(record))
	for key := range record {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}
