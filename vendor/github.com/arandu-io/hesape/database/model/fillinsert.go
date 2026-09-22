package model

import (
	"context"
	"github.com/arandu-io/hesape/auth"
)

// FillForInsert returns the rows, enriched with whatever the model would
// have put on them -- its defaults and its timestamps -- without making a
// model per row on the way to the database.
//
// It is the path a seeder and an importer take: one statement for a thousand
// rows, with the columns a save would have written.
func (b *Builder[T]) FillForInsert(values []map[string]any) ([]map[string]any, error) {
	if len(values) == 0 {
		return nil, nil
	}
	out := make([]map[string]any, 0, len(values))
	for _, row := range values {
		instance, err := b.NewModelInstance(row)
		if err != nil {
			return nil, err
		}
		if instance.UsesTimestamps() {
			instance.UpdateTimestamps()
		}
		out = append(out, instance.getAttributesForInsert())
	}
	return out, nil
}

// FillAndInsert runs FillForInsert and inserts the result.
func (b *Builder[T]) FillAndInsert(ctx context.Context, g auth.Grant, values []map[string]any) (bool, error) {
	rows, err := b.FillForInsert(values)
	if err != nil {
		return false, err
	}
	return b.Insert(ctx, g, rows...)
}

// FillAndInsertOrIgnore runs FillForInsert and inserts the result, dropping
// rows that violate a unique index.
func (b *Builder[T]) FillAndInsertOrIgnore(ctx context.Context, g auth.Grant, values []map[string]any) (bool, error) {
	rows, err := b.FillForInsert(values)
	if err != nil {
		return false, err
	}
	return b.InsertOrIgnore(ctx, g, rows...)
}

// FillAndInsertGetID runs FillForInsert for one row, inserts it, and
// returns the value generated for the primary key.
func (b *Builder[T]) FillAndInsertGetID(ctx context.Context, g auth.Grant, values map[string]any) (int64, error) {
	rows, err := b.FillForInsert([]map[string]any{values})
	if err != nil {
		return 0, err
	}
	return b.InsertGetID(ctx, g, rows[0], b.model.GetKeyName())
}

// InsertOrIgnore writes values as new rows, dropping the ones that violate a
// unique index rather than failing the statement.
func (b *Builder[T]) InsertOrIgnore(ctx context.Context, g auth.Grant, values ...map[string]any) (bool, error) {
	prepared, rows, err := b.prepareWrite(g, values)
	if err != nil {
		return false, err
	}
	if len(rows) == 0 {
		return true, nil
	}

	if err := prepared.validateWriteQuery(); err != nil {
		return false, err
	}
	sql := b.model.Grammar.CompileInsertOrIgnore(prepared.query, rows)

	bindings := make([]any, 0, len(rows)*len(rows[0]))
	for _, row := range rows {
		for _, column := range sortedKeys(row) {
			bindings = append(bindings, row[column])
		}
	}
	return b.model.connection.Insert(ctx, sql, cleanBindings(bindings))
}

// IncrementOrCreate returns the row matching attributes with column set to
// def, or increments column by step on the row that was already there.
func (b *Builder[T]) IncrementOrCreate(ctx context.Context, g auth.Grant, attributes map[string]any, column string, def, step any) (*T, error) {
	if column == "" {
		column = "count"
	}
	if def == nil {
		def = 1
	}
	if step == nil {
		step = 1
	}

	instance, err := b.firstOrCreate(ctx, g, attributes, map[string]any{column: def})
	if err != nil {
		return nil, err
	}
	if instance.WasRecentlyCreated {
		return b.result(instance), nil
	}

	q := instance.NewModelQuery()
	instance.setKeysForSaveQuery(q)
	if _, err := q.Increment(ctx, g, column, step, nil); err != nil {
		return nil, err
	}
	return b.result(instance), instance.Refresh(ctx, g)
}

// UseWritePDO points this builder's statement at the write connection, even
// though it reads.
//
// It is how a read that has to see what was just written avoids the
// replica lag that would otherwise make a fresh row look missing.
func (b *Builder[T]) UseWritePDO() *Builder[T] {
	b.query.UseWritePDO()
	return b
}

// OnWriteConnection returns a query pointed at the write connection.
func (m *Model[T]) OnWriteConnection() *Builder[T] { return m.NewQuery().UseWritePDO() }

// OnClone registers a callback that runs on every copy this builder makes.
func (b *Builder[T]) OnClone(callback func(*Builder[T])) *Builder[T] {
	b.onCloneCallbacks = append(b.onCloneCallbacks, callback)
	return b
}

// NewQueryForRestoration returns the query a route model binding uses to
// find a soft deleted row.
func (m *Model[T]) NewQueryForRestoration(ids ...any) *Builder[T] {
	q := m.NewQueryWithoutScopes()
	if len(ids) == 1 {
		return q.WhereKey(ids[0])
	}
	return q.WhereKey(ids)
}

// IsSoftDeletable reports whether the model soft deletes.
func (m *Model[T]) IsSoftDeletable() bool { return m.SoftDeletes }

// SoftDeleted registers a callback for the moment a row is marked deleted.
func (m *Model[T]) SoftDeleted(callback func(*Model[T]) error) *Model[T] {
	return m.RegisterModelEvent(Trashed, callback)
}

// Restoring registers a callback for the moment a row is about to be
// restored.
func (m *Model[T]) Restoring(callback func(*Model[T]) error) *Model[T] {
	return m.RegisterModelEvent(Restoring, callback)
}

// Restored registers a callback for the moment a row has been restored.
func (m *Model[T]) Restored(callback func(*Model[T]) error) *Model[T] {
	return m.RegisterModelEvent(Restored, callback)
}

// ForceDeleting registers a callback for the moment a row is about to be
// force deleted.
func (m *Model[T]) ForceDeleting(callback func(*Model[T]) error) *Model[T] {
	return m.RegisterModelEvent(ForceDeleting, callback)
}

// ForceDeleted registers a callback for the moment a row has been force
// deleted.
func (m *Model[T]) ForceDeleted(callback func(*Model[T]) error) *Model[T] {
	return m.RegisterModelEvent(ForceDeleted, callback)
}
