package model

import (
	"context"
	"fmt"

	"github.com/arandu-io/hesape/auth"
)

// SoftDeletingScopeName is the identifier the SoftDeletingScope is registered
// under.
//
// It is a constant rather than a computed name, so that WithTrashed can name
// the same scope the model registered without holding a reference to it.
const SoftDeletingScopeName = "SoftDeletingScope"

// SoftDeletingScope filters the deleted rows out of every query, and replaces
// the delete with an update.
//
// The model turns it on by setting SoftDeletes, and NewQuery registers it.
//
// The deleted_at field on the entity has to be able to hold a null: a
// *time.Time, or another type that writes NULL when it is empty. A plain
// time.Time has no null, so restoring would write the zero date and the row
// would read as deleted at the year one.
type SoftDeletingScope[T any] struct{}

// Apply adds the not-deleted filter for model to builder's query.
func (s *SoftDeletingScope[T]) Apply(builder *Builder[T], model *Model[T]) {
	builder.query.WhereNull(model.GetQualifiedDeletedAtColumn())
}

// Extend registers the delete override that turns a hard delete into a
// timestamp update.
//
// WithTrashed, WithoutTrashed, OnlyTrashed and Restore already exist as
// methods on Builder, so what Extend does is the part that is not a name: it
// points the builder's delete at an update.
func (s *SoftDeletingScope[T]) Extend(builder *Builder[T]) {
	builder.OnDelete(func(ctx context.Context, b *Builder[T], g auth.Grant) (int64, error) {
		column := s.deletedAtColumn(b)
		return b.Update(ctx, g, map[string]any{column: b.GetModel().FreshTimestamp()})
	})
}

// deletedAtColumn returns the deleted_at column name to filter or set:
// qualified when the query joins, bare otherwise, because a bare name is
// ambiguous across a join and a qualified one is refused on the left of a SET
// by some engines.
func (s *SoftDeletingScope[T]) deletedAtColumn(b *Builder[T]) string {
	if len(b.query.Joins) > 0 {
		return b.GetModel().GetQualifiedDeletedAtColumn()
	}
	return b.GetModel().GetDeletedAtColumn()
}

// GetDeletedAtColumn returns the name of the column that marks a row
// deleted, defaulting to "deleted_at".
func (m *Model[T]) GetDeletedAtColumn() string {
	if m.DeletedAtColumn == "" {
		return "deleted_at"
	}
	return m.DeletedAtColumn
}

// GetQualifiedDeletedAtColumn returns the deleted_at column name qualified
// with the model's table.
func (m *Model[T]) GetQualifiedDeletedAtColumn() string {
	return m.QualifyColumn(m.GetDeletedAtColumn())
}

// Trashed reports whether the model has been soft deleted.
func (m *Model[T]) Trashed() bool {
	value := m.GetAttribute(m.GetDeletedAtColumn())
	return value != nil && !isZero(value)
}

// IsForceDeleting reports whether the current delete bypasses soft deletes.
func (m *Model[T]) IsForceDeleting() bool { return m.forceDeleting }

// performDeleteOnModel deletes the model's row, or marks it deleted: a model
// that soft deletes and is not force deleting marks the row instead of
// removing it.
func (m *Model[T]) performDeleteOnModel(ctx context.Context, g auth.Grant) error {
	if m.SoftDeletes && !m.forceDeleting {
		return m.runSoftDelete(ctx, g)
	}

	q := m.NewModelQuery()
	m.setKeysForSaveQuery(q)
	if _, err := q.ForceDelete(ctx, g); err != nil {
		return err
	}
	m.Exists = false
	return nil
}

// runSoftDelete sets the deleted_at column to the current time instead of
// removing the row, and fires the Trashed event.
func (m *Model[T]) runSoftDelete(ctx context.Context, g auth.Grant) error {
	now := m.FreshTimestamp()
	column := m.GetDeletedAtColumn()

	columns := map[string]any{column: now}
	if err := m.SetAttribute(column, now); err != nil {
		return err
	}

	if m.UsesTimestamps() && m.GetUpdatedAtColumn() != "" {
		columns[m.GetUpdatedAtColumn()] = now
		if err := m.SetAttribute(m.GetUpdatedAtColumn(), now); err != nil {
			return err
		}
	}

	q := m.NewModelQuery()
	m.setKeysForSaveQuery(q)
	if _, err := q.Update(ctx, g, columns); err != nil {
		return err
	}

	m.SyncOriginalAttributes(sortedKeys(columns)...)
	return m.fireModelEvent(Trashed)
}

// ForceDelete removes the row even if the model soft deletes. On a model
// that does not soft delete it is a plain delete.
func (m *Model[T]) ForceDelete(ctx context.Context, g auth.Grant) (bool, error) {
	if !m.SoftDeletes {
		return m.Delete(ctx, g)
	}
	if err := m.fireModelEvent(ForceDeleting); err != nil {
		return false, err
	}

	m.forceDeleting = true
	deleted, err := m.Delete(ctx, g)
	m.forceDeleting = false
	if err != nil {
		return false, err
	}
	if deleted {
		if err := m.fireModelEvent(ForceDeleted); err != nil {
			return false, err
		}
	}
	return deleted, nil
}

// ForceDeleteQuietly removes the row even if the model soft deletes, without
// firing model events.
func (m *Model[T]) ForceDeleteQuietly(ctx context.Context, g auth.Grant) (deleted bool, err error) {
	return deleted, m.WithoutEvents(func() error {
		deleted, err = m.ForceDelete(ctx, g)
		return err
	})
}

// ForceDestroy loads the models with the given keys, including trashed ones,
// and removes each row even if the model soft deletes. It returns the number
// removed.
func (m *Model[T]) ForceDestroy(ctx context.Context, g auth.Grant, ids ...any) (int, error) {
	if len(ids) == 0 {
		return 0, nil
	}
	found, err := m.NewQuery().WithTrashed().WhereKey(ids).get(ctx, g)
	if err != nil {
		return 0, err
	}
	count := 0
	for _, model := range found {
		deleted, err := model.ForceDelete(ctx, g)
		if err != nil {
			return count, err
		}
		if deleted {
			count++
		}
	}
	return count, nil
}

// Restore clears the deleted_at column and saves the model: the row comes
// back.
func (m *Model[T]) Restore(ctx context.Context, g auth.Grant) (bool, error) {
	// A value the framework did not build has no connection to write through,
	// and no back pointer to the entity it is inside. It is the literal case,
	// and it says so rather than reporting a write that never happened -- see
	// ErrUnwired.
	if err := m.wired(); err != nil {
		return false, err
	}

	if !m.SoftDeletes {
		return false, fmt.Errorf("model: %s does not soft delete, so there is nothing to restore", m.GetTable())
	}
	if err := m.fireModelEvent(Restoring); err != nil {
		return false, err
	}
	if err := m.SetAttribute(m.GetDeletedAtColumn(), nil); err != nil {
		return false, err
	}

	m.Exists = true
	restored, err := m.Save(ctx, g)
	if err != nil {
		return false, err
	}
	if err := m.fireModelEvent(Restored); err != nil {
		return false, err
	}
	return restored, nil
}

// RestoreQuietly restores the model without firing model events.
func (m *Model[T]) RestoreQuietly(ctx context.Context, g auth.Grant) (restored bool, err error) {
	return restored, m.WithoutEvents(func() error {
		restored, err = m.Restore(ctx, g)
		return err
	})
}

// WithTrashed includes the soft-deleted rows in the query.
//
// On a model that does not soft delete it is an error rather than a query
// that quietly means something else.
func (b *Builder[T]) WithTrashed(withTrashed ...bool) *Builder[T] {
	if len(withTrashed) > 0 && !withTrashed[0] {
		return b.WithoutTrashed()
	}
	if err := b.requireSoftDeletes("withTrashed"); err != nil {
		return b.fail(err)
	}
	return b.WithoutGlobalScope(SoftDeletingScopeName)
}

// WithoutTrashed removes the global soft-delete scope and re-adds an
// explicit not-deleted filter, so trashed rows stay excluded even after the
// scope is gone.
func (b *Builder[T]) WithoutTrashed() *Builder[T] {
	if err := b.requireSoftDeletes("withoutTrashed"); err != nil {
		return b.fail(err)
	}
	b.WithoutGlobalScope(SoftDeletingScopeName)
	b.query.WhereNull(b.model.GetQualifiedDeletedAtColumn())
	return b
}

// OnlyTrashed restricts the query to soft-deleted rows only.
func (b *Builder[T]) OnlyTrashed() *Builder[T] {
	if err := b.requireSoftDeletes("onlyTrashed"); err != nil {
		return b.fail(err)
	}
	b.WithoutGlobalScope(SoftDeletingScopeName)
	b.query.WhereNotNull(b.model.GetQualifiedDeletedAtColumn())
	return b
}

// Restore un-deletes every row the query matches, in one statement.
func (b *Builder[T]) Restore(ctx context.Context, g auth.Grant) (int64, error) {
	if err := b.requireSoftDeletes("restore"); err != nil {
		return 0, err
	}
	return b.WithTrashed().Update(ctx, g, map[string]any{b.model.GetDeletedAtColumn(): nil})
}

// RestoreOrCreate finds the first trashed-or-not row matching attributes and
// restores it, or creates one from attributes and values if none matches.
func (b *Builder[T]) RestoreOrCreate(ctx context.Context, g auth.Grant, attributes, values map[string]any) (*T, error) {
	if err := b.requireSoftDeletes("restoreOrCreate"); err != nil {
		return nil, err
	}
	model, err := b.WithTrashed().firstOrCreate(ctx, g, attributes, values)
	if err != nil {
		return nil, err
	}
	if _, err := model.Restore(ctx, g); err != nil {
		return nil, err
	}
	return b.result(model), nil
}

// CreateOrRestore finds the first row matching attributes, including
// trashed, and restores it if trashed; otherwise it creates one from
// attributes and values.
func (b *Builder[T]) CreateOrRestore(ctx context.Context, g auth.Grant, attributes, values map[string]any) (*T, error) {
	if err := b.requireSoftDeletes("createOrRestore"); err != nil {
		return nil, err
	}
	model, err := b.WithTrashed().createOrFirst(ctx, g, attributes, values)
	if err != nil {
		return nil, err
	}
	if _, err := model.Restore(ctx, g); err != nil {
		return nil, err
	}
	return b.result(model), nil
}

func (b *Builder[T]) requireSoftDeletes(method string) error {
	if b.model.SoftDeletes {
		return nil
	}
	return fmt.Errorf("model: %s on %s, which does not soft delete: set SoftDeletes on the model, or the filter means nothing", method, b.model.GetTable())
}
