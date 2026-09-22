package model

import (
	"context"

	"github.com/arandu-io/hesape/auth"
)

// The reads and writes taken straight off a model -- Find, Create and their
// neighbours -- forwarded to a fresh builder.
//
// Every one of them starts from NewQuery, so the global scopes are on, which is
// what makes Find skip a soft deleted row.

// Find calls Find on a fresh query for the model.
func (m *Model[T]) Find(ctx context.Context, g auth.Grant, id any, columns ...any) (*T, error) {
	return m.NewQuery().Find(ctx, g, id, columns...)
}

// FindMany calls FindMany on a fresh query for the model.
func (m *Model[T]) FindMany(ctx context.Context, g auth.Grant, ids []any, columns ...any) (Collection[T], error) {
	return m.NewQuery().FindMany(ctx, g, ids, columns...)
}

// FindOrFail calls FindOrFail on a fresh query for the model.
func (m *Model[T]) FindOrFail(ctx context.Context, g auth.Grant, id any, columns ...any) (*T, error) {
	return m.NewQuery().FindOrFail(ctx, g, id, columns...)
}

// FindOrNew calls FindOrNew on a fresh query for the model.
func (m *Model[T]) FindOrNew(ctx context.Context, g auth.Grant, id any, columns ...any) (*T, error) {
	return m.NewQuery().FindOrNew(ctx, g, id, columns...)
}

// First calls First on a fresh query for the model.
func (m *Model[T]) First(ctx context.Context, g auth.Grant, columns ...any) (*T, error) {
	return m.NewQuery().First(ctx, g, columns...)
}

// FirstOrNew calls FirstOrNew on a fresh query for the model.
func (m *Model[T]) FirstOrNew(ctx context.Context, g auth.Grant, attributes, values map[string]any) (*T, error) {
	return m.NewQuery().FirstOrNew(ctx, g, attributes, values)
}

// FirstOrCreate calls FirstOrCreate on a fresh query for the model.
func (m *Model[T]) FirstOrCreate(ctx context.Context, g auth.Grant, attributes, values map[string]any) (*T, error) {
	return m.NewQuery().FirstOrCreate(ctx, g, attributes, values)
}

// UpdateOrCreate calls UpdateOrCreate on a fresh query for the model.
func (m *Model[T]) UpdateOrCreate(ctx context.Context, g auth.Grant, attributes, values map[string]any) (*T, error) {
	return m.NewQuery().UpdateOrCreate(ctx, g, attributes, values)
}

// Create calls Create on a fresh query for the model.
func (m *Model[T]) Create(ctx context.Context, g auth.Grant, attributes map[string]any) (*T, error) {
	return m.NewQuery().Create(ctx, g, attributes)
}

// ForceCreate calls ForceCreate on a fresh query for the model.
func (m *Model[T]) ForceCreate(ctx context.Context, g auth.Grant, attributes map[string]any) (*T, error) {
	return m.NewQuery().ForceCreate(ctx, g, attributes)
}

// With calls With on a fresh query for the model.
func (m *Model[T]) With(relations ...string) *Builder[T] {
	return m.NewQuery().With(relations...)
}

// Where calls Where on a fresh query for the model.
func (m *Model[T]) Where(column any, args ...any) *Builder[T] {
	return m.NewQuery().Where(column, args...)
}

// WhereKey calls WhereKey on a fresh query for the model.
func (m *Model[T]) WhereKey(id any) *Builder[T] { return m.NewQuery().WhereKey(id) }

// WithTrashed calls WithTrashed on a fresh query for the model.
func (m *Model[T]) WithTrashed() *Builder[T] { return m.NewQuery().WithTrashed() }

// OnlyTrashed calls OnlyTrashed on a fresh query for the model.
func (m *Model[T]) OnlyTrashed() *Builder[T] { return m.NewQuery().OnlyTrashed() }
