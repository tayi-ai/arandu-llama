package model

import (
	"context"
	"iter"
	"time"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/database/model/relations/concerns"
	"github.com/arandu-io/hesape/database/query"
	"github.com/arandu-io/hesape/pagination"
)

// The seam between the typed model and the relations that read it.
//
// # Why an adapter and not one type
//
// The relations tree declares the narrow contract it consumes -- concerns.Model
// and concerns.Builder -- and *Model[T] cannot satisfy it directly, for two
// reasons that are both about Go rather than about design.
//
// Fourteen of the twenty-seven methods on concerns.Builder are chainables that
// return Builder. Builder[T].Where returns *Builder[T]. Go has no covariant
// return, so satisfying the interface with the typed builder would mean renaming
// fourteen methods of the public fluent API.
//
// And the tree cannot be made generic to meet the model instead. Model[T] holds
// RelationResolvers as one map, and one map holds a has-many to Post beside a
// belongs-to to Team; one map value type means one non-generic Relation. MorphTo
// is heterogeneous by construction -- its dictionary is keyed by a morph alias
// resolved at run time -- and there is no type parameter to write there at all.
//
// So the erasure is confined to this file. Two unexported types, reached by one
// method each, and one function back. Nothing outside the relation boundary
// sees an untyped model.
//
// # Why it is safe
//
// The relations compare models by key and table, never by pointer identity, and
// key their dictionaries by string. Every mutation they perform -- SetRelation,
// SetAttribute, UnsetAttribute -- goes through this adapter's pointer to the
// real *Model[T], so what a relation writes is on the model the caller holds.

// modelRef is *Model[T] seen through the interface a relation asks for.
type modelRef[T any] struct {
	m *Model[T]

	// err holds what a method with nowhere to report it could not say.
	//
	// concerns.Model.Fill returns nothing and Model[T].Fill returns an error,
	// because a value can fail to fit a field. Dropping it would be the worst of
	// the three options; holding it and answering it from the next method that
	// can is the shape Builder[T].err already uses.
	err error
}

// builderRef is *Builder[T] seen the same way.
type builderRef[T any] struct{ b *Builder[T] }

var (
	_ concerns.Model   = (*modelRef[struct{}])(nil)
	_ concerns.Builder = (*builderRef[struct{}])(nil)
)

// Ref returns m as the model a relation takes.
//
// The value is cached on the model, so two calls answer the same one. Nothing
// keys a map by a model today, and a ref that was a new value every call would
// be a trap waiting for the first thing that did.
func (m *Model[T]) Ref() concerns.Model {
	if m.ref == nil {
		m.ref = &modelRef[T]{m: m}
	}
	return m.ref
}

// Ref returns b as the builder a relation takes.
func (b *Builder[T]) Ref() concerns.Builder { return &builderRef[T]{b: b} }

// Unref is the way back: the typed model behind a ref, and whether the ref was
// over this entity at all.
//
// A ref over another entity answers false rather than panicking, because the
// question "is this relation's model a Post" is one a caller is entitled to ask
// and get no for.
func Unref[T any](m concerns.Model) (*Model[T], bool) {
	ref, ok := m.(*modelRef[T])
	if !ok {
		return nil, false
	}
	return ref.m, true
}

// -- modelRef ---------------------------------------------------------------

func (r *modelRef[T]) GetTable() string                    { return r.m.GetTable() }
func (r *modelRef[T]) QualifyColumn(column string) string  { return r.m.QualifyColumn(column) }
func (r *modelRef[T]) GetKeyName() string                  { return r.m.GetKeyName() }
func (r *modelRef[T]) GetKeyType() string                  { return r.m.GetKeyType() }
func (r *modelRef[T]) GetKey() any                         { return r.m.GetKey() }
func (r *modelRef[T]) GetForeignKey() string               { return r.m.GetForeignKey() }
func (r *modelRef[T]) GetMorphClass() string               { return r.m.GetMorphClass() }
func (r *modelRef[T]) GetAttribute(key string) any         { return r.m.GetAttribute(key) }
func (r *modelRef[T]) GetAttributes() map[string]any       { return r.m.GetAttributes() }
func (r *modelRef[T]) RelationLoaded(relation string) bool { return r.m.RelationLoaded(relation) }
func (r *modelRef[T]) GetCreatedAtColumn() string          { return r.m.GetCreatedAtColumn() }
func (r *modelRef[T]) GetUpdatedAtColumn() string          { return r.m.GetUpdatedAtColumn() }
func (r *modelRef[T]) UsesTimestamps() bool                { return r.m.UsesTimestamps() }
func (r *modelRef[T]) UnsetAttribute(key string)           { r.m.UnsetAttribute(key) }
func (r *modelRef[T]) IsRelation(key string) bool          { return r.m.IsRelation(key) }
func (r *modelRef[T]) Touches(relation string) bool        { return r.m.Touches(relation) }

func (r *modelRef[T]) FreshTimestamp() time.Time { return r.m.FreshTimestamp() }

// Exists and WasRecentlyCreated are fields on the model and methods here, and
// that is the collision the adapter exists to absorb: a Go type cannot have
// both under one name, and this is a different type.
func (r *modelRef[T]) Exists() bool             { return r.m.Exists }
func (r *modelRef[T]) WasRecentlyCreated() bool { return r.m.WasRecentlyCreated }

func (r *modelRef[T]) GetRelation(relation string) (any, bool) { return r.m.GetRelation(relation) }
func (r *modelRef[T]) SetRelation(relation string, value any)  { r.m.SetRelation(relation, value) }
func (r *modelRef[T]) UnsetRelation(relation string)           { r.m.UnsetRelation(relation) }
func (r *modelRef[T]) SetTable(table string)                   { r.m.SetTable(table) }

func (r *modelRef[T]) Fill(attributes map[string]any)      { r.hold(r.m.Fill(attributes)) }
func (r *modelRef[T]) ForceFill(attributes map[string]any) { r.hold(r.m.ForceFill(attributes)) }

func (r *modelRef[T]) SetAttribute(key string, value any) {
	r.hold(r.m.SetAttribute(key, value))
}

func (r *modelRef[T]) SetRawAttributes(attributes map[string]any, sync bool) {
	r.hold(r.m.SetRawAttributes(attributes, sync))
}

func (r *modelRef[T]) WithoutEvents(callback func() error) error {
	return r.m.WithoutEvents(callback)
}

// NewInstance answers a fresh model of the same kind, as a ref.
//
// The error the typed constructor reports is held rather than dropped, and the
// next method that can report one does.
func (r *modelRef[T]) NewInstance(attributes map[string]any) concerns.Model {
	instance, err := r.m.NewInstance(attributes, false)
	if err != nil {
		failed := &modelRef[T]{m: r.m}
		failed.hold(err)
		return failed
	}
	return instance.Ref()
}

func (r *modelRef[T]) NewQuery() concerns.Builder { return r.m.NewQuery().Ref() }

func (r *modelRef[T]) Save(ctx context.Context, g auth.Grant) error {
	if err := r.taken(); err != nil {
		return err
	}
	_, err := r.m.Save(ctx, g)
	return err
}

// Delete answers rows affected where the typed one answers whether anything
// went.
func (r *modelRef[T]) Delete(ctx context.Context, g auth.Grant) (int64, error) {
	if err := r.taken(); err != nil {
		return 0, err
	}
	deleted, err := r.m.Delete(ctx, g)
	if err != nil || !deleted {
		return 0, err
	}
	return 1, nil
}

func (r *modelRef[T]) Touch(ctx context.Context, g auth.Grant) error {
	if err := r.taken(); err != nil {
		return err
	}
	return r.m.Touch(ctx, g)
}

// hold keeps the first error a method with no return could not report.
func (r *modelRef[T]) hold(err error) {
	if err != nil && r.err == nil {
		r.err = err
	}
}

// taken answers the held error once and forgets it, so that a model which
// recovered is not refused forever.
func (r *modelRef[T]) taken() error {
	err := r.err
	r.err = nil
	return err
}

// -- builderRef -------------------------------------------------------------

func (r *builderRef[T]) GetModel() concerns.Model { return r.b.GetModel().Ref() }
func (r *builderRef[T]) GetQuery() *query.Builder { return r.b.GetQuery() }

// ScopesOwnTableByTenant answers for every terminal on the typed builder at
// once, because they share the one door: prepare puts the model's tenant column
// on the statement, and nothing runs without going through it.
//
// The column it uses is the model's own, which is the reason this is answered
// here rather than left to the relation. A model that declares a different
// column, or declares none because its table is shared, is filtered on what it
// declared -- where a second filter added from outside would only know the
// default name, and would name a column a shared table does not have.
func (r *builderRef[T]) ScopesOwnTableByTenant() bool { return true }

// The chainables. Each returns this ref rather than a new one: the typed
// builder mutates and returns itself, and a ref that allocated per call would
// make a chain of ten allocate ten.
func (r *builderRef[T]) Select(columns ...any) concerns.Builder {
	r.b.Select(columns...)
	return r
}

func (r *builderRef[T]) AddSelect(columns ...any) concerns.Builder {
	r.b.AddSelect(columns...)
	return r
}

func (r *builderRef[T]) Where(column any, args ...any) concerns.Builder {
	r.b.Where(column, args...)
	return r
}

func (r *builderRef[T]) WhereIn(column any, values []any) concerns.Builder {
	r.b.WhereIn(column, values)
	return r
}

func (r *builderRef[T]) WhereNotNull(columns ...any) concerns.Builder {
	r.b.WhereNotNull(columns...)
	return r
}

func (r *builderRef[T]) WhereColumn(first any, args ...any) concerns.Builder {
	r.b.WhereColumn(first, args...)
	return r
}

func (r *builderRef[T]) WhereKey(ids ...any) concerns.Builder {
	if len(ids) == 1 {
		r.b.WhereKey(ids[0])
		return r
	}
	r.b.WhereKey(ids)
	return r
}

func (r *builderRef[T]) Join(table any, first any, args ...any) concerns.Builder {
	r.b.Join(table, first, args...)
	return r
}

func (r *builderRef[T]) GroupBy(groups ...any) concerns.Builder {
	r.b.GroupBy(groups...)
	return r
}

func (r *builderRef[T]) SelectRaw(expression string, bindings ...any) concerns.Builder {
	r.b.SelectRaw(expression, bindings...)
	return r
}

func (r *builderRef[T]) OrderBy(column any, direction ...string) concerns.Builder {
	r.b.OrderBy(column, direction...)
	return r
}

func (r *builderRef[T]) Limit(value int) concerns.Builder  { r.b.Limit(value); return r }
func (r *builderRef[T]) Offset(value int) concerns.Builder { r.b.Offset(value); return r }
func (r *builderRef[T]) Clone() concerns.Builder           { return r.b.Clone().Ref() }

// asRef is the conversion the reads hand to the paginators and to Get: one
// model, as the interface.
func asRef[T any](m *Model[T]) concerns.Model { return m.Ref() }

func (r *builderRef[T]) Get(ctx context.Context, g auth.Grant) ([]concerns.Model, error) {
	found, err := r.b.get(ctx, g)
	if err != nil {
		return nil, err
	}
	out := make([]concerns.Model, 0, len(found))
	for _, m := range found {
		out = append(out, m.Ref())
	}
	return out, nil
}

// First answers (nil, nil) for a miss, as the interface says: ErrModelNotFound
// belongs to FirstOrFail, which is a different question.
func (r *builderRef[T]) First(ctx context.Context, g auth.Grant) (concerns.Model, error) {
	found, err := r.b.first(ctx, g)
	if err != nil || found == nil {
		return nil, err
	}
	return found.Ref(), nil
}

func (r *builderRef[T]) Find(ctx context.Context, g auth.Grant, id any) (concerns.Model, error) {
	found, err := r.b.find(ctx, g, id)
	if err != nil || found == nil {
		return nil, err
	}
	return found.Ref(), nil
}

func (r *builderRef[T]) Cursor(ctx context.Context, g auth.Grant) iter.Seq2[concerns.Model, error] {
	return func(yield func(concerns.Model, error) bool) {
		// The typed cursor reports its failure through a pointer rather than
		// beside each value, so the error is read after the walk and yielded
		// then. A stream that fails halfway has already handed out rows, which
		// is why the interface carries the error beside the value at all.
		var failure error
		for m := range r.b.cursor(ctx, g, &failure) {
			if !yield(m.Ref(), nil) {
				return
			}
		}
		if failure != nil {
			yield(nil, failure)
		}
	}
}

func (r *builderRef[T]) Paginate(ctx context.Context, g auth.Grant, perPage, page int, opts pagination.Options, columns ...any) (*pagination.LengthAwarePaginator[concerns.Model], error) {
	return paginateAs(r.b, ctx, g, perPage, page, opts, asRef[T], columns...)
}

func (r *builderRef[T]) SimplePaginate(ctx context.Context, g auth.Grant, perPage, page int, opts pagination.Options, columns ...any) (*pagination.Paginator[concerns.Model], error) {
	return simplePaginateAs(r.b, ctx, g, perPage, page, opts, asRef[T], columns...)
}

func (r *builderRef[T]) CursorPaginate(ctx context.Context, g auth.Grant, perPage int, cursor *pagination.Cursor, opts pagination.Options, columns ...any) (*pagination.CursorPaginator[concerns.Model], error) {
	return cursorPaginateAs(r.b, ctx, g, perPage, cursor, opts, asRef[T], columns...)
}

func (r *builderRef[T]) Insert(ctx context.Context, g auth.Grant, values []map[string]any) error {
	_, err := r.b.Insert(ctx, g, values...)
	return err
}

func (r *builderRef[T]) Update(ctx context.Context, g auth.Grant, values map[string]any) (int64, error) {
	return r.b.Update(ctx, g, values)
}

func (r *builderRef[T]) Upsert(ctx context.Context, g auth.Grant, values []map[string]any, uniqueBy, update []string) (int64, error) {
	return r.b.Upsert(ctx, g, values, uniqueBy, update)
}

func (r *builderRef[T]) Delete(ctx context.Context, g auth.Grant) (int64, error) {
	return r.b.Delete(ctx, g)
}
