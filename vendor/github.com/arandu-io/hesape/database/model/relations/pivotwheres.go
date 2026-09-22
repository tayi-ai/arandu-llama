package relations

import (
	"context"
	"fmt"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/database/query"
)

// The or- and not- forms of the pivot constraints, and the reads that name a
// row rather than filter one.
//
// Each is the PHP's own one-line delegation. They are here rather than inline
// with the relation because they are the surface, not the mechanism: a reader
// looking for how a pivot where reaches newPivotQuery finds it in
// belongstomany.go, and a reader looking for the method they typed finds it
// here.

// OrWherePivot answers BelongsToMany::orWherePivot.
func (r *BelongsToMany) OrWherePivot(column string, args ...any) *BelongsToMany {
	qualified := r.QualifyPivotColumn(column)
	r.pivotWheres = append(r.pivotWheres, func(q *query.Builder) { q.OrWhere(qualified, args...) })
	r.Query.GetQuery().OrWhere(qualified, args...)
	return r
}

// OrWherePivotIn answers BelongsToMany::orWherePivotIn.
func (r *BelongsToMany) OrWherePivotIn(column string, values []any) *BelongsToMany {
	qualified := r.QualifyPivotColumn(column)
	r.pivotWhereIns = append(r.pivotWhereIns, func(q *query.Builder) { q.OrWhereIn(qualified, values) })
	r.Query.GetQuery().OrWhereIn(qualified, values)
	return r
}

// OrWherePivotNotIn answers BelongsToMany::orWherePivotNotIn.
func (r *BelongsToMany) OrWherePivotNotIn(column string, values []any) *BelongsToMany {
	qualified := r.QualifyPivotColumn(column)
	r.pivotWhereIns = append(r.pivotWhereIns, func(q *query.Builder) { q.OrWhereNotIn(qualified, values) })
	r.Query.GetQuery().OrWhereNotIn(qualified, values)
	return r
}

// OrWherePivotNull answers BelongsToMany::orWherePivotNull.
func (r *BelongsToMany) OrWherePivotNull(column string) *BelongsToMany {
	qualified := r.QualifyPivotColumn(column)
	r.pivotWhereNulls = append(r.pivotWhereNulls, func(q *query.Builder) { q.OrWhereNull(qualified) })
	r.Query.GetQuery().OrWhereNull(qualified)
	return r
}

// OrWherePivotNotNull answers BelongsToMany::orWherePivotNotNull.
func (r *BelongsToMany) OrWherePivotNotNull(column string) *BelongsToMany {
	qualified := r.QualifyPivotColumn(column)
	r.pivotWhereNulls = append(r.pivotWhereNulls, func(q *query.Builder) { q.OrWhereNotNull(qualified) })
	r.Query.GetQuery().OrWhereNotNull(qualified)
	return r
}

// WherePivotNotBetween answers BelongsToMany::wherePivotNotBetween.
func (r *BelongsToMany) WherePivotNotBetween(column string, from, to any) *BelongsToMany {
	qualified := r.QualifyPivotColumn(column)
	r.pivotWheres = append(r.pivotWheres, func(q *query.Builder) { q.WhereNotBetween(qualified, from, to) })
	r.Query.GetQuery().WhereNotBetween(qualified, from, to)
	return r
}

// OrWherePivotBetween answers BelongsToMany::orWherePivotBetween.
func (r *BelongsToMany) OrWherePivotBetween(column string, from, to any) *BelongsToMany {
	qualified := r.QualifyPivotColumn(column)
	r.pivotWheres = append(r.pivotWheres, func(q *query.Builder) { q.OrWhereBetween(qualified, from, to) })
	r.Query.GetQuery().OrWhereBetween(qualified, from, to)
	return r
}

// OrWherePivotNotBetween answers BelongsToMany::orWherePivotNotBetween.
func (r *BelongsToMany) OrWherePivotNotBetween(column string, from, to any) *BelongsToMany {
	qualified := r.QualifyPivotColumn(column)
	r.pivotWheres = append(r.pivotWheres, func(q *query.Builder) { q.OrWhereNotBetween(qualified, from, to) })
	r.Query.GetQuery().OrWhereNotBetween(qualified, from, to)
	return r
}

// OrderByPivotDesc answers BelongsToMany::orderByPivotDesc.
func (r *BelongsToMany) OrderByPivotDesc(column string) *BelongsToMany {
	return r.OrderByPivot(column, "desc")
}

// GetRelationExistenceQueryForSelfJoin answers the method of the same name.
//
// Exported because the PHP's is public, and because a self-referencing
// many-to-many -- a user who follows users -- is the case where somebody has to
// read it to believe the aliasing.
func (r *BelongsToMany) GetRelationExistenceQueryForSelfJoin(q Builder, parentQuery Builder, columns ...any) Builder {
	return r.getRelationExistenceQueryForSelfJoin(q, parentQuery, columns...)
}

// Find answers Relation::find, through the builder.
func (r *BaseRelation) Find(ctx context.Context, g auth.Grant, id any) (Model, error) {
	scoped, err := r.scoped(g)
	if err != nil {
		return nil, err
	}
	return scoped.Find(ctx, g, id)
}

// FindMany answers BelongsToMany::findMany.
func (r *BaseRelation) FindMany(ctx context.Context, g auth.Grant, ids []any) ([]Model, error) {
	if len(ids) == 0 {
		return []Model{}, nil
	}

	scoped, err := r.scoped(g)
	if err != nil {
		return nil, err
	}
	return scoped.WhereKey(ids...).Get(ctx, g)
}

// FindOrFail answers BelongsToMany::findOrFail: the miss is an error rather
// than a nil the caller has to remember to check.
func (r *BaseRelation) FindOrFail(ctx context.Context, g auth.Grant, id any) (Model, error) {
	found, err := r.Find(ctx, g, id)
	if err != nil {
		return nil, err
	}
	if found == nil {
		return nil, notFound(r.Related, id)
	}
	return found, nil
}

// FindOr answers BelongsToMany::findOr: the callback answers the miss.
func (r *BaseRelation) FindOr(ctx context.Context, g auth.Grant, id any, callback func() (Model, error)) (Model, error) {
	found, err := r.Find(ctx, g, id)
	if err != nil || found != nil {
		return found, err
	}
	return callback()
}

// FindSole answers BelongsToMany::findSole.
func (r *BaseRelation) FindSole(ctx context.Context, g auth.Grant, id any) (Model, error) {
	scoped, err := r.scoped(g)
	if err != nil {
		return nil, err
	}

	results, err := scoped.WhereKey(id).Limit(2).Get(ctx, g)
	if err != nil {
		return nil, err
	}
	return soleOf(results, r.Related)
}

// FirstWhere answers BelongsToMany::firstWhere.
func (r *BaseRelation) FirstWhere(ctx context.Context, g auth.Grant, column string, args ...any) (Model, error) {
	scoped, err := r.scoped(g)
	if err != nil {
		return nil, err
	}
	return scoped.Where(column, args...).First(ctx, g)
}

// FirstOrFail answers BelongsToMany::firstOrFail.
func (r *BaseRelation) FirstOrFail(ctx context.Context, g auth.Grant) (Model, error) {
	found, err := r.First(ctx, g)
	if err != nil {
		return nil, err
	}
	if found == nil {
		return nil, notFound(r.Related, nil)
	}
	return found, nil
}

// FirstOr answers BelongsToMany::firstOr.
func (r *BaseRelation) FirstOr(ctx context.Context, g auth.Grant, callback func() (Model, error)) (Model, error) {
	found, err := r.First(ctx, g)
	if err != nil || found != nil {
		return found, err
	}
	return callback()
}

// notFound answers ModelNotFoundException::setModel: the error names the table
// and the key, because "no query results" with neither is a message that sends
// the reader back to the code to find out what was being looked for.
func notFound(related Model, id any) error {
	if id == nil {
		return fmt.Errorf("%w: table %s", ErrModelNotFound, related.GetTable())
	}
	return fmt.Errorf("%w: table %s, key %v", ErrModelNotFound, related.GetTable(), id)
}

// soleOf answers the shared tail of sole and findSole.
func soleOf(results []Model, related Model) (Model, error) {
	switch len(results) {
	case 0:
		return nil, notFound(related, nil)
	case 1:
		return results[0], nil
	default:
		return nil, fmt.Errorf("%w: %d found on table %s", ErrMultipleRecordsFound, len(results), related.GetTable())
	}
}
