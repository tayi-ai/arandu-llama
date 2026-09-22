package relations

import (
	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/database/model/relations/concerns"
	"github.com/arandu-io/hesape/database/query"
)

// MorphToMany is a many-to-many whose intermediate table serves several parent
// types -- taggables holding the tags of posts and of videos both.
//
// Every statement it writes carries the type as well as the two keys, and every
// read filters on it. Without that, a post and a video sharing an id share
// their tags.
type MorphToMany struct {
	BelongsToMany

	morphType  string
	morphClass string
	inverse    bool
}

// newMorphToMany answers MorphToMany::__construct, without the where that
// narrows the pivot to one parent.
//
// The join and the type clause are both on this side of the split, and for the
// same reason: neither is about which parent is being read. An eager load walks
// the pivot for every parent at once and still must reach the related table and
// still must not pick up the tags of the video whose id matches the post's.
func newMorphToMany(q Builder, parent Model, name, table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey, relationName string, inverse bool) *MorphToMany {
	morphClass := parent.GetMorphClass()
	if inverse {
		morphClass = q.GetModel().GetMorphClass()
	}

	relation := &MorphToMany{
		BelongsToMany: *newBelongsToMany(q, parent, table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey, relationName),
		morphType:     name + "_type",
		morphClass:    morphClass,
		inverse:       inverse,
	}
	relation.InteractsWithPivotTable = concerns.InteractsWithPivotTable{Host: relation}
	relation.AliasedPivotColumns = relation.aliasedPivotColumns
	relation.ExistenceCompareKey = relation.GetExistenceCompareKey

	// baseAttachRecord in the PHP adds the type to every row it writes. Here the
	// same thing is said with the mechanism that already exists: a pivot value
	// is a column written on every attach, which is exactly what the type is.
	relation.PivotValues = append(relation.PivotValues, concerns.PivotValue{
		Column: relation.morphType,
		Value:  relation.morphClass,
	})

	// The type clause is this relation's own, and it is added here rather than
	// in an overridden addWhereConstraints because Go dispatches statically:
	// the embedded AddConstraints would never have reached an override.
	relation.Query.Where(relation.QualifyPivotColumn(relation.morphType), relation.morphClass)

	return relation
}

// NewMorphToMany answers MorphToMany::__construct.
func NewMorphToMany(q Builder, parent Model, name, table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey, relationName string, inverse bool) *MorphToMany {
	relation := newMorphToMany(q, parent, name, table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey, relationName, inverse)
	relation.addWhereConstraints()
	return relation
}

// NewMorphToManyUnconstrained builds the relation without narrowing it to one
// parent.
//
// It exists because the constraint used to be switched off through a
// process-wide flag, which meant a relation built on another goroutine while
// the flag was down came back unconstrained -- every parent's children, in a
// well-formed query nobody could tell apart from the right one. The call site
// says which it wants now, and there is no flag to leave down.
func NewMorphToManyUnconstrained(q Builder, parent Model, name, table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey, relationName string, inverse bool) *MorphToMany {
	return newMorphToMany(q, parent, name, table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey, relationName, inverse)
}

// AddEagerConstraints answers MorphToMany::addEagerConstraints.
func (r *MorphToMany) AddEagerConstraints(models []Model) error {
	if err := r.BelongsToMany.AddEagerConstraints(models); err != nil {
		return err
	}
	r.Query.Where(r.QualifyPivotColumn(r.morphType), r.morphClass)
	return nil
}

// NewPivotQuery answers MorphToMany::newPivotQuery.
func (r *MorphToMany) NewPivotQuery(g auth.Grant) (*query.Builder, error) {
	statement, err := r.BelongsToMany.NewPivotQuery(g)
	if err != nil {
		return nil, err
	}
	return statement.Where(r.morphType, r.morphClass), nil
}

// NewPivot answers MorphToMany::newPivot.
func (r *MorphToMany) NewPivot(attributes map[string]any, exists bool) Model {
	values := map[string]any{r.morphType: r.morphClass}
	for _, value := range r.PivotValues {
		values[value.Column] = value.Value
	}
	for key, value := range attributes {
		values[key] = value
	}

	if r.using != nil {
		return r.using(r.Parent, values, r.table, exists)
	}

	pivot := MorphPivotFromAttributes(r.Parent, values, r.table, exists)
	pivot.NewQueryFor = r.newQueryForTable
	pivot.SetPivotKeys(r.foreignPivotKey, r.relatedPivotKey)
	pivot.SetRelatedModel(r.Related)
	pivot.SetMorphType(r.morphType)
	pivot.SetMorphClass(r.morphClass)
	return pivot
}

// GetRelationExistenceQuery answers MorphToMany::getRelationExistenceQuery.
func (r *MorphToMany) GetRelationExistenceQuery(q Builder, parentQuery Builder, columns ...any) Builder {
	return r.BelongsToMany.
		GetRelationExistenceQuery(q, parentQuery, columns...).
		Where(r.QualifyPivotColumn(r.morphType), r.morphClass)
}

// GetMorphType answers MorphToMany::getMorphType.
func (r *MorphToMany) GetMorphType() string { return r.morphType }

// GetQualifiedMorphTypeName answers MorphToMany::getQualifiedMorphTypeName.
func (r *MorphToMany) GetQualifiedMorphTypeName() string { return r.QualifyPivotColumn(r.morphType) }

// GetMorphClass answers MorphToMany::getMorphClass.
func (r *MorphToMany) GetMorphClass() string { return r.morphClass }

// GetInverse answers MorphToMany::getInverse: whether this is the morphedByMany
// side, which is what decides whose alias goes in the type column.
func (r *MorphToMany) GetInverse() bool { return r.inverse }

// aliasedPivotColumns answers MorphToMany::aliasedPivotColumns: the type column
// comes back with the pivot, so the hydrated pivot can say which parent type it
// belonged to.
func (r *MorphToMany) aliasedPivotColumns() []any {
	columns := []any{
		r.QualifyPivotColumn(r.foreignPivotKey) + " as pivot_" + r.foreignPivotKey,
		r.QualifyPivotColumn(r.relatedPivotKey) + " as pivot_" + r.relatedPivotKey,
		r.QualifyPivotColumn(r.morphType) + " as pivot_" + r.morphType,
	}
	for _, column := range r.PivotColumns {
		columns = append(columns, r.QualifyPivotColumn(column)+" as pivot_"+column)
	}
	return columns
}
