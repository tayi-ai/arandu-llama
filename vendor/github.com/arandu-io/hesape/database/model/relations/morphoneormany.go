package relations

import (
	"context"
	"strings"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/database/model/relations/concerns"
	"github.com/arandu-io/hesape/str"
)

// MorphOneOrMany is the shared body of MorphOne and MorphMany.
//
// A has-many with a second column: the foreign key says which row, the morph
// type says which table it was on. The type column is what makes one comments
// table serve posts and videos both, and it is why the constraint is two
// clauses rather than one -- an id of 7 exists on every table.
type MorphOneOrMany struct {
	HasOneOrMany

	morphType  string
	morphClass string
}

// NewMorphOneOrMany answers MorphOneOrMany::__construct.
//
// The morph class is the parent's alias, and it is the value written into the
// type column. See morphmap.go for why it is an alias here and a class name
// there.
func NewMorphOneOrMany(query Builder, parent Model, typ, id, localKey string) MorphOneOrMany {
	return MorphOneOrMany{
		HasOneOrMany: NewHasOneOrMany(query, parent, id, localKey),
		morphType:    typ,
		morphClass:   parent.GetMorphClass(),
	}
}

// AddConstraints answers MorphOneOrMany::addConstraints.
func (r *MorphOneOrMany) AddConstraints() {
	r.GetRelationQuery().Where(r.morphType, r.morphClass)
	r.HasOneOrMany.AddConstraints()
}

// AddEagerConstraints answers MorphOneOrMany::addEagerConstraints.
//
// The type clause goes on after the keys, exactly as in the PHP: without it the
// eager load would pull the comments of the video whose id happens to match the
// post's -- one query, wrong rows, and every parent looking plausible.
func (r *MorphOneOrMany) AddEagerConstraints(models []Model) error {
	if err := r.HasOneOrMany.AddEagerConstraints(models); err != nil {
		return err
	}
	r.GetRelationQuery().Where(r.morphType, r.morphClass)
	return nil
}

// SetForeignAttributesForCreate answers
// MorphOneOrMany::setForeignAttributesForCreate.
func (r *MorphOneOrMany) SetForeignAttributesForCreate(model Model) {
	model.SetAttribute(r.GetForeignKeyName(), r.GetParentKey())
	model.SetAttribute(r.GetMorphType(), r.morphClass)
	r.ApplyInverseRelationToModel(model)
}

// Make answers HasOneOrMany::make with the morph type set.
func (r *MorphOneOrMany) Make(attributes map[string]any) Model {
	instance := r.Related.NewInstance(attributes)
	r.SetForeignAttributesForCreate(instance)
	return instance
}

// Create answers HasOneOrMany::create with the morph type set.
func (r *MorphOneOrMany) Create(ctx context.Context, g auth.Grant, attributes map[string]any) (Model, error) {
	instance := r.Related.NewInstance(attributes)
	r.SetForeignAttributesForCreate(instance)

	if err := instance.Save(ctx, g); err != nil {
		return nil, err
	}
	r.ApplyInverseRelationToModel(instance)
	return instance, nil
}

// ForceCreate answers MorphOneOrMany::forceCreate.
func (r *MorphOneOrMany) ForceCreate(ctx context.Context, g auth.Grant, attributes map[string]any) (Model, error) {
	filled := merge(attributes, map[string]any{
		r.GetForeignKeyName(): r.GetParentKey(),
		r.GetMorphType():      r.morphClass,
	})

	instance := r.Related.NewInstance(nil)
	instance.ForceFill(filled)

	if err := instance.Save(ctx, g); err != nil {
		return nil, err
	}
	return r.ApplyInverseRelationToModel(instance), nil
}

// Upsert answers MorphOneOrMany::upsert: HasOneOrMany's upsert with the morph
// type stamped on every row as well as the foreign key.
//
// Both columns are needed and for the same reason. A comments table shared by
// posts and videos keys a comment by the pair, so a row written with the id and
// not the type belongs to whichever of the two happens to share the number --
// which is a row appearing under the wrong parent, not a row appearing nowhere.
func (r *MorphOneOrMany) Upsert(ctx context.Context, g auth.Grant, values []map[string]any, uniqueBy, update []string) (int64, error) {
	typed := make([]map[string]any, 0, len(values))
	for _, row := range values {
		typed = append(typed, merge(row, map[string]any{r.GetMorphType(): r.morphClass}))
	}
	return r.HasOneOrMany.Upsert(ctx, g, typed, uniqueBy, update)
}

// GetRelationExistenceQuery answers MorphOneOrMany::getRelationExistenceQuery.
func (r *MorphOneOrMany) GetRelationExistenceQuery(q Builder, parentQuery Builder, columns ...any) Builder {
	return r.HasOneOrMany.
		GetRelationExistenceQuery(q, parentQuery, columns...).
		Where(q.GetModel().QualifyColumn(r.GetMorphType()), r.morphClass)
}

// GetQualifiedMorphType answers MorphOneOrMany::getQualifiedMorphType.
func (r *MorphOneOrMany) GetQualifiedMorphType() string { return r.morphType }

// GetMorphType answers MorphOneOrMany::getMorphType: the plain column.
func (r *MorphOneOrMany) GetMorphType() string {
	segments := strings.Split(r.morphType, ".")
	return segments[len(segments)-1]
}

// GetMorphClass answers MorphOneOrMany::getMorphClass.
func (r *MorphOneOrMany) GetMorphClass() string { return r.morphClass }

// PossibleInverseRelations answers MorphOneOrMany::getPossibleInverseRelations:
// commentable_type suggests the relation is called commentable.
func (r *MorphOneOrMany) PossibleInverseRelations() []string {
	candidates := []string{str.Camel(strings.TrimSuffix(r.GetMorphType(), "_type"))}
	return append(candidates, concerns.DefaultPossibleInverseRelations(r.GetForeignKeyName(), r.Parent, r.Related)...)
}
