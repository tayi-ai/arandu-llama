package relations

import (
	"context"
	"sort"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/database/model/relations/concerns"
)

// BelongsTo is the inverse relation, where the foreign key is on this table and
// points at the other one.
//
// The model holding the key is reachable as both the child and the parent: the
// shared half calls it the parent, and "parent" reads backwards for the row that
// carries the key.
type BelongsTo struct {
	BaseRelation
	concerns.SupportsDefaultModels
	concerns.ComparesRelatedModels

	child        Model
	foreignKey   string
	ownerKey     string
	relationName string
}

// NewBelongsTo answers BelongsTo::__construct.
func newBelongsTo(query Builder, child Model, foreignKey, ownerKey, relationName string) *BelongsTo {
	relation := &BelongsTo{
		BaseRelation: NewBaseRelation(query, child),
		child:        child,
		foreignKey:   foreignKey,
		ownerKey:     ownerKey,
		relationName: relationName,
	}
	relation.ExistenceCompareKey = relation.GetExistenceCompareKey

	relation.SupportsDefaultModels = concerns.SupportsDefaultModels{
		NewRelatedInstanceFor: relation.newRelatedInstanceFor,
	}
	relation.ComparesRelatedModels = concerns.ComparesRelatedModels{
		CompareRelated:        relation.Related,
		CompareParentKey:      relation.GetParentKey,
		CompareRelatedKeyFrom: relation.getRelatedKeyFrom,
	}

	return relation
}

// NewBelongsTo builds the relation and narrows it to its parent.
func NewBelongsTo(query Builder, child Model, foreignKey, ownerKey, relationName string) *BelongsTo {
	relation := newBelongsTo(query, child, foreignKey, ownerKey, relationName)
	relation.AddConstraints()
	return relation
}

// AddConstraints answers BelongsTo::addConstraints.
func (r *BelongsTo) AddConstraints() {
	r.Query.Where(r.GetQualifiedOwnerKeyName(), "=", r.getForeignKeyFrom(r.child))
}

// AddEagerConstraints answers BelongsTo::addEagerConstraints.
func (r *BelongsTo) AddEagerConstraints(models []Model) error {
	keys, err := r.getEagerModelKeys(models)
	if err != nil {
		return err
	}
	r.WhereInEager(r.GetQualifiedOwnerKeyName(), keys)
	return nil
}

// getEagerModelKeys answers BelongsTo::getEagerModelKeys.
//
// It gathers the foreign key off each child, skipping the nulls -- a child with
// no owner contributes nothing to the `in (...)` rather than a null that would
// match nothing and lengthen the statement.
func (r *BelongsTo) getEagerModelKeys(models []Model) ([]any, error) {
	seen := make(map[string]struct{}, len(models))
	keys := make([]any, 0, len(models))

	for _, model := range models {
		value := r.getForeignKeyFrom(model)
		if value == nil {
			continue
		}
		key, err := concerns.GetDictionaryKey(value)
		if err != nil {
			return nil, err
		}
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		keys = append(keys, value)
	}

	sortByDictionaryKey(keys)
	return keys, nil
}

// InitRelation answers BelongsTo::initRelation.
func (r *BelongsTo) InitRelation(models []Model, relation string) []Model {
	for _, model := range models {
		model.SetRelation(relation, r.GetDefaultFor(model))
	}
	return models
}

// Match answers BelongsTo::match.
//
// The dictionary here is one owner per key rather than a bucket of children,
// which is the whole difference between this and HasMany: many children share
// one owner, so the same loaded model is handed to each of them.
func (r *BelongsTo) Match(models []Model, results []Model, relation string) ([]Model, error) {
	dictionary := make(map[string]Model, len(results))
	for _, result := range results {
		key, err := concerns.GetDictionaryKey(r.getRelatedKeyFrom(result))
		if err != nil {
			return nil, err
		}
		dictionary[key] = result
	}

	for _, model := range models {
		key, err := concerns.GetDictionaryKey(r.getForeignKeyFrom(model))
		if err != nil {
			return nil, err
		}
		if owner, ok := dictionary[key]; ok {
			model.SetRelation(relation, owner)
		}
	}

	return models, nil
}

// GetResults answers BelongsTo::getResults.
func (r *BelongsTo) GetResults(ctx context.Context, g auth.Grant) (any, error) {
	if r.getForeignKeyFrom(r.child) == nil {
		return r.GetDefaultFor(r.Parent), nil
	}

	result, err := r.First(ctx, g)
	if err != nil {
		return nil, err
	}
	if result == nil {
		return r.GetDefaultFor(r.Parent), nil
	}
	return result, nil
}

// Associate answers BelongsTo::associate.
//
// It takes a model or a bare key, as the PHP does. With a model the relation is
// set too, so reading it back does not go to the database; with a key it is
// unset, because what is on the other end is not known.
func (r *BelongsTo) Associate(model any) Model {
	if related, ok := model.(Model); ok && related != nil {
		r.child.SetAttribute(r.foreignKey, related.GetAttribute(r.ownerKey))
		r.child.SetRelation(r.relationName, related)
		return r.child
	}

	r.child.SetAttribute(r.foreignKey, model)
	r.child.UnsetRelation(r.relationName)
	return r.child
}

// Dissociate answers BelongsTo::dissociate.
func (r *BelongsTo) Dissociate() Model {
	r.child.SetAttribute(r.foreignKey, nil)
	r.child.SetRelation(r.relationName, nil)
	return r.child
}

// Disassociate answers BelongsTo::disassociate, the alias the PHP keeps for the
// spelling people reach for.
func (r *BelongsTo) Disassociate() Model { return r.Dissociate() }

// Touch answers BelongsTo::touch: nothing to stamp when there is no owner.
func (r *BelongsTo) Touch(ctx context.Context, g auth.Grant) error {
	if r.GetParentKey() == nil {
		return nil
	}
	return r.BaseRelation.Touch(ctx, g)
}

// GetRelationExistenceQuery answers BelongsTo::getRelationExistenceQuery.
func (r *BelongsTo) GetRelationExistenceQuery(q Builder, parentQuery Builder, columns ...any) Builder {
	if len(columns) == 0 {
		columns = []any{"*"}
	}
	if sameTable(q, parentQuery) {
		return r.GetRelationExistenceQueryForSelfRelation(q, parentQuery, columns...)
	}
	return q.Select(columns...).WhereColumn(
		r.GetQualifiedForeignKeyName(), "=", q.GetModel().QualifyColumn(r.ownerKey),
	)
}

// GetRelationExistenceQueryForSelfRelation answers
// BelongsTo::getRelationExistenceQueryForSelfRelation.
func (r *BelongsTo) GetRelationExistenceQueryForSelfRelation(q Builder, parentQuery Builder, columns ...any) Builder {
	hash := r.GetRelationCountHash()

	q.Select(columns...).GetQuery().From(q.GetModel().GetTable() + " as " + hash)
	q.GetModel().SetTable(hash)

	return q.WhereColumn(hash+"."+r.ownerKey, "=", r.GetQualifiedForeignKeyName())
}

// GetChild answers BelongsTo::getChild.
func (r *BelongsTo) GetChild() Model { return r.child }

// GetForeignKeyName answers BelongsTo::getForeignKeyName.
func (r *BelongsTo) GetForeignKeyName() string { return r.foreignKey }

// GetQualifiedForeignKeyName answers BelongsTo::getQualifiedForeignKeyName.
func (r *BelongsTo) GetQualifiedForeignKeyName() string {
	return r.child.QualifyColumn(r.foreignKey)
}

// GetParentKey answers BelongsTo::getParentKey: the child's foreign key, which
// is the value the owner is found by.
func (r *BelongsTo) GetParentKey() any { return r.getForeignKeyFrom(r.child) }

// GetOwnerKeyName answers BelongsTo::getOwnerKeyName.
func (r *BelongsTo) GetOwnerKeyName() string { return r.ownerKey }

// GetQualifiedOwnerKeyName answers BelongsTo::getQualifiedOwnerKeyName.
func (r *BelongsTo) GetQualifiedOwnerKeyName() string {
	return r.Related.QualifyColumn(r.ownerKey)
}

// GetRelationName answers BelongsTo::getRelationName.
func (r *BelongsTo) GetRelationName() string { return r.relationName }

// GetExistenceCompareKey answers the base class's method for this relation.
func (r *BelongsTo) GetExistenceCompareKey() string { return r.GetQualifiedForeignKeyName() }

// newRelatedInstanceFor answers BelongsTo::newRelatedInstanceFor.
func (r *BelongsTo) newRelatedInstanceFor(parent Model) Model { return r.Related.NewInstance(nil) }

// getRelatedKeyFrom answers BelongsTo::getRelatedKeyFrom.
func (r *BelongsTo) getRelatedKeyFrom(model Model) any { return model.GetAttribute(r.ownerKey) }

// getForeignKeyFrom answers BelongsTo::getForeignKeyFrom.
func (r *BelongsTo) getForeignKeyFrom(model Model) any { return model.GetAttribute(r.foreignKey) }

// sortByDictionaryKey sorts keys the way Relation::getKeys does, so the emitted
// `in (...)` is the same statement between requests.
//
// It sorts on the rendered key rather than on the value, because the values are
// `any` and Go has nothing to compare two of those with. The rendering is the
// one the dictionary uses, so the order matches the buckets.
func sortByDictionaryKey(keys []any) {
	sort.SliceStable(keys, func(i, j int) bool {
		left, _ := concerns.GetDictionaryKey(keys[i])
		right, _ := concerns.GetDictionaryKey(keys[j])
		return left < right
	})
}

// NewBelongsToUnconstrained builds the relation without narrowing it to one parent.
//
// It exists because the constraint used to be switched off through a
// process-wide flag, which meant a relation built on another goroutine while
// the flag was down came back unconstrained -- every parent's children, in a
// well-formed query nobody could tell apart from the right one. The call site
// says which it wants now, and there is no flag to leave down.
func NewBelongsToUnconstrained(query Builder, child Model, foreignKey, ownerKey, relationName string) *BelongsTo {
	return newBelongsTo(query, child, foreignKey, ownerKey, relationName)
}
