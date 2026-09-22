package relations

import (
	"context"
	"sort"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/database/model/relations/concerns"
)

// MorphTo is the inverse of a polymorphic relation, where the row says both
// which table and which row.
//
// It is the one relation whose eager load is not two queries, and it cannot be:
// the parents point at several tables, and no dialect selects from a table
// named by a column. It is one query per distinct type in the batch -- three
// queries for a feed of posts, videos and links, not one per row -- and that
// distinction is the whole of what eager loading buys here.
type MorphTo struct {
	BelongsTo

	morphType string

	// models is MorphTo::$models: the parents being loaded, kept because
	// GetEager answers them rather than the related rows.
	models []Model

	// dictionary is MorphTo::$dictionary, keyed by morph type and then by
	// foreign key.
	dictionary map[string]map[string][]Model

	morphableConstraints     map[string]func(Builder)
	morphableEagerLoads      map[string][]string
	morphableEagerLoadCounts map[string][]string

	// trashed is what MorphTo::$macroBuffer holds for the three soft delete
	// methods: a decision taken before the types are known, replayed on each
	// per-type query once they are.
	trashed morphTrashed
}

// morphTrashed is which soft deleted rows a MorphTo was asked for.
//
// PHP buffers the calls and replays them, so a second call layers on top of the
// first. Here the last call wins, which is the same answer for every sequence
// anybody writes -- withTrashed().onlyTrashed() reads as "only trashed" either
// way -- and it is one field instead of a list of closures whose combined effect
// has to be simulated to be known.
type morphTrashed int

const (
	// morphTrashedDefault is a MorphTo nobody asked about soft deletes: the
	// query is left alone, which is what the PHP does when the related model
	// has no SoftDeletes and its hasMacro check fails.
	morphTrashedDefault morphTrashed = iota

	// morphTrashedWith answers withTrashed.
	morphTrashedWith

	// morphTrashedWithout answers withoutTrashed.
	morphTrashedWithout

	// morphTrashedOnly answers onlyTrashed.
	morphTrashedOnly
)

// NewMorphTo answers MorphTo::__construct.
//
// ownerKey may be empty, and empty means "the related model's own key name",
// which is not known until the type column is read.
func NewMorphTo(query Builder, parent Model, foreignKey, ownerKey, typ, relation string) *MorphTo {
	return newMorphTo(NewBelongsTo(query, parent, foreignKey, ownerKey, relation), typ)
}

// NewMorphToUnconstrained builds the relation without narrowing it to one
// parent, which is what an eager load needs: the batch resolves the type first
// and then reads every owner of that type in one statement.
func NewMorphToUnconstrained(query Builder, parent Model, foreignKey, ownerKey, typ, relation string) *MorphTo {
	return newMorphTo(NewBelongsToUnconstrained(query, parent, foreignKey, ownerKey, relation), typ)
}

func newMorphTo(belongsTo *BelongsTo, typ string) *MorphTo {
	relation2 := &MorphTo{
		BelongsTo:                *belongsTo,
		morphType:                typ,
		dictionary:               map[string]map[string][]Model{},
		morphableConstraints:     map[string]func(Builder){},
		morphableEagerLoads:      map[string][]string{},
		morphableEagerLoadCounts: map[string][]string{},
	}
	return relation2
}

// AddEagerConstraints answers MorphTo::addEagerConstraints.
//
// It adds no clause to any query. There is no single query to constrain: the
// work is bucketing the parents by type, and the queries are built one per
// bucket in GetEager.
func (r *MorphTo) AddEagerConstraints(models []Model) error {
	r.models = models
	return r.buildDictionary(models)
}

// buildDictionary answers MorphTo::buildDictionary.
func (r *MorphTo) buildDictionary(models []Model) error {
	for _, model := range models {
		typ := model.GetAttribute(r.morphType)
		if typ == nil {
			continue
		}

		morphTypeKey, err := concerns.GetDictionaryKey(typ)
		if err != nil {
			return err
		}
		if morphTypeKey == "" {
			continue
		}
		foreignKeyKey, err := concerns.GetDictionaryKey(model.GetAttribute(r.foreignKey))
		if err != nil {
			return err
		}

		if r.dictionary[morphTypeKey] == nil {
			r.dictionary[morphTypeKey] = map[string][]Model{}
		}
		r.dictionary[morphTypeKey][foreignKeyKey] = append(r.dictionary[morphTypeKey][foreignKeyKey], model)
	}
	return nil
}

// GetEager answers MorphTo::getEager: one query per morph type, matched as it
// goes, and the parents come back out.
//
// The types are visited in sorted order. PHP visits them in the order the
// dictionary was built; a Go map has none, and a query log that shuffles
// between identical requests is a log nobody can diff.
func (r *MorphTo) GetEager(ctx context.Context, g auth.Grant) ([]Model, error) {
	types := make([]string, 0, len(r.dictionary))
	for typ := range r.dictionary {
		types = append(types, typ)
	}
	sort.Strings(types)

	for _, typ := range types {
		results, err := r.getResultsByType(ctx, g, typ)
		if err != nil {
			return nil, err
		}
		if err := r.matchToMorphParents(typ, results); err != nil {
			return nil, err
		}
	}

	return r.models, nil
}

// GetResults answers MorphTo::getResults.
//
// It exists because the inherited one is wrong here, and wrong in the way that
// is hardest to see: BelongsTo::getResults reads the query this relation was
// built with, and for a MorphTo that query is over a placeholder model handed
// to the constructor -- one that only carries a connection to start from. So a
// lazy read selected from whatever table the placeholder named, with no morph
// type in the where clause. The eager path never had the problem, because it
// resolves the type first; the lazy path had no test.
//
// The type comes off the child, like everything else about a morph, and a child
// that names no type has no owner rather than an owner of the wrong kind.
func (r *MorphTo) GetResults(ctx context.Context, g auth.Grant) (any, error) {
	typ := r.child.GetAttribute(r.morphType)
	if typ == nil {
		return r.GetDefaultFor(r.Parent), nil
	}
	morphTypeKey, err := concerns.GetDictionaryKey(typ)
	if err != nil {
		return nil, err
	}
	if morphTypeKey == "" {
		return r.GetDefaultFor(r.Parent), nil
	}

	key := r.getForeignKeyFrom(r.child)
	if key == nil {
		return r.GetDefaultFor(r.Parent), nil
	}

	instance, err := CreateModelByType(morphTypeKey)
	if err != nil {
		return nil, err
	}

	ownerKey := r.ownerKey
	if ownerKey == "" {
		ownerKey = instance.GetKeyName()
	}

	q := instance.NewQuery()
	if constrain, ok := r.morphableConstraints[morphTypeKey]; ok && constrain != nil {
		constrain(q)
	}
	r.applyTrashed(q, instance)

	scoped, err := concerns.ScopeTenant(q, instance, g)
	if err != nil {
		return nil, err
	}

	result, err := scoped.Where(instance.QualifyColumn(ownerKey), "=", key).First(ctx, g)
	if err != nil {
		return nil, err
	}
	if result == nil {
		return r.GetDefaultFor(r.Parent), nil
	}
	return result, nil
}

// getResultsByType answers MorphTo::getResultsByType.
func (r *MorphTo) getResultsByType(ctx context.Context, g auth.Grant, typ string) ([]Model, error) {
	instance, err := CreateModelByType(typ)
	if err != nil {
		return nil, err
	}

	ownerKey := r.ownerKey
	if ownerKey == "" {
		ownerKey = instance.GetKeyName()
	}

	q := instance.NewQuery()
	if constrain, ok := r.morphableConstraints[typ]; ok && constrain != nil {
		constrain(q)
	}
	r.applyTrashed(q, instance)

	scoped, err := concerns.ScopeTenant(q, instance, g)
	if err != nil {
		return nil, err
	}

	return scoped.WhereIn(instance.QualifyColumn(ownerKey), r.gatherKeysByType(typ)).Get(ctx, g)
}

// gatherKeysByType answers MorphTo::gatherKeysByType.
func (r *MorphTo) gatherKeysByType(typ string) []any {
	keys := make([]string, 0, len(r.dictionary[typ]))
	for key := range r.dictionary[typ] {
		if key == "" {
			continue
		}
		keys = append(keys, key)
	}
	sort.Strings(keys)

	out := make([]any, 0, len(keys))
	for _, key := range keys {
		out = append(out, key)
	}
	return out
}

// matchToMorphParents answers MorphTo::matchToMorphParents.
func (r *MorphTo) matchToMorphParents(typ string, results []Model) error {
	for _, result := range results {
		var ownerKey any
		if r.ownerKey != "" {
			ownerKey = result.GetAttribute(r.ownerKey)
		} else {
			ownerKey = result.GetKey()
		}

		key, err := concerns.GetDictionaryKey(ownerKey)
		if err != nil {
			return err
		}

		for _, model := range r.dictionary[typ][key] {
			model.SetRelation(r.relationName, result)
		}
	}
	return nil
}

// Match answers MorphTo::match: nothing, because GetEager already matched.
func (r *MorphTo) Match(models []Model, results []Model, relation string) ([]Model, error) {
	return models, nil
}

// Associate answers MorphTo::associate: it writes both columns, the key and the
// type.
func (r *MorphTo) Associate(model any) Model {
	related, ok := model.(Model)
	if !ok || related == nil {
		r.Parent.SetAttribute(r.foreignKey, nil)
		r.Parent.SetAttribute(r.morphType, nil)
		r.Parent.SetRelation(r.relationName, nil)
		return r.Parent
	}

	key := related.GetKeyName()
	if r.ownerKey != "" && related.GetAttribute(r.ownerKey) != nil {
		key = r.ownerKey
	}

	r.Parent.SetAttribute(r.foreignKey, related.GetAttribute(key))
	r.Parent.SetAttribute(r.morphType, related.GetMorphClass())
	r.Parent.SetRelation(r.relationName, related)
	return r.Parent
}

// Dissociate answers MorphTo::dissociate.
func (r *MorphTo) Dissociate() Model {
	r.Parent.SetAttribute(r.foreignKey, nil)
	r.Parent.SetAttribute(r.morphType, nil)
	r.Parent.SetRelation(r.relationName, nil)
	return r.Parent
}

// MorphWith answers MorphTo::morphWith: nested eager loads, per morph type.
func (r *MorphTo) MorphWith(with map[string][]string) *MorphTo {
	for typ, relations := range with {
		r.morphableEagerLoads[typ] = relations
	}
	return r
}

// MorphWithCount answers MorphTo::morphWithCount: relationship counts to load
// per morph type, alongside the relations morphWith names.
func (r *MorphTo) MorphWithCount(withCount map[string][]string) *MorphTo {
	if r.morphableEagerLoadCounts == nil {
		r.morphableEagerLoadCounts = map[string][]string{}
	}
	for typ, counts := range withCount {
		r.morphableEagerLoadCounts[typ] = counts
	}
	return r
}

// GetMorphableEagerLoadCounts answers the map morphWithCount built.
func (r *MorphTo) GetMorphableEagerLoadCounts() map[string][]string {
	return r.morphableEagerLoadCounts
}

// GetMorphableEagerLoads answers the map morphWith built, for the builder that
// runs the nested loads.
func (r *MorphTo) GetMorphableEagerLoads() map[string][]string { return r.morphableEagerLoads }

// Constrain answers MorphTo::constrain: a callback per morph type, applied to
// that type's query.
func (r *MorphTo) Constrain(callbacks map[string]func(Builder)) *MorphTo {
	for typ, callback := range callbacks {
		r.morphableConstraints[typ] = callback
	}
	return r
}

// WithTrashed answers MorphTo::withTrashed: soft deleted rows come back too.
//
// A MorphTo cannot answer this when it is called. The related model is named by
// a column, so which types are involved -- and which of them are even soft
// deletable -- is not known until the type column has been read. PHP records the
// call in $macroBuffer and replays it on each per-type builder; this records it
// in a field and applies it in getResultsByType, at the same moment and for the
// same reason.
//
// A type that is not soft deletable is left alone rather than refused, which is
// what the PHP's hasMacro check does. One feed of posts and videos where only
// posts are soft deletable is the ordinary case, not a mistake.
func (r *MorphTo) WithTrashed() *MorphTo {
	r.trashed = morphTrashedWith
	return r
}

// WithoutTrashed answers MorphTo::withoutTrashed: soft deleted rows are
// excluded.
//
// The clause is added rather than a global scope being left in place. A relation
// here holds no scope registry -- see WithTrashedParents on HasOneOrManyThrough
// for the same trade -- so nothing filters deleted rows unless asked, and this
// is the asking.
func (r *MorphTo) WithoutTrashed() *MorphTo {
	r.trashed = morphTrashedWithout
	return r
}

// OnlyTrashed answers MorphTo::onlyTrashed: only soft deleted rows come back.
func (r *MorphTo) OnlyTrashed() *MorphTo {
	r.trashed = morphTrashedOnly
	return r
}

// applyTrashed is MorphTo::replayMacros for the three soft delete methods: the
// buffered decision, applied to one type's query now that the type is known.
//
// The instance is asked whether it is soft deletable at all, which is the
// hasMacro check the PHP closures make. A model without the deleted_at column
// gets no clause, because a filter on a column that does not exist is not a
// stricter query -- it is an error from the database.
func (r *MorphTo) applyTrashed(q Builder, instance Model) {
	if r.trashed == morphTrashedDefault {
		return
	}

	deletable, ok := instance.(SoftDeletableThrough)
	if !ok || !deletable.IsSoftDeletable() {
		return
	}

	column := instance.QualifyColumn(deletable.GetDeletedAtColumn())
	switch r.trashed {
	case morphTrashedWithout:
		q.GetQuery().WhereNull(column)
	case morphTrashedOnly:
		q.WhereNotNull(column)
	}
}

// GetMorphType answers MorphTo::getMorphType.
func (r *MorphTo) GetMorphType() string { return r.morphType }

// GetDictionary answers MorphTo::getDictionary.
func (r *MorphTo) GetDictionary() map[string]map[string][]Model { return r.dictionary }

// GetQualifiedOwnerKeyName answers MorphTo::getQualifiedOwnerKeyName: empty
// when there is no owner key, because there is no table to qualify it with
// until the type is read.
func (r *MorphTo) GetQualifiedOwnerKeyName() string {
	if r.ownerKey == "" {
		return ""
	}
	return r.BelongsTo.GetQualifiedOwnerKeyName()
}
