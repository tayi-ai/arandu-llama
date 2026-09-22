package model

import (
	"sort"
	"strings"

	"github.com/arandu-io/hesape/database/model/relations"
	"github.com/arandu-io/hesape/str"
)

// The functions that build a relation from a pair of models, and the two that
// answer the conventional names those relations default to.
//
// They are functions rather than methods because a relation needs both ends,
// and a method has one receiver.
//
// They moved here from a package that had no callers and was deleted. They are
// the only implementation of the key-naming conventions in the collection --
// user_id from User, role_user from Role and User, alphabetical so that the
// intermediate table is the same table read from either side -- and losing them
// would mean writing those rules a second time, differently.

// GetMorphs returns the pair of column names a polymorphic relation reads.
func GetMorphs(name, typ, id string) (string, string) {
	if typ == "" {
		typ = name + "_type"
	}
	if id == "" {
		id = name + "_id"
	}
	return typ, id
}

// JoiningTable returns the conventional name of an intermediate table, the
// two model names in alphabetical order.
//
// Alphabetical is what makes it the same table from both sides -- role_user
// whether you start at the user or at the role -- and it is why a many-to-many
// declared on both models needs no configuration at all.
func JoiningTable(parent, related relations.Model) string {
	segments := []string{
		str.Snake(parent.GetMorphClass(), "_"),
		str.Snake(related.GetMorphClass(), "_"),
	}
	sort.Strings(segments)
	return strings.Join(segments, "_")
}

// HasOne returns a has-one relation from parent to related.
//
// An empty foreignKey defaults to the parent's conventional foreign key,
// user_id for a model whose alias is user.
func HasOne(parent, related relations.Model, foreignKey, localKey string) *relations.HasOne {
	foreignKey, localKey = defaultHasKeys(parent, foreignKey, localKey)
	return relations.NewHasOne(related.NewQuery(), parent, related.QualifyColumn(foreignKey), localKey)
}

// HasMany returns a has-many relation from parent to related.
func HasMany(parent, related relations.Model, foreignKey, localKey string) *relations.HasMany {
	foreignKey, localKey = defaultHasKeys(parent, foreignKey, localKey)
	return relations.NewHasMany(related.NewQuery(), parent, related.QualifyColumn(foreignKey), localKey)
}

// BelongsTo returns a belongs-to relation from child to related.
//
// relation is the name the relation is read under, and it is required
// rather than guessed from the call site. It is not decoration: associate
// and dissociate write the loaded relation under it.
func BelongsTo(child, related relations.Model, foreignKey, ownerKey, relation string) *relations.BelongsTo {
	foreignKey, ownerKey = defaultBelongsToKeys(related, foreignKey, ownerKey, relation)
	return relations.NewBelongsTo(related.NewQuery(), child, foreignKey, ownerKey, relation)
}

// MorphOne returns a morph-one relation from parent to related.
func MorphOne(parent, related relations.Model, name, typ, id, localKey string) *relations.MorphOne {
	typ, id, localKey = defaultMorphKeys(parent, name, typ, id, localKey)
	return relations.NewMorphOne(related.NewQuery(), parent, related.QualifyColumn(typ), related.QualifyColumn(id), localKey)
}

// MorphMany returns a morph-many relation from parent to related.
func MorphMany(parent, related relations.Model, name, typ, id, localKey string) *relations.MorphMany {
	typ, id, localKey = defaultMorphKeys(parent, name, typ, id, localKey)
	return relations.NewMorphMany(related.NewQuery(), parent, related.QualifyColumn(typ), related.QualifyColumn(id), localKey)
}

// MorphTo returns a morph-to relation on parent.
//
// related only carries a connection to start from, and is replaced per
// type once the type column is read: Go needs something concrete to start
// with, where the query is otherwise built fresh per type.
func MorphTo(parent, related relations.Model, name, typ, id, ownerKey string) *relations.MorphTo {
	typ, id = GetMorphs(str.Snake(name, "_"), typ, id)
	return relations.NewMorphTo(related.NewQuery(), parent, id, ownerKey, typ, name)
}

// BelongsToMany returns a many-to-many relation from parent to related.
func BelongsToMany(parent, related relations.Model, table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey, relation string) *relations.BelongsToMany {
	table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey = defaultPivotKeys(
		parent, related, table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey)

	return relations.NewBelongsToMany(
		related.NewQuery(), parent, table,
		foreignPivotKey, relatedPivotKey, parentKey, relatedKey, relation,
	)
}

// MorphToMany returns a polymorphic many-to-many relation from parent to
// related.
func MorphToMany(parent, related relations.Model, name, table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey, relation string, inverse bool) *relations.MorphToMany {
	table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey = defaultMorphPivotKeys(
		parent, related, name, table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey)

	return relations.NewMorphToMany(
		related.NewQuery(), parent, name, table,
		foreignPivotKey, relatedPivotKey, parentKey, relatedKey, relation, inverse,
	)
}

// MorphedByMany returns the other side of a MorphToMany, where this model
// is what the intermediate table points at.
func MorphedByMany(parent, related relations.Model, name, table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey, relation string) *relations.MorphToMany {
	if foreignPivotKey == "" {
		foreignPivotKey = parent.GetForeignKey()
	}
	if relatedPivotKey == "" {
		relatedPivotKey = name + "_id"
	}
	return MorphToMany(parent, related, name, table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey, relation, true)
}

// HasManyThrough returns a has-many-through relation from farParent to
// related.
func HasManyThrough(farParent, through, related relations.Model, firstKey, secondKey, localKey, secondLocalKey string) *relations.HasManyThrough {
	firstKey, secondKey, localKey, secondLocalKey = defaultThroughKeys(farParent, through, firstKey, secondKey, localKey, secondLocalKey)
	return relations.NewHasManyThrough(related.NewQuery(), farParent, through, firstKey, secondKey, localKey, secondLocalKey)
}

// HasOneThrough returns a has-one-through relation from farParent to
// related.
func HasOneThrough(farParent, through, related relations.Model, firstKey, secondKey, localKey, secondLocalKey string) *relations.HasOneThrough {
	firstKey, secondKey, localKey, secondLocalKey = defaultThroughKeys(farParent, through, firstKey, secondKey, localKey, secondLocalKey)
	return relations.NewHasOneThrough(related.NewQuery(), farParent, through, firstKey, secondKey, localKey, secondLocalKey)
}

// The conventions, one function each, because both the constrained constructor
// and its Unconstrained twin default the same way. Two copies of a rule that
// picks a column name is two schemas one refactor apart.

func defaultHasKeys(parent relations.Model, foreignKey, localKey string) (string, string) {
	if foreignKey == "" {
		foreignKey = parent.GetForeignKey()
	}
	if localKey == "" {
		localKey = parent.GetKeyName()
	}
	return foreignKey, localKey
}

func defaultBelongsToKeys(related relations.Model, foreignKey, ownerKey, relation string) (string, string) {
	if foreignKey == "" {
		foreignKey = str.Snake(relation, "_") + "_" + related.GetKeyName()
	}
	if ownerKey == "" {
		ownerKey = related.GetKeyName()
	}
	return foreignKey, ownerKey
}

func defaultMorphKeys(parent relations.Model, name, typ, id, localKey string) (string, string, string) {
	typ, id = GetMorphs(name, typ, id)
	if localKey == "" {
		localKey = parent.GetKeyName()
	}
	return typ, id, localKey
}

func defaultPivotKeys(parent, related relations.Model, table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey string) (string, string, string, string, string) {
	if table == "" {
		table = JoiningTable(parent, related)
	}
	if foreignPivotKey == "" {
		foreignPivotKey = parent.GetForeignKey()
	}
	if relatedPivotKey == "" {
		relatedPivotKey = related.GetForeignKey()
	}
	if parentKey == "" {
		parentKey = parent.GetKeyName()
	}
	if relatedKey == "" {
		relatedKey = related.GetKeyName()
	}
	return table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey
}

func defaultMorphPivotKeys(parent, related relations.Model, name, table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey string) (string, string, string, string, string) {
	if table == "" {
		table = str.Plural(name)
	}
	if foreignPivotKey == "" {
		foreignPivotKey = name + "_id"
	}
	if relatedPivotKey == "" {
		relatedPivotKey = related.GetForeignKey()
	}
	if parentKey == "" {
		parentKey = parent.GetKeyName()
	}
	if relatedKey == "" {
		relatedKey = related.GetKeyName()
	}
	return table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey
}

func defaultThroughKeys(farParent, through relations.Model, firstKey, secondKey, localKey, secondLocalKey string) (string, string, string, string) {
	if firstKey == "" {
		firstKey = farParent.GetForeignKey()
	}
	if secondKey == "" {
		secondKey = through.GetForeignKey()
	}
	if localKey == "" {
		localKey = farParent.GetKeyName()
	}
	if secondLocalKey == "" {
		secondLocalKey = through.GetKeyName()
	}
	return firstKey, secondKey, localKey, secondLocalKey
}
