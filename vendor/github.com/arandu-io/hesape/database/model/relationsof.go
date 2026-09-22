package model

import (
	"github.com/arandu-io/hesape/database/model/relations"
	"github.com/arandu-io/hesape/str"
)

// The relation constructors as an application writes them.
//
// The functions in relationfactories.go take the narrow interface the relations
// tree consumes, and they are what does the work; these take the typed models
// and hand over the refs. The difference at the call site is the whole point:
//
//	func (u *User) Posts() *relations.HasMany {
//		return model.HasManyOf(u, Posts, "", "")
//	}
//
// rather than remembering to write .Ref() twice and finding out at run time
// which of the two was the parent.
//
// An empty key means the convention: user_id from a User, id from the parent,
// role_user from a Role and a User sorted so the intermediate table is the same
// table read from either side. Naming one is for the schema that does not follow
// it, and naming one that matches the convention is noise.
//
// # Reading the result back
//
// A relation loads the narrow interface, so what a caller reads is
// model.Related[User, Post](user, "posts"), which hands the typed collection
// back. That one call is the price of a relation tree that cannot be generic --
// see ref.go for why it cannot.
//
// # Two call sites, and which constructor each takes
//
// Every constructor here has an Unconstrained twin, and the difference is one
// where clause: whether the relation is narrowed to the parent it was built on.
//
//   - Reading one model's relation takes the plain one. u.Posts() is that
//     user's posts, so the narrowing is the point.
//   - Registering the relation in RelationResolvers takes the Unconstrained
//     one. The builder resolves it from a model with no key -- the prototype
//     it queries through -- and narrows it to the whole batch afterwards, so a
//     narrowing already on the query is one for a parent that does not exist.
//
// The second is not a variation of the first, it is the reason the split
// exists: a constrained relation registered as a resolver compiles, runs, and
// emits `where user_id = 0 and user_id in (1, 2)`, which matches nothing and
// reads as a parent with no children.
//
// The choice used to be a process-wide flag switched off around a constructor.
// It is two constructors now, because a relation built on another goroutine
// while the flag was down came back unconstrained -- every parent's rows, in a
// statement nobody could tell from the right one.

// HasOneOf returns a has-one from parent to related.
func HasOneOf[P, C any](parent *Model[P], related *Model[C], foreignKey, localKey string) *relations.HasOne {
	return HasOne(parent.Ref(), related.Ref(), foreignKey, localKey)
}

// HasManyOf returns a has-many from parent to related.
func HasManyOf[P, C any](parent *Model[P], related *Model[C], foreignKey, localKey string) *relations.HasMany {
	return HasMany(parent.Ref(), related.Ref(), foreignKey, localKey)
}

// BelongsToModel returns a belongs-to from child to related.
//
// The relation name is what the error message says when the key is missing, so
// it is the name the method has: "user", not "userRelation".
func BelongsToModel[C, P any](child *Model[C], related *Model[P], foreignKey, ownerKey, relation string) *relations.BelongsTo {
	return BelongsTo(child.Ref(), related.Ref(), foreignKey, ownerKey, relation)
}

// BelongsToManyOf returns a many-to-many from parent to related.
//
// An empty table is the conventional intermediate name, which is the two table
// names in the singular, sorted, joined by an underscore.
func BelongsToManyOf[P, C any](parent *Model[P], related *Model[C], table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey, relation string) *relations.BelongsToMany {
	return BelongsToMany(parent.Ref(), related.Ref(), table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey, relation)
}

// MorphOneOf returns a morph-one from parent to related.
func MorphOneOf[P, C any](parent *Model[P], related *Model[C], name, typ, id, localKey string) *relations.MorphOne {
	return MorphOne(parent.Ref(), related.Ref(), name, typ, id, localKey)
}

// MorphManyOf returns a morph-many from parent to related.
func MorphManyOf[P, C any](parent *Model[P], related *Model[C], name, typ, id, localKey string) *relations.MorphMany {
	return MorphMany(parent.Ref(), related.Ref(), name, typ, id, localKey)
}

// MorphToOf returns a morph-to on parent.
//
// related is the model whose connection the relation starts from; the table it
// finally reads is resolved from the type column, through the morph map.
func MorphToOf[P, C any](parent *Model[P], related *Model[C], name, typ, id, ownerKey string) *relations.MorphTo {
	return MorphTo(parent.Ref(), related.Ref(), name, typ, id, ownerKey)
}

// MorphToManyOf returns a polymorphic many-to-many from parent to related.
func MorphToManyOf[P, C any](parent *Model[P], related *Model[C], name, table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey, relation string, inverse bool) *relations.MorphToMany {
	return MorphToMany(parent.Ref(), related.Ref(), name, table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey, relation, inverse)
}

// MorphedByManyOf returns the other side of a MorphToManyOf.
func MorphedByManyOf[P, C any](parent *Model[P], related *Model[C], name, table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey, relation string) *relations.MorphToMany {
	return MorphedByMany(parent.Ref(), related.Ref(), name, table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey, relation)
}

// HasManyThroughOf returns a has-many-through from farParent to related, by way
// of through.
func HasManyThroughOf[F, T2, C any](farParent *Model[F], through *Model[T2], related *Model[C], firstKey, secondKey, localKey, secondLocalKey string) *relations.HasManyThrough {
	return HasManyThrough(farParent.Ref(), through.Ref(), related.Ref(), firstKey, secondKey, localKey, secondLocalKey)
}

// HasOneThroughOf returns a has-one-through from farParent to related, by way of
// through.
func HasOneThroughOf[F, T2, C any](farParent *Model[F], through *Model[T2], related *Model[C], firstKey, secondKey, localKey, secondLocalKey string) *relations.HasOneThrough {
	return HasOneThrough(farParent.Ref(), through.Ref(), related.Ref(), firstKey, secondKey, localKey, secondLocalKey)
}

// The same constructors, not narrowed to one parent. See the note at the top of
// this file for which call site takes which.

// HasOneOfUnconstrained is HasOneOf without the narrowing to parent.
func HasOneOfUnconstrained[P, C any](parent *Model[P], related *Model[C], foreignKey, localKey string) *relations.HasOne {
	p, c := parent.Ref(), related.Ref()
	foreignKey, localKey = defaultHasKeys(p, foreignKey, localKey)
	return relations.NewHasOneUnconstrained(c.NewQuery(), p, c.QualifyColumn(foreignKey), localKey)
}

// HasManyOfUnconstrained is HasManyOf without the narrowing to parent.
func HasManyOfUnconstrained[P, C any](parent *Model[P], related *Model[C], foreignKey, localKey string) *relations.HasMany {
	p, c := parent.Ref(), related.Ref()
	foreignKey, localKey = defaultHasKeys(p, foreignKey, localKey)
	return relations.NewHasManyUnconstrained(c.NewQuery(), p, c.QualifyColumn(foreignKey), localKey)
}

// BelongsToModelUnconstrained is BelongsToModel without the narrowing to the
// row child points at.
func BelongsToModelUnconstrained[C, P any](child *Model[C], related *Model[P], foreignKey, ownerKey, relation string) *relations.BelongsTo {
	c, p := child.Ref(), related.Ref()
	foreignKey, ownerKey = defaultBelongsToKeys(p, foreignKey, ownerKey, relation)
	return relations.NewBelongsToUnconstrained(p.NewQuery(), c, foreignKey, ownerKey, relation)
}

// BelongsToManyOfUnconstrained is BelongsToManyOf without the narrowing to
// parent. The join onto the intermediate table stays: it is how the related
// table is reached at all, for one parent or for a hundred.
func BelongsToManyOfUnconstrained[P, C any](parent *Model[P], related *Model[C], table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey, relation string) *relations.BelongsToMany {
	p, c := parent.Ref(), related.Ref()
	table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey = defaultPivotKeys(p, c, table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey)
	return relations.NewBelongsToManyUnconstrained(c.NewQuery(), p, table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey, relation)
}

// MorphOneOfUnconstrained is MorphOneOf without the narrowing to parent.
func MorphOneOfUnconstrained[P, C any](parent *Model[P], related *Model[C], name, typ, id, localKey string) *relations.MorphOne {
	p, c := parent.Ref(), related.Ref()
	typ, id, localKey = defaultMorphKeys(p, name, typ, id, localKey)
	return relations.NewMorphOneUnconstrained(c.NewQuery(), p, c.QualifyColumn(typ), c.QualifyColumn(id), localKey)
}

// MorphManyOfUnconstrained is MorphManyOf without the narrowing to parent.
func MorphManyOfUnconstrained[P, C any](parent *Model[P], related *Model[C], name, typ, id, localKey string) *relations.MorphMany {
	p, c := parent.Ref(), related.Ref()
	typ, id, localKey = defaultMorphKeys(p, name, typ, id, localKey)
	return relations.NewMorphManyUnconstrained(c.NewQuery(), p, c.QualifyColumn(typ), c.QualifyColumn(id), localKey)
}

// MorphToOfUnconstrained is MorphToOf without the narrowing to parent.
func MorphToOfUnconstrained[P, C any](parent *Model[P], related *Model[C], name, typ, id, ownerKey string) *relations.MorphTo {
	p, c := parent.Ref(), related.Ref()
	typ, id = GetMorphs(str.Snake(name, "_"), typ, id)
	return relations.NewMorphToUnconstrained(c.NewQuery(), p, id, ownerKey, typ, name)
}

// MorphToManyOfUnconstrained is MorphToManyOf without the narrowing to parent.
func MorphToManyOfUnconstrained[P, C any](parent *Model[P], related *Model[C], name, table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey, relation string, inverse bool) *relations.MorphToMany {
	p, c := parent.Ref(), related.Ref()
	table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey = defaultMorphPivotKeys(p, c, name, table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey)
	return relations.NewMorphToManyUnconstrained(c.NewQuery(), p, name, table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey, relation, inverse)
}

// MorphedByManyOfUnconstrained is MorphedByManyOf without the narrowing to
// parent.
func MorphedByManyOfUnconstrained[P, C any](parent *Model[P], related *Model[C], name, table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey, relation string) *relations.MorphToMany {
	if foreignPivotKey == "" {
		foreignPivotKey = parent.Ref().GetForeignKey()
	}
	if relatedPivotKey == "" {
		relatedPivotKey = name + "_id"
	}
	return MorphToManyOfUnconstrained(parent, related, name, table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey, relation, true)
}

// HasManyThroughOfUnconstrained is HasManyThroughOf without the narrowing to
// farParent. The join onto the intermediate table stays, for the reason
// BelongsToManyOfUnconstrained keeps its own.
func HasManyThroughOfUnconstrained[F, T2, C any](farParent *Model[F], through *Model[T2], related *Model[C], firstKey, secondKey, localKey, secondLocalKey string) *relations.HasManyThrough {
	f, t, c := farParent.Ref(), through.Ref(), related.Ref()
	firstKey, secondKey, localKey, secondLocalKey = defaultThroughKeys(f, t, firstKey, secondKey, localKey, secondLocalKey)
	return relations.NewHasManyThroughUnconstrained(c.NewQuery(), f, t, firstKey, secondKey, localKey, secondLocalKey)
}

// HasOneThroughOfUnconstrained is HasOneThroughOf without the narrowing to
// farParent.
func HasOneThroughOfUnconstrained[F, T2, C any](farParent *Model[F], through *Model[T2], related *Model[C], firstKey, secondKey, localKey, secondLocalKey string) *relations.HasOneThrough {
	f, t, c := farParent.Ref(), through.Ref(), related.Ref()
	firstKey, secondKey, localKey, secondLocalKey = defaultThroughKeys(f, t, firstKey, secondKey, localKey, secondLocalKey)
	return relations.NewHasOneThroughUnconstrained(c.NewQuery(), f, t, firstKey, secondKey, localKey, secondLocalKey)
}
