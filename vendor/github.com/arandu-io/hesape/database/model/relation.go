package model

import (
	"fmt"

	"github.com/arandu-io/hesape/database/model/relations"
	"github.com/arandu-io/hesape/database/query"
)

// Relation is what the builder and the eager loader ask of a relation.
//
// It is the contract the relations tree implements, plus the one method the
// builder needs that the tree does not declare: every relation there has
// GetRelationExistenceQuery, but as a method on the shared body rather than as
// part of its interface, and Has cannot call a method the type it holds does not
// promise.
//
// # It is declared here and satisfied there
//
// This used to be a smaller interface of this package's own -- a Match that took
// the Grant and a batch of keys, and returned a dictionary the builder assigned
// from. It was written against a stand-in while this slice was built, and
// nothing in model/relations ever satisfied it: relations declare
// Match(models, results, relation), so the two collided on the name and no
// relation could be registered at all.
//
// The shape that went is the one that could not have worked. Keys are not
// enough to eager load: a has-many on a local key that is not the primary key
// needs that column's value and not the row's id, and a morph-to is keyed by a
// type resolved per row and does its own matching. A dictionary of key to value
// also has nothing to say for the parent that matched nothing, where the tree
// seeds every parent with an empty collection first -- so a childless parent
// read back as a relation that was never loaded.
//
// A relation is registered on the model by name, in RelationResolvers, because
// Go cannot look up a method by name and stay type safe.
type Relation interface {
	relations.Relation

	// GetRelationExistenceQuery returns the correlated subquery over the
	// related table, selecting the given expression and constrained to the
	// parent row.
	//
	// It is what Has, WhereHas and WithCount compile into an exists() or a
	// scalar subselect. q is the query it narrows, which is a fresh one on the
	// related model rather than the relation's own; parentQuery is what it
	// correlates against, and is read to tell a relation pointing back at its
	// own table apart from one pointing elsewhere.
	GetRelationExistenceQuery(q relations.Builder, parentQuery relations.Builder, columns ...any) relations.Builder
}

// Every relation can be registered, and the compiler checks it here rather than
// at the first application that tries.
//
// The relations tree makes the same assertion against its own contract, and this
// is the other half: that contract plus what the builder adds. The two drifted
// apart once, and the whole surface -- With, Load, Has, WhereHas, WithCount --
// was unreachable while every test in both packages passed.
var (
	_ Relation = (*relations.HasOne)(nil)
	_ Relation = (*relations.HasMany)(nil)
	_ Relation = (*relations.BelongsTo)(nil)
	_ Relation = (*relations.BelongsToMany)(nil)
	_ Relation = (*relations.MorphOne)(nil)
	_ Relation = (*relations.MorphMany)(nil)
	_ Relation = (*relations.MorphTo)(nil)
	_ Relation = (*relations.MorphToMany)(nil)
	_ Relation = (*relations.HasOneThrough)(nil)
	_ Relation = (*relations.HasManyThrough)(nil)

	_ BelongsToRelation = (*relations.BelongsTo)(nil)
)

// BelongsToRelation is the part of a relation that WhereBelongsTo needs: the
// column on this table, and the column on the other one it points at.
type BelongsToRelation interface {
	Relation

	// GetQualifiedForeignKeyName returns the foreign key column on this
	// table, qualified with the table name.
	GetQualifiedForeignKeyName() string

	// GetOwnerKeyName returns the column on the related table the foreign
	// key points at.
	GetOwnerKeyName() string
}

// MorphRelation is the polymorphic relation WhereMorphRelation walks.
type MorphRelation interface {
	Relation

	// GetMorphType returns the column holding the type of the related row.
	GetMorphType() string

	// RelationForMorphType returns the concrete relation for one of the
	// types a polymorphic relation can point at.
	//
	// Go cannot make a type from a string, so the relation resolves its
	// own map of type to model instead of a caller building one
	// generically.
	RelationForMorphType(morphType string) (Relation, error)
}

// GetRelationWithoutConstraints resolves relation by calling its registered
// resolver.
//
// What comes back must not be narrowed to one parent, and it is the resolver
// that promises so: this is called with the model the builder queries through,
// which for a list query is a prototype carrying no key at all. The
// Unconstrained constructors in relationsof.go are what a resolver registers,
// and the note at the top of that file is why.
func (b *Builder[T]) GetRelationWithoutConstraints(name string) (Relation, error) {
	resolver, ok := b.model.RelationResolvers[name]
	if !ok {
		return nil, fmt.Errorf("%w: %s on %s", ErrRelationNotFound, name, b.model.GetTable())
	}
	return resolver(b.model), nil
}

// existenceSubquery is the subquery Has, WhereHas and WithCount hang off: the
// relation's own, over a fresh query on the related model, correlated to parent.
//
// The query it narrows is fresh rather than the relation's own, which is what
// lets the same relation answer both an existence check and a read: the wheres
// that narrow it to one parent are on its query, and an exists() correlates by
// column instead.
func existenceSubquery(rel Relation, parent relations.Builder, column any) relations.Builder {
	return rel.GetRelationExistenceQuery(rel.GetRelated().NewQuery(), parent, column)
}

// onBaseQuery adapts a caller's constraint callback to the builder a relation
// hands back.
//
// Has and WhereHas take func(*query.Builder), because a caller narrowing an
// exists is writing wheres and nothing else. The relation answers the wider
// builder, and this is the one line between them.
func onBaseQuery(callback func(*query.Builder)) func(relations.Builder) {
	if callback == nil {
		return nil
	}
	return func(sub relations.Builder) { callback(sub.GetQuery()) }
}
