package model

import (
	"context"
	"fmt"
	"strings"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/database/model/relations"
	"github.com/arandu-io/hesape/database/query"
	"github.com/arandu-io/hesape/str"
)

// The methods that filter a query by what its relations contain.
//
// Every one of them builds SQL and runs nothing, so none takes a Grant.
// A relation whose name is not registered on the model is an error the builder
// holds until something runs -- see Builder.err.

// Has adds a filter on the count of relation matching operator and count:
// relation exists (">=" 1), does not exist ("<" 1), or any other
// comparison.
//
// Go has no default arguments, so the full form is spelled out here and the
// short ones -- WhereHas, DoesntHave, OrHas -- are the ones to reach for.
// callback may be nil.
func (b *Builder[T]) Has(relation, operator string, count int, boolean string, callback func(*query.Builder)) *Builder[T] {
	if strings.Contains(relation, ".") {
		return b.hasNested(relation, operator, count, boolean, callback)
	}

	rel, err := b.GetRelationWithoutConstraints(relation)
	if err != nil {
		return b.fail(err)
	}
	return b.addHasWhere(rel, operator, count, boolean, onBaseQuery(callback))
}

// addHasWhere adds the where clause for a Has-style filter on rel.
//
// A ">= 1" or a "< 1" is an exists rather than a count, which is the
// optimisation canUseExistsForExistenceCheck exists for: the engine can stop
// at the first matching row instead of counting every one.
//
// The callback takes the builder a relation hands back rather than the base
// query, which is what hasNested needs to go one segment deeper. The public
// methods take func(*query.Builder) and onBaseQuery is the step between.
func (b *Builder[T]) addHasWhere(rel Relation, operator string, count int, boolean string, callback func(relations.Builder)) *Builder[T] {
	if boolean == "" {
		boolean = "and"
	}
	if canUseExistsForExistenceCheck(operator, count) {
		sub := existenceSubquery(rel, b.Ref(), query.Raw("*"))
		if callback != nil {
			callback(sub)
		}
		b.query.AddWhereExistsQuery(sub.GetQuery(), boolean, operator == "<" && count == 1)
		return b
	}

	sub := existenceSubquery(rel, b.Ref(), query.Raw("count(*)"))
	if callback != nil {
		callback(sub)
	}
	return b.addWhereCountQuery(sub.GetQuery(), operator, count, boolean)
}

// addWhereCountQuery adds the subquery compared against a number, as an
// expression on the left of the operator.
//
// It used to assemble that clause here, freezing sub.ToSQL() into the
// column of a Basic where. The subquery was gone by the time anything could
// scope it, and nothing did: Users.Has("posts", ">", 3).Get(auth.SystemGrant(
// "user.list", "acme")) ran `(select count(*) from "posts" where
// "users"."id" = "posts"."user_id") > 3` with no posts.tenant_id in it, so
// one tenant's users were selected by counting every tenant's posts.
//
// query.WhereSubCount keeps the builder on the clause, and prepare scopes it
// through query.Builder.ScopeNested. That is the same door the query
// builder's own statements go through, and it is the only one.
func (b *Builder[T]) addWhereCountQuery(sub *query.Builder, operator string, count int, boolean string) *Builder[T] {
	b.query.WhereSubCount(sub, operator, count, boolean)
	return b
}

// canUseExistsForExistenceCheck reports whether operator and count can use
// an exists check instead of a count.
func canUseExistsForExistenceCheck(operator string, count int) bool {
	return (operator == ">=" || operator == "<") && count == 1
}

// hasNested adds a Has-style filter across a dotted relation path, such as
// "comments.author".
//
// Go cannot hold "a builder of some other model" in a variable, so the
// recursion happens on the relation chain instead: each segment asks the
// one before it for the next, and the existence queries nest the same way.
//
// A relation that cannot resolve the next segment is an error, never a
// query with the filter silently missing.
func (b *Builder[T]) hasNested(path, operator string, count int, boolean string, callback func(*query.Builder)) *Builder[T] {
	segments := strings.Split(path, ".")

	rel, err := b.GetRelationWithoutConstraints(segments[0])
	if err != nil {
		return b.fail(err)
	}

	chain := []Relation{rel}
	for _, segment := range segments[1:] {
		nested, ok := chain[len(chain)-1].(NestedRelation)
		if !ok {
			return b.fail(fmt.Errorf("%w: %s does not reach %s", ErrRelationNotFound, path, segment))
		}
		next, err := nested.Nested(segment)
		if err != nil {
			return b.fail(err)
		}
		chain = append(chain, next)
	}

	// The innermost relation carries the caller's constraints; every outer one
	// only has to exist.
	return b.addHasWhere(chain[0], operator, count, boolean, func(sub relations.Builder) {
		nest(sub, chain[1:], callback)
	})
}

func nest(parent relations.Builder, chain []Relation, callback func(*query.Builder)) {
	if len(chain) == 0 {
		if callback != nil {
			callback(parent.GetQuery())
		}
		return
	}
	sub := existenceSubquery(chain[0], parent, query.Raw("*"))
	nest(sub, chain[1:], callback)
	parent.GetQuery().AddWhereExistsQuery(sub.GetQuery(), "and", false)
}

// NestedRelation is the relation a dotted existence check walks through. See
// hasNested.
type NestedRelation interface {
	Relation

	// Nested returns the relation the next segment of a dotted path names,
	// on the related model.
	Nested(name string) (Relation, error)
}

// OrHas is Has joined with or.
func (b *Builder[T]) OrHas(relation, operator string, count int) *Builder[T] {
	return b.Has(relation, operator, count, "or", nil)
}

// DoesntHave adds a filter requiring relation to not exist.
func (b *Builder[T]) DoesntHave(relation, boolean string, callback func(*query.Builder)) *Builder[T] {
	return b.Has(relation, "<", 1, boolean, callback)
}

// OrDoesntHave is DoesntHave joined with or.
func (b *Builder[T]) OrDoesntHave(relation string) *Builder[T] {
	return b.DoesntHave(relation, "or", nil)
}

// WhereHas adds a filter requiring relation to exist, constrained by
// callback.
func (b *Builder[T]) WhereHas(relation string, callback func(*query.Builder)) *Builder[T] {
	return b.Has(relation, ">=", 1, "and", callback)
}

// WhereHasCount is WhereHas with an explicit operator and count instead of
// ">=" and 1.
func (b *Builder[T]) WhereHasCount(relation string, callback func(*query.Builder), operator string, count int) *Builder[T] {
	return b.Has(relation, operator, count, "and", callback)
}

// OrWhereHas is WhereHas joined with or.
func (b *Builder[T]) OrWhereHas(relation string, callback func(*query.Builder)) *Builder[T] {
	return b.Has(relation, ">=", 1, "or", callback)
}

// WhereDoesntHave adds a filter requiring relation to not exist, constrained
// by callback.
func (b *Builder[T]) WhereDoesntHave(relation string, callback func(*query.Builder)) *Builder[T] {
	return b.DoesntHave(relation, "and", callback)
}

// OrWhereDoesntHave is WhereDoesntHave joined with or.
func (b *Builder[T]) OrWhereDoesntHave(relation string, callback func(*query.Builder)) *Builder[T] {
	return b.DoesntHave(relation, "or", callback)
}

// WhereRelation adds a filter requiring relation to have a row matching
// column args.
func (b *Builder[T]) WhereRelation(relation string, column any, args ...any) *Builder[T] {
	return b.WhereHas(relation, func(sub *query.Builder) {
		sub.Where(column, args...)
	})
}

// OrWhereRelation is WhereRelation joined with or.
func (b *Builder[T]) OrWhereRelation(relation string, column any, args ...any) *Builder[T] {
	return b.OrWhereHas(relation, func(sub *query.Builder) {
		sub.Where(column, args...)
	})
}

// WhereDoesntHaveRelation adds a filter requiring relation to have no row
// matching column args.
func (b *Builder[T]) WhereDoesntHaveRelation(relation string, column any, args ...any) *Builder[T] {
	return b.WhereDoesntHave(relation, func(sub *query.Builder) {
		sub.Where(column, args...)
	})
}

// WhereMorphRelation adds a filter requiring relation, constrained to any of
// types, to have a row matching column args.
//
// Go cannot make a type from a string, so the relation resolves its own
// types through RelationForMorphType, one branch per type, ored together.
//
// A single type of "*" is refused rather than resolved: resolving it means
// selecting the distinct morph column first, which is a query, and this
// method builds SQL and runs nothing.
func (b *Builder[T]) WhereMorphRelation(relation string, types []string, column any, args ...any) *Builder[T] {
	rel, err := b.GetRelationWithoutConstraints(relation)
	if err != nil {
		return b.fail(err)
	}
	morph, ok := rel.(MorphRelation)
	if !ok {
		return b.fail(fmt.Errorf("%w: %s is not a polymorphic relation", ErrRelationNotFound, relation))
	}
	if len(types) == 0 || (len(types) == 1 && types[0] == "*") {
		return b.fail(fmt.Errorf("model: %s needs the morph types spelled out, because resolving \"*\" is a query and this builds SQL", relation))
	}

	return b.Where(func(outer *Builder[T]) {
		for _, morphType := range types {
			typed, err := morph.RelationForMorphType(morphType)
			if err != nil {
				// The failure is recorded on the outer builder and not on the
				// nested one, which is thrown away as soon as its wheres are
				// merged -- an error left on it would never be reported.
				b.fail(err)
				return
			}
			outer.OrWhere(func(branch *Builder[T]) {
				branch.Where(b.model.QualifyColumn(morph.GetMorphType()), "=", morphType)
				branch.addHasWhere(typed, ">=", 1, "and", onBaseQuery(func(sub *query.Builder) {
					sub.Where(column, args...)
				}))
			})
		}
	})
}

// WhereBelongsTo adds a filter requiring the named belongs-to relation to
// point at one of related.
//
// It is a function and not a method because the related models are of
// another type and a Go method cannot introduce a type parameter.
func WhereBelongsTo[T, R any](b *Builder[T], relationshipName string, related ...*Model[R]) *Builder[T] {
	if len(related) == 0 {
		return b.fail(fmt.Errorf("model: WhereBelongsTo was given no models to belong to"))
	}
	rel, err := b.GetRelationWithoutConstraints(relationshipName)
	if err != nil {
		return b.fail(err)
	}
	belongsTo, ok := rel.(BelongsToRelation)
	if !ok {
		return b.fail(fmt.Errorf("%w: %s is not a belongs-to relation", ErrRelationNotFound, relationshipName))
	}

	keys := make([]any, 0, len(related))
	for _, model := range related {
		keys = append(keys, model.GetAttribute(belongsTo.GetOwnerKeyName()))
	}
	b.query.WhereIn(belongsTo.GetQualifiedForeignKeyName(), keys)
	return b
}

// WithAggregate adds a subselect per relation, aliased onto the row.
//
// A name may carry an alias -- "posts as recent_posts".
//
// The subselect goes in through query.SelectSub, and the exists form
// through query.SelectExistsSub, which are the methods that record a
// subquery so the tenant can be put on it when the Grant arrives. This used
// to write the column itself, with AddSelect and SelectRaw over
// sub.ToSQL(), and a subquery compiled into a raw column is a subquery
// nothing can scope: Users.WithCount("posts").Get(auth.SystemGrant(
// "user.list", "acme")) gave every row the number of posts EVERY tenant
// had, and WithSum("orders", "total") handed one tenant another tenant's
// revenue as a scalar.
func (b *Builder[T]) WithAggregate(relations []string, column, function string) *Builder[T] {
	if len(relations) == 0 {
		return b
	}
	if b.query.Columns == nil {
		b.query.Select(query.Raw(b.model.Grammar.WrapTable(b.model.GetTable()) + ".*"))
	}

	for _, entry := range relations {
		name, alias := splitAggregateAlias(entry)

		rel, err := b.GetRelationWithoutConstraints(name)
		if err != nil {
			return b.fail(err)
		}

		var expression string
		switch {
		case function == "":
			expression = column
		case function == "exists":
			expression = b.aggregateColumn(column)
		default:
			expression = fmt.Sprintf("%s(%s)", function, b.aggregateColumn(column))
		}

		sub := existenceSubquery(rel, b.Ref(), query.Raw(expression)).GetQuery()
		if constraints, ok := b.eagerLoad[name]; ok && constraints != nil {
			// The eager-load constraints narrow the subquery, not the query the
			// subselect hangs off -- which is what callScope does to the relation's
			// own builder there.
			constraints(sub)
		}

		if alias == "" {
			alias = aggregateAlias(name, function, column)
		}

		if function == "exists" {
			b.query.SelectExistsSub(sub, alias)
			continue
		}
		if function == "" {
			sub.Limit(1)
		}
		b.query.SelectSub(sub, alias)
	}
	return b
}

// aggregateColumn wraps the column an aggregate is taken over, leaving "*"
// alone.
func (b *Builder[T]) aggregateColumn(column string) string {
	if column == "*" {
		return column
	}
	return b.model.Grammar.Wrap(column)
}

// splitAggregateAlias reads "posts as recent" into its two halves.
func splitAggregateAlias(entry string) (name, alias string) {
	segments := strings.Fields(entry)
	if len(segments) == 3 && strings.EqualFold(segments[1], "as") {
		return segments[0], segments[2]
	}
	return entry, ""
}

// aggregateAlias returns the alias WithAggregate builds when none was
// given: the relation, the function and the column, snake cased, with the
// punctuation dropped. "posts", "count", "*" is posts_count.
func aggregateAlias(name, function, column string) string {
	raw := fmt.Sprintf("%s %s %s", name, function, strings.ToLower(column))
	var kept strings.Builder
	for _, r := range raw {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == ' ', r == '_':
			kept.WriteRune(r)
		}
	}
	return str.Snake(strings.TrimSpace(kept.String()), "_")
}

// WithCount adds the row count of each relation, aliased onto the row.
func (b *Builder[T]) WithCount(relations ...string) *Builder[T] {
	return b.WithAggregate(relations, "*", "count")
}

// WithMax adds the max of column over relation, aliased onto the row.
func (b *Builder[T]) WithMax(relation, column string) *Builder[T] {
	return b.WithAggregate([]string{relation}, column, "max")
}

// WithMin adds the min of column over relation, aliased onto the row.
func (b *Builder[T]) WithMin(relation, column string) *Builder[T] {
	return b.WithAggregate([]string{relation}, column, "min")
}

// WithSum adds the sum of column over relation, aliased onto the row.
func (b *Builder[T]) WithSum(relation, column string) *Builder[T] {
	return b.WithAggregate([]string{relation}, column, "sum")
}

// WithAvg adds the average of column over relation, aliased onto the row.
func (b *Builder[T]) WithAvg(relation, column string) *Builder[T] {
	return b.WithAggregate([]string{relation}, column, "avg")
}

// WithExists adds whether relation exists, aliased onto the row.
func (b *Builder[T]) WithExists(relation string) *Builder[T] {
	return b.WithAggregate([]string{relation}, "*", "exists")
}

// loadAggregateModels is the query LoadAggregate runs: the keys of the
// collection, with the aggregate columns beside them.
func (b *Builder[T]) loadAggregateModels(ctx context.Context, g auth.Grant, keys []any, relations []string, column, function string) (models[T], error) {
	return b.WhereKey(keys).
		Select(b.model.GetQualifiedKeyName()).
		WithAggregate(relations, column, function).
		get(ctx, g)
}
