package relations

import (
	"context"

	"github.com/arandu-io/hesape/auth"
)

// EagerLoadRelation loads one relation for a batch of parents.
//
// It is four lines and it is the whole of what eager loading buys:
//
//	relation.AddEagerConstraints(models)  // one `in (...)` for every parent
//	constraints(relation)                 // the caller's extra filters
//	relation.InitRelation(models, name)   // every parent starts non-nil
//	relation.Match(models, eager, name)   // distributed in memory
//
// It lives in this package rather than on the builder because the four methods
// it calls are the four this package exists to implement, and because a reader
// asking "what does eager loading actually do" should find the answer next to
// them. The typed builder calls it; it is not a second way to load a
// relation.
//
// # The count, which is the only measurement that matters
//
// For a hundred parents, this is 1 query. Reading the relation off each parent
// instead is 100. That is the whole subject, and it is why the test for this
// counts the statements the connection was asked for rather than checking that
// the values came out right -- values come out right either way, which is
// exactly why an N+1 survives code review.
//
// MorphTo is the exception and it is inherent: its parents point at several
// tables, so it is one query per distinct type in the batch. Its GetEager does
// its own matching and Match answers the parents unchanged, which is why the
// order below is GetEager after InitRelation rather than before.
func EagerLoadRelation(ctx context.Context, g auth.Grant, models []Model, name string, relation Relation, constraints func(Relation)) ([]Model, error) {
	if len(models) == 0 {
		return models, nil
	}

	if err := relation.AddEagerConstraints(models); err != nil {
		return nil, err
	}

	if constraints != nil {
		constraints(relation)
	}

	models = relation.InitRelation(models, name)

	results, err := relation.GetEager(ctx, g)
	if err != nil {
		return nil, err
	}

	return relation.Match(models, results, name)
}
