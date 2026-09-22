package concerns

import (
	"context"

	"github.com/arandu-io/hesape/auth"
)

// ComparesRelatedModels answers the trait of the same name.
//
// The PHP trait declares two abstract methods -- getParentKey and
// getRelatedKeyFrom -- and expects the using class to supply them. Go has no
// abstract method on an embedded struct, so the two arrive as function fields
// the relation sets when it builds itself. That is the mechanical shape of
// every trait in this package that PHP completes by inheritance.
type ComparesRelatedModels struct {
	// CompareRelated is the relation's related model, whose table decides half the
	// answer.
	CompareRelated Model

	// CompareParentKey answers the trait's abstract getParentKey.
	CompareParentKey func() any

	// CompareRelatedKeyFrom answers the trait's abstract getRelatedKeyFrom.
	CompareRelatedKeyFrom func(Model) any

	// CompareOneOfMany is the SupportsPartialRelations branch: on a one-of-many
	// relation the keys matching is not enough, because the relation is one row
	// of many and the question is whether this is that row. It is nil on a
	// relation that cannot be partial, and returns whether model is the one.
	CompareOneOfMany func(ctx context.Context, g auth.Grant, model Model) (bool, error)
}

// Is answers ComparesRelatedModels::is.
//
// It takes a context and a Grant because on a one-of-many relation it runs a
// query -- `$this->query->whereKey($model->getKey())->exists()` in the PHP --
// and everything in this collection that reaches the database carries the Grant
// that authorized it.
func (c ComparesRelatedModels) Is(ctx context.Context, g auth.Grant, model Model) (bool, error) {
	if model == nil || c.CompareRelated == nil {
		return false, nil
	}

	match := CompareKeys(c.CompareParentKey(), c.CompareRelatedKeyFrom(model)) &&
		c.CompareRelated.GetTable() == model.GetTable()

	if match && c.CompareOneOfMany != nil {
		return c.CompareOneOfMany(ctx, g, model)
	}

	return match, nil
}

// IsNot answers ComparesRelatedModels::isNot.
func (c ComparesRelatedModels) IsNot(ctx context.Context, g auth.Grant, model Model) (bool, error) {
	is, err := c.Is(ctx, g, model)
	return !is, err
}

// CompareKeys answers ComparesRelatedModels::compareKeys.
//
// Two empty keys are not a match, which is the whole point of the first branch:
// an unsaved model whose key is nil would otherwise equal every other unsaved
// model. The PHP's second branch compares as integers when either side is one;
// here the two are rendered through the dictionary key, which is the same
// coercion PHP performs on an array subscript.
func CompareKeys(parentKey, relatedKey any) bool {
	if isEmptyKey(parentKey) || isEmptyKey(relatedKey) {
		return false
	}

	left, err := GetDictionaryKey(parentKey)
	if err != nil {
		return false
	}
	right, err := GetDictionaryKey(relatedKey)
	if err != nil {
		return false
	}
	return left == right
}

// isEmptyKey answers PHP's empty() for the values a key can hold: null, the
// empty string, and zero.
func isEmptyKey(key any) bool {
	switch value := key.(type) {
	case nil:
		return true
	case string:
		return value == ""
	case int:
		return value == 0
	case int64:
		return value == 0
	case uint64:
		return value == 0
	case float64:
		return value == 0
	}
	return false
}
