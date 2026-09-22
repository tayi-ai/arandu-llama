package concerns

import (
	"fmt"
	"strings"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/database/query"
)

// AsPivot answers the trait of the same name: what a model gains by being the
// row of an intermediate table.
//
// The two static constructors the PHP trait carries -- fromAttributes and
// fromRawAttributes -- are not here. They call `new static`, which is late
// static binding, and the type they have to build is the concrete pivot; they
// live on relations.Pivot and relations.MorphPivot, where the type is known.
//
// A pivot row is not the same shape as an ordinary model, and the trait exists
// for that difference: the table has no id of its own, so a save cannot be keyed
// by one. setKeysForSaveQuery keys it by the two foreign keys instead, which is
// why they are held here rather than guessed at write time.
type AsPivot struct {
	// PivotParent is AsPivot::$pivotParent: the model the relation was declared
	// on. It answers where the timestamp column names come from.
	PivotParent Model

	// PivotRelated is AsPivot::$pivotRelated.
	PivotRelated Model

	foreignKey string
	relatedKey string
}

// SetPivotKeys answers AsPivot::setPivotKeys.
func (p *AsPivot) SetPivotKeys(foreignKey, relatedKey string) *AsPivot {
	p.foreignKey = foreignKey
	p.relatedKey = relatedKey
	return p
}

// SetRelatedModel answers AsPivot::setRelatedModel.
func (p *AsPivot) SetRelatedModel(related Model) *AsPivot {
	p.PivotRelated = related
	return p
}

// GetForeignKey answers AsPivot::getForeignKey.
func (p *AsPivot) GetForeignKey() string { return p.foreignKey }

// GetRelatedKey answers AsPivot::getRelatedKey.
func (p *AsPivot) GetRelatedKey() string { return p.relatedKey }

// GetOtherKey answers AsPivot::getOtherKey, the alias of GetRelatedKey.
func (p *AsPivot) GetOtherKey() string { return p.GetRelatedKey() }

// HasTimestampAttributes answers AsPivot::hasTimestampAttributes.
//
// A pivot table carries timestamps only when the developer said withTimestamps,
// so the question is answered by looking for the column in the row rather than
// by a flag on the class.
func (p *AsPivot) HasTimestampAttributes(attributes map[string]any) bool {
	created := p.GetCreatedAtColumn()
	if created == "" {
		return false
	}
	_, ok := attributes[created]
	return ok
}

// GetCreatedAtColumn answers AsPivot::getCreatedAtColumn: the parent's, because
// the pivot has no configuration of its own.
func (p *AsPivot) GetCreatedAtColumn() string {
	if p.PivotParent != nil {
		return p.PivotParent.GetCreatedAtColumn()
	}
	return "created_at"
}

// GetUpdatedAtColumn answers AsPivot::getUpdatedAtColumn.
func (p *AsPivot) GetUpdatedAtColumn() string {
	if p.PivotParent != nil {
		return p.PivotParent.GetUpdatedAtColumn()
	}
	return "updated_at"
}

// SetKeysForSelectQuery answers AsPivot::setKeysForSelectQuery.
//
// It is what makes a pivot row addressable without an id: the pair of foreign
// keys is the key. The values come from the original attributes rather than the
// current ones, so changing which role a row points at still updates the row it
// came from instead of a row that does not exist yet.
func (p *AsPivot) SetKeysForSelectQuery(query Builder, model Model) Builder {
	if _, ok := model.GetAttributes()[model.GetKeyName()]; ok {
		return query.Where(model.GetKeyName(), model.GetKey())
	}

	return query.
		Where(p.foreignKey, model.GetAttribute(p.foreignKey)).
		Where(p.relatedKey, model.GetAttribute(p.relatedKey))
}

// GetQueueableID answers what a queued job stores so that it can find this row
// again.
//
// A pivot row usually has no id column, so there is nothing to store; the pair
// of foreign keys is written out instead, in the "foreign:value:related:value"
// shape. The row is passed in because this struct holds no attributes -- the
// same reason SetKeysForSelectQuery takes it.
func (p *AsPivot) GetQueueableID(model Model) any {
	if _, ok := model.GetAttributes()[model.GetKeyName()]; ok {
		return model.GetKey()
	}
	return fmt.Sprintf("%s:%v:%s:%v",
		p.foreignKey, model.GetAttribute(p.foreignKey),
		p.relatedKey, model.GetAttribute(p.relatedKey))
}

// SetKeysForSaveQuery answers AsPivot::setKeysForSaveQuery.
func (p *AsPivot) SetKeysForSaveQuery(query Builder, model Model) Builder {
	return p.SetKeysForSelectQuery(query, model)
}

// NewQueryForRestoration is the query that finds again the rows GetQueueableID
// wrote down.
//
// It is the exact inverse of that method, and the two are worth reading
// together. A pivot row with an id column serializes to the id and comes back
// through WhereKey. One without serializes to "foreign:value:related:value", and
// comes back by splitting that string into the pair of clauses it was made from.
//
// Several ids collapse into one query with a group per id; the branch is three
// lines, so it is folded in here rather than given a method only this one could
// reach.
//
// The identifier parameter is variadic, and a malformed identifier returns an
// error rather than building a query keyed on nothing -- which restores nothing,
// or worse, the wrong row.
//
// The model is passed for the same reason SetKeysForSelectQuery takes it: this
// concern holds no attributes and no query of its own.
//
// # What leaked
//
// It took no Grant, and a query built without one is a query with no tenant on
// it. An audit proved it with the pair of clauses this method writes: a job that
// serialized "user_id:1:role_id:admin" restored it with `select * from role_user
// where user_id = 1 and role_id = 'admin'`, over a table every customer shares,
// and the row that came back was whichever the database returned first. Save and
// Delete on the same pivot were scoped; only the way back from a queue was not.
func (p *AsPivot) NewQueryForRestoration(model Model, g auth.Grant, ids ...any) (Builder, error) {
	return p.newQueryForRestoration(model, g, ids, 2)
}

// newQueryForRestoration is AsPivot's and MorphPivot's shared body. The pairs
// argument is how many column:value pairs the composite identifier carries --
// two for a pivot, three for a morph pivot, whose third pair is the type.
func (p *AsPivot) newQueryForRestoration(model Model, g auth.Grant, ids []any, pairs int) (Builder, error) {
	if len(ids) == 0 {
		return nil, fmt.Errorf("pivot restoration was given no identifier: the queued job stored nothing to find the row on %s by", model.GetTable())
	}

	unscoped := model.NewQuery()
	if unscoped == nil {
		return nil, fmt.Errorf("pivot restoration cannot query %s: the pivot was built without a query", model.GetTable())
	}

	q, err := ScopeTenant(unscoped, model, g)
	if err != nil {
		return nil, err
	}

	// A plain id is the ordinary Model::newQueryForRestoration, and the PHP
	// hands the whole array to it rather than looping -- whereKey takes both.
	if !isCompositeID(ids[0]) {
		return q.WhereKey(ids...), nil
	}

	if len(ids) == 1 {
		segments, err := splitCompositeID(ids[0], pairs)
		if err != nil {
			return nil, err
		}
		for i := 0; i < len(segments); i += 2 {
			q = q.Where(segments[i], segments[i+1])
		}
		return q, nil
	}

	// Every id becomes its own parenthesized group, or the clauses of one row
	// would filter the rows of the next and the query would match nothing.
	base := q.GetQuery()
	for _, id := range ids {
		segments, err := splitCompositeID(id, pairs)
		if err != nil {
			return nil, err
		}
		base.OrWhere(func(nested *query.Builder) {
			for i := 0; i < len(segments); i += 2 {
				nested.Where(segments[i], segments[i+1])
			}
		})
	}
	return q, nil
}

// isCompositeID answers the PHP's str_contains($ids, ':').
func isCompositeID(id any) bool {
	text, ok := id.(string)
	return ok && strings.Contains(text, ":")
}

// splitCompositeID answers the PHP's explode(':', $id), checking the count the
// PHP indexes into blindly.
func splitCompositeID(id any, pairs int) ([]string, error) {
	text, ok := id.(string)
	if !ok {
		return nil, fmt.Errorf("pivot restoration expected a composite identifier and got %T: the identifiers of one restore have to be all keys or all pairs", id)
	}
	segments := strings.Split(text, ":")
	if len(segments) != pairs*2 {
		return nil, fmt.Errorf("pivot restoration identifier %q has %d segments, expected %d: it was not written by GetQueueableID", text, len(segments), pairs*2)
	}
	return segments, nil
}

// UnsetRelations answers AsPivot::unsetRelations. The parent and the related
// model are dropped with the relations, because a pivot holds them as ordinary
// references and a serialized pivot would otherwise drag both models with it.
func (p *AsPivot) UnsetRelations() {
	p.PivotParent = nil
	p.PivotRelated = nil
}
