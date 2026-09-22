package query

// The having vocabulary of the query builder.
//
// A having filters the groups a group by produced, so it is compiled after them
// and its bindings live in their own segment. The tenant filter never lands
// here: it is a where, because a row that belongs to another customer must not
// reach the grouping in the first place.

// addHaving adds a having clause, with the conjunction as a parameter rather
// than a trailing argument.
//
// A func opens a nested group. A bitwise operator marks the clause "Bitwise",
// a name the grammar's compileHaving switch does not match -- it lists "bit"
// -- so a bitwise having compiles as an ordinary basic one. The mismatch is
// deliberate, not a bug.
func (b *Builder) addHaving(boolean string, column any, args ...any) *Builder {
	if nested, ok := column.(func(*Builder)); ok {
		return b.HavingNested(nested, boolean)
	}

	operator, value := prepareValueAndOperator(args...)
	operator = b.acceptOperator(operator)
	if b.Err() != nil {
		return b
	}

	typ := "Basic"
	if b.isBitwiseOperator(operator) {
		typ = "Bitwise"
	}

	b.Havings = append(b.Havings, Having{
		Type: typ, Column: column, Operator: operator, Value: value, Boolean: boolean,
	})
	if !IsExpression(value) {
		b.AddBinding([]any{flattenValue(value)}, "having")
	}
	return b
}

// isBitwiseOperator reports whether operator is one of the bitwise comparison
// operators, either the package's default set or the grammar's own.
func (b *Builder) isBitwiseOperator(operator string) bool {
	if containsFold(bitwiseOperators, operator) {
		return true
	}
	if b.Grammar == nil {
		return false
	}
	return containsFold(b.Grammar.GetBitwiseOperators(), operator)
}

// OrHaving is addHaving joined with "or".
func (b *Builder) OrHaving(column any, args ...any) *Builder {
	return b.addHaving("or", column, args...)
}

// HavingNested runs callback against a fresh nested query and folds its
// havings in as one parenthesised group.
func (b *Builder) HavingNested(callback func(*Builder), boolean string) *Builder {
	nested := b.ForNestedWhere()
	callback(nested)
	return b.AddNestedHavingQuery(nested, boolean)
}

// AddNestedHavingQuery folds query's havings into b as one parenthesised
// group.
//
// A group with no clauses is dropped rather than compiled, for the reason
// AddNestedWhereQuery gives: "()" is a syntax error on every engine, and an
// empty callback is an ordinary outcome of a conditional filter.
func (b *Builder) AddNestedHavingQuery(query *Builder, boolean string) *Builder {
	if query != nil && query.Err() != nil {
		b.setError(query.Err())
	}
	if query == nil || len(query.Havings) == 0 {
		return b
	}
	if boolean == "" {
		boolean = "and"
	}
	b.Havings = append(b.Havings, Having{Type: "Nested", Query: query, Boolean: boolean})
	b.AddBinding(query.GetRawBindings()["having"], "having")
	return b
}

// HavingNull adds a having that requires columns to be null, or not null when
// not is true.
func (b *Builder) HavingNull(columns []any, boolean string, not bool) *Builder {
	if boolean == "" {
		boolean = "and"
	}
	typ := "Null"
	if not {
		typ = "NotNull"
	}
	for _, column := range columns {
		b.Havings = append(b.Havings, Having{Type: typ, Column: column, Boolean: boolean})
	}
	return b
}

// OrHavingNull is HavingNull joined with "or".
func (b *Builder) OrHavingNull(columns ...any) *Builder {
	return b.HavingNull(columns, "or", false)
}

// HavingNotNull is HavingNull with not set to true.
func (b *Builder) HavingNotNull(columns []any, boolean string) *Builder {
	return b.HavingNull(columns, boolean, true)
}

// OrHavingNotNull is HavingNotNull joined with "or".
func (b *Builder) OrHavingNotNull(columns ...any) *Builder {
	return b.HavingNotNull(columns, "or")
}

// HavingBetween adds a having that requires column to fall between the first
// two values, or outside them when not is true.
//
// Only the first two values are bound: a between has two bounds, and a longer
// list is a caller mistake that would otherwise leave bindings with no
// placeholder to fill.
func (b *Builder) HavingBetween(column any, values []any, boolean string, not bool) *Builder {
	if boolean == "" {
		boolean = "and"
	}
	b.Havings = append(b.Havings, Having{
		Type: "between", Column: column, Values: values, Boolean: boolean, Not: not,
	})

	bounds := b.CleanBindings(values)
	if len(bounds) > 2 {
		bounds = bounds[:2]
	}
	b.AddBinding(bounds, "having")
	return b
}

// HavingNotBetween is HavingBetween with not set to true.
func (b *Builder) HavingNotBetween(column any, values []any, boolean string) *Builder {
	return b.HavingBetween(column, values, boolean, true)
}

// OrHavingBetween is HavingBetween joined with "or".
func (b *Builder) OrHavingBetween(column any, values []any) *Builder {
	return b.HavingBetween(column, values, "or", false)
}

// OrHavingNotBetween is HavingBetween joined with "or" and not set to true.
func (b *Builder) OrHavingNotBetween(column any, values []any) *Builder {
	return b.HavingBetween(column, values, "or", true)
}

// OrHavingRaw adds a raw having clause joined with "or".
func (b *Builder) OrHavingRaw(sql string, bindings ...any) *Builder {
	b.Havings = append(b.Havings, Having{Type: "Raw", SQL: sql, Boolean: "or"})
	if len(bindings) > 0 {
		b.AddBinding(bindings, "having")
	}
	return b
}
