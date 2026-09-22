package query

// JoinClause is one join, and the condition it is joined on.
//
// It embeds *Builder, so a join condition is written with the same where
// vocabulary as the query itself -- On is where for two columns, and Where
// inside a join is a where. Every Builder method is reachable on a JoinClause,
// and the two that behave differently are declared below and shadow the
// embedded ones.
type JoinClause struct {
	*Builder

	// Type is the join type: inner, left, right or cross.
	Type string

	// Table is the table or subquery expression being joined.
	Table any

	// Lateral marks the join as lateral. It is a flag rather than a second
	// type, so the grammar and Joins have one type to hold.
	Lateral bool

	parentQuery *Builder
}

// NewJoinClause builds a JoinClause for a join against table.
//
// The clause carries the parent's connection, grammar and processor so that a
// nested query built inside the join compiles against the same dialect.
func NewJoinClause(parentQuery *Builder, typ string, table any) *JoinClause {
	return &JoinClause{
		Builder:     NewBuilder(parentQuery.connection, parentQuery.Grammar, parentQuery.Processor),
		Type:        typ,
		Table:       table,
		parentQuery: parentQuery,
	}
}

// NewJoinLateralClause builds a JoinClause with Lateral already set. See
// Lateral.
func NewJoinLateralClause(parentQuery *Builder, typ string, table any) *JoinClause {
	join := NewJoinClause(parentQuery, typ, table)
	join.Lateral = true
	return join
}

// On adds a join condition.
//
// It compares two columns, so neither side becomes a binding: `on('users.id',
// '=', 'posts.user_id')` names a column on the right, not the string
// "posts.user_id". This is the one difference that catches people moving a
// condition from where to on -- in a where, the right side is a value.
//
// Passing a func opens a nested group.
func (j *JoinClause) On(first any, args ...any) *JoinClause {
	if nested, ok := first.(func(*JoinClause)); ok {
		return j.onNested(nested, "and")
	}
	operator, second := prepareValueAndOperator(args...)
	operator = j.acceptOperator(operator)
	if j.Err() != nil {
		return j
	}
	j.Wheres = append(j.Wheres, Where{
		Type:     "Column",
		First:    first,
		Operator: operator,
		Second:   second,
		Boolean:  "and",
	})
	return j
}

// OrOn is On joined with "or" instead of "and".
func (j *JoinClause) OrOn(first any, args ...any) *JoinClause {
	if nested, ok := first.(func(*JoinClause)); ok {
		return j.onNested(nested, "or")
	}
	operator, second := prepareValueAndOperator(args...)
	operator = j.acceptOperator(operator)
	if j.Err() != nil {
		return j
	}
	j.Wheres = append(j.Wheres, Where{
		Type:     "Column",
		First:    first,
		Operator: operator,
		Second:   second,
		Boolean:  "or",
	})
	return j
}

// onNested runs a callback against a fresh clause and folds the result in as
// one parenthesised group, which is what whereNested does for a query.
func (j *JoinClause) onNested(callback func(*JoinClause), boolean string) *JoinClause {
	nested := j.NewJoinClause()
	callback(nested)
	if nested.Err() != nil {
		j.setError(nested.Err())
		return j
	}
	if len(nested.Wheres) == 0 {
		return j
	}
	j.Wheres = append(j.Wheres, Where{Type: "Nested", Query: nested.Builder, Boolean: boolean})
	j.AddBinding(nested.Builder.Bindings["where"], "where")
	return j
}

// NewJoinClause returns another JoinClause rather than a Builder.
//
// Go resolves an embedded method by name, and a method on JoinClause called
// NewQuery could not return *JoinClause while the embedded one returns
// *Builder, so the two are separate names: NewQuery still returns a plain
// builder, for a subquery inside the join, and this returns the clause.
func (j *JoinClause) NewJoinClause() *JoinClause {
	return NewJoinClause(j.parentQuery, j.Type, j.Table)
}

// NewParentQuery returns a fresh builder on the table the join was declared
// against.
func (j *JoinClause) NewParentQuery() *Builder {
	return j.parentQuery.NewQuery()
}
