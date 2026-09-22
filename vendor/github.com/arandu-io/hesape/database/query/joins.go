package query

// The join vocabulary that Join, LeftJoin, RightJoin and CrossJoin do not
// cover: the "join where" family, whose
// condition compares a column with a value rather than with another column, and
// MySQL's straight join.

// addJoinClause is the shared body of the exported join methods, with the two
// flags the fluent methods vary: the join type, and whether the condition is
// an "on" or a "where".
//
// The difference matters and is easy to miss: `on('users.id', '=', 'x')` names
// a column called x, and `where('users.id', '=', 'x')` binds the string.
func (b *Builder) addJoinClause(typ string, isWhere bool, table any, first any, args ...any) *Builder {
	join := NewJoinClause(b, typ, table)

	if callback, ok := first.(func(*JoinClause)); ok {
		callback(join)
	} else {
		operator, second := prepareValueAndOperator(args...)
		if isWhere {
			join.Where(first, operator, second)
		} else {
			join.On(first, operator, second)
		}
	}

	b.Joins = append(b.Joins, join)
	b.AddBinding(join.GetBindings(), "join")
	return b
}

// joinArguments drops the trailing nils of an operator and second column, so
// that a call passing neither reaches the variadic form the rest of this
// package uses.
func joinArguments(operator, second any) []any {
	args := make([]any, 0, 2)
	if operator != nil {
		args = append(args, operator)
	}
	if second != nil {
		args = append(args, second)
	}
	return args
}

// JoinWhere adds a join whose condition compares a column with a value.
func (b *Builder) JoinWhere(table any, first any, operator, second any, typ string) *Builder {
	if typ == "" {
		typ = "inner"
	}
	return b.addJoinClause(typ, true, table, first, joinArguments(operator, second)...)
}

// LeftJoinWhere is JoinWhere with typ set to "left".
func (b *Builder) LeftJoinWhere(table any, first any, operator, second any) *Builder {
	return b.JoinWhere(table, first, operator, second, "left")
}

// RightJoinWhere is JoinWhere with typ set to "right".
func (b *Builder) RightJoinWhere(table any, first any, operator, second any) *Builder {
	return b.JoinWhere(table, first, operator, second, "right")
}

// StraightJoin is MySQL's join that forbids the planner from reordering the
// tables.
//
// Every other engine is handed the type as written and rejects it, which is
// what the grammar's SupportsStraightJoins reports.
func (b *Builder) StraightJoin(table any, first any, operator, second any) *Builder {
	return b.addJoinClause("straight_join", false, table, first, joinArguments(operator, second)...)
}

// StraightJoinWhere is JoinWhere with typ set to "straight_join".
func (b *Builder) StraightJoinWhere(table any, first any, operator, second any) *Builder {
	return b.JoinWhere(table, first, operator, second, "straight_join")
}
