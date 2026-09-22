package concerns

import (
	"time"

	"github.com/arandu-io/hesape/database/query"
)

// Now is where BuildsWhereDateClauses reads the clock.
//
// The clock is a variable a test replaces and restores. Everything below reads
// it, so freezing it freezes all eighteen clauses at once.
var Now = time.Now

// WherePast adds a where clause requiring each column to be before now,
// combined with and.
func WherePast(b *query.Builder, columns ...any) *query.Builder {
	return wherePastOrFuture(b, columns, "<", "and")
}

// WhereNowOrPast adds a where clause requiring each column to be at or
// before now, combined with and.
func WhereNowOrPast(b *query.Builder, columns ...any) *query.Builder {
	return wherePastOrFuture(b, columns, "<=", "and")
}

// OrWherePast is WherePast, combined with or.
func OrWherePast(b *query.Builder, columns ...any) *query.Builder {
	return wherePastOrFuture(b, columns, "<", "or")
}

// OrWhereNowOrPast is WhereNowOrPast, combined with or.
func OrWhereNowOrPast(b *query.Builder, columns ...any) *query.Builder {
	return wherePastOrFuture(b, columns, "<=", "or")
}

// WhereFuture adds a where clause requiring each column to be after now,
// combined with and.
func WhereFuture(b *query.Builder, columns ...any) *query.Builder {
	return wherePastOrFuture(b, columns, ">", "and")
}

// WhereNowOrFuture adds a where clause requiring each column to be at or
// after now, combined with and.
func WhereNowOrFuture(b *query.Builder, columns ...any) *query.Builder {
	return wherePastOrFuture(b, columns, ">=", "and")
}

// OrWhereFuture is WhereFuture, combined with or.
func OrWhereFuture(b *query.Builder, columns ...any) *query.Builder {
	return wherePastOrFuture(b, columns, ">", "or")
}

// OrWhereNowOrFuture is WhereNowOrFuture, combined with or.
func OrWhereNowOrFuture(b *query.Builder, columns ...any) *query.Builder {
	return wherePastOrFuture(b, columns, ">=", "or")
}

// wherePastOrFuture adds one Basic where per column against the clock, with
// one binding per column.
func wherePastOrFuture(b *query.Builder, columns []any, operator, boolean string) *query.Builder {
	value := Now()

	for _, column := range columns {
		b.Wheres = append(b.Wheres, query.Where{
			Type:     "Basic",
			Column:   column,
			Boolean:  boolean,
			Operator: operator,
			Value:    value,
		})
		b.AddBinding([]any{value}, "where")
	}
	return b
}

// WhereToday adds a where clause requiring each column to fall on today's
// date. boolean empty defaults to "and".
func WhereToday(b *query.Builder, boolean string, columns ...any) *query.Builder {
	if boolean == "" {
		boolean = "and"
	}
	return whereTodayBeforeOrAfter(b, columns, "=", boolean)
}

// WhereBeforeToday adds a where clause requiring each column to fall before
// today's date, combined with and.
func WhereBeforeToday(b *query.Builder, columns ...any) *query.Builder {
	return whereTodayBeforeOrAfter(b, columns, "<", "and")
}

// WhereTodayOrBefore adds a where clause requiring each column to fall on or
// before today's date, combined with and.
func WhereTodayOrBefore(b *query.Builder, columns ...any) *query.Builder {
	return whereTodayBeforeOrAfter(b, columns, "<=", "and")
}

// WhereAfterToday adds a where clause requiring each column to fall after
// today's date, combined with and.
func WhereAfterToday(b *query.Builder, columns ...any) *query.Builder {
	return whereTodayBeforeOrAfter(b, columns, ">", "and")
}

// WhereTodayOrAfter adds a where clause requiring each column to fall on or
// after today's date, combined with and.
func WhereTodayOrAfter(b *query.Builder, columns ...any) *query.Builder {
	return whereTodayBeforeOrAfter(b, columns, ">=", "and")
}

// OrWhereToday is WhereToday, combined with or.
func OrWhereToday(b *query.Builder, columns ...any) *query.Builder {
	return WhereToday(b, "or", columns...)
}

// OrWhereBeforeToday is WhereBeforeToday, combined with or.
func OrWhereBeforeToday(b *query.Builder, columns ...any) *query.Builder {
	return whereTodayBeforeOrAfter(b, columns, "<", "or")
}

// OrWhereTodayOrBefore is WhereTodayOrBefore, combined with or.
func OrWhereTodayOrBefore(b *query.Builder, columns ...any) *query.Builder {
	return whereTodayBeforeOrAfter(b, columns, "<=", "or")
}

// OrWhereAfterToday is WhereAfterToday, combined with or.
func OrWhereAfterToday(b *query.Builder, columns ...any) *query.Builder {
	return whereTodayBeforeOrAfter(b, columns, ">", "or")
}

// OrWhereTodayOrAfter is WhereTodayOrAfter, combined with or.
func OrWhereTodayOrAfter(b *query.Builder, columns ...any) *query.Builder {
	return whereTodayBeforeOrAfter(b, columns, ">=", "or")
}

// whereTodayBeforeOrAfter adds one Date where per column against today's
// date, formatted the way the column stores it.
//
// The date is formatted with the "2006-01-02" layout, as a string on
// purpose: a Date where compares the date part, and handing the driver a
// time.Time here would let the engine decide whether the time part counts,
// which is exactly the disagreement database.Day exists to settle.
func whereTodayBeforeOrAfter(b *query.Builder, columns []any, operator, boolean string) *query.Builder {
	value := Now().Format("2006-01-02")

	for _, column := range columns {
		b.Wheres = append(b.Wheres, query.Where{
			Type:     "Date",
			Column:   column,
			Boolean:  boolean,
			Operator: operator,
			Value:    value,
		})
		if !query.IsExpression(value) {
			b.AddBinding([]any{value}, "where")
		}
	}
	return b
}
