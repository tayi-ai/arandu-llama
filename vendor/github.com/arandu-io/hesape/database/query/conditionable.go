package query

// When runs the callback only if the condition holds, and hands the builder
// back either way.
//
// The condition is a bool, because the caller writing the expression is the one
// who knows what makes it true. The callback mutates the builder and returns
// nothing: every method on Builder already mutates and returns the receiver, so
// a callback that returned something would be a second convention.
//
// A nil callback on a true condition leaves the builder untouched instead of
// failing, which is what Collection.When does with the same argument.
//
//	q.From("posts").
//		When(published, func(q *query.Builder) { q.Where("published", true) }, nil).
//		OrderBy("id")
func (b *Builder) When(condition bool, callback, otherwise func(*Builder)) *Builder {
	if condition {
		if callback != nil {
			callback(b)
		}
		return b
	}
	if otherwise != nil {
		otherwise(b)
	}
	return b
}

// Unless is When with the condition negated.
func (b *Builder) Unless(condition bool, callback, otherwise func(*Builder)) *Builder {
	return b.When(!condition, callback, otherwise)
}
