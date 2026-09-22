package query

// Expression is a value the grammar must not wrap in identifier quotes and must
// not turn into a placeholder.
//
// Everything the query builder accepts as a column or a value also accepts an
// Expression.
type Expression struct {
	value any
}

// Raw builds an Expression out of a value.
func Raw(value any) Expression {
	return Expression{value: value}
}

// GetValue returns the wrapped value. The grammar parameter is accepted and
// ignored: the value carried here is already final, and the parameter exists
// only so Expression satisfies the same signature as a grammar-aware value.
func (e Expression) GetValue(grammar Grammar) any {
	return e.value
}

// Value returns the wrapped value without requiring a grammar argument, for
// callers that have no grammar handy and do not need one.
func (e Expression) Value() any {
	return e.value
}

// String renders the expression the way the grammar concatenates it.
func (e Expression) String() string {
	return stringify(e.value)
}

// IsExpression reports whether value is an Expression or a *Expression.
func IsExpression(value any) bool {
	switch value.(type) {
	case Expression, *Expression:
		return true
	}
	return false
}
