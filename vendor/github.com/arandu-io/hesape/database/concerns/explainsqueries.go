package concerns

import (
	"context"

	"github.com/arandu-io/hesape/auth"
)

// Explainable is what ExplainsQueries asks of the query it explains.
//
// A query builder reaches for its own compiled SQL, its bindings and its
// connection; those three are written out here as an interface, because
// what is a mixin in a dynamic language is an interface in Go.
type Explainable interface {
	// ToSQL returns the query compiled to SQL.
	ToSQL() string

	// GetBindings returns the values that go with it.
	GetBindings() []any

	// GetConnection returns the connection to run EXPLAIN on, narrowed to
	// the one call Explain makes on it.
	GetConnection() ExplainConnection
}

// ExplainConnection is the connection Explain runs the EXPLAIN through.
//
// A row is a map because EXPLAIN answers a different shape on every engine --
// Postgres one text column, MySQL twelve -- and a struct for it would be a
// struct for one of the three.
type ExplainConnection interface {
	// Select runs a select. The Grant is required because this executes:
	// there is no exception for a query plan, and an EXPLAIN that skipped
	// the Policy would be a way to learn about rows the caller may not read.
	Select(ctx context.Context, g auth.Grant, query string, bindings []any) ([]map[string]any, error)
}

// Explain asks the engine what it would do with this query, without
// running it.
//
// It returns the rows and the error the select can fail with. The
// statement is 'EXPLAIN ' concatenated with the compiled SQL, with the
// query's own bindings.
func Explain(ctx context.Context, g auth.Grant, query Explainable) ([]map[string]any, error) {
	sql := query.ToSQL()
	if checked, ok := query.(interface{ Err() error }); ok {
		if err := checked.Err(); err != nil {
			return nil, err
		}
	}

	bindings := query.GetBindings()

	return query.GetConnection().Select(ctx, g, "EXPLAIN "+sql, bindings)
}
