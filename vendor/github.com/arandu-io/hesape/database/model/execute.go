package model

import (
	"context"
	"fmt"

	"github.com/arandu-io/hesape/database/query"
)

// The statements this package issues: select, insert, insert-returning-id,
// update, upsert, delete and the aggregates.
//
// They run here rather than in the query package because running a statement
// takes a Grant and building SQL does not: query.Builder is the SQL, and the
// layer that holds the authorization is the layer that issues it. The SQL
// itself is still the grammar's -- nothing here concatenates a fragment.

// runSelect runs the query's SELECT and returns the rows.
func (b *Builder[T]) runSelect(ctx context.Context) ([]query.Record, error) {
	sql := b.query.ToSQL()
	if err := b.query.Err(); err != nil {
		return nil, err
	}
	rows, err := b.model.connection.Select(ctx, sql, b.query.GetBindings(), !b.query.UsingWritePDO())
	if err != nil {
		return nil, fmt.Errorf("model: selecting from %s: %w", b.model.GetTable(), err)
	}
	if b.model.Processor != nil {
		rows = b.model.Processor.ProcessSelect(b.query, rows)
	}
	return rows, nil
}

// runInsert runs an INSERT for values, and reports whether it succeeded.
//
// The rows are sorted by column name before they are compiled and bound,
// which is the only ordering available here, since a Go map has none.
func (b *Builder[T]) runInsert(ctx context.Context, values []map[string]any) (bool, error) {
	if len(values) == 0 {
		return true, nil
	}
	if err := b.validateWriteQuery(); err != nil {
		return false, err
	}

	sql := b.model.Grammar.CompileInsert(b.query, values)
	bindings := make([]any, 0, len(values)*len(values[0]))
	for _, row := range values {
		for _, column := range sortedKeys(row) {
			bindings = append(bindings, row[column])
		}
	}
	ok, err := b.model.connection.Insert(ctx, sql, cleanBindings(bindings))
	if err != nil {
		return false, fmt.Errorf("model: inserting into %s: %w", b.model.GetTable(), err)
	}
	return ok, nil
}

// runInsertGetID runs an INSERT for one row and returns the value generated
// for sequence.
func (b *Builder[T]) runInsertGetID(ctx context.Context, values map[string]any, sequence string) (int64, error) {
	if err := b.validateWriteQuery(); err != nil {
		return 0, err
	}

	sql := b.model.Grammar.CompileInsertGetID(b.query, values, sequence)

	bindings := make([]any, 0, len(values))
	for _, column := range sortedKeys(values) {
		bindings = append(bindings, values[column])
	}

	id, err := b.model.Processor.ProcessInsertGetID(ctx, b.query, sql, cleanBindings(bindings), sequence)
	if err != nil {
		return 0, fmt.Errorf("model: inserting into %s: %w", b.model.GetTable(), err)
	}
	return id, nil
}

// runUpdate runs an UPDATE for values and returns the number of rows
// affected.
func (b *Builder[T]) runUpdate(ctx context.Context, values map[string]any) (int64, error) {
	if err := b.validateWriteQuery(); err != nil {
		return 0, err
	}

	sql := b.model.Grammar.CompileUpdate(b.query, values)
	bindings := b.model.Grammar.PrepareBindingsForUpdate(b.query.GetRawBindings(), values)

	affected, err := b.model.connection.Update(ctx, sql, cleanBindings(bindings))
	if err != nil {
		return 0, fmt.Errorf("model: updating %s: %w", b.model.GetTable(), err)
	}
	return affected, nil
}

// runUpsert issues the upsert.
//
// It goes through query.Connection's Update -- a statement that reports how many
// rows it touched.
func (b *Builder[T]) runUpsert(ctx context.Context, values []map[string]any, uniqueBy, update []string) (int64, error) {
	if err := b.validateWriteQuery(); err != nil {
		return 0, err
	}

	sql := b.model.Grammar.CompileUpsert(b.query, values, uniqueBy, update)
	bindings := make([]any, 0, len(values)*len(values[0]))
	for _, row := range values {
		for _, column := range sortedKeys(row) {
			bindings = append(bindings, row[column])
		}
	}

	affected, err := b.model.connection.Update(ctx, sql, cleanBindings(bindings))
	if err != nil {
		return 0, fmt.Errorf("model: upserting into %s: %w", b.model.GetTable(), err)
	}
	return affected, nil
}

// runDelete runs a DELETE and returns the number of rows affected.
func (b *Builder[T]) runDelete(ctx context.Context) (int64, error) {
	if err := b.validateWriteQuery(); err != nil {
		return 0, err
	}

	sql := b.model.Grammar.CompileDelete(b.query)
	bindings := b.model.Grammar.PrepareBindingsForDelete(b.query.GetRawBindings())

	affected, err := b.model.connection.Delete(ctx, sql, cleanBindings(bindings))
	if err != nil {
		return 0, fmt.Errorf("model: deleting from %s: %w", b.model.GetTable(), err)
	}
	return affected, nil
}

// validateWriteQuery applies the operator policy of the grammar that will
// actually compile the model write. SetQuery is public, so the query carried
// by a model may have been constructed with a different dialect.
func (b *Builder[T]) validateWriteQuery() error {
	compilerGrammar := b.model.Grammar
	b.query.Grammar = compilerGrammar
	b.query.ApplyBeforeQueryCallbacks()
	if err := b.query.Err(); err != nil {
		return err
	}

	// A callback may replace the query's public Grammar. Model writes are
	// compiled by model.Grammar, so make that policy authoritative after every
	// callback has run and validate the complete graph once more.
	b.query.Grammar = compilerGrammar
	b.query.ApplyBeforeQueryCallbacks()
	return b.query.Err()
}

// runAggregate returns the one row an aggregate select returns, read out of
// the column the grammar aliases as "aggregate".
func (b *Builder[T]) runAggregate(ctx context.Context, function string, columns []any) (any, error) {
	aggregate := b.clone()
	aggregate.query = aggregate.query.
		CloneWithout("columns", "orders").
		CloneWithoutBindings("select", "order")
	aggregate.query.SetAggregate(function, columns)

	rows, err := aggregate.runSelect(ctx)
	if err != nil {
		return nil, err
	}
	if len(rows) == 0 {
		return nil, nil
	}
	for key, value := range rows[0] {
		if lowerASCII(key) == "aggregate" {
			return value, nil
		}
	}
	return nil, fmt.Errorf("model: the %s query on %s came back without an aggregate column", function, b.model.GetTable())
}

// cleanBindings drops every query.Expression from bindings: an expression
// is SQL and never a binding, so it is dropped from the list rather than
// sent as a value.
func cleanBindings(bindings []any) []any {
	out := make([]any, 0, len(bindings))
	for _, binding := range bindings {
		if query.IsExpression(binding) {
			continue
		}
		out = append(out, binding)
	}
	return out
}

func lowerASCII(s string) string {
	out := []byte(s)
	for i, c := range out {
		if c >= 'A' && c <= 'Z' {
			out[i] = c + ('a' - 'A')
		}
	}
	return string(out)
}
