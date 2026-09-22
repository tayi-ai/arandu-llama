package query

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strconv"

	"github.com/arandu-io/hesape/auth"
)

// The writing half of the query builder.
//
// A write is filtered by tenant the way a read is, and the filter takes two
// shapes because the statements do. An update and a delete carry the tenant in
// their where clause, through the same scoped() every select goes through. An
// insert has no where clause, so it carries the tenant in the row: the column is
// written from the Grant, and a value the caller passed for it does not survive
// -- a row cannot be inserted into a tenant nobody authorized.

// InsertUsingGrammar is the part of a grammar that compiles an insert reading
// from a select. It is asked for rather than declared on Grammar because the
// Grammar interface is closed and this is not on it.
type InsertUsingGrammar interface {
	// CompileInsertUsing compiles an insert that reads its rows from a select
	// query.
	CompileInsertUsing(query *Builder, columns []any, sql string) string

	// CompileInsertOrIgnoreUsing compiles CompileInsertUsing's insert-or-ignore
	// variant.
	CompileInsertOrIgnoreUsing(query *Builder, columns []any, sql string) string
}

// UpdateFromGrammar is the part of the grammar that compiles Postgres's
// "update ... from" syntax.
//
// Not every grammar implements it, so callers use a type assertion against
// this interface and return an error when it fails, rather than calling a
// method that might not exist.
type UpdateFromGrammar interface {
	// CompileUpdateFrom compiles an "update ... from" statement.
	CompileUpdateFrom(query *Builder, values map[string]any) string

	// PrepareBindingsForUpdateFrom orders the bindings for an "update ... from"
	// statement to match the placeholders CompileUpdateFrom produces.
	PrepareBindingsForUpdateFrom(bindings map[string][]any, values map[string]any) []any
}

// Insert inserts one row or many.
//
// Variadic arguments accept either shape directly, without needing to inspect
// whether the caller passed one row or a list.
//
// The rows are copied before the tenant is stamped into them, so a caller's map
// does not gain a column it never asked for.
func (b *Builder) Insert(ctx context.Context, g auth.Grant, values ...map[string]any) (bool, error) {
	if len(values) == 0 {
		return true, nil
	}
	tenant, err := b.tenantFor(ctx, g)
	if err != nil {
		return false, err
	}
	rows := stampTenant(values, tenant)

	query := b.Clone()
	query.ApplyBeforeQueryCallbacks()
	if err := query.Err(); err != nil {
		return false, err
	}

	if query.connection == nil {
		return false, errors.New("query: the builder has no connection to run against")
	}
	return query.connection.Insert(
		ctx,
		query.Grammar.CompileInsert(query, rows),
		query.CleanBindings(flattenRows(rows)))
}

// InsertOrIgnore inserts rows, letting the engine drop the ones that would
// violate a unique constraint, and returns how many it kept.
func (b *Builder) InsertOrIgnore(ctx context.Context, g auth.Grant, values ...map[string]any) (int64, error) {
	if len(values) == 0 {
		return 0, nil
	}
	tenant, err := b.tenantFor(ctx, g)
	if err != nil {
		return 0, err
	}
	rows := stampTenant(values, tenant)

	query := b.Clone()
	query.ApplyBeforeQueryCallbacks()
	if err := query.Err(); err != nil {
		return 0, err
	}

	return query.affectingStatement(ctx,
		query.Grammar.CompileInsertOrIgnore(query, rows),
		query.CleanBindings(flattenRows(rows)))
}

// InsertGetID inserts a row and returns the id the engine assigned it.
//
// sequence is the name of the sequence the id comes from, which Postgres needs
// and the other engines ignore. Empty means the engine's default.
func (b *Builder) InsertGetID(ctx context.Context, g auth.Grant, values map[string]any, sequence string) (int64, error) {
	tenant, err := b.tenantFor(ctx, g)
	if err != nil {
		return 0, err
	}
	row := stampTenant([]map[string]any{values}, tenant)[0]

	query := b.Clone()
	query.ApplyBeforeQueryCallbacks()
	if err := query.Err(); err != nil {
		return 0, err
	}

	if query.Processor == nil {
		return 0, errors.New("query: the builder has no processor to read the inserted id through")
	}
	sql := query.Grammar.CompileInsertGetID(query, row, sequence)
	return query.Processor.ProcessInsertGetID(ctx, query, sql, query.CleanBindings(flattenRow(row)), sequence)
}

// InsertUsing is an insert whose rows come from a select rather than from
// values.
//
// query may be a *Builder, a func(*Builder) or a string of SQL. A *Builder or a
// callback is scoped by the same Grant before it is compiled -- an insert
// reading from a select is a read, and a read that skipped the tenant would copy
// another customer's rows into this one's table.
//
// A string is taken as written. Nothing here can add a tenant filter to SQL it
// did not build, which is the reason to hand it a builder instead.
func (b *Builder) InsertUsing(ctx context.Context, g auth.Grant, columns []any, query any) (int64, error) {
	return b.insertUsing(ctx, g, columns, query, false)
}

// InsertOrIgnoreUsing is InsertUsing letting the engine drop the rows that
// would collide.
func (b *Builder) InsertOrIgnoreUsing(ctx context.Context, g auth.Grant, columns []any, query any) (int64, error) {
	return b.insertUsing(ctx, g, columns, query, true)
}

func (b *Builder) insertUsing(ctx context.Context, g auth.Grant, columns []any, query any, ignore bool) (int64, error) {
	if _, err := b.tenantFor(ctx, g); err != nil {
		return 0, err
	}

	statement := b.Clone()
	statement.ApplyBeforeQueryCallbacks()
	if err := statement.Err(); err != nil {
		return 0, err
	}

	grammar, ok := statement.Grammar.(InsertUsingGrammar)
	if !ok {
		return 0, errors.New("query: this database engine does not support the insertUsing method")
	}

	sql, bindings, err := b.createSub(ctx, g, query)
	if err != nil {
		return 0, err
	}

	compiled := grammar.CompileInsertUsing(statement, columns, sql)
	if ignore {
		compiled = grammar.CompileInsertOrIgnoreUsing(statement, columns, sql)
	}
	return statement.affectingStatement(ctx, compiled, statement.CleanBindings(bindings))
}

// Update updates the rows matching the query with the given column values.
//
// A value may be a *Builder, which becomes the subquery it compiles to, with
// its bindings landing where its placeholders are. Such a subquery is scoped by
// the Grant first, for the reason InsertUsing's is.
//
// A value under TenantColumn is replaced by the Grant's tenant rather than
// written: an update that moved a row to another tenant would pass its own where
// clause on the way out and be unreachable afterwards.
//
// Two maps go to the grammar -- the values to compile and the bindings to send
// -- and both are keyed by column, so the grammar has to walk them in sorted key
// order for the placeholders and the values to line up.
func (b *Builder) Update(ctx context.Context, g auth.Grant, values map[string]any) (int64, error) {
	query, err := b.scoped(ctx, g)
	if err != nil {
		return 0, err
	}
	query.ApplyBeforeQueryCallbacks()
	if err := query.Err(); err != nil {
		return 0, err
	}

	compiled, bindings, err := b.prepareUpdateValues(ctx, g, values)
	if err != nil {
		return 0, err
	}
	query.ApplyBeforeQueryCallbacks()
	if err := query.Err(); err != nil {
		return 0, err
	}

	if query.connection == nil {
		return 0, errors.New("query: the builder has no connection to run against")
	}
	return query.connection.Update(
		ctx,
		query.Grammar.CompileUpdate(query, compiled),
		query.CleanBindings(query.Grammar.PrepareBindingsForUpdate(query.Bindings, bindings)))
}

// UpdateFrom runs an "update ... from" statement, which only Postgres
// compiles.
func (b *Builder) UpdateFrom(ctx context.Context, g auth.Grant, values map[string]any) (int64, error) {
	query, err := b.scoped(ctx, g)
	if err != nil {
		return 0, err
	}

	query.ApplyBeforeQueryCallbacks()
	if err := query.Err(); err != nil {
		return 0, err
	}
	grammar, ok := query.Grammar.(UpdateFromGrammar)
	if !ok {
		return 0, errors.New("query: this database engine does not support the updateFrom method")
	}

	compiled, bindings, err := b.prepareUpdateValues(ctx, g, values)
	if err != nil {
		return 0, err
	}
	query.ApplyBeforeQueryCallbacks()
	if err := query.Err(); err != nil {
		return 0, err
	}

	if query.connection == nil {
		return 0, errors.New("query: the builder has no connection to run against")
	}
	return query.connection.Update(
		ctx,
		grammar.CompileUpdateFrom(query, compiled),
		query.CleanBindings(grammar.PrepareBindingsForUpdateFrom(query.Bindings, bindings)))
}

// prepareUpdateValues splits each value into what the statement says and what
// it binds.
func (b *Builder) prepareUpdateValues(ctx context.Context, g auth.Grant, values map[string]any) (compiled map[string]any, bindings map[string]any, err error) {
	tenant, err := b.tenantFor(ctx, g)
	if err != nil {
		return nil, nil, err
	}

	compiled = make(map[string]any, len(values))
	bindings = make(map[string]any, len(values))
	for column, value := range values {
		if column == TenantColumn {
			compiled[column] = tenant
			bindings[column] = tenant
			continue
		}
		if sub, ok := value.(*Builder); ok {
			sql, subBindings, subErr := b.parseSub(ctx, g, sub)
			if subErr != nil {
				return nil, nil, subErr
			}
			compiled[column] = Raw("(" + sql + ")")
			bindings[column] = subBindings
			continue
		}
		compiled[column] = value
		bindings[column] = value
	}
	return compiled, bindings, nil
}

// UpdateOrInsert updates the first row matching attributes, or inserts one
// combining attributes and values if none matches.
//
// attributes expands into one where clause per pair, added in sorted key
// order, so that two calls with the same attributes compile to the same
// statement.
//
// values may be nil, which updates nothing and reports that the row is there.
func (b *Builder) UpdateOrInsert(ctx context.Context, g auth.Grant, attributes, values map[string]any) (bool, error) {
	query := b.Clone()
	for _, column := range sortedKeys(attributes) {
		query.Where(column, "=", attributes[column])
	}

	exists, err := query.Exists(ctx, g)
	if err != nil {
		return false, err
	}

	if !exists {
		row := make(map[string]any, len(attributes)+len(values))
		for column, value := range attributes {
			row[column] = value
		}
		for column, value := range values {
			row[column] = value
		}
		return query.Insert(ctx, g, row)
	}
	if len(values) == 0 {
		return true, nil
	}

	affected, err := query.Limit(1).Update(ctx, g, values)
	if err != nil {
		return false, err
	}
	return affected > 0, nil
}

// Upsert inserts these rows, and updates the ones that collide on uniqueBy.
//
// update names the columns an existing row takes from the new one. Nil means
// all of them; an empty non-nil slice means none, and turns the whole
// statement into a plain insert. Go's nil/empty-slice distinction is what
// keeps the two cases apart.
//
// uniqueBy should name the tenant column too. The rows carry the tenant, but the
// conflict target is a unique index, and an index that does not include the
// tenant is a constraint shared between customers.
func (b *Builder) Upsert(ctx context.Context, g auth.Grant, values []map[string]any, uniqueBy []string, update []string) (int64, error) {
	if len(uniqueBy) == 0 {
		return 0, errors.New("query: the unique columns must not be empty")
	}
	if len(values) == 0 {
		return 0, nil
	}
	if update != nil && len(update) == 0 {
		inserted, err := b.Insert(ctx, g, values...)
		if err != nil {
			return 0, err
		}
		if inserted {
			return 1, nil
		}
		return 0, nil
	}

	tenant, err := b.tenantFor(ctx, g)
	if err != nil {
		return 0, err
	}
	rows := stampTenant(values, tenant)

	if update == nil {
		update = sortedKeys(rows[0])
	}

	query := b.Clone()
	query.ApplyBeforeQueryCallbacks()
	if err := query.Err(); err != nil {
		return 0, err
	}

	// The Grammar interface here takes update as a list of column names, so
	// there is nothing in it to bind.
	return query.affectingStatement(ctx,
		query.Grammar.CompileUpsert(query, rows, uniqueBy, update),
		query.CleanBindings(flattenRows(rows)))
}

// Increment adds amount to column, along with any extra columns to set.
//
// A non-numeric amount does not compile: Go's float64 parameter type enforces
// that guarantee at the call site, before the query ever runs.
//
// extra is the columns to set alongside, and may be nil.
func (b *Builder) Increment(ctx context.Context, g auth.Grant, column string, amount float64, extra map[string]any) (int64, error) {
	return b.IncrementEach(ctx, g, map[string]float64{column: amount}, extra)
}

// IncrementEach adds separate amounts to multiple columns, along with any
// extra columns to set.
func (b *Builder) IncrementEach(ctx context.Context, g auth.Grant, columns map[string]float64, extra map[string]any) (int64, error) {
	return b.stepEach(ctx, g, columns, extra, "+")
}

// Decrement subtracts amount from column. See Increment.
func (b *Builder) Decrement(ctx context.Context, g auth.Grant, column string, amount float64, extra map[string]any) (int64, error) {
	return b.DecrementEach(ctx, g, map[string]float64{column: amount}, extra)
}

// DecrementEach subtracts separate amounts from multiple columns, along with
// any extra columns to set.
func (b *Builder) DecrementEach(ctx context.Context, g auth.Grant, columns map[string]float64, extra map[string]any) (int64, error) {
	return b.stepEach(ctx, g, columns, extra, "-")
}

// stepEach is the shared body of IncrementEach and DecrementEach, which differ
// only in the operator.
//
// The step is written into the statement rather than bound: it is a number
// this function produced, not a value that came from outside, so there is
// nothing for a placeholder to protect.
func (b *Builder) stepEach(ctx context.Context, g auth.Grant, columns map[string]float64, extra map[string]any, operator string) (int64, error) {
	values := make(map[string]any, len(columns)+len(extra))
	for column, amount := range columns {
		values[column] = Raw(b.Grammar.Wrap(column) + " " + operator + " " +
			strconv.FormatFloat(amount, 'f', -1, 64))
	}
	for column, value := range extra {
		values[column] = value
	}
	return b.Update(ctx, g, values)
}

// Delete deletes the rows matching the query.
//
// The optional id, when given, adds a where on the primary key, so that a
// single row can be removed without writing one.
func (b *Builder) Delete(ctx context.Context, g auth.Grant, id ...any) (int64, error) {
	if len(id) > 0 && id[0] != nil {
		b.Where(b.qualify("id"), "=", id[0])
	}

	query, err := b.scoped(ctx, g)
	if err != nil {
		return 0, err
	}
	query.ApplyBeforeQueryCallbacks()
	if err := query.Err(); err != nil {
		return 0, err
	}

	if query.connection == nil {
		return 0, errors.New("query: the builder has no connection to run against")
	}
	return query.connection.Delete(
		ctx,
		query.Grammar.CompileDelete(query),
		query.CleanBindings(query.Grammar.PrepareBindingsForDelete(query.Bindings)))
}

// Truncate empties the entire table, all tenants at once.
//
// Read this before calling it. A truncate takes no where clause on any engine,
// so this is the one statement in the package that cannot carry the tenant: it
// empties the table for every customer in it, and the Grant it requires
// authorizes that rather than narrowing it. Deleting one tenant's rows is
// Delete, which is filtered.
//
// It is more than one statement on some engines -- Postgres and SQL Server reset
// the sequence separately -- which is why the grammar returns a map of
// statements to bindings.
func (b *Builder) Truncate(ctx context.Context, g auth.Grant) error {
	if _, err := b.tenantFor(ctx, g); err != nil {
		return err
	}

	query := b.Clone()
	query.ApplyBeforeQueryCallbacks()
	if err := query.Err(); err != nil {
		return err
	}

	if query.connection == nil {
		return errors.New("query: the builder has no connection to run against")
	}
	// The grammar returns a map, and a Go map has no guaranteed iteration
	// order. The statements are run in sorted order so that two runs issue them
	// the same way; a grammar whose truncate depends on a specific order of its
	// statements has to say so some other way than through map order.
	statements := query.Grammar.CompileTruncate(query)
	for _, sql := range slices.Sorted(maps.Keys(statements)) {
		if _, err := query.connection.Statement(ctx, sql, statements[sql]); err != nil {
			return err
		}
	}
	return nil
}

// createSub turns a builder, a callback or a string into SQL and bindings.
func (b *Builder) createSub(ctx context.Context, g auth.Grant, query any) (string, []any, error) {
	if callback, ok := query.(func(*Builder)); ok {
		sub := b.NewQuery()
		callback(sub)
		return b.parseSub(ctx, g, sub)
	}
	return b.parseSub(ctx, g, query)
}

// parseSub turns a subquery argument into SQL and bindings.
//
// It never qualifies the table with a second database's name: reaching another
// connection from here would mean looking one up by name. A subquery on another
// database is written with the name in the table.
func (b *Builder) parseSub(ctx context.Context, g auth.Grant, query any) (string, []any, error) {
	switch sub := query.(type) {
	case *Builder:
		scoped, err := sub.scoped(ctx, g)
		if err != nil {
			return "", nil, err
		}
		sql := scoped.ToSQL()
		if err := scoped.Err(); err != nil {
			return "", nil, err
		}
		return sql, scoped.GetBindings(), nil
	case string:
		return sub, nil, nil
	case Expression:
		return stringify(sub), nil, nil
	default:
		return "", nil, errors.New("query: a subquery must be a query builder, a callback or a string")
	}
}

// tenantFor is scoped's first half: it checks the context and reads the tenant
// off the Grant, for the statements that have no where clause to put a filter
// in. See scoped for why both halves exist.
func (b *Builder) tenantFor(ctx context.Context, g auth.Grant) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	tenant := auth.Tenant(g)
	if tenant == "" {
		return "", fmt.Errorf(
			"%w: a statement on %s ran with a grant that carries no tenant. Nothing can be scoped without one, and a statement that is not scoped reads or writes every customer",
			auth.ErrForbidden, describeTable(b.GetFrom()))
	}
	return tenant, nil
}

// stampTenant copies the rows and writes the tenant into each of them.
//
// The copy is what keeps the caller's map free of a column it did not write,
// and the overwrite is what makes the Grant the only source of a tenant: a value
// the caller passed under TenantColumn does not survive this.
func stampTenant(values []map[string]any, tenant string) []map[string]any {
	out := make([]map[string]any, len(values))
	for i, row := range values {
		copied := make(map[string]any, len(row)+1)
		for column, value := range row {
			copied[column] = value
		}
		copied[TenantColumn] = tenant
		out[i] = copied
	}
	return out
}

// flattenRows flattens the rows of an insert into a single list of values, one
// level deep.
//
// The order is the sorted column order of each row, which is what the grammar
// compiles the column list from. Iterating a Go map instead would put the
// bindings in a different order on every run, and the statement would insert
// each value into a different column.
func flattenRows(values []map[string]any) []any {
	out := make([]any, 0, len(values)*4)
	for _, row := range values {
		out = append(out, flattenRow(row)...)
	}
	return out
}

func flattenRow(row map[string]any) []any {
	out := make([]any, 0, len(row))
	for _, column := range sortedKeys(row) {
		out = append(out, row[column])
	}
	return out
}
