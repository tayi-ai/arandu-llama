package query

import (
	"context"
	"errors"
	"fmt"
	"io"
	"maps"
	"slices"
	"strconv"
	"strings"

	"github.com/arandu-io/hesape/auth"
)

// The execution half of the query builder, and the errors it raises.
//
// The errors are declared here for the reason the Connection interface is
// declared here -- database imports this package, so naming them there would
// close the cycle.
var (
	// ErrRecordNotFound is what FirstOrFail returns when nothing matched.
	ErrRecordNotFound = errors.New("query: no record found for the given query")

	// ErrRecordsNotFound is what Sole returns when nothing matched.
	ErrRecordsNotFound = errors.New("query: no records found for the given query")

	// ErrMultipleRecordsFound is what Sole returns when more than one row
	// matched. The message carries the count.
	ErrMultipleRecordsFound = errors.New("query: multiple records found")
)

// TenantColumn is the column every statement this package issues is filtered
// by, and the column every row it writes carries.
//
// It is a constant and not a setting. The tenant comes off the Grant and from
// nowhere else; a per-query column name would be a second place
// for it to come from, and the query that got it wrong would still compile, still
// run, and still return another customer's rows.
const TenantColumn = "tenant_id"

// GroupLimitRow is the alias the group-limit compilation gives its row-number
// column. Get strips it out of every row, because it is scaffolding of the
// query rather than data anybody asked for.
//
// It is exported because the grammar that emits the alias and the builder that
// strips it have to agree on one string, and two spellings of it is a stray
// column in somebody's result set.
const GroupLimitRow = "hesape_row"

// GroupLimitGroup is the prefix of the user-variable assignment the MySQL
// group-limit compilation puts in the select list, and the second key Get
// strips.
const GroupLimitGroup = "@hesape_group := "

// AffectingConnection is the part of a connection that the write half of the
// builder reaches for and Connection does not declare: the statement that
// returns a row count.
//
// A connection that does not implement it still works: Update is what the
// fallback uses, and the count means the same thing.
type AffectingConnection interface {
	// AffectingStatement runs a statement and returns the number of rows
	// affected.
	AffectingStatement(ctx context.Context, query string, bindings []any) (int64, error)
}

// CursorConnection is the part of a connection that Cursor needs: a select that
// yields its rows one at a time instead of materialising them.
//
// A connection that does not implement it makes Cursor fail rather than fall
// back to a buffered select. The fallback would work and would be a lie: the
// whole reason to reach for a cursor is the result set that does not fit in
// memory, and finding out by being killed is worse than finding out by an
// error.
type CursorConnection interface {
	// Cursor runs a select and yields its rows one at a time.
	Cursor(ctx context.Context, query string, bindings []any, useReadPDO bool) (func(yield func(Record, error) bool), error)
}

// PreparesBindings turns a driver-specific value into one the driver accepts.
// ToRawSQL asks for it, and takes the bindings unchanged from a connection that
// has no opinion.
type PreparesBindings interface {
	// PrepareBindings turns a driver-specific value into one the driver
	// accepts.
	PrepareBindings(bindings []any) []any
}

// scoped is the single door every statement in this package goes through.
//
// The tenant is applied here, at the last possible moment, so that no execution
// path can be written that skips it.
//
// Three things happen, in this order.
//
// The context is checked. The Connection this package was handed carries no
// context into the driver, so the whole of what a cancelled context can do is
// stop the statement from being issued. That is checked once per execution,
// which means once per chunk of a chunked walk.
//
// The Grant is required to carry a tenant. auth.Tenant is the empty string on
// the zero Grant and on the Grant auth.SystemGrant returns when it refuses --
// asked for no tenant, or for one that cannot be a tenant. So an empty tenant
// here is a caller who never authorized anything or a caller whose system grant
// was turned down, and neither statement reaches the connection.
//
// A tenant is all this establishes. auth.SystemGrant is exported and issues a
// Grant carrying one for work no Policy answered about, so whether what arrives
// here passed a Policy is settled by `aru doctor` and by the caller going
// through a repository -- not by this check.
//
// The tenant clause is added to a COPY. The query the caller built is not
// changed, for a reason worth the allocation: running the same builder twice
// under two Grants must not leave the first tenant's filter on the second
// tenant's statement. The clause belongs to the statement, not to the builder.
//
// Existing clauses are wrapped in a group before the clause is added, so that a
// query written with an OR compiles as `tenant_id = ? and (a or b)` rather than
// `a or b and tenant_id = ?`. AND binds tighter than OR on every engine, so the
// second reading leaves `a` unfiltered -- which is one customer's rows answering
// another customer's query, with no error anywhere.
func (b *Builder) scoped(ctx context.Context, g auth.Grant) (*Builder, error) {
	tenant, err := b.tenantFor(ctx, g)
	if err != nil {
		return nil, err
	}

	scoped := b.Clone()
	if len(scoped.Wheres) > 0 {
		inner := b.Clone()
		inner.BeforeQueryCallbacks = nil
		scoped.Wheres = nil
		scoped.Bindings["where"] = nil
		scoped.Where(b.qualifyTenantColumn(), "=", tenant)
		scoped.AddNestedWhereQuery(inner, "and")
	} else {
		scoped.Where(b.qualifyTenantColumn(), "=", tenant)
	}

	if err := scoped.ScopeNested(ctx, g); err != nil {
		return nil, err
	}
	return scoped, nil
}

// ScopeNested puts the tenant on every query nested inside this one: the far
// side of a union, the subquery of a `where exists`, of a `where in` and of a
// count comparison, and the subqueries compiled into a from, a select or a join.
//
// It is the second half of scoped, and it is exported because it is the half the
// model builder has to end in too. That builder cannot call scoped whole: it
// filters by the model's own tenant column, which a model whose table has none
// sets to the empty string, and TenantColumn here is a constant on purpose --
// see its comment. The nested half has no such difference, and it is where every
// leak found so far lived, so there is one of it and both builders run it.
//
// Until it was exported the model builder reached none of it. GetModels sent
// b.query.ToSQL() straight to the connection, so scopeUnions, scopeSubqueries
// and scopeSubqueryClauses were dead code on the path the application uses:
// Users.WithCount("posts").Get(g) came back with every tenant's posts counted
// into posts_count, and WhereHas("posts", ...) filtered one tenant's users by
// another tenant's rows.
//
// It runs on the statement being built and never on the builder the caller
// holds -- scoped calls it on a clone, and the model builder on the clone
// ApplyScopes made.
func (b *Builder) ScopeNested(ctx context.Context, g auth.Grant) error {
	b.ApplyBeforeQueryCallbacks()
	if err := b.Err(); err != nil {
		return err
	}
	if err := b.scopeUnions(ctx, g); err != nil {
		return err
	}
	if err := b.scopeSubqueries(ctx, g); err != nil {
		return err
	}
	if err := b.scopeSubqueryClauses(ctx, g); err != nil {
		return err
	}
	if err := b.scopeJoins(ctx, g); err != nil {
		return err
	}
	b.ApplyBeforeQueryCallbacks()
	return b.Err()
}

// scopeJoins filters every table this query joins.
//
// A join contributes no filter of its own: the tenant clause names the table in
// the from, and everything the query reaches by joining is read whole. That is
// how a pivot row belonging to one customer came to grant a role in another --
// the role was filtered, the row that granted it was not.
//
// The relations package closes the same hole for the joins it writes itself
// (concerns.ScopeTenant), and this closes it for the joins it does not: an
// existence subquery -- whereHas, has, withCount over a BelongsToMany or a
// HasManyThrough -- is built with no Grant and reaches the connection through
// the subquery pass above, which scopes each subquery's own from and nothing it
// joins. Putting the pass here means the subquery gets it on the way through.
//
// A derived table -- the `(select ...) as alias` a subquery join compiles to --
// is skipped: it has no tenant column of its own, and the query inside it was
// scoped by scopeSubqueryClauses a moment ago. A table already carrying the
// filter is skipped too, so scoping twice does not write the clause twice.
//
// Everything else is filtered without asking whether the table is partitioned,
// and the reason is which way each mistake fails. A shared table filtered here
// yields a statement the engine refuses by name, which is a build that stops. A
// partitioned table left unfiltered yields a read that crosses customers and
// looks right.
func (b *Builder) scopeJoins(ctx context.Context, g auth.Grant) error {
	if len(b.Joins) == 0 {
		return nil
	}

	tenant, err := b.tenantFor(ctx, g)
	if err != nil {
		return err
	}

	for _, join := range b.Joins {
		table, ok := joinedTableName(join.Table)
		if !ok {
			continue
		}
		column := table + "." + TenantColumn
		if b.hasFilterOn(column) {
			continue
		}
		b.Where(column, "=", tenant)
	}
	return nil
}

// joinedTableName returns the name a joined table's columns are qualified
// by, and whether there is one at all.
//
// The alias wins when the join declares one: `users as recent` is referred to
// by the alias, and the real name resolves to nothing. An expression is not a
// table name and reports false, which is what skips a derived table.
func joinedTableName(table any) (string, bool) {
	name, ok := table.(string)
	if !ok {
		return "", false
	}

	name = strings.TrimSpace(name)
	if index := strings.Index(strings.ToLower(name), " as "); index >= 0 {
		name = strings.TrimSpace(name[index+len(" as "):])
	}
	if name == "" {
		return "", false
	}
	return name, true
}

// hasFilterOn reports whether the query already compares column with a value.
func (b *Builder) hasFilterOn(column string) bool {
	for _, where := range b.Wheres {
		if where.Type == "Basic" && fmt.Sprint(where.Column) == column {
			return true
		}
	}
	return false
}

// scopeSubqueries puts the tenant on every query nested inside a where clause.
//
// `where exists (select ... from invoices)` compiles that inner select whole,
// so a filter on the outer query says nothing about which invoices it looked
// at: the outer row belongs to this tenant, and the reason it was returned may
// be a row belonging to another. Same for a whereIn over a subquery, and same
// for a nested group that carries either.
//
// It walks the clauses rather than the flat binding list, and then rebuilds
// that list with WhereBindings -- scoping a subquery adds a binding to it, so
// leaving the old list in place would slide every value after it onto the wrong
// placeholder. This is why WhereRaw records its bindings on the clause.
func (b *Builder) scopeSubqueries(ctx context.Context, g auth.Grant) error {
	changed := false
	for i, where := range b.Wheres {
		if where.Query == nil {
			continue
		}
		switch where.Type {
		case "Exists", "In":
			// Another table, compiled whole. It needs a tenant of its own.
			scoped, err := where.Query.scoped(ctx, g)
			if err != nil {
				return err
			}
			where.Query = scoped
			b.Wheres[i] = where
			changed = true

		case "Basic":
			// A subquery on the left of a comparison -- `(select count(*) from
			// posts where ...) > 3` -- which only WhereSubCount builds, and only
			// it puts a Query on a Basic clause. The comparison is compiled from
			// the builder rather than kept as the SQL it was written as, so the
			// column is rendered again here, now that the subquery carries its
			// tenant.
			scoped, err := where.Query.scoped(ctx, g)
			if err != nil {
				return err
			}
			sql := scoped.ToSQL()
			if err := scoped.Err(); err != nil {
				b.setError(err)
				return err
			}
			where.Query = scoped
			where.Column = Raw("(" + sql + ")")
			b.Wheres[i] = where
			changed = true

		case "Nested":
			// A parenthesised group of the SAME query. It is already under the
			// outer filter and must not be scoped again -- scoped() builds one
			// of these itself to wrap what it found, so scoping it here would
			// wrap the wrapper forever. It is descended into, because the group
			// may hold an Exists.
			//
			// The group is copied first. Clone copies the where slice but not
			// the builders the clauses point at, so the group a caller wrote
			// with Where(func(group)) is the same object on the builder and on
			// every statement built from it: descending into it without a copy
			// wrote the tenant onto the caller's query. Running that builder
			// again added a second filter inside the first, and a chunked walk
			// grew one per page.
			group := where.Query.Clone()
			if err := group.scopeSubqueries(ctx, g); err != nil {
				return err
			}
			where.Query = group
			b.Wheres[i] = where
			changed = true
		}
	}
	if changed {
		b.Bindings["where"] = b.WhereBindings()
	}
	return nil
}

// scopeUnions puts the tenant on the far side of every union.
//
// A union compiles each of its queries in full, so `select from a union select
// from b` filtered only on the outer query returns every row of b, for every
// customer. The union is the one subquery this can reach: its bindings live in
// their own segment, so they can be collected again in the order the compiled
// statement reads them.
//
// The where clause of a query is the one that cannot -- a whereRaw's bindings
// are appended straight to the segment and are not recorded on the clause, so
// the list cannot be rebuilt from the clauses. A *Builder handed to WhereExists
// or WhereIn is therefore compiled as it was written: scope it before handing it
// over, or hand over a query built by a repository that already did.
func (b *Builder) scopeUnions(ctx context.Context, g auth.Grant) error {
	if len(b.Unions) == 0 {
		return nil
	}
	b.Bindings["union"] = nil
	for i, union := range b.Unions {
		scoped, err := union.Query.scoped(ctx, g)
		if err != nil {
			return err
		}
		b.Unions[i] = Union{Query: scoped, All: union.All}
		b.AddBinding(scoped.GetBindings(), "union")
	}
	return nil
}

// qualifyTenantColumn names the tenant column on the table the query reads
// from, so that a query with a join says users.tenant_id rather than a
// tenant_id the engine cannot resolve.
func (b *Builder) qualifyTenantColumn() string { return b.qualify(TenantColumn) }

// qualify names a column on the table the query reads from.
//
// The alias wins when the table has one: `users as u` qualifies as `u.id`,
// not `users as u.id` -- a tenant filter that fails to compile is a query
// somebody deletes to make the build pass.
//
// A table that is an Expression gets the bare column: there is no name to
// qualify with, and guessing one out of raw SQL is worse than not qualifying.
func (b *Builder) qualify(column string) string {
	table, ok := b.GetFrom().(string)
	if !ok || table == "" {
		return column
	}
	if i := aliasIndex(table); i >= 0 {
		table = strings.TrimSpace(table[i+4:])
	}
	return table + "." + column
}

// describeTable names the table for an error message, and says so when there is
// no name to give.
func describeTable(from any) string {
	if from == nil {
		return "an unnamed table"
	}
	if name := stringify(from); name != "" {
		return name
	}
	return "an unnamed table"
}

// runSelect runs the query's compiled select and returns its rows.
//
// There is no fetch mode to choose: a row arrives as a Record either way.
func (b *Builder) runSelect(ctx context.Context) ([]Record, error) {
	if b.connection == nil {
		return nil, errors.New("query: the builder has no connection to run against")
	}
	sql := b.ToSQL()
	if err := b.Err(); err != nil {
		return nil, err
	}
	return b.connection.Select(ctx, sql, b.GetBindings(), !b.UsingWritePDO())
}

// affectingStatement runs a statement that returns a row count, through
// AffectingConnection when the connection has it and through Update when it
// does not.
func (b *Builder) affectingStatement(ctx context.Context, query string, bindings []any) (int64, error) {
	if b.connection == nil {
		return 0, errors.New("query: the builder has no connection to run against")
	}
	if connection, ok := b.connection.(AffectingConnection); ok {
		return connection.AffectingStatement(ctx, query, bindings)
	}
	return b.connection.Update(ctx, query, bindings)
}

// Get runs the query and returns its rows.
//
// The columns are variadic and default to every column; they only apply when
// nothing was selected already. They are set on the scoped copy the tenant
// clause is added to, so the builder the caller holds is never mutated.
func (b *Builder) Get(ctx context.Context, g auth.Grant, columns ...any) ([]Record, error) {
	query, err := b.scoped(ctx, g)
	if err != nil {
		return nil, err
	}
	if query.Columns == nil {
		query.Columns = wrapColumns(columns)
	}

	rows, err := query.runSelect(ctx)
	if err != nil {
		return nil, err
	}
	if query.Processor != nil {
		rows = query.Processor.ProcessSelect(query, rows)
	}
	if query.GetGroupLimit() != nil {
		rows = withoutGroupLimitKeys(query, rows)
	}
	return rows, nil
}

// withoutGroupLimitKeys drops the bookkeeping columns a group limit adds to
// the select list.
func withoutGroupLimitKeys(query *Builder, rows []Record) []Record {
	remove := []string{GroupLimitRow}
	if limit := query.GetGroupLimit(); limit != nil && limit.Column != "" {
		column := limit.Column
		if i := strings.LastIndex(column, "."); i >= 0 {
			column = column[i+1:]
		}
		remove = append(remove,
			GroupLimitGroup+query.Grammar.Wrap(column),
			GroupLimitGroup+query.Grammar.Wrap("pivot_"+column))
	}
	for _, row := range rows {
		for _, key := range remove {
			delete(row, key)
		}
	}
	return rows
}

// wrapColumns normalizes the columns argument of Get, Pluck and their kind:
// no columns at all means every column.
func wrapColumns(columns []any) []any {
	if len(columns) == 0 {
		return []any{"*"}
	}
	return columns
}

// First returns the first row the query matches, or nil if none did.
//
// A nil Record means no row matched. The error is for a statement that
// failed, which is a different outcome and reads differently at the call
// site.
func (b *Builder) First(ctx context.Context, g auth.Grant, columns ...any) (Record, error) {
	rows, err := b.Limit(1).Get(ctx, g, columns...)
	if err != nil {
		return nil, err
	}
	if len(rows) == 0 {
		return nil, nil
	}
	return rows[0], nil
}

// FirstOrFail returns the first row the query matches, or an error if none
// did.
//
// The message is a variadic string wrapping ErrRecordNotFound, so errors.Is
// still recognises it however it was worded.
func (b *Builder) FirstOrFail(ctx context.Context, g auth.Grant, columns []any, message ...string) (Record, error) {
	row, err := b.First(ctx, g, columns...)
	if err != nil {
		return nil, err
	}
	if row != nil {
		return row, nil
	}
	if len(message) > 0 && message[0] != "" {
		return nil, fmt.Errorf("%w: %s", ErrRecordNotFound, message[0])
	}
	return nil, ErrRecordNotFound
}

// FirstOr returns the first row the query matches, or the result of callback
// if none did. It is the sibling of FindOr, spelled out here as well so that
// the two read the same and a caller who reaches for one finds the other.
//
// The result is a Record rather than a type parameter of the callback's own
// choosing: a method cannot declare type parameters beyond what its receiver
// already has, and *Builder has none.
func (b *Builder) FirstOr(ctx context.Context, g auth.Grant, columns []any, callback func() (Record, error)) (Record, error) {
	row, err := b.First(ctx, g, columns...)
	if err != nil {
		return nil, err
	}
	if row != nil {
		return row, nil
	}
	if callback == nil {
		return nil, nil
	}
	return callback()
}

// Find adds an id filter and returns the first matching row.
//
// It adds the where to the builder it is called on, so calling it twice with
// two ids asks for a row that is both.
func (b *Builder) Find(ctx context.Context, g auth.Grant, id any, columns ...any) (Record, error) {
	return b.Where("id", "=", id).First(ctx, g, columns...)
}

// FindOr adds an id filter and returns the first matching row, or the
// result of callback if none matched. The columns may be nil.
func (b *Builder) FindOr(ctx context.Context, g auth.Grant, id any, columns []any, callback func() (Record, error)) (Record, error) {
	row, err := b.Find(ctx, g, id, columns...)
	if err != nil {
		return nil, err
	}
	if row != nil {
		return row, nil
	}
	if callback == nil {
		return nil, nil
	}
	return callback()
}

// Sole returns the query's only matching row, and errors if there is not
// exactly one.
//
// It reads two rows to find out whether there is more than one, and reports
// ErrRecordsNotFound or ErrMultipleRecordsFound.
func (b *Builder) Sole(ctx context.Context, g auth.Grant, columns ...any) (Record, error) {
	rows, err := b.Limit(2).Get(ctx, g, columns...)
	if err != nil {
		return nil, err
	}
	switch len(rows) {
	case 0:
		return nil, ErrRecordsNotFound
	case 1:
		return rows[0], nil
	default:
		return nil, fmt.Errorf("%w: %d records were found", ErrMultipleRecordsFound, len(rows))
	}
}

// Value returns the value of column from the query's first matching row.
//
// A Record is a map and a map has no first, so the value is the only one when
// the row holds one, and otherwise the one under the column's own name --
// with the table qualifier and the alias stripped, which is what the driver
// keys it by.
func (b *Builder) Value(ctx context.Context, g auth.Grant, column any) (any, error) {
	row, err := b.First(ctx, g, column)
	if err != nil {
		return nil, err
	}
	return firstValue(row, column), nil
}

// RawValue returns the value of a raw expression from the query's first
// matching row.
//
// Alias the expression -- rawValue("count(*) as total") -- when the query
// already selects something: a Go map has no first column, so the alias is
// the name the value is found under.
func (b *Builder) RawValue(ctx context.Context, g auth.Grant, expression string, bindings ...any) (any, error) {
	row, err := b.SelectRaw(expression, bindings...).First(ctx, g)
	if err != nil {
		return nil, err
	}
	return firstValue(row, expression), nil
}

// SoleValue returns the value of column from the query's only matching row,
// erroring as Sole does if there is not exactly one.
func (b *Builder) SoleValue(ctx context.Context, g auth.Grant, column any) (any, error) {
	row, err := b.Sole(ctx, g, column)
	if err != nil {
		return nil, err
	}
	return firstValue(row, column), nil
}

// firstValue returns the one value of a row that holds exactly one, or the
// value under column's own name otherwise. See Value.
func firstValue(row Record, column any) any {
	if len(row) == 0 {
		return nil
	}
	if len(row) == 1 {
		for _, value := range row {
			return value
		}
	}
	if value, ok := row[stripTableForPluck(column)]; ok {
		return value
	}
	return nil
}

// Pluck returns the values of column across every matching row, in row
// order, and -- only when a key column is named -- the same values keyed by
// it. keyed is nil when no key column was given.
//
// The keys are always strings: a key value is formatted to a string on the
// way in, whatever type it arrived as.
func (b *Builder) Pluck(ctx context.Context, g auth.Grant, column any, key ...any) (values []any, keyed map[string]any, err error) {
	var keyColumn any
	if len(key) > 0 {
		keyColumn = key[0]
	}

	query, err := b.scoped(ctx, g)
	if err != nil {
		return nil, nil, err
	}
	if query.Columns == nil {
		if keyColumn == nil || stringify(keyColumn) == stringify(column) {
			query.Columns = []any{column}
		} else {
			query.Columns = []any{column, keyColumn}
		}
	}

	rows, err := query.runSelect(ctx)
	if err != nil {
		return nil, nil, err
	}
	if query.Processor != nil {
		rows = query.Processor.ProcessSelect(query, rows)
	}
	if len(rows) == 0 {
		return nil, nil, nil
	}

	// A qualified or aliased column cannot be read out of a row: the driver
	// keys the row by the column itself, so the table qualifier is stripped
	// here before the row is read.
	name := stripTableForPluck(column)
	keyName := ""
	if keyColumn != nil {
		keyName = stripTableForPluck(keyColumn)
	}
	return pluckFromColumn(rows, name, keyName)
}

// pluckFromColumn reads column (and, if key is given, key) out of every row,
// returning the values in row order and, when key was given, the same values
// keyed by it. A Record is always a map, so there is only the one function.
func pluckFromColumn(rows []Record, column, key string) ([]any, map[string]any, error) {
	values := make([]any, 0, len(rows))
	var keyed map[string]any
	if key != "" {
		keyed = make(map[string]any, len(rows))
	}
	for _, row := range rows {
		values = append(values, row[column])
		if keyed != nil {
			keyed[stringify(row[key])] = row[column]
		}
	}
	return values, keyed, nil
}

// stripTableForPluck takes the last segment of column after " as ", or after
// the dots when there is no alias.
func stripTableForPluck(column any) string {
	if column == nil {
		return ""
	}
	name := stringify(column)
	if i := aliasIndex(name); i >= 0 {
		return strings.TrimSpace(name[i+4:])
	}
	if i := strings.LastIndex(name, "."); i >= 0 {
		return name[i+1:]
	}
	return name
}

// Implode plucks column across every matching row and joins the values with
// glue.
func (b *Builder) Implode(ctx context.Context, g auth.Grant, column any, glue string) (string, error) {
	values, _, err := b.Pluck(ctx, g, column)
	if err != nil {
		return "", err
	}
	parts := make([]string, len(values))
	for i, value := range values {
		parts[i] = stringify(value)
	}
	return strings.Join(parts, glue), nil
}

// Exists reports whether the query matches at least one row.
//
// It compiles to the grammar's exists statement -- a select wrapped so the
// engine can stop at the first row -- rather than counting.
func (b *Builder) Exists(ctx context.Context, g auth.Grant) (bool, error) {
	query, err := b.scoped(ctx, g)
	if err != nil {
		return false, err
	}
	query.ApplyBeforeQueryCallbacks()
	if err := query.Err(); err != nil {
		return false, err
	}
	if query.connection == nil {
		return false, errors.New("query: the builder has no connection to run against")
	}

	rows, err := query.connection.Select(
		ctx,
		query.Grammar.CompileExists(query), query.GetBindings(), !query.UsingWritePDO())
	if err != nil {
		return false, err
	}
	if len(rows) == 0 {
		return false, nil
	}
	return truthy(rows[0]["exists"]), nil
}

// DoesntExist reports whether the query matches no rows.
func (b *Builder) DoesntExist(ctx context.Context, g auth.Grant) (bool, error) {
	exists, err := b.Exists(ctx, g)
	if err != nil {
		return false, err
	}
	return !exists, nil
}

// ExistsOr reports whether the query matches at least one row, running
// callback when it does not.
//
// The bool is the answer to the question -- did rows exist -- and the callback
// contributes only its error, because a callback that returned a value would
// have to return the same type as `true`.
func (b *Builder) ExistsOr(ctx context.Context, g auth.Grant, callback func() error) (bool, error) {
	exists, err := b.Exists(ctx, g)
	if err != nil {
		return false, err
	}
	if exists {
		return true, nil
	}
	if callback == nil {
		return false, nil
	}
	return false, callback()
}

// DoesntExistOr reports whether the query matches no rows, running callback
// when rows do exist. The bool is read as ExistsOr's is.
func (b *Builder) DoesntExistOr(ctx context.Context, g auth.Grant, callback func() error) (bool, error) {
	doesntExist, err := b.DoesntExist(ctx, g)
	if err != nil {
		return false, err
	}
	if doesntExist {
		return true, nil
	}
	if callback == nil {
		return false, nil
	}
	return false, callback()
}

// ToRawSQL returns the statement with the bindings written into it, for
// reading rather than for running.
//
// It carries an error because escaping a value needs a connection to escape
// through, and Grammar.Escape reports that rather than handing back a value
// that looks quoted and is not.
//
// It takes the Grant because what it prints has to be what would run, tenant
// clause and all. A raw SQL dump that showed a query without its tenant filter
// would teach the reader that the filter is not there.
func (b *Builder) ToRawSQL(ctx context.Context, g auth.Grant) (string, error) {
	query, err := b.scoped(ctx, g)
	if err != nil {
		return "", err
	}
	sql := query.ToSQL()
	if err := query.Err(); err != nil {
		return "", err
	}
	bindings := query.GetBindings()
	if connection, ok := query.connection.(PreparesBindings); ok {
		bindings = connection.PrepareBindings(bindings)
	}
	return substituteBindingsIntoRawSQL(query.Grammar, sql, bindings)
}

// DumpRawSQL writes the statement with its bindings substituted in.
//
// The destination is an argument, and a failed substitution is written out too
// -- a dump that printed nothing and returned would be the least helpful
// debugging tool there is.
func (b *Builder) DumpRawSQL(ctx context.Context, g auth.Grant, w io.Writer) *Builder {
	sql, err := b.ToRawSQL(ctx, g)
	if err != nil {
		fmt.Fprintln(w, err)
		return b
	}
	fmt.Fprintln(w, sql)
	return b
}

// substituteBindingsIntoRawSQL writes sql with each placeholder replaced by
// its escaped binding, in order.
//
// It walks the statement rather than replacing every "?", because a "?" inside a
// string literal is a character and not a placeholder, and Postgres has
// operators -- ?| and ?& -- that are written "??|" and "??&" for exactly this
// reason.
//
// It is a function and not a Grammar method because the Grammar interface does
// not declare one, so PostgresGrammar's override of it is out of reach here.
// What is lost is the extra escaping that override does; what is not lost is
// any statement that runs, because nothing here ever executes what it produces.
func substituteBindingsIntoRawSQL(grammar Grammar, sql string, bindings []any) (string, error) {
	escaped := make([]string, len(bindings))
	for i, binding := range bindings {
		value, err := grammar.Escape(binding, false)
		if err != nil {
			return "", err
		}
		escaped[i] = value
	}

	var out strings.Builder
	inStringLiteral := false
	for i := 0; i < len(sql); i++ {
		char := sql[i]
		pair := ""
		if i+1 < len(sql) {
			pair = sql[i : i+2]
		}

		switch {
		case pair == `\'` || pair == "''" || pair == "??":
			out.WriteString(pair)
			i++
		case char == '\'':
			out.WriteByte(char)
			inStringLiteral = !inStringLiteral
		case char == '?' && !inStringLiteral:
			if len(escaped) > 0 {
				out.WriteString(escaped[0])
				escaped = escaped[1:]
			} else {
				out.WriteByte('?')
			}
		default:
			out.WriteByte(char)
		}
	}
	return out.String(), nil
}

// Raw returns value wrapped as an Expression.
//
// It is a method on Builder for symmetry with the fluent chain; Raw is the
// package function that actually builds one.
func (b *Builder) Raw(value any) Expression { return Raw(value) }

// CastBinding returns value unchanged.
//
// A named type over a scalar is already seen by the driver as the scalar, so
// there is nothing to unwrap. It stays because it is the one place a future
// cast would go, and because a caller looking for it should find it rather
// than conclude it is missing.
func (b *Builder) CastBinding(value any) any { return value }

// CleanBindings drops the expressions from bindings, which were compiled
// into the statement and have no placeholder to fill.
func (b *Builder) CleanBindings(bindings []any) []any {
	out := make([]any, 0, len(bindings))
	for _, binding := range bindings {
		if IsExpression(binding) {
			continue
		}
		out = append(out, b.CastBinding(binding))
	}
	return out
}

// Tap hands the query to callback and returns the query.
func (b *Builder) Tap(callback func(*Builder)) *Builder {
	if callback != nil {
		callback(b)
	}
	return b
}

// Pipe hands the query to callback, and returns what callback returned, or
// the query itself if callback returned nil.
func (b *Builder) Pipe(callback func(*Builder) *Builder) *Builder {
	if callback == nil {
		return b
	}
	if out := callback(b); out != nil {
		return out
	}
	return b
}

// truthy reads the driver's answer to a yes-or-no column. The engines
// disagree: a bool, a 0 or 1, a "t", a "true" and the bytes of any of those
// all mean the same thing, and this is the one place that gets flattened.
func truthy(value any) bool {
	switch v := value.(type) {
	case nil:
		return false
	case bool:
		return v
	case int:
		return v != 0
	case int32:
		return v != 0
	case int64:
		return v != 0
	case float64:
		return v != 0
	case []byte:
		return truthyString(string(v))
	case string:
		return truthyString(v)
	default:
		return false
	}
}

func truthyString(value string) bool {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "", "0", "f", "false", "off", "no":
		return false
	default:
		return true
	}
}

// castNumber turns whatever the driver returned for an aggregate into a float.
// The second result reports whether it was a number at all, which is how a NULL
// aggregate -- min() over no rows -- stays distinguishable from a zero.
func castNumber(value any) (float64, bool) {
	switch v := value.(type) {
	case nil:
		return 0, false
	case int:
		return float64(v), true
	case int32:
		return float64(v), true
	case int64:
		return float64(v), true
	case uint64:
		return float64(v), true
	case float32:
		return float64(v), true
	case float64:
		return v, true
	case []byte:
		return parseNumber(string(v))
	case string:
		return parseNumber(v)
	case bool:
		if v {
			return 1, true
		}
		return 0, true
	default:
		return 0, false
	}
}

func parseNumber(value string) (float64, bool) {
	number, err := strconv.ParseFloat(strings.TrimSpace(value), 64)
	if err != nil {
		return 0, false
	}
	return number, true
}

// sortedKeys returns the keys of values, sorted, so that every row of a
// batch lists its columns in the same order as the first.
//
// A Go map has no order at all, so the sort is not a normalisation here but
// the only thing that makes the statement reproducible -- and the contract
// the grammar compiles against: the columns are the sorted keys, and the
// bindings arrive in that same order.
func sortedKeys(values map[string]any) []string {
	return slices.Sorted(maps.Keys(values))
}
