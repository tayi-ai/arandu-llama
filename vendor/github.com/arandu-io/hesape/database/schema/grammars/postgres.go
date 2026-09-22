package grammars

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/arandu-io/hesape/database/query"
	"github.com/arandu-io/hesape/database/schema"
)

// PostgresGrammar turns a Blueprint into Postgres DDL, and compiles the
// catalogue queries that read a Postgres schema back.
//
// What it changes about the base grammar: DDL runs inside a transaction, so a
// failed migration leaves no half-built schema; an auto-incrementing integer is
// a serial type rather than a modifier; a timestamp carries its time zone or
// explicitly does not; an index can name an operator class, which is what makes
// a full-text, spatial or vector index reachable at all; and dropping a table,
// view, type or domain is one statement for the whole list.
//
// The introspection half -- CompileTables, CompileColumns, CompileIndexes,
// CompileForeignKeys -- reads pg_catalog rather than information_schema,
// because information_schema does not describe an index at all: its name, its
// access method and the order of its columns are only in pg_index.
type PostgresGrammar struct{ *BaseGrammar }

// NewPostgresGrammar builds a PostgresGrammar for connection. DDL for this
// grammar runs inside a transaction, and its modifier order, serial types
// and fluent commands are set for Postgres.
func NewPostgresGrammar(connection schema.Connection) *PostgresGrammar {
	g := &PostgresGrammar{&BaseGrammar{
		conn:           connection,
		transactions:   true,
		modifiers:      []string{"Collate", "Nullable", "Default", "VirtualAs", "StoredAs", "GeneratedAs", "Increment"},
		serials:        []string{"bigInteger", "integer", "mediumInteger", "smallInteger", "tinyInteger"},
		fluentCommands: []string{"autoIncrementStartingValues", "comment"},
	}}
	g.self = g
	return g
}

// MaxIdentifierLength returns 63, which is NAMEDATALEN - 1.
//
// Postgres does not refuse a longer name: it truncates to this length and says
// so in a NOTICE the client library usually discards. Two index names that
// differ only after the cut therefore become one name, and the second CREATE
// INDEX fails with "already exists" -- on whichever database happens to hold
// both, which is rarely the one the migration was written on.
func (g *PostgresGrammar) MaxIdentifierLength() int { return 63 }

// CompileCreateDatabase builds on the base's SQL, appending an encoding
// clause when the connection configures a charset.
func (g *PostgresGrammar) CompileCreateDatabase(name string) (string, error) {
	sql, err := g.BaseGrammar.CompileCreateDatabase(name)
	if err != nil {
		return "", err
	}
	if charset := g.conn.GetConfig("charset"); charset != "" {
		sql += " encoding " + g.wrapValue(charset)
	}
	return sql, nil
}

// CompileSchemas builds the query that lists user schemas from
// pg_namespace, flagging which one is on the current search path.
func (g *PostgresGrammar) CompileSchemas() (string, error) {
	return "select nspname as name, nspname = any (current_schemas(false)) as \"default\" " +
		"from pg_namespace where nspname not in ('pg_catalog', 'information_schema') and nspname !~ '^pg_' " +
		"order by nspname", nil
}

// CompileTableExists builds the query that reports whether table exists in
// schemaName, or the connection's current schema when schemaName is empty.
// It matches ordinary and partitioned tables (relkind 'r' or 'p').
func (g *PostgresGrammar) CompileTableExists(schemaName, table string) string {
	from := "current_schema()"
	if schemaName != "" {
		from = g.QuoteString(schemaName)
	}
	return fmt.Sprintf(
		"select exists (select 1 from pg_class c, pg_namespace n where "+
			"n.nspname = %s and c.relname = %s and c.relnamespace = n.oid and c.relkind in ('r', 'p'))",
		from, g.QuoteString(table))
}

// CompileTables builds the query that lists ordinary and partitioned
// tables in schemas, with each table's total size and comment. Postgres
// always computes the size, so withSize is accepted but unused.
func (g *PostgresGrammar) CompileTables(schemas []string, withSize ...bool) (string, error) {
	return "select c.relname as name, n.nspname as schema, pg_total_relation_size(c.oid) as size, " +
		"obj_description(c.oid, 'pg_class') as comment from pg_class c, pg_namespace n " +
		"where c.relkind in ('r', 'p') and n.oid = c.relnamespace and " +
		g.compileSchemaWhereClause(schemas, "n.nspname") +
		" order by n.nspname, c.relname", nil
}

// CompileTypes builds the query that lists user-defined types (composite,
// enum, domain and their array forms) in schemas, excluding types owned by
// an extension.
func (g *PostgresGrammar) CompileTypes(schemas []string) (string, error) {
	return "select t.typname as name, n.nspname as schema, t.typtype as type, t.typcategory as category, " +
		"((t.typinput = 'array_in'::regproc and t.typoutput = 'array_out'::regproc) or t.typtype = 'm') as implicit " +
		"from pg_type t join pg_namespace n on n.oid = t.typnamespace " +
		"left join pg_class c on c.oid = t.typrelid " +
		"left join pg_type el on el.oid = t.typelem " +
		"left join pg_class ce on ce.oid = el.typrelid " +
		"where ((t.typrelid = 0 and (ce.relkind = 'c' or ce.relkind is null)) or c.relkind = 'c') " +
		"and not exists (select 1 from pg_depend d where d.objid in (t.oid, t.typelem) and d.deptype = 'e') and " +
		g.compileSchemaWhereClause(schemas, "n.nspname"), nil
}

// CompileViews builds the query that lists the views in schemas from
// pg_views, including each view's defining SQL.
func (g *PostgresGrammar) CompileViews(schemas []string) (string, error) {
	return "select viewname as name, schemaname as schema, definition from pg_views where " +
		g.compileSchemaWhereClause(schemas, "schemaname") +
		" order by schemaname, viewname", nil
}

// compileSchemaWhereClause builds the where-clause fragment that restricts
// column to schemas, or, when schemas is empty, to every schema except
// information_schema and the pg_* system schemas.
func (g *PostgresGrammar) compileSchemaWhereClause(schemas []string, column string) string {
	if len(schemas) > 0 {
		return column + " in (" + g.QuoteString(schemas) + ")"
	}
	return column + " <> 'information_schema' and " + column + " not like 'pg\\_%'"
}

// CompileColumns builds the query that describes table's columns in
// schemaName, or the connection's current schema when schemaName is empty:
// each column's type, collation, nullability, default expression and
// comment, in ordinal position.
func (g *PostgresGrammar) CompileColumns(schemaName, table string) (string, error) {
	from := "current_schema()"
	if schemaName != "" {
		from = g.QuoteString(schemaName)
	}
	return fmt.Sprintf(
		"select a.attname as name, t.typname as type_name, format_type(a.atttypid, a.atttypmod) as type, "+
			"(select tc.collcollate from pg_catalog.pg_collation tc where tc.oid = a.attcollation) as collation, "+
			"not a.attnotnull as nullable, "+
			"(select pg_get_expr(adbin, adrelid) from pg_attrdef where c.oid = pg_attrdef.adrelid "+
			"and pg_attrdef.adnum = a.attnum) as default, "+
			"col_description(c.oid, a.attnum) as comment "+
			"from pg_attribute a, pg_class c, pg_type t, pg_namespace n "+
			"where c.relname = %s and n.nspname = %s and a.attnum > 0 and a.attrelid = c.oid "+
			"and a.atttypid = t.oid and n.oid = c.relnamespace order by a.attnum",
		g.QuoteString(table), from), nil
}

// CompileIndexes builds the query that describes table's indexes in
// schemaName, or the connection's current schema when schemaName is empty:
// each index's name, access method, uniqueness, primary-key status and
// columns in index order. It reads pg_index rather than information_schema,
// which does not expose an index's access method or column order at all.
func (g *PostgresGrammar) CompileIndexes(schemaName, table string) (string, error) {
	from := "current_schema()"
	if schemaName != "" {
		from = g.QuoteString(schemaName)
	}
	return fmt.Sprintf(
		"select ic.relname as name, string_agg(a.attname, ',' order by indseq.ord) as columns, "+
			"am.amname as \"type\", i.indisunique as \"unique\", i.indisprimary as \"primary\" "+
			"from pg_index i join pg_class tc on tc.oid = i.indrelid "+
			"join pg_namespace tn on tn.oid = tc.relnamespace join pg_class ic on ic.oid = i.indexrelid "+
			"join pg_am am on am.oid = ic.relam "+
			"join lateral unnest(i.indkey) with ordinality as indseq(num, ord) on true "+
			"left join pg_attribute a on a.attrelid = i.indrelid and a.attnum = indseq.num "+
			"where tc.relname = %s and tn.nspname = %s "+
			"group by ic.relname, am.amname, i.indisunique, i.indisprimary",
		g.QuoteString(table), from), nil
}

// CompileForeignKeys builds the query that describes table's foreign keys
// in schemaName, or the connection's current schema when schemaName is
// empty: each constraint's name, local and foreign columns in constraint
// order, the foreign table, and its on-update and on-delete actions.
func (g *PostgresGrammar) CompileForeignKeys(schemaName, table string) (string, error) {
	from := "current_schema()"
	if schemaName != "" {
		from = g.QuoteString(schemaName)
	}
	return fmt.Sprintf(
		"select c.conname as name, string_agg(la.attname, ',' order by conseq.ord) as columns, "+
			"fn.nspname as foreign_schema, fc.relname as foreign_table, "+
			"string_agg(fa.attname, ',' order by conseq.ord) as foreign_columns, "+
			"c.confupdtype as on_update, c.confdeltype as on_delete "+
			"from pg_constraint c join pg_class tc on c.conrelid = tc.oid "+
			"join pg_namespace tn on tn.oid = tc.relnamespace join pg_class fc on c.confrelid = fc.oid "+
			"join pg_namespace fn on fn.oid = fc.relnamespace "+
			"join lateral unnest(c.conkey) with ordinality as conseq(num, ord) on true "+
			"join pg_attribute la on la.attrelid = c.conrelid and la.attnum = conseq.num "+
			"join pg_attribute fa on fa.attrelid = c.confrelid and fa.attnum = c.confkey[conseq.ord] "+
			"where c.contype = 'f' and tc.relname = %s and tn.nspname = %s "+
			"group by c.conname, fn.nspname, fc.relname, c.confupdtype, c.confdeltype",
		g.QuoteString(table), from), nil
}

// CompileCreate builds the SQL for a CREATE TABLE statement, or CREATE
// TEMPORARY TABLE when blueprint is temporary.
func (g *PostgresGrammar) CompileCreate(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	columns, err := g.getColumns(blueprint)
	if err != nil {
		return nil, err
	}
	create := "create"
	if blueprint.GetTemporary() {
		create = "create temporary"
	}
	return one(fmt.Sprintf("%s table %s (%s)", create, g.WrapTable(blueprint), strings.Join(columns, ", "))), nil
}

// CompileAdd builds the SQL that adds command.Column to blueprint's table.
func (g *PostgresGrammar) CompileAdd(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	definition, err := g.getColumn(blueprint, command.Column)
	if err != nil {
		return nil, err
	}
	return one("alter table " + g.WrapTable(blueprint) + " add column " + definition), nil
}

// CompileAutoIncrementStartingValues builds the setval call that sets
// command.Column's sequence to start at its configured starting value. It
// compiles nothing when the column is not auto-incrementing or the
// starting value is zero.
func (g *PostgresGrammar) CompileAutoIncrementStartingValues(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	value := startingValue(command.Column)
	if !command.Column.GetAutoIncrement() || value == 0 {
		return nil, nil
	}
	return one(fmt.Sprintf("select setval(pg_get_serial_sequence(%s, %s), %d, false)",
		g.QuoteString(g.WrapTable(blueprint)),
		g.QuoteString(command.Column.GetName()),
		value)), nil
}

// CompileChange builds a single ALTER TABLE with one "alter column" clause
// per change: type and collation combine into one clause, because Postgres
// requires COLLATE to ride along with a type change rather than stand on
// its own, and every other modifier the grammar declares contributes its
// own clause.
func (g *PostgresGrammar) CompileChange(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	column := command.Column

	typ, err := g.typeOf(column)
	if err != nil {
		return nil, err
	}

	collate, err := g.modify("Collate", blueprint, column)
	if err != nil {
		return nil, err
	}
	changes := []string{"type " + typ + strings.Join(collate, "")}

	for _, modifier := range g.modifiers {
		if modifier == "Collate" {
			continue
		}
		fragments, err := g.modify(modifier, blueprint, column)
		if err != nil {
			return nil, err
		}
		changes = append(changes, fragments...)
	}

	return one(fmt.Sprintf("alter table %s %s",
		g.WrapTable(blueprint),
		strings.Join(g.PrefixArray("alter column "+g.Wrap(column), changes), ", "))), nil
}

// CompilePrimary builds the SQL that adds a primary key over
// command.Columns.
func (g *PostgresGrammar) CompilePrimary(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	return one("alter table " + g.WrapTable(blueprint) + " add primary key (" + g.Columnize(command.Columns) + ")"), nil
}

// CompileUnique builds the SQL that adds a unique constraint over
// command.Columns. When command.Online or command.Algorithm asks for a
// concurrent or non-default build, it first creates the index as its own
// statement -- Postgres cannot add a unique constraint concurrently in one
// step -- and then attaches the constraint to that index; otherwise the
// constraint is added directly. Deferrability and a nulls-distinct clause
// are appended when the command sets them.
func (g *PostgresGrammar) CompileUnique(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	unique := "unique"
	if command.NullsNotDistinct != nil {
		if *command.NullsNotDistinct {
			unique += " nulls not distinct"
		} else {
			unique += " nulls distinct"
		}
	}

	var statements []string
	var sql string

	if command.Online || command.Algorithm != "" {
		concurrently := ""
		if command.Online {
			concurrently = "concurrently "
		}
		using := ""
		if command.Algorithm != "" {
			using = " using " + command.Algorithm
		}
		statements = append(statements, fmt.Sprintf("create unique index %s%s on %s%s (%s)",
			concurrently, g.Wrap(command.Index), g.WrapTable(blueprint), using, g.Columnize(command.Columns)))

		sql = fmt.Sprintf("alter table %s add constraint %s unique using index %s",
			g.WrapTable(blueprint), g.Wrap(command.Index), g.Wrap(command.Index))
	} else {
		sql = fmt.Sprintf("alter table %s add constraint %s %s (%s)",
			g.WrapTable(blueprint), g.Wrap(command.Index), unique, g.Columnize(command.Columns))
	}

	sql += deferrability(command)
	return append(statements, sql), nil
}

// CompileIndex builds a CREATE INDEX over command.Columns, adding
// CONCURRENTLY when command.Online is set and a USING clause when
// command.Algorithm names one.
func (g *PostgresGrammar) CompileIndex(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	concurrently := ""
	if command.Online {
		concurrently = "concurrently "
	}
	using := ""
	if command.Algorithm != "" {
		using = " using " + command.Algorithm
	}
	return one(fmt.Sprintf("create index %s%s on %s%s (%s)",
		concurrently, g.Wrap(command.Index), g.WrapTable(blueprint), using, g.Columnize(command.Columns))), nil
}

// CompileFullText builds a GIN index over the to_tsvector of
// command.Columns concatenated with "||", using command.Language or
// "english" when none is set, and CONCURRENTLY when command.Online is set.
func (g *PostgresGrammar) CompileFullText(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	language := command.Language
	if language == "" {
		language = "english"
	}

	columns := make([]string, len(command.Columns))
	for i, column := range command.Columns {
		columns[i] = "to_tsvector(" + g.QuoteString(language) + ", " + g.Wrap(column) + ")"
	}

	concurrently := ""
	if command.Online {
		concurrently = "concurrently "
	}

	return one(fmt.Sprintf("create index %s%s on %s using gin ((%s))",
		concurrently, g.Wrap(command.Index), g.WrapTable(blueprint), strings.Join(columns, " || "))), nil
}

// CompileSpatialIndex builds a GiST index over command.Columns, naming
// command.OperatorClass on each column when the command sets one.
func (g *PostgresGrammar) CompileSpatialIndex(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	command.Algorithm = "gist"
	if command.OperatorClass != "" {
		return g.compileIndexWithOperatorClass(blueprint, command), nil
	}
	return g.CompileIndex(blueprint, command)
}

// CompileVectorIndex builds an index over command.Columns using
// command.Algorithm, naming command.OperatorClass on each column when set
// -- the operator class is how a vector index selects its distance
// function.
func (g *PostgresGrammar) CompileVectorIndex(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	return g.compileIndexWithOperatorClass(blueprint, command), nil
}

// compileIndexWithOperatorClass builds a CREATE INDEX over
// command.Columns, appending command.OperatorClass after each column when
// set, with CONCURRENTLY and a USING clause as CompileIndex does. It is the
// shared body behind CompileSpatialIndex and CompileVectorIndex.
func (g *PostgresGrammar) compileIndexWithOperatorClass(blueprint *schema.Blueprint, command *schema.Command) []string {
	columns := make([]string, len(command.Columns))
	for i, column := range command.Columns {
		columns[i] = g.Wrap(column)
		if command.OperatorClass != "" {
			columns[i] += " " + command.OperatorClass
		}
	}

	concurrently := ""
	if command.Online {
		concurrently = "concurrently "
	}
	using := ""
	if command.Algorithm != "" {
		using = " using " + command.Algorithm
	}

	return one(fmt.Sprintf("create index %s%s on %s%s (%s)",
		concurrently, g.Wrap(command.Index), g.WrapTable(blueprint), using, strings.Join(columns, ", ")))
}

// CompileForeign extends the base foreign key SQL with deferrability and a
// NOT VALID clause when command.NotValid is set, so a large table can add
// the constraint without validating existing rows immediately.
func (g *PostgresGrammar) CompileForeign(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	statements, err := g.BaseGrammar.CompileForeign(blueprint, command)
	if err != nil {
		return nil, err
	}
	statements[0] += deferrability(command)
	if command.NotValid != nil && *command.NotValid {
		statements[0] += " not valid"
	}
	return statements, nil
}

// CompileDrop builds the SQL that drops blueprint's table.
func (g *PostgresGrammar) CompileDrop(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	return one("drop table " + g.WrapTable(blueprint)), nil
}

// CompileDropIfExists builds the SQL that drops blueprint's table if it
// exists.
func (g *PostgresGrammar) CompileDropIfExists(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	return one("drop table if exists " + g.WrapTable(blueprint)), nil
}

// CompileDropAllTables builds one DROP TABLE ... CASCADE statement naming
// every table in tables.
func (g *PostgresGrammar) CompileDropAllTables(tables []string) (string, error) {
	return "drop table " + strings.Join(g.EscapeNames(tables), ", ") + " cascade", nil
}

// CompileDropAllViews builds one DROP VIEW ... CASCADE statement naming
// every view in views.
func (g *PostgresGrammar) CompileDropAllViews(views []string) (string, error) {
	return "drop view " + strings.Join(g.EscapeNames(views), ", ") + " cascade", nil
}

// CompileDropAllTypes builds one DROP TYPE ... CASCADE statement naming
// every type in types.
func (g *PostgresGrammar) CompileDropAllTypes(types []string) (string, error) {
	return "drop type " + strings.Join(g.EscapeNames(types), ", ") + " cascade", nil
}

// CompileDropAllDomains builds one DROP DOMAIN ... CASCADE statement naming
// every domain in domains.
func (g *PostgresGrammar) CompileDropAllDomains(domains []string) (string, error) {
	return "drop domain " + strings.Join(g.EscapeNames(domains), ", ") + " cascade", nil
}

// CompileDropColumn builds the SQL that drops every column in
// command.Columns in a single ALTER TABLE.
func (g *PostgresGrammar) CompileDropColumn(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	columns := g.PrefixArray("drop column", g.WrapArray(command.Columns))
	return one("alter table " + g.WrapTable(blueprint) + " " + strings.Join(columns, ", ")), nil
}

// CompileDropPrimary builds the SQL that drops the primary key, naming the
// constraint by Postgres' default convention: <table>_pkey.
func (g *PostgresGrammar) CompileDropPrimary(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	_, table, err := schema.ParseSchemaAndTable(blueprint.GetTable())
	if err != nil {
		return nil, err
	}
	index := g.Wrap(g.conn.GetTablePrefix() + table + "_pkey")
	return one("alter table " + g.WrapTable(blueprint) + " drop constraint " + index), nil
}

// CompileDropUnique builds the SQL that drops the unique constraint named
// command.Index.
func (g *PostgresGrammar) CompileDropUnique(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	return one("alter table " + g.WrapTable(blueprint) + " drop constraint " + g.Wrap(command.Index)), nil
}

// CompileDropIndex builds the SQL that drops the index named command.Index.
func (g *PostgresGrammar) CompileDropIndex(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	return one("drop index " + g.Wrap(command.Index)), nil
}

// CompileDropFullText drops a full-text index the same way as any other
// index: Postgres has no separate syntax for it.
func (g *PostgresGrammar) CompileDropFullText(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	return g.CompileDropIndex(blueprint, command)
}

// CompileDropSpatialIndex drops a spatial index the same way as any other
// index: Postgres has no separate syntax for it.
func (g *PostgresGrammar) CompileDropSpatialIndex(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	return g.CompileDropIndex(blueprint, command)
}

// CompileDropForeign builds the SQL that drops the foreign key constraint
// named command.Index.
func (g *PostgresGrammar) CompileDropForeign(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	return one("alter table " + g.WrapTable(blueprint) + " drop constraint " + g.Wrap(command.Index)), nil
}

// CompileRename builds the SQL that renames blueprint's table to
// command.To.
func (g *PostgresGrammar) CompileRename(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	return one("alter table " + g.WrapTable(blueprint) + " rename to " + g.WrapTable(command.To)), nil
}

// CompileRenameIndex builds the SQL that renames the index command.From to
// command.To.
func (g *PostgresGrammar) CompileRenameIndex(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	return one("alter index " + g.Wrap(command.From) + " rename to " + g.Wrap(command.To)), nil
}

// CompileEnableForeignKeyConstraints builds the SQL that makes every
// deferred constraint check take effect immediately again.
func (g *PostgresGrammar) CompileEnableForeignKeyConstraints() (string, error) {
	return "SET CONSTRAINTS ALL IMMEDIATE;", nil
}

// CompileDisableForeignKeyConstraints builds the SQL that defers every
// constraint check to the end of the transaction.
func (g *PostgresGrammar) CompileDisableForeignKeyConstraints() (string, error) {
	return "SET CONSTRAINTS ALL DEFERRED;", nil
}

// CompileComment builds a COMMENT ON COLUMN statement for command.Column.
// It compiles nothing for a new column with no comment; a changed column
// with no comment clears it by setting the comment to NULL.
func (g *PostgresGrammar) CompileComment(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	comment := command.Column.GetComment()
	if comment == nil && !command.Column.GetChange() {
		return nil, nil
	}
	value := "NULL"
	if comment != nil {
		value = quote(*comment)
	}
	return one(fmt.Sprintf("comment on column %s.%s is %s",
		g.WrapTable(blueprint), g.Wrap(command.Column.GetName()), value)), nil
}

// CompileTableComment builds a COMMENT ON TABLE statement setting
// blueprint's table comment to command.Comment.
func (g *PostgresGrammar) CompileTableComment(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	return one(fmt.Sprintf("comment on table %s is %s",
		g.WrapTable(blueprint), quote(command.Comment))), nil
}

// EscapeNames quotes every name in names, segment by segment, so a
// schema-qualified "schema.table" is quoted as "schema"."table" rather
// than as one identifier.
func (g *PostgresGrammar) EscapeNames(names []string) []string {
	out := make([]string, len(names))
	for i, name := range names {
		segments := strings.Split(name, ".")
		for j, segment := range segments {
			segments[j] = g.wrapValue(segment)
		}
		out[i] = strings.Join(segments, ".")
	}
	return out
}

// typeOf returns the Postgres SQL type for column, dispatching on its
// declared type. It returns an error wrapping schema.ErrUnsupported for a
// type this driver does not implement, such as "set" or "computed".
func (g *PostgresGrammar) typeOf(column *schema.ColumnDefinition) (string, error) {
	switch column.GetType() {
	case "char":
		if length := intOr(column.GetLength(), 0); length != 0 {
			return fmt.Sprintf("char(%d)", length), nil
		}
		return "char", nil
	case "string":
		if length := intOr(column.GetLength(), 0); length != 0 {
			return fmt.Sprintf("varchar(%d)", length), nil
		}
		return "varchar", nil
	case "tinyText":
		return "varchar(255)", nil
	case "text", "mediumText", "longText":
		return "text", nil
	case "integer", "mediumInteger":
		return g.typeSerial(column, "serial", "integer"), nil
	case "bigInteger":
		return g.typeSerial(column, "bigserial", "bigint"), nil
	case "smallInteger", "tinyInteger":
		return g.typeSerial(column, "smallserial", "smallint"), nil
	case "float":
		if p := intOr(column.GetPrecision(), 0); p != 0 {
			return fmt.Sprintf("float(%d)", p), nil
		}
		return "float", nil
	case "double":
		return "double precision", nil
	case "real":
		return "real", nil
	case "decimal":
		return fmt.Sprintf("decimal(%d, %d)", column.GetTotal(), column.GetPlaces()), nil
	case "boolean":
		return "boolean", nil
	case "enum":
		return fmt.Sprintf(`varchar(255) check ("%s" in (%s))`,
			column.GetName(), g.QuoteString(column.GetAllowed())), nil
	case "set":
		return "", fmt.Errorf("%w: the set type", schema.ErrUnsupported)
	case "json":
		return "json", nil
	case "jsonb":
		return "jsonb", nil
	case "date":
		if column.GetUseCurrent() {
			column.Default(query.Raw("CURRENT_DATE"))
		}
		return "date", nil
	case "dateTime", "timestamp":
		return g.typeTimestamp(column, " without time zone"), nil
	case "dateTimeTz", "timestampTz":
		return g.typeTimestamp(column, " with time zone"), nil
	case "time":
		return "time" + precisionSuffix(column) + " without time zone", nil
	case "timeTz":
		return "time" + precisionSuffix(column) + " with time zone", nil
	case "year":
		if column.GetUseCurrent() {
			column.Default(query.Raw("EXTRACT(YEAR FROM CURRENT_DATE)"))
		}
		return g.typeSerial(column, "serial", "integer"), nil
	case "binary":
		return "bytea", nil
	case "uuid":
		return "uuid", nil
	case "ipAddress":
		return "inet", nil
	case "macAddress":
		return "macaddr", nil
	case "geometry":
		return g.typeSpatial("geometry", column), nil
	case "geography":
		return g.typeSpatial("geography", column), nil
	case "vector":
		if d := intOr(column.GetDimensions(), 0); d != 0 {
			return fmt.Sprintf("vector(%d)", d), nil
		}
		return "vector", nil
	case "computed":
		return "", fmt.Errorf("%w: the computed type", schema.ErrUnsupported)
	case "raw":
		return typeRaw(column), nil
	default:
		return "", fmt.Errorf("%w: the column type %q", schema.ErrUnsupported, column.GetType())
	}
}

// typeSerial returns serial when column is auto-incrementing, has no
// identity clause, and is not being changed; it returns plain otherwise.
// The caller supplies the width-specific pair, e.g. "bigserial"/"bigint".
func (g *PostgresGrammar) typeSerial(column *schema.ColumnDefinition, serial, plain string) string {
	if column.GetAutoIncrement() && column.GetGeneratedAs() == nil && !column.GetChange() {
		return serial
	}
	return plain
}

// typeTimestamp returns "timestamp" with an optional precision suffix,
// followed by zone (" with time zone" or " without time zone"). It sets
// CURRENT_TIMESTAMP as column's default when column.GetUseCurrent() is set.
func (g *PostgresGrammar) typeTimestamp(column *schema.ColumnDefinition, zone string) string {
	if column.GetUseCurrent() {
		column.Default(query.Raw("CURRENT_TIMESTAMP"))
	}
	return "timestamp" + precisionSuffix(column) + zone
}

// typeSpatial returns keyword alone, or keyword(subtype[,srid]) when
// column declares a subtype (e.g. "Point") and, optionally, an SRID.
// keyword is "geometry" or "geography", set by the two typeOf cases that
// call it.
func (g *PostgresGrammar) typeSpatial(keyword string, column *schema.ColumnDefinition) string {
	if column.GetSubtype() == "" {
		return keyword
	}
	srid := ""
	if column.GetSRID() != 0 {
		srid = "," + strconv.Itoa(column.GetSRID())
	}
	return fmt.Sprintf("%s(%s%s)", keyword, strings.ToLower(column.GetSubtype()), srid)
}

// modify implements dialect.modify for Postgres, dispatching modifier
// ("Collate", "Nullable", "Default", "VirtualAs", "StoredAs", "GeneratedAs"
// or "Increment") to the code that produces its SQL fragment for column. It
// returns no fragment when the column has nothing for that modifier to add.
func (g *PostgresGrammar) modify(modifier string, blueprint *schema.Blueprint, column *schema.ColumnDefinition) ([]string, error) {
	switch modifier {
	case "Collate":
		if column.GetCollation() != "" {
			return one(" collate " + g.wrapValue(column.GetCollation())), nil
		}
	case "Nullable":
		if column.GetChange() {
			if isNullable(column) {
				return one("drop not null"), nil
			}
			return one("set not null"), nil
		}
		if isNullable(column) {
			return one(" null"), nil
		}
		return one(" not null"), nil
	case "Default":
		if column.GetChange() {
			if column.GetAutoIncrement() && column.GetGeneratedAs() == nil {
				return nil, nil
			}
			if column.GetDefault() == nil {
				return one("drop default"), nil
			}
			return one("set default " + g.GetDefaultValue(column.GetDefault())), nil
		}
		if column.GetDefault() != nil {
			return one(" default " + g.GetDefaultValue(column.GetDefault())), nil
		}
	case "VirtualAs":
		return g.modifyGenerated(column, column.HasVirtualAs(), column.GetVirtualAs(), " virtual")
	case "StoredAs":
		return g.modifyGenerated(column, column.HasStoredAs(), column.GetStoredAs(), " stored")
	case "GeneratedAs":
		return g.modifyGeneratedAs(column), nil
	case "Increment":
		if !column.GetChange() &&
			!g.hasCommand(blueprint, "primary") &&
			(g.isSerial(column) || column.GetGeneratedAs() != nil) &&
			column.GetAutoIncrement() {
			return one(" primary key"), nil
		}
	}
	return nil, nil
}

// modifyGenerated is the shared body behind the VirtualAs and StoredAs
// modifiers, which differ only in keyword (" virtual" or " stored") and in
// which of column's generation fields wasSet and expression are drawn from.
//
// On a changed column, wasSet false leaves the generation expression alone,
// wasSet true with a nil expression drops it, and wasSet true with a
// non-nil expression is refused: Postgres cannot rewrite a generated
// column's expression in place. On a new column, a non-nil expression is
// rendered as "generated always as (...)" followed by keyword.
func (g *PostgresGrammar) modifyGenerated(column *schema.ColumnDefinition, wasSet bool, expression any, keyword string) ([]string, error) {
	if column.GetChange() {
		if !wasSet {
			return nil, nil
		}
		if expression == nil {
			return one("drop expression if exists"), nil
		}
		return nil, fmt.Errorf("%w: modifying generated columns", schema.ErrUnsupported)
	}
	if expression != nil {
		return one(" generated always as (" + stringOf(expression) + ")" + keyword), nil
	}
	return nil, nil
}

// modifyGeneratedAs builds the "generated always/by default as identity"
// clause when column.GetGeneratedAs() is set, with any identity sequence
// options in parentheses.
//
// On a new column it returns that clause alone, or nothing when the column
// has no identity. On a changed column it drops any existing identity
// first and adds the new clause after, except when the column is a plain
// auto-increment column gaining no identity clause, where it does neither.
func (g *PostgresGrammar) modifyGeneratedAs(column *schema.ColumnDefinition) []string {
	sql := ""
	if generated := column.GetGeneratedAs(); generated != nil {
		kind := "by default"
		if column.GetAlways() {
			kind = "always"
		}
		options := ""
		if text, ok := generated.(string); ok && text != "" {
			options = " (" + text + ")"
		}
		sql = fmt.Sprintf(" generated %s as identity%s", kind, options)
	}

	if !column.GetChange() {
		if sql == "" {
			return nil
		}
		return one(sql)
	}

	var changes []string
	if !(column.GetAutoIncrement() && sql == "") {
		changes = append(changes, "drop identity if exists")
	}
	if sql != "" {
		changes = append(changes, "add "+sql)
	}
	return changes
}

// deferrability builds the shared deferrable/initially-immediate tail that
// CompileUnique and CompileForeign both append, from command.Deferrable and
// command.InitiallyImmediate. It returns an empty string when
// command.Deferrable is unset, and only considers InitiallyImmediate when
// the constraint is deferrable.
func deferrability(command *schema.Command) string {
	sql := ""
	if command.Deferrable != nil {
		if *command.Deferrable {
			sql += " deferrable"
		} else {
			sql += " not deferrable"
		}
	}
	if command.Deferrable != nil && *command.Deferrable && command.InitiallyImmediate != nil {
		if *command.InitiallyImmediate {
			sql += " initially immediate"
		} else {
			sql += " initially deferred"
		}
	}
	return sql
}

// precisionSuffix returns "(n)" for the time types' optional precision,
// where n is column's precision. A precision of zero still renders, since
// the check is for a set precision, not a nonzero one.
func precisionSuffix(column *schema.ColumnDefinition) string {
	if precision := column.GetPrecision(); precision != nil {
		return "(" + strconv.Itoa(*precision) + ")"
	}
	return ""
}
