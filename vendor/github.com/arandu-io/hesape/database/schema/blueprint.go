package schema

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"

	"github.com/arandu-io/hesape/database/query"
)

// Blueprint is the value a migration writes against: the table under
// construction, its columns and the commands that will be run for it. It builds
// no SQL itself -- ToSQL hands each command to the Grammar -- and Builder is the
// half that executes.
//
// Neither half asks for an authorization credential, because DDL has nothing to
// offer one: a table has no tenant column to scope by, and a migration step has
// no subject to attribute a statement to.
type Blueprint struct {
	connection Connection
	grammar    Grammar

	table    string
	columns  []*ColumnDefinition
	commands []*Command

	engine    string
	charset   string
	collation string
	temporary bool
	after     string

	state *BlueprintState
}

// NewBlueprint creates a Blueprint for the given table, invoking callback
// with it before returning, if callback is not nil.
func NewBlueprint(connection Connection, table string, callback func(*Blueprint)) *Blueprint {
	b := &Blueprint{
		connection: connection,
		grammar:    connection.GetSchemaGrammar(),
		table:      table,
	}
	if callback != nil {
		callback(b)
	}
	return b
}

// Build runs the blueprint against the database.
func (b *Blueprint) Build(ctx context.Context) error {
	statements, err := b.ToSQL(ctx)
	if err != nil {
		return err
	}
	for _, statement := range statements {
		if err := b.connection.Statement(ctx, statement); err != nil {
			return err
		}
	}
	return nil
}

// ToSQL compiles the blueprint's commands into statements.
//
// It takes a context because on SQLite a blueprint that alters a table has to
// read the table's current shape from the server before it can rewrite it, and
// a read is a read. Every other driver, and SQLite creating a table rather than
// altering one, touches nothing -- but the signature cannot say "sometimes", so
// it says what the worst case does.
func (b *Blueprint) ToSQL(ctx context.Context) ([]string, error) {
	if err := b.addImpliedCommands(ctx); err != nil {
		return nil, err
	}

	var statements []string

	for _, command := range b.commands {
		if command.ShouldBeSkipped {
			continue
		}
		if err := b.checkIdentifiers(command); err != nil {
			return nil, err
		}
		if b.state != nil {
			b.state.Update(command)
		}
		sql, err := compileCommand(b.grammar, b, command)
		if err != nil {
			return nil, err
		}
		statements = append(statements, sql...)
	}

	return statements, nil
}

// addImpliedCommands expands each column's fluent index and command
// shorthand and, when altering an existing table, turns pending column
// definitions into add or change commands.
func (b *Blueprint) addImpliedCommands(ctx context.Context) error {
	b.addFluentIndexes()
	b.AddFluentCommands()

	if b.Creating() {
		return nil
	}

	for _, command := range b.commands {
		if !command.isColumnDefinition {
			continue
		}
		command.isColumnDefinition = false
		if command.Column.change {
			command.Name = "change"
		} else {
			command.Name = "add"
		}
	}

	return b.AddAlterCommands(ctx)
}

// addFluentIndexes turns each column's inline index markers (Primary,
// Unique, Index, FullText, SpatialIndex, VectorIndex) into index commands,
// clearing the marker once consumed.
func (b *Blueprint) addFluentIndexes() {
	isMySQL := b.connection.GetDriverName() == "mysql" || b.connection.GetDriverName() == "mariadb"

	for _, column := range b.columns {
		for _, index := range []string{"primary", "unique", "index", "fulltext", "spatialIndex", "vectorIndex"} {
			marker := column.marker(index)

			// A column being changed into an auto increment column needs no
			// primary key command on MySQL: the column definition carries it.
			if index == "primary" && column.autoIncrement && column.change && isMySQL {
				break
			}

			if marker == nil {
				continue
			}

			if on, ok := marker.(bool); ok {
				if on {
					b.fluentIndex(b.indexMethod(index, column), column.name, "")
					column.setMarker(index, nil)
					break
				}
				// A false marker on a changed column drops the index the
				// conventional name would have produced.
				if column.change {
					b.dropFluentIndex(index, column.name)
					column.setMarker(index, nil)
					break
				}
				continue
			}

			if name, ok := marker.(string); ok {
				b.fluentIndex(b.indexMethod(index, column), column.name, name)
				column.setMarker(index, nil)
				break
			}
		}
	}
}

// indexMethod resolves a plain "index" marker on a vector-typed column to
// "vectorIndex", leaving every other marker unchanged.
func (b *Blueprint) indexMethod(index string, column *ColumnDefinition) string {
	if index == "index" && column.typ == "vector" {
		return "vectorIndex"
	}
	return index
}

func (b *Blueprint) fluentIndex(method, column, name string) {
	switch method {
	case "primary":
		b.Primary(column, name)
	case "unique":
		b.Unique(column, name)
	case "index":
		b.Index(column, name)
	case "fulltext":
		b.FullText(column, name)
	case "spatialIndex":
		b.SpatialIndex(column, name)
	case "vectorIndex":
		b.VectorIndex(column, name)
	}
}

func (b *Blueprint) dropFluentIndex(index, column string) {
	switch index {
	case "primary":
		b.DropPrimary([]string{column})
	case "unique":
		b.DropUnique([]string{column})
	case "index":
		b.DropIndex([]string{column})
	case "fulltext":
		b.DropFullText([]string{column})
	case "spatialIndex":
		b.DropSpatialIndex([]string{column})
	}
}

// AddFluentCommands queues one command per column for every fluent command
// name the grammar declares.
func (b *Blueprint) AddFluentCommands() {
	for _, column := range b.columns {
		for _, name := range b.grammar.GetFluentCommands() {
			command := NewCommand(name)
			command.Column = column
			b.commands = append(b.commands, command)
		}
	}
}

// AddAlterCommands inserts an "alter" command around every run of commands
// the grammar marks as alterable, and builds the blueprint's BlueprintState
// if any alter command is present.
//
// The test is whether the grammar declares any alter command at all: SQLite
// is the only driver whose GetAlterCommands returns anything, because it is
// the only one that rebuilds the table.
//
// It takes a context because a blueprint with an alter command reads the
// table's current shape from the server to build its BlueprintState.
func (b *Blueprint) AddAlterCommands(ctx context.Context) error {
	alterCommands := b.grammar.GetAlterCommands()
	if len(alterCommands) == 0 {
		return nil
	}

	var commands []*Command
	lastCommandWasAlter, hasAlterCommand := false, false

	for _, command := range b.commands {
		if contains(alterCommands, command.Name) {
			hasAlterCommand = true
			lastCommandWasAlter = true
		} else if lastCommandWasAlter {
			commands = append(commands, NewCommand("alter"))
			lastCommandWasAlter = false
		}
		commands = append(commands, command)
	}

	if lastCommandWasAlter {
		commands = append(commands, NewCommand("alter"))
	}

	if hasAlterCommand {
		state, err := NewBlueprintState(ctx, b, b.connection)
		if err != nil {
			return err
		}
		b.state = state
	}

	b.commands = commands
	return nil
}

// Creating reports whether the blueprint has a create command, rather than
// only alter commands.
func (b *Blueprint) Creating() bool {
	for _, command := range b.commands {
		if !command.isColumnDefinition && command.Name == "create" {
			return true
		}
	}
	return false
}

// Create queues the command that creates the table.
func (b *Blueprint) Create() *Command { return b.addCommand("create") }

// Engine sets the storage engine to use when creating the table.
func (b *Blueprint) Engine(engine string) { b.engine = engine }

// GetEngine returns the storage engine set for the table.
func (b *Blueprint) GetEngine() string { return b.engine }

// InnoDb sets the storage engine to InnoDB.
func (b *Blueprint) InnoDb() { b.Engine("InnoDB") }

// Charset sets the character set to use when creating the table.
func (b *Blueprint) Charset(charset string) { b.charset = charset }

// GetCharset returns the character set set for the table.
func (b *Blueprint) GetCharset() string { return b.charset }

// Collation sets the collation to use when creating the table.
func (b *Blueprint) Collation(collation string) { b.collation = collation }

// GetCollation returns the collation set for the table.
func (b *Blueprint) GetCollation() string { return b.collation }

// Temporary marks the table as temporary.
func (b *Blueprint) Temporary() { b.temporary = true }

// GetTemporary reports whether the table is marked temporary.
func (b *Blueprint) GetTemporary() bool { return b.temporary }

// Drop queues the command that drops the table.
func (b *Blueprint) Drop() *Command { return b.addCommand("drop") }

// DropIfExists queues the command that drops the table if it exists.
func (b *Blueprint) DropIfExists() *Command { return b.addCommand("dropIfExists") }

// DropColumn queues the command that drops the given columns.
func (b *Blueprint) DropColumn(columns ...string) *Command {
	command := b.addCommand("dropColumn")
	command.Columns = toAnyList(columns)
	return command
}

// RenameColumn queues the command that renames a column.
func (b *Blueprint) RenameColumn(from, to string) *Command {
	command := b.addCommand("renameColumn")
	command.From, command.To = from, to
	return command
}

// DropPrimary queues the command that drops a primary key. The argument is
// the index name, or the columns it was built from.
func (b *Blueprint) DropPrimary(index any) *Command {
	return b.dropIndexCommand("dropPrimary", "primary", index)
}

// DropUnique queues the command that drops a unique index.
func (b *Blueprint) DropUnique(index any) *Command {
	return b.dropIndexCommand("dropUnique", "unique", index)
}

// DropIndex queues the command that drops an index.
func (b *Blueprint) DropIndex(index any) *Command {
	return b.dropIndexCommand("dropIndex", "index", index)
}

// DropFullText queues the command that drops a full-text index.
func (b *Blueprint) DropFullText(index any) *Command {
	return b.dropIndexCommand("dropFullText", "fulltext", index)
}

// DropSpatialIndex queues the command that drops a spatial index.
func (b *Blueprint) DropSpatialIndex(index any) *Command {
	return b.dropIndexCommand("dropSpatialIndex", "spatialIndex", index)
}

// DropForeign queues the command that drops a foreign key.
func (b *Blueprint) DropForeign(index any) *Command {
	return b.dropIndexCommand("dropForeign", "foreign", index)
}

// DropConstrainedForeignID drops both the foreign key and the column that
// carried it.
func (b *Blueprint) DropConstrainedForeignID(column string) *Command {
	b.DropForeign([]string{column})
	return b.DropColumn(column)
}

// DropForeignIDFor drops the column holding a model's foreign key. An empty
// column defaults to the model's own foreign key name.
func (b *Blueprint) DropForeignIDFor(model Model, column string) *Command {
	if column == "" {
		column = model.GetForeignKey()
	}
	return b.DropColumn(column)
}

// DropConstrainedForeignIDFor drops both the foreign key and the column
// holding a model's foreign key.
func (b *Blueprint) DropConstrainedForeignIDFor(model Model, column string) *Command {
	if column == "" {
		column = model.GetForeignKey()
	}
	return b.DropConstrainedForeignID(column)
}

// RenameIndex queues the command that renames an index.
func (b *Blueprint) RenameIndex(from, to string) *Command {
	command := b.addCommand("renameIndex")
	command.From, command.To = from, to
	return command
}

// DropTimestamps drops the created_at and updated_at columns.
func (b *Blueprint) DropTimestamps() { b.DropColumn("created_at", "updated_at") }

// DropTimestampsTz drops the created_at and updated_at columns.
func (b *Blueprint) DropTimestampsTz() { b.DropTimestamps() }

// DropSoftDeletes drops the soft-delete column. An empty column defaults to
// deleted_at.
func (b *Blueprint) DropSoftDeletes(column string) {
	if column == "" {
		column = "deleted_at"
	}
	b.DropColumn(column)
}

// DropSoftDeletesTz drops the soft-delete column.
func (b *Blueprint) DropSoftDeletesTz(column string) { b.DropSoftDeletes(column) }

// DropRememberToken drops the remember_token column.
func (b *Blueprint) DropRememberToken() { b.DropColumn("remember_token") }

// DropMorphs drops the index over a polymorphic relation's type and id
// columns, then drops both columns.
func (b *Blueprint) DropMorphs(name string, indexName ...string) {
	index := arg(indexName, 0)
	if index == "" {
		index = b.createIndexName("index", []any{name + "_type", name + "_id"})
	}
	b.DropIndex(index)
	b.DropColumn(name+"_type", name+"_id")
}

// Rename queues the command that renames the table.
func (b *Blueprint) Rename(to string) *Command {
	command := b.addCommand("rename")
	command.To = to
	return command
}

// Primary adds a primary key command over the given columns. The optional
// arguments are the index name and the algorithm.
func (b *Blueprint) Primary(columns any, args ...string) *IndexDefinition {
	return b.indexCommand("primary", columns, arg(args, 0), arg(args, 1), "")
}

// Unique adds a unique index command over the given columns.
func (b *Blueprint) Unique(columns any, args ...string) *IndexDefinition {
	return b.indexCommand("unique", columns, arg(args, 0), arg(args, 1), "")
}

// Index adds an index command over the given columns.
func (b *Blueprint) Index(columns any, args ...string) *IndexDefinition {
	return b.indexCommand("index", columns, arg(args, 0), arg(args, 1), "")
}

// FullText adds a full-text index command over the given columns.
func (b *Blueprint) FullText(columns any, args ...string) *IndexDefinition {
	return b.indexCommand("fulltext", columns, arg(args, 0), arg(args, 1), "")
}

// SpatialIndex adds a spatial index command over the given columns. The
// optional arguments are the index name and the operator class.
func (b *Blueprint) SpatialIndex(columns any, args ...string) *IndexDefinition {
	return b.indexCommand("spatialIndex", columns, arg(args, 0), "", arg(args, 1))
}

// VectorIndex adds a vector index command over the given column, defaulting
// to the hnsw algorithm and the vector_cosine_ops operator class.
func (b *Blueprint) VectorIndex(column any, name ...string) *IndexDefinition {
	return b.indexCommand("vectorIndex", column, arg(name, 0), "hnsw", "vector_cosine_ops")
}

// RawIndex adds an index command over an expression rather than a column
// list.
func (b *Blueprint) RawIndex(expression, name string) *IndexDefinition {
	return b.Index([]any{query.Raw(expression)}, name)
}

// Foreign adds a foreign key command over the given columns.
func (b *Blueprint) Foreign(columns any, name ...string) *ForeignKeyDefinition {
	definition := b.indexCommand("foreign", columns, arg(name, 0), "", "")
	return &ForeignKeyDefinition{c: definition.c}
}

// ID adds an auto-incrementing big integer primary key column. An empty
// column name defaults to "id".
func (b *Blueprint) ID(column ...string) *ColumnDefinition {
	return b.BigIncrements(defaultTo(arg(column, 0), "id"))
}

// Increments adds an auto-incrementing unsigned integer column.
func (b *Blueprint) Increments(column string) *ColumnDefinition {
	return b.UnsignedInteger(column, true)
}

// IntegerIncrements adds an auto-incrementing unsigned integer column.
func (b *Blueprint) IntegerIncrements(column string) *ColumnDefinition {
	return b.UnsignedInteger(column, true)
}

// TinyIncrements adds an auto-incrementing unsigned tiny integer column.
func (b *Blueprint) TinyIncrements(column string) *ColumnDefinition {
	return b.UnsignedTinyInteger(column, true)
}

// SmallIncrements adds an auto-incrementing unsigned small integer column.
func (b *Blueprint) SmallIncrements(column string) *ColumnDefinition {
	return b.UnsignedSmallInteger(column, true)
}

// MediumIncrements adds an auto-incrementing unsigned medium integer
// column.
func (b *Blueprint) MediumIncrements(column string) *ColumnDefinition {
	return b.UnsignedMediumInteger(column, true)
}

// BigIncrements adds an auto-incrementing unsigned big integer column.
func (b *Blueprint) BigIncrements(column string) *ColumnDefinition {
	return b.UnsignedBigInteger(column, true)
}

// Char adds a fixed-length CHAR column. An omitted length takes the package
// default, which DefaultStringLength sets.
func (b *Blueprint) Char(column string, length ...int) *ColumnDefinition {
	n := defaultStringLength
	if len(length) > 0 {
		n = length[0]
	}
	c := b.AddColumn("char", column)
	c.length = &n
	return c
}

// String adds a VARCHAR column. An omitted or zero length takes the package
// default, which DefaultStringLength sets.
func (b *Blueprint) String(column string, length ...int) *ColumnDefinition {
	n := defaultStringLength
	if len(length) > 0 && length[0] != 0 {
		n = length[0]
	}
	c := b.AddColumn("string", column)
	c.length = &n
	return c
}

// TinyText adds a TINYTEXT column.
func (b *Blueprint) TinyText(column string) *ColumnDefinition {
	return b.AddColumn("tinyText", column)
}

// Text adds a TEXT column.
func (b *Blueprint) Text(column string) *ColumnDefinition { return b.AddColumn("text", column) }

// MediumText adds a MEDIUMTEXT column.
func (b *Blueprint) MediumText(column string) *ColumnDefinition {
	return b.AddColumn("mediumText", column)
}

// LongText adds a LONGTEXT column.
func (b *Blueprint) LongText(column string) *ColumnDefinition {
	return b.AddColumn("longText", column)
}

// Integer adds an integer column. The optional arguments are autoIncrement
// and unsigned.
func (b *Blueprint) Integer(column string, args ...bool) *ColumnDefinition {
	return b.intColumn("integer", column, args)
}

// TinyInteger adds a tiny integer column.
func (b *Blueprint) TinyInteger(column string, args ...bool) *ColumnDefinition {
	return b.intColumn("tinyInteger", column, args)
}

// SmallInteger adds a small integer column.
func (b *Blueprint) SmallInteger(column string, args ...bool) *ColumnDefinition {
	return b.intColumn("smallInteger", column, args)
}

// MediumInteger adds a medium integer column.
func (b *Blueprint) MediumInteger(column string, args ...bool) *ColumnDefinition {
	return b.intColumn("mediumInteger", column, args)
}

// BigInteger adds a big integer column.
func (b *Blueprint) BigInteger(column string, args ...bool) *ColumnDefinition {
	return b.intColumn("bigInteger", column, args)
}

func (b *Blueprint) intColumn(typ, column string, args []bool) *ColumnDefinition {
	c := b.AddColumn(typ, column)
	if len(args) > 0 {
		c.autoIncrement = args[0]
	}
	if len(args) > 1 {
		c.unsigned = args[1]
	}
	return c
}

// UnsignedInteger adds an unsigned integer column, optionally
// auto-incrementing.
func (b *Blueprint) UnsignedInteger(column string, autoIncrement ...bool) *ColumnDefinition {
	return b.Integer(column, boolAt(autoIncrement, 0), true)
}

// UnsignedTinyInteger adds an unsigned tiny integer column, optionally
// auto-incrementing.
func (b *Blueprint) UnsignedTinyInteger(column string, autoIncrement ...bool) *ColumnDefinition {
	return b.TinyInteger(column, boolAt(autoIncrement, 0), true)
}

// UnsignedSmallInteger adds an unsigned small integer column, optionally
// auto-incrementing.
func (b *Blueprint) UnsignedSmallInteger(column string, autoIncrement ...bool) *ColumnDefinition {
	return b.SmallInteger(column, boolAt(autoIncrement, 0), true)
}

// UnsignedMediumInteger adds an unsigned medium integer column, optionally
// auto-incrementing.
func (b *Blueprint) UnsignedMediumInteger(column string, autoIncrement ...bool) *ColumnDefinition {
	return b.MediumInteger(column, boolAt(autoIncrement, 0), true)
}

// UnsignedBigInteger adds an unsigned big integer column, optionally
// auto-incrementing.
func (b *Blueprint) UnsignedBigInteger(column string, autoIncrement ...bool) *ColumnDefinition {
	return b.BigInteger(column, boolAt(autoIncrement, 0), true)
}

// ForeignID adds an unsigned big integer column meant to carry another
// table's key, which Constrained turns into a foreign key.
func (b *Blueprint) ForeignID(column string) *ForeignIDColumnDefinition {
	definition := &ForeignIDColumnDefinition{
		ColumnDefinition: &ColumnDefinition{typ: "bigInteger", name: column, unsigned: true},
		blueprint:        b,
	}
	b.addColumnDefinition(definition.ColumnDefinition)
	return definition
}

// ForeignIDFor adds a foreign key column sized to match the given model's
// key type (integer, ULID, or UUID) and references that model's table and
// key. An empty column defaults to the model's foreign key name.
func (b *Blueprint) ForeignIDFor(model Model, column string) *ForeignIDColumnDefinition {
	if column == "" {
		column = model.GetForeignKey()
	}

	if model.GetKeyType() == "int" {
		definition := b.ForeignID(column)
		definition.Table(model.GetTable())
		definition.ReferencesModelColumn(model.GetKeyName())
		return definition
	}

	if ulid, ok := model.(ULIDModel); ok && ulid.UsesULIDs() {
		definition := b.ForeignULID(column, 26)
		definition.Table(model.GetTable())
		definition.ReferencesModelColumn(model.GetKeyName())
		return definition
	}

	definition := b.ForeignUUID(column)
	definition.Table(model.GetTable())
	definition.ReferencesModelColumn(model.GetKeyName())
	return definition
}

// Float adds a floating point column. An omitted precision defaults to 53.
func (b *Blueprint) Float(column string, precision ...int) *ColumnDefinition {
	n := 53
	if len(precision) > 0 {
		n = precision[0]
	}
	c := b.AddColumn("float", column)
	c.precision = &n
	return c
}

// Double adds a DOUBLE column.
func (b *Blueprint) Double(column string) *ColumnDefinition { return b.AddColumn("double", column) }

// Decimal adds a DECIMAL column. The optional arguments are the total
// digits and the decimal places, defaulting to 8 and 2.
func (b *Blueprint) Decimal(column string, args ...int) *ColumnDefinition {
	total, places := 8, 2
	if len(args) > 0 {
		total = args[0]
	}
	if len(args) > 1 {
		places = args[1]
	}
	c := b.AddColumn("decimal", column)
	c.total, c.places = total, places
	return c
}

// Boolean adds a boolean column.
func (b *Blueprint) Boolean(column string) *ColumnDefinition {
	return b.AddColumn("boolean", column)
}

// Enum adds an ENUM column restricted to the given allowed values.
func (b *Blueprint) Enum(column string, allowed []string) *ColumnDefinition {
	c := b.AddColumn("enum", column)
	c.allowed = allowed
	return c
}

// Set adds a SET column restricted to the given allowed values.
func (b *Blueprint) Set(column string, allowed []string) *ColumnDefinition {
	c := b.AddColumn("set", column)
	c.allowed = allowed
	return c
}

// JSON adds a JSON column. Go initialisms are upper case, hence JSON rather
// than Json.
func (b *Blueprint) JSON(column string) *ColumnDefinition { return b.AddColumn("json", column) }

// JSONB adds a JSONB column.
func (b *Blueprint) JSONB(column string) *ColumnDefinition { return b.AddColumn("jsonb", column) }

// Date adds a DATE column.
func (b *Blueprint) Date(column string) *ColumnDefinition { return b.AddColumn("date", column) }

// DateTime adds a DATETIME column. An omitted precision takes the package
// default, which DefaultTimePrecision sets.
func (b *Blueprint) DateTime(column string, precision ...int) *ColumnDefinition {
	return b.timeColumn("dateTime", column, precision)
}

// DateTimeTz adds a timezone-aware DATETIME column.
func (b *Blueprint) DateTimeTz(column string, precision ...int) *ColumnDefinition {
	return b.timeColumn("dateTimeTz", column, precision)
}

// Time adds a TIME column.
func (b *Blueprint) Time(column string, precision ...int) *ColumnDefinition {
	return b.timeColumn("time", column, precision)
}

// TimeTz adds a timezone-aware TIME column.
func (b *Blueprint) TimeTz(column string, precision ...int) *ColumnDefinition {
	return b.timeColumn("timeTz", column, precision)
}

// Timestamp adds a TIMESTAMP column.
func (b *Blueprint) Timestamp(column string, precision ...int) *ColumnDefinition {
	return b.timeColumn("timestamp", column, precision)
}

// TimestampTz adds a timezone-aware TIMESTAMP column.
func (b *Blueprint) TimestampTz(column string, precision ...int) *ColumnDefinition {
	return b.timeColumn("timestampTz", column, precision)
}

func (b *Blueprint) timeColumn(typ, column string, precision []int) *ColumnDefinition {
	c := b.AddColumn(typ, column)
	if len(precision) > 0 {
		n := precision[0]
		c.precision = &n
	} else {
		c.precision = defaultTimePrecision()
	}
	return c
}

// Timestamps adds nullable created_at and updated_at columns.
func (b *Blueprint) Timestamps(precision ...int) []*ColumnDefinition {
	return []*ColumnDefinition{
		b.Timestamp("created_at", precision...).Nullable(),
		b.Timestamp("updated_at", precision...).Nullable(),
	}
}

// NullableTimestamps is an alias of Timestamps.
func (b *Blueprint) NullableTimestamps(precision ...int) []*ColumnDefinition {
	return b.Timestamps(precision...)
}

// TimestampsTz adds nullable timezone-aware created_at and updated_at
// columns.
func (b *Blueprint) TimestampsTz(precision ...int) []*ColumnDefinition {
	return []*ColumnDefinition{
		b.TimestampTz("created_at", precision...).Nullable(),
		b.TimestampTz("updated_at", precision...).Nullable(),
	}
}

// NullableTimestampsTz is an alias of TimestampsTz.
func (b *Blueprint) NullableTimestampsTz(precision ...int) []*ColumnDefinition {
	return b.TimestampsTz(precision...)
}

// Datetimes adds nullable DATETIME created_at and updated_at columns.
func (b *Blueprint) Datetimes(precision ...int) []*ColumnDefinition {
	return []*ColumnDefinition{
		b.DateTime("created_at", precision...).Nullable(),
		b.DateTime("updated_at", precision...).Nullable(),
	}
}

// SoftDeletes adds a nullable TIMESTAMP column used to mark soft deletion.
// An empty column defaults to deleted_at.
func (b *Blueprint) SoftDeletes(column string, precision ...int) *ColumnDefinition {
	return b.Timestamp(defaultTo(column, "deleted_at"), precision...).Nullable()
}

// SoftDeletesTz adds a nullable timezone-aware TIMESTAMP column used to mark
// soft deletion.
func (b *Blueprint) SoftDeletesTz(column string, precision ...int) *ColumnDefinition {
	return b.TimestampTz(defaultTo(column, "deleted_at"), precision...).Nullable()
}

// SoftDeletesDatetime adds a nullable DATETIME column used to mark soft
// deletion.
func (b *Blueprint) SoftDeletesDatetime(column string, precision ...int) *ColumnDefinition {
	return b.DateTime(defaultTo(column, "deleted_at"), precision...).Nullable()
}

// Year adds a YEAR column.
func (b *Blueprint) Year(column string) *ColumnDefinition { return b.AddColumn("year", column) }

// Binary adds a BINARY or VARBINARY column.
//
// Go has no default arguments, and length and fixed are of different types,
// so both are required: pass 0 and false for a plain, unbounded VARBINARY
// column.
func (b *Blueprint) Binary(column string, length int, fixed bool) *ColumnDefinition {
	c := b.AddColumn("binary", column)
	if length != 0 {
		c.length = &length
	}
	c.fixed = fixed
	return c
}

// UUID adds a UUID column. An empty column name defaults to "uuid".
func (b *Blueprint) UUID(column ...string) *ColumnDefinition {
	return b.AddColumn("uuid", defaultTo(arg(column, 0), "uuid"))
}

// ForeignUUID adds a UUID column meant to carry another table's key.
func (b *Blueprint) ForeignUUID(column string) *ForeignIDColumnDefinition {
	definition := &ForeignIDColumnDefinition{
		ColumnDefinition: &ColumnDefinition{typ: "uuid", name: column},
		blueprint:        b,
	}
	b.addColumnDefinition(definition.ColumnDefinition)
	return definition
}

// ULID adds a CHAR column sized for a ULID. An empty column name defaults
// to "ulid", and an omitted length defaults to 26.
func (b *Blueprint) ULID(column string, length ...int) *ColumnDefinition {
	n := 26
	if len(length) > 0 {
		n = length[0]
	}
	return b.Char(defaultTo(column, "ulid"), n)
}

// ForeignULID adds a CHAR column sized for a ULID, meant to carry another
// table's key.
func (b *Blueprint) ForeignULID(column string, length ...int) *ForeignIDColumnDefinition {
	n := 26
	if len(length) > 0 {
		n = length[0]
	}
	definition := &ForeignIDColumnDefinition{
		ColumnDefinition: &ColumnDefinition{typ: "char", name: column, length: &n},
		blueprint:        b,
	}
	b.addColumnDefinition(definition.ColumnDefinition)
	return definition
}

// IPAddress adds a column sized for an IP address. An empty column name
// defaults to "ip_address".
func (b *Blueprint) IPAddress(column ...string) *ColumnDefinition {
	return b.AddColumn("ipAddress", defaultTo(arg(column, 0), "ip_address"))
}

// MacAddress adds a column sized for a MAC address. An empty column name
// defaults to "mac_address".
func (b *Blueprint) MacAddress(column ...string) *ColumnDefinition {
	return b.AddColumn("macAddress", defaultTo(arg(column, 0), "mac_address"))
}

// Geometry adds a spatial GEOMETRY column. An empty subtype leaves it
// unconstrained, and an omitted SRID defaults to 0.
func (b *Blueprint) Geometry(column string, subtype string, srid ...int) *ColumnDefinition {
	c := b.AddColumn("geometry", column)
	c.subtype = subtype
	if len(srid) > 0 {
		c.srid = srid[0]
	}
	return c
}

// Geography adds a spatial GEOGRAPHY column. An omitted SRID defaults to
// 4326, which is why it is variadic rather than a plain int.
func (b *Blueprint) Geography(column string, subtype string, srid ...int) *ColumnDefinition {
	c := b.AddColumn("geography", column)
	c.subtype = subtype
	c.srid = 4326
	if len(srid) > 0 {
		c.srid = srid[0]
	}
	return c
}

// Computed adds a generated column defined by the given expression.
func (b *Blueprint) Computed(column, expression string) *ColumnDefinition {
	c := b.AddColumn("computed", column)
	c.expression = expression
	return c
}

// Vector adds a vector column, optionally sized to the given number of
// dimensions.
func (b *Blueprint) Vector(column string, dimensions ...int) *ColumnDefinition {
	c := b.AddColumn("vector", column)
	if len(dimensions) > 0 && dimensions[0] != 0 {
		n := dimensions[0]
		c.dimensions = &n
	}
	return c
}

// Morphs adds the type and id columns for a polymorphic relation, using the
// key type configured by defaultMorphKeyType. The optional arguments are the
// index name and the column to place these after.
func (b *Blueprint) Morphs(name string, args ...string) {
	switch defaultMorphKeyType {
	case "uuid":
		b.UUIDMorphs(name, args...)
	case "ulid":
		b.ULIDMorphs(name, args...)
	default:
		b.NumericMorphs(name, args...)
	}
}

// NullableMorphs adds nullable type and id columns for a polymorphic
// relation, using the key type configured by defaultMorphKeyType.
func (b *Blueprint) NullableMorphs(name string, args ...string) {
	switch defaultMorphKeyType {
	case "uuid":
		b.NullableUUIDMorphs(name, args...)
	case "ulid":
		b.NullableULIDMorphs(name, args...)
	default:
		b.NullableNumericMorphs(name, args...)
	}
}

// NumericMorphs adds the type and id columns for a polymorphic relation,
// with the id column as an unsigned big integer.
func (b *Blueprint) NumericMorphs(name string, args ...string) {
	b.morphs(name, args, false, func(column string) *ColumnDefinition {
		return b.UnsignedBigInteger(column)
	})
}

// NullableNumericMorphs adds nullable type and id columns for a polymorphic
// relation, with the id column as an unsigned big integer.
func (b *Blueprint) NullableNumericMorphs(name string, args ...string) {
	b.morphs(name, args, true, func(column string) *ColumnDefinition {
		return b.UnsignedBigInteger(column)
	})
}

// UUIDMorphs adds the type and id columns for a polymorphic relation, with
// the id column as a UUID.
func (b *Blueprint) UUIDMorphs(name string, args ...string) {
	b.morphs(name, args, false, func(column string) *ColumnDefinition {
		return b.UUID(column)
	})
}

// NullableUUIDMorphs adds nullable type and id columns for a polymorphic
// relation, with the id column as a UUID.
func (b *Blueprint) NullableUUIDMorphs(name string, args ...string) {
	b.morphs(name, args, true, func(column string) *ColumnDefinition {
		return b.UUID(column)
	})
}

// ULIDMorphs adds the type and id columns for a polymorphic relation, with
// the id column as a ULID.
func (b *Blueprint) ULIDMorphs(name string, args ...string) {
	b.morphs(name, args, false, func(column string) *ColumnDefinition {
		return b.ULID(column)
	})
}

// NullableULIDMorphs adds nullable type and id columns for a polymorphic
// relation, with the id column as a ULID.
func (b *Blueprint) NullableULIDMorphs(name string, args ...string) {
	b.morphs(name, args, true, func(column string) *ColumnDefinition {
		return b.ULID(column)
	})
}

// morphs is the shared body of the six morphs methods: a type column, a key
// column and an index over both.
func (b *Blueprint) morphs(name string, args []string, nullable bool, key func(string) *ColumnDefinition) {
	indexName, after := arg(args, 0), arg(args, 1)

	typeColumn := b.String(name + "_type")
	if nullable {
		typeColumn.Nullable()
	}
	typeColumn.After(after)

	keyColumn := key(name + "_id")
	if nullable {
		keyColumn.Nullable()
	}
	if after != "" {
		keyColumn.After(name + "_type")
	}

	b.Index([]any{name + "_type", name + "_id"}, indexName)
}

// RememberToken adds a nullable, 100-character remember_token column.
func (b *Blueprint) RememberToken() *ColumnDefinition {
	return b.String("remember_token", 100).Nullable()
}

// RawColumn adds a column whose definition is written out by hand.
func (b *Blueprint) RawColumn(column, definition string) *ColumnDefinition {
	c := b.AddColumn("raw", column)
	c.definition = definition
	return c
}

// Comment queues a command that sets a comment on the table.
func (b *Blueprint) Comment(comment string) *Command {
	command := b.addCommand("tableComment")
	command.Comment = comment
	return command
}

// indexCommand queues an index command of the given type over the given
// columns, generating a name by convention if none is given.
func (b *Blueprint) indexCommand(typ string, columns any, index, algorithm, operatorClass string) *IndexDefinition {
	list := toColumnList(columns)

	if index == "" {
		index = b.createIndexName(typ, list)
	}

	command := b.addCommand(typ)
	command.Index = index
	command.Columns = list
	command.Algorithm = algorithm
	command.OperatorClass = operatorClass

	return &IndexDefinition{c: command}
}

// dropIndexCommand queues a command that drops an index. An index given as
// a list of columns is named by the same convention that created it.
func (b *Blueprint) dropIndexCommand(command, typ string, index any) *Command {
	var columns []any
	name, isName := index.(string)

	if !isName {
		columns = toColumnList(index)
		name = b.createIndexName(typ, columns)
	}

	return b.indexCommand(command, columns, name, "", "").c
}

// createIndexName builds a conventional index name from the table, the
// given columns and the index type.
func (b *Blueprint) createIndexName(typ string, columns []any) string {
	table := b.table

	if b.connection.GetConfig("prefix_indexes") != "" {
		if i := strings.LastIndex(b.table, "."); i >= 0 {
			table = b.table[:i] + "." + b.connection.GetTablePrefix() + b.table[i+1:]
		} else {
			table = b.connection.GetTablePrefix() + b.table
		}
	}

	parts := make([]string, 0, len(columns)+2)
	parts = append(parts, table)
	for _, column := range columns {
		parts = append(parts, stringify(column))
	}
	parts = append(parts, typ)

	index := strings.ToLower(strings.Join(parts, "_"))
	index = strings.ReplaceAll(index, "-", "_")
	index = strings.ReplaceAll(index, ".", "_")

	return shortenIdentifier(index, b.maxIdentifierLength())
}

// maxIdentifierLength is the grammar's identifier limit, or zero when the
// blueprint was built without a grammar.
func (b *Blueprint) maxIdentifierLength() int {
	if b.grammar == nil {
		return 0
	}
	return b.grammar.MaxIdentifierLength()
}

// identifierHashLength is how many hex digits of the digest a shortened
// identifier carries. Six bytes is enough that a table would need on the order
// of sixteen million index names before two of them were expected to collide,
// and short enough to leave the readable half of the name readable.
const identifierHashLength = 12

// shortenIdentifier returns name unchanged when it fits within limit, and a
// deterministic name of at most limit bytes when it does not.
//
// The shortened form is a prefix of the conventional name followed by a digest
// of the whole of it, so that two names differing only past the cut still
// differ after it -- which plain truncation, the obvious fix, does not give.
// The digest is of the full name and nothing else, so the same columns on the
// same table produce the same name on every run and every machine.
//
// A limit of zero, or a name already within it, is returned untouched: the
// driver that reports no limit gets the conventional name it always got.
func shortenIdentifier(name string, limit int) string {
	if limit <= 0 || len(name) <= limit {
		return name
	}

	sum := sha256.Sum256([]byte(name))
	suffix := "_" + hex.EncodeToString(sum[:])[:identifierHashLength]

	keep := limit - len(suffix)
	if keep < 1 {
		// The limit is too small for a prefix. Return as much of the digest
		// as fits, which is still deterministic and still distinguishing.
		return suffix[1:][:limit]
	}

	return strings.TrimRight(name[:keep], "_") + suffix
}

// checkIdentifiers reports an IdentifierTooLongError when a command carries a
// name the driver will not accept.
//
// Only a name the caller wrote by hand can fail here: createIndexName shortens
// the ones this package generates before they are ever stored on the command.
func (b *Blueprint) checkIdentifiers(command *Command) error {
	limit := b.maxIdentifierLength()
	if limit <= 0 {
		return nil
	}

	for _, name := range []string{command.Index, command.To} {
		if name != "" && len(name) > limit {
			return &IdentifierTooLongError{Name: name, Limit: limit}
		}
	}

	return nil
}

// AddColumn adds a column of the given type and name to the blueprint.
func (b *Blueprint) AddColumn(typ, name string) *ColumnDefinition {
	definition := &ColumnDefinition{typ: typ, name: name}
	b.addColumnDefinition(definition)
	return definition
}

// addColumnDefinition registers a column definition on the blueprint,
// queuing an add command for it when the table is not being newly created,
// and chaining After placement when set.
func (b *Blueprint) addColumnDefinition(definition *ColumnDefinition) *ColumnDefinition {
	b.columns = append(b.columns, definition)

	if !b.Creating() {
		b.commands = append(b.commands, &Command{Column: definition, isColumnDefinition: true})
	}

	if b.after != "" {
		definition.After(b.after)
		b.after = definition.name
	}

	return definition
}

// After runs callback with every column it adds placed after the given
// column, in the order they were added.
func (b *Blueprint) After(column string, callback func(*Blueprint)) {
	b.after = column
	callback(b)
	b.after = ""
}

// GetAfter returns the column currently set as the placement anchor for
// After.
func (b *Blueprint) GetAfter() string { return b.after }

// RemoveColumn removes a column and any pending command that adds it,
// returning the blueprint for chaining.
func (b *Blueprint) RemoveColumn(name string) *Blueprint {
	columns := b.columns[:0]
	for _, column := range b.columns {
		if column.name != name {
			columns = append(columns, column)
		}
	}
	b.columns = columns

	commands := b.commands[:0]
	for _, command := range b.commands {
		if !command.isColumnDefinition || command.Column.name != name {
			commands = append(commands, command)
		}
	}
	b.commands = commands

	return b
}

// addCommand queues a new command of the given name.
func (b *Blueprint) addCommand(name string) *Command {
	command := NewCommand(name)
	b.commands = append(b.commands, command)
	return command
}

// GetTable returns the table's name.
func (b *Blueprint) GetTable() string { return b.table }

// GetPrefix returns the connection's table prefix.
func (b *Blueprint) GetPrefix() string { return b.connection.GetTablePrefix() }

// GetColumns returns the blueprint's columns.
func (b *Blueprint) GetColumns() []*ColumnDefinition { return b.columns }

// GetCommands returns the blueprint's queued commands.
func (b *Blueprint) GetCommands() []*Command { return b.commands }

// GetState returns the blueprint's BlueprintState, which is nil unless an
// alter command required reading the table's current shape.
func (b *Blueprint) GetState() *BlueprintState { return b.state }

// GetAddedColumns returns the columns being added, excluding any marked as
// changed.
func (b *Blueprint) GetAddedColumns() []*ColumnDefinition {
	var added []*ColumnDefinition
	for _, column := range b.columns {
		if !column.change {
			added = append(added, column)
		}
	}
	return added
}

// GetChangedColumns returns the columns marked as changed.
func (b *Blueprint) GetChangedColumns() []*ColumnDefinition {
	var changed []*ColumnDefinition
	for _, column := range b.columns {
		if column.change {
			changed = append(changed, column)
		}
	}
	return changed
}

// marker reads one of a column's fluent index attributes (primary, unique,
// index, fulltext, spatialIndex, vectorIndex) by name.
func (c *ColumnDefinition) marker(index string) any {
	switch index {
	case "primary":
		return c.primary
	case "unique":
		return c.unique
	case "index":
		return c.index
	case "fulltext":
		return c.fullText
	case "spatialIndex":
		return c.spatialIndex
	case "vectorIndex":
		return c.vectorIndex
	}
	return nil
}

func (c *ColumnDefinition) setMarker(index string, value any) {
	switch index {
	case "primary":
		c.primary = value
	case "unique":
		c.unique = value
	case "index":
		c.index = value
	case "fulltext":
		c.fullText = value
	case "spatialIndex":
		c.spatialIndex = value
	case "vectorIndex":
		c.vectorIndex = value
	}
}

// toColumnList turns the string, string slice or expression slice a caller
// passes for columns into the list the command carries.
func toColumnList(columns any) []any {
	switch v := columns.(type) {
	case nil:
		return nil
	case string:
		return []any{v}
	case []string:
		return toAnyList(v)
	case []any:
		return v
	default:
		return []any{v}
	}
}

func toAnyList(values []string) []any {
	out := make([]any, len(values))
	for i, value := range values {
		out[i] = value
	}
	return out
}

func contains(values []string, value string) bool {
	for _, v := range values {
		if v == value {
			return true
		}
	}
	return false
}

func defaultTo(value, fallback string) string {
	if value == "" {
		return fallback
	}
	return value
}

func boolAt(values []bool, n int) bool {
	if n < len(values) {
		return values[n]
	}
	return false
}

// stringify renders a column entry, which is a string or an expression.
func stringify(value any) string {
	switch v := value.(type) {
	case nil:
		return ""
	case string:
		return v
	case query.Expression:
		return v.String()
	case *query.Expression:
		return v.String()
	default:
		return fmt.Sprint(value)
	}
}
