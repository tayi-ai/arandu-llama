package schema

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
)

// defaultStringLength is the length a string column takes when none is given.
//
// The value is an unexported package variable because the exported name belongs
// to the setter.
var defaultStringLength = 255

// defaultTimePrecisionValue is the time column precision used when none is
// given explicitly.
var defaultTimePrecisionValue = intPointer(0)

// defaultMorphKeyType is the key type used for new morph columns when none
// is given explicitly.
var defaultMorphKeyType = "int"

// DefaultStringLength sets the default length new string columns take when
// none is given explicitly.
func DefaultStringLength(length int) { defaultStringLength = length }

// DefaultTimePrecision sets the default time column precision. A nil
// precision declares the column without one.
func DefaultTimePrecision(precision *int) { defaultTimePrecisionValue = precision }

// DefaultMorphKeyType sets the key type used for new morph columns. It
// returns an error if typ is not "int", "uuid", or "ulid".
func DefaultMorphKeyType(typ string) error {
	switch typ {
	case "int", "uuid", "ulid":
		defaultMorphKeyType = typ
		return nil
	default:
		return fmt.Errorf("schema: morph key type must be 'int', 'uuid', or 'ulid', got %q", typ)
	}
}

// MorphUsingUUIDs sets new morph columns to key on UUIDs.
func MorphUsingUUIDs() { defaultMorphKeyType = "uuid" }

// MorphUsingULIDs sets new morph columns to key on ULIDs.
func MorphUsingULIDs() { defaultMorphKeyType = "ulid" }

func defaultTimePrecision() *int {
	if defaultTimePrecisionValue == nil {
		return nil
	}
	value := *defaultTimePrecisionValue
	return &value
}

func intPointer(value int) *int { return &value }

// ErrUnsupported is returned for a schema operation the current driver does
// not support.
var ErrUnsupported = errors.New("schema: this database driver does not support the operation")

// Builder is the half of this component that executes: Blueprint and Grammar
// build strings, and everything that reaches the connection is here.
//
// No method on it asks for an authorization credential, and that is a decision
// rather than an omission. DDL names a table rather than rows, so there is no
// tenant column to scope the statement by, and it is run as a pipeline step with
// no request behind it, so there is no subject to attribute it to. The only
// credential this type could hold is one its caller invented, and a parameter
// that looks like enforcement while enforcing nothing is worse than no
// parameter, because a reader stops looking. The path to application rows is a
// different one, and it is still closed on every method.
type Builder struct {
	connection Connection
	grammar    Grammar
	resolver   func(Connection, string, func(*Blueprint)) *Blueprint
}

// NewBuilder returns a Builder for the given connection.
func NewBuilder(connection Connection) *Builder {
	return &Builder{connection: connection, grammar: connection.GetSchemaGrammar()}
}

// GetConnection returns the connection the builder runs statements against.
func (b *Builder) GetConnection() Connection { return b.connection }

// BlueprintResolver sets the function CreateBlueprint uses to construct a
// Blueprint, in place of NewBlueprint.
func (b *Builder) BlueprintResolver(resolver func(Connection, string, func(*Blueprint)) *Blueprint) {
	b.resolver = resolver
}

// CreateDatabase creates a database named name.
func (b *Builder) CreateDatabase(ctx context.Context, name string) error {
	sql, err := b.grammar.CompileCreateDatabase(name)
	if err != nil {
		return err
	}
	return b.connection.Statement(ctx, sql)
}

// DropDatabaseIfExists drops the database named name, if it exists.
func (b *Builder) DropDatabaseIfExists(ctx context.Context, name string) error {
	sql, err := b.grammar.CompileDropDatabaseIfExists(name)
	if err != nil {
		return err
	}
	return b.connection.Statement(ctx, sql)
}

// GetSchemas returns the schemas visible on this connection.
func (b *Builder) GetSchemas(ctx context.Context) ([]SchemaInfo, error) {
	sql, err := b.grammar.CompileSchemas()
	if err != nil {
		return nil, err
	}
	rows, err := b.connection.Select(ctx, sql)
	if err != nil {
		return nil, err
	}
	return b.connection.ProcessSchemas(rows), nil
}

// HasTable reports whether table exists.
func (b *Builder) HasTable(ctx context.Context, table string) (bool, error) {
	schema, name, err := ParseSchemaAndTable(table)
	if err != nil {
		return false, err
	}
	name = b.connection.GetTablePrefix() + name

	if sql := b.grammar.CompileTableExists(schema, name); sql != "" {
		value, err := b.connection.Scalar(ctx, sql)
		if err != nil {
			return false, err
		}
		return truthy(value), nil
	}

	if schema == "" {
		schema = b.GetCurrentSchemaName()
	}
	tables, err := b.GetTables(ctx, schema)
	if err != nil {
		return false, err
	}
	for _, candidate := range tables {
		if strings.EqualFold(name, candidate.Name) {
			return true, nil
		}
	}
	return false, nil
}

// HasView reports whether view exists.
func (b *Builder) HasView(ctx context.Context, view string) (bool, error) {
	schema, name, err := ParseSchemaAndTable(view)
	if err != nil {
		return false, err
	}
	name = b.connection.GetTablePrefix() + name

	if schema == "" {
		schema = b.GetCurrentSchemaName()
	}
	views, err := b.GetViews(ctx, schema)
	if err != nil {
		return false, err
	}
	for _, candidate := range views {
		if strings.EqualFold(name, candidate.Name) {
			return true, nil
		}
	}
	return false, nil
}

// GetTables returns the tables in the given schemas. With no schemas given,
// it returns tables from every schema the connection can see.
func (b *Builder) GetTables(ctx context.Context, schemas ...string) ([]TableInfo, error) {
	sql, err := b.grammar.CompileTables(nonEmpty(schemas))
	if err != nil {
		return nil, err
	}
	rows, err := b.connection.Select(ctx, sql)
	if err != nil {
		return nil, err
	}
	return b.connection.ProcessTables(rows), nil
}

// GetTableListing returns the names of tables in the given schemas.
// schemaQualified defaults to true, which returns schema-qualified names.
func (b *Builder) GetTableListing(ctx context.Context, schemas []string, schemaQualified ...bool) ([]string, error) {
	tables, err := b.GetTables(ctx, schemas...)
	if err != nil {
		return nil, err
	}
	qualified := true
	if len(schemaQualified) > 0 {
		qualified = schemaQualified[0]
	}
	names := make([]string, 0, len(tables))
	for _, table := range tables {
		if qualified {
			names = append(names, table.SchemaQualifiedName)
			continue
		}
		names = append(names, table.Name)
	}
	return names, nil
}

// GetViews returns the views in the given schemas.
func (b *Builder) GetViews(ctx context.Context, schemas ...string) ([]ViewInfo, error) {
	sql, err := b.grammar.CompileViews(nonEmpty(schemas))
	if err != nil {
		return nil, err
	}
	rows, err := b.connection.Select(ctx, sql)
	if err != nil {
		return nil, err
	}
	return b.connection.ProcessViews(rows), nil
}

// GetTypes returns the user-defined types in the given schemas.
func (b *Builder) GetTypes(ctx context.Context, schemas ...string) ([]Record, error) {
	sql, err := b.grammar.CompileTypes(nonEmpty(schemas))
	if err != nil {
		return nil, err
	}
	return b.connection.Select(ctx, sql)
}

// DropAllTypes drops every type named in types.
//
// The refusal for a driver with no types to drop comes from the grammar,
// which is the only thing that knows whether the driver supports dropping
// types at all.
func (b *Builder) DropAllTypes(ctx context.Context, types []string) error {
	if len(types) == 0 {
		return nil
	}
	sql, err := b.grammar.CompileDropAllTypes(types)
	if err != nil {
		return err
	}
	return b.connection.Statement(ctx, sql)
}

// HasColumn reports whether table has a column named column.
func (b *Builder) HasColumn(ctx context.Context, table, column string) (bool, error) {
	columns, err := b.GetColumnListing(ctx, table)
	if err != nil {
		return false, err
	}
	for _, candidate := range columns {
		if strings.EqualFold(candidate, column) {
			return true, nil
		}
	}
	return false, nil
}

// HasColumns reports whether table has every column named in columns.
func (b *Builder) HasColumns(ctx context.Context, table string, columns []string) (bool, error) {
	existing, err := b.GetColumnListing(ctx, table)
	if err != nil {
		return false, err
	}
	for _, column := range columns {
		found := false
		for _, candidate := range existing {
			if strings.EqualFold(candidate, column) {
				found = true
				break
			}
		}
		if !found {
			return false, nil
		}
	}
	return true, nil
}

// WhenTableHasColumn runs callback against table if it has a column named
// column.
func (b *Builder) WhenTableHasColumn(ctx context.Context, table, column string, callback func(*Blueprint)) error {
	has, err := b.HasColumn(ctx, table, column)
	if err != nil || !has {
		return err
	}
	return b.Table(ctx, table, callback)
}

// WhenTableDoesntHaveColumn runs callback against table if it does not have
// a column named column.
func (b *Builder) WhenTableDoesntHaveColumn(ctx context.Context, table, column string, callback func(*Blueprint)) error {
	has, err := b.HasColumn(ctx, table, column)
	if err != nil || has {
		return err
	}
	return b.Table(ctx, table, callback)
}

// WhenTableHasIndex runs callback against table if it has the given index.
func (b *Builder) WhenTableHasIndex(ctx context.Context, table string, index any, callback func(*Blueprint), typ ...string) error {
	has, err := b.HasIndex(ctx, table, index, typ...)
	if err != nil || !has {
		return err
	}
	return b.Table(ctx, table, callback)
}

// WhenTableDoesntHaveIndex runs callback against table if it does not have
// the given index.
func (b *Builder) WhenTableDoesntHaveIndex(ctx context.Context, table string, index any, callback func(*Blueprint), typ ...string) error {
	has, err := b.HasIndex(ctx, table, index, typ...)
	if err != nil || has {
		return err
	}
	return b.Table(ctx, table, callback)
}

// GetColumnType returns the type of column on table. It returns an error if
// table has no such column. The optional fullDefinition returns the full
// type definition instead of the bare type name.
func (b *Builder) GetColumnType(ctx context.Context, table, column string, fullDefinition ...bool) (string, error) {
	columns, err := b.GetColumns(ctx, table)
	if err != nil {
		return "", err
	}
	full := len(fullDefinition) > 0 && fullDefinition[0]
	for _, candidate := range columns {
		if strings.EqualFold(candidate.Name, column) {
			if full {
				return candidate.Type, nil
			}
			return candidate.TypeName, nil
		}
	}
	return "", fmt.Errorf("schema: there is no column with name %q on table %q", column, table)
}

// GetColumnListing returns the names of table's columns.
func (b *Builder) GetColumnListing(ctx context.Context, table string) ([]string, error) {
	columns, err := b.GetColumns(ctx, table)
	if err != nil {
		return nil, err
	}
	names := make([]string, 0, len(columns))
	for _, column := range columns {
		names = append(names, column.Name)
	}
	return names, nil
}

// GetColumns returns the columns of table.
func (b *Builder) GetColumns(ctx context.Context, table string) ([]ColumnInfo, error) {
	schema, name, err := ParseSchemaAndTable(table)
	if err != nil {
		return nil, err
	}
	sql, err := b.grammar.CompileColumns(schema, b.connection.GetTablePrefix()+name)
	if err != nil {
		return nil, err
	}
	rows, err := b.connection.Select(ctx, sql)
	if err != nil {
		return nil, err
	}
	return b.connection.ProcessColumns(rows), nil
}

// GetIndexes returns the indexes of table.
func (b *Builder) GetIndexes(ctx context.Context, table string) ([]IndexInfo, error) {
	schema, name, err := ParseSchemaAndTable(table)
	if err != nil {
		return nil, err
	}
	sql, err := b.grammar.CompileIndexes(schema, b.connection.GetTablePrefix()+name)
	if err != nil {
		return nil, err
	}
	rows, err := b.connection.Select(ctx, sql)
	if err != nil {
		return nil, err
	}
	return b.connection.ProcessIndexes(rows), nil
}

// GetIndexListing returns the names of table's indexes.
func (b *Builder) GetIndexListing(ctx context.Context, table string) ([]string, error) {
	indexes, err := b.GetIndexes(ctx, table)
	if err != nil {
		return nil, err
	}
	names := make([]string, 0, len(indexes))
	for _, index := range indexes {
		names = append(names, index.Name)
	}
	return names, nil
}

// HasIndex reports whether table has the given index. The index is a name
// or the list of columns it covers; the optional type narrows the match to
// primary, unique or a driver's own index type.
func (b *Builder) HasIndex(ctx context.Context, table string, index any, typ ...string) (bool, error) {
	indexes, err := b.GetIndexes(ctx, table)
	if err != nil {
		return false, err
	}

	wanted := ""
	if len(typ) > 0 {
		wanted = strings.ToLower(typ[0])
	}

	for _, candidate := range indexes {
		typeMatches := wanted == "" ||
			(wanted == "primary" && candidate.Primary) ||
			(wanted == "unique" && candidate.Unique) ||
			wanted == candidate.Type
		if !typeMatches {
			continue
		}
		switch v := index.(type) {
		case string:
			if candidate.Name == v {
				return true, nil
			}
		case []string:
			if equalStrings(candidate.Columns, v) {
				return true, nil
			}
		}
	}
	return false, nil
}

// GetForeignKeys returns the foreign keys of table.
func (b *Builder) GetForeignKeys(ctx context.Context, table string) ([]ForeignKeyInfo, error) {
	schema, name, err := ParseSchemaAndTable(table)
	if err != nil {
		return nil, err
	}
	sql, err := b.grammar.CompileForeignKeys(schema, b.connection.GetTablePrefix()+name)
	if err != nil {
		return nil, err
	}
	rows, err := b.connection.Select(ctx, sql)
	if err != nil {
		return nil, err
	}
	return b.connection.ProcessForeignKeys(rows), nil
}

// Table runs callback against a Blueprint for a table that already exists.
func (b *Builder) Table(ctx context.Context, table string, callback func(*Blueprint)) error {
	return b.build(ctx, b.CreateBlueprint(table, callback))
}

// Create creates table, configured by callback.
func (b *Builder) Create(ctx context.Context, table string, callback func(*Blueprint)) error {
	blueprint := b.CreateBlueprint(table, nil)
	blueprint.Create()
	callback(blueprint)
	return b.build(ctx, blueprint)
}

// Drop drops table.
func (b *Builder) Drop(ctx context.Context, table string) error {
	blueprint := b.CreateBlueprint(table, nil)
	blueprint.Drop()
	return b.build(ctx, blueprint)
}

// DropIfExists drops table if it exists.
func (b *Builder) DropIfExists(ctx context.Context, table string) error {
	blueprint := b.CreateBlueprint(table, nil)
	blueprint.DropIfExists()
	return b.build(ctx, blueprint)
}

// DropColumns drops the given columns from table.
func (b *Builder) DropColumns(ctx context.Context, table string, columns []string) error {
	return b.Table(ctx, table, func(blueprint *Blueprint) {
		blueprint.DropColumn(columns...)
	})
}

// DropAllTables drops every table the connection can see.
//
// A driver whose grammar cannot compile the statement returns ErrUnsupported
// instead.
func (b *Builder) DropAllTables(ctx context.Context) error {
	tables, err := b.GetTables(ctx)
	if err != nil {
		return err
	}
	if len(tables) == 0 {
		return nil
	}
	names := make([]string, 0, len(tables))
	for _, table := range tables {
		names = append(names, table.SchemaQualifiedName)
	}
	sql, err := b.grammar.CompileDropAllTables(names)
	if err != nil {
		return err
	}
	return b.connection.Statement(ctx, sql)
}

// RefreshDatabaseFile empties a SQLite database by truncating the file it
// lives in. Passing the empty path takes the file the connection is
// configured with.
//
// This is the fastest way to empty a SQLite database; DropAllTables is the
// slower alternative that works even when SQLite has no file to truncate.
//
// RefreshDatabaseFile refuses on any other driver and on an in-memory
// database, rather than truncating a path that turns out to be something
// else. This package has one Builder for every driver, so the refusal has
// to be an explicit check rather than something the type system rules out.
// Truncating a file is not a statement the database can refuse, so nothing
// downstream would have caught a mistake here.
func (b *Builder) RefreshDatabaseFile(ctx context.Context, path string) error {
	if driver := b.connection.GetDriverName(); driver != "sqlite" {
		return fmt.Errorf("%w: refreshDatabaseFile empties the file a SQLite database lives in, and this connection is %s", ErrUnsupported, driver)
	}

	if path == "" {
		path = b.connection.GetConfig("database")
	}
	if path == "" || isSqliteMemory(path) {
		return fmt.Errorf("%w: this SQLite database is in memory, so there is no file to empty -- use DropAllTables", ErrUnsupported)
	}

	if err := os.WriteFile(path, nil, 0o600); err != nil {
		return fmt.Errorf("schema: emptying the SQLite database file %s: %w", path, err)
	}
	return nil
}

// DropAllViews drops every view the connection can see.
func (b *Builder) DropAllViews(ctx context.Context) error {
	views, err := b.GetViews(ctx)
	if err != nil {
		return err
	}
	if len(views) == 0 {
		return nil
	}
	names := make([]string, 0, len(views))
	for _, view := range views {
		names = append(names, view.SchemaQualifiedName)
	}
	sql, err := b.grammar.CompileDropAllViews(names)
	if err != nil {
		return err
	}
	return b.connection.Statement(ctx, sql)
}

// Rename renames table from to to.
func (b *Builder) Rename(ctx context.Context, from, to string) error {
	blueprint := b.CreateBlueprint(from, nil)
	blueprint.Rename(to)
	return b.build(ctx, blueprint)
}

// EnableForeignKeyConstraints turns foreign key enforcement on for the
// connection.
func (b *Builder) EnableForeignKeyConstraints(ctx context.Context) error {
	sql, err := b.grammar.CompileEnableForeignKeyConstraints()
	if err != nil {
		return err
	}
	return b.connection.Statement(ctx, sql)
}

// DisableForeignKeyConstraints turns foreign key enforcement off for the
// connection.
func (b *Builder) DisableForeignKeyConstraints(ctx context.Context) error {
	sql, err := b.grammar.CompileDisableForeignKeyConstraints()
	if err != nil {
		return err
	}
	return b.connection.Statement(ctx, sql)
}

// WithoutForeignKeyConstraints disables foreign key constraints, runs
// callback, and re-enables them.
//
// The constraints are put back whether callback succeeded or not. A failure
// to put them back is reported even when callback already failed, joined to
// it with errors.Join, because a connection left with its foreign keys off
// is the more dangerous of the two.
func (b *Builder) WithoutForeignKeyConstraints(ctx context.Context, callback func() error) error {
	if err := b.DisableForeignKeyConstraints(ctx); err != nil {
		return err
	}
	err := callback()
	return errors.Join(err, b.EnableForeignKeyConstraints(ctx))
}

// EnsureVectorExtensionExists creates the Postgres "vector" extension if it
// does not already exist, in the given schema or the default schema if none
// is given.
func (b *Builder) EnsureVectorExtensionExists(ctx context.Context, schema ...string) error {
	return b.EnsureExtensionExists(ctx, "vector", schema...)
}

// EnsureExtensionExists creates the Postgres extension named name if it does
// not already exist. It returns ErrUnsupported on any other driver.
func (b *Builder) EnsureExtensionExists(ctx context.Context, name string, schema ...string) error {
	if b.connection.GetDriverName() != "pgsql" {
		return fmt.Errorf("%w: extensions are only supported by Postgres", ErrUnsupported)
	}
	sql := "create extension if not exists " + b.grammar.Wrap(name)
	if len(schema) > 0 && schema[0] != "" {
		sql += " schema " + b.grammar.Wrap(schema[0])
	}
	return b.connection.Statement(ctx, sql)
}

// build runs blueprint against the connection.
func (b *Builder) build(ctx context.Context, blueprint *Blueprint) error {
	return blueprint.Build(ctx)
}

// CreateBlueprint returns a new Blueprint for table, using the resolver set
// by BlueprintResolver if one was set, or NewBlueprint otherwise.
func (b *Builder) CreateBlueprint(table string, callback func(*Blueprint)) *Blueprint {
	if b.resolver != nil {
		return b.resolver(b.connection, table, callback)
	}
	return NewBlueprint(b.connection, table, callback)
}

// GetCurrentSchemaListing returns the schemas the connection searches, in
// priority order.
func (b *Builder) GetCurrentSchemaListing() []string { return nil }

// GetCurrentSchemaName returns the first schema from GetCurrentSchemaListing,
// or the empty string if it returns none.
func (b *Builder) GetCurrentSchemaName() string {
	listing := b.GetCurrentSchemaListing()
	if len(listing) == 0 {
		return ""
	}
	return listing[0]
}

// ParseSchemaAndTable splits a qualified name into its schema and its table.
//
// It is a package function rather than only a method because the SQLite and
// Postgres grammars call it while compiling, where they have a blueprint but no
// builder. The method below calls it, so there is one implementation.
func ParseSchemaAndTable(reference string) (schema, table string, err error) {
	segments := strings.Split(reference, ".")
	if len(segments) > 2 {
		return "", "", fmt.Errorf("schema: using three-part references is not supported, name the connection instead of %q", reference)
	}
	if len(segments) == 2 {
		return segments[0], segments[1], nil
	}
	return "", segments[0], nil
}

// ParseSchemaAndTable splits reference into its schema and table, like the
// package-level ParseSchemaAndTable. The optional withDefaultSchema is a
// schema name to fall back on, or, if empty, the connection's current
// schema.
func (b *Builder) ParseSchemaAndTable(reference string, withDefaultSchema ...string) (schema, table string, err error) {
	schema, table, err = ParseSchemaAndTable(reference)
	if err != nil || schema != "" || len(withDefaultSchema) == 0 {
		return schema, table, err
	}
	if withDefaultSchema[0] != "" {
		return withDefaultSchema[0], table, nil
	}
	return b.GetCurrentSchemaName(), table, nil
}

func nonEmpty(values []string) []string {
	out := values[:0:0]
	for _, value := range values {
		if value != "" {
			out = append(out, value)
		}
	}
	return out
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// truthy reads the value a driver returns for an "exists" query, which is a
// bool on one engine, an int on another and a string on a third.
func truthy(value any) bool {
	switch v := value.(type) {
	case nil:
		return false
	case bool:
		return v
	case int:
		return v != 0
	case int64:
		return v != 0
	case float64:
		return v != 0
	case []byte:
		return len(v) > 0 && string(v) != "0"
	case string:
		return v != "" && v != "0"
	default:
		return false
	}
}
