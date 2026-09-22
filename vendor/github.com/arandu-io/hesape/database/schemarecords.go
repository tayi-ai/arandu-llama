package database

import (
	"strings"

	"github.com/arandu-io/hesape/database/query"
	"github.com/arandu-io/hesape/database/query/processors"
	"github.com/arandu-io/hesape/database/schema"
)

// The six processors, and the conversion from a normalised record to the typed
// struct the schema package answers.
//
// The normalising is not written here: it is the query processors' work, one per
// driver, and they already know that MySQL calls a column's nullability
// "IS_NULLABLE" and PostgreSQL calls it "nullable". What is here is the last
// step, from the record they answer to the struct the schema package declares --
// which is the whole reason the two halves could not be joined before. The
// processors speak map, the schema speaks struct, and neither was going to
// change to meet the other.

// processorFor picks the query processor by driver, so that the driver-specific
// normalising is reached rather than reimplemented.
func processorFor(driver string) interface {
	ProcessTables([]query.Record) []query.Record
	ProcessViews([]query.Record) []query.Record
	ProcessColumns([]query.Record, ...string) []query.Record
	ProcessIndexes([]query.Record) []query.Record
	ProcessForeignKeys([]query.Record) []query.Record
	ProcessSchemas([]query.Record) []query.Record
} {
	switch driver {
	case "pgsql", "postgres", "postgresql":
		return processors.NewPostgresProcessor()
	case "mysql", "mariadb":
		return processors.NewMySQLProcessor()
	default:
		return processors.NewSQLiteProcessor()
	}
}

// asSchemaRecords retypes the rows. query.Record and schema.Record are both
// map[string]any, and both are aliases, so this is a slice conversion and not a
// copy of every row.
func asSchemaRecords(rows []query.Record) []schema.Record {
	out := make([]schema.Record, 0, len(rows))
	for _, row := range rows {
		out = append(out, schema.Record(row))
	}
	return out
}

// asQueryRecords is the way back, for handing rows to a processor.
func asQueryRecords(rows []schema.Record) []query.Record {
	out := make([]query.Record, 0, len(rows))
	for _, row := range rows {
		out = append(out, query.Record(row))
	}
	return out
}

func (c *schemaConnection) processor() interface {
	ProcessTables([]query.Record) []query.Record
	ProcessViews([]query.Record) []query.Record
	ProcessColumns([]query.Record, ...string) []query.Record
	ProcessIndexes([]query.Record) []query.Record
	ProcessForeignKeys([]query.Record) []query.Record
	ProcessSchemas([]query.Record) []query.Record
} {
	return processorFor(strings.ToLower(c.connection.GetDriverName()))
}

// ProcessSchemas turns a schema listing into SchemaInfo.
func (c *schemaConnection) ProcessSchemas(rows []schema.Record) []schema.SchemaInfo {
	normalised := c.processor().ProcessSchemas(asQueryRecords(rows))
	out := make([]schema.SchemaInfo, 0, len(normalised))
	for _, row := range normalised {
		out = append(out, schema.SchemaInfo{
			Name:    text(row, "name"),
			Path:    text(row, "path"),
			Default: flag(row, "default"),
		})
	}
	return out
}

// ProcessTables turns a table listing into TableInfo.
func (c *schemaConnection) ProcessTables(rows []schema.Record) []schema.TableInfo {
	normalised := c.processor().ProcessTables(asQueryRecords(rows))
	out := make([]schema.TableInfo, 0, len(normalised))
	for _, row := range normalised {
		out = append(out, schema.TableInfo{
			Name:                text(row, "name"),
			Schema:              text(row, "schema"),
			SchemaQualifiedName: text(row, "schema_qualified_name"),
			Size:                number(row, "size"),
			Comment:             text(row, "comment"),
			Collation:           text(row, "collation"),
			Engine:              text(row, "engine"),
		})
	}
	return out
}

// ProcessViews turns a view listing into ViewInfo.
func (c *schemaConnection) ProcessViews(rows []schema.Record) []schema.ViewInfo {
	normalised := c.processor().ProcessViews(asQueryRecords(rows))
	out := make([]schema.ViewInfo, 0, len(normalised))
	for _, row := range normalised {
		out = append(out, schema.ViewInfo{
			Name:                text(row, "name"),
			Schema:              text(row, "schema"),
			SchemaQualifiedName: text(row, "schema_qualified_name"),
			Definition:          text(row, "definition"),
		})
	}
	return out
}

// ProcessColumns turns a column listing into ColumnInfo.
func (c *schemaConnection) ProcessColumns(rows []schema.Record) []schema.ColumnInfo {
	normalised := c.processor().ProcessColumns(asQueryRecords(rows))
	out := make([]schema.ColumnInfo, 0, len(normalised))
	for _, row := range normalised {
		column := schema.ColumnInfo{
			Name:          text(row, "name"),
			Type:          text(row, "type"),
			TypeName:      text(row, "type_name"),
			Nullable:      flag(row, "nullable"),
			Default:       row["default"],
			AutoIncrement: flag(row, "auto_increment"),
			Comment:       text(row, "comment"),
			Collation:     text(row, "collation"),
		}
		// The generation is a pair of columns or nothing at all, and nothing at
		// all is the common case -- so the pointer stays nil rather than
		// pointing at an empty struct a caller would have to tell apart from a
		// real one.
		if kind := text(row, "generation_type"); kind != "" {
			column.Generation = &schema.Generation{
				Type:       kind,
				Expression: text(row, "generation_expression"),
			}
		}
		out = append(out, column)
	}
	return out
}

// ProcessIndexes turns an index listing into IndexInfo.
func (c *schemaConnection) ProcessIndexes(rows []schema.Record) []schema.IndexInfo {
	normalised := c.processor().ProcessIndexes(asQueryRecords(rows))
	out := make([]schema.IndexInfo, 0, len(normalised))
	for _, row := range normalised {
		out = append(out, schema.IndexInfo{
			Name:    text(row, "name"),
			Columns: list(row, "columns"),
			Type:    text(row, "type"),
			Unique:  flag(row, "unique"),
			Primary: flag(row, "primary"),
		})
	}
	return out
}

// ProcessForeignKeys turns a foreign key listing into ForeignKeyInfo.
func (c *schemaConnection) ProcessForeignKeys(rows []schema.Record) []schema.ForeignKeyInfo {
	normalised := c.processor().ProcessForeignKeys(asQueryRecords(rows))
	out := make([]schema.ForeignKeyInfo, 0, len(normalised))
	for _, row := range normalised {
		out = append(out, schema.ForeignKeyInfo{
			Name:           text(row, "name"),
			Columns:        list(row, "columns"),
			ForeignSchema:  text(row, "foreign_schema"),
			ForeignTable:   text(row, "foreign_table"),
			ForeignColumns: list(row, "foreign_columns"),
			OnUpdate:       text(row, "on_update"),
			OnDelete:       text(row, "on_delete"),
		})
	}
	return out
}
