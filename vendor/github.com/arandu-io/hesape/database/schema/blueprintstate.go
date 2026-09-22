package schema

import (
	"context"
	"strings"

	"github.com/arandu-io/hesape/database/query"
)

// BlueprintState is the shape a table already has, kept up to date as a
// blueprint's commands are compiled.
//
// SQLite cannot alter most of a table, so the SQLite grammar rebuilds it: it
// creates a table with the shape the blueprint asks for, copies the rows over,
// drops the original and renames, and this is what it reads from the server to
// do it.
type BlueprintState struct {
	blueprint  *Blueprint
	connection Connection

	columns     []*ColumnDefinition
	primaryKey  *IndexDefinition
	indexes     []*IndexDefinition
	foreignKeys []*ForeignKeyDefinition
}

// NewBlueprintState builds a BlueprintState for blueprint's table.
//
// It takes a context because the constructor reads three catalogues off the
// server -- the table's columns, indexes and foreign keys. See Blueprint.ToSQL.
func NewBlueprintState(ctx context.Context, blueprint *Blueprint, connection Connection) (*BlueprintState, error) {
	s := &BlueprintState{blueprint: blueprint, connection: connection}
	builder := NewBuilder(connection)
	table := blueprint.GetTable()

	columns, err := builder.GetColumns(ctx, table)
	if err != nil {
		return nil, err
	}
	for _, column := range columns {
		definition := &ColumnDefinition{
			name:               column.Name,
			typ:                column.TypeName,
			fullTypeDefinition: column.Type,
			autoIncrement:      column.AutoIncrement,
			collation:          column.Collation,
		}
		nullable := column.Nullable
		definition.nullable = &nullable
		if column.Comment != "" {
			comment := column.Comment
			definition.comment = &comment
		}
		if column.Default != nil {
			definition.def = query.Raw(wrapParens(stringify(column.Default)))
		}
		if column.Generation != nil {
			switch column.Generation.Type {
			case "virtual":
				definition.virtualAs = column.Generation.Expression
			case "stored":
				definition.storedAs = column.Generation.Expression
			}
		}
		s.columns = append(s.columns, definition)
	}

	indexes, err := builder.GetIndexes(ctx, table)
	if err != nil {
		return nil, err
	}
	for _, index := range indexes {
		name := "index"
		switch {
		case index.Primary:
			name = "primary"
		case index.Unique:
			name = "unique"
		}
		command := NewCommand(name)
		command.Index = index.Name
		command.Columns = toAnyList(index.Columns)

		if index.Primary {
			if s.primaryKey == nil {
				s.primaryKey = &IndexDefinition{c: command}
			}
			continue
		}
		s.indexes = append(s.indexes, &IndexDefinition{c: command})
	}

	foreignKeys, err := builder.GetForeignKeys(ctx, table)
	if err != nil {
		return nil, err
	}
	for _, foreignKey := range foreignKeys {
		command := NewCommand("foreign")
		command.Columns = toAnyList(foreignKey.Columns)
		command.On = query.Raw(foreignKey.ForeignTable)
		command.References = toAnyList(foreignKey.ForeignColumns)
		command.OnUpdate = foreignKey.OnUpdate
		command.OnDelete = foreignKey.OnDelete
		s.foreignKeys = append(s.foreignKeys, &ForeignKeyDefinition{c: command})
	}

	return s, nil
}

// GetPrimaryKey returns the table's primary key index, or nil if it has none.
func (s *BlueprintState) GetPrimaryKey() *IndexDefinition { return s.primaryKey }

// GetColumns returns the table's current columns.
func (s *BlueprintState) GetColumns() []*ColumnDefinition { return s.columns }

// GetIndexes returns the table's current indexes, other than its primary key.
func (s *BlueprintState) GetIndexes() []*IndexDefinition { return s.indexes }

// GetForeignKeys returns the table's current foreign keys.
func (s *BlueprintState) GetForeignKeys() []*ForeignKeyDefinition { return s.foreignKeys }

// Update applies one compiled command to the remembered shape, so that the
// rebuild statement sees the table as it will be rather than as it was.
func (s *BlueprintState) Update(command *Command) {
	switch command.Name {
	case "alter":
		// Already handled...

	case "add":
		s.columns = append(s.columns, command.Column)

	case "change":
		for i, column := range s.columns {
			if column.name == command.Column.name {
				s.columns[i] = command.Column
				break
			}
		}

	case "renameColumn":
		for _, column := range s.columns {
			if column.name == command.From {
				column.name = command.To
				break
			}
		}
		if s.primaryKey != nil {
			s.primaryKey.c.Columns = replaceColumns(s.primaryKey.c.Columns, command.From, command.To)
		}
		for _, index := range s.indexes {
			index.c.Columns = replaceColumns(index.c.Columns, command.From, command.To)
		}
		for _, foreignKey := range s.foreignKeys {
			foreignKey.c.Columns = replaceColumns(foreignKey.c.Columns, command.From, command.To)
		}

	case "dropColumn":
		dropped := make(map[string]bool, len(command.Columns))
		for _, column := range command.Columns {
			dropped[stringify(column)] = true
		}
		kept := s.columns[:0]
		for _, column := range s.columns {
			if !dropped[column.name] {
				kept = append(kept, column)
			}
		}
		s.columns = kept

	case "primary":
		s.primaryKey = &IndexDefinition{c: command}

	case "unique", "index":
		s.indexes = append(s.indexes, &IndexDefinition{c: command})

	case "renameIndex":
		for _, index := range s.indexes {
			if index.c.Index == command.From {
				index.c.Index = command.To
				break
			}
		}

	case "foreign":
		s.foreignKeys = append(s.foreignKeys, &ForeignKeyDefinition{c: command})

	case "dropPrimary":
		s.primaryKey = nil

	case "dropIndex", "dropUnique":
		kept := s.indexes[:0]
		for _, index := range s.indexes {
			if index.c.Index != command.Index {
				kept = append(kept, index)
			}
		}
		s.indexes = kept

	case "dropForeign":
		kept := s.foreignKeys[:0]
		for _, foreignKey := range s.foreignKeys {
			if !sameColumns(foreignKey.c.Columns, command.Columns) {
				kept = append(kept, foreignKey)
			}
		}
		s.foreignKeys = kept
	}
}

// replaceColumns returns a copy of columns, with occurrences of from replaced
// by to in every string element; non-string elements pass through unchanged.
func replaceColumns(columns []any, from, to string) []any {
	out := make([]any, len(columns))
	for i, column := range columns {
		if name, ok := column.(string); ok {
			out[i] = strings.ReplaceAll(name, from, to)
			continue
		}
		out[i] = column
	}
	return out
}

func sameColumns(a, b []any) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if stringify(a[i]) != stringify(b[i]) {
			return false
		}
	}
	return true
}

// wrapParens wraps value in parentheses.
func wrapParens(value string) string { return "(" + value + ")" }
