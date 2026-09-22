package schema

// ForeignKeyDefinition is what Blueprint.Foreign returns.
//
// Like IndexDefinition it is a typed handle over the blueprint command it was
// appended as, so chaining onto it changes the command the grammar compiles.
type ForeignKeyDefinition struct{ c *Command }

// GetCommand returns the blueprint command this definition is a handle on.
func (d *ForeignKeyDefinition) GetCommand() *Command { return d.c }

// References sets the column or columns on the referenced table that this
// foreign key points to.
func (d *ForeignKeyDefinition) References(columns ...string) *ForeignKeyDefinition {
	d.c.References = make([]any, len(columns))
	for i, column := range columns {
		d.c.References[i] = column
	}
	return d
}

// On sets the referenced table.
func (d *ForeignKeyDefinition) On(table string) *ForeignKeyDefinition {
	d.c.On = table
	return d
}

// OnDelete sets the action to take when the referenced row is deleted:
// cascade, restrict, set null or no action.
func (d *ForeignKeyDefinition) OnDelete(action string) *ForeignKeyDefinition {
	d.c.OnDelete = action
	return d
}

// OnUpdate sets the action to take when the referenced row is updated, on the
// same four actions as OnDelete.
func (d *ForeignKeyDefinition) OnUpdate(action string) *ForeignKeyDefinition {
	d.c.OnUpdate = action
	return d
}

// CascadeOnDelete sets OnDelete to cascade.
func (d *ForeignKeyDefinition) CascadeOnDelete() *ForeignKeyDefinition {
	return d.OnDelete("cascade")
}

// RestrictOnDelete sets OnDelete to restrict.
func (d *ForeignKeyDefinition) RestrictOnDelete() *ForeignKeyDefinition {
	return d.OnDelete("restrict")
}

// NullOnDelete sets OnDelete to set null.
func (d *ForeignKeyDefinition) NullOnDelete() *ForeignKeyDefinition {
	return d.OnDelete("set null")
}

// NoActionOnDelete sets OnDelete to no action.
func (d *ForeignKeyDefinition) NoActionOnDelete() *ForeignKeyDefinition {
	return d.OnDelete("no action")
}

// CascadeOnUpdate sets OnUpdate to cascade.
func (d *ForeignKeyDefinition) CascadeOnUpdate() *ForeignKeyDefinition {
	return d.OnUpdate("cascade")
}

// RestrictOnUpdate sets OnUpdate to restrict.
func (d *ForeignKeyDefinition) RestrictOnUpdate() *ForeignKeyDefinition {
	return d.OnUpdate("restrict")
}

// NullOnUpdate sets OnUpdate to set null.
func (d *ForeignKeyDefinition) NullOnUpdate() *ForeignKeyDefinition {
	return d.OnUpdate("set null")
}

// NoActionOnUpdate sets OnUpdate to no action.
func (d *ForeignKeyDefinition) NoActionOnUpdate() *ForeignKeyDefinition {
	return d.OnUpdate("no action")
}

// Deferrable sets whether the constraint may be checked at the end of the
// transaction rather than at each statement.
func (d *ForeignKeyDefinition) Deferrable(value ...bool) *ForeignKeyDefinition {
	d.c.Deferrable = boolArg(value)
	return d
}

// InitiallyImmediate sets whether a deferrable constraint starts out checked
// per statement.
func (d *ForeignKeyDefinition) InitiallyImmediate(value ...bool) *ForeignKeyDefinition {
	d.c.InitiallyImmediate = boolArg(value)
	return d
}

// NotValid sets the Postgres "not valid" constraint modifier: existing rows
// are not checked when the constraint is added.
func (d *ForeignKeyDefinition) NotValid(value ...bool) *ForeignKeyDefinition {
	d.c.NotValid = boolArg(value)
	return d
}

// Lock sets the MySQL DDL lock mode.
func (d *ForeignKeyDefinition) Lock(mode string) *ForeignKeyDefinition {
	d.c.Lock = mode
	return d
}
