package schema

// IndexDefinition is what the index methods on a Blueprint return.
//
// It is a typed handle over the *Command that was appended, so chaining onto it
// changes the command the grammar will compile.
type IndexDefinition struct{ c *Command }

// GetCommand returns the blueprint command this definition is a handle on.
//
// The grammar compiles commands, so the handle has to be able to hand its own
// over.
func (d *IndexDefinition) GetCommand() *Command { return d.c }

// Algorithm sets the index algorithm: btree, hash, gin, gist.
func (d *IndexDefinition) Algorithm(algorithm string) *IndexDefinition {
	d.c.Algorithm = algorithm
	return d
}

// Deferrable sets whether the constraint may be checked at the end of the
// transaction rather than at each statement.
func (d *IndexDefinition) Deferrable(value ...bool) *IndexDefinition {
	d.c.Deferrable = boolArg(value)
	return d
}

// InitiallyImmediate sets whether a deferrable constraint starts out checked
// per statement.
func (d *IndexDefinition) InitiallyImmediate(value ...bool) *IndexDefinition {
	d.c.InitiallyImmediate = boolArg(value)
	return d
}

// Language sets the Postgres text search configuration a full text index is
// built with.
func (d *IndexDefinition) Language(language string) *IndexDefinition {
	d.c.Language = language
	return d
}

// Lock sets the MySQL DDL lock mode.
func (d *IndexDefinition) Lock(mode string) *IndexDefinition {
	d.c.Lock = mode
	return d
}

// NullsNotDistinct sets whether two null values collide in the unique index
// rather than both being allowed.
func (d *IndexDefinition) NullsNotDistinct(value ...bool) *IndexDefinition {
	d.c.NullsNotDistinct = boolArg(value)
	return d
}

// Online sets whether the index is built without locking the table.
func (d *IndexDefinition) Online(value ...bool) *IndexDefinition {
	v := true
	if len(value) > 0 {
		v = value[0]
	}
	d.c.Online = v
	return d
}

// boolArg turns a variadic bool parameter that defaults to true when omitted
// into the pointer the Command carries, which is how "not set" stays
// different from false.
func boolArg(value []bool) *bool {
	v := true
	if len(value) > 0 {
		v = value[0]
	}
	return &v
}
