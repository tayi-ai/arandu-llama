package schema

// ColumnDefinition is what every column method on the Blueprint returns, and it
// is what the migration reads like:
// String("email", 255).Nullable().Default("").Comment("").
//
// The attribute and the fluent setter would share a name, and Go has one
// namespace for both, so every attribute is an unexported field with a Get
// prefixed getter -- the same rule query.Builder follows for GetFrom and
// GetLimit.
//
// The index markers (Index, Unique, Primary, FullText, SpatialIndex,
// VectorIndex) are any because each of three values means something different:
// nothing means "index it, name it by convention", a string names it, and false
// on a changed column drops the index it would otherwise have had.
type ColumnDefinition struct {
	name string
	typ  string

	length             *int
	fixed              bool
	precision          *int
	total              int
	places             int
	allowed            []string
	subtype            string
	srid               int
	dimensions         *int
	definition         string
	expression         string
	fullTypeDefinition string

	autoIncrement      bool
	unsigned           bool
	nullable           *bool
	def                any
	onUpdate           any
	useCurrent         bool
	useCurrentOnUpdate bool

	change   bool
	renameTo string
	after    *string
	first    *bool

	invisible *bool
	comment   *string
	charset   string
	collation string

	virtualAs     any
	virtualAsSet  bool
	virtualAsJSON string
	storedAs      any
	storedAsSet   bool
	storedAsJSON  string
	generatedAs   any
	always        bool

	startingValue *int
	from          *int

	instant bool
	lock    string

	index        any
	unique       any
	primary      any
	fullText     any
	spatialIndex any
	vectorIndex  any

	table                 string
	referencesModelColumn string
}

// GetName returns the column's name.
func (c *ColumnDefinition) GetName() string { return c.name }

// Type overrides the column's type.
func (c *ColumnDefinition) Type(typ string) *ColumnDefinition { c.typ = typ; return c }

// GetType returns the column's type.
func (c *ColumnDefinition) GetType() string { return c.typ }

// GetLength returns the column's length, or nil if none was set.
func (c *ColumnDefinition) GetLength() *int { return c.length }

// GetFixed reports whether the binary column has a fixed width.
func (c *ColumnDefinition) GetFixed() bool { return c.fixed }

// GetPrecision returns the column's precision, or nil if none was set.
func (c *ColumnDefinition) GetPrecision() *int { return c.precision }

// GetTotal returns the decimal column's total digits.
func (c *ColumnDefinition) GetTotal() int { return c.total }

// GetPlaces returns the decimal column's decimal places.
func (c *ColumnDefinition) GetPlaces() int { return c.places }

// GetAllowed returns the values an enum or set column accepts.
func (c *ColumnDefinition) GetAllowed() []string { return c.allowed }

// GetSubtype returns the subtype of a geometry or geography column.
func (c *ColumnDefinition) GetSubtype() string { return c.subtype }

// GetSRID returns the geometry or geography column's spatial reference
// identifier (SRID).
func (c *ColumnDefinition) GetSRID() int { return c.srid }

// GetDimensions returns the dimensions of a vector column, or nil if none
// was set.
func (c *ColumnDefinition) GetDimensions() *int { return c.dimensions }

// GetDefinition returns the definition of a raw column.
func (c *ColumnDefinition) GetDefinition() string { return c.definition }

// GetExpression returns the expression of a computed column.
func (c *ColumnDefinition) GetExpression() string { return c.expression }

// GetFullTypeDefinition returns the full type definition BlueprintState
// fills in from the server, so a SQLite table rebuild can repeat a column
// it did not write.
func (c *ColumnDefinition) GetFullTypeDefinition() string { return c.fullTypeDefinition }

// AutoIncrement marks the column as auto-incrementing.
func (c *ColumnDefinition) AutoIncrement() *ColumnDefinition { c.autoIncrement = true; return c }

// GetAutoIncrement reports whether the column auto-increments.
func (c *ColumnDefinition) GetAutoIncrement() bool { return c.autoIncrement }

// Unsigned marks the column as unsigned.
func (c *ColumnDefinition) Unsigned() *ColumnDefinition { c.unsigned = true; return c }

// GetUnsigned reports whether the column is unsigned.
func (c *ColumnDefinition) GetUnsigned() bool { return c.unsigned }

// Nullable allows the column to be null. With no argument it allows null;
// Nullable(false) forbids it.
func (c *ColumnDefinition) Nullable(value ...bool) *ColumnDefinition {
	v := true
	if len(value) > 0 {
		v = value[0]
	}
	c.nullable = &v
	return c
}

// GetNullable returns whether the column is nullable, or nil if Nullable was
// never called. The pointer distinguishes the two: Postgres emits nothing
// for an unset nullable and "not null" for one explicitly set to false.
func (c *ColumnDefinition) GetNullable() *bool { return c.nullable }

// Default sets the column's default value. A query Expression passes
// through to the SQL untouched; anything else is quoted.
func (c *ColumnDefinition) Default(value any) *ColumnDefinition { c.def = value; return c }

// GetDefault returns the column's default value.
func (c *ColumnDefinition) GetDefault() any { return c.def }

// OnUpdate sets the MySQL "on update" column modifier, which typeTimestamp
// sets from UseCurrentOnUpdate.
func (c *ColumnDefinition) OnUpdate(value any) *ColumnDefinition { c.onUpdate = value; return c }

// GetOnUpdate returns the column's "on update" modifier.
func (c *ColumnDefinition) GetOnUpdate() any { return c.onUpdate }

// UseCurrent sets the column's default to the current timestamp.
func (c *ColumnDefinition) UseCurrent() *ColumnDefinition { c.useCurrent = true; return c }

// GetUseCurrent reports whether the column defaults to the current
// timestamp.
func (c *ColumnDefinition) GetUseCurrent() bool { return c.useCurrent }

// UseCurrentOnUpdate sets the column to update to the current timestamp on
// row update.
func (c *ColumnDefinition) UseCurrentOnUpdate() *ColumnDefinition {
	c.useCurrentOnUpdate = true
	return c
}

// GetUseCurrentOnUpdate reports whether the column updates to the current
// timestamp on row update.
func (c *ColumnDefinition) GetUseCurrentOnUpdate() bool { return c.useCurrentOnUpdate }

// Change marks the column as replacing one that already exists.
func (c *ColumnDefinition) Change() *ColumnDefinition { c.change = true; return c }

// GetChange reports whether this definition replaces an existing column.
func (c *ColumnDefinition) GetChange() bool { return c.change }

// GetRenameTo returns the new name a change command carries when the column
// is renamed in the same statement.
func (c *ColumnDefinition) GetRenameTo() string { return c.renameTo }

// After places the column after column in the table. An empty column
// clears the modifier, which is what Blueprint's morphs family relies on.
func (c *ColumnDefinition) After(column string) *ColumnDefinition {
	if column == "" {
		c.after = nil
		return c
	}
	c.after = &column
	return c
}

// GetAfter returns the name of the column this one is placed after, or nil
// if none was set.
func (c *ColumnDefinition) GetAfter() *string { return c.after }

// First places the column first in the table.
func (c *ColumnDefinition) First(value ...bool) *ColumnDefinition {
	v := true
	if len(value) > 0 {
		v = value[0]
	}
	c.first = &v
	return c
}

// GetFirst returns whether the column is placed first in the table, or nil
// if that was never set.
func (c *ColumnDefinition) GetFirst() *bool { return c.first }

// Invisible marks the column as left out of "select *" on MySQL.
func (c *ColumnDefinition) Invisible(value ...bool) *ColumnDefinition {
	v := true
	if len(value) > 0 {
		v = value[0]
	}
	c.invisible = &v
	return c
}

// GetInvisible returns whether the column is invisible, or nil if that was
// never set.
func (c *ColumnDefinition) GetInvisible() *bool { return c.invisible }

// Comment sets the column's comment.
func (c *ColumnDefinition) Comment(comment string) *ColumnDefinition {
	c.comment = &comment
	return c
}

// GetComment returns the column's comment, or nil if none was set. It is a
// pointer because Postgres writes "comment on column ... is NULL" to clear
// one, which is a different statement from writing no comment at all.
func (c *ColumnDefinition) GetComment() *string { return c.comment }

// Charset sets the column's character set.
func (c *ColumnDefinition) Charset(charset string) *ColumnDefinition { c.charset = charset; return c }

// GetCharset returns the column's character set.
func (c *ColumnDefinition) GetCharset() string { return c.charset }

// Collation sets the column's collation.
func (c *ColumnDefinition) Collation(collation string) *ColumnDefinition {
	c.collation = collation
	return c
}

// GetCollation returns the column's collation.
func (c *ColumnDefinition) GetCollation() string { return c.collation }

// VirtualAs makes the column a generated column computed on read from
// expression. The expression may be a string or a query Expression.
func (c *ColumnDefinition) VirtualAs(expression any) *ColumnDefinition {
	c.virtualAs = expression
	c.virtualAsSet = true
	return c
}

// HasVirtualAs reports whether VirtualAs was called on this definition.
//
// On a changed column, that distinction decides between leaving the
// generation expression alone and dropping it. A nil virtualAs field alone
// cannot say whether it was never set or set to nil, so the setter records
// that it ran.
func (c *ColumnDefinition) HasVirtualAs() bool { return c.virtualAsSet }

// GetVirtualAs returns the column's virtual generation expression.
func (c *ColumnDefinition) GetVirtualAs() any { return c.virtualAs }

// GetVirtualAsJSON returns the JSON selector form the MySQL and SQLite
// grammars unwrap before generating the column.
func (c *ColumnDefinition) GetVirtualAsJSON() string { return c.virtualAsJSON }

// StoredAs makes the column a generated column computed on write from
// expression, and stored.
func (c *ColumnDefinition) StoredAs(expression any) *ColumnDefinition {
	c.storedAs = expression
	c.storedAsSet = true
	return c
}

// HasStoredAs reports whether StoredAs was called on this definition. See
// HasVirtualAs for why the setter has to record that it ran.
func (c *ColumnDefinition) HasStoredAs() bool { return c.storedAsSet }

// GetStoredAs returns the column's stored generation expression.
func (c *ColumnDefinition) GetStoredAs() any { return c.storedAs }

// GetStoredAsJSON returns the JSON selector form of the column's stored
// generation expression.
func (c *ColumnDefinition) GetStoredAsJSON() string { return c.storedAsJSON }

// GeneratedAs makes the column a Postgres identity column. With no argument
// it is the bare "generated by default as identity"; with one it carries
// the sequence options.
func (c *ColumnDefinition) GeneratedAs(expression ...any) *ColumnDefinition {
	if len(expression) == 0 {
		c.generatedAs = true
		return c
	}
	c.generatedAs = expression[0]
	return c
}

// GetGeneratedAs returns the column's identity generation expression.
func (c *ColumnDefinition) GetGeneratedAs() any { return c.generatedAs }

// Always marks the identity column as "generated always" rather than
// "generated by default".
func (c *ColumnDefinition) Always(value ...bool) *ColumnDefinition {
	v := true
	if len(value) > 0 {
		v = value[0]
	}
	c.always = v
	return c
}

// GetAlways reports whether the identity column is "generated always"
// rather than "generated by default".
func (c *ColumnDefinition) GetAlways() bool { return c.always }

// StartingValue sets the identity column's starting value.
func (c *ColumnDefinition) StartingValue(value int) *ColumnDefinition {
	c.startingValue = &value
	return c
}

// GetStartingValue returns the identity column's starting value, or nil if
// none was set.
func (c *ColumnDefinition) GetStartingValue() *int { return c.startingValue }

// From sets the identity column's starting value. It is the other spelling
// of StartingValue.
func (c *ColumnDefinition) From(startingValue int) *ColumnDefinition {
	c.from = &startingValue
	return c
}

// GetFrom returns the starting value set via From, or nil if none was set.
func (c *ColumnDefinition) GetFrom() *int { return c.from }

// Instant marks the operation to use MySQL's algorithm=instant.
func (c *ColumnDefinition) Instant() *ColumnDefinition { c.instant = true; return c }

// GetInstant reports whether the operation uses MySQL's algorithm=instant.
func (c *ColumnDefinition) GetInstant() bool { return c.instant }

// Lock sets the MySQL DDL lock mode, one of none, shared, default or
// exclusive.
func (c *ColumnDefinition) Lock(mode string) *ColumnDefinition { c.lock = mode; return c }

// GetLock returns the column's MySQL DDL lock mode.
func (c *ColumnDefinition) GetLock() string { return c.lock }

// Index adds an index on the column. With no argument the index is named by
// convention; with a string it is named; with false on a changed column the
// conventional index is dropped.
func (c *ColumnDefinition) Index(name ...any) *ColumnDefinition {
	c.index = indexMarker(name)
	return c
}

// GetIndex returns the value Index set: true, a name, or false.
func (c *ColumnDefinition) GetIndex() any { return c.index }

// Unique adds a unique index on the column, on the same three argument rule
// as Index.
func (c *ColumnDefinition) Unique(name ...any) *ColumnDefinition {
	c.unique = indexMarker(name)
	return c
}

// GetUnique returns the value Unique set: true, a name, or false.
func (c *ColumnDefinition) GetUnique() any { return c.unique }

// Primary adds a primary key on the column, on the same three argument rule
// as Index.
func (c *ColumnDefinition) Primary(name ...any) *ColumnDefinition {
	c.primary = indexMarker(name)
	return c
}

// GetPrimary returns the value Primary set: true, a name, or false.
func (c *ColumnDefinition) GetPrimary() any { return c.primary }

// FullText adds a full-text index on the column, on the same three argument
// rule as Index.
func (c *ColumnDefinition) FullText(name ...any) *ColumnDefinition {
	c.fullText = indexMarker(name)
	return c
}

// GetFullText returns the value FullText set: true, a name, or false.
func (c *ColumnDefinition) GetFullText() any { return c.fullText }

// SpatialIndex adds a spatial index on the column, on the same three
// argument rule as Index.
func (c *ColumnDefinition) SpatialIndex(name ...any) *ColumnDefinition {
	c.spatialIndex = indexMarker(name)
	return c
}

// GetSpatialIndex returns the value SpatialIndex set: true, a name, or
// false.
func (c *ColumnDefinition) GetSpatialIndex() any { return c.spatialIndex }

// VectorIndex adds a vector index on the column, on the same three argument
// rule as Index.
func (c *ColumnDefinition) VectorIndex(name ...any) *ColumnDefinition {
	c.vectorIndex = indexMarker(name)
	return c
}

// GetVectorIndex returns the value VectorIndex set: true, a name, or false.
func (c *ColumnDefinition) GetVectorIndex() any { return c.vectorIndex }

// Table sets the table a foreign ID column references. ForeignIDFor sets it
// so that Constrained knows the table without guessing it from the column
// name.
func (c *ColumnDefinition) Table(table string) *ColumnDefinition { c.table = table; return c }

// GetTable returns the table a foreign ID column references.
func (c *ColumnDefinition) GetTable() string { return c.table }

// ReferencesModelColumn sets the key name Constrained references when the
// model does not key on id.
func (c *ColumnDefinition) ReferencesModelColumn(column string) *ColumnDefinition {
	c.referencesModelColumn = column
	return c
}

// GetReferencesModelColumn returns the key name Constrained references when
// the model does not key on id.
func (c *ColumnDefinition) GetReferencesModelColumn() string { return c.referencesModelColumn }

// indexMarker turns the variadic argument of the fluent index setters into
// the value stored on the column: no argument becomes true, and the first
// argument given is used otherwise. Extra arguments are ignored rather than
// causing a panic, because a panic in a migration helps nobody.
func indexMarker(name []any) any {
	if len(name) == 0 {
		return true
	}
	return name[0]
}
