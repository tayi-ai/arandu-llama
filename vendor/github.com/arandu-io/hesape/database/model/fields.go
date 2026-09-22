package model

import (
	"database/sql"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"time"
)

// field is one column of the entity struct: the name it has in the table and
// where to find it in the struct.
//
// A row is a struct here rather than a map, and this is the translation between
// the two.
type field struct {
	column string
	index  []int
}

// entitySchema is what reflection over an entity type answers, kept so that it
// is answered once per type rather than once per row.
//
// It holds the columns twice: ordered, because GetAttributes walks them and a
// map has no order, and by name, because every read and every write of a single
// attribute looks one up. The list alone used to serve both, and the lookup was
// a linear scan of it -- so hydrating a row of C columns into a struct of F
// fields cost C times F string comparisons, and GetDirty, which looks up every
// key it just produced, cost F squared.
type entitySchema struct {
	fields []field
	byName map[string]int
}

// fieldCache holds the schema per entity type. Reflection over a struct is
// the same answer every time, and Hydrate asks for it once per row.
var fieldCache sync.Map // reflect.Type -> *entitySchema

// fieldsOf returns the columns of an entity type.
//
// Only exported fields are columns, and that is the design rather than a
// limitation of reflection: an unexported field cannot be written from outside
// its package, which is the mass-assignment allowlist the compiler enforces
// (see the package comment).
//
// The column name is the `db` tag up to its first comma; without a tag it is the
// field name in snake case. A tag of "-" means the field is not a column at all.
//
// An embedded exported struct is flattened, so the columns of an embedded type
// are columns of the outer one. An embedded struct with a `db` tag is treated as
// a column of its own instead, which is how a driver-level type
// (sql.NullString, a custom scanner) stays one value.
func fieldsOf(t reflect.Type) []field { return schemaOf(t).fields }

// schemaOf returns the cached schema of an entity type, building it on first
// sight.
func schemaOf(t reflect.Type) *entitySchema {
	if cached, ok := fieldCache.Load(t); ok {
		return cached.(*entitySchema)
	}
	fields := collectFields(t, nil)
	byName := make(map[string]int, len(fields))
	for i, f := range fields {
		// First declaration wins, which is what the linear scan did: an
		// embedded type's column does not shadow the outer type's.
		if _, taken := byName[f.column]; !taken {
			byName[f.column] = i
		}
	}
	schema := &entitySchema{fields: fields, byName: byName}
	fieldCache.Store(t, schema)
	return schema
}

func collectFields(t reflect.Type, prefix []int) []field {
	var out []field
	for i := range t.NumField() {
		f := t.Field(i)
		if f.PkgPath != "" { // unexported: not a column, by design
			continue
		}
		tag, tagged := f.Tag.Lookup("db")
		name := strings.Split(tag, ",")[0]
		if name == "-" {
			continue
		}
		if isEmbeddedModel(f) {
			continue
		}
		index := append(append([]int(nil), prefix...), i)
		if f.Anonymous && !tagged {
			inner := f.Type
			if inner.Kind() == reflect.Pointer {
				inner = inner.Elem()
			}
			if inner.Kind() == reflect.Struct && inner != reflect.TypeOf(time.Time{}) {
				out = append(out, collectFields(inner, index)...)
				continue
			}
		}
		if name == "" {
			name = columnName(f.Name)
		}
		out = append(out, field{column: name, index: index})
	}
	return out
}

// isEmbeddedModel reports whether f is the Model[T] an entity embeds.
//
// Its fields are the model's configuration -- the table, the primary key, the
// grammar, the back pointer to the entity itself -- and every one of them is
// exported, because a Go value has no subtype to override them in. Walked as an
// embedded struct they become columns: an insert on a User that embeds
// Model[User] tried to write table, primary_key, entity and grammar, alongside
// the two the developer declared.
//
// The test is the type's own identity rather than its name, so a field the
// application happens to call Model is a column like any other, and an entity
// that embeds a Model[T] of any T is skipped whatever T is. That is why the name
// is matched with a prefix: Go spells the instantiated type Model[pkg.User], and
// there is no T here to compare against.
func isEmbeddedModel(f reflect.StructField) bool {
	return f.Anonymous &&
		f.Type.Kind() == reflect.Struct &&
		f.Type.PkgPath() == modelPackage &&
		strings.HasPrefix(f.Type.Name(), "Model[")
}

// modelPackage is this package's import path, read off a type rather than
// written as a string: a package that moves takes the constant with it.
var modelPackage = reflect.TypeFor[Model[struct{}]]().PkgPath()

// columnName is the column a field with no tag gets: the field name in snake
// case, with an initialism kept whole.
//
// It is not str.Snake: that puts a delimiter before every capital, and
// Snake("ID", "_") is "i_d". A run of capitals is one word here -- ID is id,
// UserID is user_id, HTTPServer is http_server. The `db` tag is always there for
// the rest.
func columnName(name string) string {
	runes := []rune(name)
	var out []rune
	for i, r := range runes {
		upper := r >= 'A' && r <= 'Z'
		if upper && i > 0 {
			previousLower := runes[i-1] >= 'a' && runes[i-1] <= 'z' || runes[i-1] >= '0' && runes[i-1] <= '9'
			nextLower := i+1 < len(runes) && runes[i+1] >= 'a' && runes[i+1] <= 'z'
			if previousLower || nextLower {
				out = append(out, '_')
			}
		}
		out = append(out, toLowerRune(r))
	}
	return string(out)
}

func toLowerRune(r rune) rune {
	if r >= 'A' && r <= 'Z' {
		return r + ('a' - 'A')
	}
	return r
}

// fieldByColumn finds the column in an entity type, reporting whether it exists.
func fieldByColumn(t reflect.Type, column string) (field, bool) {
	schema := schemaOf(t)
	i, ok := schema.byName[column]
	if !ok {
		return field{}, false
	}
	return schema.fields[i], true
}

// valueAt reads a column off an entity value.
//
// A nil pointer anywhere along an embedded chain reads as nil rather than
// panicking, because a half-filled struct is an ordinary state between Hydrate
// and Save.
func valueAt(entity reflect.Value, f field) any {
	v := entity
	for depth, i := range f.index {
		if v.Kind() == reflect.Pointer {
			if v.IsNil() {
				return nil
			}
			v = v.Elem()
		}
		_ = depth
		v = v.Field(i)
	}
	if !v.IsValid() {
		return nil
	}
	return v.Interface()
}

// settableAt returns the addressable destination for a column, allocating
// the nil pointers it walks through.
func settableAt(entity reflect.Value, f field) (reflect.Value, bool) {
	v := entity
	for _, i := range f.index {
		if v.Kind() == reflect.Pointer {
			if v.IsNil() {
				if !v.CanSet() {
					return reflect.Value{}, false
				}
				v.Set(reflect.New(v.Type().Elem()))
			}
			v = v.Elem()
		}
		v = v.Field(i)
	}
	if !v.CanSet() {
		return reflect.Value{}, false
	}
	return v, true
}

// timeLayouts are what a driver may hand back for a timestamp column.
//
// SQLite has no date type and returns text; the drivers that do have one return
// a time.Time and never reach this list.
var timeLayouts = []string{
	time.RFC3339Nano,
	time.RFC3339,
	"2006-01-02 15:04:05.999999999 -0700 MST",
	"2006-01-02T15:04:05",
	"2006-01-02 15:04:05.999999",
	"2006-01-02 15:04:05",
	"2006-01-02",
}

// assign writes a value read from the database into a struct field.
//
// It is the half of casting that Go still needs. The other half -- int, bool,
// datetime -- is the field's own type, which the compiler already knows.
//
// A conversion that would silently produce the wrong value is refused rather
// than performed. Go converts an int to a string as a rune, so a numeric column
// landing in a string field would read as "\x07" instead of "7"; that is an
// error here.
//
// A field whose type knows how to read itself out of a column -- anything with
// a Scan method -- is handed the driver's value and left to it. That is the
// whole of array, JSON and enum support, and it is deliberately not a list of
// encodings this package understands: a type that spells its own column knows
// what is in it, and a second opinion here would be the encoding decided twice.
func assign(dst reflect.Value, value any) error {
	if value == nil {
		dst.Set(reflect.Zero(dst.Type()))
		return nil
	}

	src := reflect.ValueOf(value)

	if src.Type().AssignableTo(dst.Type()) {
		dst.Set(src)
		return nil
	}

	if dst.Kind() == reflect.Pointer {
		inner := reflect.New(dst.Type().Elem())
		if err := assign(inner.Elem(), value); err != nil {
			return err
		}
		dst.Set(inner)
		return nil
	}

	// A driver that hands back a pointer or a wrapped value is unwrapped once.
	if src.Kind() == reflect.Pointer {
		if src.IsNil() {
			dst.Set(reflect.Zero(dst.Type()))
			return nil
		}
		return assign(dst, src.Elem().Interface())
	}

	// A field that reads itself does so, before any conversion here is tried.
	// It is the type's own answer about its own column: the slice that is a
	// JSON array, the struct that is a document, and the enum that refuses a
	// value it does not know instead of accepting whatever string came back.
	//
	// A NULL is not routed here. It is answered above, as the field's zero, so
	// that a type whose Scan speaks only about the values it has does not have
	// to carry a branch for the absence of one.
	if dst.CanAddr() {
		if scanner, ok := dst.Addr().Interface().(sql.Scanner); ok {
			if err := scanner.Scan(value); err != nil {
				return fmt.Errorf("model: reading a %s out of the column: %w", dst.Type(), err)
			}
			return nil
		}
	}

	switch dst.Type() {
	case reflect.TypeOf(time.Time{}):
		return assignTime(dst, value)
	}

	switch dst.Kind() {
	case reflect.String:
		switch v := value.(type) {
		case []byte:
			dst.SetString(string(v))
			return nil
		case string:
			dst.SetString(v)
			return nil
		}
	case reflect.Bool:
		switch v := value.(type) {
		case int64:
			dst.SetBool(v != 0)
			return nil
		case float64:
			dst.SetBool(v != 0)
			return nil
		case []byte:
			dst.SetBool(len(v) == 1 && v[0] != 0)
			return nil
		}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
		reflect.Float32, reflect.Float64:
		if src.Type().ConvertibleTo(dst.Type()) && isNumeric(src.Kind()) {
			dst.Set(src.Convert(dst.Type()))
			return nil
		}
	case reflect.Slice:
		if dst.Type().Elem().Kind() == reflect.Uint8 {
			if s, ok := value.(string); ok {
				dst.SetBytes([]byte(s))
				return nil
			}
		}
	}

	return fmt.Errorf("model: cannot put a %T into a %s field", value, dst.Type())
}

func assignTime(dst reflect.Value, value any) error {
	switch v := value.(type) {
	case time.Time:
		dst.Set(reflect.ValueOf(v))
		return nil
	case []byte:
		return assignTime(dst, string(v))
	case string:
		for _, layout := range timeLayouts {
			if parsed, err := time.Parse(layout, v); err == nil {
				dst.Set(reflect.ValueOf(parsed))
				return nil
			}
		}
		return fmt.Errorf("model: %q is not a timestamp in any layout a driver writes", v)
	}
	return fmt.Errorf("model: cannot put a %T into a time.Time field", value)
}

func isNumeric(k reflect.Kind) bool {
	switch k {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
		reflect.Float32, reflect.Float64:
		return true
	}
	return false
}

// isZero reports whether a value is its type's zero value, which is how a model
// decides whether an id was set without knowing the id's type.
func isZero(value any) bool {
	if value == nil {
		return true
	}
	v := reflect.ValueOf(value)
	return v.IsZero()
}
