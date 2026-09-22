package model

import (
	"errors"
	"reflect"
	"sync"
)

// The entity is the model, and this file is what makes that true at run time.
//
// An application writes:
//
//	type User struct {
//	    model.Model[User]
//
//	    ID    string `db:"id"`
//	    Name  string `db:"name"`
//	}
//
// and then user.Save(ctx, g) reaches the method through the embedded model. Go
// promotes the method, so that half is the compiler's. What the compiler cannot
// do is tell the embedded Model[User] which User it is inside of: an embedded
// value has no way to name the value that embeds it, and in PHP the same
// question is answered for free by $this.
//
// So it is answered here, once, at the moment the model is built: the instance
// allocates the T, takes the address of the Model[T] inside it, and points that
// model's Entity at the T. The model and the entity are one allocation seen from
// two sides after that -- reading user.Name and calling user.Save() reach the
// same bytes.
//
// A T that does not embed Model[T] still works, and is what the tests and the
// relation machinery use: the model is then a value of its own with an Entity
// beside it. Both shapes are built by the same constructor and the difference is
// where the model lives, never what it does.

// embedded is where Model[T] sits inside T, or -1 when T does not embed it.
//
// It is looked up once per type rather than per row. The scan is short, but
// hydration runs it for every row of every query, and a reflect walk per row is
// how a fixed cost becomes a variable one.
type embedded struct{ index int }

var embeddedCache sync.Map // reflect.Type -> embedded

// embeddedIndex reports where Model[T] is embedded in T.
//
// It matches on the field's type rather than on its name, because the field has
// no name: an embedded field is named after its type, and a struct that declared
// a regular field called Model would otherwise be mistaken for one that embeds.
func embeddedIndex[T any]() int {
	t := reflect.TypeFor[T]()
	if t.Kind() != reflect.Struct {
		return -1
	}
	if cached, ok := embeddedCache.Load(t); ok {
		return cached.(embedded).index
	}

	found := -1
	want := reflect.TypeFor[Model[T]]()
	for i := range t.NumField() {
		field := t.Field(i)
		if field.Anonymous && field.Type == want {
			found = i
			break
		}
	}
	embeddedCache.Store(t, embedded{index: found})
	return found
}

// modelIn returns the Model[T] embedded in entity, or nil when T does not embed
// one.
//
// The pointer is interior to the entity's allocation, which is what makes the
// two one thing: writing through it is writing into the entity.
func modelIn[T any](entity *T, index int) *Model[T] {
	if entity == nil || index < 0 {
		return nil
	}
	field := reflect.ValueOf(entity).Elem().Field(index)
	if !field.CanAddr() {
		return nil
	}
	model, _ := field.Addr().Interface().(*Model[T])
	return model
}

// entityIndex is where Model[T] sits inside T, resolved once per model rather
// than once per row.
//
// Hydration builds one instance per row and resets it once, and each of those
// asked embeddedIndex, which is two map lookups on the path a thousand-row query
// takes a thousand times. The answer depends only on T, so the model carries it.
func (m *Model[T]) entityIndex() int {
	if m.embedded == 0 {
		// Stored one past the index so the zero value means "not resolved yet",
		// which is what a Model[T] built as a literal has.
		m.embedded = embeddedIndex[T]() + 2
	}
	return m.embedded - 2
}

// newEntity allocates a T and returns it with the model inside it, when there is
// one.
//
// The second return is nil for a T that does not embed Model[T], and the caller
// then builds a model of its own. It is not an error: the plain shape is what
// the relation machinery and most of this package's tests use.
func newEntity[T any]() (*T, *Model[T]) {
	entity := new(T)
	return entity, modelIn(entity, embeddedIndex[T]())
}

// ModelOf returns the model embedded in entity.
//
// It is how a caller holding the application's own struct reaches the model's
// configuration -- the table, the key, the tenant column -- without the struct
// having to expose them again.
//
// It answers nil for a T that does not embed Model[T], and for a value the
// framework did not build: a User written as a literal has a zero Model[User]
// inside it, with no connection and no back pointer, and calling a terminal on
// it returns ErrUnwired rather than panicking. The ways to make one are the
// three ErrUnwired names.
func ModelOf[T any](entity *T) *Model[T] {
	model := modelIn(entity, embeddedIndex[T]())
	if model == nil || model.Entity == nil {
		return nil
	}
	return model
}

// ErrUnwired is a terminal called on an entity the framework did not build.
//
// A struct written as a literal has a zero Model[T] inside it: no connection, no
// back pointer to itself, no table. It is the one difference a Laravel developer
// meets at this layer, because in PHP $this is free and here it is not, so the
// error says what to do rather than what went wrong.
//
// The three it names are the three ways to get a wired row: an empty one from
// the model, a stored one from the query, and one read back. It named a fourth,
// Create(ctx, g, entity), and no such call exists -- Create takes the columns as
// a map, never a struct the caller already filled.
var ErrUnwired = errors.New("model: this value was not built by the framework, so it has no connection to save through -- make one with Model.NewInstance, insert one with Builder.Create, or read one back with Find, First or Get")

// entityValue is the entity as a reflect.Value, and whether there is one.
//
// The four places that walk the entity's fields go through it rather than each
// writing reflect.ValueOf(m.Entity).Elem(): a nil Entity makes that expression
// the zero Value, and the next call on it panics with "reflect: call of
// reflect.Value.Type on zero Value" -- which is what a literal used to get
// instead of ErrUnwired.
func (m *Model[T]) entityValue() (reflect.Value, bool) {
	if m == nil || m.Entity == nil {
		return reflect.Value{}, false
	}
	return reflect.ValueOf(m.Entity).Elem(), true
}

// wired reports whether this model can reach the database, and says why not.
//
// Both halves are checked because they fail apart: an entity built by Fill has a
// connection and no back pointer, and a literal has neither.
func (m *Model[T]) wired() error {
	if m == nil || m.Entity == nil || m.connection == nil {
		return ErrUnwired
	}
	return nil
}
