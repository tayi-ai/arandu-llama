// Package enum is the contract a closed set of values satisfies, and the
// operations the layers around it derive from that contract instead of
// repeating the set.
//
// Go has no enum keyword, so a closed set is a named type over a string or an
// integer plus the methods that refuse everything outside it. The generator
// writes those methods; this package names them, so validation, serialization,
// persistence and a form can recognise the type at runtime and read the cases
// off it.
//
// The defect it exists to remove is the second copy of the set. A list of cases
// written a second time -- in a validation rule, in a <select>, in a switch --
// is a list that can disagree with the type, and nothing compares them. Every
// function here takes the type, or a value of it, and answers from that.
//
// # Two spellings, and which one a function reads
//
// A value has a shown spelling, which is String, and a stored spelling, which
// is what Value hands the column. For a text-backed set they are the same
// string. For an integer-backed one they are not: the column holds 2 and the
// form shows "normal". Every function here says which of the two it reads, and
// the ones that cannot answer for an integer-backed set say so rather than
// guessing.
package enum

import (
	"database/sql/driver"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strconv"
)

// Enum is what a generated enum type satisfies.
//
// The four methods are the ones the generated type already carries, and each
// answers a different reader: Valid is membership, String is what a reader
// sees, Label is what a form shows, and Value is what a column receives.
//
// Scan is deliberately absent. It is declared on the pointer, because it
// assigns, and an interface naming it could only be satisfied by *T -- while
// what a struct field, a map entry and a decoded request value hold is T.
// Reading a stored spelling back into the type is Cast, which needs no pointer.
type Enum interface {
	// Valid reports whether the value is one of the cases. Anything that came
	// from outside the process -- a column, a request, a message -- is a value
	// only the type can be asked about.
	Valid() bool

	// String is the shown spelling: what a log, a form control and fmt print.
	// It is also the spelling the generated parser reads back, which is why it
	// is what an option's value carries.
	String() string

	// Label is what a form shows to a person. It is separate from String
	// because the shown spelling is a contract with every reader of the type
	// and the label is a contract with nobody: changing a label must not be
	// able to change a row.
	Label() string

	// Valuer writes the value to a column, refusing one outside the set.
	driver.Valuer
}

// Option is one case as a form control shows it: the spelling that goes back in
// the request, and the label the reader sees.
type Option struct {
	// Value is what a submitted form sends back, and what the generated parser
	// accepts.
	Value string
	// Label is the text shown to the reader.
	Label string
}

// Options renders the cases for a form control, in the order they were given.
//
//	enum.Options(enums.InvoiceStatusValues()...)
//
// It takes the values rather than the type because the generated list is a
// package function -- the one place that knows the order the cases were
// declared in. A view that spells the options out again is the second copy this
// package exists to remove.
func Options[E Enum](values ...E) []Option {
	out := make([]Option, 0, len(values))
	for _, v := range values {
		out = append(out, Option{Value: v.String(), Label: v.Label()})
	}
	return out
}

// Names lists the shown spellings, in the order they were given.
//
// It is what a rule, a message or a schema needs when it has to name the cases
// as text rather than show them.
func Names[E Enum](values ...E) []string {
	out := make([]string, 0, len(values))
	for _, v := range values {
		out = append(out, v.String())
	}
	return out
}

// From reads value as an enum, following pointers, and reports whether it was
// one.
//
// It is the entry point for the layers that receive any: a validator holding a
// field of a decoded request cannot name the type, but it can ask whether the
// value carries its own set.
func From(value any) (Enum, bool) {
	if value == nil {
		return nil, false
	}
	// The pointer is followed before the type is asserted, and not after: *T
	// satisfies an interface of value methods, so a typed nil pointer would
	// assert clean here and panic on the first call to Valid.
	rv := reflect.ValueOf(value)
	for rv.Kind() == reflect.Pointer {
		if rv.IsNil() {
			return nil, false
		}
		rv = rv.Elem()
	}
	e, ok := rv.Interface().(Enum)
	return e, ok
}

// Cast spells text in the enum type of proto and returns the result, whose
// Valid answers membership.
//
// text is the stored spelling: the string a text-backed set holds, and the
// decimal number an integer-backed one holds. It is not the shown spelling of
// an integer-backed set, which no amount of reflection can turn back into a
// number -- the mapping lives in a switch inside the type.
//
// It reports false only when text cannot be a value of that type at all, and
// not when the value is merely outside the set: those are different answers,
// and collapsing them would turn "unknown case" into "unreadable input".
func Cast(proto Enum, text string) (Enum, bool) {
	if proto == nil {
		return nil, false
	}
	t := reflect.TypeOf(proto)
	if t.Kind() == reflect.Pointer {
		t = t.Elem()
	}

	v := reflect.New(t).Elem()
	switch t.Kind() {
	case reflect.String:
		v.SetString(text)
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		n, err := strconv.ParseInt(text, 10, 64)
		if err != nil || v.OverflowInt(n) {
			return nil, false
		}
		v.SetInt(n)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		n, err := strconv.ParseUint(text, 10, 64)
		if err != nil || v.OverflowUint(n) {
			return nil, false
		}
		v.SetUint(n)
	default:
		return nil, false
	}

	e, ok := v.Interface().(Enum)
	return e, ok
}

// Holds reports whether text names a case of proto's type, reading text as the
// stored spelling that Cast documents.
//
// It is the membership question asked from a layer that has a value and a
// string and no way to name the type: the type decides, and no list is
// consulted.
func Holds(proto Enum, text string) bool {
	e, ok := Cast(proto, text)
	return ok && e.Valid()
}

// Unknown returns the names in list that proto's type does not declare, and
// reports whether the comparison was possible at all.
//
// It is how a layer that was handed a list of cases -- a validation rule
// written by hand, a filter, a schema -- finds out that the list disagrees with
// the type, which is the failure nothing else catches: a case added to the type
// and not to the list, or the other way round, passes every test and is wrong
// in production.
//
// comparable is false for an integer-backed set, and that is the honest answer
// rather than a wrong one. Such a list is written in shown spellings, the type
// stores numbers, and turning one into the other needs the type's own switch.
// A caller that gets false has not been told the list is right; it has been told
// nobody can say from here.
func Unknown(proto Enum, list []string) (unknown []string, comparable bool) {
	if proto == nil {
		return nil, false
	}
	t := reflect.TypeOf(proto)
	if t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t.Kind() != reflect.String {
		return nil, false
	}

	for _, name := range list {
		if !Holds(proto, name) {
			unknown = append(unknown, name)
		}
	}
	return unknown, true
}

// ErrNotACase is returned when a value outside the set reaches encoding or
// decoding. It is a sentinel so a handler can tell a rejected case from a
// malformed document.
var ErrNotACase = errors.New("enum: value is not one of the cases")

// Marshal encodes an enum the way its underlying type encodes -- quoted for a
// text-backed set, bare for an integer-backed one -- and refuses a value
// outside the set.
//
// The encoding is unchanged because the stored spelling is the contract every
// reader already has. What is added is the refusal, which is the same one Value
// makes on the way to a column: a zero-valued struct must not be able to put an
// empty string in a message any more than in a row.
func Marshal(v Enum) ([]byte, error) {
	if v == nil {
		return nil, fmt.Errorf("%w: no value", ErrNotACase)
	}
	if !v.Valid() {
		return nil, fmt.Errorf("%w: refusing to encode %q", ErrNotACase, v.String())
	}

	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.String:
		return json.Marshal(rv.String())
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return json.Marshal(rv.Int())
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return json.Marshal(rv.Uint())
	default:
		return nil, fmt.Errorf("enum: %T is backed by %s, which is not a stored spelling", v, rv.Kind())
	}
}

// Unmarshal decodes data into the enum type of proto.
//
// A case the type does not know about is an error here, at the document that
// carries it, rather than a zero value that behaves like the first case three
// layers up.
func Unmarshal(proto Enum, data []byte) (Enum, error) {
	if proto == nil {
		return nil, fmt.Errorf("%w: no type to decode into", ErrNotACase)
	}

	var raw any
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}

	var text string
	switch value := raw.(type) {
	case string:
		text = value
	case float64:
		// A JSON number arrives as a float64 whatever it was written as, and an
		// integer-backed set is the only one that answers to a number at all.
		if value != float64(int64(value)) {
			return nil, fmt.Errorf("%w: %v is not a whole number", ErrNotACase, value)
		}
		text = strconv.FormatInt(int64(value), 10)
	default:
		return nil, fmt.Errorf("%w: cannot read %T", ErrNotACase, raw)
	}

	e, ok := Cast(proto, text)
	if !ok {
		return nil, fmt.Errorf("%w: %q cannot be a value of %T", ErrNotACase, text, proto)
	}
	if !e.Valid() {
		return nil, fmt.Errorf("%w: %q", ErrNotACase, text)
	}
	return e, nil
}
