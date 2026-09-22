package grammars

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"regexp"
	"strconv"
	"strings"

	"github.com/arandu-io/hesape/database/query"
)

// The JSON path compilation the query grammars and the schema grammars share,
// plus the small value helpers that go with it.
//
// They are methods on Grammar, which every driver grammar embeds.

// wrapJSONFieldAndPath splits "options->language->code" into the column
// and the path into it, and wraps each separately.
func (g *Grammar) wrapJSONFieldAndPath(column any) (field, path string) {
	parts := strings.SplitN(text(column), "->", 2)

	field = g.self.Wrap(parts[0])

	if len(parts) > 1 {
		path = ", " + g.wrapJSONPath(parts[1], "->")
	}

	return field, path
}

// jsonPathQuote matches a quote inside a path, with or without the
// backslashes somebody escaped it with.
var jsonPathQuote = regexp.MustCompile(`(\\+)?'`)

// wrapJSONPath quotes a JSON path into the "$.a.b.c" form the JSON
// functions expect.
func (g *Grammar) wrapJSONPath(value, delimiter string) string {
	value = jsonPathQuote.ReplaceAllString(value, "''")

	segments := strings.Split(value, delimiter)
	wrapped := make([]string, 0, len(segments))
	for _, segment := range segments {
		wrapped = append(wrapped, wrapJSONPathSegment(segment))
	}

	jsonPath := strings.Join(wrapped, ".")

	prefix := "."
	if strings.HasPrefix(jsonPath, "[") {
		prefix = ""
	}

	return "'$" + prefix + jsonPath + "'"
}

// jsonPathArrayKeys matches the trailing [0][1] of a path segment that
// indexes into an array.
var jsonPathArrayKeys = regexp.MustCompile(`(\[[^\]]+\])+$`)

// wrapJSONPathSegment quotes one segment of a JSON path, keeping any
// trailing array index outside the quotes.
func wrapJSONPathSegment(segment string) string {
	if parts := jsonPathArrayKeys.FindString(segment); parts != "" {
		key := strings.TrimSuffix(segment, parts)
		if key != "" {
			return `"` + key + `"` + parts
		}
		return parts
	}
	return `"` + segment + `"`
}

// isJSONSelector reports whether value names a JSON path rather than a
// plain column. Go initialisms are upper case throughout, hence JSON
// rather than Json.
func isJSONSelector(value any) bool {
	return strings.Contains(text(value), "->")
}

// WrapJSONSelector quotes a JSON path selector. The base implementation has
// no JSON support to offer: every grammar in this package overrides it.
//
// Wrap returns a string, so an engine with no override cannot refuse with
// an error -- the arrow stays inside the quoted identifier instead, and the
// statement fails with "no such column", which is true.
func (g *Grammar) WrapJSONSelector(value string) string {
	return g.self.WrapValue(value)
}

// WrapJSONBooleanSelector quotes a JSON path selector for a boolean
// comparison.
func (g *Grammar) WrapJSONBooleanSelector(value string) string {
	return g.self.WrapJSONSelector(value)
}

// WrapJSONBooleanValue wraps a compiled value for a JSON boolean
// comparison. The base implementation returns it unchanged.
func (g *Grammar) WrapJSONBooleanValue(value string) string { return value }

// text renders a value as the string the grammar concatenates into a
// statement.
//
// It is not a substitute for a placeholder: nothing that reaches it is a bound
// value. Columns, table names, operators and expressions pass through here;
// values go through Parameter.
func text(v any) string {
	switch t := v.(type) {
	case nil:
		return ""
	case string:
		return t
	case query.Expression:
		return text(t.Value())
	case *query.Expression:
		if t == nil {
			return ""
		}
		return text(t.Value())
	case fmt.Stringer:
		return t.String()
	case bool:
		if t {
			return "1"
		}
		return "0"
	case int:
		return strconv.Itoa(t)
	case int64:
		return strconv.FormatInt(t, 10)
	case float64:
		return strconv.FormatFloat(t, 'f', -1, 64)
	case []byte:
		return string(t)
	default:
		return fmt.Sprint(v)
	}
}

// isBool reports whether value is a bool.
func isBool(value any) bool {
	_, ok := value.(bool)
	return ok
}

// truthy reports whether an option a caller may not have set should be
// treated as true.
func truthy(value any) bool {
	switch v := value.(type) {
	case nil:
		return false
	case bool:
		return v
	case string:
		return v != "" && v != "0"
	case int:
		return v != 0
	default:
		return true
	}
}

// isStructured reports whether value is a slice or a map, which is what
// has to become JSON text before it can be bound.
//
// A byte slice is not one of them. It is a binary value on its way to a
// blob, and encoding it as JSON would store the base64 of the bytes
// instead.
func isStructured(value any) bool {
	if value == nil {
		return false
	}
	if _, ok := value.([]byte); ok {
		return false
	}

	switch reflect.TypeOf(value).Kind() {
	case reflect.Slice, reflect.Array, reflect.Map:
		return true
	default:
		return false
	}
}

// encodeJSON encodes value as JSON text, without HTML-escaping <, > and &,
// which Go's encoder does by default.
//
// A JSON path holding a "<" would otherwise stop matching the value stored
// in the column.
func encodeJSON(value any) (any, error) {
	var buffer bytes.Buffer

	encoder := json.NewEncoder(&buffer)
	encoder.SetEscapeHTML(false)

	if err := encoder.Encode(value); err != nil {
		return nil, fmt.Errorf("query/grammars: encoding a JSON binding: %w", err)
	}

	return strings.TrimRight(buffer.String(), "\n"), nil
}
