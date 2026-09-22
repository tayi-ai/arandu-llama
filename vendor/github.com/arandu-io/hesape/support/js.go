package support

import (
	"bytes"
	"encoding/json"
	"strings"
)

// Js is data turned into a JavaScript expression meant to be dropped inside an
// HTML attribute.
//
// The encoding escapes inside string literals only, leaving the structural
// characters of the JSON alone: a forward slash leaves as \/, so a literal
// cannot close a script element. Anything that is not a scalar comes back
// wrapped in a JSON.parse call, so the browser parses it instead of the
// JavaScript parser reading an object literal.
type Js struct {
	js string
}

// NewJs turns the data into a JavaScript expression, or returns the error
// encoding it raised.
func NewJs(data any) (*Js, error) {
	expression, err := convertDataToJavaScriptExpression(data)
	if err != nil {
		return nil, err
	}
	return &Js{js: expression}, nil
}

// From turns the data into a JavaScript expression, the same as [NewJs].
func From(data any) (*Js, error) { return NewJs(data) }

// Encode returns the data as JSON, escaped the way [Js] needs it. A [Jsonable]
// is asked for its own JSON first, and a value carrying a ToArray method that
// is not already a json.Marshaler is asked for its map first.
func Encode(data any) (string, error) {
	if jsonable, ok := data.(Jsonable); ok {
		encoded, err := jsonable.ToJson()
		if err != nil {
			return "", err
		}
		return jsHexEscape(encoded), nil
	}
	if arrayable, ok := data.(interface{ ToArray() map[string]any }); ok {
		if _, marshaler := data.(json.Marshaler); !marshaler {
			data = arrayable.ToArray()
		}
	}

	var buffer bytes.Buffer
	encoder := json.NewEncoder(&buffer)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(data); err != nil {
		return "", err
	}
	return jsHexEscape(strings.TrimRight(buffer.String(), "\n")), nil
}

// convertDataToJavaScriptExpression encodes the data and wraps it as an
// expression: a [Js] is already one, a string becomes a single-quoted literal,
// and everything else goes through convertJsonToJavaScriptExpression.
func convertDataToJavaScriptExpression(data any) (string, error) {
	if js, ok := data.(*Js); ok {
		return js.ToHtml(), nil
	}

	encoded, err := Encode(data)
	if err != nil {
		return "", err
	}

	if _, ok := data.(string); ok {
		return "'" + encoded[1:len(encoded)-1] + "'", nil
	}
	return convertJsonToJavaScriptExpression(encoded)
}

// convertJsonToJavaScriptExpression wraps anything that is not a scalar in a
// JSON.parse call, so the browser parses it instead of the JavaScript parser
// reading an object literal. An empty array or object, and a scalar, are left
// as they are.
func convertJsonToJavaScriptExpression(encoded string) (string, error) {
	if encoded == "[]" || encoded == "{}" {
		return encoded, nil
	}
	if strings.HasPrefix(encoded, `"`) || strings.HasPrefix(encoded, "{") || strings.HasPrefix(encoded, "[") {
		wrapped, err := Encode(encoded)
		if err != nil {
			return "", err
		}
		return "JSON.parse('" + wrapped[1:len(wrapped)-1] + "')", nil
	}
	return encoded, nil
}

// jsHexEscape walks the JSON and rewrites the contents of every string
// literal, leaving the structural characters alone.
//
// Five bytes leave as their \uXXXX form, and each one closes something if it
// travels intact. The expression is wrapped in JSON.parse('...') and dropped
// into an HTML attribute, so an apostrophe would end the JavaScript string, a
// less-than could open a tag, and an ampersand could start an entity the HTML
// parser resolves before JavaScript ever sees it. A forward slash leaves as \/
// so that a literal cannot spell a closing script tag.
//
// The encoder that feeds this has SetEscapeHTML(false), which means nothing
// upstream escapes any of them: this function is the only thing between the
// data and the page.
//
// An escape pair is carried through as it stands, both bytes together.
func jsHexEscape(encoded string) string {
	var b strings.Builder
	b.Grow(len(encoded))

	inString := false
	for i := 0; i < len(encoded); i++ {
		c := encoded[i]
		if !inString {
			b.WriteByte(c)
			if c == '"' {
				inString = true
			}
			continue
		}
		switch c {
		case '\\':
			// An escape pair is two bytes and travels together. Splitting it
			// would leave a lone backslash escaping whatever followed.
			b.WriteByte(c)
			if i+1 < len(encoded) {
				i++
				b.WriteByte(encoded[i])
			}
		case '"':
			b.WriteByte(c)
			inString = false
		case '<':
			b.WriteString(`\u003C`)
		case '>':
			b.WriteString(`\u003E`)
		case '&':
			b.WriteString(`\u0026`)
		case '\'':
			b.WriteString(`\u0027`)
		case '/':
			b.WriteString(`\/`)
		default:
			b.WriteByte(c)
		}
	}
	return b.String()
}

// ToHtml returns the expression. A nil receiver returns the empty string.
func (j *Js) ToHtml() string {
	if j == nil {
		return ""
	}
	return j.js
}

// String returns the expression, so Js satisfies fmt.Stringer.
func (j *Js) String() string { return j.ToHtml() }
