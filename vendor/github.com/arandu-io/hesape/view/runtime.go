package view

import (
	"errors"
	"fmt"
	"html/template"
	"io"
	"math"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// The functions here are what generated views call. This runtime is
// deliberately small: everything that can be decided at build time was, and
// what is left is the handful of things that cannot.

// Text renders a value as a string, for interpolation.
//
// It handles the types a view actually interpolates, and formats anything else
// with %v. It is not reflection over a struct: the field access already happened
// in generated Go, and this only turns the result into characters.
func Text(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case int:
		return strconv.Itoa(x)
	case int64:
		return strconv.FormatInt(x, 10)
	case bool:
		if x {
			return "true"
		}
		return "false"
	case float64:
		return strconv.FormatFloat(x, 'f', -1, 64)
	case nil:
		return ""
	case fmt.Stringer:
		return x.String()

	// A method value, which means the view wrote {{ .Greeting }} where it meant
	// {{ .Greeting() }}. Both compile: without the parentheses the expression is
	// the method itself, and %v prints a function as its address.
	//
	// It shipped. The showcase's plain-text verification message read
	// "Hello 0x10273d6d0, and welcome." for every person who registered, and it
	// was invisible because the HTML part beside it was written correctly -- only
	// a reader whose client prefers text ever saw it, which is the audience the
	// text part exists for.
	//
	// Answered by panicking rather than by printing something tidier: this is a
	// mistake with no correct output, and a view that renders "" or the method
	// name would hide it exactly as %v did. In development it reaches the error
	// page, which names the file and the line of the .kyse.go.
	case func() string:
		panic("view: a method was interpolated instead of its result. Write {{ .Name() }} rather than {{ .Name }}")

	default:
		if s := funcName(v); s != "" {
			panic("view: " + s + " was interpolated instead of its result. Write {{ .Name() }} rather than {{ .Name }}")
		}
		return fmt.Sprint(v)
	}
}

// funcName reports that a value is a function, and nothing otherwise.
//
// Separate from the type switch above because a method can have any signature --
// func() int, func() time.Time, func() (string, error) -- and every one of them
// is the same mistake. reflect is reached for here and nowhere else in this
// runtime, on a path that only runs when something is already wrong.
func funcName(v any) string {
	if v == nil {
		return ""
	}
	if reflect.TypeOf(v).Kind() != reflect.Func {
		return ""
	}
	return "a method"
}

// TextAttr renders a value for an attribute value between quotes.
//
// The escape is the standard library's: the six characters that can end an
// attribute or open a tag, and nothing else. It is what the position needs and
// all it needs, because a character reference inside an attribute value is
// decoded by the parser before anything downstream reads the attribute, so
// &#34; arrives at the browser as the quote it started as and never as the end
// of the value.
//
// The name is the reason this exists next to an escape that already spells
// itself out. Which position a value was written into is decided where the
// document is parsed and lost by the time the page is served, and a call that
// says the position keeps the decision in the output, where it can be read back
// and searched for.
//
// Two positions this is not for, and both look like an attribute. An attribute
// value without quotes ends at the first space, and a space is not one of the
// six. An attribute whose value is code -- onclick, or any attribute a library
// compiles and runs -- is a JavaScript position wearing an attribute's shape:
// the parser decodes the entity back into a quote before the script engine sees
// it, so &#39; closes the string it was meant to sit inside.
func TextAttr(v any) string {
	return template.HTMLEscapeString(Text(v))
}

// TextURL renders a value for an attribute that holds a URL, and refuses one
// whose scheme a browser would act on.
//
// Escaping answers a value that tries to end the attribute. It does not answer
// javascript:alert(1), which ends nothing and is a perfectly well-formed URL:
// the danger is in what the scheme means rather than in how it is written, and
// there is no spelling of it that is both harmless and still the value that was
// given. What is left is to decide which schemes may appear at all.
//
// Four may: http, https, mailto and tel. It is an allowlist and not a list of
// the dangerous ones because the dangerous list is never finished -- the first
// one ever written did not have vbscript on it -- while a closed list grows only
// when somebody decides to grow it.
//
// A value with no scheme is a relative URL and is allowed, with one shape
// refused: a slash or a backslash in each of the first two positions. That is a
// URL to another host wearing the shape of a path, and the backslash form
// exists because browsers normalise it -- "/\evil.example/x" and
// "//evil.example/x" resolve to the same other origin, and both read as local
// to a check that looks at the first character alone.
//
// A control character anywhere in the value is refused too. Browsers strip tabs
// and newlines out of a URL before resolving it, which is how "/<tab>/evil" and
// "java<tab>script:" become a URL this would otherwise have already passed.
//
// It does not percent-encode. A URL that arrives here already encoded would
// have its percent signs encoded a second time, and the value would change
// meaning on the way to being made safe.
//
// The refusal is an error, and what comes back with it is the empty string
// rather than a repaired value. A URL quietly rewritten to something harmless
// is a page saying something other than what it was asked to say, with nothing
// anywhere recording that it happened; a caller that drops the error writes
// nothing instead of writing the attack.
func TextURL(v any) (string, error) {
	raw := Text(v)

	for i := 0; i < len(raw); i++ {
		if raw[i] < 0x20 || raw[i] == 0x7f {
			return "", errors.New("view: a URL may not carry a control character. " +
				"Browsers strip tabs and newlines out of a URL before resolving it, " +
				"so a value carrying one does not resolve to the address it reads as")
		}
	}

	if i := strings.IndexByte(raw, ':'); i >= 0 && !strings.ContainsAny(raw[:i], "/?#") {
		if !schemeIsAllowed(raw[:i]) {
			return "", errors.New("view: this URL's scheme is refused. " +
				"A URL in an attribute may be relative, or use http, https, mailto or tel")
		}
		return template.HTMLEscapeString(raw), nil
	}

	if len(raw) >= 2 && isSlash(raw[0]) && isSlash(raw[1]) {
		return "", errors.New("view: a URL that starts with two slashes is refused. " +
			"It reads as a path and resolves to another host, and a backslash in " +
			"either position is the same address written to get past the reading")
	}

	return template.HTMLEscapeString(raw), nil
}

// schemeIsAllowed reports whether a URL scheme may appear in an attribute.
//
// Folded rather than lowered first, because the comparison is the whole check:
// a scheme that differs from "http" by a stripped tab or a leading space is a
// different string here and is refused for it, which is the outcome wanted.
func schemeIsAllowed(scheme string) bool {
	for _, allowed := range [...]string{"http", "https", "mailto", "tel"} {
		if strings.EqualFold(scheme, allowed) {
			return true
		}
	}
	return false
}

// isSlash reports whether a byte separates the authority of a URL, which both
// slashes do once a browser has normalised the address.
func isSlash(c byte) bool { return c == '/' || c == '\\' }

// TextJS renders a value as a JavaScript literal, inside a script element.
//
// The value comes out as data and never as code: a number as a number, a
// boolean as a boolean, nothing as null, and everything else as a quoted
// string. "1;alert(document.cookie)" is a string of twenty-four characters
// after this and not two statements, which is the whole of what this position
// needs and what escaping six HTML characters does not do -- none of the six
// appears in that payload.
//
// The quotes are written here rather than in the view. That is what lets the
// value be a literal at all, and it is also the position's one condition: this
// replaces a whole JavaScript expression, not a piece of a string the view
// opened for itself. A view that writes its own quotes around it produces a
// literal inside a literal, which is a syntax error and not a silent hole.
//
// The characters that get a hex form are the ones that end the script element
// rather than the ones that end the string: a script's content is not parsed
// for entities, so escaping it as HTML would leave &lt; sitting in the page as
// four visible characters while "</script>" inside the value closed the element
// around it. U+2028 and U+2029 are here because JavaScript ends a line on them
// as it does on a newline.
func TextJS(v any) string {
	switch x := v.(type) {
	case nil:
		return "null"
	case bool:
		if x {
			return "true"
		}
		return "false"
	case int:
		return strconv.Itoa(x)
	case int64:
		return strconv.FormatInt(x, 10)
	case float64:
		// A float that is not a number has no literal. JavaScript spells both of
		// these, but NaN and Infinity are identifiers there rather than literals,
		// and a page that shadowed either would read the shadow.
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return "null"
		}
		return strconv.FormatFloat(x, 'f', -1, 64)
	}
	return jsString(Text(v))
}

// jsString writes a Go string as a quoted JavaScript string literal.
func jsString(s string) string {
	var b strings.Builder
	b.Grow(len(s) + 2)
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '\\':
			b.WriteString(`\\`)
		case '"':
			b.WriteString(`\"`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		case '<', '>', '&', '\u2028', '\u2029':
			jsHex(&b, r)
		default:
			if r < 0x20 || r == 0x7f {
				jsHex(&b, r)
				continue
			}
			b.WriteRune(r)
		}
	}
	b.WriteByte('"')
	return b.String()
}

// jsHex writes one rune in the \uXXXX form JavaScript reads inside a string.
func jsHex(b *strings.Builder, r rune) {
	const digits = "0123456789abcdef"
	b.WriteString(`\u`)
	b.WriteByte(digits[r>>12&0xf])
	b.WriteByte(digits[r>>8&0xf])
	b.WriteByte(digits[r>>4&0xf])
	b.WriteByte(digits[r&0xf])
}

// TextCSS renders a value inside a style attribute or a style element, and
// refuses anything that is not plainly a CSS value.
//
// This position has no escape that keeps both the value and its meaning. A
// semicolon starts a declaration nobody asked for, a brace closes one rule and
// opens another, parentheses carry a url() and a url() carries a scheme, a
// slash starts a comment that hides any of it, and a backslash spells every one
// of them a second way. Escaping those would leave a value that no longer says
// what it said, so what a value may contain is a closed set instead:
//
//	letters, digits, space, and  # % . , - + _ !
//
// Enough for a colour, a length, a keyword, a font weight and an !important,
// and short of everything above. A value outside it is refused rather than
// trimmed, for the same reason a rewritten URL is refused: a stylesheet quietly
// missing the half that was dropped is a page that changed without saying so.
//
// The accepted set holds no character the HTML parser reads, which is why the
// value goes out exactly as it came in. That is a property of the set and the
// reason it is drawn where it is: widening it to a quote or an angle bracket
// would put an escape back in front of this.
func TextCSS(v any) (string, error) {
	raw := Text(v)
	for _, r := range raw {
		if !cssIsSafe(r) {
			return "", fmt.Errorf("view: a CSS value may not contain %q. "+
				"What may appear is a letter, a digit, a space, or one of # %% . , - + _ !", r)
		}
	}
	return raw, nil
}

// cssIsSafe reports whether a rune may appear in a CSS value.
func cssIsSafe(r rune) bool {
	switch {
	case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
		return true
	case r == ' ':
		return true
	}
	return strings.ContainsRune("#%.,-+_!", r)
}

// Attributes renders a set of attributes a caller handed a component, as the
// text that goes inside a tag: a leading space before each, sorted by name.
//
// # Why this exists rather than an escape
//
// Every other function here answers a value whose position the compiler read
// off the markup around it. This one answers a value whose *name* comes from
// data, and that is a different question: the position of a value is decided by
// the attribute it lands in, so an attribute nobody wrote at build time has no
// decided position. TextAttr would be applied to a URL and to a script alike.
//
// So the name is checked instead, against what it would mean:
//
//   - it has to look like an attribute name -- a letter, then letters, digits
//     and hyphens, optionally one colon and more of the same;
//   - "on" at the front is an event handler, and its value is JavaScript
//     wearing an attribute's shape;
//   - the whole HTMX family is refused, in both spellings: a verb is fetched,
//     hx-vals and hx-headers change what is sent, and hx-swap-oob decides what
//     of the page the answer replaces. A component that speaks HTMX takes the
//     ones it supports as fields, where the compiler reads the position;
//   - "x-" at the front is an Alpine directive, which this framework does not
//     serve and whose value is an expression;
//   - srcdoc is a document, and http-equiv makes the element a directive to the
//     browser -- neither is answered by escaping;
//   - "style" is refused because the policy is style-src 'self' with no
//     unsafe-inline, so a style attribute is dropped by the browser and the
//     page silently does not do what the markup says;
//   - an attribute a browser resolves as an address is refused, because TextURL
//     is what decides a scheme and this would be the one door around it;
//   - "class", "role" and the aria- family are refused because a component owns
//     them: class has a field of its own, and the other two are the promise the
//     component makes to a screen reader.
//
// What is left is the set an attribute can carry inertly -- data-, id, title,
// the form and layout attributes -- and every one of those values is escaped as
// HTML.
//
// The list of addresses overlaps the compiler's and is deliberately wider. In
// aru/internal/kyse/generate.go the position decides, so an HTMX verb is a URL
// there and goes through TextURL; here the name decides and there is no
// component behind it, so the whole HTMX family is refused rather than checked.
// The two lists disagreeing on purpose is fine; disagreeing by accident is how
// the same attribute ends up under two rules, which is what the first version of
// this function did with hx-post.
//
// The refusal is an error and what comes back beside it is the empty string,
// for the reason TextURL gives: an attribute quietly renamed or dropped is a
// page that changed without saying so. A caller that drops the error writes
// nothing instead of writing the attack.
//
// Names are sorted so the output of a given map is one string and not several,
// which is what a test can hold.
func Attributes(attrs map[string]string) (string, error) {
	if len(attrs) == 0 {
		return "", nil
	}

	names := make([]string, 0, len(attrs))
	for name := range attrs {
		names = append(names, name)
	}
	sort.Strings(names)

	var out strings.Builder
	for _, name := range names {
		if err := attributeIsInert(name); err != nil {
			return "", err
		}
		out.WriteString(" " + name + `="` + template.HTMLEscapeString(attrs[name]) + `"`)
	}
	return out.String(), nil
}

// attributeIsInert reports why an attribute name may not be written by a
// caller, or nil when it may.
func attributeIsInert(name string) error {
	if !attributeNamePattern.MatchString(name) {
		return fmt.Errorf("view: %q is not written as an attribute name. "+
			"A name is a letter, then letters, digits and hyphens, and at most one colon "+
			"-- a space or a quote in one ends it, and no escape puts it back", name)
	}

	head, _, _ := strings.Cut(name, ":")
	switch {
	case strings.HasPrefix(head, "on"):
		return fmt.Errorf("view: %q holds a script rather than text. "+
			"An attribute a browser compiles is a JavaScript position wearing an "+
			"attribute's shape, and nothing escapes a value into being data there", name)
	case isHTMX(name):
		// The whole family, not the handlers and the verbs. hx-post is fetched,
		// hx-vals and hx-headers change what is sent, hx-swap-oob decides what
		// of the page the answer replaces -- and a component that speaks HTMX
		// takes the ones it supports as fields of its own, where the compiler
		// reads the position and puts a URL through TextURL.
		//
		// Written as a family for the reason the compiler writes it as one: it
		// is open, and a list would be one name short of the page that used a
		// new one.
		return fmt.Errorf("view: %q is an HTMX attribute, and a component takes the ones it "+
			"supports as fields of its own. "+
			"A request written here would go out with the page's own credentials, "+
			"to an address nothing checked, and swap the answer into the document", name)
	case strings.HasPrefix(name, "x-"), strings.HasPrefix(name, "data-x-"):
		return fmt.Errorf("view: %q is an Alpine directive, and this framework serves none. "+
			"Client behaviour is a name in a catalogue, dispatched by ui.js, "+
			"and no attribute here is evaluated", name)
	case name == "style":
		return fmt.Errorf("view: %q is refused. "+
			"The policy is style-src 'self' with no unsafe-inline, so the browser drops it "+
			"and the page does not do what the markup says. Write a class", name)
	case attributeHoldsURL(name):
		return fmt.Errorf("view: %q holds an address, and what makes an address safe is its scheme. "+
			"A component that takes a URL takes it as a field of its own, "+
			"which is what puts it through the check", name)
	case name == "srcdoc":
		return fmt.Errorf("view: %q holds a document rather than text. "+
			"The parser undoes character references before it parses what is inside, "+
			"so escaping the value hands the markup back", name)
	case name == "http-equiv":
		return fmt.Errorf("view: %q makes the element a directive to the browser. "+
			"With a content beside it, it is a redirect or a policy, and neither is "+
			"something a caller decides about somebody else's component", name)
	case name == "class", name == "role", strings.HasPrefix(name, "aria-"):
		return fmt.Errorf("view: %q belongs to the component. "+
			"A class is added through the class field, and role and aria- are the promise "+
			"the component makes to a screen reader", name)
	}
	return nil
}

// isHTMX reports whether HTMX reads this attribute.
//
// Both spellings, because HTMX resolves every one of its attributes through a
// data- alias -- `ee(e,t)||ee(e,"data-"+t)` in the bundle -- so a check that
// only knew the short form is six characters from being bypassed. Alpine reads
// no such alias, which is why the case above is narrower.
func isHTMX(name string) bool {
	return strings.HasPrefix(name, "hx-") || strings.HasPrefix(name, "data-hx-")
}

// attributeNamePattern is the shape of a name this will write.
//
// Narrower than the HTML syntax allows, deliberately: what it accepts is what a
// component is written with, and a name outside it is far more likely to be a
// mistake than a requirement.
var attributeNamePattern = regexp.MustCompile(`^[a-z][a-z0-9-]*(:[a-z0-9-]+)?$`)

// attributeHoldsURL reports whether a browser resolves this attribute's value
// as an address.
//
// The list is of the attributes rather than of the schemes, and it is closed:
// widening it is a decision somebody makes, while a value reaching one of these
// unchecked is a decision nobody made.
func attributeHoldsURL(name string) bool {
	switch name {
	case "href", "src", "srcset", "action", "formaction", "cite", "data",
		"poster", "background", "codebase", "longdesc", "manifest", "profile",
		"ping", "usemap", "xlink:href":
		return true
	}
	return false
}

// Yield renders the section a child view declared, or nothing.
//
// A layout yields sections that a given child may not have, and the answer is
// the empty string. A missing section is a
// page without a sidebar, not an error.
func Yield(w io.Writer, sections map[string]func(io.Writer) error, name string) error {
	section, declared := sections[name]
	if !declared {
		return nil
	}
	return section(w)
}

// RenderInto renders a layout, handing it the sections of the child view.
//
// This is what `@extends` compiles to: the child does not write markup, it
// renders the layout and passes what goes in the holes: the child is evaluated
// to fill the sections before the layout runs.
func RenderInto(w io.Writer, layout string, data any, sections map[string]func(io.Writer) error) error {
	mu.RLock()
	f, known := layouts[layout]
	mu.RUnlock()

	if !known {
		return fmt.Errorf("view: %q extends a layout that does not exist. "+
			"Layouts are views too -- resources/views/%s.kyse.go", layout, layout)
	}
	return f(w, data, sections)
}

// LayoutFunc is a view that receives sections, which is what a layout is.
type LayoutFunc func(w io.Writer, data any, sections map[string]func(io.Writer) error) error

var layouts = map[string]LayoutFunc{}

// RegisterLayout records a compiled layout. Generated code calls it from init()
// when the view contains a @yield.
func RegisterLayout(name string, f LayoutFunc) {
	mu.Lock()
	defer mu.Unlock()
	if _, taken := layouts[name]; taken {
		panic("view: layout " + name + " is already registered -- a stale generated file is probably still on disk")
	}
	layouts[name] = f
}

// Include renders a partial with the same data as the page.
//
// A partial shares the page's data. That data is one typed struct, so the
// partial receives exactly it -- and a partial that wants
// something else is a partial that takes different data, which is what a
// component is for.
func Include(w io.Writer, name string, data any) error {
	mu.RLock()
	f, known := views[name]
	mu.RUnlock()

	if !known {
		return unknownView(name)
	}
	return f(w, data)
}

// CSRF writes the hidden input a form needs.
//
// The token comes from the data, through an interface the page data satisfies.
// It is not read from a global: a template that reaches for request state
// outside the data it was given is how a form ends up with another session's
// token under load.
func CSRF(w io.Writer, data any) error {
	holder, ok := data.(interface{ CSRFToken() string })
	if !ok {
		return fmt.Errorf("view: @csrf needs the page data to provide the token. " +
			"Add a CSRFToken() string method to the struct the view declares")
	}
	_, err := fmt.Fprintf(w, `<input type="hidden" name="_token" value="%s">`, holder.CSRFToken())
	return err
}
