package view

import (
	"fmt"
	"html/template"
	"iter"
	"sort"
	"strings"

	"github.com/arandu-io/hesape/collections/arr"
	"github.com/arandu-io/hesape/str"
)

// ComponentAttributeBag is the bag a component receives every attribute it
// was not declared to take: the HTML that the caller wrote on the tag and the
// component did not name. Merge is what makes a component's own class list
// and the caller's coexist instead of one overwriting the other.
//
// Keys are iterated sorted, since a Go map remembers no order of its own, so
// every method that returns a string -- String, ToHTML -- is deterministic,
// which is what a test can assert against. It is the same choice hesape/html
// made for its attribute writer.
type ComponentAttributeBag struct {
	attributes map[string]any
}

// NewComponentAttributeBag returns a new bag populated with attributes.
func NewComponentAttributeBag(attributes map[string]any) *ComponentAttributeBag {
	bag := &ComponentAttributeBag{attributes: map[string]any{}}
	bag.SetAttributes(attributes)
	return bag
}

// All returns a copy of every attribute in the bag.
func (b *ComponentAttributeBag) All() map[string]any {
	out := make(map[string]any, len(b.attributes))
	for k, v := range b.attributes {
		out[k] = v
	}
	return out
}

// keys returns the attribute names in the bag's iteration order.
func (b *ComponentAttributeBag) keys() []string {
	out := make([]string, 0, len(b.attributes))
	for k := range b.attributes {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// First returns the first attribute in the bag's iteration order, or the
// given fallback when the bag is empty.
//
// fallback is variadic so it can be omitted entirely, which is how an
// optional parameter is written in Go.
func (b *ComponentAttributeBag) First(fallback ...any) any {
	for _, k := range b.keys() {
		return b.attributes[k]
	}
	if len(fallback) > 0 {
		return fallback[0]
	}
	return nil
}

// Get returns the value stored under key, or the given fallback when key is
// absent.
func (b *ComponentAttributeBag) Get(key string, fallback ...any) any {
	if v, ok := b.attributes[key]; ok {
		return v
	}
	if len(fallback) > 0 {
		return fallback[0]
	}
	return nil
}

// Has reports whether the bag has every one of the given keys.
func (b *ComponentAttributeBag) Has(keys ...string) bool {
	for _, key := range keys {
		if _, ok := b.attributes[key]; !ok {
			return false
		}
	}
	return true
}

// HasAny reports whether the bag has at least one of the given keys.
func (b *ComponentAttributeBag) HasAny(keys ...string) bool {
	if len(b.attributes) == 0 {
		return false
	}
	for _, key := range keys {
		if b.Has(key) {
			return true
		}
	}
	return false
}

// Missing reports whether key is absent from the bag.
func (b *ComponentAttributeBag) Missing(key string) bool { return !b.Has(key) }

// Only returns a new bag containing just the given keys.
func (b *ComponentAttributeBag) Only(keys ...string) *ComponentAttributeBag {
	if keys == nil {
		return NewComponentAttributeBag(b.All())
	}
	wanted := map[string]bool{}
	for _, k := range keys {
		wanted[k] = true
	}
	values := map[string]any{}
	for k, v := range b.attributes {
		if wanted[k] {
			values[k] = v
		}
	}
	return NewComponentAttributeBag(values)
}

// Except returns a new bag with the given keys removed.
func (b *ComponentAttributeBag) Except(keys ...string) *ComponentAttributeBag {
	if keys == nil {
		return NewComponentAttributeBag(b.All())
	}
	unwanted := map[string]bool{}
	for _, k := range keys {
		unwanted[k] = true
	}
	values := map[string]any{}
	for k, v := range b.attributes {
		if !unwanted[k] {
			values[k] = v
		}
	}
	return NewComponentAttributeBag(values)
}

// Filter returns a new bag containing only the attributes for which callback
// returns true.
//
// callback takes the value and the key, in that order.
func (b *ComponentAttributeBag) Filter(callback func(value any, key string) bool) *ComponentAttributeBag {
	values := map[string]any{}
	for k, v := range b.attributes {
		if callback(v, k) {
			values[k] = v
		}
	}
	return NewComponentAttributeBag(values)
}

// WhereStartsWith returns a new bag containing only the attributes whose key
// starts with one of needles.
func (b *ComponentAttributeBag) WhereStartsWith(needles ...string) *ComponentAttributeBag {
	return b.Filter(func(_ any, key string) bool { return str.StartsWith(key, needles...) })
}

// WhereDoesntStartWith returns a new bag excluding the attributes whose key
// starts with one of needles.
func (b *ComponentAttributeBag) WhereDoesntStartWith(needles ...string) *ComponentAttributeBag {
	return b.Filter(func(_ any, key string) bool { return !str.StartsWith(key, needles...) })
}

// ThatStartWith is an alias for WhereStartsWith.
func (b *ComponentAttributeBag) ThatStartWith(needles ...string) *ComponentAttributeBag {
	return b.WhereStartsWith(needles...)
}

// OnlyProps returns a new bag containing the given prop keys, matched in
// both their written and kebab-case form.
func (b *ComponentAttributeBag) OnlyProps(keys ...string) *ComponentAttributeBag {
	return b.Only(ExtractPropNames(keys)...)
}

// ExceptProps returns a new bag with the given prop keys removed, matched in
// both their written and kebab-case form.
func (b *ComponentAttributeBag) ExceptProps(keys ...string) *ComponentAttributeBag {
	return b.Except(ExtractPropNames(keys)...)
}

// Class merges classList into the bag's "class" attribute, after converting
// it to a CSS class string.
//
// The list is wrapped and spread so that each element is read on its own: a
// slice handed over as a single argument would be rendered as one class.
func (b *ComponentAttributeBag) Class(classList any) *ComponentAttributeBag {
	return b.Merge(map[string]any{"class": arr.ToCssClasses(arr.Wrap(classList)...)})
}

// Style merges styleList into the bag's "style" attribute, after converting
// it to a CSS style string. The list is spread for the reason given on
// [ComponentAttributeBag.Class].
func (b *ComponentAttributeBag) Style(styleList any) *ComponentAttributeBag {
	return b.Merge(map[string]any{"style": arr.ToCssStyles(arr.Wrap(styleList)...)})
}

// Merge combines attributeDefaults into the bag.
//
// The component's defaults lose to what the caller wrote, except for class
// and style and for anything the component marked with Prepends: those two
// are appended rather than replaced, which is why a button can carry its own
// padding and still take a colour from the call site.
//
// # Why nothing is escaped here
//
// A bag holds text, and String is the one place that text becomes markup, so
// String is the one place that escapes. Escaping a default on the way in and
// the caller's own value never left the two halves of one attribute under
// different rules: a default carrying an ampersand came out doubly escaped
// while the value beside it came out able to end the attribute it sat in.
func (b *ComponentAttributeBag) Merge(attributeDefaults map[string]any) *ComponentAttributeBag {
	defaults := make(map[string]any, len(attributeDefaults))
	for k, v := range attributeDefaults {
		defaults[k] = v
	}

	appendable := map[string]any{}
	nonAppendable := map[string]any{}
	for k, v := range b.attributes {
		_, prepending := defaults[k].(AppendableAttributeValue)
		if k == "class" || k == "style" || prepending {
			appendable[k] = v
			continue
		}
		nonAppendable[k] = v
	}

	attributes := map[string]any{}
	for k, v := range appendable {
		var defaultsValue any
		if appendableDefault, ok := defaults[k].(AppendableAttributeValue); ok {
			defaultsValue = appendableDefault.Value
		} else {
			defaultsValue = defaults[k]
		}

		value := attributeToString(v)
		if k == "style" {
			value = str.Finish(value, ";")
		}

		attributes[k] = joinUnique(attributeToString(defaultsValue), value)
	}
	for k, v := range nonAppendable {
		attributes[k] = v
	}

	merged := make(map[string]any, len(defaults)+len(attributes))
	for k, v := range defaults {
		merged[k] = v
	}
	for k, v := range attributes {
		merged[k] = v
	}
	return NewComponentAttributeBag(merged)
}

// joinUnique joins values with a space, dropping empty pieces and any repeat
// of a piece already seen.
func joinUnique(values ...string) string {
	seen := map[string]bool{}
	kept := make([]string, 0, len(values))
	for _, v := range values {
		if v == "" || seen[v] {
			continue
		}
		seen[v] = true
		kept = append(kept, v)
	}
	return strings.Join(kept, " ")
}

// Prepends marks value as a default to be appended to, rather than replaced
// by, what the caller wrote.
func (b *ComponentAttributeBag) Prepends(value any) AppendableAttributeValue {
	return NewAppendableAttributeValue(value)
}

// IsEmpty reports whether the bag renders to an empty string.
func (b *ComponentAttributeBag) IsEmpty() bool {
	return strings.TrimSpace(b.String()) == ""
}

// IsNotEmpty reports the opposite of IsEmpty.
func (b *ComponentAttributeBag) IsNotEmpty() bool { return !b.IsEmpty() }

// GetAttributes is an alias for All.
func (b *ComponentAttributeBag) GetAttributes() map[string]any { return b.All() }

// SetAttributes replaces the bag's contents with attributes.
//
// A bag handed in under the "attributes" key is the parent's bag: it is
// merged away rather than stored, so a component that forwards its
// attributes to a child does not nest one bag inside another.
func (b *ComponentAttributeBag) SetAttributes(attributes map[string]any) {
	copied := make(map[string]any, len(attributes))
	for k, v := range attributes {
		copied[k] = v
	}

	if parent, ok := copied["attributes"].(*ComponentAttributeBag); ok {
		delete(copied, "attributes")
		copied = parent.Merge(copied).GetAttributes()
	}

	b.attributes = copied
}

// ExtractPropNames returns each key both as written and again in kebab case,
// because a prop declared as userName is written on the tag as user-name.
func ExtractPropNames(keys []string) []string {
	props := make([]string, 0, len(keys)*2)
	for _, key := range keys {
		props = append(props, key, str.Kebab(key))
	}
	return props
}

// ToHTML is an alias for String.
func (b *ComponentAttributeBag) ToHTML() string { return b.String() }

// OffsetExists reports whether offset is present in the bag with a non-nil
// value.
func (b *ComponentAttributeBag) OffsetExists(offset string) bool {
	v, ok := b.attributes[offset]
	return ok && v != nil
}

// OffsetGet is an alias for Get, without a fallback.
func (b *ComponentAttributeBag) OffsetGet(offset string) any { return b.Get(offset) }

// OffsetSet stores value under offset, initializing the bag if necessary.
func (b *ComponentAttributeBag) OffsetSet(offset string, value any) {
	if b.attributes == nil {
		b.attributes = map[string]any{}
	}
	b.attributes[offset] = value
}

// OffsetUnset removes offset from the bag.
func (b *ComponentAttributeBag) OffsetUnset(offset string) { delete(b.attributes, offset) }

// GetIterator returns the bag's attributes as a range-over-func sequence, in
// the bag's sorted key order.
func (b *ComponentAttributeBag) GetIterator() iter.Seq2[string, any] {
	return func(yield func(string, any) bool) {
		for _, k := range b.keys() {
			if !yield(k, b.attributes[k]) {
				return
			}
		}
	}
}

// JSONSerialize returns a copy of every attribute in the bag, for JSON
// encoding.
func (b *ComponentAttributeBag) JSONSerialize() map[string]any { return b.All() }

// ToArray is an alias for All.
func (b *ComponentAttributeBag) ToArray() map[string]any { return b.All() }

// String renders every attribute as name="value", space-separated.
//
// false and nil drop the attribute entirely; true writes the bare name, which
// is what checked and disabled need -- except for x-data and the wire: family,
// where true means an empty value.
//
// # The escape, and the one it replaced
//
// A value is escaped as HTML, and a name that could not survive being written
// as one is dropped rather than written.
//
// What stood here escaped a quote as \" -- faithful to the Blade method it was
// ported from, and wrong in HTML, which has no backslash escape. The parser
// reads the backslash as a character of the value and ends the attribute at the
// quote behind it, so the value
//
//	a" onerror=alert(1) x="
//
// came out as a closed attribute followed by an onerror the caller wrote. The
// name went out unchecked beside it, where a space or a quote does the same
// thing one field earlier.
//
// Dropping the bad name rather than reporting it is what a method returning
// only a string can do. It is why nothing in this framework's own components
// renders through this bag: they go through Attributes, which refuses, names
// the attribute, and gives the view's own line.
func (b *ComponentAttributeBag) String() string {
	var out strings.Builder
	for _, key := range b.keys() {
		value := b.attributes[key]

		if value == nil || !isAttributeName(key) {
			continue
		}
		if boolean, ok := value.(bool); ok {
			if !boolean {
				continue
			}
			if key == "x-data" || strings.HasPrefix(key, "wire:") {
				value = ""
			} else {
				value = key
			}
		}

		text := strings.TrimSpace(attributeToString(value))
		out.WriteString(" " + key + `="` + template.HTMLEscapeString(text) + `"`)
	}
	return strings.TrimSpace(out.String())
}

// isAttributeName reports whether name can be written as an attribute name.
//
// The set it refuses is the one the HTML syntax refuses, and no more: a name
// carries no character references, so a character that ends the name cannot be
// written as anything else. Everything outside that -- x-data, wire:model,
// @click, :class -- is a legal name here, and whether a component may carry it
// is a question this bag does not answer. Attributes answers it.
//
// The set is the compiler's, which reaches the same question from the other
// side: aru/internal/kyse/generate.go refuses these same characters where a
// view writes a name. It first read `" \t\n\r\f\"'>/=\x00"` here and that was
// short by three -- the less-than, the backtick, and the control characters
// below U+0020 that are not whitespace.
func isAttributeName(name string) bool {
	if name == "" {
		return false
	}
	if strings.ContainsAny(name, " \"'<>/=`") {
		return false
	}
	for i := 0; i < len(name); i++ {
		if name[i] < 0x20 || name[i] == 0x7f {
			return false
		}
	}
	return true
}

// attributeToString converts an attribute value to the string it renders as.
func attributeToString(value any) string {
	switch v := value.(type) {
	case nil:
		return ""
	case string:
		return v
	case AppendableAttributeValue:
		return v.String()
	case fmt.Stringer:
		return v.String()
	case bool:
		if v {
			return "1"
		}
		return ""
	default:
		return fmt.Sprint(v)
	}
}

// AppendableAttributeValue marks a default that is appended to what the
// caller wrote rather than replaced by it. Merge is the only thing that
// reads the marker.
type AppendableAttributeValue struct {
	// Value is the attribute value being prepended.
	Value any
}

// NewAppendableAttributeValue returns value wrapped as an appendable default.
func NewAppendableAttributeValue(value any) AppendableAttributeValue {
	return AppendableAttributeValue{Value: value}
}

// String returns the string form of the wrapped value.
func (v AppendableAttributeValue) String() string { return attributeToString(v.Value) }
