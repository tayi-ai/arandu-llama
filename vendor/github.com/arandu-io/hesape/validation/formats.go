package validation

import (
	"maps"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/arandu-io/hesape/collections/arr"
	"github.com/arandu-io/hesape/str"
)

// This file is how a failure becomes a sentence, and what a caller can do to
// change it.
//
// The rule table decides WHETHER a field failed. Everything here decides what it
// then says, in this order: an inline message passed to the validator, then
// validation.custom.<attribute>.<rule> from the translator, then the size
// variant of a size rule, then validation.<rule>, then the fallback messages a
// custom rule registered.
//
// The last resort is the sentence the compiled rule set carries ("is required"),
// never a bare translation key rendered to a person. So a Validator built with
// no translator still says something, and wiring one only changes the
// wording.

// sizeRules are the eight rules whose message depends on what is being measured
// -- a number, a file, a list or a string.
var sizeRules = []string{"size", "between", "min", "max", "gt", "lt", "gte", "lte"}

// Translator is the catalogue messages are read out of, narrowed to the two
// methods this package reaches for.
//
// Get returns any because a language line is a string and a key naming a group
// -- "validation.custom", "validation.attributes", "validation.min" -- is a map.
// A key with no line answers with the key itself, which is what every "is there
// a custom message" check here compares against.
//
// The locale is a parameter rather than state, as it is on
// hesape/translation.Translator: the locale belongs to the request. The empty
// string means the translator's own default.
type Translator interface {
	// Get returns the line under the key, or the key itself when there is none.
	Get(key string, replace map[string]any, locale string) any
	// Choice returns the line under the key, in the form that number calls for.
	Choice(key string, number int, replace map[string]any, locale string) string
}

// WithTranslator gives the Validator the catalogue its messages are read out of,
// which is what turns "is required" into "The name field is required.".
//
// It is optional. Without one the compiled rule set's own sentences are used,
// and every other layer -- inline messages, custom attribute names, the
// :placeholders -- still applies.
func WithTranslator(t Translator) ValidatorOption {
	return func(v *Validator) { v.trans = t }
}

// WithCustomMessages sets the sentence one field and one rule produce, keyed
// "field.rule", "rule" or "field", with "*" allowed in the field.
//
// A value is a sentence, or -- for a size rule -- a group of them keyed by
// "numeric", "file", "array" and "string".
func WithCustomMessages(messages map[string]any) ValidatorOption {
	return func(v *Validator) { v.customMessages = messages }
}

// WithCustomAttributes sets how a field is named inside a sentence, which is
// what the validation.attributes lines of a catalogue also carry.
func WithCustomAttributes(attributes map[string]string) ValidatorOption {
	return func(v *Validator) { v.customAttributes = attributes }
}

// WithCustomValues sets how one value of one field is spelled inside a sentence,
// so that ":value" reads "credit card" and not "cc".
func WithCustomValues(values map[string]map[string]string) ValidatorOption {
	return func(v *Validator) { v.customValues = values }
}

// translator is the Translator this validator reads lines out of. It is never
// nil, so a caller does not have to ask; getMessage asks v.trans directly when
// the difference matters.
func (v *Validator) translator() Translator {
	if v.trans == nil {
		return englishTranslator{}
	}
	return v.trans
}

// GetTranslator returns the catalogue messages are read out of, which is never
// nil: without one it is a translator that finds no line.
func (v *Validator) GetTranslator() Translator { return v.translator() }

// SetTranslator sets the catalogue messages are read out of.
func (v *Validator) SetTranslator(translator Translator) { v.trans = translator }

// CustomMessages returns the inline messages in force, which a rule object reads
// to build a sibling validator that says the same things.
func (v *Validator) CustomMessages() map[string]any { return v.customMessages }

// CustomAttributes returns the field names in force, read for the reason
// CustomMessages is.
func (v *Validator) CustomAttributes() map[string]string { return v.customAttributes }

// SetCustomMessages merges more inline messages into the ones in force.
func (v *Validator) SetCustomMessages(messages map[string]any) *Validator {
	if v.customMessages == nil {
		v.customMessages = map[string]any{}
	}
	maps.Copy(v.customMessages, messages)

	return v
}

// SetAttributeNames replaces the field names in force.
func (v *Validator) SetAttributeNames(attributes map[string]string) *Validator {
	v.customAttributes = attributes

	return v
}

// AddCustomAttributes merges more field names into the ones in force.
func (v *Validator) AddCustomAttributes(attributes map[string]string) *Validator {
	if v.customAttributes == nil {
		v.customAttributes = map[string]string{}
	}
	maps.Copy(v.customAttributes, attributes)

	return v
}

// SetValueNames replaces the value names in force.
func (v *Validator) SetValueNames(values map[string]map[string]string) *Validator {
	v.customValues = values

	return v
}

// AddCustomValues merges more value names into the ones in force.
func (v *Validator) AddCustomValues(values map[string]map[string]string) *Validator {
	if v.customValues == nil {
		v.customValues = map[string]map[string]string{}
	}
	maps.Copy(v.customValues, values)

	return v
}

// SetFallbackMessages sets the sentence a rule registered with Factory.Extend
// says when nothing else names one.
func (v *Validator) SetFallbackMessages(messages map[string]any) {
	v.fallbackMessages = messages
}

// SetImplicitAttributesFormatter sets how a wildcard attribute that nothing named
// is spelled.
func (v *Validator) SetImplicitAttributesFormatter(formatter func(string) string) *Validator {
	v.implicitAttributesFormatter = formatter

	return v
}

// AddReplacer registers how one rule fills the placeholders of its own
// message.
func (v *Validator) AddReplacer(rule string, replacer ReplacerFunc) {
	if v.replacers == nil {
		v.replacers = map[string]ReplacerFunc{}
	}
	v.replacers[str.Snake(rule, "_")] = replacer
}

// AddReplacers registers several replacers at once.
func (v *Validator) AddReplacers(replacers map[string]ReplacerFunc) {
	for rule, replacer := range replacers {
		v.AddReplacer(rule, replacer)
	}
}

// ---------------------------------------------------------------------------
// The message itself.
// ---------------------------------------------------------------------------

// getMessage returns the sentence one attribute and one rule produce, before its
// placeholders are filled.
func (v *Validator) getMessage(attribute, rule string) string {
	lowerRule := str.Snake(rule, "_")

	// First we will retrieve the custom message for the validation rule if one
	// exists. If a custom validation message is being used we'll return the
	// custom message, otherwise we'll keep searching for a valid message.
	if inline, found := v.getInlineMessage(attribute, lowerRule); found {
		return inline
	}

	if v.trans != nil {
		customKey := "validation.custom." + attribute + "." + lowerRule

		keys := []string{customKey}
		if slices.Contains(sizeRules, lowerRule) {
			keys = []string{customKey + "." + v.getAttributeType(attribute), customKey}
		}

		// Then a custom defined validation message for the attribute and rule.
		// This allows the developer to specify specific messages for only some
		// attributes and rules that need to get specially formed.
		if customMessage := v.getCustomMessageFromTranslator(keys); customMessage != customKey {
			return customMessage
		}

		// If the rule being validated is a "size" rule, we will need to gather
		// the specific error message for the type of attribute being validated,
		// which all have different message types.
		if slices.Contains(sizeRules, lowerRule) {
			return v.getSizeMessage(attribute, lowerRule)
		}

		key := "validation." + lowerRule

		if value := line(v.translator().Get(key, nil, ""), key); value != key {
			return value
		}
	}

	if fallback, found := v.getFromLocalArray(attribute, lowerRule, v.fallbackMessages); found {
		if sentence, isSentence := fallback.(string); isSentence && sentence != "" {
			return sentence
		}
	}

	// The last resort is the sentence the compiled rule set carries, which is
	// the one the field is drawn with.
	return v.messageFor(attribute, lowerRule)
}

// getInlineMessage returns the message the caller passed to the validator, with
// the size variant picked out when the entry is a group and the rule measures
// something.
func (v *Validator) getInlineMessage(attribute, rule string) (string, bool) {
	entry, found := v.getFromLocalArray(attribute, str.Snake(rule, "_"), v.customMessages)
	if !found {
		return "", false
	}
	if group, isGroup := asGroup(entry); isGroup && slices.Contains(sizeRules, rule) {
		sentence, held := group[v.getAttributeType(attribute)]
		return sentence, held
	}
	sentence, isSentence := entry.(string)
	return sentence, isSentence
}

// getFromLocalArray returns the message a source map holds for one attribute and
// rule, by three keys tried in order -- "attribute.rule", "rule", "attribute" --
// with a source key containing "*" matched as a pattern.
//
// The source keys are read in sorted order, because a Go map remembers none and
// two patterns that both match would otherwise answer differently between
// runs.
func (v *Validator) getFromLocalArray(attribute, lowerRule string, source map[string]any) (any, bool) {
	keys := []string{attribute + "." + lowerRule, lowerRule, attribute}

	sourceKeys := slices.Sorted(maps.Keys(source))

	for _, key := range keys {
		for _, sourceKey := range sourceKeys {
			if strings.Contains(sourceKey, "*") {
				if !wildcardPattern(sourceKey).MatchString(key) {
					continue
				}
				message := source[sourceKey]

				if group, isGroup := asGroup(message); isGroup {
					if sentence, held := group[lowerRule]; held {
						return sentence, true
					}
				}
				return message, true
			}

			if str.Is([]string{sourceKey}, key, false) {
				message := source[sourceKey]

				if sourceKey == attribute {
					if group, isGroup := asGroup(message); isGroup {
						sentence, held := group[lowerRule]
						return sentence, held
					}
				}
				return message, true
			}
		}
	}
	return nil, false
}

// getCustomMessageFromTranslator returns the first of the given keys the
// translator holds a line for, trying an exact match and then a wildcard one
// under validation.custom, and answering with the last key when it holds none.
func (v *Validator) getCustomMessageFromTranslator(keys []string) string {
	for _, key := range keys {
		if message := line(v.translator().Get(key, nil, ""), key); message != key {
			return message
		}

		// If an exact match was not found for the key, we will collapse all of
		// these messages and loop through them and try to find a wildcard match
		// for the given key. Otherwise, we will simply return the key back out.
		shortKey := strings.TrimPrefix(key, "validation.custom.")

		custom, _ := v.translator().Get("validation.custom", nil, "").(map[string]any)

		if message := v.getWildcardCustomMessages(arr.Dot(custom), shortKey, key); message != key {
			return message
		}
	}
	if len(keys) == 0 {
		return ""
	}
	return keys[len(keys)-1]
}

// getWildcardCustomMessages returns the line whose key matches the search as a
// pattern, and def when none does.
func (v *Validator) getWildcardCustomMessages(messages map[string]any, search, def string) string {
	for _, key := range slices.Sorted(maps.Keys(messages)) {
		if search == key || (strings.Contains(key, "*") && str.Is([]string{key}, search, false)) {
			return line(messages[key], def)
		}
	}
	return def
}

// getSizeMessage returns the line of a size rule.
//
// There are four types of size validation. The attribute may be a number, a
// file, a list or a string, so the line is read out of the group its type names.
func (v *Validator) getSizeMessage(attribute, rule string) string {
	key := "validation." + str.Snake(rule, "_") + "." + v.getAttributeType(attribute)

	return line(v.translator().Get(key, nil, ""), key)
}

// getAttributeType names what an attribute's size measures: numeric, file, array
// or string.
func (v *Validator) getAttributeType(attribute string) string {
	// We assume that the attributes present in the file array are files, so if
	// the attribute does not have a numeric rule and is not a file we consider
	// it a string by elimination.
	switch {
	case v.HasRule(attribute, numericRules):
		return "numeric"
	case v.HasRule(attribute, []string{"array", "list"}):
		return "array"
	}
	if _, isFile := asFile(v.GetValue(attribute)); isFile {
		return "file"
	}
	return "string"
}

// MakeReplacements fills every placeholder of a message with what actually
// happened.
//
// It is exported because a rule object reaches for it.
func (v *Validator) MakeReplacements(message, attribute, rule string, parameters []string) string {
	message = v.replaceAttributePlaceholder(message, v.GetDisplayableAttribute(attribute))

	message = v.replaceInputPlaceholder(message, attribute)
	message = v.replaceIndexPlaceholder(message, attribute)
	message = v.replacePositionPlaceholder(message, attribute)

	lowerRule := str.Snake(rule, "_")

	if replacer, registered := v.replacers[lowerRule]; registered {
		return replacer(message, attribute, lowerRule, parameters, v)
	}
	if replacer, held := replacers[lowerRule]; held {
		return replacer(v, message, attribute, lowerRule, parameters)
	}

	return message
}

// GetDisplayableAttribute returns the name of a field as a sentence names it.
//
// The default is the snake-cased name with its underscores turned into spaces:
// "password_confirmation" reads "password confirmation", LOWERCASE. The name
// sits inside the sentence ("The password confirmation field must match") and
// never at the head of one, so it is never capitalised here.
func (v *Validator) GetDisplayableAttribute(attribute string) string {
	primaryAttribute := v.getPrimaryAttribute(attribute)

	expectedAttributes := []string{attribute}
	if attribute != primaryAttribute {
		expectedAttributes = []string{attribute, primaryAttribute}
	}

	for _, name := range expectedAttributes {
		// The developer may dynamically specify the array of custom attributes
		// on this validator instance. If the attribute exists in this array it
		// is used over the other ways of pulling the attribute name.
		if inline := v.getAttributeFromLocalArray(name, v.customAttributes); inline != "" {
			return inline
		}

		// We allow for a developer to specify language lines for any attribute
		// in this application, which allows flexibility for displaying a unique
		// displayable version of the attribute name.
		if translated := v.getAttributeFromTranslations(name); translated != "" {
			return translated
		}
	}

	// When no language line has been specified for the attribute and it is also
	// an implicit attribute we will display the raw attribute's name and not
	// modify it with any of these replacements before we display the name.
	if _, implicit := v.implicitAttributes[primaryAttribute]; implicit {
		if v.implicitAttributesFormatter != nil {
			return v.implicitAttributesFormatter(attribute)
		}
		return attribute
	}

	return strings.ReplaceAll(str.Snake(attribute, "_"), "_", " ")
}

// getPrimaryAttribute returns the wildcard an attribute was expanded from: given
// "name.0", the "name.*".
func (v *Validator) getPrimaryAttribute(attribute string) string {
	for _, unparsed := range slices.Sorted(maps.Keys(v.implicitAttributes)) {
		if slices.Contains(v.implicitAttributes[unparsed], attribute) {
			return unparsed
		}
	}

	return attribute
}

// getAttributeFromTranslations returns the field name the validation.attributes
// lines carry, and empty when they carry none.
func (v *Validator) getAttributeFromTranslations(name string) string {
	attributes, isGroup := v.translator().Get("validation.attributes", nil, "").(map[string]any)
	if !isGroup {
		return ""
	}
	return v.getAttributeFromLocalArray(name, dotStrings(arr.Dot(attributes)))
}

// getAttributeFromLocalArray returns the field name a source map holds, with the
// keys read in sorted order for the reason getFromLocalArray sorts them.
func (v *Validator) getAttributeFromLocalArray(attribute string, source map[string]string) string {
	if name, held := source[attribute]; held {
		return name
	}
	for _, sourceKey := range slices.Sorted(maps.Keys(source)) {
		if strings.Contains(sourceKey, "*") && wildcardPattern(sourceKey).MatchString(attribute) {
			return source[sourceKey]
		}
	}
	return ""
}

// replaceAttributePlaceholder fills the three spellings of :attribute, which is
// how a line chooses its own capitalisation.
func (v *Validator) replaceAttributePlaceholder(message, value string) string {
	message = strings.ReplaceAll(message, ":attribute", value)
	message = strings.ReplaceAll(message, ":ATTRIBUTE", strings.ToUpper(value))
	return strings.ReplaceAll(message, ":Attribute", str.Ucfirst(value))
}

// replaceIndexPlaceholder fills :index with the member's position, counted from
// zero.
func (v *Validator) replaceIndexPlaceholder(message, attribute string) string {
	return v.replaceIndexOrPositionPlaceholder(message, attribute, "index", nil)
}

// replacePositionPlaceholder fills :position with the member's position, counted
// from one.
func (v *Validator) replacePositionPlaceholder(message, attribute string) string {
	return v.replaceIndexOrPositionPlaceholder(message, attribute, "position", func(segment int) int {
		return segment + 1
	})
}

// replaceIndexOrPositionPlaceholder is the body the index and position
// placeholders share, in all three of their spellings.
func (v *Validator) replaceIndexOrPositionPlaceholder(message, attribute, placeholder string, modifier func(int) int) string {
	if modifier == nil {
		modifier = func(value int) int { return value }
	}

	numericIndex := 1

	for _, segment := range strings.Split(attribute, ".") {
		index, err := strconv.Atoi(segment)
		if err != nil {
			continue
		}
		if numericIndex == 1 {
			message = replaceIgnoreCase(message, ":"+placeholder, strconv.Itoa(modifier(index)))
		}

		message = replaceIgnoreCase(
			message,
			":"+numberToIndexOrPositionWord(numericIndex)+"-"+placeholder,
			strconv.Itoa(modifier(index)),
		)

		numericIndex++
	}

	return message
}

// numberToIndexOrPositionWord names the first ten positions in words, and gives
// the number back for anything past them.
func numberToIndexOrPositionWord(value int) string {
	words := []string{
		1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
		6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
	}
	if value >= 1 && value < len(words) {
		return words[value]
	}
	return "other"
}

// replaceInputPlaceholder fills :input with what was actually sent, which is the
// placeholder that turns "is invalid" into a sentence somebody can act on.
func (v *Validator) replaceInputPlaceholder(message, attribute string) string {
	actualValue := v.GetValue(attribute)

	if isScalar(actualValue) || actualValue == nil {
		message = strings.ReplaceAll(message, ":input", v.GetDisplayableValue(attribute, actualValue))
	}

	return message
}

// GetDisplayableValue returns how a value is spelled inside a message, after the
// custom values an application declared and the validation.values lines.
func (v *Validator) GetDisplayableValue(attribute string, value any) string {
	if group, held := v.customValues[attribute]; held {
		if name, named := group[stringOf(value)]; named {
			return name
		}
	}

	if isArray(value) {
		return "array"
	}

	key := "validation.values." + attribute + "." + stringOf(value)

	if sentence := line(v.translator().Get(key, nil, ""), key); sentence != key {
		return sentence
	}

	if boolean, isBool := value.(bool); isBool {
		if boolean {
			return "true"
		}
		return "false"
	}

	if value == nil {
		return "empty"
	}

	return stringOf(value)
}

// getAttributeList spells a list of field names the way a sentence names each of
// them.
func (v *Validator) getAttributeList(values []string) []string {
	attributes := make([]string, len(values))

	// For each attribute in the list we will simply get its displayable form, as
	// this is convenient when replacing lists of parameters like some of the
	// replacement functions do when formatting out the validation message.
	for i, value := range values {
		attributes[i] = v.GetDisplayableAttribute(value)
	}

	return attributes
}

// ---------------------------------------------------------------------------
// The small readings the two concerns share.
// ---------------------------------------------------------------------------

// line reads a translated value as a sentence. A key naming a group answers with
// the key, which is how every caller here spells "there is no line for this".
func line(value any, key string) string {
	if sentence, isSentence := value.(string); isSentence {
		return sentence
	}
	return key
}

// param reads one parameter, answering the empty string for a position past the
// end.
func param(parameters []string, i int) string {
	if i < 0 || i >= len(parameters) {
		return ""
	}
	return parameters[i]
}

// isScalar reports whether the value is a single one: text, a bool or a number.
func isScalar(value any) bool {
	switch value.(type) {
	case string, bool, int, int8, int16, int32, int64,
		uint, uint8, uint16, uint32, uint64, float32, float64:
		return true
	}
	return false
}

// asGroup reads a message entry that is a group of lines rather than one line,
// which is the shape a size rule's four variants arrive in.
func asGroup(entry any) (map[string]string, bool) {
	switch typed := entry.(type) {
	case map[string]string:
		return typed, true
	case map[string]any:
		group := make(map[string]string, len(typed))
		for key, value := range typed {
			sentence, isSentence := value.(string)
			if !isSentence {
				return nil, false
			}
			group[key] = sentence
		}
		return group, true
	}
	return nil, false
}

// dotStrings narrows a flattened translation group to the strings in it.
func dotStrings(flattened map[string]any) map[string]string {
	out := make(map[string]string, len(flattened))
	for key, value := range flattened {
		if sentence, isSentence := value.(string); isSentence {
			out[key] = sentence
		}
	}
	return out
}

// wildcardPattern compiles a source key containing "*" into a pattern: a star
// matches one segment, and the whole key must match.
func wildcardPattern(sourceKey string) *regexp.Regexp {
	pattern := strings.ReplaceAll(regexp.QuoteMeta(sourceKey), `\*`, `([^.]*)`)
	compiled, err := regexp.Compile(`^` + pattern + `$`)
	if err != nil {
		// A key that does not compile matches nothing, which is what a source
		// key nobody meant as a pattern should do.
		return regexp.MustCompile(`$.^`)
	}
	return compiled
}

// replaceIgnoreCase replaces without regard to case, which the index and position
// placeholders are read with so that :INDEX and :Index land too.
func replaceIgnoreCase(subject, search, replacement string) string {
	if search == "" {
		return subject
	}
	lowerSubject, lowerSearch := strings.ToLower(subject), strings.ToLower(search)

	var b strings.Builder
	for {
		i := strings.Index(lowerSubject, lowerSearch)
		if i < 0 {
			b.WriteString(subject)
			return b.String()
		}
		b.WriteString(subject[:i])
		b.WriteString(replacement)
		subject, lowerSubject = subject[i+len(search):], lowerSubject[i+len(search):]
	}
}
