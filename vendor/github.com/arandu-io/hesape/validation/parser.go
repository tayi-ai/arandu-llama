package validation

import (
	"maps"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

// This file is how a rule string becomes a name and its parameters, and how
// "items.*.price" becomes one rule per item the request actually sent.
//
// Compile does the same walk at boot for a rule set written in a package-level
// variable. These are the same steps as a value, for the caller that builds a
// rule set from something that is not a literal.
//
// The wildcard half is NOT boot work and cannot be: "items.*.price" means the
// items this request sent, so Validator.explodeRules calls
// explodeWildcardRules once per request, over the compiled set.

// ExplodedRules is a rule set with every wildcard expanded, together with the
// wildcard keys that produced it.
type ExplodedRules struct {
	// Rules is one entry per attribute, holding the rules written against it in
	// the order they were written.
	Rules map[string][]string

	// Order is the order the attributes were written in, because a Go map
	// remembers none.
	Order []string

	// ImplicitAttributes is the wildcard key each expanded attribute came from,
	// which is what names a field in a message the way the caller wrote it.
	ImplicitAttributes map[string][]string
}

// ValidationRuleParser turns a rule set as it was written into the rule set the
// validator runs: one list of rules per attribute, with every wildcard
// attribute -- "items.*.price" -- replaced by the concrete keys the request
// actually carries. It holds the request data because that expansion cannot be
// decided without it, and it remembers which wildcard each expanded key came
// from, so a message can name the field the way the caller wrote it.
type ValidationRuleParser struct {
	// Data is the request the wildcards are expanded against, because "items.*"
	// means the items that were actually sent.
	Data Data

	// ImplicitAttributes is the wildcard key each expanded attribute came from,
	// filled in as the expansion runs.
	ImplicitAttributes map[string][]string
}

// NewValidationRuleParser returns a parser that expands wildcards against data.
func NewValidationRuleParser(data Data) *ValidationRuleParser {
	return &ValidationRuleParser{Data: data, ImplicitAttributes: map[string][]string{}}
}

// Explode turns the rules as they were written into the full rule set the
// validator runs.
//
// The primary purpose of this parser is to expand any "*" rules to all of the
// explicit rules needed for the given data. For example the rule names.* would
// get expanded to names.0, names.1, and so on.
func (p *ValidationRuleParser) Explode(rules Rules) *ExplodedRules {
	p.ImplicitAttributes = map[string][]string{}

	exploded := &ExplodedRules{Rules: map[string][]string{}}

	for _, attribute := range slices.Sorted(maps.Keys(rules)) {
		if !strings.Contains(attribute, "*") {
			p.mergeInto(exploded, attribute, splitChain(rules[attribute]))
			continue
		}
		for _, key := range p.explodeWildcardRules(attribute) {
			p.ImplicitAttributes[attribute] = append(p.ImplicitAttributes[attribute], key)
			p.mergeInto(exploded, key, splitChain(rules[attribute]))
		}
	}

	exploded.ImplicitAttributes = p.ImplicitAttributes

	return exploded
}

func (p *ValidationRuleParser) mergeInto(exploded *ExplodedRules, attribute string, rules []string) {
	if _, held := exploded.Rules[attribute]; !held {
		exploded.Order = append(exploded.Order, attribute)
	}
	exploded.Rules[attribute] = append(exploded.Rules[attribute], rules...)
}

// explodeWildcardRules returns the keys of the data that one wildcard attribute
// names, sorted so that two runs agree.
func (p *ValidationRuleParser) explodeWildcardRules(attribute string) []string {
	pattern := strings.ReplaceAll(regexp.QuoteMeta(attribute), `\*`, `[^.]*`)

	compiled, err := regexp.Compile(`^` + pattern + `$`)
	if err != nil {
		return nil
	}

	gathered := InitializeAndGatherData(attribute, p.Data)

	var keys []string
	for _, key := range slices.Sorted(maps.Keys(gathered)) {
		if strings.HasPrefix(key, attribute) || compiled.MatchString(key) {
			keys = append(keys, key)
		}
	}
	return keys
}

// MergeRules adds more rules to an attribute that may already have some.
func (p *ValidationRuleParser) MergeRules(results Rules, attribute, rules string) Rules {
	if results == nil {
		results = Rules{}
	}
	if existing, held := results[attribute]; held && existing != "" && rules != "" {
		results[attribute] = existing + "|" + rules
		return results
	}
	if rules != "" {
		results[attribute] = rules
	}
	return results
}

// Parse splits a rule string into its name and the parameters written after the
// colon.
//
// The name stays as it was typed, which is how every table in this package keys
// it. Its two aliases are normalized: "int" is "integer" and "bool" is
// "boolean".
//
// A malformed parameter list answers with no parameters rather than an error;
// Compile is where a malformed list is reported, with the field and the file.
func Parse(rule string) (string, []string) {
	name, rest, hasArgs := strings.Cut(rule, ":")
	name = normalizeRule(strings.TrimSpace(name))

	parameters, err := parseArgs(name, rest, hasArgs)
	if err != nil {
		return name, nil
	}
	return name, parameters
}

// normalizeRule folds the two aliases into the names the catalogue keys.
func normalizeRule(rule string) string {
	switch rule {
	case "int":
		return "integer"
	case "bool":
		return "boolean"
	}
	return rule
}

// FilterConditionalRules replaces every ConditionalRules in the set with the
// rules its condition chose.
func FilterConditionalRules(rules map[string]any, data Data) Rules {
	out := Rules{}

	for attribute, written := range rules {
		switch typed := written.(type) {
		case string:
			out[attribute] = typed
		case *ConditionalRules:
			if typed.Passes(data) {
				out[attribute] = strings.Join(typed.Rules(data), "|")
				continue
			}
			out[attribute] = strings.Join(typed.DefaultRules(data), "|")
		case []string:
			out[attribute] = strings.Join(typed, "|")
		}
	}

	return out
}

// ---------------------------------------------------------------------------
// Rules chosen by a condition.
// ---------------------------------------------------------------------------

// ConditionalRules is the rules an attribute gets when a condition holds, and
// the ones it gets when it does not.
type ConditionalRules struct {
	// condition is asked of the whole request. A caller holding a bool wraps it
	// in a func that ignores its argument.
	condition func(Data) bool

	rules        []string
	defaultRules []string
}

// NewConditionalRules returns rules chosen by the condition: rules when it
// holds, and defaultRules when it does not.
func NewConditionalRules(condition func(Data) bool, rules string, defaultRules ...string) *ConditionalRules {
	c := &ConditionalRules{condition: condition, rules: splitChain(rules)}
	if len(defaultRules) > 0 && defaultRules[0] != "" {
		c.defaultRules = splitChain(defaultRules[0])
	}
	return c
}

// Passes reports whether the condition holds for this request. A nil condition
// does not hold.
func (c *ConditionalRules) Passes(data Data) bool {
	if c.condition == nil {
		return false
	}
	return c.condition(data)
}

// Rules returns the rules used when the condition holds.
func (c *ConditionalRules) Rules(data Data) []string { return c.rules }

// DefaultRules returns the rules used when the condition does not hold.
func (c *ConditionalRules) DefaultRules(data Data) []string { return c.defaultRules }

// ---------------------------------------------------------------------------
// Rules decided per member of an array.
// ---------------------------------------------------------------------------

// NestedRules is the rules for one member of an array, decided by looking at
// that member. Build one with NewNestedRules.
type NestedRules struct {
	callback func(value any, attribute string, data Data) Rules
}

// NewNestedRules returns rules the callback decides, member by member.
func NewNestedRules(callback func(value any, attribute string, data Data) Rules) *NestedRules {
	return &NestedRules{callback: callback}
}

// Compile asks the callback what this member's rules are, and expands them. A
// nil callback compiles to nothing.
func (n *NestedRules) Compile(attribute string, value any, data Data) *ExplodedRules {
	if n.callback == nil {
		return &ExplodedRules{Rules: map[string][]string{}, ImplicitAttributes: map[string][]string{}}
	}
	return NewValidationRuleParser(data).Explode(n.callback(value, attribute, data))
}

// ---------------------------------------------------------------------------
// Reading the slice of the request one attribute names.
// ---------------------------------------------------------------------------

// InitializeAndGatherData returns the slice of the request one attribute names,
// flattened, plus the exact keys a wildcard matched.
func InitializeAndGatherData(attribute string, masterData Data) map[string]any {
	initialized := initializeAttributeOnData(attribute, masterData)

	gathered := arrDot(initialized)

	maps.Copy(gathered, extractValuesForWildcards(masterData, gathered, attribute))

	return gathered
}

// initializeAttributeOnData fills in the keys a wildcard attribute names but the
// request did not send.
//
// A wildcard attribute NOBODY SENT A VALUE FOR still has to produce a key, or
// the rules written against it never run. With "foo.*.bar" against
// {"foo": [{"baz": "x"}]} the flattened data holds only foo.0.baz, so
// explodeWildcardRules would find no key and the field would expand to nothing
// -- which is how "foo.*.bar": "required_with:foo.*.baz" passes on a request
// that should fail. Filling foo.0.bar with null is what puts the key there.
//
// The slice of the request is deep-copied first: a Go map and a Go slice are
// references, so writing the null through would write it into the request's own
// data.
func initializeAttributeOnData(attribute string, masterData Data) Data {
	explicitPath := GetLeadingExplicitAttributePath(attribute)

	data, _ := deepClone(ExtractDataFromPath(explicitPath, masterData)).(Data)

	if !strings.Contains(attribute, "*") || strings.HasSuffix(attribute, "*") {
		return data
	}

	filled, _ := dataSet(data, strings.Split(attribute, "."), nil).(Data)

	return filled
}

// dataSet writes the value at the dotted path, walking every member where the
// path says "*", and making the levels it needs on the way.
//
// It overwrites what it finds, which is safe because the target is the throwaway
// copy this gathers KEYS out of: the values are read back from the request by
// extractValuesForWildcards.
func dataSet(target any, segments []string, value any) any {
	if len(segments) == 0 {
		return value
	}
	segment, rest := segments[0], segments[1:]

	if segment == "*" {
		switch node := target.(type) {
		case Data:
			for key, inner := range node {
				node[key] = dataSet(inner, rest, value)
			}
			return node
		case map[string]any:
			return dataSet(Data(node), segments, value)
		case []any:
			for i, inner := range node {
				node[i] = dataSet(inner, rest, value)
			}
			return node
		}
		// A target the wildcard cannot walk becomes an empty array, and the
		// loop over it writes nothing.
		return Data{}
	}

	switch node := target.(type) {
	case Data:
		if len(rest) == 0 {
			node[segment] = value
			return node
		}
		inner, held := node[segment]
		if !held {
			inner = Data{}
		}
		node[segment] = dataSet(inner, rest, value)
		return node
	case map[string]any:
		return dataSet(Data(node), segments, value)
	case []any:
		i, err := strconv.Atoi(segment)
		if err != nil || i < 0 || i >= len(node) {
			// A request cannot address past the end of what it sent, and
			// growing the list here would invent a field.
			return node
		}
		if len(rest) == 0 {
			node[i] = value
			return node
		}
		node[i] = dataSet(node[i], rest, value)
		return node
	}

	// Not walkable: replace it with an array and write into that.
	made := Data{}
	if len(rest) == 0 {
		made[segment] = value
		return made
	}
	made[segment] = dataSet(Data{}, rest, value)
	return made
}

// deepClone copies a slice of the request all the way down, so that dataSet
// cannot write through to the map the caller handed in.
func deepClone(value any) any {
	switch node := value.(type) {
	case Data:
		out := make(Data, len(node))
		for key, item := range node {
			out[key] = deepClone(item)
		}
		return out
	case map[string]any:
		out := make(Data, len(node))
		for key, item := range node {
			out[key] = deepClone(item)
		}
		return out
	case []any:
		out := make([]any, len(node))
		for i, item := range node {
			out[i] = deepClone(item)
		}
		return out
	}
	return value
}

// extractValuesForWildcards reads back, out of the request itself, the value at
// every key the wildcard matched.
func extractValuesForWildcards(masterData Data, gathered map[string]any, attribute string) map[string]any {
	pattern := strings.ReplaceAll(regexp.QuoteMeta(attribute), `\*`, `[^.]+`)

	compiled, err := regexp.Compile(`^` + pattern)
	if err != nil {
		return map[string]any{}
	}

	out := map[string]any{}
	for _, key := range slices.Sorted(maps.Keys(gathered)) {
		if match := compiled.FindString(key); match != "" {
			out[match] = masterData.Get(match)
		}
	}
	return out
}

// ExtractDataFromPath returns the sub-section of the request one dotted path
// names, so that the rest of it is not walked.
func ExtractDataFromPath(attribute string, masterData Data) Data {
	results := Data{}

	if attribute == "" {
		maps.Copy(results, masterData)
		return results
	}

	if value, present := lookup(masterData, attribute); present {
		setPath(results, attribute, value)
	}

	return results
}

// GetLeadingExplicitAttributePath returns the part of a path before its first
// wildcard: "foo.bar.*.baz" gives "foo.bar", which is all of it that can be
// walked without guessing.
func GetLeadingExplicitAttributePath(attribute string) string {
	head, _, _ := strings.Cut(attribute, "*")

	return strings.TrimRight(head, ".")
}

// arrDot flattens the shapes Data holds into one map of dotted keys.
func arrDot(data Data) map[string]any {
	out := map[string]any{}
	flattenInto(out, "", data)
	return out
}

func flattenInto(out map[string]any, prefix string, value any) {
	switch node := value.(type) {
	case Data:
		if len(node) == 0 && prefix != "" {
			out[prefix] = node
			return
		}
		for key, item := range node {
			flattenInto(out, join(prefix, key), item)
		}
	case map[string]any:
		flattenInto(out, prefix, Data(node))
	case []any:
		if len(node) == 0 && prefix != "" {
			out[prefix] = node
			return
		}
		for i, item := range node {
			flattenInto(out, join(prefix, itoa(i)), item)
		}
	default:
		if prefix != "" {
			out[prefix] = value
		}
	}
}

func join(prefix, key string) string {
	if prefix == "" {
		return key
	}
	return prefix + "." + key
}

// setPath writes a value at a dotted path, making the levels it needs on the
// way.
func setPath(data Data, key string, value any) {
	segments := strings.Split(key, ".")

	current := data
	for _, segment := range segments[:len(segments)-1] {
		next, held := current[segment]
		if !held {
			made := Data{}
			current[segment] = made
			current = made
			continue
		}
		nested, isNested := asData(next)
		if !isNested {
			made := Data{}
			current[segment] = made
			current = made
			continue
		}
		current[segment] = nested
		current = nested
	}
	current[segments[len(segments)-1]] = value
}

// asData reads a value that is a nested array, in either spelling.
func asData(value any) (Data, bool) {
	switch typed := value.(type) {
	case Data:
		return typed, true
	case map[string]any:
		return Data(typed), true
	}
	return nil, false
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var digits [20]byte
	pos := len(digits)
	for i > 0 {
		pos--
		digits[pos] = byte('0' + i%10)
		i /= 10
	}
	return string(digits[pos:])
}
