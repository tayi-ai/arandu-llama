package validation

// refused names the rule spellings this package does not have, and says what to
// write instead.
//
// Every rule there is lives in specs. What is left here is a second spelling of
// one of them: an alias, a name somebody remembers wrong, or a question this
// package answers under another name. A name in here is a redirection, not a
// gap -- so a rule set carrying one fails at boot with a sentence somebody can
// act on, rather than with "unknown rule", which reads as an omission and
// invites a pull request adding it.
//
// The sentence completes: `field "x": rule "y" ` + this.
var refused = map[string]string{
	"int":        "is a shorthand nobody writes in a rule string. Write integer",
	"bool":       "is a shorthand nobody writes in a rule string. Write boolean",
	"notregex":   "is missing its underscore. Write not_regex",
	"array_keys": "is not the name of the rule. Write required_array_keys",
	"in_array_keys": "compares the KEYS of another field's array, and a submitted form has none " +
		"to compare. Write the check in Go, in the arandu:begin custom block",
	"doesnt_contain": "is the refusal of contains, and there is no such rule. Write " +
		"not_in for one value, or the check in Go, in the arandu:begin custom block",
	"base64":   "decodes a value nothing then reads. Decode it in the service, which is what does read it",
	"encoding": "is not a rule this package has. Write mimetypes for an upload",
	"filled_if": "is not a rule this package has. Write required_if, which is the " +
		"same question about a field that was sent",
	"same_as": "is not the name of the rule. Write same",
	"unique_with": "comes from a third-party package, not from this one. Write unique with " +
		"its extra conditions: unique:table,column,NULL,id,other_column,value",
}
