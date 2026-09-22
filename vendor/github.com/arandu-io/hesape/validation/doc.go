// Package validation checks a submitted request against a set of rules.
//
// A rule is written as a string, and a field's rules are one string:
//
//	var Register = validation.MustCompile(validation.Rules{
//		"name":     "required|max:255",
//		"email":    "required|email",
//		"password": "required|min:12|confirmed",
//	})
//
//	in, err := ctx.Validate(requests.Register)
//	if err != nil {
//		return err
//	}
//
// A rule set is compiled ONCE, in a package-level variable, and every rule
// string is parsed and checked there: an unknown rule, a missing or unparseable
// argument, a pattern that does not compile, a cross-field reference naming a
// field that does not exist -- each of those fails at boot, naming the field,
// the rule and the file, and all of them are reported together. A rule set that
// boots is a rule set whose names are all real.
//
// # Where the files are
//
//	attributes.go    one ValidateX method per rule
//	rulebuilders.go  the typed builders, for a rule nobody wants to spell
//	validator.go     the Validator, which runs a compiled set over one request
//	messages.go      Errors, the messages a failed run collected
//	compile.go       the boot-time compile; parser.go is the per-request half
//	rules.go         the catalogue: arity, check and message of every rule
//
// # Four things worth knowing before writing a rule string
//
//   - date_format takes a GO layout -- date_format:2006-01-02, never
//     date_format:Y-m-d. It is a layout in the host language, and it is checked
//     at boot.
//   - regex takes a GO pattern: RE2, without delimiters, without backreferences
//     and without lookaround. It is compiled at boot.
//   - `unique` and `exists` take an auth Grant and a PresenceVerifier, given
//     with WithPresence, and fail closed without one. A read is authorized like
//     any other: "the validator only counts rows" is how a count of rows becomes
//     a way to ask whether another tenant has a user with a given address.
//   - `current_password` takes a CurrentPasswordChecker. Nothing here resolves a
//     guard or a hasher, so without a checker the rule fails closed.
//
// # Reading what passed
//
// There is no reflection and there are no struct tags. The rule set is data,
// the request struct is written by hand or generated, and the values that
// passed are read out of Input one at a time -- so a field nobody declared a
// rule for cannot reach a repository through here.
package validation
