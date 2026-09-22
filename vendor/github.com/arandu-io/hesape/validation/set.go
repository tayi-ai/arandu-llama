package validation

import (
	"net/url"
	"regexp"
	"slices"
	"time"
)

// Set is a compiled, checked rule set.
//
// Build one in a package-level variable with MustCompile: the rules are then
// parsed once, at boot, and a set that boots is a set whose names are all real.
// A Set is read-only once compiled, so one is shared by every request; the
// per-request state lives in the Validator that Make builds over it.
type Set struct {
	fields   []*field // sorted by name, so failures come out in a stable order
	byName   map[string]*field
	messages Messages
	file     string
	line     int
}

// field is one input and the rules written against it, in the order written.
type field struct {
	name  string
	rules []*rule

	// primary is the wildcard name this field was expanded from --
	// "items.*.price" for "items.0.price" -- and empty for a field written
	// without one. It is what lets a message override still be keyed on the
	// name somebody wrote.
	primary string

	// bail stops the field at its first failure.
	bail bool
	// sometimes skips the field entirely when its key was not sent at all --
	// which is why the data can tell "absent" from "present and empty", and
	// that is the difference a PATCH is made of.
	sometimes bool
	// nullable stops the field when the value is NULL rather than when it is
	// empty.
	nullable bool
	// numeric is set by numeric, integer or decimal, and makes every size rule
	// on this field measure the value rather than the characters.
	numeric bool
	// file is set by file, image, mimes, mimetypes, extensions or dimensions,
	// and makes a size limit on this field mean KILOBYTES.
	file bool
	// hasImplicit records that the field declares at least one implicit rule,
	// which is what shouldStopValidating asks before it stops.
	hasImplicit bool
	// layout is the date_format layout, which the date comparisons read.
	layout string
}

// rule is one rule of one field, with whatever its boot check parsed for it.
type rule struct {
	name string
	args []string
	spec *spec

	// re is the compiled pattern of regex/not_regex, and nums the parsed
	// numeric arguments. Both are filled at boot, so a request parses nothing.
	re   *regexp.Regexp
	nums []float64
}

func (f *field) rule(name string) *rule {
	for _, r := range f.rules {
		if r.name == name {
			return r
		}
	}
	return nil
}

// Fields returns the input names this set declares, sorted.
func (s *Set) Fields() []string {
	names := make([]string, len(s.fields))
	for i, f := range s.fields {
		names[i] = f.name
	}
	return names
}

// RulesFor returns the rule names written against one field, in the order they
// were written. It is nil for a field the set does not declare.
func (s *Set) RulesFor(field string) []string {
	f, ok := s.byName[field]
	if !ok {
		return nil
	}
	names := make([]string, len(f.rules))
	for i, r := range f.rules {
		names[i] = r.name
	}
	return names
}

// Source returns where the set was compiled, which is what a boot failure
// names.
func (s *Set) Source() (file string, line int) { return s.file, s.line }

// Validate runs the set over a submitted form and returns what passed and what
// failed.
//
// It is Make plus Passes for the common case: an HTML form, no upload, no rule
// that leaves the process. A set with `unique`, `exists`, `current_password` or
// `active_url` in it goes through Make instead, which is where the Grant, the
// verifier and the context are given -- those four fail closed here, on purpose:
// a read is authorized like any other, and a rule set carries no Grant.
//
// The input is url.Values rather than Data because that is what a form arrives
// as; DataFrom is the conversion, and a name sent twice becomes a list.
//
// The returned Errors is nil when nothing failed. Assigning it to an error
// interface makes that interface non-nil even so, because the type is not nil
// -- callers ask Any(), and http.Context.Validate returns a plain nil for
// exactly this reason.
func (s *Set) Validate(values url.Values) (Input, Errors) {
	v := Make(DataFrom(values), s)
	if v.Passes() {
		return v.Validated(), nil
	}
	return v.Validated(), v.Errors()
}

// message is the sentence one failure puts on the field, after any override.
//
// An expanded field answers to the name it was written under as well: the
// override is keyed "items.*.price.required" because that is what the rule set
// declares, and the failure lands on "items.0.price".
func (s *Set) message(f *field, r *rule) string {
	if custom, ok := s.messages[f.name+"."+r.name]; ok {
		return custom
	}
	if f.primary != "" {
		if custom, ok := s.messages[f.primary+"."+r.name]; ok {
			return custom
		}
	}
	return r.spec.message(f, r)
}

// Input is what passed the rules. It is the only way to read a submitted value
// out of a validated request: a field the set does not declare is not in here,
// so a value nobody wrote a rule for cannot reach a repository by accident.
type Input struct {
	data Data
}

// Has reports whether the field was sent and passed.
func (in Input) Has(field string) bool { return in.data.Has(field) }

// String returns the value, or empty when the field was not sent. A field sent
// more than once is a list, and this is its first value -- what url.Values.Get
// answers.
func (in Input) String(field string) string {
	value := in.data.Get(field)
	if list, ok := asList(value); ok {
		if len(list) == 0 {
			return ""
		}
		return stringOf(list[0])
	}
	return stringOf(value)
}

// Strings returns every value of a field, which is what a multi-select sends. A
// field sent once is a list of one.
func (in Input) Strings(field string) []string {
	value, sent := lookup(in.data, field)
	if !sent {
		return nil
	}
	if list, ok := asList(value); ok {
		out := make([]string, len(list))
		for i, item := range list {
			out[i] = stringOf(item)
		}
		return out
	}
	return []string{stringOf(value)}
}

// Int returns the value as a whole number. It is zero when the field was not
// sent or does not hold one -- which is a rule's job to have proven, with
// `integer`.
func (in Input) Int(field string) int64 {
	n, _ := whole(in.String(field))
	return n
}

// Float returns the value as a number, zero when there is none. `numeric` or
// `decimal` is what proves there is.
func (in Input) Float(field string) float64 {
	n, _ := number(in.String(field))
	return n
}

// Bool reads a checkbox.
//
// It accepts every value `accepted` does -- "1", "on", "yes", "true" -- and not
// only the "0" and "1" that the `boolean` rule allows, because an unticked
// checkbox sends nothing and a ticked one sends "on". Anything else is false: a
// checkbox has no third answer.
func (in Input) Bool(field string) bool { return isAccepted(in.String(field)) }

// Time reads a value with the layout its rule declared. It is the zero time
// when the field was not sent or does not parse, which `date_format` is what
// prevents.
func (in Input) Time(field, layout string) time.Time {
	t, err := time.Parse(layout, in.String(field))
	if err != nil {
		return time.Time{}
	}
	return t
}

// File returns the upload that passed, and false when the field holds none.
func (in Input) File(field string) (File, bool) { return asFile(in.data.Get(field)) }

// Data returns a copy of everything that passed, in the shape the rules read
// it. It is a copy so that a caller cannot reach back into the request's own
// values through it.
func (in Input) Data() Data { return in.data.Clone() }

// Values returns a copy of everything that passed, as a submitted form, for a
// caller that hands the whole thing on.
func (in Input) Values() url.Values { return in.data.Values() }

// anyAffix reports whether the value carries any of the affixes, with the given
// match -- the shape starts_with, ends_with and their two refusals share.
func anyAffix(list []string, match func(string, string) bool, v string) bool {
	return slices.ContainsFunc(list, func(affix string) bool { return match(v, affix) })
}
