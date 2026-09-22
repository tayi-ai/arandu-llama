package validation

import (
	"context"
	"net"
	"regexp"
	"slices"
	"strings"
	"time"

	"github.com/arandu-io/hesape/auth"
)

// Validator is one submitted request, the rules written against it, and the
// answer.
//
// It is built by Make and it is not safe for concurrent use: one belongs to one
// request. The compiled Set behind it is read-only and is shared by all of them.
//
// The four rules that leave the process -- `unique`, `exists`,
// `current_password` and `active_url` -- take what they need through an option,
// and each of them fails closed when the thing it needs was not given.
type Validator struct {
	// initialRules is the compiled set as it was written, wildcards and all.
	// set is the same set with every wildcard expanded against the data this
	// request carries.
	//
	// Two of them, because the expansion depends on the data: adding a rule or
	// replacing the data has to expand again, and expanding an
	// already-expanded set would find nothing to expand.
	initialRules *Set
	set          *Set

	data Data

	messages Errors
	failed   map[string][]string
	excluded []string

	stopOnFirstFailure bool

	// currentRule is the rule being run, which is how ValidateRegex reads the
	// pattern compiled at boot instead of compiling it again on every
	// request.
	currentRule *rule

	ctx       context.Context
	grant     auth.Grant
	presence  PresenceVerifier
	passwords CurrentPasswordChecker
	dns       Resolver
	now       func() time.Time

	// What follows decides which sentence a failure carries, and how the field
	// and the value are spelled inside it. The two a rule object reads back are
	// CustomMessages and CustomAttributes; the rest is set through the With
	// options.
	//
	// trans is the catalogue. A nil one leaves the short sentences the compiled
	// rule set carries: see getMessage.
	trans                       Translator
	customMessages              map[string]any
	fallbackMessages            map[string]any
	customAttributes            map[string]string
	customValues                map[string]map[string]string
	replacers                   map[string]ReplacerFunc
	implicitAttributes          map[string][]string
	implicitAttributesFormatter func(string) string

	// after is the callbacks After registers.
	after []func()

	// ensureExponentWithinAllowedRange is the check
	// EnsureExponentWithinAllowedRangeUsing registered.
	ensureExponentWithinAllowedRange func(scale int, attribute string, value any) bool

	// exception is what SetException registered: the thing that builds the error
	// a failure is turned into.
	exception func(*Validator) error
}

var defaultResolver Resolver = net.DefaultResolver

// ValidatorOption is what a request hands a Validator that a boot-time rule set
// cannot hold: the context it runs under, the Grant that authorizes its
// queries, and the collaborators of the rules that leave the process.
type ValidatorOption func(*Validator)

// WithContext gives the Validator the request's context, so that a lookup or a
// count query carries the request's deadline. Without it the context is
// Background, and a rule that leaves the process has no deadline at all.
func WithContext(ctx context.Context) ValidatorOption {
	return func(v *Validator) { v.ctx = ctx }
}

// WithPresence gives `unique` and `exists` the Grant and the verifier they
// need.
//
// The Grant is not optional and there is no way to pass a verifier without one:
// a read is authorized like any other, and "the validator only counts rows" is
// how a count of rows becomes a way to ask whether another tenant has a user
// with a given email.
func WithPresence(g auth.Grant, p PresenceVerifier) ValidatorOption {
	return func(v *Validator) { v.grant, v.presence = g, p }
}

// WithCurrentPassword gives `current_password` its checker.
func WithCurrentPassword(c CurrentPasswordChecker) ValidatorOption {
	return func(v *Validator) { v.passwords = c }
}

// WithResolver gives `active_url` and `email:dns` the resolver to ask. The
// default is net.DefaultResolver; a test gives its own and makes no network
// call at all.
func WithResolver(r Resolver) ValidatorOption {
	return func(v *Validator) { v.dns = r }
}

// WithClock sets the clock the relative date keywords read, so that a test of
// `after:today` does not have to be run before midnight.
func WithClock(now func() time.Time) ValidatorOption {
	return func(v *Validator) { v.now = now }
}

// Make returns a Validator over the data and the rules.
//
// The rules are a compiled Set rather than an array of strings, because the
// strings are parsed and checked at boot -- see MustCompile. The data is
// copied, so that excluding a field does not reach back into the request's own
// map.
func Make(data Data, rules *Set, opts ...ValidatorOption) *Validator {
	v := &Validator{data: data.Clone(), failed: map[string][]string{}}
	for _, opt := range opts {
		opt(v)
	}
	v.explodeRules(rules)
	return v
}

// explodeRules turns "items.*.price" into one rule per item the request actually
// sent.
//
// A set with no wildcard in it is kept as it is, pointer and all: it is shared
// by every request, and a set nobody has to expand must not be copied per
// request either.
func (v *Validator) explodeRules(rules *Set) {
	v.initialRules = rules
	v.implicitAttributes = map[string][]string{}

	if !slices.ContainsFunc(rules.fields, func(f *field) bool { return strings.Contains(f.name, "*") }) {
		v.set = rules
		return
	}

	parser := NewValidationRuleParser(v.data)

	expanded := &Set{
		byName:   make(map[string]*field, len(rules.byName)),
		messages: rules.messages,
		file:     rules.file,
		line:     rules.line,
	}
	for _, f := range rules.fields {
		if !strings.Contains(f.name, "*") {
			expanded.fields = append(expanded.fields, f)
			expanded.byName[f.name] = f
			continue
		}
		for _, key := range parser.explodeWildcardRules(f.name) {
			if _, already := expanded.byName[key]; already {
				continue
			}
			member := *f
			member.name, member.primary = key, f.name
			expanded.fields = append(expanded.fields, &member)
			expanded.byName[key] = &member
			v.implicitAttributes[f.name] = append(v.implicitAttributes[f.name], key)
		}
	}
	v.set = expanded
}

// dependentRules are the rules whose parameters name OTHER FIELDS rather than
// values, and so the rules whose asterisks have to be replaced with the keys the
// attribute expanded to.
//
// Getting this list wrong fails open, which is why it is written out rather
// than inferred: every rule that decides whether a field is required is on it,
// and one that is missing goes back to being handed the literal "foo.*.baz" --
// a name no request can hold, so the rule finds nothing and passes.
var dependentRules = []string{
	"accepted_if", "after", "after_or_equal", "before", "before_or_equal",
	"confirmed", "declined_if", "different", "exclude_if", "exclude_unless",
	"exclude_with", "exclude_without", "gt", "gte", "lt", "lte", "missing_if",
	"missing_unless", "missing_with", "missing_with_all", "present_if",
	"present_unless", "present_with", "present_with_all", "prohibited",
	"prohibited_if", "prohibited_if_accepted", "prohibited_if_declined",
	"prohibited_unless", "prohibits", "required_if", "required_if_accepted",
	"required_if_declined", "required_unless", "required_with",
	"required_with_all", "required_without", "required_without_all", "same",
	"unique",
}

// uploadedFileRules is the six rules that make a field an upload PLUS the four
// size rules. The list in compile.go is the first six on their own, because that
// is what makes `max:100` mean a hundred kilobytes; this one is what
// validateAttribute asks before it reports a failed upload.
var uploadedFileRules = []string{
	"between", "dimensions", "extensions", "file", "image", "max", "mimes",
	"mimetypes", "min", "size",
}

// StopOnFirstFailure leaves after the first field that fails, rather than
// reporting every field. It returns the Validator, so calls chain.
func (v *Validator) StopOnFirstFailure() *Validator {
	v.stopOnFirstFailure = true
	return v
}

// Passes runs every rule and reports whether nothing failed.
func (v *Validator) Passes() bool {
	v.messages = Errors{}
	v.failed = map[string][]string{}
	v.excluded = nil

	for _, f := range v.set.fields {
		if v.ShouldBeExcluded(f.name) {
			v.data.Forget(f.name)
			continue
		}
		if v.stopOnFirstFailure && v.messages.IsNotEmpty() {
			break
		}
		for _, r := range f.rules {
			v.validateAttribute(f, r)

			if v.ShouldBeExcluded(f.name) {
				break
			}
			if v.ShouldStopValidating(f.name) {
				break
			}
		}
	}
	for _, f := range v.set.fields {
		if v.ShouldBeExcluded(f.name) {
			v.data.Forget(f.name)
		}
	}

	// Here we will spin through all of the "after" hooks on this validator and
	// fire them off. This gives the callbacks a chance to perform all kinds of
	// other validation that needs to get wrapped up in this operation.
	v.runAfterCallbacks()

	return v.messages.IsEmpty()
}

// Fails runs every rule and reports whether anything failed.
func (v *Validator) Fails() bool { return !v.Passes() }

// validateAttribute runs one rule against one field.
func (v *Validator) validateAttribute(f *field, r *rule) {
	// First we will get the correct keys for the given attribute in case the
	// field is nested in an array. Then we determine if the given rule accepts
	// other field names as parameters. If so, we will replace any asterisks
	// found in the parameters with the correct keys.
	//
	// This is the step that was missing, and it failed OPEN: the whole of
	// $dependentRules -- required_with, required_if, same, different, the four
	// comparisons, confirmed, prohibits, unique -- was handed the literal
	// parameter "foo.*.baz", which names nothing a request can hold. The lookup
	// found nothing, "required when the sibling is present" found no sibling,
	// and a form written with a wildcard accepted what it was written to
	// refuse.
	parameters := r.args
	if slices.Contains(dependentRules, r.name) {
		if keys := v.getExplicitKeys(f.name); len(keys) > 0 {
			parameters = replaceAsterisksInParameters(parameters, keys)
		}
	}

	value := v.GetValue(f.name)

	// If the attribute is a file, we will verify that the file upload was
	// actually successful and if it wasn't we will add a failure for the
	// attribute. Files may not successfully upload if they are too large based
	// on the server's settings so we will bail in this case.
	if isUploadedAndInvalid(value) &&
		(v.HasRule(f.name, uploadedFileRules) || v.HasRule(f.name, implicitRules)) {
		v.AddFailure(f.name, "uploaded", nil)
		return
	}

	if !v.isValidatable(f, r, value) {
		return
	}

	v.currentRule = r
	passed := r.spec.eval(v, f.name, value, parameters)
	v.currentRule = nil

	if !passed {
		v.AddFailure(f.name, r.name, parameters)
	}
}

// isUploadedAndInvalid reports an upload that did not finish.
func isUploadedAndInvalid(value any) bool {
	up, isUpload := value.(UploadedFile)
	return isUpload && !up.IsValid()
}

// getExplicitKeys returns the keys an expanded attribute filled its wildcards
// with: "foo.1.bar.spark.baz" gives [1, spark] for "foo.*.bar.*.baz".
func (v *Validator) getExplicitKeys(attribute string) []string {
	primary := v.getPrimaryAttribute(attribute)
	if !strings.Contains(primary, "*") {
		return nil
	}

	pattern, err := regexp.Compile(`^` + strings.ReplaceAll(regexp.QuoteMeta(primary), `\*`, `([^.]+)`))
	if err != nil {
		return nil
	}
	if keys := pattern.FindStringSubmatch(attribute); keys != nil {
		return keys[1:]
	}
	return nil
}

// replaceAsterisksInParameters fills each asterisk of a parameter with the
// matching key.
//
// With fewer keys than asterisks there is nothing sensible to guess, so the
// asterisks past the last key are left as they were written -- the rule then
// finds no field, which is the answer for a rule set asking for a key the
// attribute does not have.
func replaceAsterisksInParameters(parameters, keys []string) []string {
	out := make([]string, len(parameters))
	for i, field := range parameters {
		out[i] = fillAsterisks(field, keys)
	}
	return out
}

func fillAsterisks(field string, keys []string) string {
	parts := strings.Split(field, "*")
	if len(parts) == 1 {
		return field
	}

	var b strings.Builder
	b.WriteString(parts[0])
	for i, part := range parts[1:] {
		if i < len(keys) {
			b.WriteString(keys[i])
		} else {
			b.WriteString("*")
		}
		b.WriteString(part)
	}
	return b.String()
}

// isValidatable reports whether this rule runs against this value at all.
func (v *Validator) isValidatable(f *field, r *rule, value any) bool {
	if slices.Contains(excludeRules, r.name) {
		return true
	}
	return v.presentOrRuleIsImplicit(r, f.name, value) &&
		v.passesOptionalCheck(f) &&
		v.isNotNullIfMarkedAsNullable(f, r) &&
		v.hasNotFailedPreviousRuleIfPresenceRule(r, f.name)
}

// presentOrRuleIsImplicit reports whether the value is there, or the rule runs
// on a blank one.
//
// This is what keeps `min:12` on an optional box nobody typed in from saying
// "must be at least 12 characters" about a box the person deliberately left
// alone. Dropping it is a real bug, not a simplification.
func (v *Validator) presentOrRuleIsImplicit(r *rule, attribute string, value any) bool {
	if s, ok := asString(value); ok && strings.TrimSpace(s) == "" {
		return r.spec.implicit
	}
	return v.ValidatePresent(attribute, value, nil) || r.spec.implicit
}

// passesOptionalCheck reports whether `sometimes` lets the field run: it skips
// the field entirely when its key was not sent at all.
//
// It is why the data can tell absent from present-and-empty, and that difference
// is what a PATCH is made of.
func (v *Validator) passesOptionalCheck(f *field) bool {
	if !f.sometimes {
		return true
	}
	return v.data.Has(f.name)
}

// isNotNullIfMarkedAsNullable reports whether a nullable field holds something.
//
// `nullable` stops the chain when the value is NULL, not when it is empty:
// those are two different answers, and only one of them is "the client said
// there is nothing here".
func (v *Validator) isNotNullIfMarkedAsNullable(f *field, r *rule) bool {
	if r.spec.implicit || !f.nullable {
		return true
	}
	return v.GetValue(f.name) != nil
}

// hasNotFailedPreviousRuleIfPresenceRule keeps a field that already failed from
// going to the database to fail again.
func (v *Validator) hasNotFailedPreviousRuleIfPresenceRule(r *rule, attribute string) bool {
	if r.name == "unique" || r.name == "exists" {
		return !v.messages.Has(attribute)
	}
	return true
}

// ShouldStopValidating reports whether the field is finished, whatever rules it
// has left.
//
// `bail` stops the field at its first failure. A failed implicit rule stops it
// too, without being asked: `required` failing must not also produce "must be
// at least 12 characters" about the same empty box.
func (v *Validator) ShouldStopValidating(attribute string) bool {
	f, declared := v.set.byName[attribute]
	if !declared {
		return false
	}
	if f.bail {
		return v.messages.Has(attribute)
	}
	if !f.hasImplicit {
		return false
	}
	for _, name := range v.failed[attribute] {
		if slices.Contains(implicitRules, name) {
			return true
		}
	}
	return false
}

// AddFailure records that one rule refused one attribute.
//
// An exclude rule never puts a message on the field: its failure removes the
// field from the validated data instead, which is the whole of what the five
// exclude rules do.
func (v *Validator) AddFailure(attribute, rule string, parameters []string) {
	if v.messages == nil {
		v.messages = Errors{}
	}
	if slices.Contains(excludeRules, rule) {
		v.ExcludeAttribute(attribute)
		return
	}
	v.messages.Add(attribute, v.MakeReplacements(
		v.getMessage(attribute, rule), attribute, rule, parameters,
	))
	v.failed[attribute] = append(v.failed[attribute], rule)
}

// ExcludeAttribute drops the attribute from the validated data.
func (v *Validator) ExcludeAttribute(attribute string) {
	if !slices.Contains(v.excluded, attribute) {
		v.excluded = append(v.excluded, attribute)
	}
}

// ShouldBeExcluded reports whether the attribute, or a parent of it, was
// excluded.
func (v *Validator) ShouldBeExcluded(attribute string) bool {
	for _, excluded := range v.excluded {
		if attribute == excluded || strings.HasPrefix(attribute, excluded+".") {
			return true
		}
	}
	return false
}

// raisedRules are the sentences for the failures the validator reports itself,
// which no field declares and so no spec carries.
//
// `uploaded` is the only one, and it had no reachable sentence at all: the
// English line for it exists in lang.go and nothing ever raised the failure,
// because validateAttribute had no branch for an upload that did not finish.
// It is not in specs on purpose -- specs is what a rule string may name, and
// nobody writes `uploaded` into one.
var raisedRules = map[string]string{"uploaded": "failed to upload"}

// messageFor is the sentence one failure puts on the field, after any override
// the set carries.
func (v *Validator) messageFor(attribute, ruleName string) string {
	f, declared := v.set.byName[attribute]
	if !declared {
		return "is not valid"
	}
	if r := f.rule(ruleName); r != nil {
		return v.set.message(f, r)
	}
	if sentence, raised := raisedRules[ruleName]; raised {
		return sentence
	}
	return "is not valid"
}

// GetValue returns what the request holds at the attribute.
func (v *Validator) GetValue(attribute string) any { return v.data.Get(attribute) }

// SetValue writes a value at the attribute.
func (v *Validator) SetValue(attribute string, value any) { v.data[attribute] = value }

// GetData returns the request the rules run against.
func (v *Validator) GetData() Data { return v.data }

// SetData replaces the request the rules run against. The data is copied, for
// the reason Make copies it.
func (v *Validator) SetData(data Data) *Validator {
	v.data = data.Clone()
	// The wildcards were expanded against the OLD data, and "items.*.price"
	// means the items this request sent.
	v.explodeRules(v.initialRules)
	return v
}

// Attributes returns the data the rules run against.
func (v *Validator) Attributes() Data { return v.data }

// GetRules returns the rule names written against each field.
func (v *Validator) GetRules() map[string][]string {
	out := make(map[string][]string, len(v.set.fields))
	for _, f := range v.set.fields {
		out[f.name] = v.set.RulesFor(f.name)
	}
	return out
}

// SetRules replaces the compiled set this runs.
func (v *Validator) SetRules(rules *Set) *Validator {
	v.explodeRules(rules)
	return v
}

// HasRule reports whether the field declares any of the named rules.
func (v *Validator) HasRule(attribute string, rules []string) bool {
	f, declared := v.set.byName[attribute]
	if !declared {
		return false
	}
	for _, name := range rules {
		if f.rule(name) != nil {
			return true
		}
	}
	return false
}

// Errors returns the messages a run collected, running first if it has not
// been.
func (v *Validator) Errors() Errors {
	if v.messages == nil {
		v.Passes()
	}
	return v.messages
}

// Messages returns the same messages Errors does.
func (v *Validator) Messages() Errors { return v.Errors() }

// Failed returns the rules that failed, per attribute.
func (v *Validator) Failed() map[string][]string {
	if v.messages == nil {
		v.Passes()
	}
	return v.failed
}

// Validated returns the values that were declared, sent and not rejected. It is
// the only way to read a submitted value out of a validated
// request, so a field nobody wrote a rule for cannot reach a repository through
// here.
func (v *Validator) Validated() Input {
	if v.messages == nil {
		v.Passes()
	}
	out := make(Data, len(v.set.fields))
	for _, f := range v.set.fields {
		if v.messages.Has(f.name) || v.ShouldBeExcluded(f.name) {
			continue
		}
		if value, sent := lookup(v.data, f.name); sent {
			out[f.name] = value
		}
	}
	return Input{data: out}
}

// Valid returns the data of every attribute that has no message.
func (v *Validator) Valid() Data {
	if v.messages == nil {
		v.Passes()
	}
	out := make(Data, len(v.data))
	for name, value := range v.data {
		if !v.messages.Has(name) {
			out[name] = value
		}
	}
	return out
}

// Invalid returns the data of every attribute that has a message.
func (v *Validator) Invalid() Data {
	if v.messages == nil {
		v.Passes()
	}
	out := make(Data, len(v.messages))
	for name, value := range v.data {
		if v.messages.Has(name) {
			out[name] = value
		}
	}
	return out
}
