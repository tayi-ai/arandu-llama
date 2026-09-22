package validation

import (
	"errors"
	"maps"
	"slices"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/str"
	"github.com/arandu-io/hesape/support"
)

// This file is the rest of the Validator: the entry points a controller calls,
// the two ways a rule set grows after the validator was built, and how a rule of
// one's own is registered.

// Validate returns the attributes that were validated, or the error the failures
// make.
//
// The error is always a *ValidationException, so errors.As reaches the bag, the
// status and the redirect.
func (v *Validator) Validate() (Input, error) {
	if v.Fails() {
		return Input{}, v.exceptionFor()
	}

	return v.Validated(), nil
}

// ValidateWithBag is Validate with the failures named, so that two forms on one
// page do not draw each other's errors.
func (v *Validator) ValidateWithBag(errorBag string) (Input, error) {
	validated, err := v.Validate()
	if err != nil {
		var invalid *ValidationException
		if errors.As(err, &invalid) {
			invalid.ErrorBag(errorBag)
		}
		return Input{}, err
	}
	return validated, nil
}

// Safe returns the validated attributes in a container that reads them one at a
// time, narrowed to the named keys when any are given.
//
// It is a ValidatedInput either way, so a caller has one type to write against,
// and ValidatedInput.All answers with the map.
func (v *Validator) Safe(keys ...string) (*support.ValidatedInput, error) {
	if v.Fails() {
		return nil, v.exceptionFor()
	}

	input := support.NewValidatedInput(v.Validated().data)
	if len(keys) == 0 {
		return input, nil
	}
	return support.NewValidatedInput(input.All(keys...)), nil
}

// exceptionFor builds what a failure is turned into, which SetException
// replaces.
func (v *Validator) exceptionFor() error {
	if v.exception == nil {
		return NewValidationException(v)
	}
	return v.exception(v)
}

// GetException returns what a failure is turned into: the function that builds
// the error.
func (v *Validator) GetException() func(*Validator) error {
	if v.exception == nil {
		return func(v *Validator) error { return NewValidationException(v) }
	}
	return v.exception
}

// SetException sets what a failure is turned into. The signature settles the
// shape, so only a nil factory is left to reject.
func (v *Validator) SetException(exception func(*Validator) error) *Validator {
	if exception == nil {
		panic("validation: the exception factory must not be nil -- it is what a failure is turned into")
	}
	v.exception = exception

	return v
}

// After registers a callback that runs once every rule has.
//
// It is where a check that needs the whole form goes, and it is the seam a rule
// object is run through: the callback holds the Validator, so it calls
// AddFailure with the attribute and rule name of its choosing.
func (v *Validator) After(callback func(*Validator)) *Validator {
	v.after = append(v.after, func() { callback(v) })

	return v
}

// runAfterCallbacks fires the After hooks. Passes calls it once every field has
// run.
func (v *Validator) runAfterCallbacks() {
	for _, after := range v.after {
		after()
	}
}

// AddRules merges another compiled set into this validator's, so that a rule
// decided at request time joins the ones decided at boot.
//
// What is merged is a Set, already parsed and checked by MustCompile. A field
// both sets declare keeps the rules of both, in the order this one wrote them
// first.
func (v *Validator) AddRules(rules *Set) *Validator {
	if rules == nil {
		return v
	}
	v.explodeRules(mergeSets(v.initialRules, rules))

	return v
}

// Sometimes adds rules to an attribute only when the callback says so.
//
// The callback is handed the whole request, as a Fluent, and the value of the
// attribute being decided.
func (v *Validator) Sometimes(attributes []string, rules *Set, callback func(payload *support.Fluent, value any) bool) *Validator {
	if rules == nil {
		return v
	}

	payload := support.NewFluent(v.data)

	for _, attribute := range attributes {
		f, declared := rules.byName[attribute]
		if !declared {
			continue
		}
		if !callback(payload, v.GetValue(attribute)) {
			continue
		}
		v.explodeRules(mergeSets(v.initialRules, &Set{
			fields:   []*field{f},
			byName:   map[string]*field{attribute: f},
			messages: rules.messages,
		}))
	}

	return v
}

// mergeSets is the union of two compiled sets, with neither changed: a Validator
// that grows its rules must not reach back into the Set every other request
// shares.
func mergeSets(base, extra *Set) *Set {
	merged := &Set{
		byName:   make(map[string]*field, len(base.byName)+len(extra.byName)),
		messages: Messages{},
		file:     base.file,
		line:     base.line,
	}
	maps.Copy(merged.messages, base.messages)
	maps.Copy(merged.messages, extra.messages)

	for _, f := range base.fields {
		clone := *f
		clone.rules = slices.Clone(f.rules)
		merged.fields = append(merged.fields, &clone)
		merged.byName[clone.name] = &clone
	}

	for _, f := range extra.fields {
		existing, declared := merged.byName[f.name]
		if !declared {
			clone := *f
			clone.rules = slices.Clone(f.rules)
			merged.fields = append(merged.fields, &clone)
			merged.byName[clone.name] = &clone
			continue
		}
		for _, r := range f.rules {
			if existing.rule(r.name) == nil {
				existing.rules = append(existing.rules, r)
			}
		}
		existing.bail = existing.bail || f.bail
		existing.sometimes = existing.sometimes || f.sometimes
		existing.nullable = existing.nullable || f.nullable
		existing.numeric = existing.numeric || f.numeric
		existing.file = existing.file || f.file
		existing.hasImplicit = existing.hasImplicit || f.hasImplicit
		if existing.layout == "" {
			existing.layout = f.layout
		}
	}

	return merged
}

// GetRule returns the first of the given rules written against the attribute,
// and the parameters it carries. The third value reports whether one was
// found.
func (v *Validator) GetRule(attribute string, rules []string) (string, []string, bool) {
	f, declared := v.set.byName[attribute]
	if !declared {
		return "", nil, false
	}
	for _, name := range rules {
		if r := f.rule(name); r != nil {
			return r.name, r.args, true
		}
	}
	return "", nil, false
}

// GetMessageBag returns the messages a run collected, which is what Errors
// returns.
func (v *Validator) GetMessageBag() Errors { return v.Errors() }

// GetPresenceVerifier returns the verifier `unique` and `exists` count through,
// and an error when none was set.
func (v *Validator) GetPresenceVerifier() (PresenceVerifier, error) {
	if v.presence == nil {
		return nil, errors.New("validation: presence verifier has not been set")
	}
	return v.presence, nil
}

// SetPresenceVerifier gives `unique` and `exists` the Grant and the verifier they
// need.
//
// There is no way to give a verifier without a Grant: a read is authorized like
// any other. WithPresence is the same pair as an option.
func (v *Validator) SetPresenceVerifier(g auth.Grant, presenceVerifier PresenceVerifier) {
	v.grant, v.presence = g, presenceVerifier
}

// ExtensionFunc is a rule written by the application, called with the arguments
// every rule in the catalogue is.
type ExtensionFunc func(v *Validator, attribute string, value any, parameters []string) bool

// AddExtension registers a rule of the application's own.
//
// A rule name has to exist before MustCompile reads it, so an extension joins
// the one catalogue every rule set is checked against. Registering the same name
// twice panics, for the reason two answers to one rule name always do.
//
// The catalogue is package level rather than per-validator: an extension is
// registered once, at start-up, before any validator is made.
func (v *Validator) AddExtension(rule string, extension ExtensionFunc) {
	Extend(rule, extension, "")
}

// AddExtensions registers several rules of the application's own at once.
func (v *Validator) AddExtensions(extensions map[string]ExtensionFunc) {
	for rule, extension := range extensions {
		v.AddExtension(rule, extension)
	}
}

// AddImplicitExtension registers a rule of the application's own that runs even
// when the value is blank.
func (v *Validator) AddImplicitExtension(rule string, extension ExtensionFunc) {
	ExtendImplicit(rule, extension, "")
}

// AddImplicitExtensions registers several implicit rules at once.
func (v *Validator) AddImplicitExtensions(extensions map[string]ExtensionFunc) {
	for rule, extension := range extensions {
		v.AddImplicitExtension(rule, extension)
	}
}

// AddDependentExtension registers a rule of the application's own whose first
// parameter names another field, which is then checked at boot like every other
// cross-field reference.
func (v *Validator) AddDependentExtension(rule string, extension ExtensionFunc) {
	ExtendDependent(rule, extension, "")
}

// AddDependentExtensions registers several dependent rules at once.
func (v *Validator) AddDependentExtensions(extensions map[string]ExtensionFunc) {
	for rule, extension := range extensions {
		v.AddDependentExtension(rule, extension)
	}
}

// Extend registers a rule of the application's own: a rule name, the check
// behind it, and the sentence it says.
//
// It is called from an init or from main, before any rule set is compiled. An
// empty message leaves the rule saying "is not valid".
func Extend(rule string, extension ExtensionFunc, message string) {
	registerExtension(rule, extension, message, false, nil)
}

// ExtendImplicit is Extend for a rule that runs even when the value is blank.
func ExtendImplicit(rule string, extension ExtensionFunc, message string) {
	registerExtension(rule, extension, message, true, nil)
}

// ExtendDependent is Extend for a rule whose first parameter names another
// field.
func ExtendDependent(rule string, extension ExtensionFunc, message string) {
	registerExtension(rule, extension, message, false, []int{0})
}

func registerExtension(name string, extension ExtensionFunc, message string, implicit bool, refs []int) {
	name = str.Snake(name, "_")

	if extension == nil {
		panic("validation: rule " + name + " was registered with no check")
	}
	if _, taken := specs[name]; taken {
		panic("validation: rule " + name + " is already in the catalogue -- " +
			"two answers to one rule name is what this refuses to let boot")
	}
	if _, deliberate := refused[name]; deliberate {
		panic("validation: rule " + name + " is refused on purpose -- see refused.go")
	}

	if message == "" {
		message = "is not valid"
	}

	specs[name] = &spec{
		maxArgs:  -1,
		implicit: implicit,
		refs:     refs,
		eval:     evaluator(extension),
		message:  func(f *field, r *rule) string { return message },
	}

	if implicit {
		implicitRules = append(implicitRules, name)
	}
}

// ParseData returns the request as the Validator will read it, which is a copy.
//
// Nothing has to be escaped first: lookup tries the literal key before it walks
// a dotted path, so an input genuinely named "a.b" is found as itself. What is
// left is the copy, which keeps a Validator from writing through to the
// request's own map.
func (v *Validator) ParseData(data Data) Data { return data.Clone() }

// GetRulesWithoutPlaceholders returns the same map GetRules does, for the reason
// ParseData is only a copy: no key was ever escaped, so there is nothing to take
// back out.
func (v *Validator) GetRulesWithoutPlaceholders() map[string][]string { return v.GetRules() }

// EnsureExponentWithinAllowedRangeUsing sets the check `numeric` and `decimal`
// make before they read a number written in exponent notation, so that
// "1e100000" is refused rather than turned into a float nobody meant.
func (v *Validator) EnsureExponentWithinAllowedRangeUsing(callback func(scale int, attribute string, value any) bool) *Validator {
	v.ensureExponentWithinAllowedRange = callback

	return v
}

// EnsureExponentWithinAllowedRange asks that check.
//
// With nothing registered the answer is the default range -- an exponent between
// -1000 and 1000 -- and never a plain yes: a range check that accepts every
// scale is no range check.
func (v *Validator) EnsureExponentWithinAllowedRange(scale int, attribute string, value any) bool {
	if v.ensureExponentWithinAllowedRange == nil {
		return scale <= 1000 && scale >= -1000
	}
	return v.ensureExponentWithinAllowedRange(scale, attribute, value)
}
