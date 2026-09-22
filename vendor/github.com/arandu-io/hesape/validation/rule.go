package validation

// This file is the interfaces a rule object implements, and the two wrappers
// that carry one.
//
// A rule object is not a name in the catalogue and cannot be written into a rule
// string, so it is run through After:
//
//	v.After(func(v *validation.Validator) {
//		v.ValidateUsingCustomRule("password", v.GetValue("password"), rule)
//	})

// Rule is a check written as a type: it reports whether a value passes, and says
// what was wrong when it does not.
//
// Message returns a slice because both wrappers below return several: one per
// call to their fail function.
type Rule interface {
	// Passes reports whether the value satisfies the rule.
	Passes(attribute string, value any) bool
	// Message returns what the last Passes refused, one sentence per failure.
	Message() []string
}

// ValidationRule is a check that reports a problem by calling fail with the
// sentence to show, rather than by returning false. NewInvokableValidationRule
// wraps one so that it reads as a Rule.
type ValidationRule interface {
	// Validate checks the value, calling fail once per problem it finds.
	Validate(attribute string, value any, fail func(message string))
}

// ImplicitRule is a rule that runs even when the value is blank. The marker is
// the question itself, because an interface here needs a method.
type ImplicitRule interface {
	// Implicit reports whether this rule runs on a blank value.
	Implicit() bool
}

// DataAwareRule is implemented by a rule object that has to see the whole
// request and not only the one value it was written against -- a check that
// compares two fields, for instance. Before asking such a rule whether the value
// passes, the validator hands it every value it is validating. A rule that looks
// only at its own value does not implement it and is not given the data.
type DataAwareRule interface {
	// SetData hands the rule every value being validated.
	SetData(data Data)
}

// ValidatorAwareRule is implemented by a rule object that needs the validator
// running it -- to phrase its message through the same translator, to read the
// custom messages and attribute names in force, or to reach a value it was not
// handed. The validator passes itself in before asking whether the value passes,
// and the rule keeps it for the length of that run.
type ValidatorAwareRule interface {
	// SetValidator hands the rule the validator running it.
	SetValidator(validator *Validator)
}

// CompilableRules is rules decided by looking at the value, which is what
// NestedRules is.
type CompilableRules interface {
	// Compile returns the rules this attribute gets, given what it holds.
	Compile(attribute string, value any, data Data) *ExplodedRules
}

// ValidateUsingCustomRule runs one rule object and puts whatever it says on the
// attribute.
//
// A compiled Set holds rule names and not objects, so the only caller is an After
// hook -- which is why this is exported.
func (v *Validator) ValidateUsingCustomRule(attribute string, value any, rule Rule) {
	if aware, ok := rule.(ValidatorAwareRule); ok {
		aware.SetValidator(v)
	}

	if aware, ok := rule.(DataAwareRule); ok {
		aware.SetData(v.data)
	}

	if rule.Passes(attribute, value) {
		return
	}

	if v.messages == nil {
		v.messages = Errors{}
	}

	messages := rule.Message()
	if len(messages) == 0 {
		messages = []string{"is not valid"}
	}

	for _, message := range messages {
		v.messages.Add(attribute, v.MakeReplacements(message, attribute, "", nil))
	}
	v.failed[attribute] = append(v.failed[attribute], "custom")
}

// ---------------------------------------------------------------------------
// A rule written as a function.
// ---------------------------------------------------------------------------

// ClosureValidationRule is a rule written as a function rather than as a type,
// for a check that does not deserve a type of its own. The callback is handed
// the attribute, its value, a fail function and the validator, and reports a
// problem by calling fail with the sentence to show. Each call to fail adds one
// message, and a callback that never calls it passes.
type ClosureValidationRule struct {
	// Callback is the check itself.
	Callback func(attribute string, value any, fail func(message string), validator *Validator)

	// Failed reports whether the last Passes called fail at all.
	Failed bool

	// messages is what the last Passes collected: one per call to fail.
	messages []string

	validator *Validator
}

// NewClosureValidationRule returns a rule that runs the callback.
func NewClosureValidationRule(callback func(attribute string, value any, fail func(message string), validator *Validator)) *ClosureValidationRule {
	return &ClosureValidationRule{Callback: callback}
}

// Passes runs the callback, and reports whether it called fail. A nil callback
// passes.
func (r *ClosureValidationRule) Passes(attribute string, value any) bool {
	r.Failed = false
	r.messages = nil

	if r.Callback == nil {
		return true
	}

	r.Callback(attribute, value, func(message string) {
		r.Failed = true
		r.messages = append(r.messages, message)
	}, r.validator)

	return !r.Failed
}

// Message returns what the last Passes collected.
func (r *ClosureValidationRule) Message() []string { return r.messages }

// SetValidator hands the rule the validator running it, which the callback is
// given as its fourth argument.
func (r *ClosureValidationRule) SetValidator(validator *Validator) { r.validator = validator }

// ---------------------------------------------------------------------------
// A ValidationRule read as a Rule.
// ---------------------------------------------------------------------------

// InvokableValidationRule is a ValidationRule wrapped so that it reads as a Rule.
// Build one with NewInvokableValidationRule.
type InvokableValidationRule struct {
	invokable ValidationRule
	failed    bool
	messages  []string
	validator *Validator
	data      Data
	implicit  bool
}

// NewInvokableValidationRule wraps a ValidationRule as a Rule, carrying over
// whether the wrapped one is implicit.
//
// The name is New rather than Make because Make is already the Validator's
// constructor.
func NewInvokableValidationRule(invokable ValidationRule) *InvokableValidationRule {
	rule := &InvokableValidationRule{invokable: invokable}
	if marker, marked := invokable.(ImplicitRule); marked {
		rule.implicit = marker.Implicit()
	}
	return rule
}

// Passes runs the wrapped rule, handing it the data and the validator first when
// it asks for them, and reports whether it called fail. A nil rule passes.
func (r *InvokableValidationRule) Passes(attribute string, value any) bool {
	r.failed = false
	r.messages = nil

	if r.invokable == nil {
		return true
	}

	if aware, ok := r.invokable.(DataAwareRule); ok && r.validator != nil {
		aware.SetData(r.validator.GetData())
	}

	if aware, ok := r.invokable.(ValidatorAwareRule); ok && r.validator != nil {
		aware.SetValidator(r.validator)
	}

	r.invokable.Validate(attribute, value, func(message string) {
		r.failed = true
		r.messages = append(r.messages, message)
	})

	return !r.failed
}

// Invokable returns the rule this wraps.
func (r *InvokableValidationRule) Invokable() ValidationRule { return r.invokable }

// Message returns what the last Passes collected.
func (r *InvokableValidationRule) Message() []string { return r.messages }

// Implicit reports whether the wrapped rule runs on a blank value.
func (r *InvokableValidationRule) Implicit() bool { return r.implicit }

// SetData hands the wrapper every value being validated.
func (r *InvokableValidationRule) SetData(data Data) { r.data = data }

// SetValidator hands the wrapper the validator running it.
func (r *InvokableValidationRule) SetValidator(validator *Validator) { r.validator = validator }
