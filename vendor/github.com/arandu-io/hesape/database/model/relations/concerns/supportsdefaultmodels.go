package concerns

// SupportsDefaultModels answers the trait of the same name: the "give me an
// empty model rather than null" option on the relations that return one model.
//
// The PHP's $withDefault holds a bool, an array or a Closure, and reads which
// one it got at call time. The three arrive here as three fields, because a
// single `any` would move the same test from the compiler to run time without
// buying anything back.
type SupportsDefaultModels struct {
	// NewRelatedInstanceFor answers the trait's abstract method of the same
	// name. The relation sets it when it builds itself.
	NewRelatedInstanceFor func(parent Model) Model

	enabled    bool
	attributes map[string]any
	callback   func(instance, parent Model)
}

// WithDefault answers SupportsDefaultModels::withDefault with no argument,
// which is the PHP's default of true.
func (s *SupportsDefaultModels) WithDefault() *SupportsDefaultModels {
	s.enabled = true
	s.attributes = nil
	s.callback = nil
	return s
}

// WithDefaultAttributes answers withDefault($array): the default model is
// force-filled with these.
//
// The PHP spells all three of these `withDefault`, because it can read the
// argument's type at run time. The suffix here is a mechanical split of one
// overloaded name, not a new name.
func (s *SupportsDefaultModels) WithDefaultAttributes(attributes map[string]any) *SupportsDefaultModels {
	s.enabled = true
	s.attributes = attributes
	s.callback = nil
	return s
}

// WithDefaultCallback answers withDefault(Closure): the callback is handed the
// new instance and the parent.
func (s *SupportsDefaultModels) WithDefaultCallback(callback func(instance, parent Model)) *SupportsDefaultModels {
	s.enabled = true
	s.attributes = nil
	s.callback = callback
	return s
}

// WithoutDefault turns the option back off. The PHP writes withDefault(false).
func (s *SupportsDefaultModels) WithoutDefault() *SupportsDefaultModels {
	s.enabled = false
	s.attributes = nil
	s.callback = nil
	return s
}

// GetDefaultFor answers SupportsDefaultModels::getDefaultFor.
//
// nil when the option was never asked for, which is what makes the caller's
// `$this->query->first() ?: $this->getDefaultFor(...)` read the same here.
func (s *SupportsDefaultModels) GetDefaultFor(parent Model) Model {
	if !s.enabled || s.NewRelatedInstanceFor == nil {
		return nil
	}

	instance := s.NewRelatedInstanceFor(parent)
	if instance == nil {
		return nil
	}

	switch {
	case s.callback != nil:
		s.callback(instance, parent)
	case s.attributes != nil:
		for key, value := range s.attributes {
			instance.SetAttribute(key, value)
		}
	}

	return instance
}
