package validation

import (
	"fmt"
	"strings"
	"time"
)

// Validatable is implemented by every request type that checks itself.
type Validatable interface {
	Validate() Errors
}

// The helper list below is short on purpose: it is what a hand-written request
// type reaches for, and domain rules do not belong to the framework. Anything
// that has a rule name is a rule in the catalogue, not a function here.

// Required rejects an empty or blank value. It is the `required` rule for one
// value, without a rule set.
func Required(e Errors, field, value string) {
	if strings.TrimSpace(value) == "" {
		e.Add(field, "is required")
	}
}

// MinLen counts runes, not bytes: a limit measured in bytes rejects valid input
// in any language that needs more than one byte per character.
func MinLen(e Errors, field, value string, n int) {
	if len([]rune(value)) < n {
		e.Add(field, fmt.Sprintf("must be at least %d characters", n))
	}
}

// MaxLen counts runes, not bytes.
func MaxLen(e Errors, field, value string, n int) {
	if len([]rune(value)) > n {
		e.Add(field, fmt.Sprintf("must be at most %d characters", n))
	}
}

// Email checks the shape only. Deliverability is proven by sending mail, never
// by a regular expression.
//
// Whitespace is rejected rather than trimmed: an address with a space in it is
// almost always a paste accident, and silently trimming input hides the mistake
// from the person who made it.
//
// The shape itself is emailShape, which the `email` rule also calls: two
// implementations of "is this an address" would drift, and the one that drifted
// would be whichever a screen did not exercise.
func Email(e Errors, field, value string) {
	if !emailShape(value) {
		e.Add(field, "is not a valid email address")
	}
}

// NotZero reports a value that was never filled in.
//
// It is Required for everything that is not text. Required takes a string and
// the generator used to hand it a literal "" for an int, a date or an amount --
// so every required field of those types failed validation with "is required"
// no matter what was sent, and the generated create endpoint could not be used
// at all. Found by audit.
//
// A time.Time is asked rather than compared: a parsed "0001-01-01T00:00:00Z"
// carries a location the zero value does not, so == says they differ when they
// do not.
//
// Bool has no meaningful zero to reject -- false is an answer, not an absence --
// and the specification refuses `required` on a bool for that reason.
func NotZero[T comparable](e Errors, field string, value T) {
	if t, ok := any(value).(time.Time); ok {
		if t.IsZero() {
			e.Add(field, "is required")
		}
		return
	}
	var zero T
	if value == zero {
		e.Add(field, "is required")
	}
}

// Confirmed rejects a value its confirmation field does not repeat.
//
// It is what a "confirm your password" box is for, and the message goes on the
// confirmation rather than on the field itself: a form that reports "password
// does not match" next to the first box tells the person to change the one they
// meant, and they change it, and the form fails again.
//
// An empty confirmation is reported here rather than by Required, so the field
// gets one message instead of two saying the same thing.
func Confirmed(e Errors, field, value, confirmation string) {
	if value == "" {
		// Nothing to confirm yet. Whatever rule rejected the value itself has
		// already said so, and a second message about the copy is noise.
		return
	}
	if confirmation == "" {
		e.Add(field, "is required")
		return
	}
	if value != confirmation {
		e.Add(field, "does not match")
	}
}
