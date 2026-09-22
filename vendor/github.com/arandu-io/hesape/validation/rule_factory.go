package validation

import (
	"context"

	"github.com/arandu-io/hesape/auth"
)

// RuleFactory gathers the rule builders under one name, so that a rule set reads
// as a list of rules rather than as a list of constructors:
//
//	validation.Rules{
//		"role":  validation.RuleFactory{}.In("admin", "member").String(),
//		"email": "required|email|" + validation.RuleFactory{}.Unique("users").String(),
//	}
//
// It is not spelled Rule because that name is the interface a rule of one's own
// implements. It is an empty struct, and every method returns exactly what the
// matching NewX returns: a second way of NAMING a constructor, not a second
// constructor.
type RuleFactory struct{}

// In builds an `in` rule.
func (RuleFactory) In(values ...string) *In { return NewIn(values...) }

// NotIn builds a `not_in` rule.
func (RuleFactory) NotIn(values ...string) *NotIn { return NewNotIn(values...) }

// Array builds an `array` rule.
func (RuleFactory) Array(keys ...string) *ArrayRule { return NewArrayRule(keys...) }

// Date builds the date rules of one field.
func (RuleFactory) Date() *Date { return NewDate() }

// Numeric builds the numeric rules of one field.
func (RuleFactory) Numeric() *Numeric { return NewNumeric() }

// Dimensions builds a `dimensions` rule.
func (RuleFactory) Dimensions() *Dimensions { return NewDimensions() }

// Enum builds an `enum` rule.
//
// Go has no enum type to read the cases off, so they are given: a Go program
// spells an enum as a named string type with a list of values, and that list is
// what this takes.
func (RuleFactory) Enum(cases ...string) *Enum { return NewEnum(cases...) }

// Exists builds an `exists` rule.
func (RuleFactory) Exists(table string, column ...string) *Exists {
	return NewExists(table, column...)
}

// Unique builds a `unique` rule.
func (RuleFactory) Unique(table string, column ...string) *Unique {
	return NewUnique(table, column...)
}

// File builds the upload rules of one field.
func (RuleFactory) File() *FileRule { return NewFileRule() }

// ImageFile builds the upload rules of an image field.
func (RuleFactory) ImageFile(allowSvg ...bool) *FileRule { return NewImageFile(allowSvg...) }

// Image builds the upload rules of an image field, as ImageFile does.
func (RuleFactory) Image(allowSvg ...bool) *FileRule { return NewImageFile(allowSvg...) }

// Default builds the file rule an application uses everywhere unless it says
// otherwise. Nothing stores it: what comes back is a plain builder, and the
// application keeps the one it made.
func (RuleFactory) Default() *FileRule { return NewFileRule() }

// Defaults builds a file rule with those rules already merged in. Nothing stores
// it either, for the reason Default gives.
func (RuleFactory) Defaults(rules ...string) *FileRule { return NewFileRule().Rules(rules...) }

// Email builds an `email` rule.
func (RuleFactory) Email() *EmailRule { return NewEmailRule() }

// Password builds a password policy with a minimum length.
func (RuleFactory) Password(min int) *Password { return NewPassword(min) }

// ExcludeIf builds an ExcludeIf over a condition already settled.
func (RuleFactory) ExcludeIf(condition bool) *ExcludeIf { return NewExcludeIf(condition) }

// ProhibitedIf builds a ProhibitedIf over a condition already settled.
func (RuleFactory) ProhibitedIf(condition bool) *ProhibitedIf { return NewProhibitedIf(condition) }

// RequiredIf builds a RequiredIf over a condition already settled.
func (RuleFactory) RequiredIf(condition bool) *RequiredIf { return NewRequiredIf(condition) }

// AnyOf builds an AnyOf: the value has to satisfy one of the rule sets.
func (RuleFactory) AnyOf(sets ...*Set) *AnyOf { return NewAnyOf(sets...) }

// Can builds a Can: the rule passes when the subject the Grant was issued to is
// allowed the ability against the value.
func (RuleFactory) Can(allows func(g auth.Grant, ability string, arguments []string, value any) bool, ability string, arguments ...string) *Can {
	return NewCan(allows, ability, arguments...)
}

// ---------------------------------------------------------------------------
// The presence verifier.
// ---------------------------------------------------------------------------

// PresenceQuery is the one question `unique` and `exists` ask of a store: how
// many rows of the collection hold any of the values in the column, once the
// Grant's tenant and the extra conditions are applied.
//
// It is a function rather than a repository because building the query needs
// hesape/database, and this package does not import it: the application passes
// the query in, already written against whatever it stores rows in.
type PresenceQuery func(ctx context.Context, g auth.Grant, collection, column string, values []any,
	excludeID any, idColumn string, extra map[string]string) (int, error)

// DatabasePresenceVerifier is the PresenceVerifier over a relational store.
//
// The query is given rather than built -- see PresenceQuery -- so what is left
// here is the two counts and the connection name.
type DatabasePresenceVerifier struct {
	query      PresenceQuery
	connection string
}

// NewDatabasePresenceVerifier returns a verifier that counts through the given
// query.
func NewDatabasePresenceVerifier(query PresenceQuery) *DatabasePresenceVerifier {
	return &DatabasePresenceVerifier{query: query}
}

// SetConnection names the connection the query is expected to run on.
func (d *DatabasePresenceVerifier) SetConnection(connection string) {
	d.connection = connection
}

// Connection reports the connection SetConnection named, which is what the
// query is expected to run on.
func (d *DatabasePresenceVerifier) Connection() string { return d.connection }

// GetCount counts the rows holding the value, through the query this was built
// with.
func (d *DatabasePresenceVerifier) GetCount(ctx context.Context, g auth.Grant, collection, column string,
	value any, excludeID any, idColumn string, extra map[string]string) (int, error) {
	if d.query == nil {
		return 0, errNoPresenceQuery
	}
	return d.query(ctx, g, collection, column, []any{value}, excludeID, idColumn, extra)
}

// GetMultiCount counts the rows holding any of the values, through the same
// query.
func (d *DatabasePresenceVerifier) GetMultiCount(ctx context.Context, g auth.Grant, collection, column string,
	values []any, extra map[string]string) (int, error) {
	if d.query == nil {
		return 0, errNoPresenceQuery
	}
	return d.query(ctx, g, collection, column, values, nil, "", extra)
}

// errNoPresenceQuery is what a verifier with no query answers. It is an error
// rather than a zero count, because a zero count reads as "nothing is there"
// and would make `unique` pass on a row that exists.
var errNoPresenceQuery = presenceError("validation: the presence verifier was built with no query")

type presenceError string

func (e presenceError) Error() string { return string(e) }
