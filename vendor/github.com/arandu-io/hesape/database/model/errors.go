package model

import (
	"errors"
	"fmt"
)

// ErrModelNotFound is what FindOrFail and its neighbours return when no row
// matched.
//
// It is returned wrapped with the table and the ids that were looked for, so
// errors.Is keeps working and the message still says which row:
//
//	fmt.Errorf("%w: users [7]", ErrModelNotFound)
//
// It is a sentinel and not a type for the reason database.ErrNotFound is one:
// the exception classifier turns it into a 404, and a caller comparing against
// it should not have to name a struct.
var ErrModelNotFound = errors.New("model: no query results for model")

// ErrMultipleRecordsFound is what Sole returns when the query matched more than
// one row.
var ErrMultipleRecordsFound = errors.New("model: multiple records found")

// ErrNoTenant is what every executing method returns when the Grant carries no
// tenant.
//
// auth.Tenant reads the empty string off three different Grants: the zero one,
// whose caller never authorized anything; the one auth.SystemGrant returns when
// it is asked for no tenant; and the one it returns when the tenant it was asked
// for cannot be one. They are three different mistakes, and this error does not
// tell them apart -- auth.Grant.Check does, because a refused system grant
// carries the reason it was refused.
//
// What they share is the consequence, which is why one error covers all three: a
// query with no tenant in its where clause reads every customer of the system.
// So it does not run.
var ErrNoTenant = errors.New("model: the grant carries no tenant, and a query without one reads every tenant (call auth.Authorize, or auth.SystemGrant with the tenant this work belongs to)")

// ErrNoKey is what Delete returns when the model has no primary key defined.
var ErrNoKey = errors.New("model: no primary key defined on model")

// ErrEmptyCollection is what ToQuery returns for an empty collection: there
// is no model to take the table from.
var ErrEmptyCollection = errors.New("model: unable to create query for empty collection")

// ErrRowHasNoModel is what a relation load reports when the rows it was handed
// carry no model to attach anything to.
//
// A relation is attached to the model behind a row, and a row reaches its model
// only when T embeds Model[T]: a plain struct has no field to point back with,
// and a struct written as a literal has a zero model inside it. Loading onto
// either would run the query, match the rows and attach the result to nothing,
// so it says so instead -- a silent success here is a relation the next line
// reads and does not find.
//
// It is wrapped with how many of the rows were unreachable, because one literal
// in a collection of hydrated rows and a collection that is entirely the plain
// shape are different mistakes.
var ErrRowHasNoModel = errors.New("model: these rows carry no model to load a relation onto (embed Model[T] in the entity, or read the rows back through a query)")

// ErrRelationNotFound is returned when a query names a relation the model never
// registered: an eager load of "author" on a model that declares no author, a
// nested path whose second segment does not exist on the model the first
// segment reaches, or a route binding through a relation that is not there.
//
// Wrapping it always carries the name and the table, because the useful part of
// the message is which relation was asked for on which model. It is a
// programming mistake rather than a runtime condition -- the fix is the
// declaration, not a retry -- so callers usually let it travel rather than
// matching on it.
var ErrRelationNotFound = errors.New("model: call to undefined relationship")

// ErrNamedScopeNotFound is what CallNamedScope reports for a scope the model
// never registered.
//
// A scope is an entry in Model.NamedScopes, so the miss is a missing key
// rather than a missing method.
var ErrNamedScopeNotFound = errors.New("model: call to undefined named scope")

// ModelNotFoundError carries what ModelNotFoundException records beyond its
// message: the model that was looked for and the ids it was looked for by.
//
// It unwraps to ErrModelNotFound, so errors.Is keeps working and a caller that
// only wants the 404 never has to name this type. A caller that wants the ids --
// a log line, a retry that drops the missing rows -- reaches them with
// errors.As.
type ModelNotFoundError struct {
	// Model is the table the query was on: what identifies a row on this
	// side.
	Model string

	// IDs is the primary keys that were looked for.
	IDs []any
}

// Error formats the message naming the table and, when there are any, the
// ids that were not found.
func (e *ModelNotFoundError) Error() string {
	if len(e.IDs) == 0 {
		return fmt.Sprintf("%s: %s", ErrModelNotFound, e.Model)
	}
	return fmt.Sprintf("%s: %s %v", ErrModelNotFound, e.Model, e.IDs)
}

// Unwrap makes errors.Is(err, ErrModelNotFound) true. See ErrModelNotFound for
// why the sentinel is what callers compare against.
func (e *ModelNotFoundError) Unwrap() error { return ErrModelNotFound }

// SetModel records the table and ids a not-found error is about, and
// returns e.
func (e *ModelNotFoundError) SetModel(model string, ids ...any) *ModelNotFoundError {
	e.Model = model
	e.IDs = ids
	return e
}

// GetModel returns the table the query was on.
func (e *ModelNotFoundError) GetModel() string { return e.Model }

// GetIDs returns the ids that were looked for.
func (e *ModelNotFoundError) GetIDs() []any { return e.IDs }

// modelNotFound builds the wrapped ErrModelNotFound with the table and ids.
func modelNotFound(table string, ids ...any) error {
	return &ModelNotFoundError{Model: table, IDs: ids}
}

// ErrJSONEncoding is the sentinel every failure to turn a model, an attribute
// or a resource into JSON wraps.
//
// The wrapping is done by ForModel, ForAttribute and the resource helper below,
// each of which adds what encoding failed and on which row -- a cast that
// produced a value encoding/json refuses, an attribute holding something with
// no JSON form. Callers match on this with errors.Is when they want to answer
// one status for any encoding failure; the message says which row to look at.
var ErrJSONEncoding = errors.New("model: json encoding failed")

// ForModel returns the ErrJSONEncoding wrapped with the model's table and
// key, when it is the model itself that failed to encode.
//
// It is a package function because a Go error value has no type to hang a
// constructor method on, and it names the model by its table rather than
// its type: the table is what a reader of the log can look up.
func ForModel(table string, key any, message string) error {
	return fmt.Errorf("%w: model [%s] with id [%v]: %s", ErrJSONEncoding, table, key, message)
}

// ForAttribute returns the ErrJSONEncoding wrapped with the model's table
// and the attribute's key, when it is an attribute that failed to encode.
func ForAttribute(table, key, message string) error {
	return fmt.Errorf("%w: attribute [%s] of model [%s]: %s", ErrJSONEncoding, key, table, message)
}

// ForResource returns the ErrJSONEncoding wrapped with the resource's name,
// the model's table and key, when it is a resource that failed to encode.
//
// A resource here is a value an http handler names, and http/resources does
// not reach into this package, so the resource arrives as its own name
// rather than a value this package reads it from.
func ForResource(resource, table string, key any, message string) error {
	return fmt.Errorf("%w: resource [%s] with model [%s] with id [%v]: %s", ErrJSONEncoding, resource, table, key, message)
}
