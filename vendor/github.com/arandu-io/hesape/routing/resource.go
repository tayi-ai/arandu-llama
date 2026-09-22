package routing

import (
	"net/http"
	"strings"
)

// Adapter turns one controller action into an http.Handler.
//
// C is the request context a controller action receives, and it is a type
// parameter rather than a named type because that type lives above this
// package: hesape/http owns the request context, the renderer and the answer
// to a rejected form, and a router that imported it would be a router that had
// to know what an answer looks like in order to match a path.
//
// The layer that owns C supplies the adapter, and the call reads:
//
//	routing.Resource(r, "invoices", InvoiceController{}, adapt)
//
// where adapt is that layer's own function. It is the same one a single route
// already goes through -- r.Get("/x", adapt(c.Show)) -- passed by name instead
// of applied, so Resource can apply it to the seven actions it finds.
type Adapter[C any] func(func(*C) error) http.Handler

// The seven actions of a resource controller, one interface each.
//
// The usual shape of this takes one object and registers seven routes whether
// or not the methods exist -- a request to a missing one is a runtime error.
// Here each action is its own tiny interface, and Resource registers exactly
// the ones the controller implements. A route that exists is a route that
// answers.
//
// Seven interfaces rather than one with seven methods, for the same reason the
// kernel splits Bootable, Background and Closable: a controller that only lists
// and shows should not be forced to write five empty methods, and five empty
// methods is how a 500 gets committed.
type (
	// Indexer answers GET /thing -- the list.
	Indexer[C any] interface {
		Index(*C) error
	}
	// Creator answers GET /thing/create -- the empty form.
	Creator[C any] interface {
		Create(*C) error
	}
	// Storer answers POST /thing -- the form submission.
	Storer[C any] interface {
		Store(*C) error
	}
	// Shower answers GET /thing/{id} -- one record.
	Shower[C any] interface {
		Show(*C) error
	}
	// Editor answers GET /thing/{id}/edit -- the filled form.
	Editor[C any] interface {
		Edit(*C) error
	}
	// Updater answers PUT and PATCH /thing/{id}.
	Updater[C any] interface {
		Update(*C) error
	}
	// Destroyer answers DELETE /thing/{id}.
	Destroyer[C any] interface {
		Destroy(*C) error
	}
)

// Resource registers the REST routes a controller implements.
//
//	routing.Resource(r, "invoices", InvoiceController{}, adapt)
//
// The seven, in the conventional order and with the conventional names:
//
//	GET    /invoices             index    invoices.index
//	GET    /invoices/create      create   invoices.create
//	POST   /invoices             store    invoices.store
//	GET    /invoices/{id}        show     invoices.show
//	GET    /invoices/{id}/edit   edit     invoices.edit
//	PUT    /invoices/{id}        update   invoices.update
//	PATCH  /invoices/{id}        update   invoices.update
//	DELETE /invoices/{id}        destroy  invoices.destroy
//
// The order matters: /invoices/create is registered before /invoices/{id} so a
// GET of "create" reaches the form rather than being read as an id. Go's
// ServeMux prefers the more specific pattern, and registering in this order
// keeps the intent readable even where the mux would sort it out anyway.
//
// A controller implementing none of the seven registers nothing and returns
// zero routes, which is a wiring mistake worth seeing in the route table.
//
// It is a function and not a method on Router because a method cannot take a
// type parameter in Go, and C has to come from somewhere. The router is the
// first argument, so the line still reads left to right as "register, on this
// router, this resource".
func Resource[C any](r *Router, name string, controller any, adapt Adapter[C]) []*Route {
	name = strings.Trim(name, "/")
	base := "/" + name
	one := base + "/{id}"

	var out []*Route
	add := func(method, pattern string, h func(*C) error) *Route {
		route := r.handle(method, pattern, adapt(h))
		out = append(out, route)
		return route
	}

	if c, ok := controller.(Indexer[C]); ok {
		add(http.MethodGet, base, c.Index).Name(name + ".index")
	}
	if c, ok := controller.(Creator[C]); ok {
		add(http.MethodGet, base+"/create", c.Create).Name(name + ".create")
	}
	if c, ok := controller.(Storer[C]); ok {
		add(http.MethodPost, base, c.Store).Name(name + ".store")
	}
	if c, ok := controller.(Shower[C]); ok {
		add(http.MethodGet, one, c.Show).Name(name + ".show")
	}
	if c, ok := controller.(Editor[C]); ok {
		add(http.MethodGet, one+"/edit", c.Edit).Name(name + ".edit")
	}
	if c, ok := controller.(Updater[C]); ok {
		route := add(http.MethodPut, one, c.Update)
		// PATCH answers the same handler and shows the same name. Only the PUT
		// row is indexed by it, so URL generation has one answer rather than
		// two.
		route.siblings = append(route.siblings, r.handle(http.MethodPatch, one, adapt(c.Update)))
		route.Name(name + ".update")
	}
	if c, ok := controller.(Destroyer[C]); ok {
		add(http.MethodDelete, one, c.Destroy).Name(name + ".destroy")
	}
	return out
}
