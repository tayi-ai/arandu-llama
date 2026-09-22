package routing

import (
	"context"
	"errors"
	"fmt"
	"net/http"

	"github.com/arandu-io/hesape/auth"
)

// UrlRoutable is what a record implements to be reachable from a URL: it
// knows the value that stands for it in a path, the column that value lives
// in, and how to find itself again from that value.
//
// # The tenant comes from the Grant, never from the URL
//
// ResolveRouteBinding receives an auth.Grant and MUST filter by
// auth.Tenant(g). This is the exact place a multi-tenant application leaks:
// /invoices/9 arrives, the id is read off the path, a row is loaded by that id
// alone, and the reader is handed another customer's invoice. No policy was
// violated -- none was consulted. The implementation is one WHERE clause:
//
//	func (Invoice) ResolveRouteBinding(ctx context.Context, g auth.Grant, value, field string) (routing.UrlRoutable, error) {
//		if field == "" {
//			field = Invoice{}.GetRouteKeyName()
//		}
//		return invoices.FindBy(ctx, g, field, value) // scoped by auth.Tenant(g)
//	}
//
// The Grant is a parameter and not something read out of the context, because
// a parameter cannot be forgotten: a resolver written without it does not
// compile.
type UrlRoutable interface {
	// GetRouteKey returns the value that stands for this record in a URL.
	GetRouteKey() string
	// GetRouteKeyName returns the column that value lives in, "id" unless
	// the route binds by another.
	GetRouteKeyName() string
	// ResolveRouteBinding resolves value into a record. An empty field means
	// GetRouteKeyName. It returns nil when nothing matched, which is a 404
	// and not an error.
	ResolveRouteBinding(ctx context.Context, g auth.Grant, value, field string) (UrlRoutable, error)
	// ResolveChildRouteBinding is the same lookup, restricted to the children
	// of this record, which is what a scoped nested resource resolves
	// through.
	ResolveChildRouteBinding(ctx context.Context, g auth.Grant, childType, value, field string) (UrlRoutable, error)
}

// BindingFunc resolves one route parameter into the value a handler receives.
//
// It is the shape of the closure Router.Bind takes, and of the one
// RouteBinding.ForModel returns. It takes the context every query needs and
// the Grant every query is scoped by.
//
// A binder that looks a record up by value alone returns another tenant's row
// for the same id. Filter by auth.Tenant(g).
type BindingFunc func(ctx context.Context, g auth.Grant, value string, route *Route) (any, error)

// MissingModelFunc decides what a model binding that matched nothing means. It
// is the third argument of Router.Model, and returning a nil value with a nil
// error means "answer 404".
type MissingModelFunc func(ctx context.Context, value string) (any, error)

// ErrTenantlessBinding is returned when a binding is resolved under a Grant
// that carries no tenant.
//
// It is a refusal and not a warning: a query with no tenant to scope by reads
// every customer, so the binding fails rather than running unscoped.
var ErrTenantlessBinding = errors.New("routing: route binding without a tenant")

// RouteBindings holds the parameter binders registered for a router, filled
// by Router.Bind and Router.Model.
type RouteBindings struct {
	m map[string]BindingFunc
	// models are the record types registered with Model, kept next to their
	// binder because the implicit binding calls them directly with the field
	// the route declared, through the registry.
	models map[string]UrlRoutable
}

// NewRouteBindings returns an empty set of bindings.
func NewRouteBindings() *RouteBindings {
	return &RouteBindings{
		m:      make(map[string]BindingFunc),
		models: make(map[string]UrlRoutable),
	}
}

// Bind registers a binder for a parameter key.
//
// When a route carries a parameter named key, the binder resolves the raw path
// value and the handler receives what it returned rather than the string.
func (b *RouteBindings) Bind(key string, binder BindingFunc) {
	b.m[normalizeBindingKey(key)] = ForCallback(binder)
}

// Model registers a model binding for a parameter key: the parameter resolves
// through the record's own ResolveRouteBinding, which is where the tenant
// filter lives.
//
// The optional callback says what a value that matched nothing means. Without
// one, nothing found is nothing bound, which the caller answers 404.
func (b *RouteBindings) Model(key string, class UrlRoutable, callback ...MissingModelFunc) {
	name := normalizeBindingKey(key)
	b.m[name] = ForModel(class, callback...)
	if b.models == nil {
		b.models = make(map[string]UrlRoutable)
	}
	b.models[name] = class
}

// modelFor returns the record type registered for a parameter, or nil.
func (b *RouteBindings) modelFor(key string) UrlRoutable {
	if b == nil || b.models == nil {
		return nil
	}
	return b.models[normalizeBindingKey(key)]
}

// GetBindingCallback returns the binder for a key, or nil.
func (b *RouteBindings) GetBindingCallback(key string) BindingFunc {
	if b == nil || b.m == nil {
		return nil
	}
	return b.m[normalizeBindingKey(key)]
}

// ForCallback returns binder unchanged: there is no container here to resolve
// a name through, so a binder is the closure itself.
func ForCallback(binder BindingFunc) BindingFunc { return binder }

// ForModel turns a record type into a binder: the value from the path is
// handed to that type's ResolveRouteBinding, under the Grant, and what comes
// back is what the handler receives.
func ForModel(class UrlRoutable, callback ...MissingModelFunc) BindingFunc {
	return func(ctx context.Context, g auth.Grant, value string, route *Route) (any, error) {
		if value == "" {
			return nil, nil
		}
		if err := requireTenant(g); err != nil {
			return nil, err
		}
		if class == nil {
			return nil, errors.New("routing: model binding registered without a record type")
		}
		_ = route

		// The field a route declared is applied by ImplicitRouteBinding,
		// which knows which parameter it is looking at.
		model, err := class.ResolveRouteBinding(ctx, g, value, "")
		if err != nil {
			return nil, err
		}
		if model != nil {
			return model, nil
		}

		// Nothing matched. A callback decides what that means; without one it
		// is a miss, and the caller turns a miss into a 404.
		if len(callback) > 0 && callback[0] != nil {
			return callback[0](ctx, value)
		}
		return nil, nil
	}
}

// ImplicitRouteBinding resolves a route's parameters into records.
//
// There are no type hints to reflect on here, so the registry is explicit --
// Router.Bind and Router.Model -- and this walks the route's parameters
// against it.
type ImplicitRouteBinding struct {
	bindings *RouteBindings
}

// NewImplicitRouteBinding creates a resolver backed by the bindings set.
func NewImplicitRouteBinding(b *RouteBindings) *ImplicitRouteBinding {
	return &ImplicitRouteBinding{bindings: b}
}

// ResolveForRoute resolves every parameter of the route that has a binder
// registered, and installs what came back as the value the handler reads.
//
// # Two things it refuses
//
// A Grant with no tenant, before it looks anything up: a lookup with nothing
// to scope by reads every customer.
//
// A child whose parent is a record and whose route is scoped: the child is
// resolved through the parent -- /clients/{client}/invoices/{invoice} finds
// the invoice among that client's, so an invoice id belonging to somebody else
// is a 404 rather than a page. That is the second leak this file exists to
// close, and it is the one a tenant filter alone does not catch: two clients
// of the same tenant are still two clients.
func (b *ImplicitRouteBinding) ResolveForRoute(ctx context.Context, g auth.Grant, route *Route, req *http.Request) error {
	if b == nil || b.bindings == nil || route == nil || req == nil {
		return nil
	}
	if err := requireTenant(g); err != nil {
		return err
	}

	for _, name := range route.ParameterNames() {
		binder := b.bindings.GetBindingCallback(name)
		if binder == nil {
			continue
		}

		value := route.OriginalParameter(req, name)
		if value == "" {
			continue
		}

		if parent, scoped := b.scopedParent(route, req, name); scoped {
			model, err := parent.ResolveChildRouteBinding(ctx, g, name, value, route.BindingFieldFor(name))
			if err != nil {
				return fmt.Errorf("routing: binding %q: %w", name, err)
			}
			if model == nil {
				return fmt.Errorf("routing: binding %q: no record for %q under its parent", name, value)
			}
			route.SetParameter(req, name, model)
			continue
		}

		var resolved any
		var err error
		if class := b.bindings.modelFor(name); class != nil {
			// A model binding resolves through the record type itself, with
			// the field the route declared -- {post:slug} looks the post up by
			// slug rather than by id.
			var model UrlRoutable
			model, err = class.ResolveRouteBinding(ctx, g, value, route.BindingFieldFor(name))
			if model != nil {
				resolved = model
			}
		} else {
			resolved, err = binder(ctx, g, value, route)
		}
		if err != nil {
			return fmt.Errorf("routing: binding %q: %w", name, err)
		}
		if resolved == nil {
			return fmt.Errorf("routing: binding %q: no record for %q", name, value)
		}
		route.SetParameter(req, name, resolved)
	}
	return nil
}

// scopedParent reports whether name should be resolved through the parameter
// before it, and returns that parent: the parent is a record, the route does
// not forbid scoping, and the route either asked for it or bound the child by
// a field.
func (b *ImplicitRouteBinding) scopedParent(route *Route, req *http.Request, name string) (UrlRoutable, bool) {
	parent, ok := route.ParentOfParameter(req, name).(UrlRoutable)
	if !ok {
		return nil, false
	}
	if route.PreventsScopedBindings() {
		return nil, false
	}
	if route.EnforcesScopedBindings() || route.BindingFieldFor(name) != "" {
		return parent, true
	}
	return nil, false
}

// requireTenant refuses a Grant that cannot scope a query.
func requireTenant(g auth.Grant) error {
	if auth.Tenant(g) == "" {
		return fmt.Errorf("%w: the Grant carries no tenant, and a lookup with nothing to scope by reads every customer", ErrTenantlessBinding)
	}
	return nil
}

// normalizeBindingKey is what Bind and Model do to a key: a parameter written
// with dashes is registered under underscores, because that is how it reaches
// the route.
func normalizeBindingKey(key string) string {
	out := []byte(key)
	for i := range out {
		if out[i] == '-' {
			out[i] = '_'
		}
	}
	return string(out)
}
