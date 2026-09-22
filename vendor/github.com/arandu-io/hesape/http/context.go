package http

import (
	"context"
	"encoding/json"
	stdhttp "net/http"

	"github.com/arandu-io/hesape/log"
	"github.com/arandu-io/hesape/toon"
)

// Context is what a controller action receives.
//
// It is everything a controller action gets, and no more: the request,
// the response, and helpers that answer. Nothing more -- and the "nothing more"
// is the point. There is no database handle here, no repository, no Grant. A
// controller that could reach the data layer would be a controller that skipped
// the service, and therefore the policy that guards it.
//
// It is a struct rather than an interface because it has no second
// implementation and never will. One way to do one thing.
type Context struct {
	// Response and Request are exported: a handler that needs the standard
	// library reaches for it directly instead of waiting for a wrapper.
	Response stdhttp.ResponseWriter
	Request  *stdhttp.Request

	render Renderer
	urls   URLGenerator
}

// NewContext builds the Context an action is called with.
//
// It is exported because the thing that calls actions -- the router -- is no
// longer in this package, and the two fields it fills are unexported so that
// nothing else can swap a renderer or a URL table mid-request. hesape/routing
// calls it once per request; a test that drives an action directly calls it with
// nil for whichever of the two that action does not use.
func NewContext(w stdhttp.ResponseWriter, r *stdhttp.Request, render Renderer, urls URLGenerator) *Context {
	return &Context{Response: w, Request: r, render: render, urls: urls}
}

// Renderer draws a named view with typed data.
//
// It is an interface here, and implemented in the view package, for one reason:
// the view package imports http, so http importing the view package back would
// be a cycle. The kernel wires the concrete one at boot.
type Renderer interface {
	Render(ctx context.Context, w stdhttp.ResponseWriter, status int, name string, data any) error
}

// URLGenerator turns the name of a route into its path.
//
// It is an interface here for the reason Renderer is: the route table lives in
// hesape/routing, which imports this package to build a Context, so naming its
// type here would be a cycle. hesape/routing.Routes satisfies it.
//
// The method is Route and not URL because that is what the table calls it, and
// because Context.URL is the caller-facing name -- one word per side, and
// neither of them shadows net/url.
type URLGenerator interface {
	Route(name string, params ...string) (string, error)
}

// ExceptionHandler turns an error that reached the edge into an answer.
//
// Declared here and implemented in hesape/exception, which must not import this
// package: exception produces middleware through pipeline precisely so that the
// error path stays below the request path. The kernel wires the concrete one at
// boot, and hesape/routing calls it instead of panicking.
type ExceptionHandler interface {
	// Report records the error wherever failures are recorded. It is separate
	// from Render because an error that was answered still has to be reported,
	// and a report that only happens when a page is drawn misses every failure
	// on a queued job.
	Report(ctx context.Context, err error)
	// Render writes the answer: the debug page in development, a status page
	// anywhere else.
	Render(w stdhttp.ResponseWriter, r *stdhttp.Request, err error)
}

// Ctx returns the request context, which carries the Collector, the logger and
// the request id.
func (c *Context) Ctx() context.Context { return c.Request.Context() }

// Param reads a path parameter: /invoices/{id} gives Param("id").
func (c *Context) Param(name string) string { return c.Request.PathValue(name) }

// Query reads a query string parameter.
func (c *Context) Query(name string) string { return c.Request.URL.Query().Get(name) }

// Input reads a form field, from the body or the query string.
//
// Named Input rather than Form because Input is the word the vocabulary
// already uses for it, and the vocabulary is the point.
func (c *Context) Input(name string) string { return c.Request.FormValue(name) }

// URL is the path of a named route, with its parameters filled in order.
//
//	ctx.URL("posts.show", post.ID)   -> "/posts/01J.../"
//
// It is what a controller hands a view instead of building a path by hand.
// "/posts/"+id compiles and keeps compiling after the route moves; this stops
// working the moment the name is wrong, and says so.
//
// An unknown name or a wrong number of parameters returns empty and is logged
// at ERROR with the name. Empty is what the views already treat as "there is no
// link here" -- a page with a missing button is recoverable, and a template
// renderer that panics takes the whole page down to report something a missing
// link would have said better.
func (c *Context) URL(name string, params ...string) string {
	if c.urls == nil {
		return ""
	}
	out, err := c.urls.Route(name, params...)
	if err != nil {
		log.For(c.Ctx()).Error("building a URL", "route", name, "error", err)
		return ""
	}
	return out
}

// View renders a page. The data is a typed struct, never a map.
//
//	return ctx.View("invoices/index", IndexData{Invoices: list})
//
// A map would compile and render blank on a typo, which is the failure this
// framework exists to make impossible.
func (c *Context) View(name string, data any) error {
	return c.renderWith(stdhttp.StatusOK, name, data)
}

// Fragment renders a partial with a status, for HTMX.
//
// The status matters: a form that failed validation answers 422 with the form
// fragment, so the browser and the logs agree with each other. Answering 200
// would make both of them believe it worked.
//
// # The layout has to let htmx swap it, and by default htmx does not
//
// This comment used to end with "and HTMX swaps it in", stated as fact. It does
// not. htmx's default response handling is
//
//	[{code:"204", swap:false}, {code:"[23]..", swap:true}, {code:"[45]..", swap:false, error:true}]
//
// in the copy this framework embeds (2.0.4), and a 422 matches the third entry.
// The fragment is fetched, the status is right, the body is correct -- and it is
// thrown away. The person sees the form they submitted, unchanged, with no
// message on it: the same failure that made a guard's 403 invisible, on the
// answer every form in an application gives.
//
// Neither HX-Retarget nor HX-Reswap rescues it. Both are read after shouldSwap
// has already been decided from the table above; they change where and how, not
// whether. What decides whether is the configuration, and htmx reads it from the
// document without any script running:
//
//	<meta name="htmx-config" content='{"responseHandling":[
//	  {"code":"204","swap":false},
//	  {"code":"422","swap":true},
//	  {"code":"[23]..","swap":true},
//	  {"code":"[45]..","swap":false,"error":true}]}'>
//
// 422 before the catch-all, because htmx takes the first entry that matches.
// That line belongs in the application's layout, once -- it is the layout that
// decides what a fragment answer means, and a per-page opt-in would be a second
// way to answer a rejected form. A meta tag is not a script, so it costs
// nothing against a `script-src 'self'` policy and nothing in Node.
//
// A refusal is a different thing and does not go through here: 403, 419 and 429
// are not a form coming back with messages on it, and they answer through
// Refuse.
func (c *Context) Fragment(status int, name string, data any) error {
	return c.renderWith(status, name, data)
}

func (c *Context) renderWith(status int, name string, data any) error {
	if c.render == nil {
		return errNoRenderer
	}
	return c.render.Render(c.Ctx(), c.Response, status, name, data)
}

// Redirect answers a redirect, and does the right thing under HTMX.
//
// An HTMX request that gets a 302 follows it inside the fragment, so the whole
// page ends up nested in a div. HX-Redirect is the header that makes the browser
// navigate instead. Handling it here means no application has to remember.
func (c *Context) Redirect(to string) error {
	Redirect(c.Response, c.Request, to)
	return nil
}

// RedirectRoute redirects to a named route, with its parameters filled in order.
//
// It is Redirect over URL, and it exists so that the address a person is sent to
// after a POST is looked up by name like every other address in the application.
// An unknown name is logged by URL and answered as a redirect to "/", because
// the alternative -- a Location header that is empty -- is a browser that stays
// where it is with no explanation.
func (c *Context) RedirectRoute(name string, params ...string) error {
	to := c.URL(name, params...)
	if to == "" {
		to = "/"
	}
	return c.Redirect(to)
}

// Redirect is Context.Redirect for the code that holds a raw ResponseWriter: a
// middleware, or a handler registered with Get rather than Action.
//
// It is one function and not one per caller. The branch below existed in three
// copies -- here, in the auth module's handlers, and in the route guards -- and
// the middle one had already been wrong once, answering HX-Redirect with 200 to
// every client including the ones that do not read the header, which left a
// browser with scripts off on a blank page after signing in. Three copies of a
// six-line decision is three chances to be the copy that is wrong.
//
// 303 and not 302: after a POST, 303 is what tells the browser to GET the next
// address instead of posting the body to it again.
func Redirect(w stdhttp.ResponseWriter, r *stdhttp.Request, to string) {
	if r.Header.Get("HX-Request") == "true" {
		// A body alongside HX-Redirect is swapped in before the browser
		// navigates, so there is deliberately none: 204.
		w.Header().Set("HX-Redirect", to)
		w.WriteHeader(stdhttp.StatusNoContent)
		return
	}
	stdhttp.Redirect(w, r, to, stdhttp.StatusSeeOther)
}

// Refuse answers a refusal that the person in front of the browser can see.
//
// The status and the sentence are the caller's and are unchanged: 403 stays 403
// and 419 stays 419, so logs, monitoring and every non-HTMX client see exactly
// what they saw before.
//
// What it adds is the HTMX half. htmx does not swap a 4xx -- its response
// handling is configured `{code:"[45]..", swap:false, error:true}` in the copy
// this framework embeds -- so a guard's 403 and an expired CSRF token both
// arrived as a fired event and a blank screen: the person clicked, nothing at
// all happened, and the sentence explaining why was in a response body that was
// thrown away. HX-Refresh makes the browser reload the page as an ordinary
// navigation, and that request is not an HTMX one, so the same middleware
// answers the same refusal and the browser renders it. Nothing loops: a reload
// is a GET, which CSRF verification does not check, and which a role guard
// answers as a plain page.
//
// # What a reload can and cannot show
//
// It reloads the page the person is ON, which is not always the address that was
// refused. When the two are the same -- an expired token on a form, a fragment
// of the page itself -- the refusal comes back as a full page and is read. When
// they differ -- a boosted link into an area this account may not open -- the
// person gets their own page back, correct and unexplained, and learns only that
// the link does nothing.
//
// That half is deliberately not solved by sending them to the refused address
// instead: the address would come off the request, and a request target
// beginning with "//" is parsed into URL.Path host and all, so handing it to
// location.href is an open redirect built out of a refusal. Something visible
// and incomplete beats something invisible, and beats a hole.
//
// It is one function and not a branch in each middleware, for the reason
// Redirect above is one function: the last time this decision existed in three
// copies, one of them was wrong.
func Refuse(w stdhttp.ResponseWriter, r *stdhttp.Request, status int, message string) {
	if r.Header.Get("HX-Request") == "true" {
		w.Header().Set("HX-Refresh", "true")
	}
	stdhttp.Error(w, message, status)
}

// JsonResource is what [Context.JSON] takes: the fields a type is allowed to
// answer with, declared once beside the type rather than remembered at each
// place it is answered from.
//
// It is declared here rather than imported from the resource layer, which
// declares the same contract: that layer builds a response and imports this
// package to do it, so an import back would be a cycle. Go compares an
// interface by its methods, so a resource satisfies this one by being what it
// already is, and the two names are one contract rather than two.
type JsonResource interface {
	// ToArray returns the fields that may leave, by name.
	ToArray() map[string]any

	// With returns what goes beside them at the top level of the response:
	// metadata about the answer rather than the thing being answered with.
	With() map[string]any
}

// JSON answers with JSON. It exists for the endpoints that are genuinely an
// API; a page answers with View.
//
// It takes a resource and not a value, and the signature is the whole point of
// it. An encoder handed an entity answers with whatever fields the entity
// happens to have, including the ones somebody adds to it later without ever
// reading this handler: a password hash, an internal note, the identifier of
// the account a row belongs to. A resource cannot do that, because what leaves
// is a list somebody wrote.
//
// The answer is the fields under a "data" key, and whatever With returns beside
// it. The key is fixed here and does not follow the resource layer's
// configurable wrapping: this is one answer with one shape, and an endpoint
// that needs another builds its body with the resource layer and writes it to
// [Context.Response] itself.
//
// A field carrying a value that reports itself missing is left out, which is
// what a conditional field is for -- the field a person may not see is absent
// rather than present and empty.
//
// The body is built before anything is written. An encoder writing straight
// into the response has already sent the status and half an object by the time
// it reports that it cannot marshal the rest, and neither can be taken back.
func (c *Context) JSON(status int, resource JsonResource) error {
	encoded, err := json.Marshal(resourceBody(resource))
	if err != nil {
		return err
	}

	c.Response.Header().Set("Content-Type", "application/json; charset=utf-8")
	c.Response.WriteHeader(status)
	_, err = c.Response.Write(encoded)
	return err
}

// TOON answers with Token-Oriented Object Notation for a caller that
// explicitly chooses the AI-oriented representation. JSON remains the default
// response and request format; TOON is never selected from an Accept header.
//
// TOON takes the same JsonResource as [Context.JSON] and applies the same field
// selection, conditional-field filter, and top-level metadata semantics. The
// complete body is encoded before any header or status is written, leaving the
// response untouched when encoding fails.
func (c *Context) TOON(status int, resource JsonResource) error {
	encoded, err := toon.Marshal(resourceBody(resource))
	if err != nil {
		return err
	}

	c.Response.Header().Set("Content-Type", toon.MediaType+"; charset=utf-8")
	c.Response.WriteHeader(status)
	_, err = c.Response.Write(encoded)
	return err
}

func resourceBody(resource JsonResource) map[string]any {
	data := make(map[string]any)
	for name, value := range resource.ToArray() {
		if missing, ok := value.(interface{ IsMissing() bool }); ok && missing.IsMissing() {
			continue
		}
		data[name] = value
	}

	body := map[string]any{"data": data}
	for name, value := range resource.With() {
		body[name] = value
	}
	return body
}

// Status answers with a status and no body.
func (c *Context) Status(code int) error {
	c.Response.WriteHeader(code)
	return nil
}

// errNoRenderer is what a View call gets when no view layer was wired.
//
// It names the fix, because the alternative is a nil dereference in a stack
// trace that points at the framework rather than at the missing line in
// bootstrap/app.go.
var errNoRenderer = &noRendererError{}

type noRendererError struct{}

func (*noRendererError) Error() string {
	return "arandu: this handler rendered a view and no view layer is wired. " +
		"Register it in bootstrap/app.go: k.Register(view.NewModule())"
}
