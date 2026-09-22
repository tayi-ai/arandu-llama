package view

import (
	"net/http"

	hhttp "github.com/arandu-io/hesape/http"
	"github.com/arandu-io/hesape/routing"
)

// Module serves the embedded assets and wires the renderer.
//
// It is a module for one reason: it was not, and every application shipped a
// page that asked for its own stylesheet and got 404.
//
// The route used to be exported as Mount, a plain function, with a comment
// arguing that "a module that exists only to serve two files is a module people
// have to remember to register". That reasoning was backwards, and the code
// proved it: Mount had zero call sites across every repository -- the kernel did
// not call it, the generated main.go did not call it, and neither did the
// generated sign-in screen. That screen emits three tags, and against a real
// server all three answered 404: no stylesheet, no HTMX, no client behaviour.
// Found by audit, reproduced end to end.
//
// A function has to be remembered. A module appears in the Register call next to
// events, jobs and the scheduler, which is where somebody reading main.go
// already looks to learn what an application is made of.
type Module struct{}

// NewModule returns the module.
//
//	k.Register(view.NewModule(), auth.New(...), events.NewModule())
func NewModule() *Module { return &Module{} }

// The compile-time assertion that this satisfies the module contract belongs
// here, and it is missing on purpose: foundation.Module is one of the surfaces
// that package says it is still waiting on the layer below to declare. It comes
// back the day it lands, and until then the three methods below are the whole
// contract, matched structurally.

// Name is the module identifier.
func (*Module) Name() string { return "view" }

// Routes registers the content-addressed asset route.
//
// One route, one handler. The hash in the path is what makes the response
// cacheable forever, and what makes a deploy invalidate it without anybody
// clearing anything.
func (*Module) Routes(r *routing.Router) {
	r.Get(AssetPath+"{hash}/{name}", http.HandlerFunc(Handler))
}

// ReloadTag supplies the development live-reload tag to the kernel.
//
// foundation.ReloadTagger, an optional interface, asked for only in
// development. The kernel gives the address of the stream it serves; this
// package owns the script and the asset it is served as.
func (*Module) ReloadTag(stream string) string { return ReloadTag(stream) }

// Renderer supplies the view renderer to the kernel.
//
// It is the RendererProvider the kernel looks for, an optional interface: the
// kernel asks every registered module whether it brings one, before any route
// is registered. That is what makes ctx.View work without the application
// calling a wiring function that somebody eventually forgets.
func (*Module) Renderer() hhttp.Renderer { return NewRenderer() }
