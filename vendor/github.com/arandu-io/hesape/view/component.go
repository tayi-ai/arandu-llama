package view

import (
	"fmt"
	"iter"
	"sync"
)

// Component is the contract a view component satisfies.
//
// Go has no abstract class, so the contract is this interface, and the
// shared state lives in BaseComponent, which a component embeds.
type Component interface {
	// Render returns the view the component draws: a *View, something with a
	// ToHTML method, a func(map[string]any) any, or the name of a registered
	// view.
	Render() any
}

// BaseComponent holds the state a component needs.
//
// A component embeds it to get the attribute bag, the alias name and the
// methods the compiler-generated code calls around a render.
type BaseComponent struct {
	// ComponentName is the alias the component was written as.
	ComponentName string

	// Attributes is the bag of everything the caller wrote that the component
	// did not declare.
	Attributes *ComponentAttributeBag

	// Props are the names the component declares. They are what
	// IgnoredParameterNames answers with.
	Props []string
}

// componentState holds the package-level state a component needs: the
// shared factory, the registered resolver and the inline-view cache.
var componentState struct {
	mu        sync.RWMutex
	factory   *Factory
	resolver  func(name string, data map[string]any) Component
	viewCache map[string]string
}

// Data returns the values available to the component's view: the attribute
// bag, plus whatever the embedding type passes in. A component states its
// data explicitly rather than exposing it through reflection.
func (c *BaseComponent) Data() map[string]any {
	if c.Attributes == nil {
		c.Attributes = c.newAttributeBag(nil)
	}
	data := c.Attributes.All()
	data["attributes"] = c.Attributes
	return data
}

// WithName sets the alias the component was written as, and returns c for
// chaining.
func (c *BaseComponent) WithName(name string) *BaseComponent {
	c.ComponentName = name
	return c
}

// WithAttributes replaces the component's attribute bag, and returns c for
// chaining.
func (c *BaseComponent) WithAttributes(attributes map[string]any) *BaseComponent {
	if c.Attributes == nil {
		c.Attributes = c.newAttributeBag(nil)
	}
	c.Attributes.SetAttributes(attributes)
	return c
}

// newAttributeBag returns a new attribute bag populated with attributes.
func (c *BaseComponent) newAttributeBag(attributes map[string]any) *ComponentAttributeBag {
	return NewComponentAttributeBag(attributes)
}

// ShouldRender reports whether the component draws at all. The default is
// true; a component that embeds BaseComponent and wants to disappear under
// some condition shadows this method.
func (c *BaseComponent) ShouldRender() bool { return true }

// IgnoredParameterNames returns a copy of Props, the parameter names the
// component declares.
func (c *BaseComponent) IgnoredParameterNames() []string {
	out := make([]string, len(c.Props))
	copy(out, c.Props)
	return out
}

// View draws view with data, through the factory the component is attached
// to -- the same one ForgetFactory drops.
func (c *BaseComponent) View(view string, data map[string]any) *View {
	return componentFactory().Make(view, data)
}

// ResolveView calls c.Render and normalizes the result: a *View or anything
// with a ToHTML method is returned as it is; a func is wrapped so it
// resolves when the data arrives; a string is the name of a registered view.
//
// A view here is compiled into the binary, never written to disk at request
// time.
func ResolveView(c Component) any {
	rendered := c.Render()

	switch value := rendered.(type) {
	case *View:
		return value
	case interface{ ToHTML() string }:
		return value
	case func(map[string]any) any:
		return func(data map[string]any) any {
			resolved := value(data)
			if v, ok := resolved.(*View); ok {
				return v
			}
			return resolved
		}
	default:
		return rendered
	}
}

// Resolve builds the named component with data, through the resolver
// registered by ResolveComponentsUsing. There is no container to fall back
// to, so a resolver that cannot build the component is an error naming it.
func Resolve(name string, data map[string]any) (Component, error) {
	componentState.mu.RLock()
	resolver := componentState.resolver
	componentState.mu.RUnlock()

	if resolver == nil {
		return nil, fmt.Errorf("view: no resolver can build the component %q. Call view.ResolveComponentsUsing first", name)
	}

	component := resolver(name, data)
	if component == nil {
		return nil, fmt.Errorf("view: the component resolver returned nothing for %q", name)
	}
	return component, nil
}

// ResolveComponentsUsing registers the function Resolve calls to build a
// component by name.
func ResolveComponentsUsing(resolver func(name string, data map[string]any) Component) {
	componentState.mu.Lock()
	componentState.resolver = resolver
	componentState.mu.Unlock()
}

// ForgetComponentsResolver clears the resolver registered by
// ResolveComponentsUsing.
func ForgetComponentsResolver() {
	componentState.mu.Lock()
	componentState.resolver = nil
	componentState.mu.Unlock()
}

// ForgetFactory clears the factory componentFactory built, so the next call
// builds a new one.
func ForgetFactory() {
	componentState.mu.Lock()
	componentState.factory = nil
	componentState.mu.Unlock()
}

// FlushCache clears the inline-view cache.
func FlushCache() {
	componentState.mu.Lock()
	componentState.viewCache = map[string]string{}
	componentState.mu.Unlock()
}

// componentFactory returns the shared factory, building one on first use.
func componentFactory() *Factory {
	componentState.mu.Lock()
	defer componentState.mu.Unlock()
	if componentState.factory == nil {
		componentState.factory = NewFactory()
	}
	return componentState.factory
}

// AnonymousComponent is a component with no behaviour: a view name and the
// data to draw it with, which is what a component written as markup alone
// compiles to.
type AnonymousComponent struct {
	BaseComponent

	view string
	data map[string]any
}

// NewAnonymousComponent returns a component that draws view with a copy of
// data.
func NewAnonymousComponent(view string, data map[string]any) *AnonymousComponent {
	copied := make(map[string]any, len(data))
	for k, v := range data {
		copied[k] = v
	}
	return &AnonymousComponent{view: view, data: copied}
}

// Render returns the component's view name.
func (c *AnonymousComponent) Render() any { return c.view }

// Data returns the values available to the component's view.
//
// The parent bag comes first, then this component's own attributes, then the
// data it was built with, and the bag itself last under "attributes" -- the
// order is what decides which value wins.
func (c *AnonymousComponent) Data() map[string]any {
	if c.Attributes == nil {
		c.Attributes = c.newAttributeBag(nil)
	}

	data := map[string]any{}
	if parent, ok := c.data["attributes"].(*ComponentAttributeBag); ok {
		for k, v := range parent.GetAttributes() {
			data[k] = v
		}
	}
	for k, v := range c.Attributes.GetAttributes() {
		data[k] = v
	}
	for k, v := range c.data {
		data[k] = v
	}
	data["attributes"] = c.Attributes
	return data
}

// DynamicComponent draws the component whose name is only known at render
// time. It resolves directly to the name of a registered view, with no
// second compile pass.
type DynamicComponent struct {
	BaseComponent

	// Component is the name of the component to draw.
	Component string
}

// NewDynamicComponent returns a component that draws the named component.
func NewDynamicComponent(component string) *DynamicComponent {
	return &DynamicComponent{Component: component}
}

// Render returns the name of the component to draw.
func (c *DynamicComponent) Render() any { return c.Component }

// InvokableComponentVariable is a component method exposed to the view as a
// value: the view writes the name and the call happens when the value is
// drawn, not before.
type InvokableComponentVariable struct {
	callable func() any
}

// NewInvokableComponentVariable wraps callable so its result is resolved
// lazily.
func NewInvokableComponentVariable(callable func() any) *InvokableComponentVariable {
	return &InvokableComponentVariable{callable: callable}
}

// ResolveDisplayableValue is an alias for Invoke.
func (v *InvokableComponentVariable) ResolveDisplayableValue() any { return v.Invoke() }

// Invoke calls the wrapped callable and returns its result, or nil if none
// was given.
func (v *InvokableComponentVariable) Invoke() any {
	if v.callable == nil {
		return nil
	}
	return v.callable()
}

// GetIterator invokes the callable and returns its result as a
// range-over-func sequence: a []any yields each element, and any other value
// yields once.
func (v *InvokableComponentVariable) GetIterator() iter.Seq2[int, any] {
	return func(yield func(int, any) bool) {
		switch resolved := v.Invoke().(type) {
		case nil:
			return
		case []any:
			for i, item := range resolved {
				if !yield(i, item) {
					return
				}
			}
		default:
			yield(0, resolved)
		}
	}
}

// String invokes the callable and renders its result as text.
func (v *InvokableComponentVariable) String() string { return Text(v.Invoke()) }
