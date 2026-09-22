package engines

// Engine turns a view path and data into rendered content.
//
// Views are compiled into Go functions at build time, so the engine
// delegates to the compiled function registry -- there is no runtime
// compilation.
type Engine interface {
	Get(path string, data map[string]any) (string, error)
}

// Base holds one field: the view drawn last, which an error message can
// name. An engine embeds this to answer for it.
type Base struct {
	lastRendered string
}

// GetLastRendered returns the path of the view drawn last.
func (e *Base) GetLastRendered() string { return e.lastRendered }

// setLastRendered records the path being drawn.
func (e *Base) setLastRendered(path string) { e.lastRendered = path }

// CompilerInterface is the contract an engine needs from a compiler.
//
// It is declared where it is consumed rather than beside the implementation,
// which is what keeps view -> engines -> compilers -> view from closing into
// a cycle.
type CompilerInterface interface {
	// GetCompiledPath returns the path the compiled output for path is
	// written to.
	GetCompiledPath(path string) string

	// IsExpired reports whether the compiled output for path is stale.
	IsExpired(path string) (bool, error)

	// Compile compiles path, writing fresh output for it.
	Compile(path string) error
}

// Resolver maps engine names to constructors, building and caching each
// engine on first use.
//
// The only engine registered today is the compiled-view engine; this type
// exists for future extension.
type Resolver struct {
	resolvers map[string]func() Engine
	resolved  map[string]Engine
}

// NewResolver returns a resolver with no engines registered.
func NewResolver() *Resolver {
	return &Resolver{
		resolvers: map[string]func() Engine{},
		resolved:  map[string]Engine{},
	}
}

// Register stores a constructor for the given engine name.
func (r *Resolver) Register(name string, resolver func() Engine) {
	delete(r.resolved, name)
	r.resolvers[name] = resolver
}

// Resolve returns the engine for name, creating it once and caching.
func (r *Resolver) Resolve(name string) (Engine, error) {
	if eng, ok := r.resolved[name]; ok {
		return eng, nil
	}
	resolver, ok := r.resolvers[name]
	if !ok {
		return nil, &engineNotFoundError{name: name}
	}
	eng := resolver()
	r.resolved[name] = eng
	return eng, nil
}

// Forget removes a resolved engine from the cache.
func (r *Resolver) Forget(name string) {
	delete(r.resolved, name)
}

type engineNotFoundError struct{ name string }

func (e *engineNotFoundError) Error() string {
	return "view: engine " + e.name + " is not registered"
}

// Compiler wraps a compiled view with the bookkeeping an engine needs:
// whether a path was already found current in this process, and which path
// is being drawn when something fails.
//
// There is no runtime compilation on the hot path: compilation runs at
// build time and produces Go functions directly.
type Compiler struct {
	Base

	compiler             CompilerInterface
	evaluate             func(path string, data map[string]any) (string, error)
	compiledOrNotExpired map[string]bool
}

// NewCompiler returns an engine that calls compiler around evaluate.
//
// evaluate is how the engine actually runs a compiled view: a compiled view
// here is a Go function already linked into the binary, not a file to read
// and include at request time.
func NewCompiler(compiler CompilerInterface, evaluate func(path string, data map[string]any) (string, error)) *Compiler {
	return &Compiler{
		compiler:             compiler,
		evaluate:             evaluate,
		compiledOrNotExpired: map[string]bool{},
	}
}

// Get compiles path if its output is stale, then evaluates it.
func (c *Compiler) Get(path string, data map[string]any) (string, error) {
	c.setLastRendered(path)

	if c.compiler != nil && !c.compiledOrNotExpired[path] {
		expired, err := c.compiler.IsExpired(path)
		if err != nil {
			return "", err
		}
		if expired {
			if err := c.compiler.Compile(path); err != nil {
				return "", err
			}
		}
	}

	compiled := path
	if c.compiler != nil {
		compiled = c.compiler.GetCompiledPath(path)
	}

	if c.evaluate == nil {
		return "", &engineNotFoundError{name: "compiler"}
	}

	result, err := c.evaluate(compiled, data)
	if err != nil {
		return "", err
	}

	if c.compiledOrNotExpired == nil {
		c.compiledOrNotExpired = map[string]bool{}
	}
	c.compiledOrNotExpired[path] = true
	return result, nil
}

// GetCompiler returns the compiler this engine drives.
func (c *Compiler) GetCompiler() CompilerInterface { return c.compiler }

// ForgetCompiledOrNotExpired clears the cache of paths known to be compiled
// and not expired.
func (c *Compiler) ForgetCompiledOrNotExpired() {
	c.compiledOrNotExpired = map[string]bool{}
}

// File reads a raw file from disk. That is the view build reading a
// .kyse.go source before compiling it, never a request.
type File struct {
	Base

	read func(path string) (string, error)
}

// NewFile returns an engine that reads a file with read.
func NewFile(read func(path string) (string, error)) *File {
	return &File{read: read}
}

// Get reads path from disk, ignoring data.
func (e *File) Get(path string, _ map[string]any) (string, error) {
	e.setLastRendered(path)
	return e.read(path)
}
