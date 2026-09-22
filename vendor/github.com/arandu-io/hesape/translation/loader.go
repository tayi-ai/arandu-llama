package translation

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"maps"
	"path"
	"sort"
	"sync"
)

// Lines is one group of a catalogue, in one locale: the item path of every line
// mapped to the line itself.
//
// The item path is dotted. A file that nests "string" under "min" is read as
// the single item "min.string", so the file keeps the shape a human edits and
// the lookup keeps one flat form.
type Lines map[string]string

// AppNamespace is the namespace of the application's own lines, the one a key
// with no "::" resolves in. It is the key the lines are stored under, and it
// travels through [Translator.AddLines].
const AppNamespace = "*"

// JSONGroup is the group the per locale JSON catalogue is loaded under. It is
// paired with [AppNamespace], so the JSON lines of a locale live at
// loaded["*"]["*"][locale] -- which is where [Translator.Get] looks first.
const JSONGroup = "*"

// Loader answers with the lines of a group in a locale.
//
// It returns nil for a group it does not carry, which is not an error: a
// catalogue is not required to translate everything, and the [Translator] falls
// through to the fallback locale and then to the English lines that ship here.
//
// AddNamespace and AddJSONPath are on the contract because [Translator]
// forwards to them; a loader with no notion of a path implements them as
// no-ops, as [ArrayLoader] does.
type Loader interface {
	// Load reads one group of one locale. An empty namespace means
	// AppNamespace, and the pair ("*", "*") asks for the JSON catalogue of
	// the locale.
	Load(locale, group, namespace string) Lines

	// AddNamespace answers addNamespace(): it points a namespace at the place
	// its lines are loaded from.
	AddNamespace(namespace, hint string)

	// AddJSONPath registers a directory holding per locale JSON catalogues.
	AddJSONPath(path string)

	// Namespaces is every registered namespace, mapped to its hint.
	Namespaces() map[string]string
}

// ArrayLoader is a catalogue written in Go rather than read from files. It is
// what a test translates against.
//
// It is safe for concurrent use: one loader answers every request, and
// [ArrayLoader.AddMessages] may be called while it does.
type ArrayLoader struct {
	mu sync.RWMutex
	// messages is namespace -> locale -> group -> lines.
	messages map[string]map[string]map[string]Lines
}

// NewArrayLoader answers ArrayLoader::__construct(). The loader starts empty;
// [ArrayLoader.AddMessages] fills it.
func NewArrayLoader() *ArrayLoader {
	return &ArrayLoader{messages: make(map[string]map[string]map[string]Lines)}
}

// Load reads one group of one locale. An empty namespace is read as
// [AppNamespace].
func (l *ArrayLoader) Load(locale, group, namespace string) Lines {
	if namespace == "" {
		namespace = AppNamespace
	}
	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.messages[namespace][locale][group]
}

// AddNamespace answers ArrayLoader::addNamespace(), which does nothing: an
// array loader holds its namespaced lines already and has no path to hint at.
func (l *ArrayLoader) AddNamespace(namespace, hint string) {}

// AddJSONPath answers ArrayLoader::addJsonPath(), which does nothing for the
// same reason.
func (l *ArrayLoader) AddJSONPath(path string) {}

// AddMessages stores one group of one locale, and returns the loader so that
// calls chain.
//
// An empty namespace means [AppNamespace]. The lines are copied, so the caller
// keeps no way to write into a loader that requests are reading.
func (l *ArrayLoader) AddMessages(locale, group string, messages Lines, namespace string) *ArrayLoader {
	if namespace == "" {
		namespace = AppNamespace
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.messages == nil {
		l.messages = make(map[string]map[string]map[string]Lines)
	}
	if l.messages[namespace] == nil {
		l.messages[namespace] = make(map[string]map[string]Lines)
	}
	if l.messages[namespace][locale] == nil {
		l.messages[namespace][locale] = make(map[string]Lines)
	}
	l.messages[namespace][locale][group] = maps.Clone(messages)
	return l
}

// Namespaces is always empty for an [ArrayLoader].
func (l *ArrayLoader) Namespaces() map[string]string { return map[string]string{} }

// FileLoader is a catalogue read from a filesystem.
//
// A group lives at <path>/<locale>/<group>.json, a namespaced group at the
// hint registered for the namespace, an override of a namespaced group at
// <path>/vendor/<namespace>/<locale>/<group>.json, and the JSON catalogue of a
// locale at <path>/<locale>.json. Those are the four shapes it reads. A
// language file is data, never code.
//
// Every path is read through one fs.FS. A caller reading the project's lang
// directory passes os.DirFS; the English lines that ship with this package are
// a FileLoader over the embedded catalogue.
//
// It is safe for concurrent use.
type FileLoader struct {
	mu        sync.RWMutex
	files     fs.FS
	paths     []string
	jsonPaths []string
	hints     map[string]string

	// cache holds every file that has been read, by name, so a group is parsed
	// once. A file that is absent is cached as nil, which is what keeps a
	// missing group from being stat'd on every request.
	cache map[string]Lines
}

// NewFileLoader answers FileLoader::__construct(). It reads paths out of files,
// and every path holds the locale directories.
//
// It parses everything under those paths before returning, and reports a
// malformed file as an error here: a language file that cannot be read is a
// boot failure, not a sentence that goes missing on the one request that needed
// it. [FileLoader.AddPath] and [FileLoader.AddJSONPath] cannot report that way
// -- they answer void methods -- so a file added later that does not parse is
// skipped when it is asked for.
func NewFileLoader(files fs.FS, paths ...string) (*FileLoader, error) {
	l := &FileLoader{
		files: files,
		paths: append([]string(nil), paths...),
		hints: make(map[string]string),
		cache: make(map[string]Lines),
	}
	for _, p := range l.paths {
		if err := l.warm(p); err != nil {
			return nil, err
		}
	}
	return l, nil
}

// Load answers FileLoader::load(). The pair ("*", "*") asks for the JSON
// catalogue of the locale; an empty or "*" namespace asks the ordinary paths;
// anything else asks the hint registered for that namespace, and then the
// overrides the application published for it.
func (l *FileLoader) Load(locale, group, namespace string) Lines {
	if group == JSONGroup && namespace == AppNamespace {
		return l.loadJSONPaths(locale)
	}
	if namespace == "" || namespace == AppNamespace {
		return l.loadPaths(l.snapshot(&l.paths), locale, group)
	}
	return l.loadNamespaced(locale, group, namespace)
}

// loadNamespaced reads a group registered under a namespace. A namespace with
// no hint has no lines.
func (l *FileLoader) loadNamespaced(locale, group, namespace string) Lines {
	l.mu.RLock()
	hint, registered := l.hints[namespace]
	l.mu.RUnlock()
	if !registered {
		return nil
	}
	lines := l.loadPaths([]string{hint}, locale, group)
	return l.loadNamespaceOverrides(lines, locale, group, namespace)
}

// loadNamespaceOverrides answers FileLoader::loadNamespaceOverrides(). It lets
// an application replace a line a module ships by publishing it under
// vendor/<namespace>, without editing the module.
func (l *FileLoader) loadNamespaceOverrides(lines Lines, locale, group, namespace string) Lines {
	out := maps.Clone(lines)
	if out == nil {
		out = make(Lines)
	}
	for _, p := range l.snapshot(&l.paths) {
		maps.Copy(out, l.read(path.Join(p, "vendor", namespace, locale, group+".json")))
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// loadPaths answers FileLoader::loadPaths(). Every path is read in turn and a
// later one replaces what an earlier one said, which is what array_replace_
// recursive does over the reduce.
func (l *FileLoader) loadPaths(paths []string, locale, group string) Lines {
	var out Lines
	for _, p := range paths {
		lines := l.read(path.Join(p, locale, group+".json"))
		if len(lines) == 0 {
			continue
		}
		if out == nil {
			out = make(Lines, len(lines))
		}
		maps.Copy(out, lines)
	}
	return out
}

// loadJSONPaths answers FileLoader::loadJsonPaths(). The JSON catalogue of a
// locale is one file keyed by the sentence itself, and the JSON paths are read
// before the ordinary ones.
func (l *FileLoader) loadJSONPaths(locale string) Lines {
	var out Lines
	for _, p := range append(l.snapshot(&l.jsonPaths), l.snapshot(&l.paths)...) {
		lines := l.read(path.Join(p, locale+".json"))
		if len(lines) == 0 {
			continue
		}
		if out == nil {
			out = make(Lines, len(lines))
		}
		maps.Copy(out, lines)
	}
	// The catalogue of this project is one file per group under the locale
	// directory -- lang/en/community.json -- as well as the single per-locale
	// file above. The lookup a page makes through Translator.Get goes through
	// this map, so each group must reach it flattened, or a line that exists in
	// the catalogue reads as missing and the key prints where the sentence was.
	for _, p := range append(l.snapshot(&l.jsonPaths), l.snapshot(&l.paths)...) {
		entries, err := fs.ReadDir(l.files, path.Join(p, locale))
		if err != nil {
			continue
		}
		sort.Slice(entries, func(i, j int) bool { return entries[i].Name() < entries[j].Name() })
		for _, entry := range entries {
			if entry.IsDir() || path.Ext(entry.Name()) != ".json" {
				continue
			}
			lines := l.read(path.Join(p, locale, entry.Name()))
			if len(lines) == 0 {
				continue
			}
			if out == nil {
				out = make(Lines, len(lines))
			}
			maps.Copy(out, lines)
		}
	}
	return out
}

// AddNamespace answers FileLoader::addNamespace(). The hint is the path the
// namespace's locale directories sit in.
func (l *FileLoader) AddNamespace(namespace, hint string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.hints == nil {
		l.hints = make(map[string]string)
	}
	l.hints[namespace] = hint
}

// Namespaces answers FileLoader::namespaces(). It returns a copy, so a caller
// cannot register a namespace by writing into what it was handed.
func (l *FileLoader) Namespaces() map[string]string {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return maps.Clone(l.hints)
}

// AddPath registers a directory of catalogues. The path is read the first time
// a group under it is asked for; a file there that does not parse is skipped,
// because there is nowhere here to report it from. [NewFileLoader] is where a
// malformed catalogue is an error.
func (l *FileLoader) AddPath(p string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.paths = append(l.paths, p)
}

// AddJSONPath registers a directory holding per locale JSON catalogues.
func (l *FileLoader) AddJSONPath(p string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.jsonPaths = append(l.jsonPaths, p)
}

// Paths answers FileLoader::paths(). It returns a copy of the registered
// catalogue paths, in the order they are read.
func (l *FileLoader) Paths() []string { return l.snapshot(&l.paths) }

// JSONPaths answers FileLoader::jsonPaths(). It returns a copy of the
// registered JSON catalogue paths.
func (l *FileLoader) JSONPaths() []string { return l.snapshot(&l.jsonPaths) }

// snapshot copies one of the path slices under the read lock, so a Load that is
// walking it cannot see an AddPath halfway through.
func (l *FileLoader) snapshot(list *[]string) []string {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return append([]string(nil), (*list)...)
}

// read parses one file, once. A file that is absent, or that does not parse, is
// remembered as having no lines: a group is asked for on every request, and
// reporting the same broken file on each of them is noise the boot already
// carried.
func (l *FileLoader) read(name string) Lines {
	l.mu.RLock()
	lines, done := l.cache[name]
	l.mu.RUnlock()
	if done {
		return lines
	}

	lines, err := l.parse(name)
	if err != nil {
		lines = nil
	}

	l.mu.Lock()
	defer l.mu.Unlock()
	if l.cache == nil {
		l.cache = make(map[string]Lines)
	}
	l.cache[name] = lines
	return lines
}

// parse reads one catalogue file and flattens it. An absent file is no lines
// and no error.
func (l *FileLoader) parse(name string) (Lines, error) {
	raw, err := fs.ReadFile(l.files, name)
	if err != nil {
		return nil, nil
	}
	var nested map[string]any
	if err := json.Unmarshal(raw, &nested); err != nil {
		return nil, fmt.Errorf("translation: parsing %s: %w", name, err)
	}
	lines := make(Lines)
	if err := flatten(lines, "", nested); err != nil {
		return nil, fmt.Errorf("translation: %s: %w", name, err)
	}
	return lines, nil
}

// warm parses every file under one path, so that a malformed catalogue is an
// error where the application boots rather than a sentence that goes missing.
//
// It walks the locale directories, the per locale JSON files beside them, and
// the vendor tree an application publishes namespace overrides into.
func (l *FileLoader) warm(p string) error {
	entries, err := fs.ReadDir(l.files, p)
	if err != nil {
		// A path with no directory is not an error: a project that ships no
		// lang directory still translates, out of the bundled catalogue.
		return nil
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Name() < entries[j].Name() })

	for _, entry := range entries {
		name := entry.Name()
		if !entry.IsDir() {
			if path.Ext(name) == ".json" {
				if err := l.warmFile(path.Join(p, name)); err != nil {
					return err
				}
			}
			continue
		}
		if name == "vendor" {
			if err := l.warmVendor(path.Join(p, name)); err != nil {
				return err
			}
			continue
		}
		if err := l.warmLocale(path.Join(p, name)); err != nil {
			return err
		}
	}
	return nil
}

// warmVendor parses <path>/vendor/<namespace>/<locale>/<group>.json.
func (l *FileLoader) warmVendor(p string) error {
	namespaces, err := fs.ReadDir(l.files, p)
	if err != nil {
		return nil
	}
	for _, namespace := range namespaces {
		if !namespace.IsDir() {
			continue
		}
		locales, err := fs.ReadDir(l.files, path.Join(p, namespace.Name()))
		if err != nil {
			continue
		}
		for _, locale := range locales {
			if !locale.IsDir() {
				continue
			}
			if err := l.warmLocale(path.Join(p, namespace.Name(), locale.Name())); err != nil {
				return err
			}
		}
	}
	return nil
}

// warmLocale parses every group of one locale directory. Anything that is not a
// .json file is ignored, so a lang directory can hold a README.
func (l *FileLoader) warmLocale(p string) error {
	groups, err := fs.ReadDir(l.files, p)
	if err != nil {
		return nil
	}
	for _, group := range groups {
		if group.IsDir() || path.Ext(group.Name()) != ".json" {
			continue
		}
		if err := l.warmFile(path.Join(p, group.Name())); err != nil {
			return err
		}
	}
	return nil
}

// warmFile parses one file into the cache, reporting what does not parse.
func (l *FileLoader) warmFile(name string) error {
	lines, err := l.parse(name)
	if err != nil {
		return err
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	l.cache[name] = lines
	return nil
}

// flatten writes every line of a parsed group under its dotted item path.
func flatten(out Lines, prefix string, in map[string]any) error {
	for key, value := range in {
		item := key
		if prefix != "" {
			item = prefix + "." + key
		}
		switch v := value.(type) {
		case string:
			out[item] = v
		case map[string]any:
			if err := flatten(out, item, v); err != nil {
				return err
			}
		default:
			return fmt.Errorf("item %q is %T, and a translation line is a string", item, value)
		}
	}
	return nil
}

// chain reads a set of loaders as one, the first that carries a group winning
// over the ones after it.
//
// It is how the English lines that ship with this package are answered after
// the application catalogue, for a [Translator] built over a loader of any
// kind.
type chain []Loader

func (c chain) Load(locale, group, namespace string) Lines {
	var out Lines
	// Backwards, so that the first loader of the chain has the last word.
	for i := len(c) - 1; i >= 0; i-- {
		lines := c[i].Load(locale, group, namespace)
		if len(lines) == 0 {
			continue
		}
		if out == nil {
			out = make(Lines, len(lines))
		}
		maps.Copy(out, lines)
	}
	return out
}

func (c chain) AddNamespace(namespace, hint string) {
	for _, l := range c {
		l.AddNamespace(namespace, hint)
	}
}

func (c chain) AddJSONPath(p string) {
	for _, l := range c {
		l.AddJSONPath(p)
	}
}

func (c chain) Namespaces() map[string]string {
	out := make(map[string]string)
	for _, l := range c {
		maps.Copy(out, l.Namespaces())
	}
	return out
}

// AddPath forwards to every loader of the chain that has one.
func (c chain) AddPath(p string) {
	for _, l := range c {
		if adder, ok := l.(interface{ AddPath(string) }); ok {
			adder.AddPath(p)
		}
	}
}
