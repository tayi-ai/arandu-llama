package view

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// FileViewFinder locates view files on disk by name and path.
//
// Views are compiled into the binary and registered by name at boot, so the
// finder is used during development and by the view build step -- never at
// request time in production. A view that is not compiled in is a boot-time
// error, not a runtime one.
type FileViewFinder struct {
	mu         sync.RWMutex
	files      fs.StatFS
	paths      []string
	extensions []string
	hints      map[string][]string
	views      map[string]string // name -> absolute path, cache
}

// NewFileViewFinder returns a finder that reads from files; nil means the
// real filesystem.
//
// files comes first because it is what makes the finder testable without
// writing a temporary tree.
func NewFileViewFinder(files fs.StatFS, paths, extensions []string) *FileViewFinder {
	if extensions == nil {
		extensions = []string{".kyse.go"}
	}
	resolved := make([]string, 0, len(paths))
	for _, p := range paths {
		if rp, err := filepath.Abs(p); err == nil {
			resolved = append(resolved, rp)
		} else {
			resolved = append(resolved, p)
		}
	}
	return &FileViewFinder{
		files:      files,
		paths:      resolved,
		extensions: extensions,
		hints:      map[string][]string{},
		views:      map[string]string{},
	}
}

// GetFilesystem returns the filesystem the finder reads from.
//
// nil is the real filesystem: the finder reaches for os directly when nobody
// handed it one, which is the case in every non-test caller.
func (f *FileViewFinder) GetFilesystem() fs.StatFS {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return f.files
}

// exists reports whether the candidate path is a file the finder can read.
func (f *FileViewFinder) exists(candidate string) bool {
	f.mu.RLock()
	files := f.files
	f.mu.RUnlock()

	if files == nil {
		_, err := os.Stat(candidate)
		return err == nil
	}
	_, err := files.Stat(strings.TrimPrefix(filepath.ToSlash(candidate), "/"))
	return err == nil
}

// Find returns the absolute path of a view, or an error naming where it
// looked. It caches results; call Flush to invalidate.
//
//	finder.Find("posts/index")
//	finder.Find("errors::404")       // namespace hint "errors"
func (f *FileViewFinder) Find(name string) (string, error) {
	f.mu.RLock()
	if cached, ok := f.views[name]; ok {
		f.mu.RUnlock()
		return cached, nil
	}
	f.mu.RUnlock()

	path, err := f.findUncached(name)
	if err != nil {
		return "", err
	}

	f.mu.Lock()
	f.views[name] = path
	f.mu.Unlock()
	return path, nil
}

func (f *FileViewFinder) findUncached(name string) (string, error) {
	normalized := strings.ReplaceAll(name, ".", "/")

	if f.hasHintInformation(name) {
		return f.findNamespaced(name)
	}

	var tried []string
	for _, dir := range f.paths {
		for _, ext := range f.extensions {
			candidate := filepath.Join(dir, normalized+ext)
			tried = append(tried, candidate)
			if f.exists(candidate) {
				return candidate, nil
			}
		}
	}
	return "", fmt.Errorf("view: %q not found. Looked in: %s", name, strings.Join(tried, ", "))
}

func (f *FileViewFinder) findNamespaced(name string) (string, error) {
	at := strings.Index(name, hintPathDelimiter)
	ns := name[:at]
	rest := name[at+len(hintPathDelimiter):]

	f.mu.RLock()
	hints, ok := f.hints[ns]
	f.mu.RUnlock()
	if !ok {
		return "", fmt.Errorf("view: namespace %q is not registered", ns)
	}

	normalized := strings.ReplaceAll(rest, ".", "/")
	var tried []string
	for _, dir := range hints {
		for _, ext := range f.extensions {
			candidate := filepath.Join(dir, normalized+ext)
			tried = append(tried, candidate)
			if f.exists(candidate) {
				return candidate, nil
			}
		}
	}
	return "", fmt.Errorf("view: %q (namespace %s) not found. Looked in: %s", name, ns, strings.Join(tried, ", "))
}

// AddLocation appends a directory the finder searches.
func (f *FileViewFinder) AddLocation(location string) error {
	resolved, err := filepath.Abs(location)
	if err != nil {
		return fmt.Errorf("view: cannot resolve %q: %w", location, err)
	}
	f.mu.Lock()
	f.paths = append(f.paths, resolved)
	f.mu.Unlock()
	return nil
}

// PrependLocation prepends a directory the finder searches.
func (f *FileViewFinder) PrependLocation(location string) error {
	resolved, err := filepath.Abs(location)
	if err != nil {
		return fmt.Errorf("view: cannot resolve %q: %w", location, err)
	}
	f.mu.Lock()
	f.paths = append([]string{resolved}, f.paths...)
	f.mu.Unlock()
	return nil
}

// AddNamespace registers a namespace hint that maps to one or more
// directories.
func (f *FileViewFinder) AddNamespace(ns string, hints []string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if existing, ok := f.hints[ns]; ok {
		f.hints[ns] = append(existing, hints...)
	} else {
		f.hints[ns] = hints
	}
}

// PrependNamespace prepends directories to a namespace hint.
func (f *FileViewFinder) PrependNamespace(ns string, hints []string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if existing, ok := f.hints[ns]; ok {
		f.hints[ns] = append(hints, existing...)
	} else {
		f.hints[ns] = hints
	}
}

// ReplaceNamespace replaces all hints for a namespace.
func (f *FileViewFinder) ReplaceNamespace(ns string, hints []string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.hints[ns] = hints
}

// AddExtension adds a file extension to search, without removing duplicates.
func (f *FileViewFinder) AddExtension(ext string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	// Remove existing and prepend (so it is tried first).
	filtered := make([]string, 0, len(f.extensions))
	for _, e := range f.extensions {
		if e != ext {
			filtered = append(filtered, e)
		}
	}
	f.extensions = append([]string{ext}, filtered...)
}

// HasHintInformation reports whether the view name carries a namespace prefix.
func (f *FileViewFinder) HasHintInformation(name string) bool {
	return f.hasHintInformation(name)
}

func (f *FileViewFinder) hasHintInformation(name string) bool {
	at := strings.Index(name, hintPathDelimiter)
	return at > 0
}

// GetPaths returns the directories searched.
func (f *FileViewFinder) GetPaths() []string {
	f.mu.RLock()
	defer f.mu.RUnlock()
	out := make([]string, len(f.paths))
	copy(out, f.paths)
	return out
}

// SetPaths replaces the search directories.
func (f *FileViewFinder) SetPaths(paths []string) {
	resolved := make([]string, 0, len(paths))
	for _, p := range paths {
		if rp, err := filepath.Abs(p); err == nil {
			resolved = append(resolved, rp)
		} else {
			resolved = append(resolved, p)
		}
	}
	f.mu.Lock()
	f.paths = resolved
	f.mu.Unlock()
}

// GetHints returns the registered namespace hints.
func (f *FileViewFinder) GetHints() map[string][]string {
	f.mu.RLock()
	defer f.mu.RUnlock()
	out := make(map[string][]string, len(f.hints))
	for k, v := range f.hints {
		copied := make([]string, len(v))
		copy(copied, v)
		out[k] = copied
	}
	return out
}

// GetExtensions returns the file extensions searched, in order.
func (f *FileViewFinder) GetExtensions() []string {
	f.mu.RLock()
	defer f.mu.RUnlock()
	out := make([]string, len(f.extensions))
	copy(out, f.extensions)
	return out
}

// Flush clears the path cache.
func (f *FileViewFinder) Flush() {
	f.mu.Lock()
	f.views = map[string]string{}
	f.mu.Unlock()
}

// GetViews returns the cached name-to-path map.
func (f *FileViewFinder) GetViews() map[string]string {
	f.mu.RLock()
	defer f.mu.RUnlock()
	out := make(map[string]string, len(f.views))
	for k, v := range f.views {
		out[k] = v
	}
	return out
}
