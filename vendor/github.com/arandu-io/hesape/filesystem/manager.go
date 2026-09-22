package filesystem

import (
	"context"
	"fmt"
	"io"
	"strings"
)

// Config is one disk's configuration.
//
// It is typed rather than a map because a disk configured with a misspelled key
// is a disk that boots and then behaves like a different one -- the local driver
// silently rooted at the working directory, the public URL silently absent. A
// struct makes the misspelling a compile error, which is the whole argument for
// build configuration being typed here.
type Config struct {
	// Driver names which creator builds the adapter: "local", "scoped", or a
	// name registered with [Disks.Extend].
	Driver string

	// Root is the directory the local driver writes under.
	Root string

	// URL is the public address files on this disk are reachable at, without a
	// trailing slash, and empty when they are not reachable without a session.
	//
	// Empty is the right answer for almost every disk here: a public address
	// carries no authorization, so a disk that has one is one whose contents are
	// world-readable to anybody holding the address. See [Disk.URL].
	URL string

	// Visibility is what a file written to this disk gets when the caller does
	// not say: [VisibilityPublic] or [VisibilityPrivate]. Empty means private,
	// which is the only default a tenant's file can have.
	Visibility string

	// Disk and Prefix configure the "scoped" driver: every key on the built disk
	// is stored under Prefix on the disk named by Disk.
	Disk   string
	Prefix string

	// ServeSigned makes [ServeFile] require a valid signature. See
	// [LocalFilesystemAdapter.ShouldServeSignedUrls].
	ServeSigned bool
}

// DriverCreator builds an adapter out of a configuration.
//
// It is what [Disks.Extend] registers. It receives the disk's name because a
// driver often
// wants it in an error message, and because the local driver puts it in the
// signed-URL route.
type DriverCreator func(name string, cfg Config) (Adapter, error)

// Drive returns a disk by name. The empty name is the default disk.
//
// It is [Disks.Disk] under a second name, so a project moving over finds the
// call it wrote either way.
func (ds *Disks) Drive(name string) (*Disk, error) { return ds.Disk(name) }

// Cloud returns the disk named as the cloud disk when the set was built.
func (ds *Disks) Cloud() (*Disk, error) {
	name := ds.GetDefaultCloudDriver()
	if name == "" {
		return nil, fmt.Errorf("%w: no cloud disk was named when the set was built", ErrNoDisk)
	}
	return ds.Disk(name)
}

// GetDefaultDriver returns the name of the disk that "" resolves to.
func (ds *Disks) GetDefaultDriver() string {
	ds.mu.RLock()
	defer ds.mu.RUnlock()
	return ds.defaultDisk
}

// GetDefaultCloudDriver returns the name of the disk [Disks.Cloud] answers with.
func (ds *Disks) GetDefaultCloudDriver() string {
	ds.mu.RLock()
	defer ds.mu.RUnlock()
	return ds.cloudDisk
}

// Set registers an already-built disk under a name and returns the set, so
// wiring chains.
func (ds *Disks) Set(name string, disk *Disk) *Disks {
	ds.mu.Lock()
	defer ds.mu.Unlock()
	ds.byName[name] = disk
	return ds
}

// Extend registers a driver creator under a name.
//
// It is what an adapter living in its own module hooks into: the S3 driver is
// github.com/arandu-io/hesape/filesystem/s3, a separate module because in Go
// there is no optional dependency, and the application that imports
// it registers it here in one line. That is why there is no CreateS3Driver on
// this type -- the root module cannot import its own submodule, and a creator
// that lied about being able to build one would fail at boot instead of at
// compile time.
func (ds *Disks) Extend(driver string, creator DriverCreator) *Disks {
	ds.mu.Lock()
	defer ds.mu.Unlock()
	ds.creators[driver] = creator
	return ds
}

// Build makes a disk out of a configuration without registering it.
//
// It is the one-off disk, for a job that needs a directory nobody named in
// configuration. Register it with [Disks.Set] or
// [Disks.Add] if it should be reachable by name.
func (ds *Disks) Build(name string, cfg Config) (*Disk, error) {
	switch cfg.Driver {
	case "local", "":
		return ds.CreateLocalDriver(name, cfg)
	case "scoped":
		return ds.CreateScopedDriver(name, cfg)
	}

	ds.mu.RLock()
	creator, ok := ds.creators[cfg.Driver]
	ds.mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("filesystem: no driver called %q. Register one with Disks.Extend", cfg.Driver)
	}
	adapter, err := creator(name, cfg)
	if err != nil {
		return nil, err
	}
	return NewDisk(name, adapter, cfg), nil
}

// CreateLocalDriver builds a disk over a directory.
func (ds *Disks) CreateLocalDriver(name string, cfg Config) (*Disk, error) {
	adapter, err := NewLocalFilesystemAdapter(cfg.Root)
	if err != nil {
		return nil, err
	}
	adapter.DiskName(name)
	adapter.ShouldServeSignedUrls(cfg.ServeSigned)
	return NewDisk(name, adapter, cfg), nil
}

// CreateScopedDriver builds a disk that is a subtree of another disk.
//
// Everything it stores lands at <prefix>/<tenant>/<key> on the disk named by
// Config.Disk: the scope narrows which part of that disk this one reaches, and
// the tenant still separates two customers inside the scope. Scoping only ever
// narrows, so a scoped disk cannot be used to step out of a tenant or out of
// the subtree it was given.
func (ds *Disks) CreateScopedDriver(name string, cfg Config) (*Disk, error) {
	if cfg.Disk == "" {
		return nil, fmt.Errorf("filesystem: the scoped disk %q names no disk to scope", name)
	}
	if cfg.Prefix == "" {
		return nil, fmt.Errorf("filesystem: the scoped disk %q names no prefix, and scoping nothing is the disk it scopes", name)
	}
	inner, err := ds.Disk(cfg.Disk)
	if err != nil {
		return nil, err
	}
	return NewDisk(name, NewScopedAdapter(inner.GetAdapter(), cfg.Prefix), cfg), nil
}

// ForgetDisk drops disks from the set, so the next lookup rebuilds or fails.
func (ds *Disks) ForgetDisk(names ...string) *Disks {
	ds.mu.Lock()
	defer ds.mu.Unlock()
	for _, name := range names {
		delete(ds.byName, name)
	}
	return ds
}

// Purge drops one disk. The empty name purges the default disk.
func (ds *Disks) Purge(name string) {
	if name == "" {
		name = ds.GetDefaultDriver()
	}
	ds.ForgetDisk(name)
}

// ScopedAdapter is an [Adapter] that stores everything under a prefix of
// another one.
//
// It is the "scoped" driver: a path prefix and nothing else. It is deliberately
// not a place to put a tenant -- the tenant prefix comes from [Key] and from the
// Grant, and a driver that could add one would be a second place isolation is
// decided.
type ScopedAdapter struct {
	inner  Adapter
	prefix string
}

// NewScopedAdapter wraps an adapter so every stored path goes under prefix.
func NewScopedAdapter(inner Adapter, prefix string) *ScopedAdapter {
	return &ScopedAdapter{inner: inner, prefix: strings.Trim(prefix, "/") + "/"}
}

var _ Adapter = (*ScopedAdapter)(nil)

// Inner returns the adapter underneath.
func (a *ScopedAdapter) Inner() Adapter { return a.inner }

func (a *ScopedAdapter) scope(storedPath string) string { return a.prefix + storedPath }

// Put writes under the prefix.
func (a *ScopedAdapter) Put(ctx context.Context, storedPath string, body io.Reader, contentType string) error {
	return a.inner.Put(ctx, a.scope(storedPath), body, contentType)
}

// Get reads from under the prefix.
func (a *ScopedAdapter) Get(ctx context.Context, storedPath string) (File, error) {
	f, err := a.inner.Get(ctx, a.scope(storedPath))
	if err != nil {
		return File{}, err
	}
	// The caller asked in unscoped paths and gets its own word back, the same
	// way Disk hands back keys rather than stored paths.
	f.Key = storedPath
	return f, nil
}

// Stat answers about the prefixed path.
func (a *ScopedAdapter) Stat(ctx context.Context, storedPath string) (Info, error) {
	i, err := a.inner.Stat(ctx, a.scope(storedPath))
	if err != nil {
		return Info{}, err
	}
	i.Key = storedPath
	return i, nil
}

// Exists reports on the prefixed path.
func (a *ScopedAdapter) Exists(ctx context.Context, storedPath string) (bool, error) {
	return a.inner.Exists(ctx, a.scope(storedPath))
}

// Delete removes the prefixed path.
func (a *ScopedAdapter) Delete(ctx context.Context, storedPath string) error {
	return a.inner.Delete(ctx, a.scope(storedPath))
}

// List answers in unprefixed paths, so a caller never sees the scope.
func (a *ScopedAdapter) List(ctx context.Context, prefix string) ([]string, error) {
	paths, err := a.inner.List(ctx, a.scope(prefix))
	if err != nil {
		return nil, err
	}
	out := make([]string, 0, len(paths))
	for _, p := range paths {
		// A driver answering outside the scope it was given has a bug, and
		// passing the path on would hand the caller a path it cannot address.
		if !strings.HasPrefix(p, a.prefix) {
			continue
		}
		out = append(out, strings.TrimPrefix(p, a.prefix))
	}
	return out, nil
}
