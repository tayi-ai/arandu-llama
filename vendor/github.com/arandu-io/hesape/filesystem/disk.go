package filesystem

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"mime"
	"path"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/arandu-io/hesape/auth"
)

// ErrNotFound is returned when a key does not exist for this tenant.
//
// It is the same error whether the file is absent or belongs to somebody else,
// and that is deliberate: distinguishing them would tell a caller which keys
// exist in other tenants.
var ErrNotFound = errors.New("filesystem: not found")

// ErrNoDisk is returned by [Disks.Disk] for a name nobody registered.
var ErrNoDisk = errors.New("filesystem: no such disk")

// Info is what a file is, without its content.
//
// Stat answers with it, Get carries it inside a [File], and [Serve] writes it
// into the response headers. One type, because a size that means one thing in a
// listing and another in a download is how a Content-Length ends up wrong.
type Info struct {
	// Key is the key the caller asked for, never the stored path. A caller that
	// saw the tenant prefix here would start building paths out of it.
	Key         string
	Size        int64
	ContentType string
	ModifiedAt  time.Time
}

// File is what a Get returns: the [Info] plus the bytes.
type File struct {
	Info
	// Body is the content. The caller closes it.
	Body io.ReadCloser
}

// Adapter is what a driver implements.
//
// It takes stored paths, already resolved by [Key], and knows nothing about
// tenants or Grants. That is the point: the isolation is enforced once, in this
// package, instead of in each driver -- and a driver that forgets it cannot,
// because it is never told which tenant it is serving.
//
// Six methods, because they are the six an object store and a directory both
// have. Copy, Move, DeleteDirectory and every listing narrower than this one are
// composed out of them by [Disk], so a new driver is six methods and not twenty.
type Adapter interface {
	// Put writes body at path, creating whatever it needs to.
	Put(ctx context.Context, path string, body io.Reader, contentType string) error
	// Get reads it back. It returns ErrNotFound when the path is not there.
	Get(ctx context.Context, path string) (File, error)
	// Stat answers the same metadata Get carries, without the body.
	Stat(ctx context.Context, path string) (Info, error)
	// Exists reports whether the path is there.
	Exists(ctx context.Context, path string) (bool, error)
	// Delete removes it. Removing what is not there is not an error.
	Delete(ctx context.Context, path string) error
	// List returns the stored paths under a prefix, in no promised order.
	List(ctx context.Context, prefix string) ([]string, error)
}

// Disk is what an application calls.
//
// Every method takes an auth.Grant. That is not ceremony: it is the only thing
// standing between "the application stores files" and "any handler can read any
// customer's files by building a string". The Grant decides the tenant, the
// tenant decides the prefix, and the prefix is not reachable from the key.
//
// Reads are not exempt. AllFiles, Get, Stat and Exists take a Grant for the same
// reason Put does, and a listing is the one call where forgetting
// it hands over the names of every file in the system.
type Disk struct {
	name    string
	adapter Adapter
	config  Config

	// mu guards the three callbacks below, which are set at wiring time and
	// read on every request.
	mu                 sync.RWMutex
	serveUsing         ServeCallback
	temporaryURL       TemporaryURLCallback
	temporaryUploadURL TemporaryUploadURLCallback
}

// NewDisk names an adapter.
//
// The name is what appears in an error and in `aru` output; it is not part of
// any path, so renaming a disk does not move a file.
//
// The optional [Config] is what [Disk.URL] reads a public base address out of,
// and what [Disk.GetConfig] answers with. Passing none leaves it zero, which is
// a disk that has no public address -- and saying so is the correct answer for
// one.
func NewDisk(name string, a Adapter, cfg ...Config) *Disk {
	d := &Disk{name: name, adapter: a}
	if len(cfg) > 0 {
		d.config = cfg[0]
	}
	return d
}

// Name returns the name this disk was registered under.
func (d *Disk) Name() string { return d.name }

// GetAdapter returns the driver underneath.
//
// It exists so [URLSigner] can ask whether the driver is a [Presigner], and so
// a test can assert against the fake it installed. It is not a way around the
// Grant: an Adapter takes stored paths, and the only thing that produces one is
// [Key], which needs a Grant.
func (d *Disk) GetAdapter() Adapter { return d.adapter }

// GetDriver returns the same driver [Disk.GetAdapter] does.
//
// There is one object here rather than a driver wrapping an adapter, so both
// names answer with it rather than one of them being missing for a reason
// nobody could act on.
func (d *Disk) GetDriver() Adapter { return d.adapter }

// GetConfig returns the configuration this disk was built with.
func (d *Disk) GetConfig() Config { return d.config }

// Put writes a file under the tenant of the Grant.
//
// An empty contentType is inferred from the key. Inferring from the key and not
// from what an upload announced is deliberate: the client-supplied type is a
// string an attacker chose, and storing it is how an upload becomes stored XSS
// three months later when something serves it back.
func (d *Disk) Put(ctx context.Context, g auth.Grant, key string, body io.Reader, contentType string) error {
	full, err := Key(g, key)
	if err != nil {
		return err
	}
	if contentType == "" {
		contentType = TypeOf(key)
	}
	if err := d.adapter.Put(ctx, full, body, contentType); err != nil {
		return d.wrap("put", key, err)
	}
	return nil
}

// Get reads one back. The caller closes File.Body.
func (d *Disk) Get(ctx context.Context, g auth.Grant, key string) (File, error) {
	full, err := Key(g, key)
	if err != nil {
		return File{}, err
	}
	f, err := d.adapter.Get(ctx, full)
	if err != nil {
		return File{}, d.wrap("get", key, err)
	}
	// The driver answers in stored paths; the caller asked in keys, and gets
	// its own word back.
	f.Key = key
	return f, nil
}

// Stat answers what Get would carry, without moving the bytes.
func (d *Disk) Stat(ctx context.Context, g auth.Grant, key string) (Info, error) {
	full, err := Key(g, key)
	if err != nil {
		return Info{}, err
	}
	i, err := d.adapter.Stat(ctx, full)
	if err != nil {
		return Info{}, d.wrap("stat", key, err)
	}
	i.Key = key
	return i, nil
}

// Exists reports whether the key is there for this tenant.
func (d *Disk) Exists(ctx context.Context, g auth.Grant, key string) (bool, error) {
	full, err := Key(g, key)
	if err != nil {
		return false, err
	}
	ok, err := d.adapter.Exists(ctx, full)
	if err != nil {
		return false, d.wrap("exists", key, err)
	}
	return ok, nil
}

// Missing is Exists inverted, and it exists because the two readings are not
// equally easy to get right: `if !ok` after a call that also returns an error is
// where a failed lookup silently becomes "the file is not there".
func (d *Disk) Missing(ctx context.Context, g auth.Grant, key string) (bool, error) {
	ok, err := d.Exists(ctx, g, key)
	if err != nil {
		return false, err
	}
	return !ok, nil
}

// Size returns how many bytes the file holds.
func (d *Disk) Size(ctx context.Context, g auth.Grant, key string) (int64, error) {
	i, err := d.Stat(ctx, g, key)
	if err != nil {
		return 0, err
	}
	return i.Size, nil
}

// LastModified returns when the file was last written.
func (d *Disk) LastModified(ctx context.Context, g auth.Grant, key string) (time.Time, error) {
	i, err := d.Stat(ctx, g, key)
	if err != nil {
		return time.Time{}, err
	}
	return i.ModifiedAt, nil
}

// MimeType returns the stored content type.
//
// It is the type [Put] inferred from the key, never the one an upload
// announced, so it is the same string [Serve] will send -- which is what makes
// it safe to show to a person or to switch on.
func (d *Disk) MimeType(ctx context.Context, g auth.Grant, key string) (string, error) {
	i, err := d.Stat(ctx, g, key)
	if err != nil {
		return "", err
	}
	if i.ContentType == "" {
		return TypeOf(key), nil
	}
	return i.ContentType, nil
}

// Checksum returns the SHA-256 of the file's contents, in lowercase hex.
//
// It is SHA-256 and not MD5: the reason to hash a stored file is to answer "is
// this the same file", and answering it with a hash that can be made to collide
// on purpose is answering a different question. There is no algorithm option,
// for the same reason there is no second anything here.
//
// It reads the whole file, because a checksum of part of one is not a checksum.
func (d *Disk) Checksum(ctx context.Context, g auth.Grant, key string) (string, error) {
	f, err := d.Get(ctx, g, key)
	if err != nil {
		return "", err
	}
	defer f.Body.Close()

	sum := sha256.New()
	if _, err := io.Copy(sum, f.Body); err != nil {
		return "", d.wrap("checksum", key, err)
	}
	return hex.EncodeToString(sum.Sum(nil)), nil
}

// Delete removes a file. Removing what is not there is not an error.
func (d *Disk) Delete(ctx context.Context, g auth.Grant, key string) error {
	full, err := Key(g, key)
	if err != nil {
		return err
	}
	if err := d.adapter.Delete(ctx, full); err != nil {
		return d.wrap("delete", key, err)
	}
	return nil
}

// AllFiles returns every key under a directory, recursively, without the tenant
// part, sorted.
//
// The empty directory means everything this tenant has. Sorted because a listing
// that changes order between two identical calls turns a paginated screen into
// a bug report nobody can reproduce.
//
// The argument is matched as a prefix, so "invoices/" is the directory and
// "invoices" also catches "invoices-2025.pdf". Only the caller knows which was
// meant -- [Disk.Files] and [Disk.Directories] are the ones that read it as a
// directory.
func (d *Disk) AllFiles(ctx context.Context, g auth.Grant, directory string) ([]string, error) {
	full, err := prefixPath(g, directory)
	if err != nil {
		return nil, err
	}
	paths, err := d.adapter.List(ctx, full)
	if err != nil {
		return nil, d.wrap("list", directory, err)
	}
	tenantPrefix := auth.Tenant(g) + "/"
	keys := make([]string, 0, len(paths))
	for _, p := range paths {
		// A driver that answers with something outside the prefix it was given
		// is a driver with a bug, and passing it on would be this package
		// handing a caller another tenant's key. Dropped, not returned.
		if !strings.HasPrefix(p, tenantPrefix) {
			continue
		}
		keys = append(keys, strings.TrimPrefix(p, tenantPrefix))
	}
	sort.Strings(keys)
	return keys, nil
}

// Files returns the keys directly inside a directory, without descending.
//
// The empty directory is the tenant's root. What "directly inside" means is
// decided here and not by the driver: an object store has no directories at all,
// and a listing that differed between local disk and S3 would be a module that
// works until it is deployed.
func (d *Disk) Files(ctx context.Context, g auth.Grant, directory string) ([]string, error) {
	dir := asDirectory(directory)
	all, err := d.AllFiles(ctx, g, dir)
	if err != nil {
		return nil, err
	}
	out := make([]string, 0, len(all))
	for _, key := range all {
		if !strings.Contains(strings.TrimPrefix(key, dir), "/") {
			out = append(out, key)
		}
	}
	return out, nil
}

// Directories returns the directories directly inside a directory.
//
// A directory exists because a key underneath it does; there is nothing else for
// one to be. Every name comes back with a trailing slash, so what it is cannot be
// mistaken and it can be passed straight back in.
func (d *Disk) Directories(ctx context.Context, g auth.Grant, directory string) ([]string, error) {
	return d.directories(ctx, g, directory, false)
}

// AllDirectories returns every directory under a directory, at any depth.
func (d *Disk) AllDirectories(ctx context.Context, g auth.Grant, directory string) ([]string, error) {
	return d.directories(ctx, g, directory, true)
}

func (d *Disk) directories(ctx context.Context, g auth.Grant, directory string, recursive bool) ([]string, error) {
	dir := asDirectory(directory)
	all, err := d.AllFiles(ctx, g, dir)
	if err != nil {
		return nil, err
	}
	seen := map[string]bool{}
	for _, key := range all {
		rest := strings.TrimPrefix(key, dir)
		for i, c := range rest {
			if c != '/' {
				continue
			}
			seen[dir+rest[:i+1]] = true
			if !recursive {
				break
			}
		}
	}
	out := make([]string, 0, len(seen))
	for name := range seen {
		out = append(out, name)
	}
	sort.Strings(out)
	return out, nil
}

// asDirectory turns a directory argument into the prefix the rest of the
// package speaks. The empty directory is the tenant's root, and
// anything else gets the trailing slash that makes "invoices" mean the folder
// rather than every key starting with those eight letters.
func asDirectory(directory string) string {
	if directory == "" || strings.HasSuffix(directory, "/") {
		return directory
	}
	return directory + "/"
}

// Copy duplicates a file inside one tenant.
//
// Both keys are resolved against the same Grant, so a copy cannot cross into
// another tenant in either direction -- which is the mistake worth preventing,
// because "copy this into the shared folder" is how it gets asked for.
func (d *Disk) Copy(ctx context.Context, g auth.Grant, src, dst string) error {
	f, err := d.Get(ctx, g, src)
	if err != nil {
		return err
	}
	defer f.Body.Close()
	return d.Put(ctx, g, dst, f.Body, f.ContentType)
}

// Move copies and then deletes the source.
//
// In that order. The other order loses the file when the write fails, and the
// write is the half that fails.
func (d *Disk) Move(ctx context.Context, g auth.Grant, src, dst string) error {
	if src == dst {
		return nil
	}
	if err := d.Copy(ctx, g, src, dst); err != nil {
		return err
	}
	return d.Delete(ctx, g, src)
}

// DeleteDirectory removes everything under a directory, for this tenant only.
//
// It is the one operation whose blast radius is worth stating out loud: an
// empty directory deletes everything the tenant has. It still cannot reach past
// the tenant, because the prefix is resolved the same way a key is.
func (d *Disk) DeleteDirectory(ctx context.Context, g auth.Grant, directory string) error {
	keys, err := d.AllFiles(ctx, g, directory)
	if err != nil {
		return err
	}
	for _, key := range keys {
		if err := d.Delete(ctx, g, key); err != nil {
			return err
		}
	}
	return nil
}

// Append adds a line to the end of a file, creating it when it is not there.
//
// It is read-modify-write and not an append syscall, because an object store has
// no such thing: two concurrent Appends to the same key end with one of the two,
// not both. The answer to a file several writers extend at once is the queue,
// not this.
func (d *Disk) Append(ctx context.Context, g auth.Grant, key, data string) error {
	return d.join(ctx, g, key, data, false)
}

// Prepend adds a line to the front of a file, creating it when it is not there.
//
// It reads the whole file into memory to do it. That is what prepending to a
// file is, on a directory and on a bucket alike.
func (d *Disk) Prepend(ctx context.Context, g auth.Grant, key, data string) error {
	return d.join(ctx, g, key, data, true)
}

func (d *Disk) join(ctx context.Context, g auth.Grant, key, data string, front bool) error {
	existing, err := d.Get(ctx, g, key)
	switch {
	case errors.Is(err, ErrNotFound):
		return d.Put(ctx, g, key, strings.NewReader(data), "")
	case err != nil:
		return err
	}
	body, err := io.ReadAll(existing.Body)
	closeErr := existing.Body.Close()
	if err != nil {
		return d.wrap("read", key, err)
	}
	if closeErr != nil {
		return d.wrap("read", key, closeErr)
	}

	var b strings.Builder
	b.Grow(len(body) + len(data) + 1)
	if front {
		b.WriteString(data)
		b.WriteString("\n")
		b.Write(body)
	} else {
		b.Write(body)
		b.WriteString("\n")
		b.WriteString(data)
	}
	return d.Put(ctx, g, key, strings.NewReader(b.String()), existing.ContentType)
}

// wrap names the disk and the key in an error, and lets ErrNotFound through
// unchanged in the sense that errors.Is still sees it.
func (d *Disk) wrap(op, key string, err error) error {
	return fmt.Errorf("filesystem: %s %s on disk %q: %w", op, key, d.name, err)
}

// TypeOf infers a content type from a key's extension, falling back to a type
// that browsers download rather than render.
//
// The fallback matters: serving an unknown file as text/html is how an upload
// becomes stored XSS.
func TypeOf(key string) string {
	if t := mime.TypeByExtension(path.Ext(key)); t != "" {
		return t
	}
	return "application/octet-stream"
}

// Disks is the set of configured disks, and the one place a name becomes a
// disk.
//
// Disks are registered at boot from configuration, and a handler asks for one
// by name.
// The zero value is not usable; call [NewDisks].
type Disks struct {
	mu          sync.RWMutex
	byName      map[string]*Disk
	defaultDisk string
	cloudDisk   string
	creators    map[string]DriverCreator
}

// NewDisks returns an empty set whose default disk will be defaultName.
//
// The default may be registered after this call -- configuration names it
// before it is wired -- and asking for it before it exists is an error, not a
// panic at boot.
//
// An optional second name is the cloud disk, which is what [Disks.Cloud]
// answers with. A project with one disk does not have to name it.
func NewDisks(defaultName string, cloud ...string) *Disks {
	ds := &Disks{
		byName:      map[string]*Disk{},
		defaultDisk: defaultName,
		creators:    map[string]DriverCreator{},
	}
	if len(cloud) > 0 {
		ds.cloudDisk = cloud[0]
	}
	return ds
}

// Add registers an adapter under a name and returns the disk it became.
//
// Registering the same name twice replaces it, because the alternative -- an
// error at boot from a config file that listed a disk twice -- is a crash for a
// mistake with an obvious reading.
func (ds *Disks) Add(name string, a Adapter) *Disk {
	d := NewDisk(name, a)
	ds.mu.Lock()
	defer ds.mu.Unlock()
	ds.byName[name] = d
	return d
}

// Disk returns a disk by name. The empty name is the default disk.
//
// There is no second accessor for the default. A caller that has no opinion
// passes "", and a caller that does passes the name; one function, so nothing
// has to decide which of two to reach for.
func (ds *Disks) Disk(name string) (*Disk, error) {
	ds.mu.RLock()
	defer ds.mu.RUnlock()
	if name == "" {
		name = ds.defaultDisk
	}
	if name == "" {
		return nil, fmt.Errorf("%w: no disk was asked for and none is the default", ErrNoDisk)
	}
	d, ok := ds.byName[name]
	if !ok {
		return nil, fmt.Errorf("%w: %q. Registered: %s", ErrNoDisk, name, strings.Join(ds.names(), ", "))
	}
	return d, nil
}

// Names returns the registered names, sorted.
func (ds *Disks) Names() []string {
	ds.mu.RLock()
	defer ds.mu.RUnlock()
	return ds.names()
}

func (ds *Disks) names() []string {
	out := make([]string, 0, len(ds.byName))
	for name := range ds.byName {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}
