package filesystem

import (
	"context"
	"crypto/rand"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"strings"
)

// LocalFilesystemAdapter stores files in a directory.
//
// It is the [Adapter] that needs nothing installed, which makes it the right
// one for development and for a single machine. For anything with more than one
// replica, github.com/arandu-io/hesape/filesystem/s3 is the same contract over
// the S3 protocol -- and Cloudflare R2 is the default there.
//
// Like every Adapter it is handed stored paths and never a Grant, so the tenant
// prefix is not something this file could forget: it was applied by [Key]
// before the call.
type LocalFilesystemAdapter struct {
	root string
	// dir is the root held open, and it is what every operation goes through.
	//
	// A path built by joining strings is confined only for as long as nothing
	// on the way to it is a symbolic link: the join answers about the string,
	// and the kernel answers about the links. This one is confined by the
	// kernel, one component at a time, at the moment the syscall runs -- so
	// there is no window between deciding a path is inside the root and using
	// it.
	//
	// What it confines is the root, and not a directory inside it: a link
	// written relative to where it sits, from one tenant's directory into
	// another's, never leaves the root and is followed. A link whose target is
	// written as an absolute path is refused wherever it points, because the
	// open root resolves the names it walked and an absolute target is not one
	// of them.
	//
	// The followed case is the smaller exposure and it is stated rather than
	// papered over -- whatever can plant that link can already read both
	// directories, and nothing a caller can ask of this adapter plants one.
	// Leaving the root is the one worth refusing, because there the
	// application's own credentials reach files no key names and write bytes
	// where nothing asked them to.
	dir *os.Root
	// diskName and serveSigned are what lets a link be turned back into a disk.
	// See [LocalFilesystemAdapter.DiskName]
	// and [LocalFilesystemAdapter.ShouldServeSignedUrls]; both are set at wiring
	// time and read afterwards.
	diskName    string
	serveSigned bool
}

// NewLocalFilesystemAdapter returns an adapter rooted at a directory.
//
// The root is created if it does not exist, because the alternative is an
// application that boots fine and fails on the first upload.
func NewLocalFilesystemAdapter(root string) (*LocalFilesystemAdapter, error) {
	if root == "" {
		return nil, errors.New("filesystem: the local adapter needs a root directory")
	}
	absolute, err := filepath.Abs(root)
	if err != nil {
		return nil, fmt.Errorf("filesystem: resolving %s: %w", root, err)
	}
	if err := os.MkdirAll(absolute, 0o755); err != nil {
		return nil, fmt.Errorf("filesystem: creating %s: %w", absolute, err)
	}
	dir, err := os.OpenRoot(absolute)
	if err != nil {
		return nil, fmt.Errorf("filesystem: opening %s: %w", absolute, err)
	}
	return &LocalFilesystemAdapter{root: absolute, dir: dir}, nil
}

var _ Adapter = (*LocalFilesystemAdapter)(nil)

// Root returns the directory this adapter writes under.
//
// It is here for an error message and for a test that wants to look at what
// landed on disk. It is not a way to build a path: what goes under the root is
// decided by [Key], and this returns the part of the answer that carries no
// tenant.
func (a *LocalFilesystemAdapter) Root() string { return a.root }

// partialPrefix names an upload in flight.
//
// It is a prefix rather than a suffix so List can skip it by looking at the base
// name only, and it starts with a dot so it sorts away from real keys. A stored
// path can never begin with it: [CleanKey] resolves the key, and this prefix is
// only ever produced here.
const partialPrefix = ".hesape-partial-"

// Put writes a file, creating whatever directories it needs.
//
// The content type is not stored: on disk it is the extension, and keeping a
// sidecar file per object to hold one string is a second thing to keep in sync.
// Get infers it, which is what a static file server does anyway.
func (a *LocalFilesystemAdapter) Put(_ context.Context, storedPath string, body io.Reader, _ string) error {
	name, err := a.file(storedPath)
	if err != nil {
		return err
	}
	if err := a.dir.MkdirAll(path.Dir(name), 0o755); err != nil {
		return fmt.Errorf("filesystem: creating the directory for %s: %w", storedPath, err)
	}

	// Written to a temporary name and renamed, so a reader never sees a
	// half-written file and a crash mid-upload leaves nothing to clean up.
	//
	// The name is unique per call. It used to be the path plus ".partial", which
	// is the same name for every concurrent upload of the same key: two requests
	// opened it with O_TRUNC, interleaved their bytes into one file, and both
	// renamed it into place. The stored object was neither upload. Two people
	// replacing the same attachment is not a rare race -- it is a retry. Found
	// by audit.
	f, tmp, err := a.partial(path.Dir(name))
	if err != nil {
		return fmt.Errorf("filesystem: writing %s: %w", storedPath, err)
	}
	// Nobody reads a partial file, but the mode a file is created with is
	// masked by the umask and the stored object is 0644 like every other.
	if err := f.Chmod(0o644); err != nil {
		_ = f.Close()
		_ = a.dir.Remove(tmp)
		return fmt.Errorf("filesystem: writing %s: %w", storedPath, err)
	}
	if _, err := io.Copy(f, body); err != nil {
		_ = f.Close()
		_ = a.dir.Remove(tmp)
		return fmt.Errorf("filesystem: writing %s: %w", storedPath, err)
	}
	if err := f.Close(); err != nil {
		_ = a.dir.Remove(tmp)
		return fmt.Errorf("filesystem: writing %s: %w", storedPath, err)
	}
	if err := a.dir.Rename(tmp, name); err != nil {
		_ = a.dir.Remove(tmp)
		return fmt.Errorf("filesystem: writing %s: %w", storedPath, err)
	}
	return nil
}

// partial creates the file an upload is written to before it is renamed into
// place, beside its destination and inside the root. It returns the open file
// and the name to rename or remove it by.
//
// os.CreateTemp is not usable here: it takes a directory name and opens it the
// way any other path is opened, so a directory component that is a symbolic
// link puts the partial file wherever the link points -- and the rename after
// it carries the upload there. Everything this adapter opens goes through the
// root, and this is no exception.
//
// The retry is what os.CreateTemp does. O_EXCL turns a name somebody else
// already took into an error rather than into a truncated file, and the count
// is a bound on a loop whose exit is a random draw.
func (a *LocalFilesystemAdapter) partial(dir string) (*os.File, string, error) {
	for range 10000 {
		name := path.Join(dir, partialPrefix+rand.Text())
		f, err := a.dir.OpenFile(name, os.O_RDWR|os.O_CREATE|os.O_EXCL, 0o644)
		if errors.Is(err, fs.ErrExist) {
			continue
		}
		if err != nil {
			return nil, "", err
		}
		return f, name, nil
	}
	return nil, "", errors.New("no unused name for a partial upload")
}

// Get reads a file back. The caller closes File.Body.
func (a *LocalFilesystemAdapter) Get(ctx context.Context, storedPath string) (File, error) {
	info, err := a.Stat(ctx, storedPath)
	if err != nil {
		return File{}, err
	}
	name, err := a.file(storedPath)
	if err != nil {
		return File{}, err
	}
	f, err := a.dir.Open(name)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return File{}, ErrNotFound
		}
		return File{}, fmt.Errorf("filesystem: reading %s: %w", storedPath, err)
	}
	return File{Info: info, Body: f}, nil
}

// Stat answers what Get would carry, without opening the file.
func (a *LocalFilesystemAdapter) Stat(_ context.Context, storedPath string) (Info, error) {
	name, err := a.file(storedPath)
	if err != nil {
		return Info{}, err
	}
	info, err := a.dir.Stat(name)
	if errors.Is(err, fs.ErrNotExist) {
		return Info{}, ErrNotFound
	}
	if err != nil {
		return Info{}, fmt.Errorf("filesystem: reading %s: %w", storedPath, err)
	}
	// A directory is not a file, and answering with one would let a caller
	// download the name of a folder.
	if info.IsDir() {
		return Info{}, ErrNotFound
	}
	return Info{
		Key:         storedPath,
		Size:        info.Size(),
		ContentType: TypeOf(storedPath),
		ModifiedAt:  info.ModTime(),
	}, nil
}

// Exists reports whether the path is a file that is there.
func (a *LocalFilesystemAdapter) Exists(_ context.Context, storedPath string) (bool, error) {
	name, err := a.file(storedPath)
	if err != nil {
		return false, err
	}
	info, err := a.dir.Stat(name)
	if errors.Is(err, fs.ErrNotExist) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("filesystem: checking %s: %w", storedPath, err)
	}
	return !info.IsDir(), nil
}

// Delete removes a file. Removing what is not there is not an error: the caller
// wanted it gone, and it is.
func (a *LocalFilesystemAdapter) Delete(_ context.Context, storedPath string) error {
	name, err := a.file(storedPath)
	if err != nil {
		return err
	}
	if err := a.dir.Remove(name); err != nil && !errors.Is(err, fs.ErrNotExist) {
		return fmt.Errorf("filesystem: deleting %s: %w", storedPath, err)
	}
	return nil
}

// List returns the stored paths under a prefix.
//
// The prefix is a string match and not a directory: "invoices/" and "invoices"
// select different sets, and [Disk] is what decides which one a caller meant.
// The walk starts at the deepest directory the prefix names, so listing one
// tenant does not read another tenant's directory entries.
func (a *LocalFilesystemAdapter) List(_ context.Context, prefix string) ([]string, error) {
	// The directory the walk starts in, resolved through the same containment
	// check every other operation goes through rather than joined by hand.
	// Joining by hand was the one path into this package that never verified the
	// result stayed under the root -- so List was the operation that would have
	// walked out of it. Found by audit.
	dir := prefix
	if !strings.HasSuffix(dir, "/") {
		if i := strings.LastIndex(dir, "/"); i >= 0 {
			dir = dir[:i+1]
		} else {
			dir = ""
		}
	}
	start, err := a.resolve(dir)
	if err != nil {
		return nil, err
	}

	// The walk goes over the root's own file system rather than over the
	// process's, so the directory it starts in is resolved through the root and
	// no entry under it can be reached by leaving it.
	var out []string
	err = fs.WalkDir(a.dir.FS(), start, func(p string, entry fs.DirEntry, err error) error {
		if err != nil {
			if errors.Is(err, fs.ErrNotExist) {
				// A prefix nothing was ever written under has no directory, and
				// an empty list is the right answer rather than an error.
				return fs.SkipAll
			}
			return err
		}
		// An upload in flight is not a stored object yet, and the name is
		// matched on the base rather than the whole path: a directory named
		// after the prefix would otherwise hide everything under it.
		if entry.IsDir() || strings.HasPrefix(entry.Name(), partialPrefix) {
			return nil
		}
		// Neither is anything that is not a plain file. Put stores regular
		// files and nothing else, so a symbolic link in here was put on the
		// disk by something other than this adapter, and answering with its
		// name hands the caller a key for whatever it points at.
		if !entry.Type().IsRegular() {
			return nil
		}
		if strings.HasPrefix(p, prefix) {
			out = append(out, p)
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("filesystem: listing %s: %w", prefix, err)
	}
	return out, nil
}

// file resolves a stored path to a file name under the root, and refuses one
// that escapes.
//
// The escape check happens twice: [CleanKey] rejects "../" in the key, and this
// rejects a resolved path outside the root. Two checks because the first is
// about the key and the second is about the filesystem -- an Adapter is a public
// interface, so this one is also what stands between a hand-built path and the
// rest of the disk.
func (a *LocalFilesystemAdapter) file(storedPath string) (string, error) {
	name, err := a.resolve(storedPath)
	if err != nil {
		return "", err
	}
	// The root itself is a directory, and every operation here is about a file.
	if name == "." {
		return "", ErrBadKey
	}
	return name, nil
}

// resolve turns a stored path into a name to be opened against the root, and
// refuses one that could not be under it. "." is the root itself, which is what
// makes an empty prefix mean "everything".
//
// The name is relative and slash-separated because that is what the open root
// takes, and what it takes is the point: a name resolved against the root is
// checked component by component as the syscall walks it, so a symbolic link
// pointing out of the root stops the operation instead of redirecting it.
//
// The string comparison below stays, and it is not the containment check any
// more -- it is what turns a key carrying "../" into [ErrBadKey] rather than
// into a syscall error. Comparing strings cannot be the containment check: it
// answers about the key, and the kernel answers about the links, and between
// the two answers is where a link planted after the check gets followed.
func (a *LocalFilesystemAdapter) resolve(storedPath string) (string, error) {
	// A NUL byte truncates a path in every syscall that takes one, so a path
	// carrying it can name a different file than it appears to.
	if strings.ContainsRune(storedPath, 0) {
		return "", ErrBadKey
	}
	full := filepath.Join(a.root, filepath.FromSlash(storedPath))
	if full != a.root && !strings.HasPrefix(full, a.root+string(filepath.Separator)) {
		return "", ErrBadKey
	}
	name, err := filepath.Rel(a.root, full)
	if err != nil {
		return "", ErrBadKey
	}
	return filepath.ToSlash(name), nil
}
