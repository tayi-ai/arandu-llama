package filesystem

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"path/filepath"

	"github.com/arandu-io/hesape/auth"
)

// DirectoryAware is the optional half of an [Adapter] whose store has real
// directories.
//
// A directory on disk exists whether or not anything is in it. In an object
// store there are no directories at all: a key with slashes in it is one string,
// and "the invoices folder" is a prefix that exists exactly as long as a key
// under it does. Both readings are correct for their store, and this interface
// is what lets [Disk] give the same answer on either -- see [Disk.MakeDirectory]
// for what "the same answer" means when there is nothing to make.
type DirectoryAware interface {
	// MakeDirectory creates the directory and its parents.
	MakeDirectory(ctx context.Context, storedPath string) error
	// DirectoryExists reports whether the directory is there.
	DirectoryExists(ctx context.Context, storedPath string) (bool, error)
}

// FileExists reports whether the key is there for this tenant.
//
// It is [Disk.Exists] under a second name. Both are here because the pair with
// [Disk.DirectoryExists] is what makes the question unambiguous at a call site
// that deals in both.
func (d *Disk) FileExists(ctx context.Context, g auth.Grant, key string) (bool, error) {
	return d.Exists(ctx, g, key)
}

// FileMissing reports whether the key is not there for this tenant.
func (d *Disk) FileMissing(ctx context.Context, g auth.Grant, key string) (bool, error) {
	return d.Missing(ctx, g, key)
}

// DirectoryExists reports whether a directory is there for this tenant.
//
// On a driver with real directories it is the driver's answer. On one without,
// it is whether the tenant has any key under that prefix -- which is what a
// directory is in an object store, and the only reading under which the same
// module behaves the same way on both.
func (d *Disk) DirectoryExists(ctx context.Context, g auth.Grant, directory string) (bool, error) {
	full, err := prefixPath(g, asDirectory(directory))
	if err != nil {
		return false, err
	}
	if aware, ok := d.adapter.(DirectoryAware); ok {
		ok, err := aware.DirectoryExists(ctx, full)
		if err != nil {
			return false, d.wrap("directory exists", directory, err)
		}
		return ok, nil
	}
	paths, err := d.adapter.List(ctx, full)
	if err != nil {
		return false, d.wrap("directory exists", directory, err)
	}
	return len(paths) > 0, nil
}

// DirectoryMissing is [Disk.DirectoryExists] inverted, and it exists for the
// reason [Disk.Missing] does: `if !ok` after a call that also returns an error
// is where a failed lookup silently becomes "it is not there".
func (d *Disk) DirectoryMissing(ctx context.Context, g auth.Grant, directory string) (bool, error) {
	ok, err := d.DirectoryExists(ctx, g, directory)
	if err != nil {
		return false, err
	}
	return !ok, nil
}

// MakeDirectory creates a directory for this tenant.
//
// On a driver with real directories it makes one. On an object store it does
// nothing and reports success, because there is nothing to make: the directory
// exists as soon as a key under it does, and it is not there before that no
// matter what this call did. The alternative -- writing an empty marker object
// to stand for a folder -- puts a key in every listing that [Disk.Get] then
// answers [ErrNotFound] for, which is a file that exists and cannot be read.
func (d *Disk) MakeDirectory(ctx context.Context, g auth.Grant, directory string) error {
	full, err := prefixPath(g, asDirectory(directory))
	if err != nil {
		return err
	}
	aware, ok := d.adapter.(DirectoryAware)
	if !ok {
		return nil
	}
	if err := aware.MakeDirectory(ctx, full); err != nil {
		return d.wrap("make directory", directory, err)
	}
	return nil
}

var _ DirectoryAware = (*LocalFilesystemAdapter)(nil)

// MakeDirectory creates the directory and its parents under the root.
func (a *LocalFilesystemAdapter) MakeDirectory(_ context.Context, storedPath string) error {
	name, err := a.resolve(storedPath)
	if err != nil {
		return err
	}
	if err := a.dir.MkdirAll(name, 0o755); err != nil {
		return fmt.Errorf("filesystem: creating %s: %w", storedPath, err)
	}
	return nil
}

// DirectoryExists reports whether the path is a directory under the root.
func (a *LocalFilesystemAdapter) DirectoryExists(_ context.Context, storedPath string) (bool, error) {
	name, err := a.resolve(storedPath)
	if err != nil {
		return false, err
	}
	info, err := a.dir.Stat(name)
	if errors.Is(err, fs.ErrNotExist) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("filesystem: reading %s: %w", storedPath, err)
	}
	return info.IsDir(), nil
}

// Pather is the optional half of an [Adapter] that can name a stored file on
// the machine the process is running on.
//
// It exists for the one thing a stream cannot do: handing a path to another
// program -- a thumbnailer, a virus scanner, ffmpeg. An object store has no
// such path, and the caller has to download first, which is why this is
// optional and why [Disk.Path] says so rather than inventing one.
type Pather interface {
	Path(storedPath string) (string, error)
}

// ErrNoPath is returned by [Disk.Path] on a driver whose files are not on this
// machine.
var ErrNoPath = errors.New("filesystem: this driver's files have no path on this machine")

// Path returns where a key is stored on this machine.
//
// It carries the tenant prefix, because that is what the path IS -- this is the
// one method that hands one out, and it does so only to a caller that already
// holds a Grant for that key. Do not build another key out of it: the way to
// reach a second file is a second [Key] call, which checks the Grant again.
func (d *Disk) Path(g auth.Grant, key string) (string, error) {
	full, err := Key(g, key)
	if err != nil {
		return "", err
	}
	pather, ok := d.adapter.(Pather)
	if !ok {
		return "", fmt.Errorf("%w: disk %q", ErrNoPath, d.name)
	}
	path, err := pather.Path(full)
	if err != nil {
		return "", d.wrap("path", key, err)
	}
	return path, nil
}

var _ Pather = (*LocalFilesystemAdapter)(nil)

// Path returns the absolute path of a stored path under the root.
//
// The name is walked through the open root before it is joined, so a directory
// component that is a symbolic link out of the root is refused rather than
// handed to whatever program the caller was about to run. A link as the last
// component is refused for the same reason: this adapter stores regular files
// and nothing else, so a link is not a stored object, and what the caller would
// open is whatever it points at.
//
// A name nothing is stored under yet is still answered. The string is the
// answer to "where would this go", and the caller that creates it holds the
// only moment at which the answer could be checked again.
func (a *LocalFilesystemAdapter) Path(storedPath string) (string, error) {
	name, err := a.file(storedPath)
	if err != nil {
		return "", err
	}
	switch info, err := a.dir.Lstat(name); {
	case errors.Is(err, fs.ErrNotExist):
	case err != nil:
		return "", fmt.Errorf("filesystem: reading %s: %w", storedPath, err)
	case info.Mode()&fs.ModeSymlink != 0:
		return "", ErrBadKey
	}
	return filepath.Join(a.root, filepath.FromSlash(name)), nil
}
