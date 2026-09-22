package filesystem

import (
	"context"
	"errors"
	"fmt"
	"io/fs"

	"github.com/arandu-io/hesape/auth"
)

// The two visibilities, as the strings they are stored and compared as.
const (
	// VisibilityPublic is a file the store itself will hand to anybody who has
	// its address.
	VisibilityPublic = "public"
	// VisibilityPrivate is a file the store will not, which is the only correct
	// setting for anything a tenant uploaded.
	VisibilityPrivate = "private"
)

// ErrNoVisibility is returned by [Disk.GetVisibility] and [Disk.SetVisibility]
// on a driver with no such concept.
var ErrNoVisibility = errors.New("filesystem: this driver has no visibility to read or set")

// VisibilityAware is the optional half of an [Adapter] whose store has a notion
// of a file being world-readable: a unix mode on a directory, an ACL on an
// object store.
//
// It is an optional interface and not a method on [Adapter] for the reason
// [Presigner] is: a driver that has no such concept would otherwise carry a
// method returning an error nobody can act on. Adding it to Adapter would also
// break every driver already written against that interface, including the S3
// module.
//
// # Visibility is not authorization
//
// This is worth stating where somebody will read it before reaching for
// SetVisibility. Making a file public does not remove the Policy: every read
// still goes through [Disk], which still takes an [auth.Grant]. What
// it changes is whether the STORE will serve the same bytes to somebody who
// never came through the application at all -- which is why the default here is
// private and why a tenant's upload must stay that way.
type VisibilityAware interface {
	// Visibility returns [VisibilityPublic] or [VisibilityPrivate].
	Visibility(ctx context.Context, storedPath string) (string, error)
	// SetVisibility sets it.
	SetVisibility(ctx context.Context, storedPath, visibility string) error
}

// GetVisibility returns whether the store itself would hand this file to
// somebody who has not been through a Policy.
//
// It answers [ErrNoVisibility] on a driver with no such concept, which is not a
// failure of the call: a store that has no public mode has no visibility to
// report, and the honest answer is that rather than "private", which would read
// as a guarantee this package did not make.
func (d *Disk) GetVisibility(ctx context.Context, g auth.Grant, key string) (string, error) {
	full, err := Key(g, key)
	if err != nil {
		return "", err
	}
	aware, ok := d.adapter.(VisibilityAware)
	if !ok {
		return "", fmt.Errorf("%w: disk %q", ErrNoVisibility, d.name)
	}
	visibility, err := aware.Visibility(ctx, full)
	if err != nil {
		return "", d.wrap("visibility", key, err)
	}
	return visibility, nil
}

// SetVisibility sets it. See [VisibilityAware] for why this is not a way to
// authorize anything.
func (d *Disk) SetVisibility(ctx context.Context, g auth.Grant, key, visibility string) error {
	if visibility != VisibilityPublic && visibility != VisibilityPrivate {
		return fmt.Errorf("filesystem: %q is not a visibility: %q or %q", visibility, VisibilityPublic, VisibilityPrivate)
	}
	full, err := Key(g, key)
	if err != nil {
		return err
	}
	aware, ok := d.adapter.(VisibilityAware)
	if !ok {
		return fmt.Errorf("%w: disk %q", ErrNoVisibility, d.name)
	}
	if err := aware.SetVisibility(ctx, full, visibility); err != nil {
		return d.wrap("visibility", key, err)
	}
	return nil
}

// The modes a visibility is on a directory: what Flysystem's local adapter
// uses, and what a person reading `ls -l` expects to find.
const (
	publicFileMode  fs.FileMode = 0o644
	privateFileMode fs.FileMode = 0o600
)

var _ VisibilityAware = (*LocalFilesystemAdapter)(nil)

// Visibility reads the file's mode.
//
// Any read bit outside the owner's is public: a file group-readable but not
// world-readable is still reachable by somebody who is not the process, and
// calling that private would be the more dangerous of the two mistakes.
func (a *LocalFilesystemAdapter) Visibility(_ context.Context, storedPath string) (string, error) {
	name, err := a.file(storedPath)
	if err != nil {
		return "", err
	}
	info, err := a.dir.Stat(name)
	if errors.Is(err, fs.ErrNotExist) {
		return "", ErrNotFound
	}
	if err != nil {
		return "", fmt.Errorf("filesystem: reading %s: %w", storedPath, err)
	}
	if info.Mode().Perm()&0o044 != 0 {
		return VisibilityPublic, nil
	}
	return VisibilityPrivate, nil
}

// SetVisibility chmods the file.
func (a *LocalFilesystemAdapter) SetVisibility(_ context.Context, storedPath, visibility string) error {
	name, err := a.file(storedPath)
	if err != nil {
		return err
	}
	mode := privateFileMode
	if visibility == VisibilityPublic {
		mode = publicFileMode
	}
	if err := a.dir.Chmod(name, mode); err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return ErrNotFound
		}
		return fmt.Errorf("filesystem: chmod %s: %w", storedPath, err)
	}
	return nil
}
