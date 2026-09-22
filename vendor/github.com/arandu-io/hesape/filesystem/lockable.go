package filesystem

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// ErrLockTimeout is returned when a lock was asked for without blocking and
// somebody else holds it.
var ErrLockTimeout = errors.New("filesystem: the file is locked by another process")

// ErrNoFileLocking is returned on a platform whose kernel this package has no
// advisory lock for. Every unix does; it is here so the type still compiles
// where one does not, instead of the whole collection failing to build.
var ErrNoFileLocking = errors.New("filesystem: file locking is not available on this platform")

// LockableFile is a file held open with an advisory lock around it.
//
// It exists for one job: two processes
// writing the same file -- a session, a cache entry, a compiled view -- where a
// reader must see either the old contents or the new ones and never half of
// each. [Filesystem.SharedGet] and [Filesystem.Put] with lock are the two
// callers, and between them they are the whole pattern.
//
// The lock is advisory, which means it only holds against somebody who also
// asks for it. That is not a weakness of this type: it is what flock is on every
// unix, and it is why the reading half has to take a shared lock rather than
// trusting the writer to be quick.
//
// It is not tenant storage. A path here is a path, with no prefix and no Grant
// -- see [Filesystem] for that line, and [Disk] for the other side of it.
type LockableFile struct {
	path   string
	file   *os.File
	locked bool
}

// NewLockableFile opens a file, creating the directory that holds it when the
// mode asks for creation.
//
// mode is the flag set os.OpenFile takes: os.O_RDONLY to read, and
// os.O_CREATE|os.O_RDWR to write.
func NewLockableFile(path string, mode int) (*LockableFile, error) {
	if mode&(os.O_CREATE) != 0 {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			return nil, fmt.Errorf("filesystem: creating the directory for %s: %w", path, err)
		}
	}
	file, err := os.OpenFile(path, mode, 0o644)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, fmt.Errorf("filesystem: %s: %w", path, ErrNotFound)
		}
		return nil, fmt.Errorf("filesystem: opening %s: %w", path, err)
	}
	return &LockableFile{path: path, file: file}, nil
}

// File returns the open file underneath.
//
// It is here so a caller that already holds the lock can hand the handle to
// io.Copy rather than reading the whole file into memory. Locking stays this
// type's job: the handle is the same one Read and Write use.
func (f *LockableFile) File() *os.File { return f.file }

// Read returns up to length bytes from the current position. A length of zero
// or less reads to the end of the file.
func (f *LockableFile) Read(length int64) ([]byte, error) {
	if length <= 0 {
		body, err := io.ReadAll(f.file)
		if err != nil {
			return nil, fmt.Errorf("filesystem: reading %s: %w", f.path, err)
		}
		return body, nil
	}
	body := make([]byte, length)
	n, err := io.ReadFull(f.file, body)
	if err != nil && !errors.Is(err, io.EOF) && !errors.Is(err, io.ErrUnexpectedEOF) {
		return nil, fmt.Errorf("filesystem: reading %s: %w", f.path, err)
	}
	return body[:n], nil
}

// Size returns how many bytes the file holds.
func (f *LockableFile) Size() (int64, error) {
	info, err := f.file.Stat()
	if err != nil {
		return 0, fmt.Errorf("filesystem: reading %s: %w", f.path, err)
	}
	return info.Size(), nil
}

// Write appends the contents at the current position and flushes them.
func (f *LockableFile) Write(contents []byte) (int, error) {
	n, err := f.file.Write(contents)
	if err != nil {
		return n, fmt.Errorf("filesystem: writing %s: %w", f.path, err)
	}
	if err := f.file.Sync(); err != nil {
		return n, fmt.Errorf("filesystem: writing %s: %w", f.path, err)
	}
	return n, nil
}

// Truncate empties the file and rewinds to the start.
func (f *LockableFile) Truncate() error {
	if _, err := f.file.Seek(0, io.SeekStart); err != nil {
		return fmt.Errorf("filesystem: truncating %s: %w", f.path, err)
	}
	if err := f.file.Truncate(0); err != nil {
		return fmt.Errorf("filesystem: truncating %s: %w", f.path, err)
	}
	return nil
}

// GetSharedLock takes a read lock. Several readers may hold one at once, and no
// writer may hold an exclusive lock while any of them do.
//
// block waits for the lock; without it, a lock somebody else holds returns
// [ErrLockTimeout] straight away.
func (f *LockableFile) GetSharedLock(block bool) error {
	if err := lockFile(f.file, false, block); err != nil {
		return err
	}
	f.locked = true
	return nil
}

// GetExclusiveLock takes a write lock. Exactly one holder, and no shared lock
// alongside it.
func (f *LockableFile) GetExclusiveLock(block bool) error {
	if err := lockFile(f.file, true, block); err != nil {
		return err
	}
	f.locked = true
	return nil
}

// ReleaseLock drops whichever lock is held. Releasing none is not an error.
func (f *LockableFile) ReleaseLock() error {
	if !f.locked {
		return nil
	}
	if err := unlockFile(f.file); err != nil {
		return err
	}
	f.locked = false
	return nil
}

// Close releases the lock and closes the file.
//
// The lock first, and unconditionally: closing a descriptor drops the flock on
// every unix, but a caller reading this has to be able to see that the release
// happened rather than infer it from the kernel's behaviour.
func (f *LockableFile) Close() error {
	if err := f.ReleaseLock(); err != nil {
		_ = f.file.Close()
		return err
	}
	if err := f.file.Close(); err != nil {
		return fmt.Errorf("filesystem: closing %s: %w", f.path, err)
	}
	return nil
}
