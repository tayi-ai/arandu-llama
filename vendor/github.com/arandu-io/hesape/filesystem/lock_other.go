//go:build !unix

package filesystem

import "os"

// lockFile refuses on a platform with no advisory lock this package speaks.
//
// It returns an error rather than pretending to lock. A no-op here would mean
// [Filesystem.Put] with lock true silently writes without one, and the corrupt
// file that follows would only appear under load, on the platform nobody tests.
func lockFile(_ *os.File, _, _ bool) error { return ErrNoFileLocking }

// unlockFile is the other half, and refuses for the same reason.
func unlockFile(_ *os.File) error { return ErrNoFileLocking }
