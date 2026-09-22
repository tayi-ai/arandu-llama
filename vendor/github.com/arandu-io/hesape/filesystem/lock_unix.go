//go:build unix

package filesystem

import (
	"errors"
	"fmt"
	"os"
	"syscall"
)

// lockFile takes an advisory lock on an open file.
//
// flock and not fcntl: fcntl locks are released when ANY descriptor on the file
// is closed in the process, so an unrelated open-and-close elsewhere in the same
// binary silently drops the lock this one is holding. flock is bound to the open
// file description instead, which is what makes it survivable in a server that
// touches the same path from two goroutines.
func lockFile(file *os.File, exclusive, block bool) error {
	how := syscall.LOCK_SH
	if exclusive {
		how = syscall.LOCK_EX
	}
	if !block {
		how |= syscall.LOCK_NB
	}
	for {
		err := syscall.Flock(int(file.Fd()), how)
		switch {
		case err == nil:
			return nil
		case errors.Is(err, syscall.EINTR):
			// A signal arriving during a blocking wait is not a failure to lock.
			// Without this retry, a process that takes any signal at all -- which
			// under the Go runtime is every process, because preemption uses one
			// -- would report the file as locked by somebody else.
			continue
		case errors.Is(err, syscall.EWOULDBLOCK):
			return fmt.Errorf("%w: %s", ErrLockTimeout, file.Name())
		default:
			return fmt.Errorf("filesystem: locking %s: %w", file.Name(), err)
		}
	}
}

// unlockFile drops the advisory lock.
func unlockFile(file *os.File) error {
	for {
		err := syscall.Flock(int(file.Fd()), syscall.LOCK_UN)
		if errors.Is(err, syscall.EINTR) {
			continue
		}
		if err != nil {
			return fmt.Errorf("filesystem: unlocking %s: %w", file.Name(), err)
		}
		return nil
	}
}
