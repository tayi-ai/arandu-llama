//go:build !linux && !darwin

package pipeline

import "os"

const durableLockSupported = false

func openDurableRegular(*os.Root, string, int, os.FileMode) (*os.File, error) {
	return nil, ErrDurableConfiguration
}
func acquireDurableLock(*os.Root) (*os.File, error) {
	return nil, ErrDurableConfiguration
}
