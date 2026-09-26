//go:build linux || darwin

package pipeline

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"syscall"
)

const durableLockSupported = true

func openDurableRegular(root *os.Root, name string, flag int, mode os.FileMode) (*os.File, error) {
	prefix := ""
	for _, part := range strings.Split(name, string(filepath.Separator)) {
		prefix = filepath.Join(prefix, part)
		info, err := root.Lstat(prefix)
		if errors.Is(err, os.ErrNotExist) && prefix == name && flag&os.O_CREATE != 0 {
			break
		}
		if err != nil {
			return nil, err
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return nil, ErrDurableReceipt
		}
	}
	file, err := root.OpenFile(name, flag|syscall.O_NOFOLLOW|syscall.O_NONBLOCK, mode)
	if err != nil {
		return nil, err
	}
	info, err := file.Stat()
	if err != nil || !info.Mode().IsRegular() {
		file.Close()
		return nil, ErrDurableReceipt
	}
	current, err := root.Lstat(name)
	if err != nil || current.Mode()&os.ModeSymlink != 0 || !os.SameFile(info, current) {
		file.Close()
		return nil, ErrDurableReceipt
	}
	return file, nil
}

func acquireDurableLock(root *os.Root) (*os.File, error) {
	file, err := openDurableRegular(root, "execution.lock", os.O_CREATE|os.O_RDWR, 0600)
	if err != nil {
		return nil, err
	}
	if err := syscall.Flock(int(file.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		file.Close()
		if errors.Is(err, syscall.EWOULDBLOCK) || errors.Is(err, syscall.EAGAIN) {
			return nil, ErrDurableBusy
		}
		return nil, err
	}
	// The lock inode is permanent; never unlink it while another process can open it.
	if err := errors.Join(file.Sync(), syncDirectory(root)); err != nil {
		file.Close()
		return nil, err
	}
	return file, nil
}
