//go:build !libtorch || !cgo

package local

import "context"

// Run refuses native calculation without the required backend.
func Run(context.Context, Config) (Progress, error) { return Progress{}, ErrUnavailable }
