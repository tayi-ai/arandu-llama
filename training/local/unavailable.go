//go:build !libtorch || !cgo

package local

import (
	"context"
	"errors"
)

// ErrUnavailable refuses local training without the pinned native backend.
var ErrUnavailable = errors.New("local training: LibTorch backend unavailable")

// Config fixes one bounded, resumable local Ornith training delivery.
type Config struct {
	BundleDir         string
	ModelDir          string
	DataPath          string
	CheckpointRoot    string
	InitialCheckpoint string
	MaxTokens         int
	MaxSteps          int
}

// Progress identifies the latest complete checkpoint of this delivery.
type Progress struct {
	Step       uint64
	ExampleID  string
	Checkpoint string
}

// LatestCheckpoint refuses inspection without the native backend.
func LatestCheckpoint(Config) (Progress, error) { return Progress{}, ErrUnavailable }

// Run refuses training without the native backend.
func Run(context.Context, Config) (Progress, error) { return Progress{}, ErrUnavailable }
