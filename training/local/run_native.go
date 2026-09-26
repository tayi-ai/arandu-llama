//go:build libtorch && cgo

package local

import (
	"context"
	"errors"
)

// Run resumes one bounded delivery from its last verified checkpoint.
// A redelivery after a committed step never repeats that example.
func Run(ctx context.Context, c Config) (Progress, error) {
	if ctx == nil {
		return Progress{}, errors.New("local training: context required")
	}
	if err := ctx.Err(); err != nil {
		return Progress{}, err
	}
	owned, err := c.snapshot()
	if err != nil {
		return Progress{}, err
	}
	c = owned
	progress, err := LatestCheckpoint(c)
	if err != nil {
		return Progress{}, err
	}
	_, base, _, err := curriculumPosition(c.DataPath, c.InitialCheckpoint, c)
	if err != nil {
		return Progress{}, err
	}
	remaining := int(base.Step) + c.MaxSteps - int(progress.Step)
	if remaining <= 0 {
		return progress, nil
	}
	rows, _, next, err := curriculumPosition(c.DataPath, progress.Checkpoint, c)
	if err != nil {
		return Progress{}, err
	}
	if next >= len(rows) || len(rows[next].InputIDs) > c.MaxTokens {
		return progress, nil
	}
	if err := run(ctx, c.BundleDir, c.ModelDir, c.DataPath, c.CheckpointRoot,
		progress.Checkpoint, "", c.MaxTokens, remaining, c, nil); err != nil {
		return Progress{}, err
	}
	return LatestCheckpoint(c)
}

func runStageDelivery(ctx context.Context, c Config, hooks *stepHooks) error {
	return run(ctx, c.BundleDir, c.ModelDir, c.DataPath, c.CheckpointRoot, c.InitialCheckpoint, "", c.MaxTokens, c.MaxSteps, c, hooks)
}
