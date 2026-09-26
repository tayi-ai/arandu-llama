//go:build libtorch && cgo

package local

import (
	"context"
	"errors"
	"fmt"
	"github.com/tayi-ai/arandu-llama/training/decoder"
	"github.com/tayi-ai/arandu-llama/training/torch"
	"path/filepath"
)

func runCurriculum(ctx context.Context, loaded *decoder.LoadedTextModel, data, previous, output string, maxTokens, maxSteps int, c Config) error {
	rows, prior, next, err := curriculumPosition(data, previous, c)
	if err != nil {
		return err
	}
	completed := 0
	for i := next; i < len(rows) && completed < maxSteps; i++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		if len(rows[i].InputIDs) > maxTokens {
			break
		}
		row, err := selectedID(data, rows[i].ID)
		if err != nil {
			return err
		}
		tables, err := decoder.TextRotary(ctx, len(row.InputIDs), torch.MPSDevice(), c.Recipe.Rotary)
		if err != nil {
			return err
		}
		for layer := range loaded.Model.Layers {
			if loaded.Model.Layers[layer].Weights.Full != nil {
				loaded.Model.Layers[layer].Cosine, loaded.Model.Layers[layer].Sine = tables.Cosine, tables.Sine
			}
		}
		target := filepath.Join(output, fmt.Sprintf("step-%03d", prior.Step+1))
		stepErr := resumeNext(ctx, loaded, row, previous, target, c.Recipe)
		for layer := range loaded.Model.Layers {
			if loaded.Model.Layers[layer].Weights.Full != nil {
				loaded.Model.Layers[layer].Cosine, loaded.Model.Layers[layer].Sine = nil, nil
			}
		}
		closeErr := tables.Close()
		if err := errors.Join(stepErr, closeErr); err != nil {
			return err
		}
		previous = target
		prior.Step++
		completed++
	}
	if completed == 0 {
		return errors.New("no admitted curriculum step was run")
	}
	fmt.Printf("phase=bounded_curriculum_complete steps=%d last_step=%d max_tokens=%d\n", completed, prior.Step, maxTokens)
	return nil
}
