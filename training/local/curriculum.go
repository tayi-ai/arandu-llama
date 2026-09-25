//go:build libtorch && cgo

package local

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/tayi-ai/arandu-llama/training/ornith"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

const admittedAranduSHA = "da8d5054e9ab77848c0fd5b2637f0ea7d3bc9912f3ac9dc6ed687044a1e5b85c"

func curriculumPosition(data, previous string) ([]example, stepManifest, int, error) {
	file, err := os.Open(data)
	if err != nil {
		return nil, stepManifest{}, 0, err
	}
	defer file.Close()
	hash := sha256.New()
	decoder := json.NewDecoder(io.TeeReader(file, hash))
	rows := make([]example, 0, 103)
	seen := map[string]bool{}
	lastLength := 0
	for {
		var row example
		if err := decoder.Decode(&row); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, stepManifest{}, 0, err
		}
		if row.ID == "" || seen[row.ID] || len(row.InputIDs) < lastLength || len(row.InputIDs) != len(row.Labels) || row.PromptTokens < 1 || row.PromptTokens >= len(row.InputIDs) {
			return nil, stepManifest{}, 0, errors.New("curriculum order or geometry differs")
		}
		seen[row.ID] = true
		lastLength = len(row.InputIDs)
		rows = append(rows, row)
	}
	if hex.EncodeToString(hash.Sum(nil)) != admittedAranduSHA || len(rows) != 103 {
		return nil, stepManifest{}, 0, errors.New("admitted curriculum digest or count differs")
	}
	body, err := os.ReadFile(filepath.Join(previous, "manifest.json"))
	if err != nil {
		return nil, stepManifest{}, 0, err
	}
	var prior stepManifest
	if err := json.Unmarshal(body, &prior); err != nil {
		return nil, stepManifest{}, 0, err
	}
	for i, row := range rows {
		if row.ID == prior.ExampleID {
			if prior.Step != uint64(i+1) {
				return nil, stepManifest{}, 0, errors.New("checkpoint step differs from dataset cursor")
			}
			return rows, prior, i + 1, nil
		}
	}
	return nil, stepManifest{}, 0, errors.New("checkpoint example absent from admitted curriculum")
}

func nextCurriculumID(data, previous string, maxTokens int) (string, error) {
	rows, _, next, err := curriculumPosition(data, previous)
	if err != nil {
		return "", err
	}
	if next >= len(rows) || len(rows[next].InputIDs) > maxTokens {
		return "", errors.New("no next complete example within admitted token cap")
	}
	return rows[next].ID, nil
}

func runCurriculum(ctx context.Context, loaded *ornith.LoadedTextModel, data, previous, output string, maxTokens, maxSteps int) error {
	rows, prior, next, err := curriculumPosition(data, previous)
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
		tables, err := ornith.TextRotary(ctx, len(row.InputIDs), torch.MPSDevice(), rotaryDigest)
		if err != nil {
			return err
		}
		for layer := range loaded.Model.Layers {
			if layer%4 == 3 {
				loaded.Model.Layers[layer].Cosine, loaded.Model.Layers[layer].Sine = tables.Cosine, tables.Sine
			}
		}
		target := filepath.Join(output, fmt.Sprintf("step-%03d", prior.Step+1))
		stepErr := resumeNext(ctx, loaded, row, previous, target)
		for layer := range loaded.Model.Layers {
			if layer%4 == 3 {
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
