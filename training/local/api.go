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
)

// ErrUnavailable means the local native backend cannot run on this host.
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

func (c Config) validate() error {
	if !filepath.IsAbs(c.BundleDir) || !filepath.IsAbs(c.ModelDir) ||
		!filepath.IsAbs(c.DataPath) || !filepath.IsAbs(c.CheckpointRoot) ||
		!filepath.IsAbs(c.InitialCheckpoint) || c.MaxTokens < 2 || c.MaxTokens > 1024 ||
		c.MaxSteps < 1 || c.MaxSteps > 20 {
		return errors.New("local training: absolute paths and bounded work required")
	}
	return nil
}

func readManifest(path string) (stepManifest, error) {
	var item stepManifest
	data, err := os.ReadFile(filepath.Join(path, "manifest.json"))
	if err != nil {
		return item, err
	}
	if err := json.Unmarshal(data, &item); err != nil || item.Step == 0 ||
		item.BaseRevision != "489cb97981b8654bcfcf30ce1f94ed1b62e07b53" ||
		item.ExampleID == "" || item.AdapterFileSHA == "" || item.OptimizerFileSHA == "" {
		return stepManifest{}, errors.New("local training: checkpoint manifest differs")
	}
	for _, pair := range [][2]string{
		{"adapter_model.safetensors", item.AdapterFileSHA},
		{"optimizer_moments.safetensors", item.OptimizerFileSHA},
	} {
		file, err := os.Open(filepath.Join(path, pair[0]))
		if err != nil {
			return stepManifest{}, err
		}
		stat, statErr := file.Stat()
		if statErr != nil || !stat.Mode().IsRegular() {
			file.Close()
			return stepManifest{}, errors.New("local training: checkpoint file is not regular")
		}
		hash := sha256.New()
		_, copyErr := io.Copy(hash, file)
		closeErr := file.Close()
		if err := errors.Join(copyErr, closeErr); err != nil {
			return stepManifest{}, err
		}
		if hex.EncodeToString(hash.Sum(nil)) != pair[1] {
			return stepManifest{}, errors.New("local training: checkpoint hash differs")
		}
	}
	return item, nil
}

// LatestCheckpoint reconciles durable steps, refusing gaps and corrupt files.
func LatestCheckpoint(c Config) (Progress, error) {
	if err := c.validate(); err != nil {
		return Progress{}, err
	}
	rows, base, _, err := curriculumPosition(c.DataPath, c.InitialCheckpoint)
	if err != nil {
		return Progress{}, err
	}
	base, err = readManifest(c.InitialCheckpoint)
	if err != nil {
		return Progress{}, err
	}
	current := Progress{Step: base.Step, ExampleID: base.ExampleID, Checkpoint: c.InitialCheckpoint}
	for i := int(base.Step); i < len(rows); i++ {
		path := filepath.Join(c.CheckpointRoot, fmt.Sprintf("step-%03d", i+1))
		info, err := os.Lstat(path)
		if errors.Is(err, os.ErrNotExist) {
			break
		}
		if err != nil {
			return Progress{}, err
		}
		if !info.IsDir() {
			return Progress{}, errors.New("local training: checkpoint directory differs")
		}
		item, err := readManifest(path)
		if err != nil || item.Step != uint64(i+1) || item.ExampleID != rows[i].ID {
			return Progress{}, errors.New("local training: checkpoint sequence differs")
		}
		current = Progress{Step: item.Step, ExampleID: item.ExampleID, Checkpoint: path}
	}
	return current, nil
}

// Run resumes one bounded delivery from its last verified checkpoint.
// A redelivery after a committed step never repeats that example.
func Run(ctx context.Context, c Config) (Progress, error) {
	if ctx == nil {
		return Progress{}, errors.New("local training: context required")
	}
	if err := ctx.Err(); err != nil {
		return Progress{}, err
	}
	progress, err := LatestCheckpoint(c)
	if err != nil {
		return Progress{}, err
	}
	_, base, _, err := curriculumPosition(c.DataPath, c.InitialCheckpoint)
	if err != nil {
		return Progress{}, err
	}
	remaining := int(base.Step) + c.MaxSteps - int(progress.Step)
	if remaining <= 0 {
		return progress, nil
	}
	rows, _, next, err := curriculumPosition(c.DataPath, progress.Checkpoint)
	if err != nil {
		return Progress{}, err
	}
	if next >= len(rows) || len(rows[next].InputIDs) > c.MaxTokens {
		return progress, nil
	}
	if err := run(ctx, c.BundleDir, c.ModelDir, c.DataPath, c.CheckpointRoot,
		progress.Checkpoint, "", c.MaxTokens, remaining); err != nil {
		return Progress{}, err
	}
	return LatestCheckpoint(c)
}
