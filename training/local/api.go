package local

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

func readManifest(path string, c Config) (stepManifest, error) {
	var item stepManifest
	data, err := os.ReadFile(filepath.Join(path, "manifest.json"))
	if err != nil {
		return item, err
	}
	if err := json.Unmarshal(data, &item); err != nil || item.Step == 0 ||
		item.BaseRevision != c.Recipe.BaseRevision ||
		item.ExampleID == "" || item.AdapterFileSHA == "" || item.OptimizerFileSHA == "" {
		return stepManifest{}, errors.New("local training: checkpoint manifest differs")
	}
	if item.RecipeSHA256 != c.Recipe.Digest() && !(item.RecipeSHA256 == "" && c.AdmittedCheckpoints[item.Step] == fmtHash(data)) {
		return stepManifest{}, errors.New("local training: checkpoint recipe binding differs")
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
	owned, err := c.snapshot()
	if err != nil {
		return Progress{}, err
	}
	c = owned
	rows, base, _, err := curriculumPosition(c.DataPath, c.InitialCheckpoint, c)
	if err != nil {
		return Progress{}, err
	}
	base, err = readManifest(c.InitialCheckpoint, c)
	if err != nil {
		return Progress{}, err
	}
	current := Progress{Step: base.Step, ExampleID: base.ExampleID, Checkpoint: c.InitialCheckpoint}
	gap := false
	for i := int(base.Step); i < len(rows); i++ {
		path := filepath.Join(c.CheckpointRoot, fmt.Sprintf("step-%03d", i+1))
		info, err := os.Lstat(path)
		if errors.Is(err, os.ErrNotExist) {
			gap = true
			continue
		}
		if err != nil {
			return Progress{}, err
		}
		if gap {
			return Progress{}, errors.New("local training: checkpoint gap precedes a durable step")
		}
		if !info.IsDir() {
			return Progress{}, errors.New("local training: checkpoint directory differs")
		}
		item, err := readManifest(path, c)
		if err != nil || item.Step != uint64(i+1) || item.ExampleID != rows[i].ID {
			return Progress{}, errors.New("local training: checkpoint sequence differs")
		}
		current = Progress{Step: item.Step, ExampleID: item.ExampleID, Checkpoint: path}
	}
	return current, nil
}
