package decoder

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/tayi-ai/arandu-llama/checkpoint"
)

func TestAdapterCheckpointBulkMemoryBudget(t *testing.T) {
	source := AdapterCheckpoint{MaxBytes: 200, MaxWorkingBytes: 472, Limits: checkpoint.Limits{MaxHeaderBytes: 64}}
	// File + header + three full payloads + two largest-tensor copies.
	if err := adapterWorkingBudget(200, 48, 32, source); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"one-byte-short", "file-too-big", "overflow", "invalid-largest", "short-file"} {
		t.Run(name, func(t *testing.T) {
			configuration, size, payload, largest := source, int64(200), int64(48), int64(32)
			switch name {
			case "one-byte-short":
				configuration.MaxWorkingBytes--
			case "file-too-big":
				configuration.MaxBytes--
			case "overflow":
				configuration.MaxBytes, configuration.MaxWorkingBytes = math.MaxInt64, math.MaxInt64
				size, payload = math.MaxInt64, math.MaxInt64
			case "invalid-largest":
				largest = payload + 1
			case "short-file":
				size = 7
			}
			if err := adapterWorkingBudget(size, payload, largest, configuration); !errors.Is(err, ErrAdapterCheckpoint) {
				t.Fatalf("invalid budget accepted: %v", err)
			}
		})
	}
}

func TestAdapterCheckpointFileSnapshotHasIndependentOwnership(t *testing.T) {
	path := filepath.Join(t.TempDir(), "adapter.safetensors")
	original := []byte("immutable checkpoint fixture")
	if err := os.WriteFile(path, original, 0600); err != nil {
		t.Fatal(err)
	}
	hash := sha256.Sum256(original)
	source := AdapterCheckpoint{Path: path, FileSHA256: hex.EncodeToString(hash[:]), MaxBytes: int64(len(original)), MaxWorkingBytes: 1024,
		Limits: checkpoint.Limits{MaxHeaderBytes: 32, MaxChunkBytes: 3}}
	snapshot, err := readAdapterSnapshot(context.Background(), source, 4, 4)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("mutated file after snapshot"), 0600); err != nil {
		t.Fatal(err)
	}
	if string(snapshot) != string(original) {
		t.Fatal("file mutation changed the immutable snapshot")
	}
	if _, err := readAdapterSnapshot(context.Background(), source, 4, 4); !errors.Is(err, ErrAdapterCheckpoint) {
		t.Fatalf("changed file identity accepted: %v", err)
	}
	source.Path = filepath.Dir(path)
	if _, err := readAdapterSnapshot(context.Background(), source, 4, 4); !errors.Is(err, ErrAdapterCheckpoint) {
		t.Fatalf("directory accepted as regular file: %v", err)
	}
	source.Path = filepath.Join(filepath.Dir(path), "symlink")
	if err := os.Symlink(path, source.Path); err != nil {
		t.Fatal(err)
	}
	if _, err := readAdapterSnapshot(context.Background(), source, 4, 4); !errors.Is(err, ErrAdapterCheckpoint) {
		t.Fatalf("symlink accepted as regular file: %v", err)
	}
}
