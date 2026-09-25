//go:build libtorch && cgo

package local

import (
	"os"
	"path/filepath"
	"testing"
)

func TestPinnedLocalCheckpointReconciles(t *testing.T) {
	root := os.Getenv("TAYI_LOCAL_TEST_ROOT")
	if root == "" {
		t.Skip("local checkpoint fixture is not configured")
	}
	base := filepath.Join(root, "runtime", "arasa-local-train-20260925", "bounded-1024")
	config := Config{
		BundleDir: filepath.Join(root, "runtime", "native-training-20260919", "release-canary-v21-r18", "arasa-native-v1"),
		ModelDir: filepath.Join(root, "runtime", "arasa-local-train-20260925", "model"),
		DataPath: filepath.Join(root, "runtime", "master-20260923", "arandu-ornith-v1", "tokenized.jsonl"),
		CheckpointRoot: base, InitialCheckpoint: filepath.Join(base, "step-005"),
		MaxTokens: 1024, MaxSteps: 14,
	}
	got, err := LatestCheckpoint(config)
	if err != nil {
		t.Fatal(err)
	}
	if got.Step < 5 || got.Checkpoint == "" || got.ExampleID == "" {
		t.Fatalf("incomplete checkpoint: %+v", got)
	}
}
