package native_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	decoder "github.com/tayi-ai/arandu-llama/training/decoder"
	services "github.com/tayi-ai/arandu-llama/training/native"
)

func TestForwardTracePreservesPartialEvidenceAndRejectsOverwrite(t *testing.T) {
	directory := t.TempDir()
	if err := os.Chmod(directory, 0700); err != nil {
		t.Fatal(err)
	}
	trace, err := services.NewExperimentNativeForwardTrace(directory, 0)
	if err != nil {
		t.Fatal(err)
	}
	observation := decoder.ForwardObservation{Stage: decoder.StageEmbedding, Layer: -1}
	if err := trace.Observe(context.Background(), observation); err != nil {
		t.Fatal(err)
	}
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := trace.Observe(cancelled, observation); err == nil {
		t.Fatal("cancelled observation was written")
	}
	receipt, err := trace.Close()
	if err != nil {
		t.Fatal(err)
	}
	body, err := os.ReadFile(filepath.Join(directory, receipt.Name))
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(body)
	if receipt.Records != 1 || receipt.Bytes != len(body) || receipt.SHA256 != hex.EncodeToString(digest[:]) {
		t.Fatal("partial trace receipt differs")
	}
	receiptBody, err := os.ReadFile(filepath.Join(directory, receipt.Name+".receipt.json"))
	if err != nil {
		t.Fatal("cancelled calibration trace lost its receipt:", err)
	}
	var persisted services.ExperimentNativeForwardTraceReceipt
	if json.Unmarshal(receiptBody, &persisted) != nil || persisted != receipt {
		t.Fatal("persisted partial receipt differs")
	}
	if _, err := services.NewExperimentNativeForwardTrace(directory, 0); err == nil {
		t.Fatal("existing evidence was replaced")
	}
	if err := trace.Observe(context.Background(), observation); err == nil {
		t.Fatal("closed trace accepted a record")
	}
	if _, err := trace.Close(); err == nil {
		t.Fatal("double close accepted")
	}
	info, err := os.Stat(filepath.Join(directory, receipt.Name))
	if err != nil || info.Mode().Perm() != 0600 {
		t.Fatal("trace file permissions differ")
	}
}

func TestForwardTraceEnforcesRecordBudgetAndPrivateDirectory(t *testing.T) {
	directory := t.TempDir()
	if err := os.Chmod(directory, 0755); err != nil {
		t.Fatal(err)
	}
	if _, err := services.NewExperimentNativeForwardTrace(directory, 0); err == nil {
		t.Fatal("public directory accepted")
	}
	if err := os.Chmod(directory, 0700); err != nil {
		t.Fatal(err)
	}
	for _, index := range []int{-1, 8} {
		if _, err := services.NewExperimentNativeForwardTrace(directory, index); err == nil {
			t.Fatal("invalid calibration index accepted")
		}
	}
	trace, err := services.NewExperimentNativeForwardTrace(directory, 7)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 128; i++ {
		if err := trace.Observe(context.Background(), decoder.ForwardObservation{}); err != nil {
			t.Fatal(err)
		}
	}
	if err := trace.Observe(context.Background(), decoder.ForwardObservation{}); err == nil {
		t.Fatal("record budget exceeded")
	}
	receipt, err := trace.Close()
	if err != nil || receipt.Records != 128 {
		t.Fatal("bounded trace was not preserved")
	}
}
