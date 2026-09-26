package native

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	decoder "github.com/tayi-ai/arandu-llama/training/decoder"
)

// ExperimentNativeForwardTrace writes only copied activation statistics and hashes.
// It does not retain tensors, change precision, or authorize model execution.
// The admitted job owns the existing private directory. The file is exclusive,
// limited to 128 records and 1 MiB, and synced after every observation.
type ExperimentNativeForwardTrace struct {
	mu      sync.Mutex
	file    *os.File
	records int
	bytes   int
}

// ExperimentNativeForwardTraceReceipt identifies a closed, possibly partial trace.
// Its presence does not mean that calibration passed.
type ExperimentNativeForwardTraceReceipt struct {
	Name    string `json:"name"`
	SHA256  string `json:"sha256"`
	Records int    `json:"records"`
	Bytes   int    `json:"bytes"`
}

// NewExperimentNativeForwardTrace reserves one of the eight calibration trace files.
// No directory or earlier evidence is created, replaced, or removed here.
func NewExperimentNativeForwardTrace(directory string, index int) (*ExperimentNativeForwardTrace, error) {
	if index < 0 || index >= 8 || !filepath.IsAbs(directory) {
		return nil, errors.New("experiment native: invalid forward trace destination")
	}
	info, err := os.Lstat(directory)
	if err != nil || !info.IsDir() || info.Mode().Perm()&0077 != 0 {
		return nil, errors.New("experiment native: forward trace requires the private job directory")
	}
	file, err := os.OpenFile(filepath.Join(directory, fmt.Sprintf("forward-%02d.jsonl", index)), os.O_RDWR|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return nil, errors.New("experiment native: forward trace already exists or cannot be reserved")
	}
	return &ExperimentNativeForwardTrace{file: file}, nil
}

// Observe synchronously preserves a single bounded module observation.
func (trace *ExperimentNativeForwardTrace) Observe(ctx context.Context, observation decoder.ForwardObservation) error {
	if trace == nil || ctx == nil {
		return errors.New("experiment native: missing forward trace context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	trace.mu.Lock()
	defer trace.mu.Unlock()
	if trace.file == nil || trace.records >= 128 {
		return errors.New("experiment native: forward trace closed or record budget exhausted")
	}
	body, err := json.Marshal(struct {
		At          time.Time                  `json:"at"`
		Observation decoder.ForwardObservation `json:"observation"`
	}{time.Now().UTC(), observation})
	if err != nil || len(body) > 8192 || trace.bytes+len(body)+1 > 1<<20 {
		return errors.New("experiment native: forward observation exceeds the evidence budget")
	}
	body = append(body, '\n')
	n, err := trace.file.Write(body)
	trace.bytes += n
	if err != nil {
		return err
	}
	trace.records++
	return trace.file.Sync()
}

// Close preserves partial traces on failure and returns their content digest.
func (trace *ExperimentNativeForwardTrace) Close() (ExperimentNativeForwardTraceReceipt, error) {
	var receipt ExperimentNativeForwardTraceReceipt
	if trace == nil {
		return receipt, nil
	}
	trace.mu.Lock()
	defer trace.mu.Unlock()
	if trace.file == nil {
		return receipt, errors.New("experiment native: forward trace already closed")
	}
	file := trace.file
	trace.file = nil
	if err := file.Sync(); err != nil {
		return receipt, errors.Join(err, file.Close())
	}
	body := make([]byte, trace.bytes)
	if _, err := file.ReadAt(body, 0); err != nil && len(body) != 0 {
		return receipt, errors.Join(err, file.Close())
	}
	digest := sha256.Sum256(body)
	receipt = ExperimentNativeForwardTraceReceipt{filepath.Base(file.Name()), hex.EncodeToString(digest[:]), trace.records, trace.bytes}
	if err := file.Close(); err != nil {
		return receipt, err
	}
	// This independent receipt also survives a failed or cancelled calibration.
	// It certifies only the trace bytes, never mathematical acceptance.
	return receipt, experimentNativeWriteJSON(file.Name()+".receipt.json", receipt)
}
