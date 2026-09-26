package decoder_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/decoder"
	"github.com/tayi-ai/arandu-llama/training/fusioncache"
	"github.com/tayi-ai/arandu-llama/training/pipeline"
)

func recoverySHA(body []byte) string {
	h := sha256.Sum256(body)
	return hex.EncodeToString(h[:])
}

func TestRecoveryBatchVerifiesBytesTargetsAndBounds(t *testing.T) {
	for _, mode := range []string{"valid", "hash", "length", "targets", "dataset", "student", "examples", "supervised", "budget", "working", "total-tokens", "mask", "token", "duplicate", "unknown", "trailing", "symlink", "cancel"} {
		t.Run(mode, func(t *testing.T) {
			model := fusioncache.ModelIdentity{Name: "student", Revision: strings.Repeat("a", 40), WeightsSHA256: strings.Repeat("b", 64), TokenizerSHA256: strings.Repeat("c", 64), TemplateSHA256: strings.Repeat("d", 64), RuntimeSHA256: strings.Repeat("e", 64), Vocabulary: 6}
			data := pipeline.ArtifactRef{ID: "recovery", SHA256: strings.Repeat("f", 64)}
			d := decoder.RecoveryBatchDocument{Version: 1, Dataset: data, Student: model, Aggregation: decoder.RecoveryTokenMean, SupervisedTokens: 3,
				Examples: []pipeline.Example{{ID: "long", DatasetDigest: data.SHA256, Tokens: []int64{1, 2, 3}, PromptTokens: 1}, {ID: "short", DatasetDigest: data.SHA256, Tokens: []int64{1, 4}, PromptTokens: 1}}}
			limits := decoder.RecoveryLimits{MaxBytes: 1 << 20, MaxExamples: 2, MaxTotalTokens: 5, MaxTokens: 3, WorkingBytes: 4 << 20}
			targets, err := d.TargetsDigest(limits)
			if err != nil {
				t.Fatal(err)
			}
			expected := decoder.RecoveryExpectation{Dataset: data, Student: model, Aggregation: decoder.RecoveryTokenMean, TargetsSHA256: targets, Examples: 2, SupervisedTokens: 3}
			switch mode {
			case "targets":
				expected.TargetsSHA256 = strings.Repeat("0", 64)
			case "dataset":
				expected.Dataset.SHA256 = strings.Repeat("0", 64)
			case "student":
				expected.Student.TokenizerSHA256 = strings.Repeat("0", 64)
			case "examples":
				expected.Examples = 1
			case "supervised":
				expected.SupervisedTokens = 2
			case "total-tokens":
				limits.MaxTotalTokens = 4
			case "working":
				limits.WorkingBytes = 1
			case "mask":
				d.Examples[0].PromptTokens = 0
			case "token":
				d.Examples[0].Tokens[2] = 6
			case "duplicate":
				d.Examples[1].ID = d.Examples[0].ID
			}
			body, err := json.Marshal(d)
			if err != nil {
				t.Fatal(err)
			}
			if mode == "unknown" {
				body = append([]byte(`{"unknown":true,`), body[1:]...)
			}
			if mode == "trailing" {
				body = append(body, []byte(` {}`)...)
			}
			path := filepath.Join(t.TempDir(), "batch.json")
			if err := os.WriteFile(path, body, 0600); err != nil {
				t.Fatal(err)
			}
			if mode == "symlink" {
				link := path + ".link"
				if err := os.Symlink(path, link); err != nil {
					t.Fatal(err)
				}
				path = link
			}
			digest, size := recoverySHA(body), int64(len(body))
			if mode == "hash" {
				digest = strings.Repeat("0", 64)
			}
			if mode == "length" {
				size++
			}
			if mode == "budget" {
				limits.MaxBytes = size - 1
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if mode == "cancel" {
				cancel()
			}
			batch, err := decoder.ReadRecoveryBatch(ctx, path, digest, size, expected, limits)
			if mode == "valid" {
				if err != nil || batch == nil {
					t.Fatal(err)
				}
			} else if err == nil || batch != nil {
				t.Fatalf("invalid batch escaped: %v", err)
			}
			if mode == "cancel" && !errors.Is(err, context.Canceled) {
				t.Fatal(err)
			}
		})
	}
}
