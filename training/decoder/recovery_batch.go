package decoder

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"slices"
	"strings"

	"github.com/tayi-ai/arandu-llama/training/fusioncache"
	"github.com/tayi-ai/arandu-llama/training/pipeline"
)

// ErrRecovery reports refused causal recovery data or native admission.
var ErrRecovery = errors.New("decoder: causal recovery refused")

// RecoveryTokenMean identifies a mean over supervised tokens, not over examples.
const RecoveryTokenMean = "supervised-token-mean-v1"

// RecoveryLimits bounds encoded bytes, rows and tokens before native execution.
// MaxBytes bounds the input file, not the JSON decoder's complete heap footprint.
type RecoveryLimits struct {
	MaxBytes       int64
	MaxExamples    int
	MaxTotalTokens int64
	MaxTokens      int
	// WorkingBytes reserves encoded snapshots and copied token payloads. It is
	// additional to native StepWorkingBytes, not a cap on Go heap/JSON metadata.
	WorkingBytes int64
}

// RecoveryBatchDocument is the versioned, ordered causal input. Dataset identifies
// the original recovery dataset; its digest is distinct from the serialized batch
// artifact digest. Tokenization and templates are pinned in Student.
type RecoveryBatchDocument struct {
	Version          int                       `json:"version"`
	Dataset          pipeline.ArtifactRef      `json:"dataset"`
	Student          fusioncache.ModelIdentity `json:"student"`
	Aggregation      string                    `json:"aggregation"`
	Examples         []pipeline.Example        `json:"examples"`
	SupervisedTokens int64                     `json:"supervised_tokens"`
}

// RecoveryExpectation comes from an independent admitted manifest. TargetsSHA256
// pins the ordered examples, including their complete prefixes and prompt masks.
type RecoveryExpectation struct {
	Dataset          pipeline.ArtifactRef
	Student          fusioncache.ModelIdentity
	Aggregation      string
	TargetsSHA256    string
	Examples         int
	SupervisedTokens int64
}

// RecoveryBatch is an immutable byte-verified causal batch. Its zero value is not
// admitted. Byte integrity does not prove dataset quality or model qualification.
type RecoveryBatch struct {
	document RecoveryBatchDocument
	admitted bool
}

// TargetsDigest hashes the canonical ordered targets after validating all rows.
// A caller must independently freeze this digest before admitting a source.
func (d RecoveryBatchDocument) TargetsDigest(limits RecoveryLimits) (string, error) {
	if err := d.validate(limits); err != nil {
		return "", err
	}
	body, err := json.Marshal(d.Examples)
	if err != nil || int64(len(body)) > limits.MaxBytes {
		return "", errors.Join(ErrRecovery, err)
	}
	return assemblyHash(body), nil
}

func validRecoveryLimits(l RecoveryLimits) bool {
	return l.MaxBytes > 0 && l.MaxBytes <= 256<<20 && l.MaxExamples > 0 && l.MaxExamples <= 1<<20 &&
		l.MaxTotalTokens > 0 && l.MaxTotalTokens <= 1<<30 && l.MaxTokens >= 2 && l.MaxTokens <= 1<<20 &&
		l.WorkingBytes >= 3*l.MaxBytes+16*l.MaxTotalTokens
}

func validRecoveryStudent(s fusioncache.ModelIdentity) bool {
	if strings.TrimSpace(s.Name) == "" || len(s.Name) > 1024 || (len(s.Revision) != 40 && len(s.Revision) != 64) || s.Revision != strings.ToLower(s.Revision) || s.Vocabulary < 2 {
		return false
	}
	if _, err := hex.DecodeString(s.Revision); err != nil {
		return false
	}
	return validAssemblyHash(s.WeightsSHA256) && validAssemblyHash(s.TokenizerSHA256) && validAssemblyHash(s.TemplateSHA256) && validAssemblyHash(s.RuntimeSHA256)
}

func (e RecoveryExpectation) validate(l RecoveryLimits) error {
	if !validRecoveryLimits(l) || e.Dataset.ID == "" || len(e.Dataset.ID) > 1024 || !validAssemblyHash(e.Dataset.SHA256) ||
		!validRecoveryStudent(e.Student) || e.Aggregation != RecoveryTokenMean || !validAssemblyHash(e.TargetsSHA256) ||
		e.Examples < 1 || e.Examples > l.MaxExamples || e.SupervisedTokens < 1 || e.SupervisedTokens > l.MaxTotalTokens {
		return ErrRecovery
	}
	return nil
}

func (d RecoveryBatchDocument) validate(l RecoveryLimits) error {
	if !validRecoveryLimits(l) || d.Version != 1 || d.Dataset.ID == "" || len(d.Dataset.ID) > 1024 || !validAssemblyHash(d.Dataset.SHA256) ||
		!validRecoveryStudent(d.Student) || d.Aggregation != RecoveryTokenMean || len(d.Examples) < 1 || len(d.Examples) > l.MaxExamples {
		return ErrRecovery
	}
	var tokens, supervised int64
	seen := make(map[string]bool, len(d.Examples))
	for _, row := range d.Examples {
		if row.ID == "" || len(row.ID) > 1024 || seen[row.ID] || row.DatasetDigest != d.Dataset.SHA256 || len(row.Tokens) > l.MaxTokens ||
			row.PromptTokens < 1 || row.PromptTokens >= len(row.Tokens) || int64(len(row.Tokens)) > l.MaxTotalTokens-tokens {
			return ErrRecovery
		}
		seen[row.ID] = true
		tokens += int64(len(row.Tokens))
		supervised += int64(len(row.Tokens) - row.PromptTokens)
		for _, token := range row.Tokens {
			if token < 0 || token >= int64(d.Student.Vocabulary) {
				return ErrRecovery
			}
		}
	}
	if supervised != d.SupervisedTokens || supervised < 1 {
		return ErrRecovery
	}
	return nil
}

// ReadRecoveryBatch verifies regular-file identity, exact length and SHA while
// reading a bounded snapshot. Only that verified snapshot is decoded. Unknown
// fields, trailing documents and divergent independent expectations are refused.
func ReadRecoveryBatch(ctx context.Context, path, fileSHA256 string, size int64, expected RecoveryExpectation, limits RecoveryLimits) (batch *RecoveryBatch, err error) {
	if ctx == nil || !validAssemblyHash(fileSHA256) || expected.validate(limits) != nil || size < 1 || size > limits.MaxBytes {
		return nil, ErrRecovery
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	info, err := os.Lstat(path)
	if err != nil || !info.Mode().IsRegular() || info.Size() != size {
		return nil, errors.Join(ErrRecovery, err)
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() {
		err = errors.Join(err, f.Close())
		if err != nil {
			batch = nil
		}
	}()
	opened, err := f.Stat()
	if err != nil || !os.SameFile(info, opened) {
		return nil, errors.Join(ErrRecovery, err)
	}
	body := make([]byte, int(size))
	h := sha256.New()
	for offset := 0; offset < len(body); {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		end := min(offset+(64<<10), len(body))
		if _, err := io.ReadFull(f, body[offset:end]); err != nil {
			return nil, err
		}
		_, _ = h.Write(body[offset:end])
		offset = end
	}
	after, err := f.Stat()
	if err != nil || after.Size() != size || !after.ModTime().Equal(info.ModTime()) || hex.EncodeToString(h.Sum(nil)) != fileSHA256 {
		return nil, errors.Join(ErrRecovery, err)
	}
	var d RecoveryBatchDocument
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&d); err != nil {
		return nil, errors.Join(ErrRecovery, err)
	}
	if err := decoder.Decode(new(any)); !errors.Is(err, io.EOF) {
		return nil, ErrRecovery
	}
	digest, err := d.TargetsDigest(limits)
	if err != nil || digest != expected.TargetsSHA256 || d.Dataset != expected.Dataset || d.Student != expected.Student || d.Aggregation != expected.Aggregation ||
		len(d.Examples) != expected.Examples || d.SupervisedTokens != expected.SupervisedTokens {
		return nil, errors.Join(ErrRecovery, err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return &RecoveryBatch{document: d, admitted: true}, nil
}

func (b *RecoveryBatch) snapshot() *RecoveryBatch {
	if b == nil || !b.admitted {
		return nil
	}
	out := *b
	out.document.Examples = slices.Clone(b.document.Examples)
	for i := range out.document.Examples {
		out.document.Examples[i].Tokens = slices.Clone(out.document.Examples[i].Tokens)
	}
	return &out
}
