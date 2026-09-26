// Package teachers implements native, identity-bound teacher cache production.
package teachers

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"unsafe"

	llama "github.com/tayi-ai/arandu-llama"
	"github.com/tayi-ai/arandu-llama/training/fusioncache"
)

// NativeBindingEvidence is an externally qualified runtime-to-executable
// binding. Hashing a runtime artifact alone does not identify linked code. The
// executable digest is checked against the running process's executable file;
// externally loaded shared libraries still require independent qualification.
// This record makes no claim of compatibility with a different model runtime.
type NativeBindingEvidence struct {
	Backend          string `json:"backend"`
	CaptureVersion   string `json:"capture_version"`
	RuntimeSHA256    string `json:"runtime_sha256"`
	ExecutableSHA256 string `json:"executable_sha256"`
}

// NativeConfig selects one exact model and its externally admitted artifacts.
// Tokenizer/template files are hashed, not executed: gold token equivalence is
// the external Expectation's responsibility. Capture.Tensor is the fallback
// tensor for probability-only capture and must be result_norm. Feature tensors
// are selected exclusively by the frozen Expectation, one forward per tensor.
type NativeConfig struct {
	Identity              fusioncache.ModelIdentity
	ModelPath             string
	TokenizerPath         string
	TemplatePath          string
	RuntimePath           string
	BindingEvidencePath   string
	BindingEvidenceSHA256 string
	CaptureVersion        string
	GPULayers             int
	MainGPU               int
	LockWeights           bool
	Capture               llama.TeacherCaptureOptions
}

// NativeLimits bounds aggregate frozen inputs and host capture/cache buffers.
// Cache.MaxExamples, MaxPositions, MaxFeatureValues, MaxMappingPairs and MaxBytes
// apply across all expectations; Cache.MaxTokens remains a per-example bound.
// MaxResidentBytes is a conservative estimate of live host payloads, including
// configured native output/window caps and response copies. It is not measured
// resident memory or total memory admission: weights, KV, decoder workspaces,
// allocator/GC overhead and linked backend allocations are excluded.
type NativeLimits struct {
	Cache            fusioncache.Limits
	MaxTeachers      int
	MaxTotalTokens   int
	MaxResidentBytes int64
}

// NativeStats exposes completed ownership transitions and actual native calls.
// ResidentTeachers is always zero or one; failed captures count as calls.
type NativeStats struct {
	OpenedTeachers   int64
	ClosedTeachers   int64
	ResidentTeachers int
	NativeCaptures   int64
	CapturedExamples int64
}

type prefixRow struct{ example, row int }
type nativeJob struct {
	expected fusioncache.Expectation
	config   NativeConfig
	rows     map[string]prefixRow
}

// NativeFactory implements fusioncache.Factory with one resident teacher and
// frozen jobs. Close releases an active teacher and permanently closes factory.
// Native calls are synchronous; cancellation is checked around those calls and
// never returns while native work still runs.
type NativeFactory struct {
	mu       sync.Mutex
	jobs     map[fusioncache.ModelIdentity]*nativeJob
	active   *nativeProducer
	closed   bool
	closeErr error
	stats    NativeStats
}

var _ fusioncache.Factory = (*NativeFactory)(nil)

func rejected(message string) error {
	return fmt.Errorf("%w: native teacher: %s", fusioncache.ErrContract, message)
}

func validDigest(value string) bool {
	if len(value) != 64 {
		return false
	}
	decoded, err := hex.DecodeString(value)
	return err == nil && len(decoded) == sha256.Size && value == strings.ToLower(value)
}

// NewNativeFactory freezes and validates all jobs without opening weights.
// Each teacher identity has exactly one expectation and one config. Files are
// verified at Open, so construction neither attests nor executes a backend.
func NewNativeFactory(expectations []fusioncache.Expectation, configs []NativeConfig, limits NativeLimits) (*NativeFactory, error) {
	if limits.MaxTeachers <= 0 || limits.MaxTotalTokens <= 0 || limits.MaxResidentBytes <= 0 ||
		len(expectations) == 0 || len(expectations) > limits.MaxTeachers || len(configs) != len(expectations) {
		return nil, rejected("explicit bounds and one config per teacher required")
	}
	configByIdentity := make(map[fusioncache.ModelIdentity]NativeConfig, len(configs))
	for _, config := range configs {
		if _, exists := configByIdentity[config.Identity]; exists {
			return nil, rejected("duplicate teacher config")
		}
		configByIdentity[config.Identity] = config
	}
	factory := &NativeFactory{jobs: make(map[fusioncache.ModelIdentity]*nativeJob, len(expectations))}
	remainingExamples, remainingPositions := limits.Cache.MaxExamples, limits.Cache.MaxPositions
	remainingFeatures, remainingMappings, remainingTokens := limits.Cache.MaxFeatureValues, limits.Cache.MaxMappingPairs, limits.MaxTotalTokens
	remainingBytes := limits.Cache.MaxBytes
	for _, expected := range expectations {
		if err := fusioncache.ValidateExpectation(expected, limits.Cache); err != nil {
			return nil, err
		}
		config, exists := configByIdentity[expected.Teacher]
		if !exists || factory.jobs[expected.Teacher] != nil {
			return nil, rejected("missing config or duplicate teacher expectation")
		}
		if err := validateConfig(config, expected, limits); err != nil {
			return nil, err
		}
		if len(expected.Examples) > remainingExamples || len(expected.Mapping.Pairs) > remainingMappings {
			return nil, rejected("aggregate example or mapping bound exceeded")
		}
		remainingExamples -= len(expected.Examples)
		remainingMappings -= len(expected.Mapping.Pairs)
		featureWidth := 0
		for _, feature := range expected.Features {
			featureWidth += feature.Dimension // ValidateExpectation already bounds the sum.
		}
		for _, example := range expected.Examples {
			positions := len(example.TeacherTokens) - example.PromptTokens
			if len(example.TeacherTokens) > remainingTokens || positions > remainingPositions ||
				featureWidth > 0 && positions > remainingFeatures/featureWidth {
				return nil, rejected("aggregate token, position or feature bound exceeded")
			}
			remainingTokens -= len(example.TeacherTokens)
			remainingPositions -= positions
			remainingFeatures -= positions * featureWidth
		}
		// Bound raw components before JSON's temporary encoding allocation.
		if !rawFits(expected, config, remainingBytes) {
			return nil, rejected("aggregate frozen input byte bound exceeded")
		}
		encoded, err := json.Marshal(struct {
			Expected fusioncache.Expectation
			Config   NativeConfig
		}{expected, config})
		if err != nil || int64(len(encoded)) > remainingBytes {
			return nil, rejected("aggregate frozen input encoding bound exceeded")
		}
		remainingBytes -= int64(len(encoded))
		var frozen struct {
			Expected fusioncache.Expectation
			Config   NativeConfig
		}
		if err := json.Unmarshal(encoded, &frozen); err != nil {
			return nil, err
		}
		job := &nativeJob{expected: frozen.Expected, config: frozen.Config, rows: make(map[string]prefixRow)}
		for exampleIndex, example := range job.expected.Examples {
			for target := example.PromptTokens; target < len(example.TeacherTokens); target++ {
				digest := fusioncache.PrefixDigest(example.TeacherTokens[:target])
				// Shared gold prefixes refer to the first frozen example. Native
				// causality makes its row valid for every identical input prefix.
				if previous, exists := job.rows[digest]; exists {
					prior := job.expected.Examples[previous.example]
					end := prior.PromptTokens + previous.row
					if !slices.Equal(prior.TeacherTokens[:end], example.TeacherTokens[:target]) {
						return nil, rejected("prefix digest collision")
					}
				} else {
					job.rows[digest] = prefixRow{exampleIndex, target - example.PromptTokens}
				}
			}
		}
		factory.jobs[expected.Teacher] = job
	}
	return factory, nil
}

func rawFits(expected fusioncache.Expectation, config NativeConfig, maximum int64) bool {
	add := func(size int64) bool {
		if size < 0 || size > maximum {
			return false
		}
		maximum -= size
		return true
	}
	for _, model := range []fusioncache.ModelIdentity{expected.Teacher, expected.Student, config.Identity} {
		for _, value := range []string{model.Name, model.Revision, model.WeightsSHA256, model.TokenizerSHA256, model.TemplateSHA256, model.RuntimeSHA256} {
			if !add(int64(len(value))) {
				return false
			}
		}
	}
	for _, value := range []string{config.ModelPath, config.TokenizerPath, config.TemplatePath, config.RuntimePath, config.BindingEvidencePath,
		config.BindingEvidenceSHA256, config.CaptureVersion, config.Capture.Tensor, expected.MappingSHA256,
		expected.Mapping.EvidenceSHA256, expected.Mapping.TeacherTokenizerSHA256, expected.Mapping.StudentTokenizerSHA256} {
		if !add(int64(len(value))) {
			return false
		}
	}
	if !add(int64(len(expected.Mapping.Pairs)) * 16) {
		return false
	}
	for _, example := range expected.Examples {
		if !add(int64(len(example.TeacherTokens)) * 16) {
			return false
		}
		for _, value := range []string{example.DatasetID, example.DatasetSHA256, example.ID, example.Role} {
			if !add(int64(len(value))) {
				return false
			}
		}
	}
	for _, feature := range expected.Features {
		if !add(int64(len(feature.Tensor)) + int64(len(feature.DType)) + 16) {
			return false
		}
	}
	return true
}

func validateConfig(config NativeConfig, expected fusioncache.Expectation, limits NativeLimits) error {
	o := config.Capture
	if config.ModelPath == "" || config.TokenizerPath == "" || config.TemplatePath == "" || config.RuntimePath == "" ||
		config.BindingEvidencePath == "" || !validDigest(config.BindingEvidenceSHA256) || config.CaptureVersion != llama.TeacherCaptureVersion ||
		config.GPULayers < -1 || config.GPULayers > math.MaxInt32 || config.MainGPU < 0 || config.MainGPU > math.MaxInt32 ||
		(o.CPUOnly && config.GPULayers != 0) || o.TopK < 1 || o.TopK > limits.Cache.MaxTopK || o.TopK > expected.Teacher.Vocabulary ||
		expected.Teacher.Vocabulary > math.MaxInt32 || o.Tensor != "result_norm" || o.ContextTokens < 256 || o.ContextTokens > math.MaxInt32 ||
		o.ContextTokens%256 != 0 || o.WindowTokens < 1 || o.WindowTokens > o.ContextTokens || o.Threads < 1 || o.Threads > 256 ||
		o.MaxOutputBytes < 1 || o.MaxWindowBytes < 1 {
		return rejected("invalid artifacts, capture version, placement or native bounds")
	}
	featureWidth := int64(0)
	for _, feature := range expected.Features {
		if feature.DType != "float32" || feature.Dimension > math.MaxInt32 {
			return rejected("native feature dtype or width differs")
		}
		if feature.Tensor != "result_norm" && feature.Tensor != "l_out-"+strconv.Itoa(feature.Layer) {
			return rejected("feature must name final normalization or its exact decoder layer")
		}
		featureWidth += int64(feature.Dimension)
	}
	remaining := limits.MaxResidentBytes
	consume := func(size int64) bool {
		if size < 0 || size > remaining {
			return false
		}
		remaining -= size
		return true
	}
	if !consume(o.MaxOutputBytes) || !consume(o.MaxWindowBytes) {
		return rejected("native buffer caps exceed host payload budget")
	}
	rowBytes := int64(unsafe.Sizeof(fusioncache.Signal{})) + 64
	for _, product := range [][2]int64{{int64(o.TopK), int64(unsafe.Sizeof(fusioncache.TeacherProbability{}))},
		{int64(len(expected.Features)), int64(unsafe.Sizeof(fusioncache.Feature{}))}, {featureWidth, 8}} {
		if rowBytes > remaining || product[0] > (remaining-rowBytes)/product[1] {
			return rejected("response geometry exceeds host payload budget")
		}
		rowBytes += product[0] * product[1]
	}
	if !consume(rowBytes) {
		return rejected("response copy exceeds host payload budget")
	}
	maxInputBytes := int64(0)
	for _, example := range expected.Examples {
		if len(example.TeacherTokens) > o.ContextTokens {
			return rejected("gold example exceeds native context")
		}
		positions := int64(len(example.TeacherTokens) - example.PromptTokens)
		if rowBytes > remaining/positions || !consume(rowBytes*positions) {
			return rejected("cached signals exceed host payload budget")
		}
		maxInputBytes = max(maxInputBytes, int64(len(example.TeacherTokens))*4+positions*4)
	}
	if !consume(maxInputBytes) {
		return rejected("input conversion exceeds host payload budget")
	}
	return nil
}

// Open verifies artifact hashes and the explicit native binding, then loads one
// teacher. An unknown identity or a second resident teacher is refused before IO.
func (f *NativeFactory) Open(ctx context.Context, identity fusioncache.ModelIdentity) (fusioncache.Producer, error) {
	if f == nil || ctx == nil {
		return nil, rejected("factory and context required")
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.closed || f.active != nil {
		return nil, errors.Join(rejected("factory closed or teacher already resident"), f.closeErr)
	}
	job := f.jobs[identity]
	if job == nil {
		return nil, rejected("teacher identity was not frozen")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	config := job.config
	for _, artifact := range []struct{ path, digest string }{
		{config.TokenizerPath, identity.TokenizerSHA256}, {config.TemplatePath, identity.TemplateSHA256},
		{config.RuntimePath, identity.RuntimeSHA256},
	} {
		if _, err := hashFile(ctx, artifact.path, artifact.digest, 0); err != nil {
			return nil, err
		}
	}
	evidence, err := hashFile(ctx, config.BindingEvidencePath, config.BindingEvidenceSHA256, 16<<10)
	if err != nil {
		return nil, err
	}
	var binding NativeBindingEvidence
	decoder := json.NewDecoder(bytes.NewReader(evidence))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&binding); err != nil {
		return nil, rejected("invalid native binding evidence")
	}
	var trailing any
	if decoder.Decode(&trailing) != io.EOF || binding.Backend != "llama.cpp" || binding.CaptureVersion != llama.TeacherCaptureVersion ||
		binding.RuntimeSHA256 != identity.RuntimeSHA256 || !validDigest(binding.ExecutableSHA256) {
		return nil, rejected("runtime binding does not identify this native capture")
	}
	executable, err := os.Executable()
	if err != nil {
		return nil, err
	}
	if _, err := hashFile(ctx, executable, binding.ExecutableSHA256, 0); err != nil {
		return nil, err
	}
	modelOptions := []llama.ModelOption{llama.WithGPULayers(config.GPULayers), llama.WithMainGPU(strconv.Itoa(config.MainGPU))}
	if config.LockWeights {
		modelOptions = append(modelOptions, llama.WithMLock())
	}
	model, err := llama.OpenTeacherModel(config.ModelPath, identity.WeightsSHA256, modelOptions...)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, errors.Join(err, model.Close())
	}
	producer := &nativeProducer{factory: f, job: job, model: model, cache: make(map[int][]fusioncache.Signal)}
	f.active = producer
	f.stats.OpenedTeachers++
	f.stats.ResidentTeachers = 1
	return producer, nil
}

func hashFile(ctx context.Context, path, expected string, keepAtMost int64) (data []byte, err error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { err = errors.Join(err, file.Close()) }()
	before, err := file.Stat()
	if err != nil || !before.Mode().IsRegular() || keepAtMost > 0 && before.Size() > keepAtMost {
		return nil, rejected("artifact type or bound differs")
	}
	digest := sha256.New()
	var kept bytes.Buffer
	buffer := make([]byte, 32<<10)
	var size int64
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		count, readErr := file.Read(buffer)
		size += int64(count)
		if keepAtMost > 0 && size > keepAtMost {
			return nil, rejected("artifact grew beyond bound")
		}
		_, _ = digest.Write(buffer[:count])
		if keepAtMost > 0 {
			_, _ = kept.Write(buffer[:count])
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return nil, readErr
		}
	}
	after, err := file.Stat()
	if err != nil || before.Size() != after.Size() || !before.ModTime().Equal(after.ModTime()) || size != after.Size() ||
		hex.EncodeToString(digest.Sum(nil)) != expected {
		return nil, rejected("artifact digest or stability differs")
	}
	return kept.Bytes(), nil
}

// Stats returns an ownership/capture counter snapshot without exposing weights.
func (f *NativeFactory) Stats() NativeStats {
	if f == nil {
		return NativeStats{}
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.stats
}

// Close waits for any active synchronous capture, releases its teacher and
// propagates cleanup errors. A failed close never permits a successor teacher.
func (f *NativeFactory) Close() error {
	if f == nil {
		return nil
	}
	f.mu.Lock()
	f.closed = true
	active, prior := f.active, f.closeErr
	f.mu.Unlock()
	if active != nil {
		return errors.Join(prior, active.Close())
	}
	return prior
}
