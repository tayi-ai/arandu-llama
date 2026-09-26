package llama

/*
#include "wrapper.h"
#include <stdlib.h>
extern bool goProgressCallback(float progress, void* user_data);
static inline llama_progress_callback_wrapper teacher_progress_callback() {
    return goProgressCallback;
}
*/
import "C"

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"unsafe"
)

// TeacherCaptureVersion identifies full-row gold-prefix decoding, FP32 graph
// features, full-vocabulary Float64 softmax and deterministic top-k tie order.
const TeacherCaptureVersion = "arandu-llama.teacher-forced.v1/llama.cpp.90c26fcd"

// TeacherModel owns verified, adapter-free teacher weights. Close releases the
// weights before loading another teacher; captures on one model are serialized.
// The native model is intentionally not exposed for mutation or adapter loading.
type TeacherModel struct {
	mu        sync.Mutex
	model     *Model
	modelSHA  string
	cpuOnly   bool
	placement string
	closed    bool
}

// OpenTeacherModel verifies a regular GGUF against expectedSHA256 before and
// after loading, including file identity, size and modification time. It forces
// mmap off so subsequent file changes cannot mutate the loaded teacher. Model
// options still select placement; WithGPULayers(0) selects CPU devices only.
// Tensor splitting is refused because this loader does not implement it.
func OpenTeacherModel(path, expectedSHA256 string, options ...ModelOption) (*TeacherModel, error) {
	decoded, err := hex.DecodeString(expectedSHA256)
	if path == "" || err != nil || len(decoded) != sha256.Size || expectedSHA256 != strings.ToLower(expectedSHA256) {
		return nil, errors.New("teacher capture: path and lowercase model SHA-256 required")
	}
	before, err := verifyTeacherFile(path, expectedSHA256)
	if err != nil {
		return nil, err
	}
	config := defaultModelConfig
	for _, option := range options {
		if option == nil {
			return nil, errors.New("teacher capture: nil model option")
		}
		option(&config)
	}
	if config.tensorSplit != "" || config.gpuLayers < -1 || config.gpuLayers > math.MaxInt32 {
		return nil, errors.New("teacher capture: invalid placement or unsupported tensor splitting")
	}
	mainGPU := 0
	if config.mainGPU != "" {
		var parseErr error
		mainGPU, parseErr = strconv.Atoi(config.mainGPU)
		if parseErr != nil || mainGPU < 0 || mainGPU > math.MaxInt32 {
			return nil, errors.New("teacher capture: main GPU must be a nonnegative integer")
		}
	}
	model, err := loadTeacherWeights(path, config)
	if err != nil {
		return nil, err
	}
	after, err := verifyTeacherFile(path, expectedSHA256)
	if err != nil || !os.SameFile(before, after) || before.Size() != after.Size() || !before.ModTime().Equal(after.ModTime()) {
		return nil, errors.Join(errors.New("teacher capture: model file changed during loading"), err, model.Close())
	}
	return &TeacherModel{model: model, modelSHA: expectedSHA256, cpuOnly: config.gpuLayers == 0,
		placement: fmt.Sprintf("gpu_layers=%d;main_gpu=%d", config.gpuLayers, mainGPU)}, nil
}

func loadTeacherWeights(path string, config modelConfig) (*Model, error) {
	cPath, mainGPU := C.CString(path), C.CString(config.mainGPU)
	defer C.free(unsafe.Pointer(cPath))
	defer C.free(unsafe.Pointer(mainGPU))
	params := C.llama_wrapper_model_params{n_gpu_layers: C.int(config.gpuLayers), mlock: C.bool(config.mlock), main_gpu: mainGPU}
	var callbackID uintptr
	var idPointer *uintptr
	if config.progressCallback != nil {
		progressCallbackMutex.Lock()
		progressCallbackCounter++
		callbackID = progressCallbackCounter
		progressCallbackMutex.Unlock()
		progressCallbackRegistry.Store(callbackID, config.progressCallback)
		idPointer = new(uintptr)
		*idPointer = callbackID
		params.progress_callback = C.teacher_progress_callback()
		params.progress_callback_user_data = unsafe.Pointer(idPointer)
	} else {
		params.disable_progress_callback = C.bool(config.disableProgressCallback)
	}
	modelPointer := C.llama_wrapper_teacher_model_load(cPath, params)
	runtime.KeepAlive(idPointer)
	if modelPointer == nil {
		if callbackID != 0 {
			progressCallbackRegistry.Delete(callbackID)
		}
		return nil, fmt.Errorf("teacher capture: %s", C.GoString(C.llama_wrapper_last_error()))
	}
	model := &Model{modelPtr: modelPointer, ProgressCallbackID: callbackID}
	runtime.SetFinalizer(model, (*Model).Close)
	return model, nil
}

func verifyTeacherFile(path, expected string) (os.FileInfo, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	before, err := file.Stat()
	if err != nil || !before.Mode().IsRegular() {
		return nil, errors.New("teacher capture: model must be a regular file")
	}
	digest := sha256.New()
	if _, err := io.Copy(digest, file); err != nil {
		return nil, err
	}
	after, err := file.Stat()
	if err != nil || before.Size() != after.Size() || !before.ModTime().Equal(after.ModTime()) || hex.EncodeToString(digest.Sum(nil)) != expected {
		return nil, errors.New("teacher capture: model content digest differs")
	}
	return after, nil
}

// Close releases the teacher weights once and refuses subsequent captures.
func (m *TeacherModel) Close() error {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil
	}
	m.closed = true
	if m.model == nil {
		return nil
	}
	return m.model.Close()
}

// TeacherCaptureOptions admits numerical payload and decode-window bounds.
// Tensor is exactly "result_norm" (after final normalization) or "l_out-N"
// (decoder N output, currently admitted only for llama and qwen35 builders).
// Unsupported names, layers and graph geometries are errors, never aliases.
// ContextTokens is a positive multiple of 256 (the pinned native KV alignment)
// so the context allocation does not silently round up beyond this bound.
// MaxOutputBytes covers output rows and conversion buffers, excluding allocator
// overhead and constant-sized identity metadata. MaxWindowBytes covers logits,
// feature rows and native top-k scratch, excluding weights, KV cache,
// graph intermediates and native allocator overhead; admit those separately.
// CPUOnly requires weights loaded with WithGPULayers(0) and disables context
// operation/KV offload. KV uses FP32, attention is causal with flash disabled.
type TeacherCaptureOptions struct {
	TopK           int
	Tensor         string
	ContextTokens  int
	WindowTokens   int
	Threads        int
	CPUOnly        bool
	MaxOutputBytes int64
	MaxWindowBytes int64
}

// TeacherProbability is a token's full-vocabulary softmax probability at
// temperature one. Top-k selection does not renormalize the retained subset.
type TeacherProbability struct {
	TokenID     int32
	Probability float64
}

// TeacherCaptureRow names the exact gold-prefix boundary that produced a row.
// Position p predicts GoldNextToken == suppliedTokens[p+1]. Features are the
// exact named tensor at p, not a feature of the predicted token at p+1.
type TeacherCaptureRow struct {
	Position      int32
	GoldNextToken int32
	TopK          []TeacherProbability
	RetainedMass  float64
	Features      []float32
}

// TeacherCapture contains native teacher-forced probabilities and feature rows.
// ModelSHA256 is verified file content, TokenDigest binds the entire supplied
// sequence and positions, and SnapshotDigest also binds procedure and options.
// Tensor remains the actual graph name; consumers must explicitly admit its
// mapping to a student layer and may not label result_norm as a decoder output.
type TeacherCapture struct {
	Version        string
	ModelSHA256    string
	Quantisation   string
	Placement      string
	TokenDigest    string
	SnapshotDigest string
	Tensor         string
	DType          string
	Width          int
	Vocabulary     int
	Rows           []TeacherCaptureRow
}

// TeacherTokenDigest hashes the procedure tag, then length-prefixed little
// endian int32 token IDs and positions. A changed gold suffix also changes the
// identity even when the requested early causal row is numerically unchanged.
func TeacherTokenDigest(tokens, positions []int32) string {
	hash := sha256.New()
	_, _ = hash.Write([]byte(TeacherCaptureVersion + "\n"))
	writeInt32s(hash, tokens)
	writeInt32s(hash, positions)
	return hex.EncodeToString(hash.Sum(nil))
}

// CaptureTeacherForced decodes only supplied tokens, from a fresh context, and
// returns requested next-token distributions together with exact feature rows.
// Positions must strictly increase in [0,len(tokens)-2]. For a completion after
// promptTokens tokens, use promptTokens-1 through len(tokens)-2. No tokenizer,
// sampler, free generation, adapter or implicit truncation participates.
// Every decode-window row is marked as an output, independent of selected rows.
// Native context and temporary buffers are released on success or failure.
// The caller must not mutate inputs during this synchronous native call.
func (m *TeacherModel) CaptureTeacherForced(tokens, positions []int32, options TeacherCaptureOptions) (*TeacherCapture, error) {
	if m == nil {
		return nil, errors.New("teacher capture: model required")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed || m.model == nil {
		return nil, errors.New("teacher capture: model is closed")
	}
	if options.CPUOnly && !m.cpuOnly {
		return nil, errors.New("teacher capture: CPUOnly requires weights loaded with WithGPULayers(0)")
	}
	if len(tokens) < 2 || len(tokens) > math.MaxInt32 || len(positions) < 1 || len(positions) > len(tokens)-1 ||
		options.TopK < 1 || options.TopK > math.MaxInt32 || options.Tensor == "" || len(options.Tensor) > 64 || strings.ContainsRune(options.Tensor, 0) ||
		options.ContextTokens < len(tokens) || options.ContextTokens > math.MaxInt32 || options.ContextTokens%256 != 0 || options.WindowTokens < 1 ||
		options.WindowTokens > options.ContextTokens || options.Threads < 1 || options.Threads > 256 ||
		options.MaxOutputBytes < 1 || options.MaxWindowBytes < 1 {
		return nil, errors.New("teacher capture: invalid sequence or resource limits")
	}
	for i, token := range tokens {
		if token < 0 {
			return nil, fmt.Errorf("teacher capture: negative token at %d", i)
		}
	}
	for i, position := range positions {
		if position < 0 || int(position) >= len(tokens)-1 || i > 0 && position <= positions[i-1] {
			return nil, errors.New("teacher capture: positions must increase and have a supplied next token")
		}
	}
	tensor := C.CString(options.Tensor)
	defer C.free(unsafe.Pointer(tensor))
	var vocabulary, width C.int
	if C.llama_wrapper_teacher_geometry(m.model.modelPtr, tensor, &vocabulary, &width) != 0 {
		return nil, fmt.Errorf("teacher capture: %s", C.GoString(C.llama_wrapper_last_error()))
	}
	if options.TopK > int(vocabulary) {
		return nil, errors.New("teacher capture: top-k exceeds vocabulary")
	}
	for _, token := range tokens {
		if token >= int32(vocabulary) {
			return nil, errors.New("teacher capture: token outside vocabulary")
		}
	}
	// Every multiplication below is bounded before an output allocation.
	rowBytes := int64(options.TopK)*(12+int64(unsafe.Sizeof(TeacherProbability{}))) + int64(width)*4 + 8 + int64(unsafe.Sizeof(TeacherCaptureRow{}))
	if rowBytes > options.MaxOutputBytes/int64(len(positions)) || rowBytes > int64(math.MaxInt)/int64(len(positions)) ||
		(2*int64(vocabulary)+int64(width))*4 > options.MaxWindowBytes {
		return nil, errors.New("teacher capture: numeric payload or window budget exceeded")
	}
	quantisation, err := m.model.Describe()
	if err != nil {
		return nil, err
	}
	ids := make([]C.int, len(positions)*options.TopK)
	probabilities := make([]float64, len(ids))
	mass := make([]float64, len(positions))
	features := make([]float32, len(positions)*int(width))
	result := C.llama_wrapper_capture_teacher(m.model.modelPtr, tensor,
		(*C.int)(unsafe.Pointer(&tokens[0])), C.int(len(tokens)), (*C.int)(unsafe.Pointer(&positions[0])), C.int(len(positions)),
		C.int(options.TopK), C.int(options.ContextTokens), C.int(options.WindowTokens), C.int(options.Threads), C.bool(options.CPUOnly),
		C.longlong(options.MaxWindowBytes), &ids[0], (*C.double)(unsafe.Pointer(&probabilities[0])), C.longlong(len(ids)),
		(*C.double)(unsafe.Pointer(&mass[0])), (*C.float)(unsafe.Pointer(&features[0])), C.longlong(len(features)))
	runtime.KeepAlive(tokens)
	runtime.KeepAlive(positions)
	runtime.KeepAlive(ids)
	runtime.KeepAlive(probabilities)
	runtime.KeepAlive(mass)
	runtime.KeepAlive(features)
	runtime.KeepAlive(m)
	if result != 0 {
		return nil, fmt.Errorf("teacher capture: %s", C.GoString(C.llama_wrapper_last_error()))
	}
	capture := &TeacherCapture{Version: TeacherCaptureVersion, ModelSHA256: m.modelSHA, Quantisation: quantisation, Placement: m.placement,
		TokenDigest: TeacherTokenDigest(tokens, positions), Tensor: options.Tensor, DType: "f32", Width: int(width), Vocabulary: int(vocabulary),
		Rows: make([]TeacherCaptureRow, len(positions))}
	for i, position := range positions {
		row := TeacherCaptureRow{Position: position, GoldNextToken: tokens[position+1], RetainedMass: mass[i],
			TopK: make([]TeacherProbability, options.TopK), Features: features[i*int(width) : (i+1)*int(width)]}
		for k := range row.TopK {
			offset := i*options.TopK + k
			row.TopK[k] = TeacherProbability{TokenID: int32(ids[offset]), Probability: probabilities[offset]}
		}
		capture.Rows[i] = row
	}
	hash := sha256.New()
	for _, identity := range []string{TeacherCaptureVersion, capture.ModelSHA256, quantisation, m.placement, capture.TokenDigest, options.Tensor} {
		writeString(hash, identity)
	}
	for _, number := range []int64{int64(options.TopK), int64(options.ContextTokens), int64(options.WindowTokens), int64(options.Threads), options.MaxOutputBytes, options.MaxWindowBytes} {
		_ = binary.Write(hash, binary.LittleEndian, number)
	}
	_ = binary.Write(hash, binary.LittleEndian, options.CPUOnly)
	capture.SnapshotDigest = hex.EncodeToString(hash.Sum(nil))
	return capture, nil
}
