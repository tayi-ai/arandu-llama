package teachercapture_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"testing"

	llama "github.com/tayi-ai/arandu-llama"
)

const fixtureWidth, fixtureVocabulary = 8, 16

// tinyTeacher writes a complete deterministic, two-layer F32 GGUF. No external
// model, network, tokenizer or accelerator is needed by this native test.
func tinyTeacher(t *testing.T) (string, string, []float32) {
	t.Helper()
	type tensor struct {
		name string
		dims []uint64
		data []float32
	}
	var tensors []tensor
	add := func(name string, dims ...uint64) {
		count := 1
		for _, d := range dims {
			count *= int(d)
		}
		data := make([]float32, count)
		for i := range data {
			data[i] = float32(.12 * math.Sin(float64(i+1)*.37+float64(len(tensors))*1.19))
			if len(dims) == 1 {
				data[i] = 1 + float32(i)*.01
			}
		}
		tensors = append(tensors, tensor{name, dims, data})
	}
	add("token_embd.weight", fixtureWidth, fixtureVocabulary)
	add("output_norm.weight", fixtureWidth)
	add("output.weight", fixtureWidth, fixtureVocabulary)
	output := tensors[2].data
	// Equal output columns make the native top-k tie order observable.
	copy(output[fixtureWidth:2*fixtureWidth], output[:fixtureWidth])
	for layer := 0; layer < 2; layer++ {
		prefix := fmt.Sprintf("blk.%d.", layer)
		add(prefix+"attn_norm.weight", fixtureWidth)
		for _, name := range []string{"attn_q", "attn_k", "attn_v", "attn_output"} {
			add(prefix+name+".weight", fixtureWidth, fixtureWidth)
		}
		add(prefix+"ffn_norm.weight", fixtureWidth)
		add(prefix+"ffn_gate.weight", fixtureWidth, 16)
		add(prefix+"ffn_down.weight", 16, fixtureWidth)
		add(prefix+"ffn_up.weight", fixtureWidth, 16)
	}
	metadata := []struct {
		key   string
		value any
	}{
		{"general.architecture", "llama"}, {"general.name", "native-teacher-capture-fixture"},
		{"llama.context_length", uint32(256)}, {"llama.embedding_length", uint32(fixtureWidth)},
		{"llama.feed_forward_length", uint32(16)}, {"llama.block_count", uint32(2)},
		{"llama.attention.head_count", uint32(2)}, {"llama.attention.head_count_kv", uint32(2)},
		{"llama.rope.dimension_count", uint32(4)}, {"llama.attention.layer_norm_rms_epsilon", float32(1e-5)},
		{"tokenizer.ggml.model", "none"}, {"llama.vocab_size", uint32(fixtureVocabulary)},
	}
	var header, data bytes.Buffer
	write := func(buffer *bytes.Buffer, value any) {
		if err := binary.Write(buffer, binary.LittleEndian, value); err != nil {
			t.Fatal(err)
		}
	}
	writeString := func(value string) {
		write(&header, uint64(len(value)))
		header.WriteString(value)
	}
	header.WriteString("GGUF")
	write(&header, uint32(3))
	write(&header, uint64(len(tensors)))
	write(&header, uint64(len(metadata)))
	for _, field := range metadata {
		writeString(field.key)
		switch value := field.value.(type) {
		case string:
			write(&header, uint32(8))
			writeString(value)
		case uint32:
			write(&header, uint32(4))
			write(&header, value)
		case float32:
			write(&header, uint32(6))
			write(&header, value)
		}
	}
	for _, tensor := range tensors {
		writeString(tensor.name)
		write(&header, uint32(len(tensor.dims)))
		write(&header, tensor.dims)
		write(&header, uint32(0)) // GGML_TYPE_F32
		write(&header, uint64(data.Len()))
		write(&data, tensor.data)
		for data.Len()%32 != 0 {
			data.WriteByte(0)
		}
	}
	for header.Len()%32 != 0 {
		header.WriteByte(0)
	}
	header.Write(data.Bytes())
	path := filepath.Join(t.TempDir(), "teacher.gguf")
	if err := os.WriteFile(path, header.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	return path, fmt.Sprintf("%x", sha256.Sum256(header.Bytes())), output
}

func openTeacher(t *testing.T) (*llama.TeacherModel, []float32) {
	t.Helper()
	path, digest, output := tinyTeacher(t)
	model, err := llama.OpenTeacherModel(path, digest, llama.WithGPULayers(0))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := model.Close(); err != nil {
			t.Error(err)
		}
	})
	return model, output
}

func captureOptions() llama.TeacherCaptureOptions {
	return llama.TeacherCaptureOptions{TopK: fixtureVocabulary, Tensor: "result_norm", ContextTokens: 256,
		WindowTokens: 3, Threads: 1, CPUOnly: true, MaxOutputBytes: 64 << 10, MaxWindowBytes: 64 << 10}
}

func capture(t *testing.T, model *llama.TeacherModel, tokens, positions []int32, options llama.TeacherCaptureOptions) *llama.TeacherCapture {
	t.Helper()
	got, err := model.CaptureTeacherForced(tokens, positions, options)
	if err != nil {
		t.Fatal(err)
	}
	return got
}

func TestTeacherForcedNativeSoftmaxAndFeatures(t *testing.T) {
	model, output := openTeacher(t)
	tokens, positions := []int32{1, 2, 3, 4, 5, 6}, []int32{1, 2, 3, 4}
	options := captureOptions()
	got := capture(t, model, tokens, positions, options)
	if got.Tensor != "result_norm" || got.DType != "f32" || got.Width != fixtureWidth || got.Vocabulary != fixtureVocabulary ||
		got.Version != llama.TeacherCaptureVersion || !strings.Contains(got.Version, "90c26fcd") || got.Quantisation == "" ||
		len(got.ModelSHA256) != 64 || len(got.SnapshotDigest) != 64 || got.TokenDigest != llama.TeacherTokenDigest(tokens, positions) {
		t.Fatalf("incomplete capture identity: %+v", got)
	}
	maximumSoftmaxError := 0.0
	for i, row := range got.Rows {
		if row.Position != positions[i] || row.GoldNextToken != tokens[positions[i]+1] || len(row.Features) != fixtureWidth {
			t.Fatalf("incorrect causal row: %+v", row)
		}
		// Independent full-vocabulary reference: known output weights times the
		// actual final-normalization feature, followed by stable Float64 softmax.
		logits := make([]float64, fixtureVocabulary)
		maximum := math.Inf(-1)
		for token := range logits {
			for column, feature := range row.Features {
				logits[token] += float64(feature) * float64(output[token*fixtureWidth+column])
			}
			maximum = math.Max(maximum, logits[token])
		}
		denominator := 0.0
		order := make([]int, len(logits))
		for token, logit := range logits {
			denominator += math.Exp(logit - maximum)
			order[token] = token
		}
		sort.Slice(order, func(a, b int) bool {
			if logits[order[a]] == logits[order[b]] {
				return order[a] < order[b]
			}
			return logits[order[a]] > logits[order[b]]
		})
		for k, probability := range row.TopK {
			want := math.Exp(logits[order[k]]-maximum) / denominator
			maximumSoftmaxError = math.Max(maximumSoftmaxError, math.Abs(probability.Probability-want))
			if probability.TokenID != int32(order[k]) || math.Abs(probability.Probability-want) > 1e-7 {
				t.Fatalf("row %d rank %d: %+v, expected token %d p=%.12g", i, k, probability, order[k], want)
			}
		}
		if math.Abs(row.RetainedMass-1) > 1e-12 {
			t.Fatalf("full vocabulary mass = %.16g", row.RetainedMass)
		}
	}
	if repeated := capture(t, model, tokens, positions, options); !reflect.DeepEqual(got, repeated) {
		t.Fatal("fresh-context capture is not repeatable")
	}
	// A one-row numerical window budget must work by reducing the window,
	// without reducing the supplied sequence or requested rows.
	boundedOptions := options
	boundedOptions.MaxWindowBytes = (2*fixtureVocabulary + fixtureWidth) * 4
	bounded := capture(t, model, tokens, positions, boundedOptions)
	for i, row := range got.Rows {
		for k, probability := range row.TopK {
			other := bounded.Rows[i].TopK[k]
			if probability.TokenID != other.TokenID || math.Abs(probability.Probability-other.Probability) > 1e-7 {
				t.Fatal("bounded window changed distribution")
			}
		}
	}
	options.TopK = 3
	partial := capture(t, model, tokens, positions, options)
	t.Logf("native softmax maximum absolute error %.12g; first-row top-3 retained mass %.12g; repeated capture bitwise identical",
		maximumSoftmaxError, partial.Rows[0].RetainedMass)
	for i, row := range partial.Rows {
		mass := 0.0
		for k, probability := range row.TopK {
			if probability != got.Rows[i].TopK[k] {
				t.Fatal("top-k probabilities were renormalized or altered")
			}
			mass += probability.Probability
		}
		if mass != row.RetainedMass || mass >= 1 || mass <= 0 {
			t.Fatalf("incorrect retained mass: %v", row.RetainedMass)
		}
	}
}

func TestTeacherForcedCausalityWindowsAndActualLayers(t *testing.T) {
	model, _ := openTeacher(t)
	tokens := []int32{1, 2, 3, 4, 5, 6}
	positions := []int32{0, 1, 2, 3, 4}
	options := captureOptions()
	all := capture(t, model, tokens, positions, options)
	subset := capture(t, model, tokens, []int32{1, 4}, options)
	if !reflect.DeepEqual(subset.Rows, []llama.TeacherCaptureRow{all.Rows[1], all.Rows[4]}) {
		t.Fatal("selected rows changed native graph outputs")
	}
	suffix := append([]int32(nil), tokens...)
	suffix[len(suffix)-1] = 10
	changedSuffix := capture(t, model, suffix, positions, options)
	if changedSuffix.TokenDigest == all.TokenDigest || changedSuffix.SnapshotDigest == all.SnapshotDigest {
		t.Fatal("gold suffix was omitted from identity")
	}
	for i, row := range all.Rows {
		if !reflect.DeepEqual(row.Features, changedSuffix.Rows[i].Features) || !reflect.DeepEqual(row.TopK, changedSuffix.Rows[i].TopK) {
			t.Fatal("future token affected causal row")
		}
	}
	prefix := append([]int32(nil), tokens...)
	prefix[1] = 10
	changedPrefix := capture(t, model, prefix, positions, options)
	if reflect.DeepEqual(changedPrefix.Rows[2].TopK, all.Rows[2].TopK) || reflect.DeepEqual(changedPrefix.Rows[2].Features, all.Rows[2].Features) {
		t.Fatal("gold prefix did not affect subsequent rows")
	}
	options.WindowTokens = 2
	windowed := capture(t, model, tokens, positions, options)
	if windowed.SnapshotDigest == all.SnapshotDigest {
		t.Fatal("decode window omitted from procedure identity")
	}
	for i, row := range all.Rows {
		for j, feature := range row.Features {
			if math.Abs(float64(feature-windowed.Rows[i].Features[j])) > 1e-5 {
				t.Fatal("window boundary changed feature")
			}
		}
		for k, probability := range row.TopK {
			other := windowed.Rows[i].TopK[k]
			if probability.TokenID != other.TokenID || math.Abs(probability.Probability-other.Probability) > 1e-7 {
				t.Fatal("window boundary changed distribution")
			}
		}
	}
	for _, tensor := range []string{"l_out-0", "l_out-1"} {
		options.Tensor = tensor
		layer := capture(t, model, tokens, positions, options)
		if layer.Tensor != tensor || reflect.DeepEqual(layer.Rows[0].Features, all.Rows[0].Features) {
			t.Fatal("decoder feature aliased to final normalization")
		}
		for i, row := range layer.Rows {
			for k, probability := range row.TopK {
				if math.Abs(probability.Probability-windowed.Rows[i].TopK[k].Probability) > 1e-7 {
					t.Fatal("feature selection changed teacher probabilities")
				}
			}
		}
	}
}

func TestTeacherCaptureRejectsInvalidInputsAndCleansUp(t *testing.T) {
	model, _ := openTeacher(t)
	for _, invalid := range []struct {
		name              string
		tokens, positions []int32
		change            func(*llama.TeacherCaptureOptions)
	}{
		{name: "empty"}, {name: "negative-token", tokens: []int32{-1, 2}, positions: []int32{0}},
		{name: "outside-vocabulary", tokens: []int32{1, fixtureVocabulary}, positions: []int32{0}},
		{name: "last-position", tokens: []int32{1, 2}, positions: []int32{1}},
		{name: "negative-position", tokens: []int32{1, 2}, positions: []int32{-1}},
		{name: "duplicate-position", tokens: []int32{1, 2, 3}, positions: []int32{0, 0}},
		{name: "reverse-position", tokens: []int32{1, 2, 3}, positions: []int32{1, 0}},
		{name: "top-k", change: func(o *llama.TeacherCaptureOptions) { o.TopK = fixtureVocabulary + 1 }},
		{name: "top-k-overflow", change: func(o *llama.TeacherCaptureOptions) { o.TopK = math.MaxInt }},
		{name: "context-truncation", change: func(o *llama.TeacherCaptureOptions) { o.ContextTokens = 1 }},
		{name: "context-alignment", change: func(o *llama.TeacherCaptureOptions) { o.ContextTokens = 257 }},
		{name: "empty-window", change: func(o *llama.TeacherCaptureOptions) { o.WindowTokens = 0 }},
		{name: "threads", change: func(o *llama.TeacherCaptureOptions) { o.Threads = 0 }},
		{name: "output-budget", change: func(o *llama.TeacherCaptureOptions) { o.MaxOutputBytes = 1 }},
		{name: "window-budget", change: func(o *llama.TeacherCaptureOptions) { o.MaxWindowBytes = 1 }},
		{name: "unknown-tensor", change: func(o *llama.TeacherCaptureOptions) { o.Tensor = "hidden.0" }},
		{name: "outside-layer", change: func(o *llama.TeacherCaptureOptions) { o.Tensor = "l_out-2" }},
		{name: "noncanonical-layer", change: func(o *llama.TeacherCaptureOptions) { o.Tensor = "l_out-01" }},
		{name: "nul-tensor", change: func(o *llama.TeacherCaptureOptions) { o.Tensor = "result_norm\x00" }},
	} {
		t.Run(invalid.name, func(t *testing.T) {
			options := captureOptions()
			tokens, positions := invalid.tokens, invalid.positions
			if invalid.change != nil {
				invalid.change(&options)
				tokens, positions = []int32{1, 2}, []int32{0}
			}
			if got, err := model.CaptureTeacherForced(tokens, positions, options); err == nil || got != nil {
				t.Fatalf("accepted invalid capture: %+v, %v", got, err)
			}
		})
	}
	capture(t, model, []int32{1, 2}, []int32{0}, captureOptions())
	if err := model.Close(); err != nil {
		t.Fatal(err)
	}
	if got, err := model.CaptureTeacherForced([]int32{1, 2}, []int32{0}, captureOptions()); err == nil || got != nil {
		t.Fatal("closed teacher accepted capture")
	}
	var nilModel *llama.TeacherModel
	if _, err := nilModel.CaptureTeacherForced(nil, nil, captureOptions()); err == nil {
		t.Fatal("nil teacher accepted capture")
	}
}

func TestTeacherCaptureVerifiesModelIdentity(t *testing.T) {
	path, digest, _ := tinyTeacher(t)
	for _, invalid := range []string{"", strings.Repeat("0", 64), strings.ToUpper(digest)} {
		if model, err := llama.OpenTeacherModel(path, invalid, llama.WithGPULayers(0)); err == nil || model != nil {
			t.Fatal("unverified model was admitted")
		}
	}
	if model, err := llama.OpenTeacherModel(filepath.Dir(path), digest); err == nil || model != nil {
		t.Fatal("directory was admitted")
	}
	for _, options := range [][]llama.ModelOption{{nil}, {llama.WithGPULayers(-2)}, {llama.WithMainGPU("invalid")}, {llama.WithTensorSplit("1,1")}} {
		if model, err := llama.OpenTeacherModel(path, digest, options...); err == nil || model != nil {
			t.Fatal("unsupported model option was admitted")
		}
	}
}

func TestTeacherCaptureRejectsNonfiniteNativeFeatures(t *testing.T) {
	path, _, _ := tinyTeacher(t)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	// The last tensor is a dense FFN-up matrix: poison a real weight while
	// preserving valid GGUF structure and a matching verified file digest.
	binary.LittleEndian.PutUint32(data[len(data)-4:], math.Float32bits(float32(math.NaN())))
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	model, err := llama.OpenTeacherModel(path, fmt.Sprintf("%x", sha256.Sum256(data)), llama.WithGPULayers(0))
	if err != nil {
		t.Fatal(err)
	}
	for attempt := 0; attempt < 2; attempt++ {
		got, err := model.CaptureTeacherForced([]int32{1, 2}, []int32{0}, captureOptions())
		if got != nil || err == nil || !strings.Contains(err.Error(), "nonfinite") {
			t.Fatalf("native failure was not refused: got=%+v err=%v", got, err)
		}
	}
	if err := model.Close(); err != nil {
		t.Fatal(err)
	}
	valid, _ := openTeacher(t)
	capture(t, valid, []int32{1, 2}, []int32{0}, captureOptions())
}

func TestTeacherCaptureOwnsImmutableLoadedWeights(t *testing.T) {
	path, digest, _ := tinyTeacher(t)
	model, err := llama.OpenTeacherModel(path, digest, llama.WithGPULayers(0), llama.WithMMap(true))
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	before := capture(t, model, []int32{1, 2}, []int32{0}, captureOptions())
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	binary.LittleEndian.PutUint32(data[len(data)-4:], math.Float32bits(float32(math.NaN())))
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	after := capture(t, model, []int32{1, 2}, []int32{0}, captureOptions())
	if !reflect.DeepEqual(before, after) || before.ModelSHA256 != digest {
		t.Fatal("file mutation changed the owned teacher snapshot")
	}
}

func TestTeacherCaptureVersionMatchesPinnedSource(t *testing.T) {
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Join(filepath.Dir(file), "..", "..", "..")
	command := exec.Command("git", "ls-tree", "HEAD", "llama.cpp")
	command.Dir = root
	output, err := command.Output()
	if err != nil {
		t.Skipf("source gitlink unavailable: %v", err)
	}
	fields := strings.Fields(string(output))
	if len(fields) < 3 || len(fields[2]) < 8 || !strings.HasSuffix(llama.TeacherCaptureVersion, "/llama.cpp."+fields[2][:8]) {
		t.Fatalf("capture version %q differs from gitlink %q", llama.TeacherCaptureVersion, output)
	}
}
