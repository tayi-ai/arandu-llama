package native_test

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"math"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	decoder "github.com/tayi-ai/arandu-llama/training/decoder"
	services "github.com/tayi-ai/arandu-llama/training/native"
	"github.com/tayi-ai/arandu-llama/training/tokenizer"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

func nativeCPU(t *testing.T) {
	t.Helper()
	if !torch.Enabled() {
		t.Skip("requires the explicit libtorch CPU build")
	}
}

func contentDigest(raw []byte) string {
	digest := sha256.Sum256(raw)
	return hex.EncodeToString(digest[:])
}

func floatContent(values []float32) []byte {
	raw := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(value))
	}
	return raw
}

func tensorSpec(shape []int64, dtype torch.DType, raw []byte) decoder.AssemblyTensor {
	return decoder.AssemblyTensor{ReferenceName: "fixture.weight", Shape: shape, DType: dtype, Device: torch.CPUDevice(), Bytes: int64(len(raw)), SHA256: contentDigest(raw)}
}

func TestNativeTensorDigestHashesNoncontiguousViewsInLogicalOrder(t *testing.T) {
	nativeCPU(t)
	values := make([]float32, 28)
	for i := range values {
		values[i] = float32(i-9) / 4
	}
	original, err := torch.FromFloat32(values, []int64{4, 7}, torch.CPUDevice(), false)
	if err != nil {
		t.Fatal(err)
	}
	defer original.Close()
	transposed, err := original.Transpose(0, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer transposed.Close()
	var reordered []float32
	for column := range 7 {
		for row := range 4 {
			reordered = append(reordered, values[row*7+column])
		}
	}
	for _, fixture := range []struct {
		name  string
		value *torch.Tensor
		spec  decoder.AssemblyTensor
	}{
		{"contiguous", original, tensorSpec([]int64{4, 7}, torch.Float32, floatContent(values))},
		{"transposed", transposed, tensorSpec([]int64{7, 4}, torch.Float32, floatContent(reordered))},
	} {
		t.Run(fixture.name, func(t *testing.T) {
			for _, chunk := range []int{4, 7, 16, 64, 4 << 20} {
				digest, err := services.ExperimentNativeTensorDigest(context.Background(), fixture.value, fixture.spec, chunk)
				if err != nil || digest != fixture.spec.SHA256 {
					t.Fatalf("chunk=%d digest=%s error=%v", chunk, digest, err)
				}
			}
		})
	}
	after, err := original.Float32Values()
	if err != nil || !slices.Equal(after, values) {
		t.Fatal("hashing changed or closed the borrowed source")
	}
}

func TestNativeTensorDigestPreservesFP16BytesAndScalarGeometry(t *testing.T) {
	nativeCPU(t)
	// Positive/negative zero, the smallest subnormal, one and minus two.
	half := []byte{0, 0, 0, 128, 1, 0, 0, 60, 0, 192}
	value, err := torch.FromBytes(half, []int64{1, 5}, torch.Float16, torch.CPUDevice(), false)
	if err != nil {
		t.Fatal(err)
	}
	defer value.Close()
	expected := tensorSpec([]int64{1, 5}, torch.Float16, half)
	if digest, err := services.ExperimentNativeTensorDigest(context.Background(), value, expected, 5); err != nil || digest != expected.SHA256 {
		t.Fatalf("FP16 bytes changed: %s %v", digest, err)
	}
	negativeZero := []byte{0, 0, 0, 128}
	scalar, err := torch.FromBytes(negativeZero, []int64{}, torch.Float32, torch.CPUDevice(), false)
	if err != nil {
		t.Fatal(err)
	}
	defer scalar.Close()
	expected = tensorSpec([]int64{}, torch.Float32, negativeZero)
	if digest, err := services.ExperimentNativeTensorDigest(context.Background(), scalar, expected, 4); err != nil || digest != expected.SHA256 {
		t.Fatalf("scalar bytes changed: %s %v", digest, err)
	}
}

func TestNativeTensorDigestRejectsMetadataContentAndNonfiniteValues(t *testing.T) {
	nativeCPU(t)
	raw := floatContent([]float32{1, 2, 3, 4})
	value, err := torch.FromBytes(raw, []int64{2, 2}, torch.Float32, torch.CPUDevice(), false)
	if err != nil {
		t.Fatal(err)
	}
	defer value.Close()
	for _, mutate := range []func(*decoder.AssemblyTensor){
		func(s *decoder.AssemblyTensor) { s.Shape = []int64{4} },
		func(s *decoder.AssemblyTensor) { s.DType = torch.Float16 },
		func(s *decoder.AssemblyTensor) { s.Device = torch.CUDADevice(0) },
		func(s *decoder.AssemblyTensor) { s.Bytes-- },
		func(s *decoder.AssemblyTensor) { s.Trainable = true },
		func(s *decoder.AssemblyTensor) { s.SHA256 = strings.Repeat("f", 64) },
		func(s *decoder.AssemblyTensor) { s.SHA256 = "" },
	} {
		expected := tensorSpec([]int64{2, 2}, torch.Float32, raw)
		mutate(&expected)
		if digest, err := services.ExperimentNativeTensorDigest(context.Background(), value, expected, 4); !errors.Is(err, services.ErrExperimentNativeReplica) || digest != "" {
			t.Fatalf("invalid tensor admitted: %s %v", digest, err)
		}
	}
	for _, chunk := range []int{0, 3, 4<<20 + 1} {
		if _, err := services.ExperimentNativeTensorDigest(context.Background(), value, tensorSpec([]int64{2, 2}, torch.Float32, raw), chunk); !errors.Is(err, services.ErrExperimentNativeReplica) {
			t.Fatal("invalid chunk admitted")
		}
	}
	for _, bad := range []float32{float32(math.NaN()), float32(math.Inf(-1))} {
		raw := floatContent([]float32{0, bad})
		value, err := torch.FromBytes(raw, []int64{2}, torch.Float32, torch.CPUDevice(), false)
		if err != nil {
			t.Fatal(err)
		}
		digest, hashErr := services.ExperimentNativeTensorDigest(context.Background(), value, tensorSpec([]int64{2}, torch.Float32, raw), 4)
		closeErr := value.Close()
		if !errors.Is(hashErr, services.ErrExperimentNativeReplica) || digest != "" || closeErr != nil {
			t.Fatal("nonfinite tensor was admitted by its matching content hash")
		}
	}
}

type cancelBetweenChunks struct {
	context.Context
	cancel context.CancelFunc
	calls  atomic.Int32
}

func (c *cancelBetweenChunks) Err() error {
	if c.calls.Add(1) == 5 {
		c.cancel()
	}
	return c.Context.Err()
}

func TestNativeTensorDigestCancellationLeavesBorrowedTensorOpen(t *testing.T) {
	nativeCPU(t)
	values := []float32{1, 2, 3, 4, 5, 6}
	raw := floatContent(values)
	value, err := torch.FromBytes(raw, []int64{2, 3}, torch.Float32, torch.CPUDevice(), false)
	if err != nil {
		t.Fatal(err)
	}
	defer value.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	between := &cancelBetweenChunks{Context: ctx, cancel: cancel}
	digest, err := services.ExperimentNativeTensorDigest(between, value, tensorSpec([]int64{2, 3}, torch.Float32, raw), 4)
	if !errors.Is(err, context.Canceled) || digest != "" || between.calls.Load() < 5 {
		t.Fatalf("cancellation did not reach a chunk boundary: %v", err)
	}
	after, err := value.Float32Values()
	if err != nil || !slices.Equal(after, values) {
		t.Fatal("cancelled inspection changed source ownership")
	}
}

func TestNativeCalibrationUsesCompleteVocabularyAndCandidateOrder(t *testing.T) {
	logits := make([]float32, 248320)
	uniform, err := services.ExperimentNativeVocabularyLogProbabilities(testNativeExperiment(), logits)
	if err != nil {
		t.Fatal(err)
	}
	for _, value := range uniform {
		if math.Abs(value+math.Log(float64(len(logits)))) > 1e-12 {
			t.Fatal("calibration used four-choice normalization")
		}
	}
	ids := services.ExperimentNativeRecipe(testNativeExperiment()).Data.CandidateVocabularyIDs
	partition := float64(len(logits) - 4)
	for i, id := range ids {
		logits[id] = float32(i + 1)
		partition += math.Exp(float64(i + 1))
	}
	got, err := services.ExperimentNativeVocabularyLogProbabilities(testNativeExperiment(), logits)
	if err != nil {
		t.Fatal(err)
	}
	for i, value := range got {
		if math.Abs(value-(float64(i+1)-math.Log(partition))) > 2e-12 {
			t.Fatalf("candidate%d=%g", i, value)
		}
	}
	for i := range logits {
		logits[i] += 64
	}
	shifted, err := services.ExperimentNativeVocabularyLogProbabilities(testNativeExperiment(), logits)
	if err != nil || shifted != got {
		t.Fatal("stable normalization changed under an exactly representable shift")
	}
	logits = make([]float32, 248320)
	logits[0] = 20 // Outside A/B/C/D: it must still affect every score.
	got, err = services.ExperimentNativeVocabularyLogProbabilities(testNativeExperiment(), logits)
	want := -20 - math.Log(1+float64(len(logits)-1)*math.Exp(-20))
	if err != nil || math.Abs(got[0]-want) > 2e-11 || got[0] > -19 {
		t.Fatalf("outside-vocabulary-candidate contribution omitted: %v %v", got, err)
	}
}

func TestNativeCalibrationRejectsIncompleteOrNonfiniteVocabulary(t *testing.T) {
	if _, err := services.ExperimentNativeVocabularyLogProbabilities(testNativeExperiment(), make([]float32, 4)); !errors.Is(err, services.ErrExperimentNativeReplica) {
		t.Fatal("four logits are insufficient for historical calibration")
	}
	for _, bad := range []float32{float32(math.NaN()), float32(math.Inf(1))} {
		logits := make([]float32, 248320)
		logits[len(logits)-1] = bad
		if result, err := services.ExperimentNativeVocabularyLogProbabilities(testNativeExperiment(), logits); !errors.Is(err, services.ErrExperimentNativeReplica) || result != [4]float64{} {
			t.Fatal("noncandidate nonfinite logit was ignored")
		}
	}
	logits := make([]float32, 248320)
	logits[0] = math.MaxFloat32
	for _, id := range services.ExperimentNativeRecipe(testNativeExperiment()).Data.CandidateVocabularyIDs {
		logits[id] = -math.MaxFloat32
	}
	result, err := services.ExperimentNativeVocabularyLogProbabilities(testNativeExperiment(), logits)
	if err != nil || math.IsInf(result[0], 0) || math.IsNaN(result[0]) {
		t.Fatal("finite FP32 range overflowed Go float64 log normalization")
	}
}

func TestDecoderReplicaHasNoFixtureGeometryOrUnqualifiedConstructionBypass(t *testing.T) {
	loaded := &decoder.LoadedTextModel{Model: &decoder.TextModel{Layers: make([]decoder.Layer, 2)}}
	options := services.ExperimentDecoderReplicaOptions{Limits: decoder.Limits{MaxTokens: 32, LogitRows: 2, MaxCheckpointBytes: 1 << 20},
		HashChunkBytes: 64 << 10, ParameterCopyBytes: 3 * 557056 * 4,
		SourceManifestSHA256:          services.ExperimentNativeRecipe(testNativeExperiment()).Model.ManifestSHA256,
		ExpectedRotaryFrequencySHA256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}
	if replica, err := services.NewExperimentDecoderReplica(testNativeExperiment(), loaded, &tokenizer.Tokenizer{}, &decoder.AssemblyPlan{}, options); !errors.Is(err, services.ErrExperimentNativeReplica) || replica != nil {
		t.Fatal("unqualified CPU fixture became a production replica")
	}
	if loaded.Model == nil || len(loaded.Model.Layers) != 2 {
		t.Fatal("failed constructor consumed caller ownership")
	}
	for _, value := range []*services.ExperimentDecoderReplica{nil, {}} {
		if _, err := value.Inspect(context.Background()); !errors.Is(err, services.ErrExperimentNativeReplica) {
			t.Fatal("uninitialized replica inspected")
		}
		if _, err := value.InitialParameters(context.Background()); !errors.Is(err, services.ErrExperimentNativeReplica) {
			t.Fatal("uninitialized replica exported initial weights")
		}
		if _, err := value.Gradient(context.Background(), services.ExperimentNativeExample{}, nil); !errors.Is(err, services.ErrExperimentNativeReplica) {
			t.Fatal("uninitialized replica calculated gradients")
		}
		if _, err := value.Calibration(context.Background(), services.ExperimentNativeExample{}); !errors.Is(err, services.ErrExperimentNativeReplica) {
			t.Fatal("uninitialized replica calibrated")
		}
		if _, err := value.Install(context.Background(), nil); !errors.Is(err, services.ErrExperimentNativeReplica) {
			t.Fatal("uninitialized replica installed parameters")
		}
		if err := value.Close(); err != nil {
			t.Fatal(err)
		}
		if err := value.Close(); err != nil {
			t.Fatal("close is not idempotent")
		}
	}
}
