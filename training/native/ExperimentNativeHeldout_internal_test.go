package native

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tayi-ai/arandu-llama/checkpoint"
	decoder "github.com/tayi-ai/arandu-llama/training/decoder"
	"github.com/tayi-ai/arandu-llama/training/tokenizer"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

type experimentHeldoutRow struct {
	ExampleID       string   `json:"example_id"`
	Subject         string   `json:"subject"`
	Prompt          string   `json:"prompt"`
	Choices         []string `json:"choices"`
	ReferenceResult string   `json:"reference_result"`
	SourceRevision  string   `json:"source_revision"`
	Split           string   `json:"split"`
	VerifierID      string   `json:"verifier_id"`
	MiningStatus    string   `json:"mining_status"`
}

type experimentHeldoutScore struct {
	ExampleID               string     `json:"example_id"`
	Subject                 string     `json:"subject"`
	Reference               string     `json:"reference"`
	BasePrediction          string     `json:"base_prediction"`
	AdaptedPrediction       string     `json:"adapted_prediction"`
	BaseCorrect             bool       `json:"base_correct"`
	AdaptedCorrect          bool       `json:"adapted_correct"`
	BaseMargin              float64    `json:"base_margin"`
	AdaptedMargin           float64    `json:"adapted_margin"`
	BaseLogProbabilities    [4]float64 `json:"base_log_probabilities"`
	AdaptedLogProbabilities [4]float64 `json:"adapted_log_probabilities"`
}

func TestExperimentPinnedHeldout(t *testing.T) {
	root := os.Getenv("TAYI_R18_ARTIFACT_ROOT")
	heldoutPath := os.Getenv("TAYI_R18_HELDOUT_PATH")
	checkpointPath := os.Getenv("TAYI_R18_CHECKPOINT_PATH")
	outputPath := os.Getenv("TAYI_R18_EVAL_OUTPUT")
	if root == "" || heldoutPath == "" || checkpointPath == "" || outputPath == "" {
		t.Skip("R18 heldout evaluator is opt-in")
	}
	experiment, evidence := privateNativeEvidence(t)
	shardIndex, err := strconv.Atoi(os.Getenv("TAYI_R18_SHARD_INDEX"))
	if err != nil {
		t.Fatal(err)
	}
	shardCount, err := strconv.Atoi(os.Getenv("TAYI_R18_SHARD_COUNT"))
	if err != nil || shardCount < 1 || shardIndex < 0 || shardIndex >= shardCount {
		t.Fatal("invalid evaluation shard")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 80*time.Minute)
	defer cancel()
	rows, err := experimentReadHeldout(heldoutPath, evidence.SourceRevision)
	if err != nil || len(rows) != 512 {
		t.Fatalf("heldout: rows=%d err=%v", len(rows), err)
	}
	body, err := os.ReadFile(heldoutPath)
	if err != nil || experimentHeldoutDigest(body) != evidence.HeldoutSHA256 {
		t.Fatalf("heldout identity differs: %v", err)
	}
	selected := make([]experimentHeldoutRow, 0, (len(rows)+shardCount-1)/shardCount)
	for index, row := range rows {
		if index%shardCount == shardIndex {
			selected = append(selected, row)
		}
	}
	if len(selected) == 0 {
		t.Fatal("empty evaluation shard")
	}

	manifestPath := filepath.Join(root, experiment.config.BundleName, "manifest.json")
	manifestBody, err := experimentReadFile(manifestPath, 256<<10)
	if err != nil || experimentHash(manifestBody) != evidence.TrainRelease {
		t.Fatalf("R18 train release identity differs: %v", err)
	}
	release := ExperimentNativeRelease{ManifestSHA256: evidence.TrainRelease}
	bundle, err := experimentLoadNativeBundle(experiment, root, "train", release)
	if err != nil {
		t.Fatal(err)
	}
	bundlePath := filepath.Join(root, experiment.config.BundleName)
	trainingBytes, err := experimentReadFile(filepath.Join(bundlePath, "training.json"), 32<<20)
	if err != nil {
		t.Fatal(err)
	}
	data, err := ReadExperimentTraining(experiment, strings.NewReader(string(trainingBytes)))
	if err != nil {
		t.Fatal(err)
	}
	if err := experimentValidateDiagnosticDisjoint(rows, data); err != nil {
		t.Fatal(err)
	}
	codecBytes, err := experimentReadFile(filepath.Join(bundlePath, "tokenizer.json"), 32<<20)
	if err != nil {
		t.Fatal(err)
	}
	codec, err := tokenizer.Load(ctx, strings.NewReader(string(codecBytes)), bundle.manifest.Files["tokenizer.json"], tokenizer.DefaultLimits())
	if err != nil {
		t.Fatal(err)
	}

	maximum := int64(0)
	for _, row := range selected {
		prompt, err := experimentRenderHeldoutPrompt(row, data.Demos)
		if err != nil {
			t.Fatal(err)
		}
		ids, err := codec.Encode(ctx, prompt)
		if err != nil {
			t.Fatal(err)
		}
		if n := int64(len(ids) + 1); n > maximum {
			maximum = n
		}
	}
	if maximum < 2 || maximum > 4096 {
		t.Fatalf("invalid shard token maximum %d", maximum)
	}
	memory, err := ExperimentNativeMemory(maximum)
	if err != nil {
		t.Fatal(err)
	}
	available, err := diagnosticHostAvailableRAM("/proc/meminfo")
	if err != nil || available < memory.HostPreloadBytes {
		t.Fatalf("host RAM admission failed: available=%d required=%d err=%v", available, memory.HostPreloadBytes, err)
	}
	probeCtx, probeCancel := context.WithTimeout(ctx, 15*time.Second)
	probe, err := diagnosticProbeGPUs(probeCtx, .80)
	probeCancel()
	if err != nil || len(probe.GPUs) != 2 {
		t.Fatalf("GPU admission failed: %+v err=%v", probe, err)
	}

	index, err := experimentReadFile(filepath.Join(bundlePath, "model.safetensors.index.json"), 32<<20)
	if err != nil {
		t.Fatal(err)
	}
	config, err := experimentReadFile(filepath.Join(bundlePath, "config.json"), 32<<20)
	if err != nil {
		t.Fatal(err)
	}
	reference, err := experimentReadFile(filepath.Join(bundlePath, "initial-reference.json"), 32<<20)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := decoder.PlanTextAssembly(index, config, reference, decoder.AssemblyIdentity{
		IndexSHA256:          bundle.manifest.Files["model.safetensors.index.json"],
		ConfigSHA256:         bundle.manifest.Files["config.json"],
		ReferenceSHA256:      bundle.manifest.Files["initial-reference.json"],
		InitialAdapterSHA256: experiment.config.Recipe.LoRA.ExpectedInitialDigest,
	}, memory.assemblyLimits(experiment))
	if err != nil {
		t.Fatal(err)
	}
	baseBytes, err := experimentReadFile(filepath.Join(bundlePath, "base-manifest.json"), 32<<20)
	if err != nil {
		t.Fatal(err)
	}
	sources, err := experimentNativeVerifySources(experiment, ctx, bundle.contract.BasePath, baseBytes)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := decoder.InspectAssemblySources(ctx, plan, sources); err != nil {
		t.Fatal(err)
	}

	count, err := torch.CUDADeviceCount()
	if err != nil || count != 2 {
		t.Fatalf("CUDA device count=%d err=%v", count, err)
	}
	for device := 0; device < 2; device++ {
		if err := torch.SetCUDAMemoryFraction(device, .80); err != nil {
			t.Fatal(err)
		}
	}
	initial, err := decoder.InitializeAdapter(ctx, experiment.config.Initializer)
	if err != nil {
		t.Fatal(err)
	}
	defer initial.Close()
	loaded, err := decoder.LoadTextAssembly(ctx, plan, sources, initial)
	if err != nil {
		t.Fatal(err)
	}
	replica, err := NewExperimentDecoderReplica(experiment, loaded, codec, plan, ExperimentDecoderReplicaOptions{
		Limits:         decoder.Limits{MaxTokens: maximum, LogitRows: 2, MaxCheckpointBytes: memory.CheckpointBytes},
		HashChunkBytes: 4 << 20, ParameterCopyBytes: 557056 * 4 * 3,
		SourceManifestSHA256: experiment.config.Recipe.Model.ManifestSHA256,

		ExpectedRotaryFrequencySHA256: experiment.config.RotarySHA256,
	})
	if err != nil {
		_ = loaded.Close()
		t.Fatal(err)
	}
	defer replica.Close()
	initialReceipt, err := replica.Inspect(ctx)
	if err != nil || initialReceipt.AdapterSHA256 !=
		experiment.config.Recipe.LoRA.ExpectedInitialDigest {
		t.Fatalf("initial replica identity differs: %+v err=%v", initialReceipt, err)
	}

	scores := make([]experimentHeldoutScore, len(selected))
	for i, row := range selected {
		base, err := experimentScoreHeldout(ctx, replica, row, data.Demos)
		if err != nil {
			t.Fatalf("base score %s: %v", row.ExampleID, err)
		}
		scores[i] = experimentHeldoutScore{
			ExampleID: row.ExampleID, Subject: row.Subject, Reference: row.ReferenceResult,
			BasePrediction: base.Prediction, BaseCorrect: base.Correct, BaseMargin: base.Margin,
			BaseLogProbabilities: base.LogProbabilities,
		}
	}
	parameters, err := experimentReadHeldoutCheckpoint(experiment, evidence, checkpointPath)
	if err != nil {
		t.Fatal(err)
	}
	installed, err := replica.Install(ctx, parameters)
	clear(parameters)
	if err != nil || installed != evidence.FinalAdapter {
		t.Fatalf("final adapter install identity=%s err=%v", installed, err)
	}
	for i, row := range selected {
		adapted, err := experimentScoreHeldout(ctx, replica, row, data.Demos)
		if err != nil {
			t.Fatalf("adapted score %s: %v", row.ExampleID, err)
		}
		scores[i].AdaptedPrediction = adapted.Prediction
		scores[i].AdaptedCorrect = adapted.Correct
		scores[i].AdaptedMargin = adapted.Margin
		scores[i].AdaptedLogProbabilities = adapted.LogProbabilities
	}
	if err := experimentWriteHeldoutScores(outputPath, scores); err != nil {
		t.Fatal(err)
	}
	t.Logf("R18 heldout shard %d/%d complete: rows=%d", shardIndex, shardCount, len(scores))
}

type experimentHeldoutSingle struct {
	Prediction       string
	Correct          bool
	Margin           float64
	LogProbabilities [4]float64
}

func experimentReadHeldout(path, sourceRevision string) ([]experimentHeldoutRow, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	rows := make([]experimentHeldoutRow, 0, 512)
	seen := make(map[string]bool, 512)
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64<<10), 1<<20)
	previous := ""
	for scanner.Scan() {
		var row experimentHeldoutRow
		if err := json.Unmarshal(scanner.Bytes(), &row); err != nil {
			return nil, err
		}
		if !strings.HasPrefix(row.ExampleID, "mmlu-heldout-") || seen[row.ExampleID] ||
			row.ExampleID <= previous || row.SourceRevision != sourceRevision ||
			row.Split != "heldout" || row.MiningStatus != "heldout_never_mined" ||
			row.VerifierID != "multiple_choice.exact_letter.v1" || len(row.Choices) != 4 {
			return nil, errors.New("invalid frozen heldout row")
		}
		if _, err := experimentLabelIndex(row.ReferenceResult); err != nil {
			return nil, err
		}
		seen[row.ExampleID] = true
		previous = row.ExampleID
		rows = append(rows, row)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return rows, nil
}

func experimentRenderHeldoutPrompt(row experimentHeldoutRow, demos []ExperimentTrainingRow) (string, error) {
	if len(demos) != 4 || !strings.HasSuffix(row.Prompt, experimentAnswerSuffix) {
		return "", errors.New("invalid heldout prompt protocol")
	}
	for _, choice := range row.Choices {
		if strings.TrimSpace(choice) == "" {
			return "", errors.New("empty heldout choice")
		}
	}
	chunks := make([]string, 0, 5)
	for _, demo := range demos {
		if err := validateExperimentQuestion(demo, "calibration"); err != nil {
			return "", err
		}
		chunks = append(chunks, strings.TrimSuffix(demo.Prompt, experimentAnswerSuffix)+"\nAnswer: "+demo.ReferenceResult)
	}
	chunks = append(chunks, strings.TrimSuffix(row.Prompt, experimentAnswerSuffix)+"\nAnswer:")
	return strings.Join(chunks, "\n\n"), nil
}

func experimentScoreHeldout(ctx context.Context, replica *ExperimentDecoderReplica, row experimentHeldoutRow, demos []ExperimentTrainingRow) (result experimentHeldoutSingle, err error) {
	if replica == nil {
		return result, ErrExperimentNativeReplica
	}
	replica.mu.Lock()
	defer replica.mu.Unlock()
	if err := replica.ready(ctx, true); err != nil {
		return result, err
	}
	prompt, err := experimentRenderHeldoutPrompt(row, demos)
	if err != nil {
		return result, err
	}

	example := ExperimentNativeExample{
		ExampleID: row.ExampleID, Prompt: prompt,
		CandidateVocabularyIDs: ExperimentNativeRecipe(replica.experiment).Data.CandidateVocabularyIDs,
		LogitRows:              2,
	}
	ids, restore, err := replica.prepare(ctx, example)
	if err != nil {
		return result, err
	}
	var snapshot *decoder.Snapshot
	var logitsRow *torch.Tensor
	defer func() {
		if logitsRow != nil {
			err = errors.Join(err, logitsRow.Close())
		}
		if snapshot != nil {
			err = errors.Join(err, snapshot.Close())
		}
		err = errors.Join(err, restore())
	}()
	snapshot, err = replica.loaded.Model.Forward(ctx, ids, replica.options.Limits)
	if err != nil {
		return result, err
	}
	info, err := snapshot.Logits.Info()
	if err != nil || len(info.Shape) != 3 || info.Shape[0] != 1 || info.Shape[1] != 2 || info.Shape[2] != 248320 || info.DType != torch.Float32 {
		return result, errors.Join(ErrExperimentNativeReplica, err)
	}
	logitsRow, err = snapshot.Logits.Select(1, 0)
	if err != nil {
		return result, err
	}
	values, err := logitsRow.Float32Values()
	if err != nil {
		return result, err
	}
	result.LogProbabilities, err = ExperimentNativeVocabularyLogProbabilities(replica.experiment, values)
	if err != nil {
		return result, err
	}

	correct, err := experimentLabelIndex(row.ReferenceResult)
	if err != nil {
		return result, err
	}
	best := 0
	for i := 1; i < 4; i++ {
		if result.LogProbabilities[i] > result.LogProbabilities[best] {
			best = i
		}
	}
	competitor := math.Inf(-1)
	for i := 0; i < 4; i++ {
		if i != correct && result.LogProbabilities[i] > competitor {
			competitor = result.LogProbabilities[i]
		}
	}
	result.Prediction = string(rune('A' + best))
	result.Correct = best == correct
	result.Margin = result.LogProbabilities[correct] - competitor
	if math.IsNaN(result.Margin) || math.IsInf(result.Margin, 0) {
		return experimentHeldoutSingle{}, errors.New("nonfinite heldout margin")
	}
	return result, ctx.Err()
}

func experimentReadHeldoutCheckpoint(experiment *NativeExperiment, evidence nativeDiagnosticEvidence, path string) ([]float32, error) {
	digest, err := experimentNativeFileHash(path, 16<<20, false)
	if err != nil || digest != evidence.FinalArtifact {
		return nil, errors.New("R18 checkpoint artifact identity differs")
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return nil, err
	}
	index, err := checkpoint.OpenSafetensors(file, info.Size(), experimentCheckpointLimits())
	if err != nil {
		return nil, err
	}
	layout := ExperimentNativeParameterLayout()
	values := make([]float32, 0, ExperimentNativeRecipe(experiment).LoRA.Parameters)
	for _, expected := range layout {
		name := experimentCheckpointExportName(expected.Name)
		tensor, ok := index.Tensor(name)
		if !ok || tensor.DType != "F32" || len(tensor.Shape) != 2 ||
			tensor.Shape[0] != uint64(expected.Shape[0]) || tensor.Shape[1] != uint64(expected.Shape[1]) {
			return nil, ErrExperimentNativeCheckpoint
		}
		reader, err := index.TensorReader(name)
		if err != nil {
			return nil, err
		}
		body := make([]byte, tensor.Size())
		if _, err := io.ReadFull(reader, body); err != nil {
			return nil, err
		}

		if len(body)%4 != 0 {
			return nil, ErrExperimentNativeCheckpoint
		}
		for offset := 0; offset < len(body); offset += 4 {
			value := math.Float32frombits(binary.LittleEndian.Uint32(body[offset : offset+4]))
			if !experimentFinite32(value) {
				return nil, ErrExperimentNativeCheckpoint
			}
			values = append(values, value)
		}
	}
	if len(values) != ExperimentNativeRecipe(experiment).LoRA.Parameters {
		return nil, ErrExperimentNativeCheckpoint
	}
	adapterDigest, err := ExperimentNativeParameterDigest(experiment, values)
	if err != nil || adapterDigest != evidence.FinalAdapter {
		return nil, errors.Join(ErrExperimentNativeCheckpoint, err)
	}
	return values, nil
}

func experimentWriteHeldoutScores(path string, scores []experimentHeldoutScore) error {
	if len(scores) == 0 {
		return errors.New("empty heldout score set")
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(file)
	encoder.SetEscapeHTML(false)
	for _, score := range scores {
		if err := encoder.Encode(score); err != nil {
			_ = file.Close()
			return err
		}
	}
	return errors.Join(file.Sync(), file.Close())
}

func experimentHeldoutDigest(body []byte) string {
	sum := sha256.Sum256(body)
	return hex.EncodeToString(sum[:])
}

func experimentValidateDiagnosticDisjoint(rows []experimentHeldoutRow, data ExperimentTrainingData) error {
	training := make(map[string]bool, len(data.Examples)+len(data.Demos))
	for _, row := range append(append([]ExperimentTrainingRow(nil), data.Demos...), data.Examples...) {
		index := strings.LastIndexByte(row.ExampleID, '-')
		if index < 0 || index == len(row.ExampleID)-1 {
			return errors.New("invalid training question identity")
		}
		training[row.ExampleID[index+1:]] = true
	}
	subjects := make(map[string]bool, 57)
	for _, row := range rows {
		index := strings.LastIndexByte(row.ExampleID, '-')
		if index < 0 || index == len(row.ExampleID)-1 || training[row.ExampleID[index+1:]] {
			return errors.New("diagnostic heldout overlaps R18 training question identity")
		}
		subjects[row.Subject] = true
	}
	if len(rows) != 512 || len(subjects) != 57 {
		return errors.New("diagnostic heldout coverage differs")
	}
	return nil
}
