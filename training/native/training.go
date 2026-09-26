package native

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"reflect"
	"slices"
	"sync"
	"time"
)

// ErrExperimentNativeTraining identifies a rejected native training operation.
var ErrExperimentNativeTraining = errors.New("experiment: native training rejected")

// ExperimentNativeParameter identifies a contiguous FP32 slice in canonical order.
type ExperimentNativeParameter struct {
	Name  string  `json:"name"`
	Shape []int64 `json:"shape"`
}

// ExperimentNativeReplicaReceipt records measured identities, not qualifications
// inferred from geometry. Inspect must measure the frozen base and live adapter;
// the concrete backend owns source verification and resource/GPU admission.
type ExperimentNativeReplicaReceipt struct {
	AdapterSHA256        string                      `json:"adapter_sha256"`
	BaseSHA256           string                      `json:"base_sha256"`
	SourceManifestSHA256 string                      `json:"source_manifest_sha256"`
	Layout               []ExperimentNativeParameter `json:"layout"`
}

// ExperimentNativeExample contains the existing four-demonstration raw prompt.
// The replica encodes Prompt and appends the A vocabulary ID exactly once. It
// retains two logit rows and scores the first row, in the specified A/B/C/D order.
type ExperimentNativeExample struct {
	ExampleID              string
	Prompt                 string
	CandidateVocabularyIDs [4]int
	LogitRows              int
}

// ExperimentNativeLossGradient supplies the already scaled derivative of four raw
// logits. The replica must call it exactly once and perform no further scaling,
// averaging, clipping, loss computation or optimizer update.
type ExperimentNativeLossGradient func([4]float64) ([4]float64, error)

// ExperimentNativeGradient owns the replica's scaled FP32 gradient in receipt order.
// InputSequenceLength includes the appended A ID; Logits equals callback input.
type ExperimentNativeGradient struct {
	Logits              [4]float64
	ValuesF32           []float32
	InputSequenceLength int64
}

// ExperimentNativeReplica provides model operations without transport or optimizer
// policy. Methods are synchronous. Install returns the measured adapter digest
// after copying every parameter; a partial install is an error, never retried.
// Run takes ownership of the replica and closes it on every return.
type ExperimentNativeReplica interface {
	Inspect(context.Context) (ExperimentNativeReplicaReceipt, error)
	Gradient(context.Context, ExperimentNativeExample, ExperimentNativeLossGradient) (ExperimentNativeGradient, error)
	Install(context.Context, []float32) (string, error)
	Close() error
}

// ExperimentGradientExchange matches the existing native collective's operation.
// It gathers all 20 scaled local sums, invokes the rank-zero AdamW callback once,
// and returns identical new parameters after broadcast admission. A failure
// aborts the session, including any optimizer update already applied remotely.
type ExperimentGradientExchange interface {
	Exchange(context.Context, uint32, []float32) ([]float32, error)
	Close() error
}

// ExperimentNativeCheckpoint supplies the installed adapter to a rank-zero writer.
// Slices are borrowed only during WriteCheckpoint and must not be modified or
// retained. Optimizer state, when persisted, belongs to the caller's optimizer.
type ExperimentNativeCheckpoint struct {
	Step           int
	Candidate      bool
	AdapterSHA256  string
	ScheduleSHA256 string
	Layout         []ExperimentNativeParameter
	Parameters     []float32
}

// ExperimentNativeCheckpointReceipt identifies a durably written checkpoint artifact.
// ArtifactSHA256 hashes the writer's actual bytes; AdapterSHA256 hashes tensor
// names and FP32 content, which is a different identity from the file hash.
type ExperimentNativeCheckpointReceipt struct {
	Step           int    `json:"step"`
	Candidate      bool   `json:"candidate"`
	AdapterSHA256  string `json:"adapter_sha256"`
	ArtifactSHA256 string `json:"artifact_sha256"`
}

// ExperimentNativeCheckpointWriter owns checkpoint I/O. Only rank zero calls it.
type ExperimentNativeCheckpointWriter interface {
	WriteCheckpoint(context.Context, ExperimentNativeCheckpoint) (ExperimentNativeCheckpointReceipt, error)
}

// ExperimentNativeProgress reports an installed update, after its required checkpoint
// succeeds. A callback failure stops the run and prevents the next gradient.
type ExperimentNativeProgress struct {
	Rank                   int                               `json:"rank"`
	Node                   string                            `json:"node"`
	Step                   int                               `json:"step"`
	Examples               []ExperimentNativeExampleProgress `json:"examples"`
	InputSequenceLength    int64                             `json:"input_sequence_length"`
	TotalSequencePositions int64                             `json:"total_sequence_positions"`
	AdapterSHA256          string                            `json:"adapter_sha256"`
	CoverageSHA256         string                            `json:"local_coverage_sha256"`
}

// ExperimentNativeExampleProgress records one of the update's sequential examples.
// Loss and logits are per example; they are never averaged as progress metadata.
type ExperimentNativeExampleProgress struct {
	ExampleID           string         `json:"example_id"`
	InputSequenceLength int64          `json:"input_sequence_length"`
	Logits              [4]float64     `json:"logits_abcd"`
	Loss                ExperimentLoss `json:"loss"`
}

// ExperimentNativeTrainingResult separates planned global coverage from this rank's
// measured installed updates. Completed never means the other ranks completed;
// the owning fleet job must reconcile all 20 results. Errors preserve partial
// installed counts and receipts, but Completed remains false.
type ExperimentNativeTrainingResult struct {
	Rank                    int                                 `json:"rank"`
	Node                    string                              `json:"node"`
	World                   int                                 `json:"world"`
	Completed               bool                                `json:"completed"`
	InstalledUpdates        int                                 `json:"installed_updates"`
	LocalExamples           int                                 `json:"local_examples"`
	SequencePositions       int64                               `json:"sequence_positions"`
	GlobalScheduledExamples int                                 `json:"global_scheduled_examples"`
	ScheduleSHA256          string                              `json:"schedule_sha256"`
	LocalCoverageSHA256     string                              `json:"local_coverage_sha256"`
	Initial                 ExperimentNativeReplicaReceipt      `json:"initial"`
	Final                   ExperimentNativeReplicaReceipt      `json:"final"`
	Checkpoints             []ExperimentNativeCheckpointReceipt `json:"checkpoints"`
}

// ExperimentNativeTraining is the single-use application case for the frozen epoch.
// It performs no file opening, tokenization, GPU admission, network connection,
// job launch, retry or resume. Those collaborators must be qualified and admitted
// by the caller before Run; constructing this case does not qualify a backend.
// The fixed training deadline propagates to collaborators; the owning job must
// still bound any collaborator that fails to honor context cancellation.
type ExperimentNativeTraining struct {
	experiment *NativeExperiment

	mu       sync.Mutex
	used     bool
	rank     int
	world    int
	replica  ExperimentNativeReplica
	exchange ExperimentGradientExchange
	writer   ExperimentNativeCheckpointWriter
	progress func(context.Context, ExperimentNativeProgress) error
}

// NewExperimentNativeTraining fixes one rank and its collaborators. A successful Run
// or failed Run consumes the case permanently. Rank zero requires a writer;
// other ranks never invoke one. Progress is optional and synchronous.
func NewExperimentNativeTraining(experiment *NativeExperiment, rank, world int, replica ExperimentNativeReplica, exchange ExperimentGradientExchange, writer ExperimentNativeCheckpointWriter, progress func(context.Context, ExperimentNativeProgress) error) (*ExperimentNativeTraining, error) {
	if experiment == nil {
		return nil, errors.New("native experiment: installation is not configured")
	}

	if world != experimentReplicas || rank < 0 || rank >= world || experimentNativeNil(replica) || experimentNativeNil(exchange) || (rank == 0 && experimentNativeNil(writer)) {
		return nil, fmt.Errorf("%w: exactly 20 ranks and explicit collaborators required", ErrExperimentNativeTraining)
	}
	return &ExperimentNativeTraining{experiment: experiment, rank: rank, world: world, replica: replica, exchange: exchange, writer: writer, progress: progress}, nil
}

// RunFrozen reads and hash-admits the original corpus using ReadExperimentTraining.
// The caller owns the reader. Even a read/admission failure closes collaborators
// and consumes the case, so no technical retry can occur through this object.
func (training *ExperimentNativeTraining) RunFrozen(ctx context.Context, reader io.Reader) (ExperimentNativeTrainingResult, error) {
	return training.execute(ctx, func() (ExperimentTrainingData, error) {
		return ReadExperimentTraining(training.experiment, reader)
	})
}

// Run consumes already admitted typed data from ReadExperimentTraining. It checks
// identity fields and all semantic constraints but cannot attest the original
// bytes from a struct: the caller owns that provenance. Prefer RunFrozen at I/O
// boundaries. The caller must not mutate data while this synchronous call runs.
func (training *ExperimentNativeTraining) Run(ctx context.Context, data ExperimentTrainingData) (ExperimentNativeTrainingResult, error) {
	return training.execute(ctx, func() (ExperimentTrainingData, error) { return data, nil })
}

func (training *ExperimentNativeTraining) execute(ctx context.Context, read func() (ExperimentTrainingData, error)) (result ExperimentNativeTrainingResult, err error) {
	if training == nil {
		return result, fmt.Errorf("%w: nil training case", ErrExperimentNativeTraining)
	}
	training.mu.Lock()
	if training.used {
		training.mu.Unlock()
		return result, fmt.Errorf("%w: training case already consumed", ErrExperimentNativeTraining)
	}
	training.used = true
	training.mu.Unlock()
	if training.world != experimentReplicas || training.rank < 0 || training.rank >= training.world || experimentNativeNil(training.replica) || experimentNativeNil(training.exchange) || (training.rank == 0 && experimentNativeNil(training.writer)) {
		return result, fmt.Errorf("%w: training case is not initialized", ErrExperimentNativeTraining)
	}
	defer func() {
		err = errors.Join(err, training.exchange.Close(), training.replica.Close())
		if err != nil {
			result.Completed = false
		}
	}()
	if ctx == nil {
		return result, fmt.Errorf("%w: context required", ErrExperimentNativeTraining)
	}
	if err := ctx.Err(); err != nil {
		return result, err
	}
	recipe := ExperimentNativeRecipe(training.experiment)
	ctx, cancel := context.WithTimeout(ctx, time.Duration(recipe.Guards.TrainSeconds)*time.Second)
	defer cancel()
	data, err := read()
	if err != nil {
		return result, err
	}
	if err := experimentNativeData(training.experiment, data, recipe); err != nil {
		return result, err
	}
	schedule, err := ExperimentTrainingSchedule(training.experiment, data.Examples)
	if err != nil {
		return result, err
	}
	result = ExperimentNativeTrainingResult{Rank: training.rank, Node: recipe.Distribution.Nodes[training.rank], World: training.world,
		GlobalScheduledExamples: len(data.Examples), ScheduleSHA256: experimentNativeJSONDigest(schedule)}
	rows := make(map[string]ExperimentTrainingRow, len(data.Examples))
	for _, row := range data.Examples {
		rows[row.ExampleID] = row
	}
	if err := ctx.Err(); err != nil {
		return result, err
	}
	initial, err := training.replica.Inspect(ctx)
	if err != nil {
		return result, err
	}
	if err := experimentNativeReceipt(initial, recipe); err != nil {
		return result, err
	}
	if initial.AdapterSHA256 != recipe.LoRA.ExpectedInitialDigest {
		return result, fmt.Errorf("%w: initial adapter identity differs", ErrExperimentNativeTraining)
	}
	result.Initial = experimentNativeCopyReceipt(initial)
	installedDigest := initial.AdapterSHA256
	coverage := make([]ExperimentTrainingAssignment, 0, experimentUpdates+1)
	for _, batch := range schedule {
		if err := ctx.Err(); err != nil {
			return result, err
		}
		assignments := batch.Assignments[training.rank]
		var localSum []float32
		var localTokens int64
		observations := make([]ExperimentNativeExampleProgress, 0, len(assignments.Examples))
		for _, assignment := range assignments.Examples {
			if err := ctx.Err(); err != nil {
				return result, err
			}
			gradient, observation, err := training.exampleGradient(ctx, rows[assignment.ExampleID], data.Demos, recipe)
			if err != nil {
				return result, err
			}
			if gradient.InputSequenceLength > math.MaxInt64-result.SequencePositions-localTokens {
				return result, fmt.Errorf("%w: token count overflow", ErrExperimentNativeTraining)
			}
			localSum, err = ExperimentNativeAccumulateGradient(training.experiment, localSum, gradient.ValuesF32)
			if err != nil {
				return result, err
			}
			localTokens += gradient.InputSequenceLength
			observations = append(observations, observation)
		}
		if err := ctx.Err(); err != nil {
			return result, err
		}
		parameters, err := training.exchange.Exchange(ctx, uint32(batch.Update), localSum)
		if err != nil {
			return result, err
		}
		digest, err := ExperimentNativeParameterDigest(training.experiment, parameters)
		if err != nil {
			return result, err
		}
		if err := ctx.Err(); err != nil {
			return result, err
		}
		installedDigest, err = training.replica.Install(ctx, parameters)
		if err != nil {
			return result, err
		}
		if installedDigest != digest {
			return result, fmt.Errorf("%w: installed parameters differ from exchange", ErrExperimentNativeTraining)
		}
		coverage = append(coverage, assignments.Examples...)
		result.InstalledUpdates = batch.Update
		result.LocalExamples += len(assignments.Examples)
		result.SequencePositions += localTokens
		result.LocalCoverageSHA256 = experimentNativeJSONDigest(coverage)
		if err := ctx.Err(); err != nil {
			return result, err
		}
		if training.rank == 0 && slices.Contains(recipe.Optimization.SaveSteps[:], batch.Update) {
			request := ExperimentNativeCheckpoint{Step: batch.Update, Candidate: batch.Update == recipe.Optimization.CandidateStep,
				AdapterSHA256: digest, ScheduleSHA256: result.ScheduleSHA256, Layout: ExperimentNativeParameterLayout(), Parameters: parameters}
			receipt, err := training.writer.WriteCheckpoint(ctx, request)
			if err != nil {
				return result, err
			}
			if receipt.Step != request.Step || receipt.Candidate != request.Candidate || receipt.AdapterSHA256 != digest || !experimentNativeHash(receipt.ArtifactSHA256) {
				return result, fmt.Errorf("%w: checkpoint receipt differs", ErrExperimentNativeTraining)
			}
			result.Checkpoints = append(result.Checkpoints, receipt)
		}
		if err := ctx.Err(); err != nil {
			return result, err
		}
		if training.progress != nil {
			if err := training.progress(ctx, ExperimentNativeProgress{Rank: training.rank, Node: assignments.Node, Step: batch.Update, Examples: observations,
				InputSequenceLength: localTokens, TotalSequencePositions: result.SequencePositions, AdapterSHA256: digest, CoverageSHA256: result.LocalCoverageSHA256}); err != nil {
				return result, err
			}
		}
	}
	if err := ctx.Err(); err != nil {
		return result, err
	}
	final, err := training.replica.Inspect(ctx)
	if err != nil {
		return result, err
	}
	result.Final = experimentNativeCopyReceipt(final)
	if err := experimentNativeReceipt(final, recipe); err != nil {
		return result, err
	}
	if final.AdapterSHA256 != installedDigest || final.AdapterSHA256 == initial.AdapterSHA256 || final.BaseSHA256 != initial.BaseSHA256 {
		return result, fmt.Errorf("%w: final adapter or frozen base identity differs", ErrExperimentNativeTraining)
	}
	if err := ctx.Err(); err != nil {
		return result, err
	}
	result.Completed = true
	return result, nil
}

func (training *ExperimentNativeTraining) exampleGradient(ctx context.Context, row ExperimentTrainingRow, demos []ExperimentTrainingRow, recipe ExperimentRecipe) (ExperimentNativeGradient, ExperimentNativeExampleProgress, error) {
	observation := ExperimentNativeExampleProgress{ExampleID: row.ExampleID}
	prompt, err := RenderExperimentTrainingPrompt(row, demos)
	if err != nil {
		return ExperimentNativeGradient{}, observation, err
	}
	var callbackErr error
	calls := 0
	gradient, err := training.replica.Gradient(ctx, ExperimentNativeExample{ExampleID: row.ExampleID, Prompt: prompt,
		CandidateVocabularyIDs: recipe.Data.CandidateVocabularyIDs, LogitRows: 2}, func(logits [4]float64) ([4]float64, error) {
		calls++
		if calls != 1 {
			callbackErr = fmt.Errorf("%w: derivative callback must run once", ErrExperimentNativeTraining)
			return [4]float64{}, callbackErr
		}
		if err := ctx.Err(); err != nil {
			callbackErr = err
			return [4]float64{}, callbackErr
		}
		observation.Loss, callbackErr = MultiTeacherLoss(training.experiment, logits, row)
		if callbackErr != nil {
			return [4]float64{}, callbackErr
		}
		observation.Logits = logits
		scaled := observation.Loss.LogitGradient
		for i := range scaled {
			scaled[i] *= ExperimentAdamWLossScale
		}
		return scaled, nil
	})
	if err != nil || callbackErr != nil {
		return ExperimentNativeGradient{}, observation, errors.Join(err, callbackErr)
	}
	if calls != 1 || gradient.Logits != observation.Logits || gradient.InputSequenceLength < 2 {
		return ExperimentNativeGradient{}, observation, fmt.Errorf("%w: gradient observation or token count differs", ErrExperimentNativeTraining)
	}
	observation.InputSequenceLength = gradient.InputSequenceLength
	return gradient, observation, nil
}

// ExperimentNativeAccumulateGradient copies the first scaled gradient, then sums
// subsequent contributions in FP32 without averaging or unscaling. The returned
// accumulator belongs to the caller and may be passed back as sum; next is never
// retained or changed. A rejected sum must be discarded. This keeps sequential
// example graphs separate even when a replica reuses its output gradient buffer.
func ExperimentNativeAccumulateGradient(experiment *NativeExperiment, sum, next []float32) ([]float32, error) {
	if experiment == nil {
		return nil, errors.New("native experiment: installation is not configured")
	}

	if err := experimentNativeValues(experiment, next); err != nil {
		return nil, err
	}
	if sum == nil {
		return slices.Clone(next), nil
	}
	if err := experimentNativeValues(experiment, sum); err != nil {
		return nil, err
	}
	for i, value := range next {
		sum[i] = float32(sum[i] + value)
		if !experimentFinite32(sum[i]) {
			return nil, fmt.Errorf("%w: local FP32 gradient sum overflow", ErrExperimentNativeTraining)
		}
	}
	return sum, nil
}

// ExperimentNativeAdamWUpdate adapts the application's existing optimizer to the
// collective's rank-ordered callback. It performs no additional scale/mean/clip
// arithmetic. The supplied optimizer must start at step zero with the fixed
// 557056 FP32 parameters; restoring an optimizer is not admitted here.
func ExperimentNativeAdamWUpdate(experiment *NativeExperiment, optimizer *ExperimentAdamW) (func([][]float32) ([]float32, error), error) {
	if experiment == nil {
		return nil, errors.New("native experiment: installation is not configured")
	}

	state := optimizer.Snapshot()
	if state.Step != 0 || state.NumericMode != ExperimentAdamWNumericMode || len(state.Parameters) != ExperimentNativeRecipe(experiment).LoRA.Parameters {
		return nil, fmt.Errorf("%w: fresh fixed-geometry optimizer required", ErrExperimentNativeTraining)
	}
	return func(ordered [][]float32) ([]float32, error) {
		if len(ordered) != ExperimentAdamWReplicas {
			return nil, fmt.Errorf("%w: exactly 20 ordered gradient sums required", ErrExperimentNativeTraining)
		}
		ranks := make([]ExperimentRankGradient, len(ordered))
		for rank, values := range ordered {
			ranks[rank] = ExperimentRankGradient{Rank: rank, Values: values}
		}
		if _, err := optimizer.Update(ranks); err != nil {
			return nil, err
		}
		return optimizer.Snapshot().Parameters, nil
	}, nil
}

// ExperimentNativeParameterLayout returns the fixed 32-tensor layer/q-v/A-B order.
// Shapes and names are new caller-owned Go values on every call.
func ExperimentNativeParameterLayout() []ExperimentNativeParameter {
	result := make([]ExperimentNativeParameter, 0, 32)
	for layer := 3; layer < 32; layer += 4 {
		for _, projection := range []struct {
			name string
			out  int64
		}{{"q_proj", 8192}, {"v_proj", 1024}} {
			for _, letter := range []string{"A", "B"} {
				shape := []int64{4, 4096}
				if letter == "B" {
					shape = []int64{projection.out, 4}
				}
				result = append(result, ExperimentNativeParameter{Name: fmt.Sprintf("base_model.model.model.language_model.layers.%d.self_attn.%s.lora_%s.default.weight", layer, projection.name, letter), Shape: shape})
			}
		}
	}
	return result
}

// ExperimentNativeParameterDigest hashes each canonical name followed immediately by
// its contiguous little-endian FP32 content, matching the initial digest format.
// Shape validation uses the fixed layout; nonfinite or incomplete values fail.
func ExperimentNativeParameterDigest(experiment *NativeExperiment, parameters []float32) (string, error) {
	if experiment == nil {
		return "", errors.New("native experiment: installation is not configured")
	}

	if err := experimentNativeValues(experiment, parameters); err != nil {
		return "", err
	}
	digest := sha256.New()
	var buffer [4096]byte
	position := 0
	for _, parameter := range ExperimentNativeParameterLayout() {
		_, _ = digest.Write([]byte(parameter.Name))
		end := position + int(parameter.Shape[0]*parameter.Shape[1])
		for position < end {
			count := min(end-position, len(buffer)/4)
			for i, value := range parameters[position : position+count] {
				binary.LittleEndian.PutUint32(buffer[i*4:], math.Float32bits(value))
			}
			_, _ = digest.Write(buffer[:count*4])
			position += count
		}
	}
	return hex.EncodeToString(digest.Sum(nil)), nil
}

func experimentNativeValues(experiment *NativeExperiment, values []float32) error {
	if experiment == nil {
		return errors.New("native experiment: installation is not configured")
	}

	if len(values) != ExperimentNativeRecipe(experiment).LoRA.Parameters {
		return fmt.Errorf("%w: expected 557056 FP32 values", ErrExperimentNativeTraining)
	}
	for _, value := range values {
		if !experimentFinite32(value) {
			return fmt.Errorf("%w: nonfinite FP32 value", ErrExperimentNativeTraining)
		}
	}
	return nil
}

func experimentNativeReceipt(receipt ExperimentNativeReplicaReceipt, recipe ExperimentRecipe) error {
	if !experimentNativeHash(receipt.AdapterSHA256) || !experimentNativeHash(receipt.BaseSHA256) || receipt.SourceManifestSHA256 != recipe.Model.ManifestSHA256 || !reflect.DeepEqual(receipt.Layout, ExperimentNativeParameterLayout()) {
		return fmt.Errorf("%w: measured replica identity or parameter layout differs", ErrExperimentNativeTraining)
	}
	return nil
}

func experimentNativeCopyReceipt(receipt ExperimentNativeReplicaReceipt) ExperimentNativeReplicaReceipt {
	receipt.Layout = slices.Clone(receipt.Layout)
	for i := range receipt.Layout {
		receipt.Layout[i].Shape = slices.Clone(receipt.Layout[i].Shape)
	}
	return receipt
}

func experimentNativeData(experiment *NativeExperiment, data ExperimentTrainingData, recipe ExperimentRecipe) error {
	if experiment == nil {
		return errors.New("native experiment: installation is not configured")
	}

	if data.SHA256 != recipe.Data.SHA256 || data.SchemaVersion != 1 || data.DatasetID != recipe.Data.ID || data.Protocol != recipe.Data.Protocol || len(data.Demos) != recipe.Data.Demonstrations {
		return fmt.Errorf("%w: frozen corpus identity differs", ErrExperimentNativeTraining)
	}
	if err := ValidateExperimentTrainingRows(experiment, data.Examples); err != nil {
		return err
	}
	seen := make(map[string]bool, experimentExamples+4)
	for _, row := range data.Examples {
		seen[row.ExampleID] = true
	}
	for _, demo := range data.Demos {
		if err := validateExperimentQuestion(demo, "calibration"); err != nil {
			return err
		}
		if seen[demo.ExampleID] {
			return fmt.Errorf("%w: duplicate demonstration identity", ErrExperimentNativeTraining)
		}
		seen[demo.ExampleID] = true
	}
	return nil
}

func experimentNativeHash(value string) bool {
	decoded, err := hex.DecodeString(value)
	return err == nil && len(decoded) == sha256.Size && hex.EncodeToString(decoded) == value
}

func experimentNativeJSONDigest(value any) string {
	// All callers supply fixed structs with only integers, strings and slices.
	body, _ := json.Marshal(value)
	digest := sha256.Sum256(body)
	return hex.EncodeToString(digest[:])
}

func experimentNativeNil(value any) bool {
	if value == nil {
		return true
	}
	v := reflect.ValueOf(value)
	switch v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return v.IsNil()
	default:
		return false
	}
}
