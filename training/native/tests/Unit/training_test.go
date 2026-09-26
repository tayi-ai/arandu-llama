package native_test

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	services "github.com/tayi-ai/arandu-llama/training/native"
)

// These fakes attest identities only to exercise application control flow. They
// do not qualify native arithmetic, source files, resources or actual devices.
type replica struct {
	inspectCalls, installCalls, closes int
	examples                           []services.ExperimentNativeExample
	derivatives                        [][4]float64
	values                             []float32
	digest                             string
	initial                            services.ExperimentNativeReplicaReceipt
	gradientHook                       func(*services.ExperimentNativeGradient)
	installHook                        func() error
	finalHook                          func(*services.ExperimentNativeReplicaReceipt)
	callbackMode                       string
	closeErr                           error
	deadline                           time.Time
}

func newReplica() *replica {
	r := services.ExperimentNativeRecipe(testNativeExperiment())
	return &replica{values: make([]float32, r.LoRA.Parameters), digest: r.LoRA.ExpectedInitialDigest,
		initial: services.ExperimentNativeReplicaReceipt{AdapterSHA256: r.LoRA.ExpectedInitialDigest, BaseSHA256: strings.Repeat("b", 64),
			SourceManifestSHA256: r.Model.ManifestSHA256, Layout: services.ExperimentNativeParameterLayout()}}
}

func (r *replica) Inspect(ctx context.Context) (services.ExperimentNativeReplicaReceipt, error) {
	r.inspectCalls++
	r.deadline, _ = ctx.Deadline()
	receipt := r.initial
	if r.inspectCalls > 1 {
		receipt.AdapterSHA256 = r.digest
		if r.finalHook != nil {
			r.finalHook(&receipt)
		}
	}
	return receipt, nil
}

func (r *replica) Gradient(_ context.Context, example services.ExperimentNativeExample, derivative services.ExperimentNativeLossGradient) (services.ExperimentNativeGradient, error) {
	r.examples = append(r.examples, example)
	logits := [4]float64{.3, -.2, .8, -.7}
	if r.callbackMode == "swallow_nonfinite_error" {
		_, _ = derivative([4]float64{math.NaN()})
		return services.ExperimentNativeGradient{ValuesF32: r.values, InputSequenceLength: 10}, nil
	}
	gradient, err := derivative(logits)
	if err != nil {
		return services.ExperimentNativeGradient{}, err
	}
	if r.callbackMode == "twice" {
		_, _ = derivative(logits)
	}
	r.derivatives = append(r.derivatives, gradient)
	for i, value := range gradient {
		r.values[i] = float32(value)
	}
	result := services.ExperimentNativeGradient{Logits: logits, ValuesF32: r.values, InputSequenceLength: int64(100 + len(example.ExampleID))}
	if r.gradientHook != nil {
		r.gradientHook(&result)
	}
	return result, nil
}

func (r *replica) Install(_ context.Context, parameters []float32) (string, error) {
	r.installCalls++
	if r.installHook != nil {
		if err := r.installHook(); err != nil {
			return "", err
		}
	}
	digest, err := services.ExperimentNativeParameterDigest(testNativeExperiment(), parameters)
	r.digest = digest
	return digest, err
}

func (r *replica) Close() error { r.closes++; return r.closeErr }

type exchange struct {
	steps      []uint32
	gradients  [][4]float32
	parameters []float32
	failStep   uint32
	hook       func([]float32)
	closes     int
}

var failure = errors.New("injected operation failure")

func newExchange() *exchange {
	return &exchange{parameters: make([]float32, services.ExperimentNativeRecipe(testNativeExperiment()).LoRA.Parameters)}
}

func (e *exchange) Exchange(_ context.Context, step uint32, values []float32) ([]float32, error) {
	e.steps = append(e.steps, step)
	e.gradients = append(e.gradients, [4]float32{values[0], values[1], values[2], values[3]})
	if step == e.failStep {
		return nil, failure
	}
	e.parameters[0] = float32(step) * .001
	if e.hook != nil {
		e.hook(e.parameters)
	}
	return e.parameters, nil
}

func (e *exchange) Close() error { e.closes++; return nil }

type writer struct {
	steps    []int
	failStep int
	hook     func(*services.ExperimentNativeCheckpointReceipt)
}

func (w *writer) WriteCheckpoint(_ context.Context, request services.ExperimentNativeCheckpoint) (services.ExperimentNativeCheckpointReceipt, error) {
	w.steps = append(w.steps, request.Step)
	if request.Step == w.failStep {
		return services.ExperimentNativeCheckpointReceipt{}, failure
	}
	digest, err := services.ExperimentNativeParameterDigest(testNativeExperiment(), request.Parameters)
	if err != nil || digest != request.AdapterSHA256 || !reflect.DeepEqual(request.Layout, services.ExperimentNativeParameterLayout()) {
		return services.ExperimentNativeCheckpointReceipt{}, errors.New("writer received inconsistent adapter")
	}
	receipt := services.ExperimentNativeCheckpointReceipt{Step: request.Step, Candidate: request.Candidate,
		AdapterSHA256: digest, ArtifactSHA256: strings.Repeat("c", 64)}
	if w.hook != nil {
		w.hook(&receipt)
	}
	return receipt, nil
}

func syntheticData() services.ExperimentTrainingData {
	r := services.ExperimentNativeRecipe(testNativeExperiment())
	data := services.ExperimentTrainingData{SchemaVersion: 1, DatasetID: r.Data.ID, Protocol: r.Data.Protocol, SHA256: r.Data.SHA256}
	categories := []string{"anchor_preserve", "teacher_a_teacher_b_gain", "teacher_a_only_gain", "teacher_b_only_gain", "persistent_error_all"}
	question := func(id string, index int) services.ExperimentTrainingRow {
		return services.ExperimentTrainingRow{ExampleID: id, Role: "calibration-expansion", Split: "development",
			Prompt:  fmt.Sprintf("Question %d?\nA. one\nB. two\nC. three\nD. four\nAnswer with only the letter A, B, C, or D.", index),
			Choices: []string{"one", "two", "three", "four"}, ReferenceResult: string(rune('A' + index%4)), TeacherCategory: categories[index%len(categories)], TeacherProbabilities: map[string]services.ExperimentCandidateProbabilities{"anchor": services.ExperimentCandidateProbabilities{"A": .1, "B": .2, "C": .3, "D": .4}, "teacher-a": services.ExperimentCandidateProbabilities{"A": .4, "B": .3, "C": .2, "D": .1}, "teacher-b": services.ExperimentCandidateProbabilities{"A": .2, "B": .3, "C": .1, "D": .4}}}
	}
	for i := range r.Data.Examples {
		data.Examples = append(data.Examples, question(fmt.Sprintf("synthetic-%03d", i), i))
	}
	for i := range r.Data.Demonstrations {
		demo := question(fmt.Sprintf("demo-%d", i), i)
		demo.Role = "calibration"
		data.Demos = append(data.Demos, demo)
	}
	return data
}

func trainingCase(t *testing.T, rank int, r *replica, e *exchange, w *writer, progress func(context.Context, services.ExperimentNativeProgress) error) *services.ExperimentNativeTraining {
	t.Helper()
	value, err := services.NewExperimentNativeTraining(testNativeExperiment(), rank, 20, r, e, w, progress)
	if err != nil {
		t.Fatal(err)
	}
	return value
}

func jsonDigest(value any) string {
	body, _ := json.Marshal(value)
	digest := sha256.Sum256(body)
	return hex.EncodeToString(digest[:])
}

func TestNativeEpochConsumesEachAssignmentOnceAcross20Ranks(t *testing.T) {
	data := syntheticData()
	schedule, err := services.ExperimentTrainingSchedule(testNativeExperiment(), data.Examples)
	if err != nil {
		t.Fatal(err)
	}
	byID := make(map[string]services.ExperimentTrainingRow)
	for _, row := range data.Examples {
		byID[row.ExampleID] = row
	}
	seen := make(map[string]bool)
	for rank := range 20 {
		r, e, w := newReplica(), newExchange(), &writer{}
		var progress []services.ExperimentNativeProgress
		training := trainingCase(t, rank, r, e, w, func(_ context.Context, event services.ExperimentNativeProgress) error {
			if r.installCalls != event.Step || len(e.steps) != event.Step || event.AdapterSHA256 != r.digest {
				t.Fatal("progress preceded verified installation")
			}
			if rank == 0 && event.Step >= 9 && len(w.steps) < 1 {
				t.Fatal("progress preceded checkpoint")
			}
			progress = append(progress, event)
			return nil
		})
		result, err := training.Run(context.Background(), data)
		wantLocal := 18
		if rank < 18 {
			wantLocal = 19
		}
		if err != nil || !result.Completed || result.World != 20 || result.Rank != rank || result.InstalledUpdates != 18 || result.LocalExamples != wantLocal || result.GlobalScheduledExamples != 378 {
			t.Fatalf("rank%d result=%+v err=%v", rank, result, err)
		}
		if r.closes != 1 || e.closes != 1 || r.inspectCalls != 2 || len(r.examples) != wantLocal || r.installCalls != 18 || len(progress) != 18 {
			t.Fatalf("rank%d operation counts differ", rank)
		}
		if remaining := time.Until(r.deadline); remaining <= 0 || remaining > 5400*time.Second {
			t.Fatal("fixed training deadline was not propagated")
		}
		var covered []services.ExperimentTrainingAssignment
		var totalTokens int64
		exampleIndex := 0
		for step, event := range progress {
			assignments := schedule[step].Assignments[rank].Examples
			if e.steps[step] != uint32(step+1) || len(event.Examples) != len(assignments) {
				t.Fatal("rank update geometry changed")
			}
			var summed [4]float32
			for local, assignment := range assignments {
				example := r.examples[exampleIndex]
				observation := event.Examples[local]
				if example.ExampleID != assignment.ExampleID || observation.ExampleID != assignment.ExampleID || example.LogitRows != 2 || example.CandidateVocabularyIDs != services.ExperimentNativeRecipe(testNativeExperiment()).Data.CandidateVocabularyIDs {
					t.Fatal("rank schedule or scoring contract changed")
				}
				if seen[example.ExampleID] {
					t.Fatal("example executed twice")
				}
				seen[example.ExampleID] = true
				covered = append(covered, assignment)
				prompt, err := services.RenderExperimentTrainingPrompt(byID[example.ExampleID], data.Demos)
				if err != nil || prompt != example.Prompt {
					t.Fatal("existing prompt renderer was not preserved")
				}
				loss, err := services.MultiTeacherLoss(testNativeExperiment(), observation.Logits, byID[example.ExampleID])
				if err != nil || loss != observation.Loss {
					t.Fatal("existing loss was not preserved")
				}
				for i, derivative := range loss.LogitGradient {
					want := derivative * services.ExperimentAdamWLossScale
					if r.derivatives[exampleIndex][i] != want {
						t.Fatal("loss derivative was scaled other than exactly once")
					}
					summed[i] += float32(want)
				}
				exampleIndex++
			}
			if e.gradients[step] != summed {
				t.Fatal("sequential local gradients were not summed")
			}
			totalTokens += event.InputSequenceLength
			if event.TotalSequencePositions != totalTokens || event.CoverageSHA256 != jsonDigest(covered) {
				t.Fatal("local progress evidence differs")
			}
		}
		if result.ScheduleSHA256 != jsonDigest(schedule) || result.LocalCoverageSHA256 != jsonDigest(covered) || result.SequencePositions != totalTokens || result.Initial.BaseSHA256 != result.Final.BaseSHA256 || result.Initial.AdapterSHA256 == result.Final.AdapterSHA256 {
			t.Fatal("final evidence differs")
		}
		if rank == 0 {
			if !slices.Equal(w.steps, []int{9, 18}) || len(result.Checkpoints) != 2 || result.Checkpoints[0].Candidate || !result.Checkpoints[1].Candidate {
				t.Fatal("only step18 may be the candidate")
			}
		} else if len(w.steps) != 0 || len(result.Checkpoints) != 0 {
			t.Fatal("nonzero rank wrote a checkpoint")
		}
		if _, err := training.Run(context.Background(), data); !errors.Is(err, services.ErrExperimentNativeTraining) || r.closes != 1 || len(e.steps) != 18 {
			t.Fatal("single-use epoch admitted a retry")
		}
	}
	if len(seen) != 378 {
		t.Fatalf("coverage=%d", len(seen))
	}
}

func TestNativeFailureStopsBeforeNextExampleAndClosesCollaborators(t *testing.T) {
	for _, test := range []struct {
		name                        string
		configure                   func(*replica, *exchange, *writer)
		gradients, installs, events int
	}{
		{"exchange", func(_ *replica, e *exchange, _ *writer) { e.failStep = 3 }, 4, 2, 2},
		{"installation", func(r *replica, _ *exchange, _ *writer) {
			r.installHook = func() error { return failure }
		}, 2, 1, 0},
		{"checkpoint9", func(_ *replica, _ *exchange, w *writer) { w.failStep = 9 }, 10, 9, 8},
		{"checkpoint18", func(_ *replica, _ *exchange, w *writer) { w.failStep = 18 }, 19, 18, 17},
	} {
		t.Run(test.name, func(t *testing.T) {
			r, e, w := newReplica(), newExchange(), &writer{}
			test.configure(r, e, w)
			events := 0
			training := trainingCase(t, 0, r, e, w, func(context.Context, services.ExperimentNativeProgress) error { events++; return nil })
			result, err := training.Run(context.Background(), syntheticData())
			if !errors.Is(err, failure) || result.Completed || len(r.examples) != test.gradients || r.installCalls != test.installs || events != test.events || r.closes != 1 || e.closes != 1 {
				t.Fatalf("failure did not stop: result=%+v err=%v gradients=%d installs=%d events=%d", result, err, len(r.examples), r.installCalls, events)
			}
			if _, err := training.Run(context.Background(), syntheticData()); !errors.Is(err, services.ErrExperimentNativeTraining) || len(r.examples) != test.gradients {
				t.Fatal("failed epoch was retried")
			}
		})
	}
}

func TestNativeCancellationBetweenPhases(t *testing.T) {
	for _, phase := range []string{"before", "gradient", "exchange", "install", "progress"} {
		t.Run(phase, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			r, e, w := newReplica(), newExchange(), &writer{}
			wantGradients, wantExchanges, wantInstalls := 2, 1, 1
			var progress func(context.Context, services.ExperimentNativeProgress) error
			switch phase {
			case "before":
				cancel()
				wantGradients, wantExchanges, wantInstalls = 0, 0, 0
			case "gradient":
				r.gradientHook = func(*services.ExperimentNativeGradient) { cancel() }
				wantGradients, wantExchanges, wantInstalls = 1, 0, 0
			case "exchange":
				e.hook = func([]float32) { cancel() }
				wantInstalls = 0
			case "install":
				r.installHook = func() error { cancel(); return nil }
			case "progress":
				progress = func(context.Context, services.ExperimentNativeProgress) error { cancel(); return nil }
			}
			result, err := trainingCase(t, 0, r, e, w, progress).Run(ctx, syntheticData())
			if !errors.Is(err, context.Canceled) || result.Completed || len(r.examples) != wantGradients || len(e.steps) != wantExchanges || r.installCalls != wantInstalls || r.closes != 1 || e.closes != 1 {
				t.Fatalf("phase=%s err=%v gradients=%d exchanges=%d installs=%d", phase, err, len(r.examples), len(e.steps), r.installCalls)
			}
		})
	}
}

func TestNativeProgressAndCleanupFailuresCannotReportCompletion(t *testing.T) {
	r, e := newReplica(), newExchange()
	result, err := trainingCase(t, 0, r, e, &writer{}, func(context.Context, services.ExperimentNativeProgress) error {
		return failure
	}).Run(context.Background(), syntheticData())
	if !errors.Is(err, failure) || result.Completed || result.InstalledUpdates != 1 || len(r.examples) != 2 || r.closes != 1 || e.closes != 1 {
		t.Fatal("progress failure did not stop after the installed update")
	}
	r, e = newReplica(), newExchange()
	r.closeErr = failure
	result, err = trainingCase(t, 0, r, e, &writer{}, nil).Run(context.Background(), syntheticData())
	if !errors.Is(err, failure) || result.Completed || result.InstalledUpdates != 18 || r.closes != 1 || e.closes != 1 {
		t.Fatal("cleanup failure reported completion")
	}
}

func TestNativeRejectsUnmeasuredIdentityAndInvalidBackendResults(t *testing.T) {
	for _, test := range []struct {
		name      string
		configure func(*replica, *exchange, *writer)
	}{
		{"initial_digest", func(r *replica, _ *exchange, _ *writer) { r.initial.AdapterSHA256 = strings.Repeat("e", 64) }},
		{"base_digest_absent", func(r *replica, _ *exchange, _ *writer) { r.initial.BaseSHA256 = "" }},
		{"source_manifest", func(r *replica, _ *exchange, _ *writer) { r.initial.SourceManifestSHA256 = strings.Repeat("f", 64) }},
		{"tensor_name", func(r *replica, _ *exchange, _ *writer) { r.initial.Layout[0].Name = "other" }},
		{"tensor_shape", func(r *replica, _ *exchange, _ *writer) { r.initial.Layout[0].Shape[0] = 5 }},
		{"callback_twice", func(r *replica, _ *exchange, _ *writer) { r.callbackMode = "twice" }},
		{"swallowed_callback_error", func(r *replica, _ *exchange, _ *writer) { r.callbackMode = "swallow_nonfinite_error" }},
		{"gradient_nonfinite", func(r *replica, _ *exchange, _ *writer) {
			r.gradientHook = func(g *services.ExperimentNativeGradient) { g.ValuesF32[0] = float32(math.Inf(1)) }
		}},
		{"gradient_shape", func(r *replica, _ *exchange, _ *writer) {
			r.gradientHook = func(g *services.ExperimentNativeGradient) { g.ValuesF32 = g.ValuesF32[:1] }
		}},
		{"different_logits", func(r *replica, _ *exchange, _ *writer) {
			r.gradientHook = func(g *services.ExperimentNativeGradient) { g.Logits[0]++ }
		}},
		{"missing_input_count", func(r *replica, _ *exchange, _ *writer) {
			r.gradientHook = func(g *services.ExperimentNativeGradient) { g.InputSequenceLength = 0 }
		}},
		{"exchange_nonfinite", func(_ *replica, e *exchange, _ *writer) { e.hook = func(p []float32) { p[0] = float32(math.NaN()) } }},
		{"checkpoint_identity", func(_ *replica, _ *exchange, w *writer) {
			w.hook = func(r *services.ExperimentNativeCheckpointReceipt) { r.AdapterSHA256 = strings.Repeat("d", 64) }
		}},
		{"frozen_base_changed", func(r *replica, _ *exchange, _ *writer) {
			r.finalHook = func(r *services.ExperimentNativeReplicaReceipt) { r.BaseSHA256 = strings.Repeat("d", 64) }
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			r, e, w := newReplica(), newExchange(), &writer{}
			test.configure(r, e, w)
			result, err := trainingCase(t, 0, r, e, w, nil).Run(context.Background(), syntheticData())
			if err == nil || result.Completed || r.closes != 1 || e.closes != 1 {
				t.Fatalf("invalid backend result accepted: %v", err)
			}
			if strings.HasPrefix(test.name, "callback_") || test.name == "swallowed_callback_error" {
				if len(e.steps) != 0 || r.installCalls != 0 {
					t.Fatal("callback failure reached exchange or install")
				}
			}
		})
	}
}

func TestNativeFrozenReaderAndTypedDataAdmission(t *testing.T) {
	r, e := newReplica(), newExchange()
	training := trainingCase(t, 0, r, e, &writer{}, nil)
	if _, err := training.RunFrozen(context.Background(), strings.NewReader("{}")); err == nil || len(r.examples) != 0 || r.inspectCalls != 0 || r.closes != 1 || e.closes != 1 {
		t.Fatal("unbound corpus reached native replica")
	}
	for _, mutate := range []func(*services.ExperimentTrainingData){
		func(d *services.ExperimentTrainingData) { d.SHA256 = "" },
		func(d *services.ExperimentTrainingData) { d.Examples[1] = d.Examples[0] },
		func(d *services.ExperimentTrainingData) { d.Demos[0].ExampleID = d.Examples[0].ExampleID },
		func(d *services.ExperimentTrainingData) {
			d.Examples[0].TeacherProbabilities["anchor"]["A"] = math.NaN()
		},
	} {
		data := syntheticData()
		mutate(&data)
		r, e := newReplica(), newExchange()
		if _, err := trainingCase(t, 0, r, e, &writer{}, nil).Run(context.Background(), data); err == nil || r.inspectCalls != 0 || r.closes != 1 || e.closes != 1 {
			t.Fatal("invalid typed corpus reached native replica")
		}
	}
}

func TestNativeConstructorAndZeroValueRejectInvalidTopology(t *testing.T) {
	for _, topology := range [][2]int{{0, 21}, {0, 22}, {-1, 20}, {20, 20}} {
		if _, err := services.NewExperimentNativeTraining(testNativeExperiment(), topology[0], topology[1], newReplica(), newExchange(), &writer{}, nil); !errors.Is(err, services.ErrExperimentNativeTraining) {
			t.Fatal("topology changed")
		}
	}
	var missing *replica
	if _, err := services.NewExperimentNativeTraining(testNativeExperiment(), 0, 20, missing, newExchange(), &writer{}, nil); err == nil {
		t.Fatal("typed nil replica admitted")
	}
	if _, err := services.NewExperimentNativeTraining(testNativeExperiment(), 0, 20, newReplica(), newExchange(), nil, nil); err == nil {
		t.Fatal("rank zero without writer admitted")
	}
	var zero services.ExperimentNativeTraining
	if _, err := zero.Run(context.Background(), syntheticData()); err == nil {
		t.Fatal("zero value admitted")
	}
}

func TestNativeAdamWAdapterUsesExistingRankReductionAndUnscaling(t *testing.T) {
	initial := make([]float32, services.ExperimentNativeRecipe(testNativeExperiment()).LoRA.Parameters)
	initial[0] = .25
	optimizer, err := services.NewExperimentAdamW(initial)
	if err != nil {
		t.Fatal(err)
	}
	update, err := services.ExperimentNativeAdamWUpdate(testNativeExperiment(), optimizer)
	if err != nil {
		t.Fatal(err)
	}
	ordered := make([][]float32, 20)
	for rank := range ordered {
		ordered[rank] = make([]float32, len(initial))
		ordered[rank][0] = float32(float64(rank+1) * .01 * services.ExperimentAdamWLossScale)
	}
	// Rank zero sums its regular .01 contribution and the sequential extra .21
	// contribution before exchange. The global batch remains 21 examples.
	ordered[0][0] += float32(.21 * services.ExperimentAdamWLossScale)
	parameters, err := update(ordered)
	if err != nil {
		t.Fatal(err)
	}
	state := optimizer.Snapshot()
	// The unscaled mean is .11. With zero prior moments, m1=.011 and v1=.0000121.
	if state.Step != 1 || math.Abs(float64(state.FirstMoment[0])-.011) > 2e-9 || math.Abs(float64(state.SecondMoment[0])-.0000121) > 4e-12 || math.Abs(float64(parameters[0])-(.25-2e-6)) > 1e-8 {
		t.Fatalf("AdamW scale/reduction changed: first=%g second=%g parameter=%g", state.FirstMoment[0], state.SecondMoment[0], parameters[0])
	}
	parameters[0] = 9
	if optimizer.Snapshot().Parameters[0] == 9 {
		t.Fatal("callback result aliases optimizer")
	}
	if _, err := services.ExperimentNativeAdamWUpdate(testNativeExperiment(), optimizer); err == nil {
		t.Fatal("resumed optimizer admitted")
	}
	if _, err := update(ordered[:19]); err == nil || optimizer.Snapshot().Step != 1 {
		t.Fatal("incomplete world changed optimizer")
	}
}

func TestNativeParameterDigestMatchesIndependentWireEncoding(t *testing.T) {
	layout := services.ExperimentNativeParameterLayout()
	values := make([]float32, services.ExperimentNativeRecipe(testNativeExperiment()).LoRA.Parameters)
	for i := range values {
		values[i] = float32((i%37)-18) / 1024
	}
	hash := sha256.New()
	position := 0
	for _, tensor := range layout {
		_, _ = hash.Write([]byte(tensor.Name))
		count := int(tensor.Shape[0] * tensor.Shape[1])
		if err := binary.Write(hash, binary.LittleEndian, values[position:position+count]); err != nil {
			t.Fatal(err)
		}
		position += count
	}
	got, err := services.ExperimentNativeParameterDigest(testNativeExperiment(), values)
	if err != nil || got != hex.EncodeToString(hash.Sum(nil)) || position != 557056 || len(layout) != 32 {
		t.Fatalf("digest/layout differs: %s %v", got, err)
	}
	layout[0].Shape[0] = 0
	if services.ExperimentNativeParameterLayout()[0].Shape[0] != 4 {
		t.Fatal("layout aliases shared state")
	}
	values[0] = float32(math.NaN())
	if _, err := services.ExperimentNativeParameterDigest(testNativeExperiment(), values); err == nil {
		t.Fatal("nonfinite parameters accepted")
	}
}
