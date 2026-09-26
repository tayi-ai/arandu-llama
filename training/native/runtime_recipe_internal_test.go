package native

import (
	"context"
	"reflect"
	"testing"

	decoder "github.com/tayi-ai/arandu-llama/training/decoder"
)

// Only Gradient may be used by the canary calculation. Embedding the remaining
// interface makes an unexpected native operation fail without allocating one.
type recipeCandidateReplica struct {
	ExperimentNativeReplica
	gradient func(ExperimentNativeExample, ExperimentNativeLossGradient) (ExperimentNativeGradient, error)
}

func (r recipeCandidateReplica) Gradient(_ context.Context, example ExperimentNativeExample, derivative ExperimentNativeLossGradient) (ExperimentNativeGradient, error) {
	return r.gradient(example, derivative)
}

func TestRuntimeCandidatesFollowAdmittedRecipeWithoutChangingCanaryMath(t *testing.T) {
	row := ExperimentTrainingRow{ExampleID: "synthetic-question", ReferenceResult: "C", TeacherCategory: "persistent_error_all"}
	const prompt = "Synthetic question\nAnswer:"
	logits := [4]float64{.1, -.2, .4, .3}
	for _, candidates := range [][4]int{{357, 417, 351, 414}, {4, 9, 1, 7}, {0, 248319, 2, 6}} {
		config := testNativeConfiguration()
		config.Recipe.Data.CandidateVocabularyIDs = candidates
		experiment, err := NewNativeExperiment(config)
		if err != nil {
			t.Fatal(err)
		}
		expected := ExperimentNativeExample{ExampleID: row.ExampleID, Prompt: prompt, CandidateVocabularyIDs: candidates, LogitRows: 2}
		// Calibration and training use this same pure example constructor.
		if got := nativeExample(experiment, row, prompt); got != expected {
			t.Fatalf("example candidates: got %+v, want %+v", got, expected)
		}
		loss, err := MultiTeacherLoss(experiment, logits, row)
		if err != nil {
			t.Fatal(err)
		}
		calls := 0
		replica := recipeCandidateReplica{gradient: func(example ExperimentNativeExample, derivative ExperimentNativeLossGradient) (ExperimentNativeGradient, error) {
			calls++
			if example != expected {
				t.Fatalf("canary used tokens outside admitted recipe: %+v", example)
			}
			scaled, err := derivative(logits)
			if err != nil {
				return ExperimentNativeGradient{}, err
			}
			values := make([]float32, 4)
			for i, value := range scaled {
				if value != loss.LogitGradient[i]*ExperimentAdamWLossScale {
					t.Fatal("candidate mapping changed backward scaling")
				}
				values[i] = float32(value)
			}
			return ExperimentNativeGradient{Logits: logits, ValuesF32: values, InputSequenceLength: 5}, nil
		}}
		gradient, losses, err := experimentNativeCanaryGradient(experiment, context.Background(), replica, row, prompt, 2)
		if err != nil || calls != 2 || len(losses) != 2 || gradient.InputSequenceLength != 5 {
			t.Fatalf("canary result: calls=%d losses=%d error=%v", calls, len(losses), err)
		}
		if losses[0] != loss || losses[1] != loss {
			t.Fatal("candidate mapping changed the measured loss")
		}
		for i, value := range gradient.ValuesF32 {
			single := float32(loss.LogitGradient[i] * ExperimentAdamWLossScale)
			if value != single+single {
				t.Fatal("candidate mapping changed sequential FP32 accumulation")
			}
		}
	}
}

func TestRuntimeAssemblyRefusesInsufficientConfiguredBudgets(t *testing.T) {
	plan, err := ExperimentNativeMemory(1129)
	if err != nil {
		t.Fatal(err)
	}
	for name, lower := range map[string]func(*decoder.AssemblyLimits){
		"copies":   func(l *decoder.AssemblyLimits) { l.TensorCopyBytes-- },
		"input":    func(l *decoder.AssemblyLimits) { l.MaxInputElements-- },
		"score":    func(l *decoder.AssemblyLimits) { l.MaxScoreElements-- },
		"working":  func(l *decoder.AssemblyLimits) { l.MaxWorkingElements-- },
		"tokens":   func(l *decoder.AssemblyLimits) { l.Sequence.MaxTokens-- },
		"sequence": func(l *decoder.AssemblyLimits) { l.Sequence.MaxOwnedElements-- },
	} {
		t.Run(name, func(t *testing.T) {
			config := testNativeConfiguration()
			lower(&config.Assembly)
			experiment, err := NewNativeExperiment(config)
			if err != nil {
				t.Fatalf("positive budget rejected before actual sequence is known: %v", err)
			}
			limits, err := plan.assemblyLimits(experiment)
			if err == nil || !reflect.DeepEqual(limits, decoder.AssemblyLimits{}) {
				t.Fatal("runtime enlarged an insufficient installation budget")
			}
		})
	}
}

func TestRuntimeAssemblyPreservesExplicitLimitsAndHistoricalPlan(t *testing.T) {
	plan, err := ExperimentNativeMemory(1129)
	if err != nil {
		t.Fatal(err)
	}
	config := testNativeConfiguration()
	experiment, err := NewNativeExperiment(config)
	if err != nil {
		t.Fatal(err)
	}
	limits, err := plan.assemblyLimits(experiment)
	if err != nil || !reflect.DeepEqual(limits, config.Assembly) {
		t.Fatalf("explicit historical limits changed: %v", err)
	}
	config.Assembly.HeaderLimits.MaxHeaderBytes /= 2
	config.Assembly.HeaderLimits.MaxTensors /= 2
	config.Assembly.HeaderLimits.MaxDimensions /= 2
	config.Assembly.HeaderLimits.MaxMetadataEntries /= 2
	config.Assembly.HeaderLimits.MaxChunkBytes /= 2
	config.Assembly.HashChunkBytes /= 2
	experiment, err = NewNativeExperiment(config)
	if err != nil {
		t.Fatal(err)
	}
	smaller, err := ExperimentNativeMemory(8)
	if err != nil {
		t.Fatal(err)
	}
	limits, err = smaller.assemblyLimits(experiment)
	if err != nil {
		t.Fatal(err)
	}
	if limits.HeaderLimits != config.Assembly.HeaderLimits || limits.HashChunkBytes != config.Assembly.HashChunkBytes {
		t.Fatal("runtime discarded explicit parsing or hash limits")
	}
	if limits.Sequence.ChunkTokens != 8 || limits.Sequence.MaxTokens != 8 || limits.Sequence.MaxOwnedElements != smaller.SequenceElements || limits.MaxWorkingElements != smaller.WorkingElements {
		t.Fatal("runtime changed numerical chunking or ignored measured sequence bounds")
	}
	limits.PersistentBytes[0] = 0
	limits.DeviceByLayer[0] = 1
	if experiment.config.Assembly.PersistentBytes[0] == 0 || experiment.config.Assembly.DeviceByLayer[0] != 0 {
		t.Fatal("runtime assembly aliases the admitted installation")
	}
}

func TestNativeInstallationRejectsOmittedBudgetsAndUnsupportedChunking(t *testing.T) {
	for name, change := range map[string]func(*decoder.AssemblyLimits){
		"copies":          func(l *decoder.AssemblyLimits) { l.TensorCopyBytes = 0 },
		"input":           func(l *decoder.AssemblyLimits) { l.MaxInputElements = 0 },
		"score":           func(l *decoder.AssemblyLimits) { l.MaxScoreElements = 0 },
		"working":         func(l *decoder.AssemblyLimits) { l.MaxWorkingElements = 0 },
		"tokens":          func(l *decoder.AssemblyLimits) { l.Sequence.MaxTokens = 0 },
		"sequence":        func(l *decoder.AssemblyLimits) { l.Sequence.MaxOwnedElements = 0 },
		"chunking":        func(l *decoder.AssemblyLimits) { l.Sequence.ChunkTokens = 4 },
		"hash":            func(l *decoder.AssemblyLimits) { l.HashChunkBytes = 0 },
		"header":          func(l *decoder.AssemblyLimits) { l.HeaderLimits.MaxHeaderBytes = 0 },
		"expanded-header": func(l *decoder.AssemblyLimits) { l.HeaderLimits.MaxHeaderBytes++ },
	} {
		t.Run(name, func(t *testing.T) {
			config := testNativeConfiguration()
			change(&config.Assembly)
			if _, err := NewNativeExperiment(config); err == nil {
				t.Fatal("implicit budget or unsupported execution mode accepted")
			}
		})
	}
}
