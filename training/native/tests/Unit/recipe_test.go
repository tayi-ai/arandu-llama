package native_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"reflect"
	"strings"
	"testing"

	services "github.com/tayi-ai/arandu-llama/training/native"
)

func experimentRow(id, category string) services.ExperimentTrainingRow {
	return services.ExperimentTrainingRow{
		ExampleID: id, Role: "calibration-expansion", Split: "development", Prompt: "Question?",
		Choices: []string{"first", "second", "third", "fourth"}, ReferenceResult: "D", TeacherCategory: category, TeacherProbabilities: map[string]services.ExperimentCandidateProbabilities{"anchor": services.ExperimentCandidateProbabilities{"A": 1, "B": 2, "C": 3, "D": 4}, "teacher-a": services.ExperimentCandidateProbabilities{"A": 7, "B": 1, "C": 1, "D": 1}, "teacher-b": services.ExperimentCandidateProbabilities{"A": 1, "B": 1, "C": 1, "D": 7}},
	}
}

func experimentRows() []services.ExperimentTrainingRow {
	categories := []string{"anchor_preserve", "teacher_b_only_gain", "persistent_error_all", "teacher_a_teacher_b_gain", "teacher_a_only_gain"}
	rows := make([]services.ExperimentTrainingRow, 378)
	for index := range rows {
		rows[index] = experimentRow(fmt.Sprintf("development-%03d", index), categories[index%len(categories)])
	}
	return rows
}

func experimentRecipeNear(t *testing.T, name string, got, want, tolerance float64) {
	t.Helper()
	if math.IsNaN(got) || math.IsInf(got, 0) || math.Abs(got-want) > tolerance {
		t.Fatalf("%s = %.17g, want %.17g within %g", name, got, want, tolerance)
	}
}

func TestExperimentNativeRecipePreservesIntentWithoutClaimingBackendReadiness(t *testing.T) {
	r := services.ExperimentNativeRecipe(testNativeExperiment())
	if r.Version != "native-go-v2" || r.Model.Repository != "fixture/student" ||
		r.Model.Revision != "fixture-revision" || !r.Model.Frozen ||
		r.Data.SHA256 != "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" ||
		r.Data.Examples != 378 || r.Data.Demonstrations != 4 || r.Data.Epochs != 1 ||
		r.Data.CandidateVocabularyIDs != [4]int{357, 417, 351, 414} || r.Data.SelectionAllowed || r.Data.SealedAllowed {
		t.Fatalf("model or corpus contract changed: %+v", r)
	}
	if r.LoRA.Rank != 4 || r.LoRA.Alpha != 8 || r.LoRA.Targets != [2]string{"q_proj", "v_proj"} ||
		r.LoRA.Tensors != 32 || r.LoRA.Parameters != 557056 || !r.LoRA.Fresh || r.LoRA.Dropout != 0 ||
		r.LoRA.Bias != "none" || r.LoRA.ExpectedInitialDigest != "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" {
		t.Fatalf("LoRA contract changed: %+v", r.LoRA)
	}
	o := r.Optimization
	if o.Optimizer != "AdamW" || o.Seed != 83 || o.LearningRate != 2e-6 || o.Betas != [2]float64{.9, .999} ||
		o.Epsilon != 1e-8 || o.WeightDecay != 0 || o.GradientClip != 1 || o.BackwardLossScale != 1.0/1024 ||
		o.Updates != 18 || o.SaveSteps != [2]int{9, 18} || o.CandidateStep != 18 {
		t.Fatalf("optimization contract changed: %+v", o)
	}
	if r.Distribution.Processes != 20 || r.Distribution.GPUs != 40 || r.Distribution.GPUsPerProcess != 2 ||
		r.Distribution.GlobalBatch != 21 || r.Distribution.LocalBatch != 2 ||
		r.Distribution.LocalBatchMode != "sequential_extra_example_on_rank_update_minus_one" ||
		r.Distribution.LayersPerGPU != [2]int{16, 16} ||
		!reflect.DeepEqual(r.Distribution.Nodes[:], services.ExperimentNativeNodeIDs(testNativeExperiment())) {
		t.Fatalf("distribution contract changed: %+v", r.Distribution)
	}
	if r.Guards.VRAMFraction != .8 || r.Guards.HostReserveBytes != 6<<30 || !r.Guards.PreloadRAMQualificationRequired ||
		r.Guards.CanarySeconds != 1800 || r.Guards.TrainSeconds != 5400 || r.Guards.TechnicalRetries != 0 || r.Guards.AutomaticPromotion ||
		r.Backend.Status != "unqualified" || r.Backend.ReadyGPU || r.Backend.BitwiseEquivalent || r.Backend.LossPrecision != "go_float64" {
		t.Fatalf("unqualified backend or guard contract changed: %+v", r)
	}
	body, err := json.Marshal(r)
	if err != nil || !bytes.Contains(body, []byte(`"ready_gpu":false`)) || !bytes.Contains(body, []byte(`"bitwise_equivalent":false`)) {
		t.Fatalf("recipe must serialize explicit limitations: %s, %v", body, err)
	}
	r.Backend.UnqualifiedCapabilities[0] = "mutated"
	if services.ExperimentNativeRecipe(testNativeExperiment()).Backend.UnqualifiedCapabilities[0] == "mutated" {
		t.Fatal("recipe calls share mutable status state")
	}
}

func TestExperimentTeacherWeightsAndUniformLossGolden(t *testing.T) {
	tests := []struct {
		category string
		gate     string
		weight   float64
		teacher  [4]float64
	}{
		{"anchor_preserve", "decoder_anchor_1.0", .70, [4]float64{.1, .2, .3, .4}},
		{"teacher_a_teacher_b_gain", "teacher_a_0.75_teacher_b_0.25", .45, [4]float64{.55, .1, .1, .25}},
		{"teacher_a_only_gain", "teacher_a_1.0", .35, [4]float64{.7, .1, .1, .1}},
		{"teacher_b_only_gain", "teacher_b_1.0", .25, [4]float64{.1, .1, .1, .7}},
		{"persistent_error_all", "hard_label_only", 0, [4]float64{.25, .25, .25, .25}},
	}
	for _, tt := range tests {
		t.Run(tt.category, func(t *testing.T) {
			loss, err := services.MultiTeacherLoss(testNativeExperiment(), [4]float64{}, experimentRow("example", tt.category))
			if err != nil {
				t.Fatal(err)
			}
			if loss.TeacherGate != tt.gate || loss.TeacherWeight != tt.weight || loss.Prediction != "A" {
				t.Fatalf("teacher gate, weight or tie order changed: %+v", loss)
			}
			experimentRecipeNear(t, "hard", loss.Hard, math.Log(4), 1e-15)
			experimentRecipeNear(t, "distill", loss.Distill, math.Log(4), 1e-15)
			experimentRecipeNear(t, "total", loss.Total, math.Log(4), 1e-15)
			for index, teacher := range tt.teacher {
				experimentRecipeNear(t, "teacher", loss.Teacher[index], teacher, 1e-15)
				target := tt.weight * teacher
				if index == 3 {
					target += 1 - tt.weight
				}
				experimentRecipeNear(t, "target", loss.Target[index], target, 1e-15)
				experimentRecipeNear(t, "gradient", loss.LogitGradient[index], .25-target, 1e-15)
			}
		})
	}
}

func TestExperimentLossGradientMatchesIndependentCentralDifference(t *testing.T) {
	// This independent objective uses exp/sum/log probabilities without the
	// implementation's shifted log-softmax or analytic gradient formula.
	independent := func(logits [4]float64, target [4]float64) float64 {
		denominator := 0.0
		for _, value := range logits {
			denominator += math.Exp(value)
		}
		value := 0.0
		for index := range logits {
			value -= target[index] * math.Log(math.Exp(logits[index])/denominator)
		}
		return value
	}
	for _, category := range []string{"anchor_preserve", "teacher_a_teacher_b_gain", "teacher_a_only_gain", "teacher_b_only_gain", "persistent_error_all"} {
		row := experimentRow("example", category)
		logits := [4]float64{-.8, .1, 1.3, -.2}
		loss, err := services.MultiTeacherLoss(testNativeExperiment(), logits, row)
		if err != nil {
			t.Fatal(err)
		}
		experimentRecipeNear(t, category+" objective", loss.Total, independent(logits, loss.Target), 1e-14)
		const epsilon = 1e-5
		gradientSum := 0.0
		for index := range logits {
			plus, minus := logits, logits
			plus[index] += epsilon
			minus[index] -= epsilon
			derivative := (independent(plus, loss.Target) - independent(minus, loss.Target)) / (2 * epsilon)
			experimentRecipeNear(t, category+" derivative", loss.LogitGradient[index], derivative, 1e-9)
			gradientSum += loss.LogitGradient[index]
		}
		experimentRecipeNear(t, "gradient sum", gradientSum, 0, 1e-15)
	}
}

func TestExperimentLossStableShiftAndDeterministicTies(t *testing.T) {
	row := experimentRow("example", "persistent_error_all")
	a, err := services.MultiTeacherLoss(testNativeExperiment(), [4]float64{0, 1, 1, -2}, row)
	if err != nil {
		t.Fatal(err)
	}
	b, err := services.MultiTeacherLoss(testNativeExperiment(), [4]float64{10000, 10001, 10001, 9998}, row)
	if err != nil {
		t.Fatal(err)
	}
	if a.Prediction != "B" || b.Prediction != "B" {
		t.Fatal("equal best logits must choose the first letter")
	}
	experimentRecipeNear(t, "shift invariant loss", a.Total, b.Total, 1e-14)
	large, err := services.MultiTeacherLoss(testNativeExperiment(), [4]float64{math.MaxFloat64, math.MaxFloat64, math.MaxFloat64, math.MaxFloat64}, row)
	if err != nil {
		t.Fatal(err)
	}
	experimentRecipeNear(t, "large tied logits", large.Total, math.Log(4), 1e-15)
	underflow, err := services.MultiTeacherLoss(testNativeExperiment(), [4]float64{1000, -1000, 0, 0}, row)
	if err != nil || math.IsNaN(underflow.Total) || math.IsInf(underflow.Total, 0) {
		t.Fatalf("finite extreme logit loss failed: %+v, %v", underflow, err)
	}
}

func TestExperimentRejectsInvalidLabelsLogitsAndUsedTeachers(t *testing.T) {
	for _, value := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		if _, err := services.MultiTeacherLoss(testNativeExperiment(), [4]float64{0, value, 0, 0}, experimentRow("example", "anchor_preserve")); err == nil {
			t.Fatal("nonfinite logit was admitted")
		}
	}
	for _, probabilities := range []services.ExperimentCandidateProbabilities{
		nil, {"A": 1, "B": 0, "C": 0}, {"A": 1, "B": 0, "C": 0, "E": 0},
		{"A": 0, "B": 0, "C": 0, "D": 0}, {"A": -1, "B": 1, "C": 1, "D": 1},
		{"A": math.NaN(), "B": 1, "C": 1, "D": 1}, {"A": math.Inf(1), "B": 1, "C": 1, "D": 1},
		{"A": math.MaxFloat64, "B": math.MaxFloat64, "C": 1, "D": 1},
	} {
		row := experimentRow("example", "anchor_preserve")
		row.TeacherProbabilities["anchor"] = probabilities
		if _, err := services.MultiTeacherLoss(testNativeExperiment(), [4]float64{}, row); err == nil {
			t.Fatalf("invalid used teacher was admitted: %v", probabilities)
		}
	}
	row := experimentRow("example", "persistent_error_all")
	row.TeacherProbabilities["anchor"], row.TeacherProbabilities["teacher-a"], row.TeacherProbabilities["teacher-b"] = nil, nil, nil
	if _, err := services.MultiTeacherLoss(testNativeExperiment(), [4]float64{}, row); err != nil {
		t.Fatalf("hard-label-only loss must not require unused teachers: %v", err)
	}
	row.ReferenceResult = "E"
	if _, err := services.MultiTeacherLoss(testNativeExperiment(), [4]float64{}, row); err == nil {
		t.Fatal("invalid reference admitted")
	}
	row.ReferenceResult, row.TeacherCategory = "A", "unknown"
	if _, err := services.MultiTeacherLoss(testNativeExperiment(), [4]float64{}, row); err == nil {
		t.Fatal("unknown category admitted")
	}
}

func TestExperimentScheduleIsPermutationInvariantAndCoversEveryExampleOnce(t *testing.T) {
	rows := experimentRows()
	firstID := rows[0].ExampleID
	want, err := services.ExperimentTrainingSchedule(testNativeExperiment(), rows)
	if err != nil {
		t.Fatal(err)
	}
	if rows[0].ExampleID != firstID {
		t.Fatal("scheduling changed the input order")
	}
	shuffled := append([]services.ExperimentTrainingRow(nil), rows...)
	random := rand.New(rand.NewSource(83))
	random.Shuffle(len(shuffled), func(i, j int) { shuffled[i], shuffled[j] = shuffled[j], shuffled[i] })
	got, err := services.ExperimentTrainingSchedule(testNativeExperiment(), shuffled)
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("schedule depends on input order: %v", err)
	}
	order, err := services.ExperimentTrainingOrder(testNativeExperiment(), rows)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 18 {
		t.Fatalf("got %d updates", len(got))
	}
	seen := map[string]bool{}
	var rankCounts [20]int
	for step, batch := range got {
		if batch.Update != step+1 {
			t.Fatal("incorrect update index")
		}
		for rank, group := range batch.Assignments {
			wantCount := 1
			if rank == step {
				wantCount = 2
			}
			if group.Rank != rank || group.Node != services.ExperimentNativeNodeIDs(testNativeExperiment())[rank] || len(group.Examples) != wantCount {
				t.Fatalf("invalid rank assignment group: %+v", group)
			}
			for local, assignment := range group.Examples {
				wantIndex := step*21 + rank
				if local == 1 {
					wantIndex = step*21 + 20
				}
				if assignment.Rank != rank || assignment.Node != group.Node || assignment.ExampleID != order[wantIndex].ExampleID || seen[assignment.ExampleID] {
					t.Fatalf("invalid or duplicate assignment: %+v", assignment)
				}
				seen[assignment.ExampleID] = true
				rankCounts[rank]++
			}
		}
	}
	if len(seen) != 378 {
		t.Fatalf("covered %d rows", len(seen))
	}
	for rank, count := range rankCounts {
		want := 18
		if rank < 18 {
			want = 19
		}
		if count != want {
			t.Fatal("rank received incorrect example count")
		}
	}
}

func TestExperimentEpochValidationRejectsInvalidRowsWithoutDroppingThem(t *testing.T) {
	changes := map[string]func([]services.ExperimentTrainingRow) []services.ExperimentTrainingRow{
		"short epoch": func(r []services.ExperimentTrainingRow) []services.ExperimentTrainingRow { return r[:377] },
		"duplicate id": func(r []services.ExperimentTrainingRow) []services.ExperimentTrainingRow {
			r[1].ExampleID = r[0].ExampleID
			return r
		},
		"empty id": func(r []services.ExperimentTrainingRow) []services.ExperimentTrainingRow {
			r[0].ExampleID = ""
			return r
		},
		"evaluation role": func(r []services.ExperimentTrainingRow) []services.ExperimentTrainingRow {
			r[0].Role = "evaluation"
			return r
		},
		"test split": func(r []services.ExperimentTrainingRow) []services.ExperimentTrainingRow {
			r[0].Split = "test"
			return r
		},
		"three choices": func(r []services.ExperimentTrainingRow) []services.ExperimentTrainingRow {
			r[0].Choices = r[0].Choices[:3]
			return r
		},
		"empty choice": func(r []services.ExperimentTrainingRow) []services.ExperimentTrainingRow {
			r[0].Choices[2] = " "
			return r
		},
		"invalid label": func(r []services.ExperimentTrainingRow) []services.ExperimentTrainingRow {
			r[0].ReferenceResult = "AA"
			return r
		},
		"NaN teacher": func(r []services.ExperimentTrainingRow) []services.ExperimentTrainingRow {
			r[0].TeacherProbabilities["anchor"]["A"] = math.NaN()
			return r
		},
	}
	for name, change := range changes {
		t.Run(name, func(t *testing.T) {
			if _, err := services.ExperimentTrainingSchedule(testNativeExperiment(), change(experimentRows())); err == nil {
				t.Fatal("invalid epoch admitted")
			}
		})
	}
}

func TestExperimentReadsOnlyFrozenTrainingCorpusAndMatchesCalibrationOrder(t *testing.T) {
	path := os.Getenv("TAYI_EXPERIMENT_TRAINING_DATA")
	if path == "" {
		t.Skip("set TAYI_EXPERIMENT_TRAINING_DATA to opt into the frozen development-corpus check")
	}
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	data, err := services.ReadExperimentTraining(testNativeExperiment(), bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	order, err := services.ExperimentTrainingOrder(testNativeExperiment(), data.Examples)
	if err != nil || len(data.Examples) != 378 || len(data.Demos) != 4 {
		t.Fatalf("invalid frozen corpus geometry: %v", err)
	}
	want := []string{"mmlu-development-8ebfccf18df2260b", "mmlu-development-22c107120de16db7",
		"mmlu-development-4728a5e2a0675af1", "mmlu-development-d72c72a439088882",
		"mmlu-development-ef5fbb5514bde8b6", "mmlu-development-3bdd992e62d49f24",
		"mmlu-development-46b8563a559137fc", "mmlu-development-9042a871013b6566"}
	for index, id := range want {
		if order[index].ExampleID != id {
			t.Fatalf("canonical example %d = %s, want %s", index, order[index].ExampleID, id)
		}
	}
	if _, err := services.ExperimentTrainingSchedule(testNativeExperiment(), data.Examples); err != nil {
		t.Fatal(err)
	}
	for _, changed := range [][]byte{[]byte(`{"examples":[]}`), append(append([]byte(nil), body...), '\n')} {
		if _, err := services.ReadExperimentTraining(testNativeExperiment(), bytes.NewReader(changed)); err == nil {
			t.Fatal("unfrozen input was admitted")
		}
	}
}

func TestExperimentTrainingReaderRejectsUnfrozenInputWithoutExternalFiles(t *testing.T) {
	for _, body := range []string{"", `{"examples":[]}`, `{"training_allowed":true,"examples":[]}`, `{"selection_holdout_included":true}`} {
		if _, err := services.ReadExperimentTraining(testNativeExperiment(), strings.NewReader(body)); err == nil || !strings.Contains(err.Error(), "SHA256") {
			t.Fatalf("unfrozen bytes must fail their identity check: %v", err)
		}
	}
	if _, err := services.ReadExperimentTraining(testNativeExperiment(), nil); err == nil {
		t.Fatal("missing reader admitted")
	}
}

func TestExperimentRendersExactlyFourDemonstrations(t *testing.T) {
	const suffix = "\nAnswer with only the letter A, B, C, or D."
	row := experimentRow("question", "persistent_error_all")
	row.Prompt = "Final question" + suffix
	demos := make([]services.ExperimentTrainingRow, 4)
	for index := range demos {
		demos[index] = experimentRow(fmt.Sprintf("demo-%d", index), "")
		demos[index].Role = "calibration"
		demos[index].Prompt = fmt.Sprintf("Demo %d", index) + suffix
		demos[index].ReferenceResult = string(rune('A' + index))
	}
	got, err := services.RenderExperimentTrainingPrompt(row, demos)
	want := "Demo 0\nAnswer: A\n\nDemo 1\nAnswer: B\n\nDemo 2\nAnswer: C\n\nDemo 3\nAnswer: D\n\nFinal question\nAnswer:"
	if err != nil || got != want {
		t.Fatalf("prompt = %q, error %v", got, err)
	}
	row.Prompt = "Internal" + suffix + "\nextra"
	got, err = services.RenderExperimentTrainingPrompt(row, demos)
	if err != nil || !strings.HasSuffix(got, row.Prompt+"\nAnswer:") {
		t.Fatal("only an exact terminal instruction may be removed")
	}
	if _, err := services.RenderExperimentTrainingPrompt(row, demos[:3]); err == nil {
		t.Fatal("three demonstrations admitted")
	}
}
