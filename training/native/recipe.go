package native

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"math"
	"slices"
	"sort"
	"strings"
)

const (
	experimentExamples     = 378
	experimentReplicas     = ExperimentAdamWReplicas
	experimentUpdates      = ExperimentAdamWMaxUpdates
	experimentAnswerSuffix = "\nAnswer with only the letter A, B, C, or D."
)

// ExperimentRecipe fixes scientific intent. It does not authorize or start execution.
type ExperimentRecipe struct {
	Version      string                  `json:"version"`
	Model        ExperimentModelRecipe   `json:"model"`
	Data         ExperimentDataRecipe    `json:"data"`
	LoRA         ExperimentLoRARecipe    `json:"lora"`
	Optimization ExperimentOptimization  `json:"optimization"`
	Distribution ExperimentDistribution  `json:"distribution"`
	Guards       ExperimentRecipeGuards  `json:"guards"`
	Backend      ExperimentBackendStatus `json:"backend"`
}

// ExperimentModelRecipe identifies the frozen checkpoint and intended native dtypes.
type ExperimentModelRecipe struct {
	Repository         string `json:"repository"`
	Revision           string `json:"revision"`
	ManifestSHA256     string `json:"manifest_sha256"`
	Frozen             bool   `json:"frozen"`
	BasePrecision      string `json:"base_precision"`
	AttentionPrecision string `json:"attention_precision"`
}

// ExperimentDataRecipe admits one frozen development corpus and four answer labels.
type ExperimentTeacherRule struct {
	Sources map[string]float64 `json:"sources"`
	Weight  float64            `json:"weight"`
	Gate    string             `json:"gate"`
}

type ExperimentDataRecipe struct {
	TeacherRules           map[string]ExperimentTeacherRule `json:"teacher_rules"`
	ID                     string                           `json:"id"`
	SHA256                 string                           `json:"sha256"`
	Examples               int                              `json:"examples"`
	Demonstrations         int                              `json:"demonstrations"`
	Epochs                 int                              `json:"epochs"`
	Protocol               string                           `json:"protocol"`
	SelectionAllowed       bool                             `json:"selection_allowed"`
	SealedAllowed          bool                             `json:"sealed_allowed"`
	CandidateVocabularyIDs [4]int                           `json:"candidate_vocabulary_ids_abcd"`
}

// ExperimentLoRARecipe records the adapter geometry and required initial identity.
type ExperimentLoRARecipe struct {
	Rank                  int       `json:"rank"`
	Alpha                 int       `json:"alpha"`
	Targets               [2]string `json:"targets"`
	Tensors               int       `json:"tensors"`
	Parameters            int       `json:"parameters"`
	Dropout               float64   `json:"dropout"`
	Bias                  string    `json:"bias"`
	Precision             string    `json:"precision"`
	Fresh                 bool      `json:"fresh"`
	ExpectedInitialDigest string    `json:"expected_initial_digest"`
}

// ExperimentOptimization fixes the update rule shared with the native Go optimizer.
type ExperimentOptimization struct {
	Optimizer         string     `json:"optimizer"`
	Seed              int        `json:"seed"`
	LearningRate      float64    `json:"learning_rate"`
	Betas             [2]float64 `json:"betas"`
	Epsilon           float64    `json:"epsilon"`
	WeightDecay       float64    `json:"weight_decay"`
	GradientClip      float64    `json:"gradient_clip"`
	ClipEpsilon       float64    `json:"clip_epsilon"`
	BackwardLossScale float64    `json:"backward_loss_scale"`
	Updates           int        `json:"updates"`
	SaveSteps         [2]int     `json:"save_steps"`
	CandidateStep     int        `json:"candidate_step"`
	Reduction         string     `json:"gradient_reduction"`
	UpdateOrder       string     `json:"update_order"`
}

// ExperimentDistribution assigns one model replica to both GPUs on each fleet node.
type ExperimentDistribution struct {
	Nodes          [ExperimentAdamWReplicas]string `json:"nodes"`
	Processes      int                             `json:"processes"`
	GPUs           int                             `json:"gpus"`
	GPUsPerProcess int                             `json:"gpus_per_process"`
	LocalBatch     int                             `json:"local_batch"`
	LocalBatchMode string                          `json:"local_batch_mode"`
	GlobalBatch    int                             `json:"global_batch"`
	LayersPerGPU   [2]int                          `json:"layers_per_gpu"`
}

// ExperimentRecipeGuards records fixed reserves and prerequisites for future admission.
type ExperimentRecipeGuards struct {
	VRAMFraction                    float64 `json:"vram_fraction"`
	HostReserveBytes                int64   `json:"host_reserve_bytes"`
	PreloadRAMQualificationRequired bool    `json:"preload_ram_qualification_required"`
	CanarySeconds                   int     `json:"canary_seconds"`
	TrainSeconds                    int     `json:"train_seconds"`
	TechnicalRetries                int     `json:"technical_retries"`
	CalibrationRows                 int     `json:"calibration_rows"`
	ScoreTolerance                  float64 `json:"score_tolerance"`
	BaseFileCount                   int     `json:"base_file_count"`
	AutomaticPromotion              bool    `json:"automatic_promotion"`
}

// ExperimentBackendStatus separates executable CPU mathematics from GPU qualification.
type ExperimentBackendStatus struct {
	Status                  string   `json:"status"`
	LossPrecision           string   `json:"loss_precision"`
	BitwiseEquivalent       bool     `json:"bitwise_equivalent"`
	ReadyGPU                bool     `json:"ready_gpu"`
	UnqualifiedCapabilities []string `json:"unqualified_capabilities"`
}

// ExperimentNativeRecipe describes a new engine contract, not numerical equivalence
// with the preceding engine. CPU loss arithmetic below uses Go float64.
func ExperimentNativeRecipe(experiment *NativeExperiment) ExperimentRecipe {
	if experiment == nil {
		return ExperimentRecipe{}
	}
	recipe := experiment.config.Recipe
	recipe.Backend.UnqualifiedCapabilities = slices.Clone(recipe.Backend.UnqualifiedCapabilities)
	recipe.Data.TeacherRules = maps.Clone(recipe.Data.TeacherRules)
	for key, rule := range recipe.Data.TeacherRules {
		rule.Sources = maps.Clone(rule.Sources)
		recipe.Data.TeacherRules[key] = rule
	}
	return recipe
}

// ExperimentCandidateProbabilities preserves the source A/B/C/D keys and allows
// validation to distinguish a missing entry from a legitimate zero.
type ExperimentCandidateProbabilities map[string]float64

// ExperimentTrainingRow contains only the fields needed to validate and train a question.
type ExperimentTrainingRow struct {
	ExampleID            string                                      `json:"example_id"`
	Role                 string                                      `json:"a2_role"`
	Split                string                                      `json:"split"`
	Prompt               string                                      `json:"prompt"`
	Choices              []string                                    `json:"choices"`
	ReferenceResult      string                                      `json:"reference_result"`
	TeacherCategory      string                                      `json:"teacher_category"`
	TeacherProbabilities map[string]ExperimentCandidateProbabilities `json:"teacher_probabilities,omitempty"`
}

// ExperimentTrainingData is the validated, hash-bound development input.
type ExperimentTrainingData struct {
	SchemaVersion int                     `json:"schema_version"`
	DatasetID     string                  `json:"dataset_id"`
	Protocol      string                  `json:"protocol"`
	SHA256        string                  `json:"sha256"`
	Demos         []ExperimentTrainingRow `json:"demos"`
	Examples      []ExperimentTrainingRow `json:"examples"`
}

// ReadExperimentTraining admits only the frozen training bytes. It opens no paths
// and reads no selection or final evaluation data.
func ReadExperimentTraining(experiment *NativeExperiment, reader io.Reader) (ExperimentTrainingData, error) {
	if experiment == nil {
		return ExperimentTrainingData{}, errors.New("native experiment: installation is not configured")
	}

	const limit = 8 << 20
	if reader == nil {
		return ExperimentTrainingData{}, errors.New("experiment: missing training reader")
	}
	body, err := io.ReadAll(io.LimitReader(reader, limit+1))
	if err != nil || len(body) > limit {
		return ExperimentTrainingData{}, errors.New("experiment: training input is unreadable or oversized")
	}
	digest := sha256.Sum256(body)
	if hex.EncodeToString(digest[:]) !=
		experiment.config.Recipe.Data.SHA256 {
		return ExperimentTrainingData{}, errors.New("experiment: training SHA256 differs from the frozen corpus")
	}
	var wire struct {
		ExperimentTrainingData
		TrainingAllowed          *bool `json:"training_allowed"`
		ProtectionIncluded       *bool `json:"protection_included"`
		SelectionHoldoutIncluded *bool `json:"selection_holdout_included"`
		SealedFinalIncluded      *bool `json:"sealed_final_included"`
		HeldoutIncluded          *bool `json:"heldout_included"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		return ExperimentTrainingData{}, fmt.Errorf("experiment: decoding training input: %w", err)
	}
	if wire.TrainingAllowed == nil || !*wire.TrainingAllowed {
		return ExperimentTrainingData{}, errors.New("experiment: corpus is not admitted for training")
	}
	for _, flag := range []*bool{wire.ProtectionIncluded, wire.SelectionHoldoutIncluded, wire.SealedFinalIncluded, wire.HeldoutIncluded} {
		if flag == nil || *flag {
			return ExperimentTrainingData{}, errors.New("experiment: forbidden or absent evaluation-data flag")
		}
	}
	data := wire.ExperimentTrainingData
	recipe := ExperimentNativeRecipe(experiment)
	if data.SchemaVersion != 1 || data.DatasetID != recipe.Data.ID || data.Protocol != recipe.Data.Protocol || len(data.Demos) != 4 {
		return ExperimentTrainingData{}, errors.New("experiment: unexpected training protocol or demonstration geometry")
	}
	if err := ValidateExperimentTrainingRows(experiment, data.Examples); err != nil {
		return ExperimentTrainingData{}, err
	}
	seen := make(map[string]bool, experimentExamples+4)
	for _, row := range data.Examples {
		seen[row.ExampleID] = true
	}
	for _, row := range data.Demos {
		if err := validateExperimentQuestion(row, "calibration"); err != nil {
			return ExperimentTrainingData{}, err
		}
		if seen[row.ExampleID] {
			return ExperimentTrainingData{}, errors.New("experiment: duplicate demonstration or training identity")
		}
		seen[row.ExampleID] = true
	}
	data.SHA256 = experiment.config.Recipe.Data.SHA256

	return data, nil
}

func validateExperimentQuestion(row ExperimentTrainingRow, role string) error {
	if strings.TrimSpace(row.ExampleID) == "" || row.ExampleID != strings.TrimSpace(row.ExampleID) ||
		row.Role != role || row.Split != "development" || strings.TrimSpace(row.Prompt) == "" || len(row.Choices) != 4 {
		return fmt.Errorf("experiment: invalid development question %q", row.ExampleID)
	}
	if _, err := experimentLabelIndex(row.ReferenceResult); err != nil {
		return err
	}
	for _, choice := range row.Choices {
		if strings.TrimSpace(choice) == "" {
			return fmt.Errorf("experiment: empty choice in %q", row.ExampleID)
		}
	}
	return nil
}

// ValidateExperimentTrainingRows validates the complete epoch without silently
// dropping duplicates, changing categories or replacing teacher distributions.
func ValidateExperimentTrainingRows(experiment *NativeExperiment, rows []ExperimentTrainingRow) error {
	if experiment == nil {
		return errors.New("native experiment: installation is not configured")
	}

	if len(rows) != experimentExamples {
		return fmt.Errorf("experiment: expected %d training rows, got %d", experimentExamples, len(rows))
	}
	seen := make(map[string]bool, len(rows))
	for _, row := range rows {
		if err := validateExperimentQuestion(row, "calibration-expansion"); err != nil {
			return err
		}
		if seen[row.ExampleID] {
			return fmt.Errorf("experiment: duplicate training identity %q", row.ExampleID)
		}
		seen[row.ExampleID] = true
		if _, _, _, err := experimentTeacher(experiment, row); err != nil {
			return fmt.Errorf("experiment: teacher for %q: %w", row.ExampleID, err)
		}
	}
	return nil
}

// ExperimentTrainingOrder reproduces category sorting and SHA256(example_id)
// round-robin ordering. It leaves the caller's slice unchanged.
func ExperimentTrainingOrder(experiment *NativeExperiment, rows []ExperimentTrainingRow) ([]ExperimentTrainingRow, error) {
	if experiment == nil {
		return nil, errors.New("native experiment: installation is not configured")
	}

	if err := ValidateExperimentTrainingRows(experiment, rows); err != nil {
		return nil, err
	}
	groups := make(map[string][]ExperimentTrainingRow)
	for _, row := range rows {
		groups[row.TeacherCategory] = append(groups[row.TeacherCategory], row)
	}
	categories := make([]string, 0, len(groups))
	for category, group := range groups {
		categories = append(categories, category)
		sort.Slice(group, func(i, j int) bool {
			a, b := sha256.Sum256([]byte(group[i].ExampleID)), sha256.Sum256([]byte(group[j].ExampleID))
			if comparison := bytes.Compare(a[:], b[:]); comparison != 0 {
				return comparison < 0
			}
			return group[i].ExampleID < group[j].ExampleID
		})
	}
	sort.Strings(categories)
	ordered := make([]ExperimentTrainingRow, 0, len(rows))
	for index := 0; len(ordered) < len(rows); index++ {
		for _, category := range categories {
			if index < len(groups[category]) {
				ordered = append(ordered, groups[category][index])
			}
		}
	}
	return ordered, nil
}

// ExperimentTrainingAssignment identifies one example consumed by one replica.
type ExperimentTrainingAssignment struct {
	Rank      int    `json:"rank"`
	Node      string `json:"node"`
	ExampleID string `json:"example_id"`
}

// ExperimentTrainingRankAssignments contains one or two sequential examples. Both
// gradients use the same installed parameters and are summed before exchange.
type ExperimentTrainingRankAssignments struct {
	Rank     int                            `json:"rank"`
	Node     string                         `json:"node"`
	Examples []ExperimentTrainingAssignment `json:"examples"`
}

// ExperimentTrainingBatch contains 21 examples distributed across all 20 replicas.
type ExperimentTrainingBatch struct {
	Update      int                                                        `json:"update"`
	Assignments [ExperimentAdamWReplicas]ExperimentTrainingRankAssignments `json:"assignments"`
}

// ExperimentTrainingSchedule preserves consecutive groups of 21 canonical examples.
// Every rank gets one example; rank update-1 receives the group's final example
// as a second sequential contribution. No padding, duplication or drop occurs.
func ExperimentTrainingSchedule(experiment *NativeExperiment, rows []ExperimentTrainingRow) ([]ExperimentTrainingBatch, error) {
	if experiment == nil {
		return nil, errors.New("native experiment: installation is not configured")
	}

	ordered, err := ExperimentTrainingOrder(experiment, rows)
	if err != nil {
		return nil, err
	}
	nodes := ExperimentNativeRecipe(experiment).Distribution.Nodes
	batches := make([]ExperimentTrainingBatch, experimentUpdates)
	for step := range batches {
		batches[step].Update = step + 1
		for rank := range experimentReplicas {
			assignment := ExperimentTrainingAssignment{Rank: rank, Node: nodes[rank], ExampleID: ordered[step*ExperimentAdamWGlobalBatch+rank].ExampleID}
			batches[step].Assignments[rank] = ExperimentTrainingRankAssignments{Rank: rank, Node: nodes[rank], Examples: []ExperimentTrainingAssignment{assignment}}
		}
		extra := ExperimentTrainingAssignment{Rank: step, Node: nodes[step], ExampleID: ordered[step*ExperimentAdamWGlobalBatch+experimentReplicas].ExampleID}
		batches[step].Assignments[step].Examples = append(batches[step].Assignments[step].Examples, extra)
	}
	return batches, nil
}

func experimentLabelIndex(label string) (int, error) {
	if len(label) != 1 || label[0] < 'A' || label[0] > 'D' {
		return 0, errors.New("experiment: reference must be A, B, C or D")
	}
	return int(label[0] - 'A'), nil
}

func experimentNormalize(values ExperimentCandidateProbabilities) ([4]float64, error) {
	var out [4]float64
	if len(values) != 4 {
		return out, errors.New("teacher distribution must contain exactly A, B, C and D")
	}
	total := 0.0
	for index := range out {
		value, exists := values[string(rune('A'+index))]
		if !exists || math.IsNaN(value) || math.IsInf(value, 0) || value < 0 {
			return out, errors.New("teacher distribution contains a missing, negative or nonfinite probability")
		}
		out[index], total = value, total+value
	}
	if total <= 0 || math.IsInf(total, 0) {
		return out, errors.New("teacher distribution has zero or nonfinite mass")
	}
	for index := range out {
		out[index] /= total
	}
	return out, nil
}

func experimentTeacher(experiment *NativeExperiment, row ExperimentTrainingRow) ([4]float64, string, float64, error) {
	if experiment == nil {
		return [4]float64{}, "", 0, errors.New("native experiment: installation is not configured")
	}

	var distribution [4]float64
	rule, ok := experiment.config.Recipe.Data.TeacherRules[row.TeacherCategory]
	if !ok {
		return distribution, "", 0, errors.New("teacher category is not admitted")
	}
	if rule.Weight == 0 {
		return [4]float64{.25, .25, .25, .25}, rule.Gate, 0, nil
	}
	ids := make([]string, 0, len(rule.Sources))
	for id := range rule.Sources {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		source, err := experimentNormalize(row.TeacherProbabilities[id])
		if err != nil {
			return distribution, "", 0, err
		}
		for i := range distribution {
			distribution[i] += rule.Sources[id] * source[i]
		}
	}
	return distribution, rule.Gate, rule.Weight, nil
}

// ExperimentLoss records four-choice cross entropy and its unscaled logit derivative.
type ExperimentLoss struct {
	Hard          float64    `json:"hard"`
	Distill       float64    `json:"distill"`
	Total         float64    `json:"total"`
	TeacherWeight float64    `json:"teacher_weight"`
	TeacherGate   string     `json:"teacher_gate"`
	Teacher       [4]float64 `json:"teacher_abcd"`
	Target        [4]float64 `json:"target_abcd"`
	Probabilities [4]float64 `json:"probabilities_abcd"`
	LogitGradient [4]float64 `json:"logit_gradient_abcd"`
	Prediction    string     `json:"prediction"`
}

// MultiTeacherLoss computes four-choice mixed cross entropy and its derivative
// with respect to those four logits in float64. The derivative is unscaled and
// not divided by the replica count; the native backend must apply both later.
func MultiTeacherLoss(experiment *NativeExperiment, logits [4]float64, row ExperimentTrainingRow) (ExperimentLoss, error) {
	if experiment == nil {
		return ExperimentLoss{}, errors.New("native experiment: installation is not configured")
	}

	var loss ExperimentLoss
	label, err := experimentLabelIndex(row.ReferenceResult)
	if err != nil {
		return loss, err
	}
	loss.Teacher, loss.TeacherGate, loss.TeacherWeight, err = experimentTeacher(experiment, row)
	if err != nil {
		return ExperimentLoss{}, err
	}
	best := 0
	for index, value := range logits {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return ExperimentLoss{}, errors.New("experiment: nonfinite candidate logit")
		}
		if value > logits[best] {
			best = index
		}
	}
	var shifted [4]float64
	sum := 0.0
	for index := range logits {
		shifted[index] = logits[index] - logits[best]
		if math.IsInf(shifted[index], 0) {
			return ExperimentLoss{}, errors.New("experiment: candidate logit range overflows float64")
		}
		loss.Probabilities[index] = math.Exp(shifted[index])
		sum += loss.Probabilities[index]
	}
	logSum := math.Log(sum)
	for index := range logits {
		logProbability := shifted[index] - logSum
		loss.Probabilities[index] /= sum
		loss.Target[index] = loss.TeacherWeight * loss.Teacher[index]
		if index == label {
			loss.Hard = -logProbability
			loss.Target[index] += 1 - loss.TeacherWeight
		}
		loss.Distill -= loss.Teacher[index] * logProbability
		loss.LogitGradient[index] = loss.Probabilities[index] - loss.Target[index]
	}
	loss.Total = (1-loss.TeacherWeight)*loss.Hard + loss.TeacherWeight*loss.Distill
	loss.Prediction = string(rune('A' + best))
	if math.IsNaN(loss.Total) || math.IsInf(loss.Total, 0) || math.IsInf(loss.Distill, 0) || math.IsInf(loss.Hard, 0) {
		return ExperimentLoss{}, errors.New("experiment: nonfinite cross entropy")
	}
	return loss, nil
}

// RenderExperimentTrainingPrompt reproduces the four-demo raw answer-space prompt.
// Tokenizer identity and the candidate IDs still require native qualification.
func RenderExperimentTrainingPrompt(row ExperimentTrainingRow, demos []ExperimentTrainingRow) (string, error) {
	if len(demos) != 4 {
		return "", errors.New("experiment: exactly four demonstrations are required")
	}
	if err := validateExperimentQuestion(row, "calibration-expansion"); err != nil {
		return "", err
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
