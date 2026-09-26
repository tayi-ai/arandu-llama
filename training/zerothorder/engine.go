package zerothorder

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"strings"
	"time"

	"github.com/tayi-ai/arandu-llama/training/adapter"
)

// Zeroth-order optimisation: training without a backward pass.
//
// The gradient is not computed, it is estimated. A direction is drawn from a
// seed, the parameters are moved along it in both directions, and the difference
// in loss between the two is the projection of the gradient onto that direction.
// The direction is never stored: the seed reconstructs it.
//
// A step needs two forward passes without an autograd graph, optimizer moments
// or retained activations. Choosing this method belongs to the recipe owner.
//
// What crosses the network per step is two float64 and one uint64. The
// gradient never travels, which is why the bandwidth of the private network
// stopped being the question this technique has to answer.
//
// Nothing in this file computes a forward pass. It states the arithmetic and
// leaves the model to an engine that already holds the weights on the cards.

// Reading is one scored evaluation: the loss the engine reported for an example
// under a particular perturbation.
type Reading struct {
	Loss    float64       `json:"loss"`
	Tokens  int           `json:"tokens"`
	Elapsed time.Duration `json:"elapsed"`
}

// Estimate is what one zeroth-order step learned.
//
// Gradient is the projection of the gradient onto the direction, not the
// gradient. Reconstructing the update needs the seed as well, which is why both
// travel together and why neither is useful alone.
type Estimate struct {
	Direction adapter.Direction `json:"direction"`
	Plus      Reading           `json:"plus"`
	Minus     Reading           `json:"minus"`
	Gradient  float64           `json:"gradient"`
}

// Step estimates the gradient along one direction with two forward passes.
//
// The order is fixed: plus, then minus, then back to unperturbed. Leaving the
// engine perturbed after a step is how a later measurement silently reads a
// model nobody meant to evaluate.
func Step(ctx context.Context, engine Engine, example Example, d adapter.Direction) (Estimate, error) {
	if d.Scale <= 0 {
		return Estimate{}, errors.New("cluster: the perturbation scale has to be positive; a zero step measures the same point twice and estimates nothing")
	}
	plus, err := probeAt(ctx, engine, example, d, +1)
	if err != nil {
		return Estimate{}, restoreAfter(ctx, engine, d, err)
	}
	minus, err := probeAt(ctx, engine, example, d, -1)
	if err != nil {
		return Estimate{}, restoreAfter(ctx, engine, d, err)
	}
	if err := engine.Perturb(ctx, d, 0); err != nil {
		return Estimate{}, fmt.Errorf("cluster: the engine stayed perturbed after the step: %w", err)
	}
	// The central difference: (L(+e) - L(-e)) / 2e. Central rather than forward
	// because the forward difference carries a first-order bias that does not
	// vanish, and two passes are already being paid for.
	estimate := Estimate{Direction: d, Plus: plus, Minus: minus}
	estimate.Gradient = (plus.Loss - minus.Loss) / (2 * d.Scale)
	if math.IsNaN(estimate.Gradient) || math.IsInf(estimate.Gradient, 0) {
		return estimate, fmt.Errorf("cluster: the estimate is not finite: L(+)=%v L(-)=%v scale=%v", plus.Loss, minus.Loss, d.Scale)
	}
	return estimate, nil
}

func probeAt(ctx context.Context, engine Engine, example Example, d adapter.Direction, sign float64) (Reading, error) {
	if err := engine.Perturb(ctx, d, sign); err != nil {
		return Reading{}, err
	}
	return engine.Score(ctx, example)
}

// restoreAfter puts the engine back on the snapshot when a probe failed, and
// reports both errors if the restore fails too.
//
// A step that failed half way is the same hazard as one that finished without
// restoring: whatever measures next reads the candidate of a probe nobody
// completed. The restore is attempted even when the failure was a cancelled
// context, because an interrupted run still has to leave the policy applied.
func restoreAfter(ctx context.Context, engine Engine, d adapter.Direction, err error) error {
	if restore := engine.Perturb(ctx, d, 0); restore != nil {
		return errors.Join(err, fmt.Errorf("cluster: the engine stayed perturbed after the failed probe: %w", restore))
	}
	return err
}

// Engine is what holds the weights and answers with a loss.
//
// An interface because this package does no arithmetic on a tensor and should
// not learn how. The engine that trains is the in-process binding behind the
// llama build tag; the HTTP engine scores but cannot probe, and says so.
type Engine interface {
	// Perturb applies the candidate S + sign*d.Scale*Z, built by the same
	// constructor the accepted step uses. A sign of zero re-applies the
	// snapshot the probes started from.
	Perturb(ctx context.Context, d adapter.Direction, sign float64) error
	// Score returns the loss the engine measures for one example.
	Score(ctx context.Context, e Example) (Reading, error)
}

// Example is one scored item: the text, and how many tokens of it are the
// answer being scored rather than the prompt leading to it.
type Example struct {
	ID     string `json:"id"`
	Prompt string `json:"prompt"`
	// Completion is the part the loss is measured over. A loss that included the
	// prompt would reward a policy for predicting text it was given.
	Completion string `json:"completion"`
	// Tokens, when present, is the sequence to score instead of the text, and
	// PromptTokens is how many of them are the prompt. The gate this technique
	// has to pass is reproducing a loss the trainer measured, which means scoring
	// the token sequence it scored: a GGUF carries its own vocabulary and a
	// requantisation is not obliged to preserve the merges, so going through the
	// engine's tokeniser would introduce a difference nobody could attribute.
	Tokens       []int32 `json:"-"`
	PromptTokens int     `json:"-"`
}

// LlamaEngine reaches a resident llama.cpp server.
//
// The server is the engine and not the CLI because the weights have to stay on
// the cards between probes. Loading 15.32 GiB per probe would cost more than the
// forward pass it exists to make cheap.
type LlamaEngine struct {
	BaseURL string
	Client  *http.Client
	apiKey  string
	// Bank names the adapters the server has loaded, in order. The first is the
	// adapter being trained; the rest are slots the server can rescale through
	// POST /lora-adapters.
	//
	// Rescaling a second adapter was how this engine probed, and it is the
	// defect T02 measured: the server sums adapters in product space, so a
	// direction loaded as an adapter moves the output along ZB*ZA while the
	// accepted step moves it along B*ZA + ZB*A. The slots are kept so a server
	// can be reset to the policy alone; they are never scaled away from zero.
	Bank []int
}

// LogValue keeps the resident server key out of structured logs.
func (e *LlamaEngine) LogValue() slog.Value {
	if e == nil {
		return slog.AnyValue(nil)
	}
	return slog.GroupValue(
		slog.String("base_url", e.BaseURL), slog.Int("adapter_slots", len(e.Bank)),
		slog.String("api_key", redactedSecret(e.apiKey)),
	)
}

// MarshalJSON keeps the resident server key out of debug dumps.
func (e *LlamaEngine) MarshalJSON() ([]byte, error) {
	if e == nil {
		return []byte("null"), nil
	}
	return json.Marshal(map[string]any{
		"base_url": e.BaseURL, "adapter_slots": len(e.Bank), "api_key": redactedSecret(e.apiKey),
	})
}

func redactedSecret(value string) string {
	if value == "" {
		return ""
	}
	return "[redacted]"
}

// ErrProductSpaceProbe is returned when the server engine is asked to probe.
var ErrProductSpaceProbe = errors.New("cluster: the server can only scale a second adapter, which moves the output by c*ZB*ZA while the accepted step moves it by c*(B*ZA + ZB*A); on the scalar oracle those derivatives are -35 and 1, so the server engine scores but does not probe")

// NewLlamaEngine returns an engine with the timeout a scored example needs.
func NewLlamaEngine(baseURL string) *LlamaEngine {
	return newLlamaEngine(baseURL, "")
}

// NewAuthenticatedLlamaEngine returns an engine for a private resident server.
// The key is held separately from the URL and is sent only in the authorization
// header, so neither process arguments nor endpoint diagnostics disclose it.
func NewAuthenticatedLlamaEngine(baseURL, apiKey string) *LlamaEngine {
	return newLlamaEngine(baseURL, apiKey)
}

func newLlamaEngine(baseURL, apiKey string) *LlamaEngine {
	return &LlamaEngine{
		BaseURL: baseURL, apiKey: apiKey,
		// Generous: a 4176-token example was measured at 4.76 s of prompt
		// processing, and a loaded card under contention is slower than a bench.
		Client: &http.Client{Timeout: 5 * time.Minute},
	}
}

type loraScale struct {
	ID    int     `json:"id"`
	Scale float64 `json:"scale"`
}

// Perturb refuses to probe and, at sign zero, resets the server to the policy
// alone: slot zero at scale one, every other slot at zero.
//
// The only mechanism the server offers is the one the oracle refutes, so a
// non-zero sign is an error rather than a measurement. The reset stays because
// a server left with a direction slot scaled is a server every later Score
// reads through it.
func (e *LlamaEngine) Perturb(ctx context.Context, d adapter.Direction, sign float64) error {
	if len(e.Bank) == 0 {
		return errors.New("cluster: the engine has no adapter slots; load the policy before scoring")
	}
	if sign != 0 {
		return ErrProductSpaceProbe
	}
	scales := make([]loraScale, 0, len(e.Bank))
	for i, id := range e.Bank {
		scale := 0.0
		if i == 0 {
			scale = 1.0
		}
		scales = append(scales, loraScale{ID: id, Scale: scale})
	}
	body, err := json.Marshal(scales)
	if err != nil {
		return err
	}
	_, err = e.post(ctx, "/lora-adapters", body)
	return err
}

// scoreRequest asks the engine to read an example back to us rather than write
// one.
//
// Echo is the whole point. Without it the engine reports the likelihood of text
// it chose itself, which is a different quantity: the training signal here is how
// well the model predicts a completion somebody else wrote, token by token, and
// that is what every backpropagated run in this project measured. MaxTokens is
// zero because nothing is to be generated.
type scoreRequest struct {
	Prompt      string      `json:"prompt"`
	MaxTokens   int         `json:"max_tokens"`
	Echo        bool        `json:"echo"`
	LogProbs    int         `json:"logprobs"`
	Temperature float64     `json:"temperature"`
	LoRA        []loraScale `json:"lora,omitempty"`
}

// scoreAnswer reads the shape this build answers in.
//
// The logprobs arrive under `content`, not under the legacy `token_logprobs`.
// Reading the wrong one returns an empty list rather than an error, and an empty
// list is indistinguishable from "this engine cannot score" -- which is exactly
// the wrong conclusion it produced once already.
type scoreAnswer struct {
	Choices []struct {
		LogProbs struct {
			Content []struct {
				LogProb *float64 `json:"logprob"`
			} `json:"content"`
		} `json:"logprobs"`
	} `json:"choices"`
}

// Completion is one deterministic generation from the resident MX server.
type Completion struct {
	Text             string `json:"text"`
	PromptTokens     int    `json:"prompt_tokens"`
	CompletionTokens int    `json:"completion_tokens"`
}

type completionRequest struct {
	Prompt      string      `json:"prompt"`
	MaxTokens   int         `json:"max_tokens"`
	Seed        int         `json:"seed"`
	Temperature float64     `json:"temperature"`
	LogProbs    int         `json:"logprobs,omitempty"`
	LoRA        []loraScale `json:"lora,omitempty"`
}

type completionAnswer struct {
	Choices []struct {
		Text     string `json:"text"`
		LogProbs struct {
			Content []struct {
				Token       string  `json:"token"`
				LogProb     float64 `json:"logprob"`
				TopLogProbs []struct {
					Token   string  `json:"token"`
					LogProb float64 `json:"logprob"`
				} `json:"top_logprobs"`
			} `json:"content"`
		} `json:"logprobs"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
}

type chatTemplateMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatTemplateRequest struct {
	Messages            []chatTemplateMessage `json:"messages"`
	AddGenerationPrompt bool                  `json:"add_generation_prompt"`
	TemplateOptions     struct {
		EnableThinking bool `json:"enable_thinking"`
	} `json:"chat_template_kwargs"`
}

// renderPrompt delegates formatting to the installed backend's chat template.
// The application supplies user content once and preserves non-thinking mode;
// an unavailable or invalid template fails before requesting any generation.
func (e *LlamaEngine) renderPrompt(ctx context.Context, prompt string) (string, error) {
	body, err := json.Marshal(chatTemplateRequest{
		Messages:            []chatTemplateMessage{{Role: "user", Content: prompt}},
		AddGenerationPrompt: true,
	})
	if err != nil {
		return "", err
	}
	answer, err := e.post(ctx, "/apply-template", body)
	if err != nil {
		return "", err
	}
	var parsed struct {
		Prompt string `json:"prompt"`
	}
	if err := json.Unmarshal(answer, &parsed); err != nil {
		return "", errors.New("cluster: the engine returned an invalid chat template answer")
	}
	if strings.TrimSpace(parsed.Prompt) == "" {
		return "", errors.New("cluster: the engine returned an empty rendered prompt")
	}
	return parsed.Prompt, nil
}

// ChoiceMargin is a first-answer-token preference measurement for bounded
// multiple-choice tasks. The raw model log probabilities come from the
// resident quantized representation before verifier classification.
type ChoiceMargin struct {
	Correct       string             `json:"correct"`
	Competitor    string             `json:"competitor"`
	CorrectScore  float64            `json:"correct_score"`
	CompeteScore  float64            `json:"competitor_score"`
	Margin        float64            `json:"margin"`
	OptionScores  map[string]float64 `json:"option_scores"`
	GeneratedText string             `json:"generated_text"`
}

// Complete generates through the private resident server. Bounds are repeated
// here because this method is also used by offline dataset mining callers.
func (e *LlamaEngine) Complete(ctx context.Context, prompt string, maxTokens int) (Completion, error) {
	return e.complete(ctx, prompt, maxTokens, nil)
}

// CompleteCandidateIndex generates with exactly one resident adapter enabled.
// IDs are zero-based in llama-server and bounded by the persisted session recipe.
func (e *LlamaEngine) CompleteCandidateIndex(ctx context.Context, adapterID, adapterCount int, prompt string, maxTokens int) (Completion, error) {
	if adapterCount < 1 || adapterCount > 16 || adapterID < 0 || adapterID >= adapterCount {
		return Completion{}, errors.New("cluster: requested candidate is not in the resident adapter bank")
	}
	return e.complete(ctx, prompt, maxTokens, &adapterID)
}

func (e *LlamaEngine) complete(ctx context.Context, prompt string, maxTokens int, adapterID *int) (Completion, error) {
	if strings.TrimSpace(prompt) == "" || len(prompt) > 4096 {
		return Completion{}, errors.New("cluster: completion prompt has to contain between 1 and 4096 bytes")
	}
	if maxTokens < 1 || maxTokens > 1024 {
		return Completion{}, errors.New("cluster: completion max tokens has to be between 1 and 1024")
	}
	rendered, err := e.renderPrompt(ctx, prompt)
	if err != nil {
		return Completion{}, err
	}
	request := completionRequest{
		Prompt: rendered, MaxTokens: maxTokens,
		Seed: 59, Temperature: 0,
	}
	if adapterID != nil {
		request.LoRA = []loraScale{{ID: *adapterID, Scale: 1}}
	}
	body, err := json.Marshal(request)
	if err != nil {
		return Completion{}, err
	}
	answer, err := e.post(ctx, "/v1/completions", body)
	if err != nil {
		return Completion{}, err
	}
	var parsed completionAnswer
	if err := json.Unmarshal(answer, &parsed); err != nil {
		return Completion{}, fmt.Errorf("cluster: reading the completion answer: %w", err)
	}
	if len(parsed.Choices) != 1 {
		return Completion{}, errors.New("cluster: the engine did not return exactly one completion")
	}
	return Completion{
		Text: parsed.Choices[0].Text, PromptTokens: parsed.Usage.PromptTokens,
		CompletionTokens: parsed.Usage.CompletionTokens,
	}, nil
}

// ChoiceMargin measures the correct option against the strongest declared
// alternative. It asks for raw top-token log probabilities and refuses a
// measurement if any declared option is absent, instead of treating a missing
// candidate as negative infinity and inflating the margin.
func (e *LlamaEngine) ChoiceMargin(ctx context.Context, prompt, correct string, options []string) (ChoiceMargin, error) {
	return e.choiceMargin(ctx, prompt, correct, options, nil)
}

// ChoiceMarginCandidate measures one complete factor-space candidate loaded in
// the resident server. Every other candidate is held at zero for this request.
func (e *LlamaEngine) ChoiceMarginCandidate(ctx context.Context, adapterID int, prompt, correct string, options []string) (ChoiceMargin, error) {
	if !containsAdapter(e.Bank, adapterID) {
		return ChoiceMargin{}, errors.New("cluster: requested candidate is not in the resident adapter bank")
	}
	return e.choiceMargin(ctx, prompt, correct, options, &adapterID)
}

// ChoiceMarginCandidateIndex measures one of the complete candidates declared
// by the persistent session recipe. IDs are zero-based in llama-server.
func (e *LlamaEngine) ChoiceMarginCandidateIndex(ctx context.Context, adapterID, adapterCount int, prompt, correct string, options []string) (ChoiceMargin, error) {
	if adapterCount < 1 || adapterCount > 16 || adapterID < 0 || adapterID >= adapterCount {
		return ChoiceMargin{}, errors.New("cluster: requested candidate is not in the resident adapter bank")
	}
	return e.choiceMargin(ctx, prompt, correct, options, &adapterID)
}

func (e *LlamaEngine) choiceMargin(ctx context.Context, prompt, correct string, options []string, adapterID *int) (ChoiceMargin, error) {
	if strings.TrimSpace(prompt) == "" || len(prompt) > 4096 {
		return ChoiceMargin{}, errors.New("cluster: choice prompt has to contain between 1 and 4096 bytes")
	}
	if len(options) < 2 || len(options) > 8 {
		return ChoiceMargin{}, errors.New("cluster: choice margin needs between two and eight options")
	}
	declared := make(map[string]bool, len(options))
	for _, option := range options {
		if option != strings.TrimSpace(option) || len(option) != 1 {
			return ChoiceMargin{}, errors.New("cluster: each choice option has to be one visible token label")
		}
		declared[option] = true
	}
	if !declared[correct] || len(declared) != len(options) {
		return ChoiceMargin{}, errors.New("cluster: correct choice must be one unique declared option")
	}
	rendered, err := e.renderPrompt(ctx, prompt)
	if err != nil {
		return ChoiceMargin{}, err
	}
	request := completionRequest{
		Prompt: rendered, MaxTokens: 1,
		Seed: 59, Temperature: 0, LogProbs: 256,
	}
	if adapterID != nil {
		request.LoRA = []loraScale{{ID: *adapterID, Scale: 1}}
	}
	body, err := json.Marshal(request)
	if err != nil {
		return ChoiceMargin{}, err
	}
	answer, err := e.post(ctx, "/v1/completions", body)
	if err != nil {
		return ChoiceMargin{}, err
	}
	var parsed completionAnswer
	if err := json.Unmarshal(answer, &parsed); err != nil {
		return ChoiceMargin{}, fmt.Errorf("cluster: reading the choice margin answer: %w", err)
	}
	if len(parsed.Choices) != 1 || len(parsed.Choices[0].LogProbs.Content) != 1 {
		return ChoiceMargin{}, errors.New("cluster: choice margin needs exactly one generated token probability")
	}
	scores := make(map[string]float64, len(options))
	content := parsed.Choices[0].LogProbs.Content[0]
	for _, probability := range append(content.TopLogProbs, struct {
		Token   string  `json:"token"`
		LogProb float64 `json:"logprob"`
	}{Token: content.Token, LogProb: content.LogProb}) {
		label := strings.TrimSpace(probability.Token)
		if declared[label] {
			if prior, exists := scores[label]; !exists || probability.LogProb > prior {
				scores[label] = probability.LogProb
			}
		}
	}
	for _, option := range options {
		if _, exists := scores[option]; !exists {
			return ChoiceMargin{}, fmt.Errorf("cluster: option %s is absent from the admitted top-token probabilities", option)
		}
	}
	competitor := ""
	competitorScore := math.Inf(-1)
	for _, option := range options {
		if option != correct && scores[option] > competitorScore {
			competitor, competitorScore = option, scores[option]
		}
	}
	return ChoiceMargin{
		Correct: correct, Competitor: competitor, CorrectScore: scores[correct],
		CompeteScore: competitorScore, Margin: scores[correct] - competitorScore,
		OptionScores: scores, GeneratedText: parsed.Choices[0].Text,
	}, nil
}

func containsAdapter(bank []int, id int) bool {
	for _, candidate := range bank {
		if candidate == id {
			return true
		}
	}
	return false
}

// Score asks the engine what the example costs under the current perturbation.
//
// The loss is the mean negative log likelihood over the completion's tokens,
// which is the same quantity the Python trainer reports, so the two are
// comparable. They have to be: the gate this technique has to pass is
// reproducing a loss the backpropagated runs already measured.
func (e *LlamaEngine) Score(ctx context.Context, example Example) (Reading, error) {
	return e.score(ctx, example, nil)
}

// ScoreCandidateIndex scores one complete candidate declared by the resident
// session. IDs are zero-based in llama-server and bounded by the stored recipe.
func (e *LlamaEngine) ScoreCandidateIndex(ctx context.Context, adapterID, adapterCount int, example Example) (Reading, error) {
	if adapterCount < 1 || adapterCount > 16 || adapterID < 0 || adapterID >= adapterCount {
		return Reading{}, errors.New("cluster: requested candidate is not in the resident adapter bank")
	}
	return e.score(ctx, example, &adapterID)
}

func (e *LlamaEngine) score(ctx context.Context, example Example, adapterID *int) (Reading, error) {
	started := time.Now()
	request := scoreRequest{
		Prompt:      example.Prompt + example.Completion,
		MaxTokens:   0,
		Echo:        true,
		LogProbs:    1,
		Temperature: 0,
	}
	if adapterID != nil {
		request.LoRA = []loraScale{{ID: *adapterID, Scale: 1}}
	}
	body, err := json.Marshal(request)
	if err != nil {
		return Reading{}, err
	}
	answer, err := e.post(ctx, "/v1/completions", body)
	if err != nil {
		return Reading{}, err
	}
	var parsed scoreAnswer
	if err := json.Unmarshal(answer, &parsed); err != nil {
		return Reading{}, fmt.Errorf("cluster: reading the engine's answer: %w", err)
	}
	if len(parsed.Choices) == 0 {
		return Reading{}, errors.New("cluster: the engine returned no choices; a loss cannot be read from it")
	}
	// The first token has no logprob -- nothing preceded it to predict it -- and
	// the engine reports that as null rather than zero. Counting it as zero would
	// divide by one token too many and quietly lower every loss.
	sum, n := 0.0, 0
	for _, token := range parsed.Choices[0].LogProbs.Content {
		if token.LogProb == nil {
			continue
		}
		sum += *token.LogProb
		n++
	}
	if n == 0 {
		return Reading{}, errors.New("cluster: the engine echoed no token probabilities; check that it was asked with echo and logprobs")
	}
	return Reading{Loss: -sum / float64(n), Tokens: n, Elapsed: time.Since(started)}, nil
}

func (e *LlamaEngine) post(ctx context.Context, path string, body []byte) ([]byte, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, e.BaseURL+path, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", "application/json")
	if e.apiKey != "" {
		request.Header.Set("Authorization", "Bearer "+e.apiKey)
	}
	answer, err := e.Client.Do(request)
	if err != nil {
		return nil, fmt.Errorf("cluster: reaching the engine at %s: %w", path, err)
	}
	defer answer.Body.Close()
	read, err := io.ReadAll(answer.Body)
	if err != nil {
		return nil, err
	}
	if answer.StatusCode >= 300 {
		return read, fmt.Errorf("cluster: the engine answered HTTP %d on %s: %s", answer.StatusCode, path, read)
	}
	return read, nil
}

// ScoreBatch is the loss of several examples read as one number.
//
// The sums are added and the counts are added, and the division happens once at
// the end. Averaging the per-example means instead would weight a three-token
// completion the same as a forty-eight-token one, and the figure would stop
// being the quantity a backpropagating trainer reports for the same data --
// which is the whole reason this loss is measured here rather than guessed.
//
// Reading.Loss is a mean and Reading.Tokens is what it is a mean over, so the
// sum is recovered by multiplying. That is exact in float64 for these
// magnitudes, and it keeps Engine.Score with one job.
//
// The engine is scored example by example because a context holds one sequence
// at a time; batching would mean either a longer context or several contexts,
// and both cost memory this card does not have to spare.
func ScoreBatch(ctx context.Context, engine Engine, examples []Example) (Reading, error) {
	if len(examples) == 0 {
		return Reading{}, errors.New("cluster: a batch needs at least one example")
	}
	started := time.Now()
	sum, tokens := 0.0, 0
	for _, e := range examples {
		if ctx.Err() != nil {
			return Reading{}, ctx.Err()
		}
		reading, err := engine.Score(ctx, e)
		if err != nil {
			return Reading{}, fmt.Errorf("cluster: scoring %s: %w", e.ID, err)
		}
		sum += reading.Loss * float64(reading.Tokens)
		tokens += reading.Tokens
	}
	if tokens == 0 {
		return Reading{}, errors.New("cluster: the batch scored no position; a mean over nothing reads the same as a perfect prediction")
	}
	return Reading{Loss: sum / float64(tokens), Tokens: tokens, Elapsed: time.Since(started)}, nil
}

// StepOver is one zeroth-order step taken against a batch.
//
// It is the same central difference as Step, over the batch loss instead of one
// example's. That matters for what the step means: a direction is accepted or
// rejected by whether it lowers the loss across the whole batch, so a
// perturbation that helps one example and hurts three is rejected. Stepping on
// one example measures only that example's response to the direction.
//
// The cost is linear in the batch: two scores per example per step instead of
// two. That is the price of an estimate that is about the data rather than
// about one row of it.
func StepOver(ctx context.Context, engine Engine, examples []Example, d adapter.Direction) (Estimate, error) {
	if d.Scale <= 0 {
		return Estimate{}, errors.New("cluster: the perturbation scale has to be positive; a zero step measures the same point twice and estimates nothing")
	}
	if len(examples) == 0 {
		return Estimate{}, errors.New("cluster: a step needs at least one example")
	}
	plus, err := probeBatchAt(ctx, engine, examples, d, +1)
	if err != nil {
		return Estimate{}, restoreAfter(ctx, engine, d, err)
	}
	minus, err := probeBatchAt(ctx, engine, examples, d, -1)
	if err != nil {
		return Estimate{}, restoreAfter(ctx, engine, d, err)
	}
	if err := engine.Perturb(ctx, d, 0); err != nil {
		return Estimate{}, fmt.Errorf("cluster: the engine stayed perturbed after the step: %w", err)
	}
	estimate := Estimate{Direction: d, Plus: plus, Minus: minus}
	estimate.Gradient = (plus.Loss - minus.Loss) / (2 * d.Scale)
	if math.IsNaN(estimate.Gradient) || math.IsInf(estimate.Gradient, 0) {
		return estimate, fmt.Errorf("cluster: the estimate is not finite: L(+)=%v L(-)=%v scale=%v", plus.Loss, minus.Loss, d.Scale)
	}
	return estimate, nil
}

func probeBatchAt(ctx context.Context, engine Engine, examples []Example, d adapter.Direction, sign float64) (Reading, error) {
	if err := engine.Perturb(ctx, d, sign); err != nil {
		return Reading{}, err
	}
	return ScoreBatch(ctx, engine, examples)
}
