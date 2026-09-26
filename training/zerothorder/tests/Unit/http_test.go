package unit_test

import (
	"github.com/tayi-ai/arandu-llama/training/adapter"

	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	tests "github.com/tayi-ai/arandu-llama/training/adapter/tests/Fixtures"
	services "github.com/tayi-ai/arandu-llama/training/zerothorder"
)

// The server engine against a fake llama-server, and the one thing it must not
// be allowed to do.
//
// llama-server can load several adapters and rescale each through POST
// /lora-adapters, and the first zeroth-order engine used that: the policy at
// scale one and a direction adapter at scale c. What that measures is the
// derivative of c*(alpha/r)*ZB*ZA -- the product of the direction's own
// factors -- while the accepted step adds c*ZA to A and c*ZB to B, whose
// derivative is B*ZA + ZB*A. On the scalar oracle of the T02 addendum, A = 2,
// B = 3, ZA = 5, ZB = -7, those two numbers are -35 and 1. A probe reading -35
// where the step moves along +1 is a training run that walks away from its
// own measurement, and on the fleet it presented as a loss that did not fall.
//
// The server cannot build a candidate in factor space, so it is not allowed to
// probe at all. This test stands the fake server up, gives the engine both
// slots, and checks that a step neither succeeds nor ever asks the server to
// scale the direction slot away from zero. On the engine before the fix it
// succeeds with gradient -35, and the failure message prints that number.

// fakeLlamaServer answers the two endpoints the engine speaks: it keeps the
// scale of every adapter id and scores as the sum over adapters of
// scale * B * A for the 1x1 fixtures it is handed, returning the output as a
// single token whose logprob is minus that output, so the loss reads as the
// output itself.
type fakeLlamaServer struct {
	mu       sync.Mutex
	adapters []string
	scales   map[int]float64
	// directionScales records every scale the engine ever asked for on a slot
	// other than the first. A probe through the server is exactly a non-zero
	// entry here.
	directionScales []float64
}

func (f *fakeLlamaServer) handle(t *testing.T) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		f.mu.Lock()
		defer f.mu.Unlock()
		switch r.URL.Path {
		case "/lora-adapters":
			var asked []struct {
				ID    int     `json:"id"`
				Scale float64 `json:"scale"`
			}
			if err := json.NewDecoder(r.Body).Decode(&asked); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			for _, a := range asked {
				f.scales[a.ID] = a.Scale
				if a.ID != 0 {
					f.directionScales = append(f.directionScales, a.Scale)
				}
			}
			w.Write([]byte("{}"))
		case "/v1/completions":
			output := 0.0
			for id, path := range f.adapters {
				a, b := scalarFactors(t, path)
				output += f.scales[id] * b * a
			}
			answer := map[string]any{"choices": []map[string]any{{
				"logprobs": map[string]any{"content": []map[string]any{{"logprob": -output}}},
			}}}
			json.NewEncoder(w).Encode(answer)
		default:
			http.NotFound(w, r)
		}
	}
}

// scalarFactors reads the single element of lora_a and lora_b out of a 1x1
// fixture, through the same reader the application uses.
func scalarFactors(t *testing.T, path string) (a, b float64) {
	t.Helper()
	container, err := adapter.ReadGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, tensor := range container.Tensors {
		value := float64(math.Float32frombits(binary.LittleEndian.Uint32(body[container.DataOffset+tensor.Offset:])))
		switch {
		case strings.HasSuffix(tensor.Name, ".lora_a"):
			a = value
		case strings.HasSuffix(tensor.Name, ".lora_b"):
			b = value
		}
	}
	return a, b
}

func TestTheServerEngineIsNotAllowedToProbe(t *testing.T) {
	dir := t.TempDir()
	policy := filepath.Join(dir, "policy.gguf")
	direction := filepath.Join(dir, "direction.gguf")
	tests.WriteLoRAGGUF(t, policy, "oracle", 1, []tests.LoRATensor{
		{Name: "blk.0.attn_q.weight.lora_a", Dims: []uint64{1, 1}, Data: []float32{2}},
		{Name: "blk.0.attn_q.weight.lora_b", Dims: []uint64{1, 1}, Data: []float32{3}},
	})
	tests.WriteLoRAGGUF(t, direction, "oracle", 1, []tests.LoRATensor{
		{Name: "blk.0.attn_q.weight.lora_a", Dims: []uint64{1, 1}, Data: []float32{5}},
		{Name: "blk.0.attn_q.weight.lora_b", Dims: []uint64{1, 1}, Data: []float32{-7}},
	})

	fake := &fakeLlamaServer{adapters: []string{policy, direction}, scales: map[int]float64{0: 1}}
	server := httptest.NewServer(fake.handle(t))
	defer server.Close()

	engine := services.NewLlamaEngine(server.URL)
	engine.Bank = []int{0, 1}

	// The server's Score is a valid measurement on its own, and stays one.
	base, err := engine.Score(context.Background(), services.Example{ID: "oracle"})
	if err != nil {
		t.Fatal(err)
	}
	if base.Loss != 6 {
		t.Fatalf("the unperturbed server read %v, want B*A = 6", base.Loss)
	}

	estimate, err := services.Step(context.Background(), engine, services.Example{ID: "oracle"}, adapter.Direction{Seed: 1, Scale: 1e-3})
	if err == nil {
		t.Fatalf("the server engine probed and estimated gradient=%v; scaling a second adapter measures ZB*ZA = -35 where the step moves along B*ZA + ZB*A = 1, so the server must refuse to probe", estimate.Gradient)
	}
	fake.mu.Lock()
	defer fake.mu.Unlock()
	for _, scale := range fake.directionScales {
		if scale != 0 {
			t.Fatalf("the engine asked the server to scale the direction slot to %v; that is the product-space probe the oracle refutes", scale)
		}
	}
}

func TestTheResidentServerKeyTravelsOnlyInAuthorization(t *testing.T) {
	const key = "resident-server-secret"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer "+key {
			t.Fatalf("authorization header = %q", got)
		}
		if strings.Contains(r.URL.String(), key) {
			t.Fatal("resident server key leaked into the request URL")
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{
			"logprobs": map[string]any{"content": []map[string]any{{"logprob": -0.25}}},
		}}})
	}))
	defer server.Close()

	engine := services.NewAuthenticatedLlamaEngine(server.URL, key)
	reading, err := engine.Score(context.Background(), services.Example{ID: "auth"})
	if err != nil {
		t.Fatal(err)
	}
	if reading.Loss != 0.25 || reading.Tokens != 1 {
		t.Fatalf("authenticated score = %+v", reading)
	}
	dump, err := json.Marshal(engine)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(dump), key) || !strings.Contains(string(dump), "[redacted]") {
		t.Fatalf("resident engine dump did not redact its key: %s", dump)
	}
}

func TestChoiceMarginUsesRawDeclaredOptionProbabilities(t *testing.T) {
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request map[string]any
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		if serveResidentTemplateFixture(t, w, r, request, "Answer with A-D.") {
			return
		}
		if request["logprobs"] != float64(256) || request["max_tokens"] != float64(1) {
			t.Fatalf("choice request = %+v", request)
		}
		if request["prompt"] != "fixture-rendered: Answer with A-D." {
			t.Fatalf("choice prompt was not rendered once: %q", request["prompt"])
		}
		if _, exists := request["model"]; exists {
			t.Fatal("choice request overrode the resident model")
		}
		requests++
		if requests == 1 {
			if _, exists := request["lora"]; exists {
				t.Fatalf("base choice unexpectedly selected an adapter: %+v", request)
			}
		} else if got := request["lora"].([]any)[0].(map[string]any); got["id"] != float64(2) || got["scale"] != float64(1) {
			t.Fatalf("candidate choice selected %+v", request["lora"])
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{
				"text": "B",
				"logprobs": map[string]any{"content": []map[string]any{{
					"token": "B", "logprob": -0.2,
					"top_logprobs": []map[string]any{
						{"token": " A", "logprob": -2.0}, {"token": "B", "logprob": -0.2},
						{"token": "C", "logprob": -1.1}, {"token": "D", "logprob": -3.0},
					},
				}}},
			}},
		})
	}))
	defer server.Close()

	engine := services.NewLlamaEngine(server.URL)
	engine.Bank = []int{0, 1, 2}
	margin, err := engine.ChoiceMargin(
		context.Background(), "Answer with A-D.", "C", []string{"A", "B", "C", "D"},
	)
	if err != nil {
		t.Fatal(err)
	}
	if margin.Competitor != "B" || math.Abs(margin.Margin-(-0.9)) > 1e-9 {
		t.Fatalf("choice margin = %+v", margin)
	}
	if _, err := engine.ChoiceMarginCandidate(context.Background(), 2, "Answer with A-D.", "C", []string{"A", "B", "C", "D"}); err != nil {
		t.Fatal(err)
	}
	if _, err := engine.ChoiceMarginCandidate(context.Background(), 9, "Answer with A-D.", "C", []string{"A", "B", "C", "D"}); err == nil {
		t.Fatal("choice margin accepted an adapter outside the resident bank")
	}
}

func TestScoreCandidateSelectsDeclaredResidentAdapter(t *testing.T) {
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request map[string]any
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		requests++
		if requests == 1 {
			if _, exists := request["lora"]; exists {
				t.Fatalf("base score unexpectedly selected an adapter: %+v", request)
			}
		} else if got := request["lora"].([]any)[0].(map[string]any); got["id"] != float64(2) || got["scale"] != float64(1) {
			t.Fatalf("candidate score selected %+v", request["lora"])
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"logprobs": map[string]any{"content": []map[string]any{
				{"logprob": nil}, {"logprob": -0.5},
			}}}},
		})
	}))
	defer server.Close()

	engine := services.NewLlamaEngine(server.URL)
	example := services.Example{Prompt: "prompt", Completion: " answer"}
	if _, err := engine.Score(context.Background(), example); err != nil {
		t.Fatal(err)
	}
	if _, err := engine.ScoreCandidateIndex(context.Background(), 2, 3, example); err != nil {
		t.Fatal(err)
	}
	if _, err := engine.ScoreCandidateIndex(context.Background(), 3, 3, example); err == nil {
		t.Fatal("score accepted an adapter outside the resident bank")
	}
}

func TestResidentCompletionIsDeterministicAndBounded(t *testing.T) {
	const key = "resident-server-secret"
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer "+key {
			t.Fatalf("authorization header = %q", got)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if serveResidentTemplateFixture(t, w, r, body, "responda") {
			return
		}
		if body["temperature"] != float64(0) || body["seed"] != float64(59) || body["max_tokens"] != float64(32) {
			t.Fatalf("completion request = %+v", body)
		}
		if body["prompt"] != "fixture-rendered: responda" {
			t.Fatalf("completion prompt was not rendered once: %q", body["prompt"])
		}
		if _, exists := body["model"]; exists {
			t.Fatal("completion request overrode the resident model")
		}
		requests++
		if requests == 1 {
			if _, exists := body["lora"]; exists {
				t.Fatalf("base completion unexpectedly selected an adapter: %+v", body)
			}
		} else {
			got := body["lora"].([]any)[0].(map[string]any)
			if got["id"] != float64(1) || got["scale"] != float64(1) {
				t.Fatalf("candidate completion selected %+v", body["lora"])
			}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"text": "TAYI_FLASH_OK"}},
			"usage":   map[string]any{"prompt_tokens": 4, "completion_tokens": 3},
		})
	}))
	defer server.Close()

	engine := services.NewAuthenticatedLlamaEngine(server.URL, key)
	result, err := engine.Complete(context.Background(), "responda", 32)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "TAYI_FLASH_OK" || result.PromptTokens != 4 || result.CompletionTokens != 3 {
		t.Fatalf("completion = %+v", result)
	}
	if _, err := engine.CompleteCandidateIndex(context.Background(), 1, 2, "responda", 32); err != nil {
		t.Fatal(err)
	}
	if _, err := engine.CompleteCandidateIndex(context.Background(), 2, 2, "responda", 32); err == nil {
		t.Fatal("completion accepted an adapter outside the resident bank")
	}
	if _, err := engine.Complete(context.Background(), "responda", 1025); err == nil {
		t.Fatal("completion accepted more than 1024 tokens")
	}
}

func serveResidentTemplateFixture(t *testing.T, w http.ResponseWriter, r *http.Request, body map[string]any, prompt string) bool {
	t.Helper()
	if r.Method != http.MethodPost {
		t.Fatalf("resident request method = %s", r.Method)
	}
	if r.URL.Path != "/apply-template" {
		if r.URL.Path != "/v1/completions" {
			t.Fatalf("unexpected resident endpoint: %s", r.URL.Path)
		}
		return false
	}
	messages, ok := body["messages"].([]any)
	if !ok || len(messages) != 1 {
		t.Fatalf("chat template messages = %+v", body["messages"])
	}
	message, ok := messages[0].(map[string]any)
	if !ok || message["role"] != "user" || message["content"] != prompt || body["add_generation_prompt"] != true {
		t.Fatalf("chat template content = %+v", body)
	}
	options, ok := body["chat_template_kwargs"].(map[string]any)
	if !ok || options["enable_thinking"] != false {
		t.Fatalf("non-thinking template mode was not preserved: %+v", body)
	}
	if _, exists := body["model"]; exists {
		t.Fatal("chat template request overrode the resident model")
	}
	_ = json.NewEncoder(w).Encode(map[string]string{"prompt": "fixture-rendered: " + prompt})
	return true
}
