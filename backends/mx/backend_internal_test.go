package mx

import (
	"strings"
	"testing"
	"time"
)

func TestMXProcessSpecsAreBoundedAndUseNoShell(t *testing.T) {
	b := &MXBackend{runtimeRoot: "/runtime/mx", modelRoot: t.TempDir()}
	rpc, err := b.RPC(MXRPCRequest{
		BindIP: "100.64.0.2", Port: 50052, Devices: []int{0, 1}, Deadline: time.Hour,
	})
	if err != nil {
		t.Fatal(err)
	}
	if rpc.Path != "/usr/bin/timeout" || !strings.Contains(strings.Join(rpc.Args, " "), "/runtime/mx/bin/ggml-rpc-server") {
		t.Fatalf("unexpected RPC process: %+v", rpc)
	}
	if got := runtimeSetting(rpc.Env, "GGML_CUDA_Q8_1_CACHE"); got != "0" {
		t.Fatalf("RPC q8_1 cache setting = %q, want 0", got)
	}
	if _, err := b.RPC(MXRPCRequest{BindIP: "0.0.0.0", Port: 50052, Devices: []int{0, 1}, Deadline: time.Hour}); err == nil {
		t.Fatal("public RPC bind was accepted")
	}

	endpoints := make([]string, 20)
	for i := range endpoints {
		endpoints[i] = "100.64.0." + decimal(i+2) + ":50052"
	}
	if _, err := b.Generate(MXGenerateRequest{RPCEndpoints: endpoints, Prompt: "TAYI_OK", MaxTokens: 8, Context: 4096, Deadline: time.Hour}); err == nil {
		t.Fatal("missing admitted model shard was accepted")
	}
	endpoints[19] = endpoints[0]
	if _, err := b.Generate(MXGenerateRequest{RPCEndpoints: endpoints, Prompt: "TAYI_OK", MaxTokens: 8, Context: 4096, Deadline: time.Hour}); err == nil || !strings.Contains(err.Error(), "unique") {
		t.Fatalf("duplicate endpoint refusal = %v", err)
	}
}

func TestMXRuntimeEnvironmentCannotInheritUnsafeQ81CacheSetting(t *testing.T) {
	t.Setenv("GGML_CUDA_Q8_1_CACHE", "1")
	environment := mxRuntimeEnvironment()
	seen := 0
	for _, value := range environment {
		if strings.HasPrefix(value, "GGML_CUDA_Q8_1_CACHE=") {
			seen++
		}
	}
	if seen != 1 || runtimeSetting(environment, "GGML_CUDA_Q8_1_CACHE") != "0" {
		t.Fatalf("qualified environment has %d cache settings: %v", seen, environment)
	}
}

func TestMXGenerateTerminatesAfterOneConversationTurn(t *testing.T) {
	b := &MXBackend{runtimeRoot: "/runtime/mx", modelRoot: t.TempDir()}
	endpoints := make([]string, 20)
	for i := range endpoints {
		endpoints[i] = "100.64.0." + decimal(i+2) + ":50052"
	}
	process := b.generateProcess("/models/deepseek.gguf", MXGenerateRequest{
		RPCEndpoints: endpoints, Prompt: "TAYI_FLASH_OK", MaxTokens: 1024,
		Context: 4096, Deadline: time.Hour,
	})
	joined := strings.Join(process.Args, " ")
	if !strings.Contains(joined, "--single-turn") {
		t.Fatalf("generation can remain in the interactive prompt loop: %s", joined)
	}
}

func TestMXResidentServerIsPrivateBoundAndKeepsAuthenticationOutOfArguments(t *testing.T) {
	b := &MXBackend{runtimeRoot: "/runtime/mx", modelRoot: t.TempDir()}
	endpoints := make([]string, 20)
	for i := range endpoints {
		endpoints[i] = "100.64.0." + decimal(i+2) + ":50052"
	}
	request := MXServeRequest{
		RPCEndpoints: endpoints, BindIP: "100.64.0.1", Port: 50053,
		Context: 4096, Deadline: 6 * time.Hour,
		Adapters: []MXAdapter{
			{Path: "/artifacts/probe-plus/adapter.gguf", SHA256: strings.Repeat("a", 64)},
			{Path: "/artifacts/probe-minus/adapter.gguf", SHA256: strings.Repeat("b", 64)},
		},
	}
	process := b.serveProcess("/models/deepseek.gguf", request)
	joined := strings.Join(process.Args, " ")
	for _, required := range []string{
		"/runtime/mx/bin/llama-server", "--rpc", "--host 100.64.0.1",
		"--port 50053", "--alias tayi-flash", "--metrics", "--no-host",
		"--lora-scaled /artifacts/probe-plus/adapter.gguf:0,/artifacts/probe-minus/adapter.gguf:0",
	} {
		if !strings.Contains(joined, required) {
			t.Fatalf("resident server args omit %q: %s", required, joined)
		}
	}
	if strings.Count(joined, "--lora-scaled") != 1 {
		t.Fatalf("resident server must pass one comma-separated adapter list: %s", joined)
	}
	if strings.Contains(joined, "api-key") {
		t.Fatalf("resident server exposed authentication in arguments: %s", joined)
	}
	if got := runtimeSetting(process.Env, "GGML_CUDA_Q8_1_CACHE"); got != "0" {
		t.Fatalf("resident server q8_1 cache setting = %q, want 0", got)
	}

	request.Adapters = nil
	request.BindIP = "0.0.0.0"
	if _, err := b.Serve(request); err == nil || !strings.Contains(err.Error(), "private") {
		t.Fatalf("public server bind refusal = %v", err)
	}
	request.BindIP, request.Port = "100.64.0.1", 8080
	if _, err := b.Serve(request); err == nil || !strings.Contains(err.Error(), "port") {
		t.Fatalf("unadmitted server port refusal = %v", err)
	}
}

func runtimeSetting(environment []string, name string) string {
	prefix := name + "="
	for _, value := range environment {
		if strings.HasPrefix(value, prefix) {
			return strings.TrimPrefix(value, prefix)
		}
	}
	return ""
}

func decimal(v int) string {
	const digits = "0123456789"
	if v < 10 {
		return string(digits[v])
	}
	return string([]byte{digits[v/10], digits[v%10]})
}
