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

func decimal(v int) string {
	const digits = "0123456789"
	if v < 10 {
		return string(digits[v])
	}
	return string([]byte{digits[v/10], digits[v%10]})
}
