package unit_test

import (
	"testing"

	mx "github.com/tayi-ai/arandu-llama/backends/mx"
)

func TestMXManifestKeepsTheQualifiedBuildIdentity(t *testing.T) {
	m := mx.AdmittedMXManifest()
	if m.Backend != mx.MXBackendID || m.Revision != "a245214d8df6304762c7688c6b8ee45652c5c8e5" {
		t.Fatalf("unexpected MX identity: %+v", m)
	}
	if m.CUDAArchitecture != "75" || m.SchedulerBackends != 64 {
		t.Fatalf("qualified build settings drifted: %+v", m)
	}
	if len(m.Binaries) != 4 || m.Binaries["libllama.so.0.3.0"] != "92cb8adca8177feb417a9c3aee856ee1a72f1390f387459b9a533fd52c0408ad" {
		t.Fatalf("qualified binary set has %d entries", len(m.Binaries))
	}
	if m.RuntimeEnvironment["GGML_CUDA_Q8_1_CACHE"] != "0" {
		t.Fatalf("qualified runtime environment drifted: %+v", m.RuntimeEnvironment)
	}
}

func TestMXBackendRefusesUnverifiedInstallation(t *testing.T) {
	if _, err := mx.NewMXBackend(mx.MXConfig{RuntimeRoot: t.TempDir(), ModelRoot: t.TempDir()}); err == nil {
		t.Fatal("unverified runtime was accepted")
	}
}
