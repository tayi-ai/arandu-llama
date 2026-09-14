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
	if len(m.Binaries) != 3 {
		t.Fatalf("qualified binary set has %d entries", len(m.Binaries))
	}
}

func TestMXBackendRefusesUnverifiedInstallation(t *testing.T) {
	if _, err := mx.NewMXBackend(mx.MXConfig{RuntimeRoot: t.TempDir(), ModelRoot: t.TempDir()}); err == nil {
		t.Fatal("unverified runtime was accepted")
	}
}
