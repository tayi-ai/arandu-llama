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

func TestMXManifestPinsQ2SeparatelyWithoutChangingTheQ4Default(t *testing.T) {
	q4 := mx.AdmittedMXManifest()
	q2, err := mx.AdmittedMXManifestFor(mx.MXModelQ2K)
	if err != nil {
		t.Fatal(err)
	}
	if q4.ModelRecipe != mx.MXModelMXFP4 || q4.ModelShards != 12 || q4.ModelQuantisation != "MXFP4" {
		t.Fatalf("Q4 default drifted: %+v", q4)
	}
	if q2.ModelRecipe != mx.MXModelQ2K || q2.ModelShards != 7 || q2.ModelQuantisation != "Q2_K" {
		t.Fatalf("Q2 recipe is incomplete: %+v", q2)
	}
	if q2.ModelRevision != "58d8ac86298fdf85a2440defee08b1abcad32e45" || q2.ModelDigest != "676c159423ab746ba1ce6036f072d84d977921276429927f336f58aefaea9999" {
		t.Fatalf("Q2 source identity drifted: %+v", q2)
	}
	if _, err := mx.AdmittedMXManifestFor(mx.MXModelRecipe("q2-by-name-only")); err == nil {
		t.Fatal("an unpinned Q2 recipe was accepted")
	}
}

func TestMXBackendRefusesUnverifiedInstallation(t *testing.T) {
	if _, err := mx.NewMXBackend(mx.MXConfig{RuntimeRoot: t.TempDir(), ModelRoot: t.TempDir()}); err == nil {
		t.Fatal("unverified runtime was accepted")
	}
}
