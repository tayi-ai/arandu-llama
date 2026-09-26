package unit_test

import (
	"strings"
	"testing"

	mx "github.com/tayi-ai/arandu-llama/backends/mx"
)

func mxIdentity() mx.MXModelIdentity {
	return mx.MXModelIdentity{
		Recipe: "example-base", Repository: "example/model", Revision: strings.Repeat("a", 40),
		Quantisation: "Q4_K_M", Manifest: strings.Repeat("b", 64), FirstShard: "model.gguf", Shards: 1,
	}
}

func TestMXCatalogKeepsTheQualifiedBuildIdentity(t *testing.T) {
	model := mxIdentity()
	catalog, err := mx.NewMXCatalog([]mx.MXModelIdentity{model})
	if err != nil {
		t.Fatal(err)
	}
	m, err := catalog.ManifestFor(model.Recipe)
	if err != nil {
		t.Fatal(err)
	}
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

func TestMXCatalogSelectsIndependentModelsWithoutImplicitDefaults(t *testing.T) {
	first, second := mxIdentity(), mxIdentity()
	second.Recipe, second.Repository = "example-second", "other/model"
	second.Revision, second.Manifest = strings.Repeat("c", 64), strings.Repeat("d", 64)
	second.Quantisation, second.FirstShard, second.Shards = "Q6_K", "part-00001-of-00002.gguf", 2
	entries := []mx.MXModelIdentity{first, second}
	catalog, err := mx.NewMXCatalog(entries)
	if err != nil {
		t.Fatal(err)
	}
	entries[0] = second
	for _, want := range []mx.MXModelIdentity{first, second} {
		got, err := catalog.ModelFor(want.Recipe)
		if err != nil || got != want {
			t.Fatalf("selected identity = %+v, %v; want %+v", got, err, want)
		}
		manifest, err := catalog.ManifestFor(want.Recipe)
		if err != nil || manifest.ModelDigest != want.Manifest || manifest.ModelRevision != want.Revision || manifest.ModelRepository != want.Repository || manifest.ModelFirstShard != want.FirstShard || manifest.ModelShards != want.Shards || manifest.ModelQuantisation != want.Quantisation {
			t.Fatalf("selected manifest = %+v, %v", manifest, err)
		}
		recipe, err := catalog.RecipeForDigest(want.Manifest)
		if err != nil || recipe != want.Recipe {
			t.Fatalf("persisted digest selected %q, %v", recipe, err)
		}
		// A caller owns its returned maps and values, never the catalog state.
		manifest.Binaries["llama-cli"] = "changed"
		manifest.RuntimeEnvironment["GGML_CUDA_Q8_1_CACHE"] = "1"
		got.Manifest = "changed"
		again, err := catalog.ManifestFor(want.Recipe)
		if err != nil || again.ModelDigest != want.Manifest || again.Binaries["llama-cli"] == "changed" || again.RuntimeEnvironment["GGML_CUDA_Q8_1_CACHE"] != "0" {
			t.Fatalf("catalog identity changed through output: %+v, %v", again, err)
		}
	}
	for _, invalid := range []mx.MXModelRecipe{"", "unknown", "example-base "} {
		if _, err := catalog.ManifestFor(invalid); err == nil {
			t.Fatalf("unadmitted selector %q chose a model", invalid)
		}
	}
	if _, err := catalog.RecipeForDigest(strings.Repeat("e", 64)); err == nil {
		t.Fatal("unknown persisted digest chose a model")
	}
	for _, empty := range []*mx.MXCatalog{nil, {}} {
		if _, err := empty.ModelFor(first.Recipe); err == nil {
			t.Fatal("empty catalog selected a model")
		}
		if _, err := empty.RecipeForDigest(first.Manifest); err == nil {
			t.Fatal("empty catalog resolved a digest")
		}
	}
}

func TestMXCatalogRejectsMalformedAndAmbiguousIdentities(t *testing.T) {
	changes := map[string]func(*mx.MXModelIdentity){
		"missing recipe":        func(m *mx.MXModelIdentity) { m.Recipe = "" },
		"option selector":       func(m *mx.MXModelIdentity) { m.Recipe = "--help" },
		"blank repository":      func(m *mx.MXModelIdentity) { m.Repository = "" },
		"repository whitespace": func(m *mx.MXModelIdentity) { m.Repository = "org/model\n" },
		"mutable revision":      func(m *mx.MXModelIdentity) { m.Revision = "main" },
		"uppercase revision":    func(m *mx.MXModelIdentity) { m.Revision = strings.Repeat("A", 40) },
		"zero revision":         func(m *mx.MXModelIdentity) { m.Revision = strings.Repeat("0", 40) },
		"missing manifest":      func(m *mx.MXModelIdentity) { m.Manifest = "" },
		"nonhex manifest":       func(m *mx.MXModelIdentity) { m.Manifest = strings.Repeat("z", 64) },
		"blank quantisation":    func(m *mx.MXModelIdentity) { m.Quantisation = "" },
		"shard traversal":       func(m *mx.MXModelIdentity) { m.FirstShard = "../model.gguf" },
		"absolute shard":        func(m *mx.MXModelIdentity) { m.FirstShard = "/model.gguf" },
		"shard separator":       func(m *mx.MXModelIdentity) { m.FirstShard = "a,model.gguf" },
		"empty shards":          func(m *mx.MXModelIdentity) { m.Shards = 0 },
		"unbounded shards":      func(m *mx.MXModelIdentity) { m.Shards = 100000 },
	}
	for name, change := range changes {
		t.Run(name, func(t *testing.T) {
			model := mxIdentity()
			change(&model)
			if _, err := mx.NewMXCatalog([]mx.MXModelIdentity{model}); err == nil {
				t.Fatal("malformed identity was admitted")
			}
			if _, err := mx.NewMXBackend(mx.MXConfig{RuntimeRoot: t.TempDir(), ModelRoot: t.TempDir(), Model: model}); err == nil {
				t.Fatal("malformed backend identity was admitted")
			}
		})
	}
	if _, err := mx.NewMXCatalog(nil); err == nil {
		t.Fatal("empty catalog was admitted")
	}
	first, second := mxIdentity(), mxIdentity()
	if _, err := mx.NewMXCatalog([]mx.MXModelIdentity{first, second}); err == nil {
		t.Fatal("duplicate recipe was admitted")
	}
	second.Recipe = "example-second"
	if _, err := mx.NewMXCatalog([]mx.MXModelIdentity{first, second}); err == nil {
		t.Fatal("ambiguous persisted model digest was admitted")
	}
}

func TestMXBackendRefusesUnverifiedInstallation(t *testing.T) {
	if _, err := mx.NewMXBackend(mx.MXConfig{RuntimeRoot: t.TempDir(), ModelRoot: t.TempDir(), Model: mxIdentity()}); err == nil {
		t.Fatal("unverified runtime was accepted")
	}
	if _, err := mx.NewMXBackend(mx.MXConfig{RuntimeRoot: t.TempDir(), ModelRoot: t.TempDir()}); err == nil {
		t.Fatal("absent model identity selected an implicit default")
	}
}
