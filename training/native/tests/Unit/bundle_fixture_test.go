package native_test

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	services "github.com/tayi-ai/arandu-llama/training/native"
)

type nativeFixture struct {
	experiment            *services.NativeExperiment
	root, bundle, library string
	contract              services.ExperimentNativeContract
	manifest              services.ExperimentNativeManifest
	cpu                   services.ExperimentNativeCPUQualification
	canary                services.ExperimentNativeCanaryProof
	release               services.ExperimentNativeRelease
	now                   time.Time
	real                  bool
}

func hash(body []byte) string { digest := sha256.Sum256(body); return hex.EncodeToString(digest[:]) }

func encode(t *testing.T, value any) []byte {
	t.Helper()
	body, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return body
}

func write(t *testing.T, path string, body []byte, mode os.FileMode) {
	t.Helper()
	if err := os.WriteFile(path, body, mode); err != nil {
		t.Fatal(err)
	}
}

// newNativeFixture creates complete synthetic bundle integrity, without model
// weights, a runtime executable, private source artifacts or environment access.
// Its qualification claims are test data; nothing in this fixture is executed.
func newNativeFixture(t *testing.T, _ bool) *nativeFixture {
	t.Helper()
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	config := testNativeConfiguration()
	f := &nativeFixture{root: root, real: true, now: time.Date(2026, 9, 19, 12, 0, 0, 0, time.UTC)}
	f.bundle = filepath.Join(root, config.BundleName)
	if err := os.MkdirAll(filepath.Join(f.bundle, "runtime"), 0700); err != nil {
		t.Fatal(err)
	}
	f.library = filepath.Join(root, "libtorch.so")
	library := []byte("CPU test library; never loaded")
	write(t, f.library, library, 0600)
	files := map[string][]byte{}
	for _, name := range []string{"training.json", "config.json", "model.safetensors.index.json", "tokenizer.json", "initial-reference.json", "calibration-reference.json"} {
		files[name] = []byte("synthetic " + name)
	}
	var inventory []map[string]string
	for _, name := range []string{"config.json", "model.safetensors.index.json", "tokenizer.json"} {
		inventory = append(inventory, map[string]string{"path": name, "sha256": hash(files[name])})
	}
	files["base-manifest.json"] = encode(t, map[string]any{"files": inventory})
	files["runtime/tayi-native"] = []byte("CPU test executable; never executed")
	files["tokenization-proof.json"] = encode(t, map[string]any{"all_ids_match": true, "examples": 378, "mismatches": []string{}, "model_weights_loaded": false, "oracle": "huggingface/tokenizers", "python_executed": false, "tokens": 1000, "version": "0.22.2"})
	for name, body := range files {
		mode := os.FileMode(0600)
		if name == "runtime/tayi-native" {
			mode = 0700
		}
		write(t, filepath.Join(f.bundle, name), body, mode)
	}
	config.Recipe.Model.ManifestSHA256 = hash(files["base-manifest.json"])
	config.Recipe.Data.SHA256 = hash(files["training.json"])
	config.IndexSHA256 = hash(files["model.safetensors.index.json"])
	config.ConfigSHA256 = hash(files["config.json"])
	config.TokenizerSHA256 = hash(files["tokenizer.json"])
	config.ReferenceSHA256 = hash(files["initial-reference.json"])
	document := encode(t, config)
	path := filepath.Join(root, "installation.json")
	write(t, path, document, 0600)
	f.experiment, err = services.LoadNativeExperiment(path, hash(document))
	if err != nil {
		t.Fatal(err)
	}
	recipe := services.ExperimentNativeRecipe(f.experiment)
	f.manifest = services.ExperimentNativeManifest{SchemaVersion: 2, ContractVersion: services.ExperimentNativeContractVersion,
		ModelRepo: recipe.Model.Repository, ModelRevision: recipe.Model.Revision, ModelManifestSHA256: recipe.Model.ManifestSHA256,
		TrainingSHA256: recipe.Data.SHA256, InitialDigest: recipe.LoRA.ExpectedInitialDigest, LibTorchVersion: "2.14.0",
		Files: map[string]string{}, Libraries: map[string]string{f.library: hash(library)}}
	f.contract = services.ExperimentNativeContract{ConfigurationSHA256: f.experiment.RecipeSHA256(), BasePath: config.BasePath, Nodes: services.ExperimentNativeNodeIDs(f.experiment),
		Admission: services.ExperimentNativeAdmission{SchemaVersion: 2, ContractVersion: services.ExperimentNativeContractVersion,
			ModelManifestSHA256: recipe.Model.ManifestSHA256, TopologyIDs: services.ExperimentNativeNodeIDs(f.experiment), NNodes: 20, ProcessWorldSize: 20,
			GPUsPerNode: 2, GlobalBatchSize: 21, OptimizerSteps: 18, Examples: 378, Epochs: 1, VRAMFraction: "0.80",
			DeadlinesSeconds: map[string]int{"canary": 900, "train": 5400},
			Jobs:             map[string]services.ExperimentNativeJobIdentity{"canary": {ID: "native-canary", Generation: 1}, "train": {ID: "native-train", Generation: 2}},
			NotBeforeUTC:     f.now.Add(-time.Hour).Format(time.RFC3339), ExpiresAtUTC: f.now.Add(6 * time.Hour).Format(time.RFC3339)}}
	f.cpu = services.ExperimentNativeCPUQualification{SchemaVersion: 1, Passed: true, ForwardVJP: true, InitialDigest: recipe.LoRA.ExpectedInitialDigest, TestsSHA256: hash([]byte("synthetic CPU qualification"))}
	f.seal(t, false)
	return f
}
