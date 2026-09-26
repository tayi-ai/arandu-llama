package native_test

import (
	services "github.com/tayi-ai/arandu-llama/training/native"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestNativeAdmissionRequiresMatchingDecodedAndSerializedContract(t *testing.T) {
	f := newNativeFixture(t, false)
	body := encode(t, f.contract)
	if _, err := services.AdmissionForContract(f.experiment, f.contract, body, "canary", f.release); err != nil {
		t.Fatal(err)
	}
	f.contract.Admission.ExpiresAtUTC = "2026-09-19T23:00:00Z"
	if _, err := services.AdmissionForContract(f.experiment, f.contract, body, "canary", f.release); err == nil {
		t.Fatal("different contract bytes and metadata admitted")
	}
}

func (f *nativeFixture) seal(t *testing.T, canary bool) {
	t.Helper()
	write(t, filepath.Join(f.bundle, "contract.json"), encode(t, f.contract), 0600)
	f.manifest.Files = map[string]string{}
	for _, name := range []string{"contract.json", "training.json", "base-manifest.json", "config.json", "model.safetensors.index.json",
		"tokenizer.json", "initial-reference.json", "calibration-reference.json", "tokenization-proof.json", "runtime/tayi-native"} {
		body, err := os.ReadFile(filepath.Join(f.bundle, name))
		if err != nil {
			t.Fatal(err)
		}
		f.manifest.Files[name] = hash(body)
	}
	if !f.real {
		f.manifest.Files["base-manifest.json"], f.manifest.Files["training.json"] = f.manifest.ModelManifestSHA256, f.manifest.TrainingSHA256
	}
	f.manifest.RuntimeSHA256 = f.manifest.Files["runtime/tayi-native"]
	evidence := services.ExperimentNativeEvidence{RuntimeSHA256: f.manifest.RuntimeSHA256, Libraries: f.manifest.Libraries, Inputs: map[string]string{}}
	for name, digest := range f.manifest.Files {
		if name != "runtime/tayi-native" {
			evidence.Inputs[name] = digest
		}
	}
	f.cpu.Evidence = evidence
	body := encode(t, f.cpu)
	write(t, filepath.Join(f.bundle, "cpu-qualification.json"), body, 0600)
	f.manifest.Files["cpu-qualification.json"] = hash(body)
	if canary {
		f.canary.Evidence = evidence
		body := encode(t, f.canary)
		write(t, filepath.Join(f.bundle, "canary-proof.json"), body, 0600)
		f.manifest.Files["canary-proof.json"] = hash(body)
	}
	body = encode(t, f.manifest)
	write(t, filepath.Join(f.bundle, "manifest.json"), body, 0600)
	f.release.ManifestSHA256 = hash(body)
}

func (f *nativeFixture) job(t *testing.T, mode string) services.Job {
	t.Helper()
	identity := f.contract.Admission.Jobs[mode]
	return services.Job{ID: identity.ID, Generation: identity.Generation, Action: "experiment.train", ContractVersion: services.ExperimentNativeContractVersion,
		RuntimeDigest: f.release.ManifestSHA256, ModelRecipe: "fixture-decoder", ModelDigest: f.manifest.ModelManifestSHA256,
		Request: encode(t, map[string]string{"mode": mode, "contract_sha256": f.manifest.Files["contract.json"]})}
}

func noOutput(t *testing.T, f *nativeFixture) {
	t.Helper()
	if _, err := os.Stat(filepath.Join(f.root, "runs")); !os.IsNotExist(err) {
		t.Fatal("rejected admission reserved output")
	}
}

func TestNativeContractRejectsVersionTopologyJobsAndLeaseBeforeIntegrityWork(t *testing.T) {
	for name, change := range map[string]func(*services.ExperimentNativeContract){
		"native_v2": func(c *services.ExperimentNativeContract) { c.Admission.ContractVersion = "tayi.experiment.native.v2" },
		"excluded_nine": func(c *services.ExperimentNativeContract) {
			c.Nodes[0], c.Admission.TopologyIDs[0] = "nine", "nine"
		},
		"controller_zero": func(c *services.ExperimentNativeContract) {
			c.Nodes[0], c.Admission.TopologyIDs[0] = "zero", "zero"
		},
		"old_full_topology": func(c *services.ExperimentNativeContract) {
			c.Nodes, c.Admission.TopologyIDs = append(services.ExperimentNativeNodeIDs(testNativeExperiment()), "extra-node"), append(services.ExperimentNativeNodeIDs(testNativeExperiment()), "extra-node")
			c.Admission.NNodes, c.Admission.ProcessWorldSize = len(c.Nodes), len(c.Nodes)
		},
		"legacy_version": func(c *services.ExperimentNativeContract) { c.Admission.ContractVersion = "tayi.experiment.ddp21.v1" },
		"legacy_schema":  func(c *services.ExperimentNativeContract) { c.Admission.SchemaVersion = 1 },
		"model_cache":    func(c *services.ExperimentNativeContract) { c.BasePath = "/arbitrary" },
		"rank_order":     func(c *services.ExperimentNativeContract) { c.Nodes[0], c.Nodes[1] = c.Nodes[1], c.Nodes[0] },
		"node_count":     func(c *services.ExperimentNativeContract) { c.Admission.NNodes = 21 },
		"process_world":  func(c *services.ExperimentNativeContract) { c.Admission.ProcessWorldSize = 42 },
		"gpus":           func(c *services.ExperimentNativeContract) { c.Admission.GPUsPerNode = 1 },
		"updates":        func(c *services.ExperimentNativeContract) { c.Admission.OptimizerSteps = 17 },
		"examples":       func(c *services.ExperimentNativeContract) { c.Admission.Examples = 377 },
		"vram":           func(c *services.ExperimentNativeContract) { c.Admission.VRAMFraction = "0.95" },
		"long_deadline":  func(c *services.ExperimentNativeContract) { c.Admission.DeadlinesSeconds["canary"] = 1801 },
		"duplicate_job":  func(c *services.ExperimentNativeContract) { c.Admission.Jobs["train"] = c.Admission.Jobs["canary"] },
		"zero_generation": func(c *services.ExperimentNativeContract) {
			c.Admission.Jobs["canary"] = services.ExperimentNativeJobIdentity{ID: "native-canary"}
		},
		"long_lease": func(c *services.ExperimentNativeContract) { c.Admission.ExpiresAtUTC = "2026-09-22T00:00:00Z" },
		"non_utc":    func(c *services.ExperimentNativeContract) { c.Admission.NotBeforeUTC = "2026-09-19T08:00:00-03:00" },
	} {
		t.Run(name, func(t *testing.T) {
			f := newNativeFixture(t, false)
			change(&f.contract)
			f.seal(t, false)
			_, err := services.LoadExperimentNativeAdmissionForRelease(f.experiment, f.root, "canary", f.release)
			if err == nil || strings.Contains(err.Error(), "integrity failed") {
				t.Fatalf("native contract guard was bypassed: %v", err)
			}
			noOutput(t, f)
		})
	}
}

func TestNativePinnedAdmissionRejectsTamperingQualificationAndUnsafeFiles(t *testing.T) {
	f := newNativeFixture(t, true)
	if _, err := services.LoadExperimentNativeAdmissionForRelease(f.experiment, f.root, "train", f.release); err == nil {
		t.Fatal("train admitted without canary proof")
	}
	f.cpu.ForwardVJP = false
	f.seal(t, false)
	if _, err := services.LoadExperimentNativeAdmissionForRelease(f.experiment, f.root, "canary", f.release); err == nil || !strings.Contains(err.Error(), "CPU") {
		t.Fatalf("unqualified CPU admitted: %v", err)
	}
	f.cpu.ForwardVJP = true
	f.seal(t, false)
	write(t, filepath.Join(f.bundle, "runtime", "tayi-native"), []byte("tampered"), 0700)
	if _, err := services.LoadExperimentNativeAdmissionForRelease(f.experiment, f.root, "canary", f.release); err == nil {
		t.Fatal("tampered runtime admitted")
	}
	write(t, filepath.Join(f.bundle, "runtime", "tayi-native"), []byte("CPU test executable; never executed"), 0700)
	write(t, filepath.Join(f.bundle, "unmanifested.py"), []byte("never executed"), 0600)
	if _, err := services.LoadExperimentNativeAdmissionForRelease(f.experiment, f.root, "canary", f.release); err == nil {
		t.Fatal("unmanifested runtime admitted")
	}
	if err := os.Remove(filepath.Join(f.bundle, "unmanifested.py")); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(f.library, f.library+".real"); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(f.library+".real", f.library); err != nil {
		t.Fatal(err)
	}
	if _, err := services.FileHash(f.library, 1<<20, false); err == nil {
		t.Fatalf("symlink library admitted: %v", err)
	}
	noOutput(t, f)
}
