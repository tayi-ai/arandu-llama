package native

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"io/fs"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"
)

// ExperimentNativeContractVersion separates the native contract from the disabled
// Python runtime contract while retaining the fleet action experiment.train.
const ExperimentNativeContractVersion = "tayi.experiment.native.v3"

// ExperimentNativeNodeIDs returns the versioned compute topology in rank order.
// Membership and exclusions come only from the deployment's admitted recipe.
// Each caller owns its copy and cannot change subsequent admissions.
func ExperimentNativeNodeIDs(experiment *NativeExperiment) []string {
	if experiment == nil {
		return nil
	}
	return slices.Clone(experiment.config.Recipe.Distribution.Nodes[:])
}

// ExperimentNativeBundleName is the only native bundle admitted below ArtifactRoot.

// The native canary release has passed CPU qualification. Training additionally
// requires the complete admitted CUDA canary proof in a newly reviewed release.

// Loading, calibration and four complete updates share one bounded qualification window.
const experimentNativeCanarySeconds = 1800

// ExperimentNativeRelease is a trusted bootstrap input, never a job/request field.
// Empty or malformed identity keeps native admission closed.
type ExperimentNativeRelease struct{ ManifestSHA256 string }

// ExperimentNativeManifest pins native code, its complete reviewed library inventory,
// scientific inputs and qualification documents. Library keys are absolute
// regular-file paths; Files keys belong to the fixed relative bundle allowlist.
type ExperimentNativeManifest struct {
	SchemaVersion       int               `json:"schema_version"`
	ContractVersion     string            `json:"contract_version"`
	ModelRepo           string            `json:"model_repo"`
	ModelRevision       string            `json:"model_revision"`
	ModelManifestSHA256 string            `json:"model_manifest_sha256"`
	TrainingSHA256      string            `json:"training_sha256"`
	InitialDigest       string            `json:"initial_digest"`
	RuntimeSHA256       string            `json:"runtime_sha256"`
	LibTorchVersion     string            `json:"libtorch_version"`
	Files               map[string]string `json:"files"`
	Libraries           map[string]string `json:"libraries"`
}

// ExperimentNativeJobIdentity is the same immutable identity carried by fleet jobs.
type ExperimentNativeJobIdentity = experimentJobIdentity

// ExperimentNativeAdmission retains the existing bounded fleet geometry and lease.
// SchemaVersion is 2 and ContractVersion is ExperimentNativeContractVersion.
type ExperimentNativeAdmission experimentAdmission

// ExperimentNativeContract pins the existing model cache and authoritative rank order.
// Scientific numerical behavior remains owned by the fixed native recipe.
type ExperimentNativeContract struct {
	ConfigurationSHA256 string                    `json:"configuration_sha256"`
	Admission           ExperimentNativeAdmission `json:"admission"`
	Nodes               []string                  `json:"nodes"`
	BasePath            string                    `json:"base_path"`
}

// ExperimentNativeEvidence binds a reviewed qualification to code, libraries and all
// scientific bundle files, excluding qualification documents themselves.
type ExperimentNativeEvidence struct {
	RuntimeSHA256 string            `json:"runtime_sha256"`
	Libraries     map[string]string `json:"libraries"`
	Inputs        map[string]string `json:"inputs"`
}

// ExperimentNativeCPUQualification is a pinned rollup of independent CPU numerical
// gates. It does not attest GPU memory, CUDA execution or fleet completion.
type ExperimentNativeCPUQualification struct {
	SchemaVersion int                      `json:"schema_version"`
	Evidence      ExperimentNativeEvidence `json:"evidence"`
	Passed        bool                     `json:"passed"`
	ForwardVJP    bool                     `json:"forward_vjp"`
	InitialDigest string                   `json:"initial_digest"`
	TestsSHA256   string                   `json:"tests_sha256"`
}

// ExperimentNativeCanaryNode identifies one successful native canary result. The
// result digest binds the measured per-node output reviewed by the coordinator.
type ExperimentNativeCanaryNode struct {
	Node                        string `json:"node"`
	Rank                        int    `json:"rank"`
	State                       string `json:"state"`
	Exit                        *int   `json:"exit"`
	GPUAdmission                bool   `json:"gpu_admission"`
	ResultSHA256                string `json:"result_sha256"`
	TelemetrySHA256             string `json:"telemetry_sha256"`
	CalibrationExamples         int    `json:"calibration_examples"`
	CalibrationReferenceSHA256  string `json:"calibration_reference_sha256"`
	ReferenceProbabilitiesMatch bool   `json:"reference_probabilities_match"`
}

// ExperimentNativeCanaryProof admits train only after all admitted canary identities
// succeeded against the same binary, libraries, contract and scientific inputs.
// Adding this proof requires a newly reviewed manifest pin; it cannot hash the
// manifest that contains itself.
type ExperimentNativeCanaryProof struct {
	SchemaVersion  int                          `json:"schema_version"`
	Evidence       ExperimentNativeEvidence     `json:"evidence"`
	Job            ExperimentNativeJobIdentity  `json:"job"`
	CompletedAtUTC string                       `json:"completed_at_utc"`
	Nodes          []ExperimentNativeCanaryNode `json:"nodes"`
}

type experimentNativeBundle struct {
	manifest        ExperimentNativeManifest
	contract        ExperimentNativeContract
	bytes           []byte
	canaryCompleted time.Time
	canaryProof     ExperimentNativeCanaryProof
}

func experimentNativeInputFiles() []string {
	return []string{"contract.json", "training.json", "base-manifest.json", "config.json", "model.safetensors.index.json",
		"tokenizer.json", "initial-reference.json", "calibration-reference.json", "tokenization-proof.json"}
}

// LoadExperimentNativeAdmission uses only the reviewed compiled-in release pin.
// It is deliberately closed while qualification and production wiring remain
// incomplete. Loading never creates a directory or launches a process.
func LoadExperimentNativeAdmission(experiment *NativeExperiment, artifactRoot, mode string) (Admission, error) {
	if experiment == nil {
		return Admission{}, errors.New("native experiment: installation is not configured")
	}

	return LoadExperimentNativeAdmissionForRelease(experiment, artifactRoot, mode, ExperimentNativeRelease{ManifestSHA256: experiment.config.ManifestSHA256})
}

// LoadExperimentNativeAdmissionForRelease accepts an explicit bootstrap trust
// identity from bootstrap wiring or the supervisor's explicit child environment,
// never from job/request data. Expired contracts remain readable for status/cancellation;
// ValidateStart is required before dispatch and repeated by the node builder.
func LoadExperimentNativeAdmissionForRelease(experiment *NativeExperiment, artifactRoot, mode string, release ExperimentNativeRelease) (Admission, error) {
	if experiment == nil {
		return Admission{}, errors.New("native experiment: installation is not configured")
	}

	bundle, err := experimentLoadNativeBundle(experiment, artifactRoot, mode, release)
	if err != nil {
		return Admission{}, err
	}
	return experimentNativeAdmission(experiment, bundle.contract, bundle.bytes, mode, release)
}

// ValidateExperimentNativeCanaryResults binds the original collected canary results
// to the proof in the currently trusted train release. Adding the proof changes
// the manifest hash, so originalJob retains its original RuntimeDigest. Every
// scientific input, executable and evidence-file digest must still match.
func ValidateExperimentNativeCanaryResults(experiment *NativeExperiment, artifactRoot string, originalJob Job, results []ExperimentNativeResultEnvelope) error {
	if experiment == nil {
		return errors.New("native experiment: installation is not configured")
	}

	return ValidateExperimentNativeCanaryResultsForRelease(experiment, artifactRoot, originalJob, results, ExperimentNativeRelease{ManifestSHA256: experiment.config.ManifestSHA256})
}

// ValidateExperimentNativeCanaryResultsForRelease is the explicit bootstrap variant.
// The release comes from trusted wiring, never a job request. Callers must first
// verify each Fleet terminal state, log digest and persisted receipt identity;
// the supplied DTOs do not carry Fleet's transport or process attestations.
// This function is read-only and never reinterprets a Fleet JSON digest as the
// digest of a remote result.json or resources.jsonl file.
func ValidateExperimentNativeCanaryResultsForRelease(experiment *NativeExperiment, artifactRoot string, originalJob Job, results []ExperimentNativeResultEnvelope, release ExperimentNativeRelease) error {
	if experiment == nil {
		return errors.New("native experiment: installation is not configured")
	}

	bundle, err := experimentLoadNativeBundle(experiment, artifactRoot, "train", release)
	if err != nil {
		return err
	}
	fail := func() error {
		return errors.New("experiment.train: collected canary results differ from the pinned proof")
	}
	if !experimentNativeResultHash(originalJob.RuntimeDigest) || len(results) != len(bundle.contract.Nodes) {
		return fail()
	}
	prior, err := experimentNativeAdmission(experiment, bundle.contract, bundle.bytes, "canary", ExperimentNativeRelease{ManifestSHA256: originalJob.RuntimeDigest})
	if err != nil {
		return err
	}
	expected := prior.Job
	if originalJob.ID != expected.ID || originalJob.Generation != expected.Generation || originalJob.Action != expected.Action ||
		originalJob.ContractVersion != expected.ContractVersion || originalJob.ModelRecipe != expected.ModelRecipe ||
		originalJob.ModelDigest != expected.ModelDigest || !bytes.Equal(originalJob.Request, expected.Request) {
		return fail()
	}
	finalDigest := ""
	for rank, outcome := range results {
		body, err := json.Marshal(outcome)
		if err != nil {
			return fail()
		}
		if _, err := ParseExperimentNativeResult(experiment, string(body), originalJob, prior.Nodes[rank], rank, "canary"); err != nil {
			return fail()
		}
		proof := bundle.canaryProof.Nodes[rank]
		if outcome.RuntimeSHA256 != bundle.canaryProof.Evidence.RuntimeSHA256 || outcome.ContractSHA256 != bundle.manifest.Files["contract.json"] ||
			outcome.Node != proof.Node || outcome.Rank != proof.Rank || outcome.GPUAdmission != proof.GPUAdmission ||
			outcome.ResultSHA256 != proof.ResultSHA256 || outcome.TelemetrySHA256 != proof.TelemetrySHA256 ||
			outcome.CalibrationExamples != proof.CalibrationExamples || outcome.CalibrationReferenceSHA256 != proof.CalibrationReferenceSHA256 ||
			outcome.ReferenceProbabilitiesMatch != proof.ReferenceProbabilitiesMatch ||
			finalDigest != "" && outcome.FinalAdapterSHA256 != finalDigest {
			return fail()
		}
		finalDigest = outcome.FinalAdapterSHA256
	}
	return nil
}

func experimentNativeAdmission(experiment *NativeExperiment, contract ExperimentNativeContract, body []byte, mode string, release ExperimentNativeRelease) (Admission, error) {
	if experiment == nil {
		return Admission{}, errors.New("native experiment: installation is not configured")
	}

	if mode != "canary" && mode != "train" {
		return Admission{}, errors.New("experiment.train: invalid native mode")
	}
	a := experimentAdmission(contract.Admission)
	if a.SchemaVersion != 2 || a.ContractVersion != ExperimentNativeContractVersion || contract.ConfigurationSHA256 != experiment.RecipeSHA256() || contract.BasePath !=
		experiment.config.BasePath {
		return Admission{}, errors.New("experiment.train: native contract version or model cache differs")
	}
	identity := a.Jobs[mode]
	request, _ := json.Marshal(experimentRequest{Mode: mode, ContractSHA256: experimentHash(body)})
	job := Job{ID: identity.ID, Action: "experiment.train", Generation: identity.Generation,
		ContractVersion: ExperimentNativeContractVersion, RuntimeDigest: release.ManifestSHA256,
		ModelRecipe: experiment.config.ModelRecipe, ModelDigest: experiment.config.Recipe.Model.ManifestSHA256, Request: request}
	if err := experimentValidateNativeIdentity(experiment, a, job, mode, contract.Nodes); err != nil {
		return Admission{}, err
	}
	before, expires, err := a.window()
	if err != nil || expires.Sub(before) > 24*time.Hour {
		return Admission{}, errors.New("experiment.train: invalid native lease; maximum duration is 24 hours")
	}
	return Admission{Job: job, Nodes: slices.Clone(contract.Nodes), Deadline: time.Duration(a.DeadlinesSeconds[mode]) * time.Second,
		NotBefore: before, ExpiresAt: expires}, nil
}

func experimentValidateNativeIdentity(experiment *NativeExperiment, a experimentAdmission, job Job, mode string, nodes []string) error {
	if experiment == nil {
		return errors.New("native experiment: installation is not configured")
	}

	ids := ExperimentNativeNodeIDs(experiment)
	if a.SchemaVersion != 2 || a.ContractVersion != ExperimentNativeContractVersion || a.ModelManifestSHA256 !=
		experiment.config.Recipe.Model.ManifestSHA256 ||
		!slices.Equal(a.TopologyIDs, ids) || !slices.Equal(nodes, ids) ||
		a.NNodes != len(ids) || a.ProcessWorldSize != a.NNodes || a.GPUsPerNode != 2 || a.GlobalBatchSize != 21 ||
		a.OptimizerSteps != 18 || a.Examples != a.GlobalBatchSize*a.OptimizerSteps || a.Epochs != 1 || a.VRAMFraction != "0.80" {
		return errors.New("experiment.train: native admission topology or training geometry mismatch")
	}
	if len(a.DeadlinesSeconds) != 2 || a.DeadlinesSeconds["canary"] < 1 || a.DeadlinesSeconds["canary"] > experimentNativeCanarySeconds ||
		a.DeadlinesSeconds["train"] < 1 || a.DeadlinesSeconds["train"] > 5400 || len(a.Jobs) != 2 {
		return errors.New("experiment.train: native admission deadlines or job set mismatch")
	}
	for _, name := range []string{"canary", "train"} {
		identity := a.Jobs[name]
		if !experimentJobID.MatchString(identity.ID) || identity.Generation == 0 {
			return errors.New("experiment.train: invalid native admitted job identity")
		}
	}
	if a.Jobs["canary"].ID == a.Jobs["train"].ID || a.Jobs[mode].ID != job.ID || a.Jobs[mode].Generation != job.Generation {
		return errors.New("experiment.train: native job is not the admitted mode and generation")
	}
	return nil
}

func experimentLoadNativeBundle(experiment *NativeExperiment, root, mode string, release ExperimentNativeRelease) (experimentNativeBundle, error) {
	if experiment == nil {
		return experimentNativeBundle{}, errors.New("native experiment: installation is not configured")
	}

	var result experimentNativeBundle
	fail := func(reason string) (experimentNativeBundle, error) {
		return result, errors.New("experiment.train: " + reason)
	}
	if !experimentSHA256(release.ManifestSHA256) {
		return result, ErrExperimentNativeBackendUnqualified
	}
	if mode != "canary" && mode != "train" {
		return fail("invalid native mode")
	}
	if err := experimentDirectory(root); err != nil {
		return fail("unsafe native artifact root")
	}
	bundle := filepath.Join(root, experiment.config.BundleName)
	if err := experimentDirectory(bundle); err != nil {
		return fail("unsafe native bundle directory")
	}
	body, err := experimentReadFile(filepath.Join(bundle, "manifest.json"), 256<<10)
	if err != nil || experimentHash(body) != release.ManifestSHA256 || experimentDecode(body, &result.manifest, true) != nil {
		return fail("native manifest is unreadable or differs from the trusted release")
	}
	m := result.manifest
	if m.SchemaVersion != 2 || m.ContractVersion != ExperimentNativeContractVersion || m.ModelRepo !=
		experiment.config.Recipe.Model.Repository ||
		m.ModelRevision !=
			experiment.config.Recipe.Model.Revision ||
		m.ModelManifestSHA256 !=
			experiment.config.Recipe.Model.ManifestSHA256 ||
		m.TrainingSHA256 !=
			experiment.config.Recipe.Data.SHA256 ||
		m.InitialDigest !=
			experiment.config.Recipe.LoRA.ExpectedInitialDigest ||
		m.LibTorchVersion != "2.14.0" ||
		!experimentSHA256(m.RuntimeSHA256) || m.Files["runtime/tayi-native"] != m.RuntimeSHA256 ||
		m.Files["base-manifest.json"] != m.ModelManifestSHA256 || m.Files["training.json"] != m.TrainingSHA256 ||
		len(m.Libraries) == 0 || len(m.Libraries) > 64 {
		return fail("native manifest provenance, runtime or scientific identity differs")
	}
	for path, digest := range m.Libraries {
		if !experimentNativeLibraryPath(path) || !experimentSHA256(digest) {
			return fail("native library inventory is invalid")
		}
	}
	files := append(experimentNativeInputFiles(), "runtime/tayi-native", "cpu-qualification.json")
	if _, exists := m.Files["canary-proof.json"]; exists {
		files = append(files, "canary-proof.json")
	} else if mode == "train" {
		return fail("native train requires a pinned complete-topology canary proof")
	}
	if len(m.Files) != len(files) {
		return fail("native manifest contains an unexpected file set")
	}
	seen := 0
	err = filepath.WalkDir(bundle, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		relative, err := filepath.Rel(bundle, path)
		if err != nil {
			return err
		}
		if entry.IsDir() && (relative == "." || relative == "runtime") {
			return nil
		}
		if !entry.Type().IsRegular() || relative != "manifest.json" && !slices.Contains(files, relative) {
			return errors.New("unmanifested or unsafe native bundle entry")
		}
		seen++
		return nil
	})
	if err != nil || seen != len(files)+1 {
		return fail("native bundle contains missing, unmanifested or unsafe entries")
	}
	result.bytes, err = experimentReadFile(filepath.Join(bundle, "contract.json"), 1<<20)
	if err != nil || experimentHash(result.bytes) != m.Files["contract.json"] || experimentDecode(result.bytes, &result.contract, true) != nil {
		return fail("invalid native contract")
	}
	admission, err := experimentNativeAdmission(experiment, result.contract, result.bytes, mode, release)
	if err != nil {
		return result, err
	}
	for _, name := range files {
		expected := m.Files[name]
		if !experimentSHA256(expected) {
			return fail("native bundle file hash is missing or invalid")
		}
		limit := int64(32 << 20)
		if name == "runtime/tayi-native" {
			limit = 512 << 20
		}
		digest, err := experimentNativeFileHash(filepath.Join(bundle, name), limit, name == "runtime/tayi-native")
		if err != nil || digest != expected {
			return fail("native bundle file integrity failed: " + name)
		}
	}
	var source struct {
		Files []struct {
			Path   string `json:"path"`
			SHA256 string `json:"sha256"`
		} `json:"files"`
	}
	body, err = experimentReadFile(filepath.Join(bundle, "base-manifest.json"), 32<<20)
	if err != nil || experimentDecode(body, &source, false) != nil {
		return fail("invalid frozen native model inventory")
	}
	for _, name := range []string{"config.json", "model.safetensors.index.json", "tokenizer.json"} {
		found := false
		for _, file := range source.Files {
			if file.Path == name && file.SHA256 == m.Files[name] {
				found = true
			}
		}
		if !found {
			return fail("native model input differs from the frozen source inventory")
		}
	}
	var cpu ExperimentNativeCPUQualification
	body, err = experimentReadFile(filepath.Join(bundle, "cpu-qualification.json"), 1<<20)
	if err != nil || experimentDecode(body, &cpu, true) != nil || cpu.SchemaVersion != 1 || !cpu.Passed || !cpu.ForwardVJP ||
		cpu.InitialDigest !=
			experiment.config.Recipe.LoRA.ExpectedInitialDigest ||
		!experimentSHA256(cpu.TestsSHA256) || !experimentNativeEvidenceMatches(cpu.Evidence, m) {
		return fail("native CPU forward/backward qualification is absent or mismatched")
	}
	var tokenization struct {
		AllIDsMatch        bool              `json:"all_ids_match"`
		Examples           int               `json:"examples"`
		Mismatches         []json.RawMessage `json:"mismatches"`
		ModelWeightsLoaded bool              `json:"model_weights_loaded"`
		Oracle             string            `json:"oracle"`
		PythonExecuted     bool              `json:"python_executed"`
		Tokens             int               `json:"tokens"`
		Version            string            `json:"version"`
	}
	body, err = experimentReadFile(filepath.Join(bundle, "tokenization-proof.json"), 1<<20)
	if err != nil || experimentDecode(body, &tokenization, true) != nil || !tokenization.AllIDsMatch || tokenization.Examples != 378 ||
		len(tokenization.Mismatches) != 0 || tokenization.Tokens <= 0 || tokenization.Oracle != "huggingface/tokenizers" ||
		tokenization.Version != "0.22.2" || tokenization.PythonExecuted || tokenization.ModelWeightsLoaded {
		return fail("native full-corpus tokenization qualification is absent or mismatched")
	}
	if mode == "train" {
		var proof ExperimentNativeCanaryProof
		body, err = experimentReadFile(filepath.Join(bundle, "canary-proof.json"), 1<<20)
		if err != nil || experimentDecode(body, &proof, true) != nil || proof.SchemaVersion != 1 || !experimentNativeEvidenceMatches(proof.Evidence, m) ||
			proof.Job != result.contract.Admission.Jobs["canary"] || len(proof.Nodes) != len(admission.Nodes) {
			return fail("native canary proof identity or node count differs")
		}
		completed, err := time.Parse(time.RFC3339, proof.CompletedAtUTC)
		_, offset := completed.Zone()
		if err != nil || offset != 0 || completed.Before(admission.NotBefore) || !completed.Before(admission.ExpiresAt) {
			return fail("native canary proof falls outside the admission lease")
		}
		result.canaryCompleted = completed
		for rank, node := range proof.Nodes {
			if node.Node != admission.Nodes[rank] || node.Rank != rank || node.State != "succeeded" || node.Exit == nil ||
				*node.Exit != 0 || !node.GPUAdmission || !experimentSHA256(node.ResultSHA256) || !experimentSHA256(node.TelemetrySHA256) ||
				node.CalibrationExamples != 8 || node.CalibrationReferenceSHA256 != m.Files["calibration-reference.json"] || !node.ReferenceProbabilitiesMatch {
				return fail("native train requires every admitted canary to pass GPU admission, eight reference probabilities and exit zero with telemetry")
			}
		}
		result.canaryProof = proof
	}
	return result, nil
}

func experimentNativeEvidenceMatches(evidence ExperimentNativeEvidence, manifest ExperimentNativeManifest) bool {
	if evidence.RuntimeSHA256 != manifest.RuntimeSHA256 || !maps.Equal(evidence.Libraries, manifest.Libraries) || len(evidence.Inputs) != len(experimentNativeInputFiles()) {
		return false
	}
	for _, name := range experimentNativeInputFiles() {
		if evidence.Inputs[name] != manifest.Files[name] {
			return false
		}
	}
	return true
}

func experimentNativeLibraryPath(path string) bool {
	return filepath.IsAbs(path) && filepath.Clean(path) == path && !strings.ContainsAny(path, "\x00\r\n:") &&
		(strings.HasSuffix(filepath.Base(path), ".so") || strings.Contains(filepath.Base(path), ".so."))
}

func experimentNativeFileHash(path string, limit int64, executable bool) (string, error) {
	if err := experimentDirectory(filepath.Dir(path)); err != nil {
		return "", err
	}
	before, err := os.Lstat(path)
	if err != nil || !before.Mode().IsRegular() || before.Size() < 1 || before.Size() > limit || executable && before.Mode().Perm()&0111 == 0 {
		return "", errors.New("unsafe native file or executable")
	}
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	opened, err := file.Stat()
	if err != nil || !os.SameFile(before, opened) {
		return "", errors.New("native file changed while opening")
	}
	digest := sha256.New()
	n, err := io.CopyBuffer(digest, io.LimitReader(file, limit+1), make([]byte, 64<<10))
	after, statErr := file.Stat()
	if err != nil || statErr != nil || n != before.Size() || n > limit || after.Size() != before.Size() || !after.ModTime().Equal(before.ModTime()) {
		return "", errors.New("native file changed or could not be read completely")
	}
	return hex.EncodeToString(digest.Sum(nil)), nil
}
