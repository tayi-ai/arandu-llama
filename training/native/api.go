package native

import (
	"encoding/json"
	"errors"
	"reflect"
	"time"
)

// Config returns an owned snapshot of the admitted installation recipe.
func (e *NativeExperiment) Config() NativeExperimentConfig {
	if e == nil {
		return NativeExperimentConfig{}
	}
	body, _ := json.Marshal(e.config)
	var result NativeExperimentConfig
	_ = json.Unmarshal(body, &result)
	return result
}

// SourceIdentity reports the explicit document supplied to LoadNativeExperiment.
// In-memory configurations have no source path and cannot be passed to a child
// process by pretending that an environment-loaded document exists.
func (e *NativeExperiment) SourceIdentity() (path, digest string) {
	if e == nil {
		return "", ""
	}
	return e.sourcePath, e.sourceSHA256
}

// Bundle is a detached snapshot of a validated native artifact bundle.
type Bundle struct {
	Manifest        ExperimentNativeManifest
	Contract        ExperimentNativeContract
	ContractBytes   []byte
	CanaryCompleted time.Time
	CanaryProof     ExperimentNativeCanaryProof
}

// LoadBundle applies the native artifact, provenance and qualification gates.
// It does not authorize or start work; the application owns those decisions.
func LoadBundle(experiment *NativeExperiment, root, mode string, release ExperimentNativeRelease) (Bundle, error) {
	value, err := experimentLoadNativeBundle(experiment, root, mode, release)
	if err != nil {
		return Bundle{}, err
	}
	return Bundle{Manifest: value.manifest, Contract: value.contract, ContractBytes: value.bytes, CanaryCompleted: value.canaryCompleted, CanaryProof: value.canaryProof}, nil
}

// AdmissionForContract validates a contract's numerical identity and lease.
// The contract and bytes must come from LoadBundle, never an untrusted request.
func AdmissionForContract(experiment *NativeExperiment, contract ExperimentNativeContract, body []byte, mode string, release ExperimentNativeRelease) (Admission, error) {
	var decoded ExperimentNativeContract
	if experimentDecode(body, &decoded, true) != nil || !reflect.DeepEqual(decoded, contract) {
		return Admission{}, errors.New("experiment.train: contract bytes differ from the supplied identity")
	}
	return experimentNativeAdmission(experiment, contract, body, mode, release)
}

// FileHash validates a bounded regular artifact and returns its content digest.
func FileHash(path string, limit int64, executable bool) (string, error) {
	return experimentNativeFileHash(path, limit, executable)
}
