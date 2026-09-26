package native

import (
	"bytes"
	"encoding/json"
	"errors"
	"regexp"
	"time"
)

var experimentJobID = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,64}$`)

// ErrExperimentNativeBackendUnqualified rejects execution without an admitted backend.
var ErrExperimentNativeBackendUnqualified = errors.New("experiment.train: native Go backend is not qualified; the Python training runtime is disabled")

// Job is the immutable execution identity supplied by an authorized orchestrator.
// It contains no transport, process supervisor, tenant, or authorization behavior.
type Job struct {
	ID              string          `json:"id"`
	Action          string          `json:"action"`
	Generation      uint64          `json:"generation,omitempty"`
	ContractVersion string          `json:"contract_version,omitempty"`
	RuntimeDigest   string          `json:"runtime_digest,omitempty"`
	ModelRecipe     string          `json:"model_recipe,omitempty"`
	ModelDigest     string          `json:"model_digest,omitempty"`
	Request         json.RawMessage `json:"request,omitempty"`
}

func sameJob(a, b Job) bool {
	return a.ID == b.ID && a.Action == b.Action && a.Generation == b.Generation &&
		a.ContractVersion == b.ContractVersion && a.RuntimeDigest == b.RuntimeDigest &&
		a.ModelRecipe == b.ModelRecipe && a.ModelDigest == b.ModelDigest && bytes.Equal(a.Request, b.Request)
}

// Admission is a detached, validated execution identity and time window.
// The application must authorize its dispatch and call ValidateStart.
type Admission struct {
	Job       Job
	Nodes     []string
	Deadline  time.Duration
	NotBefore time.Time
	ExpiresAt time.Time
}

type experimentRequest struct {
	Mode           string `json:"mode"`
	ContractSHA256 string `json:"contract_sha256"`
}

type experimentJobIdentity struct {
	ID         string `json:"id"`
	Generation uint64 `json:"generation"`
}

type experimentAdmission struct {
	SchemaVersion       int                              `json:"schema_version"`
	ContractVersion     string                           `json:"contract_version"`
	ModelManifestSHA256 string                           `json:"model_manifest_sha256"`
	TopologyIDs         []string                         `json:"topology_ids"`
	NNodes              int                              `json:"nnodes"`
	ProcessWorldSize    int                              `json:"process_world_size"`
	GPUsPerNode         int                              `json:"gpus_per_node"`
	GlobalBatchSize     int                              `json:"global_batch_size"`
	OptimizerSteps      int                              `json:"optimizer_steps"`
	Examples            int                              `json:"examples"`
	Epochs              int                              `json:"epochs"`
	VRAMFraction        string                           `json:"vram_fraction"`
	DeadlinesSeconds    map[string]int                   `json:"deadlines_seconds"`
	Jobs                map[string]experimentJobIdentity `json:"jobs"`
	NotBeforeUTC        string                           `json:"not_before_utc"`
	ExpiresAtUTC        string                           `json:"expires_at_utc"`
}
