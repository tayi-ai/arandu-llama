package pipeline

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"reflect"
	"slices"

	"github.com/tayi-ai/arandu-llama/checkpoint"
	"github.com/tayi-ai/arandu-llama/training/protection"
)

// ErrConstrained reports a refused constrained-stage contract or checkpoint.
var ErrConstrained = errors.New("pipeline: constrained stage refused")

// ErrConstrainedAttempt reports an intent without an accepted checkpoint.
// Rejected or interrupted attempts require explicit review and a new admitted
// execution; this handler never automatically repeats numerical attempts.
var ErrConstrainedAttempt = errors.New("pipeline: constrained attempt requires review")

// ConstrainedTensor fixes the canonical parameter order, names and shapes.
type ConstrainedTensor struct {
	Name  string   `json:"name"`
	Shape []uint64 `json:"shape"`
}

// ConstrainedSource pins a signal or qualification artifact. Exactly one origin
// is required: Input must occur verbatim in Stage.Inputs, or StageID and
// ReceiptSHA256 identify a completed predecessor containing Artifact verbatim.
// External input artifacts are resolved under ConstrainedConfig.SourceDirectory.
// Predecessor artifacts are resolved under StageContext.ArtifactDirectory.
type ConstrainedSource struct {
	Artifact      StageArtifact `json:"artifact"`
	Input         ArtifactRef   `json:"input"`
	StageID       string        `json:"stage_id"`
	ReceiptSHA256 string        `json:"receipt_sha256"`
}

// ConstrainedUpdate fixes one objective's immutable sources and solver policy.
// Their scientific interpretation belongs to the explicitly qualified Factory.
type ConstrainedUpdate struct {
	Config  StepConfig          `json:"config"`
	Sources []ConstrainedSource `json:"sources"`
}

// ConstrainedLimits bounds configuration, parameter geometry and artifact bytes.
// MaxTotalBytes reserves MaxCheckpointBytes + 2*MaxReceiptBytes per admitted step
// for parameters, evidence and intent. Each update admits exactly one attempt.
// Model/solver resident memory is independently admitted by Placement and Factory.
type ConstrainedLimits struct {
	MaxProtocolBytes   int64             `json:"max_protocol_bytes"`
	MaxReceiptBytes    int64             `json:"max_receipt_bytes"`
	MaxCheckpointBytes int64             `json:"max_checkpoint_bytes"`
	MaxTotalBytes      int64             `json:"max_total_bytes"`
	MaxSourceBytes     int64             `json:"max_source_bytes"`
	MaxParameters      int               `json:"max_parameters"`
	MaxSteps           int               `json:"max_steps"`
	Checkpoint         checkpoint.Limits `json:"checkpoint"`
}

// ConstrainedProtocol freezes all numerical choices without scientific defaults.
// InitialParametersSHA256 and subsequent parameter hashes use ordered UTF-8
// names followed by each tensor's little-endian FP32 values, as decoder does.
type ConstrainedProtocol struct {
	Version                    int                 `json:"version"`
	RecipeSHA256               string              `json:"recipe_sha256"`
	PlacementSHA256            string              `json:"placement_sha256"`
	FactoryQualificationSHA256 string              `json:"factory_qualification_sha256"`
	StageID                    string              `json:"stage_id"`
	InitialParametersSHA256    string              `json:"initial_parameters_sha256"`
	Layout                     []ConstrainedTensor `json:"layout"`
	Updates                    []ConstrainedUpdate `json:"updates"`
	Limits                     ConstrainedLimits   `json:"limits"`
}

// Digest returns the SHA-256 of the bounded canonical JSON encoding, including
// its final newline. It validates resource bounds before encoding the protocol.
func (p ConstrainedProtocol) Digest() (string, error) {
	if _, err := p.validate(); err != nil {
		return "", err
	}
	body, err := constrainedJSON(p, p.Limits.MaxProtocolBytes)
	if err != nil {
		return "", err
	}
	return constrainedSHA(body), nil
}

// ConstrainedSourceFile carries a verified origin and the private resolved path.
// Factory must consume the pinned bytes, checking SHA while reading; a path is
// not a snapshot. Sources must remain immutable for this stage's lifetime.
type ConstrainedSourceFile struct {
	Source ConstrainedSource
	Path   string
}

// ConstrainedRequest identifies one serialized model session. Parameters is nil
// only for the first update; Factory then loads the externally pinned initial
// adapter. Later requests contain a private copy of the verified prior vector.
// Factory must build the exact objective from Sources and release every owned
// resource in Close. It must not detach compute or modify scientific policy.
type ConstrainedRequest struct {
	Context    StageContext
	Step       int
	Parameters []float32
	Sources    []ConstrainedSourceFile
}

// ConstrainedSession owns one concrete Model and all resources backing it.
type ConstrainedSession struct {
	Model Model
	Close func() error
}

// ConstrainedFactory loads a qualified model and binds the exact step objective.
// Byte verification does not qualify this callback's numerical implementation.
type ConstrainedFactory func(context.Context, ConstrainedRequest) (ConstrainedSession, error)

// ConstrainedConfig explicitly installs one exact stage, recipe and placement.
// Only the Factory owns backend loading; no model discovery or optimizer exists.
type ConstrainedConfig struct {
	Recipe          Recipe
	Placement       Placement
	Protocol        ConstrainedProtocol
	ProtocolSHA256  string
	SourceDirectory string
	Factory         ConstrainedFactory
}

// ConstrainedStage executes Step and durably publishes accepted candidates.
// Run and Reconcile share a cancelable in-process gate. DurableRuntime owns the
// cross-process execution lock; direct callers must provide the equivalent lock.
// Verify checks recorded evidence and artifacts, without rerunning model compute.
type ConstrainedStage struct {
	config     ConstrainedConfig
	stage      Stage
	parameters int
	gate       chan struct{}
}

var _ StageHandler = (*ConstrainedStage)(nil)

// NewConstrainedStage snapshots an explicitly hashed protocol and exact wiring.
// No files are created and no model is loaded during construction or admission.
func NewConstrainedStage(c ConstrainedConfig) (*ConstrainedStage, error) {
	n, err := c.Protocol.validate()
	if err != nil || c.Factory == nil || !digest(c.ProtocolSHA256) || !filepath.IsAbs(c.SourceDirectory) {
		return nil, errors.Join(ErrConstrained, err)
	}
	body, err := constrainedJSON(c.Protocol, c.Protocol.Limits.MaxProtocolBytes)
	if err != nil || constrainedSHA(body) != c.ProtocolSHA256 {
		return nil, errors.Join(ErrConstrained, err)
	}
	var copied ConstrainedProtocol
	if err := json.Unmarshal(body, &copied); err != nil {
		return nil, err
	}
	c.Protocol, c.Recipe, c.Placement = copied, cloneRecipe(c.Recipe), clonePlacement(c.Placement)
	c.SourceDirectory = filepath.Clean(c.SourceDirectory)
	h := &ConstrainedStage{config: c, parameters: n, gate: make(chan struct{}, 1)}
	for _, stage := range c.Recipe.Stages {
		if stage.ID == copied.StageID {
			h.stage = cloneStage(stage)
		}
	}
	if err := h.Admit(context.Background(), c.Recipe, h.stage, c.Placement); err != nil {
		return nil, err
	}
	return h, nil
}

// Admit verifies exact recipe, placement, phase, data roles and source origins.
func (h *ConstrainedStage) Admit(ctx context.Context, recipe Recipe, stage Stage, placement Placement) error {
	if h == nil || ctx == nil {
		return ErrConstrained
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	p := h.config.Protocol
	r, err := recipe.Digest()
	if err != nil || r != p.RecipeSHA256 || jsonSHA256(placement) != p.PlacementSHA256 ||
		!reflect.DeepEqual(recipe, h.config.Recipe) || !reflect.DeepEqual(placement, h.config.Placement) || !reflect.DeepEqual(stage, h.stage) ||
		(stage.Phase != PhaseFusion && stage.Phase != PhaseRecovery && stage.Phase != PhaseVariantRecovery) ||
		len(p.Updates) != stage.MaxSteps || p.FactoryQualificationSHA256 != placement.QualificationSHA256 ||
		!digest(placement.RuntimeSHA256) || !identifier(placement.Backend) || placement.MemoryBytes <= 0 || stage.MaxTokens > placement.MaxTokens {
		return errors.Join(ErrConstrained, err)
	}
	for _, update := range p.Updates {
		for _, pair := range update.Config.Protection {
			if pair.Positive.DatasetDigest != recipe.Protection.SHA256 || len(pair.Positive.Tokens) > stage.MaxTokens || len(pair.Negative.Tokens) > stage.MaxTokens {
				return ErrConstrained
			}
		}
		for _, source := range update.Sources {
			if source.StageID == "" {
				if !slices.Contains(stage.Inputs, source.Input) || source.Input.SHA256 != source.Artifact.SHA256 {
					return ErrConstrained
				}
			} else {
				found := false
				for _, earlier := range recipe.Stages {
					if earlier.ID == stage.ID {
						break
					}
					if earlier.ID == source.StageID {
						found = true
					}
				}
				if !found {
					return ErrConstrained
				}
			}
		}
	}
	return nil
}

func (p ConstrainedProtocol) validate() (int, error) {
	l := p.Limits
	c := l.Checkpoint
	if p.Version != 1 || !digest(p.RecipeSHA256) || !digest(p.PlacementSHA256) || !digest(p.FactoryQualificationSHA256) || !identifier(p.StageID) || !digest(p.InitialParametersSHA256) ||
		l.MaxProtocolBytes < 1 || l.MaxProtocolBytes > 64<<20 || l.MaxReceiptBytes < 1 || l.MaxReceiptBytes > 256<<20 ||
		l.MaxCheckpointBytes < 1 || l.MaxCheckpointBytes > 1<<40 || l.MaxSourceBytes < 1 || l.MaxSourceBytes > 1<<40 ||
		l.MaxParameters < 1 || l.MaxParameters > 1<<20 || l.MaxSteps < 1 || l.MaxSteps > 1_000_000 || len(p.Updates) < 1 || len(p.Updates) > l.MaxSteps ||
		c.MaxHeaderBytes < 1 || c.MaxHeaderBytes > 100_000_000 || c.MaxTensors < 1 || c.MaxTensors > 65536 || c.MaxDimensions < 1 || c.MaxDimensions > 32 ||
		c.MaxMetadataEntries < 1 || c.MaxChunkBytes < 4 || c.MaxChunkBytes > 4<<20 || len(p.Layout) < 1 || len(p.Layout) > c.MaxTensors ||
		l.MaxTotalBytes < l.MaxCheckpointBytes+2*l.MaxReceiptBytes || int64(len(p.Updates)) > l.MaxTotalBytes/(l.MaxCheckpointBytes+2*l.MaxReceiptBytes) {
		return 0, ErrConstrained
	}
	n := 0
	seen := map[string]bool{}
	for _, tensor := range p.Layout {
		if tensor.Name == "" || len(tensor.Name) > 1024 || seen[tensor.Name] || len(tensor.Shape) < 1 || len(tensor.Shape) > c.MaxDimensions {
			return 0, ErrConstrained
		}
		seen[tensor.Name] = true
		size := uint64(1)
		for _, d := range tensor.Shape {
			if d == 0 || d > uint64(l.MaxParameters)/size {
				return 0, ErrConstrained
			}
			size *= d
		}
		if int(size) > l.MaxParameters-n {
			return 0, ErrConstrained
		}
		n += int(size)
	}
	if int64(n)*4 > l.MaxCheckpointBytes {
		return 0, ErrConstrained
	}
	for _, update := range p.Updates {
		x := update.Config
		if !finite(x.Lambda) || x.Lambda <= 0 || !finite(x.MinimumGain) || x.MinimumGain < 0 ||
			len(x.Protection) < 1 || protection.ValidateBounds(x.Solver, n, len(x.Protection)) != nil ||
			len(update.Sources) < 1 || len(update.Sources) > 64 {
			return 0, ErrConstrained
		}
		ids := map[string]bool{}
		for _, pair := range x.Protection {
			if ValidateProtectedPair(pair) != nil || ids[pair.ID] {
				return 0, ErrConstrained
			}
			ids[pair.ID] = true
		}
		var total int64
		paths := map[string]bool{}
		for _, source := range update.Sources {
			a := source.Artifact
			if !constrainedArtifact(a, l.MaxSourceBytes) || a.Bytes > l.MaxSourceBytes-total {
				return 0, ErrConstrained
			}
			total += a.Bytes
			if source.StageID == "" {
				if source.ReceiptSHA256 != "" || !identifier(source.Input.ID) || !digest(source.Input.SHA256) {
					return 0, ErrConstrained
				}
			} else if !identifier(source.StageID) || !digest(source.ReceiptSHA256) || source.Input != (ArtifactRef{}) {
				return 0, ErrConstrained
			}
			key := source.StageID + "/" + a.Path
			if paths[key] {
				return 0, ErrConstrained
			}
			paths[key] = true
		}
	}
	return n, nil
}

func constrainedArtifact(a StageArtifact, limit int64) bool {
	return filepath.IsLocal(a.Path) && a.Path != "." && filepath.Clean(a.Path) == a.Path && len(a.Path) <= 1024 && !bytes.ContainsRune([]byte(a.Path), 0) && digest(a.SHA256) && a.Bytes > 0 && a.Bytes <= limit
}

type constrainedBuffer struct {
	bytes.Buffer
	limit int64
}

func (b *constrainedBuffer) Write(v []byte) (int, error) {
	if int64(len(v)) > b.limit-int64(b.Len()) {
		return 0, ErrConstrained
	}
	return b.Buffer.Write(v)
}
func constrainedJSON(v any, limit int64) ([]byte, error) {
	b := &constrainedBuffer{limit: limit}
	if err := json.NewEncoder(b).Encode(v); err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}
func constrainedSHA(body []byte) string {
	sum := sha256.Sum256(body)
	return hex.EncodeToString(sum[:])
}

func (h *ConstrainedStage) parameterSHA(values []float32) (string, error) {
	if len(values) != h.parameters {
		return "", ErrConstrained
	}
	digest := sha256.New()
	var scalar [4]byte
	offset := 0
	for _, tensor := range h.config.Protocol.Layout {
		_, _ = digest.Write([]byte(tensor.Name))
		size := 1
		for _, d := range tensor.Shape {
			size *= int(d)
		}
		for _, v := range values[offset : offset+size] {
			if !finite(float64(v)) {
				return "", ErrConstrained
			}
			binary.LittleEndian.PutUint32(scalar[:], math.Float32bits(v))
			_, _ = digest.Write(scalar[:])
		}
		offset += size
	}
	return hex.EncodeToString(digest.Sum(nil)), nil
}

func (h *ConstrainedStage) enter(ctx context.Context) error {
	if h == nil || ctx == nil {
		return ErrConstrained
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	select {
	case h.gate <- struct{}{}:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (h *ConstrainedStage) stageIdentity(c StageContext) (ReceiptIdentity, error) {
	if err := h.Admit(context.Background(), c.Execution.Recipe, c.Stage, c.Execution.Placement); err != nil {
		return ReceiptIdentity{}, err
	}
	x := c.Execution
	if !identifier(x.RunID) || !identifier(x.TenantID) || x.Generation == 0 || !digest(x.TargetSHA256) || !filepath.IsAbs(c.ArtifactDirectory) {
		return ReceiptIdentity{}, ErrConstrained
	}
	if c.Stage.ParentStage != "" && (c.Parent == nil || c.Parent.StageID != c.Stage.ParentStage || !c.Parent.Result.Complete || !digest(c.Parent.SHA256)) {
		return ReceiptIdentity{}, fmt.Errorf("%w: completed parent required", ErrConstrained)
	}
	return ReceiptIdentity{RunID: x.RunID, TenantID: x.TenantID, RecipeSHA256: h.config.Protocol.RecipeSHA256, TargetSHA256: x.TargetSHA256, PlacementSHA256: h.config.Protocol.PlacementSHA256, Generation: x.Generation}, nil
}
