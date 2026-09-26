package pipeline

import (
	"context"
	"errors"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
)

var ErrBinding = errors.New("pipeline: artifact binding refused")

const (
	BindingDerivedInput = "derived_input"
	BindingPredecessor  = "predecessor"
)

// PredecessorOutput selects one exact output, without predicting its future
// digest. Schema names the consumer's frozen format contract; the concrete
// consumer must validate that format before using its numerical contents.
type PredecessorOutput struct {
	StageID  string `json:"stage_id"`
	Path     string `json:"path"`
	Schema   string `json:"schema"`
	MaxBytes int64  `json:"max_bytes"`
}

// ArtifactBinding is an explicit v2 source union. A derived input has its own
// pinned bytes and names the independent underlying dataset in Stage.Inputs.
// A predecessor selector is resolved only from this execution's complete receipts.
type ArtifactBinding struct {
	Kind        string            `json:"kind"`
	Input       ArtifactRef       `json:"input,omitzero"`
	Artifact    StageArtifact     `json:"artifact,omitzero"`
	Predecessor PredecessorOutput `json:"predecessor,omitzero"`
}

// BoundArtifact retains the exact result of resolution. Absolute installation
// paths are deliberately absent, so evidence is portable with its artifact root.
type BoundArtifact struct {
	BindingSHA256 string          `json:"binding_sha256"`
	StageID       string          `json:"stage_id,omitempty"`
	ReceiptSHA256 string          `json:"receipt_sha256,omitempty"`
	Identity      ReceiptIdentity `json:"identity,omitzero"`
	Artifact      StageArtifact   `json:"artifact"`
}

type BoundArtifactFile struct {
	Binding BoundArtifact
	Path    string
}

// Validate checks authority and bounds without requiring any future output.
// The binding is frozen as part of the consuming stage's complete protocol.
func (b ArtifactBinding) Validate(recipe Recipe, stage Stage, maxBytes int64) error {
	if b.validate(maxBytes) != nil || recipe.Validate() != nil {
		return ErrBinding
	}
	found := false
	for _, s := range recipe.Stages {
		if reflect.DeepEqual(s, stage) {
			found = true
		}
	}
	if !found {
		return ErrBinding
	}
	switch b.Kind {
	case BindingDerivedInput:
		if !slices.Contains(stage.Inputs, b.Input) ||
			(b.Input != recipe.Training && b.Input != recipe.Recovery && b.Input != recipe.Calibration) {
			return ErrBinding
		}
	case BindingPredecessor:
		p := b.Predecessor
		found = false
		for _, earlier := range recipe.Stages {
			if earlier.ID == stage.ID {
				break
			}
			if earlier.ID == p.StageID {
				found = true
			}
		}
		if !found {
			return ErrBinding
		}
	default:
		return ErrBinding
	}
	return nil
}

func (b ArtifactBinding) validate(maxBytes int64) error {
	if maxBytes < 1 || maxBytes > 1<<40 {
		return ErrBinding
	}
	switch b.Kind {
	case BindingDerivedInput:
		if b.Predecessor != (PredecessorOutput{}) || !identifier(b.Input.ID) || !digest(b.Input.SHA256) || !constrainedArtifact(b.Artifact, maxBytes) {
			return ErrBinding
		}
	case BindingPredecessor:
		p := b.Predecessor
		if b.Input != (ArtifactRef{}) || b.Artifact != (StageArtifact{}) || !identifier(p.StageID) || !identifier(p.Schema) || p.MaxBytes < 1 || p.MaxBytes > maxBytes || !filepath.IsLocal(p.Path) || p.Path == "." || filepath.Clean(p.Path) != p.Path || len(p.Path) > 1024 || strings.ContainsRune(p.Path, 0) {
			return ErrBinding
		}
	default:
		return ErrBinding
	}
	return nil
}

// ValidateBounds checks only the union and limits. Admission must also call
// Validate with the exact recipe and stage before resolution.
func (b ArtifactBinding) ValidateBounds(maxBytes int64) error { return b.validate(maxBytes) }

// ByteLimit is the frozen upper bound; resolved predecessor bytes may be smaller.
func (b ArtifactBinding) ByteLimit() int64 {
	if b.Kind == BindingPredecessor {
		return b.Predecessor.MaxBytes
	}
	return b.Artifact.Bytes
}

// ResolveArtifact consumes only the verified Completed receipts supplied by
// DurableRuntime. It additionally checks receipt/identity/generation, uniqueness,
// rooted path and actual file bytes. It does not manufacture a receipt or discover
// files. Concrete handlers still verify the content's scientific format and role.
func ResolveArtifact(ctx context.Context, c StageContext, sourceDirectory string, b ArtifactBinding, maxBytes int64) (BoundArtifactFile, error) {
	var out BoundArtifactFile
	if ctx == nil {
		return out, ErrBinding
	}
	if err := ctx.Err(); err != nil {
		return out, err
	}
	if err := b.Validate(c.Execution.Recipe, c.Stage, maxBytes); err != nil {
		return out, err
	}
	x := c.Execution
	rsha, err := x.Recipe.Digest()
	if err != nil || !identifier(x.RunID) || !identifier(x.TenantID) || !digest(x.TargetSHA256) || x.Generation == 0 {
		return out, ErrBinding
	}
	identity := ReceiptIdentity{RunID: x.RunID, TenantID: x.TenantID, RecipeSHA256: rsha, TargetSHA256: x.TargetSHA256, PlacementSHA256: jsonSHA256(x.Placement), Generation: x.Generation}
	out.Binding.BindingSHA256 = jsonSHA256(b)
	directory := sourceDirectory
	if b.Kind == BindingDerivedInput {
		out.Binding.Artifact = b.Artifact
	} else {
		directory = c.ArtifactDirectory
		matches := 0
		for _, r := range c.Completed {
			if r.StageID != b.Predecessor.StageID {
				continue
			}
			matches++
			if r.Version != 1 || !r.Result.Complete || r.Result.Steps < 1 || r.SHA256 != receiptSHA256(r) || !sameIdentity(r.Identity, identity) || r.Identity.Generation == 0 || r.Identity.Generation > x.Generation {
				return out, ErrBinding
			}
			artifacts := 0
			for _, a := range r.Result.Artifacts {
				if a.Path == b.Predecessor.Path {
					artifacts++
					if !constrainedArtifact(a, b.Predecessor.MaxBytes) {
						return out, ErrBinding
					}
					out.Binding.Artifact = a
				}
			}
			if artifacts != 1 {
				return out, ErrBinding
			}
			out.Binding.StageID, out.Binding.ReceiptSHA256, out.Binding.Identity = r.StageID, r.SHA256, r.Identity
		}
		if matches != 1 {
			return out, ErrBinding
		}
	}
	if !filepath.IsAbs(directory) {
		return out, ErrBinding
	}
	root, err := constrainedRoot(directory)
	if err != nil {
		return out, errors.Join(ErrBinding, err)
	}
	a := out.Binding.Artifact
	_, err = constrainedRead(ctx, root, a.Path, a.Bytes, a.SHA256, false)
	err = errors.Join(err, root.Close())
	if err != nil {
		return out, errors.Join(ErrBinding, err)
	}
	out.Path = filepath.Join(directory, a.Path)
	return out, ctx.Err()
}
