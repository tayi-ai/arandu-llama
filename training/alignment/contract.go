// Package alignment persists explicitly qualified ridge feature projections.
// Reconstruction qualification does not establish language-model capability.
package alignment

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"reflect"
	"slices"
	"strings"

	"github.com/tayi-ai/arandu-llama/training/fusioncache"
	"github.com/tayi-ai/arandu-llama/training/pipeline"
)

var ErrContract = errors.New("alignment: stage contract refused")

// Source admits exactly one origin. Input identifies the underlying dataset in
// Stage.Inputs; Artifact independently pins the derived Samples document through
// the frozen Protocol. Every sample must belong to that Input. Alternatively,
// StageID and ReceiptSHA256 pin a completed predecessor containing Artifact.
type Source struct {
	Artifact      pipeline.StageArtifact `json:"artifact"`
	Input         pipeline.ArtifactRef   `json:"input"`
	StageID       string                 `json:"stage_id"`
	ReceiptSHA256 string                 `json:"receipt_sha256"`
}

// Samples retains both numerical and causal provenance. Heldout is a disjoint
// alignment validation partition, never the recipe's sealed final evaluation.
type Samples struct {
	Version            int                         `json:"version"`
	MatchingPlanSHA256 string                      `json:"matching_plan_sha256"`
	SourceModel        fusioncache.ModelIdentity   `json:"source_model"`
	TargetModel        fusioncache.ModelIdentity   `json:"target_model"`
	TokenMappingSHA256 string                      `json:"token_mapping_sha256"`
	Source             fusioncache.FeatureSpec     `json:"source"`
	Target             fusioncache.FeatureSpec     `json:"target"`
	Fit                []fusioncache.FeatureSample `json:"fit"`
	Heldout            []fusioncache.FeatureSample `json:"heldout"`
}

// Qualification freezes maximum squared error per scalar on both partitions.
// Zero is an explicit exact-reconstruction threshold, not an omitted default.
type Qualification struct {
	MaxFitMSE     float64 `json:"max_fit_mse"`
	MaxHeldoutMSE float64 `json:"max_heldout_mse"`
}

type Fit struct {
	MatchingPlan       fusioncache.MatchingPlan `json:"matching_plan"`
	MatchingPlanSHA256 string                   `json:"matching_plan_sha256"`
	Source             Source                   `json:"source"`
	Qualification      Qualification            `json:"qualification"`
}

// Limits bounds all serialized artifacts, projection storage and cubic work.
// MaxSolveWork conservatively counts scalar operations before the synchronous
// FitProjection call. Cancellation is checked before and after that bounded call.
type Limits struct {
	MaxProtocolBytes int64                        `json:"max_protocol_bytes"`
	MaxSourceBytes   int64                        `json:"max_source_bytes"`
	MaxArtifactBytes int64                        `json:"max_artifact_bytes"`
	MaxTotalBytes    int64                        `json:"max_total_bytes"`
	MaxFits          int                          `json:"max_fits"`
	MaxSolveWork     int64                        `json:"max_solve_work"`
	Projection       fusioncache.ProjectionLimits `json:"projection"`
}

type Protocol struct {
	Version             int    `json:"version"`
	RecipeSHA256        string `json:"recipe_sha256"`
	PlacementSHA256     string `json:"placement_sha256"`
	QualificationSHA256 string `json:"qualification_sha256"`
	StageID             string `json:"stage_id"`
	Fits                []Fit  `json:"fits"`
	Limits              Limits `json:"limits"`
}

// Digest hashes canonical JSON including its final newline, after validation.
func (p Protocol) Digest() (string, error) {
	if err := p.validate(); err != nil {
		return "", err
	}
	b, err := boundedJSON(p, p.Limits.MaxProtocolBytes)
	if err != nil {
		return "", err
	}
	return bodySHA(b), nil
}

type Config struct {
	Recipe          pipeline.Recipe
	Placement       pipeline.Placement
	Protocol        Protocol
	ProtocolSHA256  string
	SourceDirectory string
}

// Stage fits frozen layer correspondences; it never chooses layers or ridge.
// DurableRuntime owns cross-process locking. Direct callers must provide the
// same lock and keep source/artifact directories private and immutable.
type Stage struct {
	c     Config
	stage pipeline.Stage
	gate  chan struct{}
}

var _ pipeline.StageHandler = (*Stage)(nil)

func NewStage(c Config) (*Stage, error) {
	digest, err := c.Protocol.Digest()
	if err != nil || digest != c.ProtocolSHA256 || !filepath.IsAbs(c.SourceDirectory) {
		return nil, errors.Join(ErrContract, err)
	}
	// Validate caller-owned collections before snapshotting, so a malformed
	// recipe or placement cannot force an unbounded JSON allocation.
	h := &Stage{c: c}
	for _, s := range c.Recipe.Stages {
		if s.ID == c.Protocol.StageID {
			h.stage = s
		}
	}
	if err := h.Admit(context.Background(), c.Recipe, h.stage, c.Placement); err != nil {
		return nil, err
	}
	// Clone every nested slice so installation mutation cannot change admission.
	body, err := boundedJSON(c, c.Protocol.Limits.MaxProtocolBytes+4<<20)
	if err != nil {
		return nil, err
	}
	var snapshot Config
	if err := json.Unmarshal(body, &snapshot); err != nil {
		return nil, err
	}
	h = &Stage{c: snapshot, gate: make(chan struct{}, 1)}
	for _, s := range snapshot.Recipe.Stages {
		if s.ID == snapshot.Protocol.StageID {
			h.stage = s
		}
	}
	if err := h.Admit(context.Background(), snapshot.Recipe, h.stage, snapshot.Placement); err != nil {
		return nil, err
	}
	return h, nil
}

// Admit checks exact protocol, placement, model pins and allowed dataset roles.
// It creates no files and performs no fitting or feature capture.
func (h *Stage) Admit(ctx context.Context, recipe pipeline.Recipe, stage pipeline.Stage, placement pipeline.Placement) error {
	if h == nil || ctx == nil {
		return ErrContract
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if !validSHA(placement.RuntimeSHA256) || !identifier(placement.Backend) || len(placement.Nodes) > 1024 {
		return ErrContract
	}
	nodes := map[string]bool{}
	for _, node := range placement.Nodes {
		if !identifier(node) || nodes[node] {
			return ErrContract
		}
		nodes[node] = true
	}
	p := h.c.Protocol
	rsha, err := recipe.Digest()
	psha, perr := fusioncache.Digest(placement)
	if err != nil || perr != nil || rsha != p.RecipeSHA256 || psha != p.PlacementSHA256 ||
		!reflect.DeepEqual(recipe, h.c.Recipe) || !reflect.DeepEqual(stage, h.stage) || !reflect.DeepEqual(placement, h.c.Placement) ||
		stage.Phase != pipeline.PhaseAlignment || stage.MaxSteps != len(p.Fits) || p.QualificationSHA256 != placement.QualificationSHA256 ||
		!validSHA(placement.RuntimeSHA256) || !identifier(placement.Backend) || len(placement.Nodes) > 1024 ||
		placement.MemoryBytes < h.residentReservation() || stage.MaxTokens > placement.MaxTokens {
		return ErrContract
	}
	for _, f := range p.Fits {
		plan := f.MatchingPlan
		if plan.TargetModel.WeightsSHA256 != recipe.Student.SHA256 || !slices.Contains(stage.Inputs, recipe.Student) {
			return ErrContract
		}
		teacher := false
		for _, ref := range recipe.Teachers {
			if ref.SHA256 == plan.SourceModel.WeightsSHA256 && slices.Contains(stage.Inputs, ref) {
				teacher = true
			}
		}
		if !teacher {
			return ErrContract
		}
		for _, ids := range [][]fusioncache.SampleIdentity{plan.Fit, plan.Heldout} {
			for _, id := range ids {
				ref := recipe.Training
				if id.Role == "calibration" {
					ref = recipe.Calibration
				} else if id.Role != "train" {
					return ErrContract
				}
				if id.DatasetSHA256 != ref.SHA256 || !slices.Contains(stage.Inputs, ref) || id.TargetIndex >= stage.MaxTokens {
					return ErrContract
				}
			}
		}
		if f.Source.StageID == "" {
			if !slices.Contains(stage.Inputs, f.Source.Input) || (f.Source.Input != recipe.Training && f.Source.Input != recipe.Calibration) {
				return ErrContract
			}
			for _, ids := range [][]fusioncache.SampleIdentity{plan.Fit, plan.Heldout} {
				for _, id := range ids {
					if id.DatasetSHA256 != f.Source.Input.SHA256 {
						return ErrContract
					}
				}
			}
		} else {
			found := false
			for _, earlier := range recipe.Stages {
				if earlier.ID == stage.ID {
					break
				}
				if earlier.ID == f.Source.StageID {
					found = true
				}
			}
			if !found {
				return ErrContract
			}
		}
	}
	return nil
}

// residentReservation includes conservative expansion of typed JSON, current
// input/output buffers, solver arrays, configuration snapshots and receipt
// metadata. It is an admission reservation, not a measured process RSS. Only
// one projection remains resident; prior projections are discarded after verify.
func (h *Stage) residentReservation() int64 {
	l := h.c.Protocol.Limits
	var sourceBytes int64
	for _, f := range h.c.Protocol.Fits {
		sourceBytes = max(sourceBytes, f.Source.Artifact.Bytes)
	}
	return 8<<20 + 16*l.MaxProtocolBytes + 128*sourceBytes + 128*l.MaxArtifactBytes + 32*l.Projection.MaxElements + int64(len(h.c.Protocol.Fits))*256
}

func (p Protocol) validate() error {
	l := p.Limits
	x := l.Projection
	if p.Version != 1 || !validSHA(p.RecipeSHA256) || !validSHA(p.PlacementSHA256) || !validSHA(p.QualificationSHA256) || !identifier(p.StageID) ||
		l.MaxProtocolBytes < 1 || l.MaxProtocolBytes > 64<<20 || l.MaxSourceBytes < 1 || l.MaxSourceBytes > 256<<20 ||
		l.MaxArtifactBytes < 1 || l.MaxArtifactBytes > 256<<20 || l.MaxTotalBytes < l.MaxArtifactBytes ||
		l.MaxFits < 1 || l.MaxFits > 4096 || len(p.Fits) < 1 || len(p.Fits) > l.MaxFits || int64(len(p.Fits)) > l.MaxTotalBytes/l.MaxArtifactBytes ||
		l.MaxSolveWork < 1 || l.MaxSolveWork > 1_000_000_000 || x.MaxSamples < 2 || x.MaxSamples > 1<<20 || x.MaxDimension < 1 || x.MaxDimension > 1<<20 || x.MaxElements < 1 || x.MaxElements > 1<<26 {
		return ErrContract
	}
	seen := map[string]bool{}
	for _, f := range p.Fits {
		m := f.MatchingPlan
		if seen[m.ID] || !identifier(m.ID) || !validModel(m.SourceModel) || !validModel(m.TargetModel) || !validSHA(m.TokenMappingSHA256) ||
			!validSpec(m.Source, x.MaxDimension) || !validSpec(m.Target, x.MaxDimension) || !finite(m.Ridge) || m.Ridge <= 0 ||
			!finite(f.Qualification.MaxFitMSE) || f.Qualification.MaxFitMSE < 0 || !finite(f.Qualification.MaxHeldoutMSE) || f.Qualification.MaxHeldoutMSE < 0 ||
			len(m.Fit) < 1 || len(m.Heldout) < 1 || len(m.Fit) > x.MaxSamples || len(m.Heldout) > x.MaxSamples-len(m.Fit) {
			return ErrContract
		}
		seen[m.ID] = true
		sha, err := fusioncache.Digest(m)
		if err != nil || sha != f.MatchingPlanSHA256 {
			return ErrContract
		}
		rows, examples := map[string]bool{}, map[string]bool{}
		for part, ids := range [][]fusioncache.SampleIdentity{m.Fit, m.Heldout} {
			for _, id := range ids {
				key := id.DatasetSHA256 + "/" + id.ExampleID
				row := fmt.Sprintf("%s/%d", key, id.TargetIndex)
				if !validSHA(id.DatasetSHA256) || !identifier(id.ExampleID) || (id.Role != "train" && id.Role != "calibration") || id.TargetIndex < 1 || !validSHA(id.TeacherPrefixSHA256) || !validSHA(id.StudentPrefixSHA256) || rows[row] || (part == 1 && examples[key]) {
					return ErrContract
				}
				rows[row] = true
				if part == 0 {
					examples[key] = true
				}
			}
		}
		// Bound input, Gram, RHS, basis and scratch exactly as FitProjection does.
		n, d, out := int64(len(m.Fit)), int64(m.Source.Dimension), int64(m.Target.Dimension)
		size := min(n, d)
		remaining := x.MaxElements
		products := [][2]int64{{n + int64(len(m.Heldout)), d}, {n + int64(len(m.Heldout)), out}, {size, size}, {size, out}, {1, size}, {1, out}}
		if n < d {
			products = append(products, [2]int64{n, d})
		}
		if !withinProducts(remaining, products) || !withinProducts(l.MaxSolveWork, [][2]int64{{size * size, size}, {size * size, out * 2}, {n * d, size + 2*out}, {int64(len(m.Heldout)) + n, 2 * size * (d + out)}}) {
			return ErrContract
		}
		s := f.Source
		if !artifactValid(s.Artifact, l.MaxSourceBytes) {
			return ErrContract
		}
		if s.StageID == "" {
			if !identifier(s.Input.ID) || !validSHA(s.Input.SHA256) || s.ReceiptSHA256 != "" {
				return ErrContract
			}
		} else if !identifier(s.StageID) || !validSHA(s.ReceiptSHA256) || s.Input != (pipeline.ArtifactRef{}) {
			return ErrContract
		}
	}
	return nil
}

func withinProducts(remaining int64, terms [][2]int64) bool {
	for _, t := range terms {
		if t[0] <= 0 || t[1] <= 0 || t[0] > remaining/t[1] {
			return false
		}
		remaining -= t[0] * t[1]
	}
	return true
}
func validSpec(s fusioncache.FeatureSpec, max int) bool {
	return s.Layer >= 0 && len(s.Tensor) > 0 && len(s.Tensor) <= 1024 && (s.DType == "float32" || s.DType == "float64") && s.Dimension > 0 && s.Dimension <= max
}
func validModel(m fusioncache.ModelIdentity) bool {
	if strings.TrimSpace(m.Name) == "" || len(m.Name) > 1024 || (len(m.Revision) != 40 && len(m.Revision) != 64) || strings.ToLower(m.Revision) != m.Revision {
		return false
	}
	if _, err := hex.DecodeString(m.Revision); err != nil {
		return false
	}
	return validSHA(m.WeightsSHA256) && validSHA(m.TokenizerSHA256) && validSHA(m.TemplateSHA256) && validSHA(m.RuntimeSHA256) && m.Vocabulary >= 2
}
func artifactValid(a pipeline.StageArtifact, max int64) bool {
	return filepath.IsLocal(a.Path) && a.Path != "." && filepath.Clean(a.Path) == a.Path && len(a.Path) <= 1024 && !bytes.ContainsRune([]byte(a.Path), 0) && validSHA(a.SHA256) && a.Bytes > 0 && a.Bytes <= max
}
func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }
func identifier(s string) bool {
	if len(s) < 1 || len(s) > 128 || s == "." || s == ".." {
		return false
	}
	for _, c := range s {
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '-' || c == '_' || c == '.') {
			return false
		}
	}
	return true
}
func validSHA(s string) bool {
	if len(s) != 64 {
		return false
	}
	b, err := hex.DecodeString(s)
	return err == nil && hex.EncodeToString(b) == s
}
func bodySHA(b []byte) string { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }

type boundedBuffer struct {
	bytes.Buffer
	limit int64
}

func (b *boundedBuffer) Write(p []byte) (int, error) {
	if int64(len(p)) > b.limit-int64(b.Len()) {
		return 0, ErrContract
	}
	return b.Buffer.Write(p)
}
func boundedJSON(v any, max int64) ([]byte, error) {
	b := &boundedBuffer{limit: max}
	if err := json.NewEncoder(b).Encode(v); err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}
