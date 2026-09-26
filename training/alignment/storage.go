package alignment

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"

	"github.com/tayi-ai/arandu-llama/training/fusioncache"
	"github.com/tayi-ai/arandu-llama/training/pipeline"
)

// Evidence stores the complete projection and measured reconstruction report in
// one atomic artifact. The source pin, protocol and lineage remain inspectable.
type Evidence struct {
	Version        int                      `json:"version"`
	Identity       pipeline.ReceiptIdentity `json:"identity"`
	StageID        string                   `json:"stage_id"`
	Step           int                      `json:"step"`
	ProtocolSHA256 string                   `json:"protocol_sha256"`
	FitSHA256      string                   `json:"fit_sha256"`
	ParentSHA256   string                   `json:"parent_sha256"`
	PreviousSHA256 string                   `json:"previous_sha256"`
	Source         Source                   `json:"source"`
	Projection     fusioncache.Projection   `json:"projection"`
	Report         fusioncache.FitReport    `json:"report"`
	Binding        pipeline.BoundArtifact   `json:"binding,omitzero"`
	Intent         pipeline.StageArtifact   `json:"intent,omitzero"`
}

type alignmentIntent struct {
	Version        int                      `json:"version"`
	Identity       pipeline.ReceiptIdentity `json:"identity"`
	StageID        string                   `json:"stage_id"`
	Step           int                      `json:"step"`
	ProtocolSHA256 string                   `json:"protocol_sha256"`
	FitSHA256      string                   `json:"fit_sha256"`
	ParentSHA256   string                   `json:"parent_sha256"`
	PreviousSHA256 string                   `json:"previous_sha256"`
	Binding        pipeline.BoundArtifact   `json:"binding"`
}

var ErrAttempt = errors.New("alignment: unresolved attempt requires review")

func (h *Stage) binding(ctx context.Context, c pipeline.StageContext, f Fit) (pipeline.BoundArtifact, error) {
	if h.c.Protocol.Version == 1 {
		return pipeline.BoundArtifact{}, nil
	}
	file, err := pipeline.ResolveArtifact(ctx, c, h.c.SourceDirectory, f.Source.Binding, h.c.Protocol.Limits.MaxSourceBytes)
	return file.Binding, err
}

func rooted(path string) (*os.Root, error) {
	if !filepath.IsAbs(path) {
		return nil, ErrContract
	}
	i, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if !i.IsDir() || i.Mode()&os.ModeSymlink != 0 {
		return nil, ErrContract
	}
	return os.OpenRoot(path)
}

// regular refuses symlinks at every component, special files and replacement
// between lstat/open. os.Root also prevents path escape during resolution.
func regular(root *os.Root, name string) (*os.File, error) {
	if !filepath.IsLocal(name) || name == "." || filepath.Clean(name) != name {
		return nil, ErrContract
	}
	parts := strings.Split(name, string(filepath.Separator))
	var before os.FileInfo
	for i := range parts {
		v, err := root.Lstat(filepath.Join(parts[:i+1]...))
		if err != nil {
			return nil, err
		}
		if v.Mode()&os.ModeSymlink != 0 || (i < len(parts)-1 && !v.IsDir()) {
			return nil, ErrContract
		}
		before = v
	}
	if !before.Mode().IsRegular() {
		return nil, ErrContract
	}
	f, err := root.Open(name)
	if err != nil {
		return nil, err
	}
	after, err := f.Stat()
	if err != nil || !after.Mode().IsRegular() || !os.SameFile(before, after) {
		f.Close()
		return nil, ErrContract
	}
	return f, nil
}

type contextReader struct {
	ctx context.Context
	r   io.Reader
}

func (r contextReader) Read(p []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	return r.r.Read(p)
}

func read(ctx context.Context, root *os.Root, name string, max int64, expected string) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	f, err := regular(root, name)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	before, err := f.Stat()
	if err != nil || before.Size() < 1 || before.Size() > max || (expected != "" && before.Size() != max) {
		return nil, ErrContract
	}
	body, err := io.ReadAll(io.LimitReader(contextReader{ctx, f}, before.Size()+1))
	if err != nil {
		return nil, err
	}
	after, err := f.Stat()
	if err != nil || int64(len(body)) != before.Size() || after.Size() != before.Size() || !before.ModTime().Equal(after.ModTime()) || (expected != "" && bodySHA(body) != expected) {
		return nil, ErrContract
	}
	return body, ctx.Err()
}

func decode(body []byte, v any) error {
	d := json.NewDecoder(bytes.NewReader(body))
	d.DisallowUnknownFields()
	if err := d.Decode(v); err != nil {
		return errors.Join(ErrContract, err)
	}
	if d.Decode(new(any)) != io.EOF {
		return ErrContract
	}
	return nil
}

func receiptSHA(r pipeline.StageReceipt) string {
	r.SHA256 = ""
	s, _ := fusioncache.Digest(r)
	return s
}
func sameIdentity(a, b pipeline.ReceiptIdentity) bool {
	a.Generation, b.Generation = 0, 0
	return a == b
}

func (h *Stage) identity(ctx context.Context, c pipeline.StageContext) (pipeline.ReceiptIdentity, error) {
	if err := h.Admit(ctx, c.Execution.Recipe, c.Stage, c.Execution.Placement); err != nil {
		return pipeline.ReceiptIdentity{}, err
	}
	x := c.Execution
	id := pipeline.ReceiptIdentity{RunID: x.RunID, TenantID: x.TenantID, RecipeSHA256: h.c.Protocol.RecipeSHA256, TargetSHA256: x.TargetSHA256, PlacementSHA256: h.c.Protocol.PlacementSHA256, Generation: x.Generation}
	if !identifier(x.RunID) || !identifier(x.TenantID) || x.Generation == 0 || !validSHA(x.TargetSHA256) || !filepath.IsAbs(c.ArtifactDirectory) || c.Parent == nil {
		return id, ErrContract
	}
	p := c.Parent
	if p.Version != 1 || p.StageID != c.Stage.ParentStage || !p.Result.Complete || p.Result.Steps < 1 || p.SHA256 != receiptSHA(*p) || !sameIdentity(p.Identity, id) || p.Identity.Generation == 0 || p.Identity.Generation > id.Generation {
		return id, ErrContract
	}
	return id, nil
}

func (h *Stage) source(ctx context.Context, c pipeline.StageContext, f Fit) (Samples, error) {
	var samples Samples
	id, err := h.identity(ctx, c)
	if err != nil {
		return samples, err
	}
	s := f.Source
	dir := h.c.SourceDirectory
	artifact := s.Artifact
	if h.c.Protocol.Version == 2 {
		bound, err := h.binding(ctx, c, f)
		if err != nil {
			return samples, err
		}
		artifact = bound.Artifact
		if s.Binding.Kind == pipeline.BindingPredecessor {
			dir = c.ArtifactDirectory
		}
	} else if s.StageID != "" {
		dir = c.ArtifactDirectory
		found := false
		for _, r := range c.Completed {
			if r.Version == 1 && r.StageID == s.StageID && r.Result.Complete && r.Result.Steps > 0 && r.SHA256 == s.ReceiptSHA256 && receiptSHA(r) == r.SHA256 && sameIdentity(r.Identity, id) && r.Identity.Generation > 0 && r.Identity.Generation <= id.Generation && slices.Contains(r.Result.Artifacts, s.Artifact) {
				found = true
			}
		}
		if !found {
			return samples, fmt.Errorf("%w: completed source receipt absent", ErrContract)
		}
	}
	root, err := rooted(dir)
	if err != nil {
		return samples, err
	}
	defer root.Close()
	body, err := read(ctx, root, artifact.Path, artifact.Bytes, artifact.SHA256)
	if err != nil {
		return samples, err
	}
	if err := decode(body, &samples); err != nil {
		return samples, err
	}
	p := f.MatchingPlan
	if samples.Version != 1 || samples.MatchingPlanSHA256 != f.MatchingPlanSHA256 || samples.SourceModel != p.SourceModel || samples.TargetModel != p.TargetModel || samples.TokenMappingSHA256 != p.TokenMappingSHA256 || samples.Source != p.Source || samples.Target != p.Target || len(samples.Fit) != len(p.Fit) || len(samples.Heldout) != len(p.Heldout) {
		return samples, ErrContract
	}
	for part, rows := range [][]fusioncache.FeatureSample{samples.Fit, samples.Heldout} {
		ids := p.Fit
		if part == 1 {
			ids = p.Heldout
		}
		for i, row := range rows {
			if err := ctx.Err(); err != nil {
				return samples, err
			}
			if row.Identity != ids[i] || len(row.Source) != p.Source.Dimension || len(row.Target) != p.Target.Dimension {
				return samples, ErrContract
			}
			for _, v := range [][]float64{row.Source, row.Target} {
				for _, x := range v {
					if !finite(x) {
						return samples, ErrContract
					}
				}
			}
		}
	}
	return samples, nil
}

func (h *Stage) name(step int) string { return fmt.Sprintf("alignment-%s-%06d.json", h.stage.ID, step) }

func (h *Stage) intentName(step int) string {
	return strings.TrimSuffix(h.name(step), ".json") + "-intent.json"
}

func intentFor(e Evidence) alignmentIntent {
	return alignmentIntent{Version: 2, Identity: e.Identity, StageID: e.StageID, Step: e.Step, ProtocolSHA256: e.ProtocolSHA256, FitSHA256: e.FitSHA256, ParentSHA256: e.ParentSHA256, PreviousSHA256: e.PreviousSHA256, Binding: e.Binding}
}

// Link publishes a complete synced file without replacing an existing result.
// Interrupted temporary files are not evidence and never enter the receipt.
func writeAtomic(ctx context.Context, root *os.Root, name string, body []byte) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	var nonce [16]byte
	if _, err := rand.Read(nonce[:]); err != nil {
		return err
	}
	tmp := ".alignment-" + hex.EncodeToString(nonce[:]) + ".tmp"
	f, err := root.OpenFile(tmp, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return err
	}
	defer root.Remove(tmp)
	_, writeErr := io.Copy(f, contextReader{ctx, bytes.NewReader(body)})
	syncErr := f.Sync()
	closeErr := f.Close()
	if err := errors.Join(writeErr, syncErr, closeErr, ctx.Err()); err != nil {
		return err
	}
	if err := root.Link(tmp, name); err != nil {
		return err
	}
	if err := root.Remove(tmp); err != nil {
		return err
	}
	return syncDirectory(root)
}

func syncDirectory(root *os.Root) error {
	f, err := root.Open(".")
	if err != nil {
		return err
	}
	return errors.Join(f.Sync(), f.Close())
}
func syncArtifact(root *os.Root, name string) error {
	f, err := regular(root, name)
	if err != nil {
		return err
	}
	return errors.Join(f.Sync(), f.Close(), syncDirectory(root))
}

func (h *Stage) verifyEvidence(ctx context.Context, c pipeline.StageContext, step int, prior string, generation uint64, body []byte) (Evidence, error) {
	var e Evidence
	if err := decode(body, &e); err != nil {
		return e, err
	}
	id, err := h.identity(ctx, c)
	if err != nil {
		return e, err
	}
	f := h.c.Protocol.Fits[step-1]
	fsha, _ := fusioncache.Digest(f)
	canonical, err := boundedJSON(e, h.c.Protocol.Limits.MaxArtifactBytes)
	if err != nil || !bytes.Equal(canonical, body) || e.Version != h.c.Protocol.Version || !sameIdentity(e.Identity, id) || e.Identity.Generation == 0 || e.Identity.Generation > id.Generation || e.Identity.Generation < generation || e.StageID != c.Stage.ID || e.Step != step || e.ProtocolSHA256 != h.c.ProtocolSHA256 || e.FitSHA256 != fsha || e.ParentSHA256 != c.Parent.SHA256 || e.PreviousSHA256 != prior || e.Source != f.Source {
		return e, ErrContract
	}
	bound, err := h.binding(ctx, c, f)
	if err != nil || e.Binding != bound {
		return e, errors.Join(ErrContract, err)
	}
	if h.c.Protocol.Version == 1 {
		if e.Intent != (pipeline.StageArtifact{}) {
			return e, ErrContract
		}
	} else {
		if e.Intent.Path != h.intentName(step) || !artifactValid(e.Intent, h.c.Protocol.Limits.MaxArtifactBytes) {
			return e, ErrContract
		}
		root, err := rooted(c.ArtifactDirectory)
		if err != nil {
			return e, err
		}
		intentBody, readErr := read(ctx, root, e.Intent.Path, e.Intent.Bytes, e.Intent.SHA256)
		if err := errors.Join(readErr, root.Close()); err != nil {
			return e, err
		}
		want, err := boundedJSON(intentFor(e), h.c.Protocol.Limits.MaxArtifactBytes)
		if err != nil || !bytes.Equal(intentBody, want) {
			return e, ErrContract
		}
	}
	samples, err := h.source(ctx, c, f)
	if err != nil {
		return e, err
	}
	if e.Projection.SourceDimension != f.MatchingPlan.Source.Dimension || e.Projection.TargetDimension != f.MatchingPlan.Target.Dimension || e.Projection.Dual != (len(samples.Fit) < f.MatchingPlan.Source.Dimension) {
		return e, ErrContract
	}
	if e.Projection.Dual {
		basis := make([][]float64, len(samples.Fit))
		for i, s := range samples.Fit {
			basis[i] = s.Source
		}
		if !reflect.DeepEqual(basis, e.Projection.Basis) {
			return e, ErrContract
		}
	}
	report, err := h.evaluate(ctx, f, samples, e.Projection)
	if err != nil || report != e.Report {
		return e, errors.Join(ErrContract, err)
	}
	return e, nil
}

// evaluate applies the saved coefficients and recomputes both errors. It never
// solves, reselects a layer, tunes ridge, or consumes sealed final evaluation.
func (h *Stage) evaluate(ctx context.Context, f Fit, s Samples, p fusioncache.Projection) (fusioncache.FitReport, error) {
	r := fusioncache.FitReport{PlanSHA256: f.MatchingPlanSHA256, Dual: p.Dual}
	if err := p.Validate(f.MatchingPlanSHA256, h.c.Protocol.Limits.Projection); err != nil {
		return r, err
	}
	for part, rows := range [][]fusioncache.FeatureSample{s.Fit, s.Heldout} {
		squared := 0.0
		for _, row := range rows {
			if err := ctx.Err(); err != nil {
				return r, err
			}
			prediction, err := p.Apply(row.Source, f.MatchingPlanSHA256, h.c.Protocol.Limits.Projection)
			if err != nil {
				return r, err
			}
			for i, target := range row.Target {
				d := prediction[i] - target
				squared += d * d
			}
		}
		if !finite(squared) {
			return r, ErrContract
		}
		count := int64(len(rows)) * int64(f.MatchingPlan.Target.Dimension)
		threshold := f.Qualification.MaxFitMSE
		if part == 0 {
			r.FitSquaredError, r.FitValues = squared, count
		} else {
			r.HeldoutSquaredError, r.HeldoutValues = squared, count
			threshold = f.Qualification.MaxHeldoutMSE
		}
		if count <= 0 || squared/float64(count) > threshold {
			return r, fmt.Errorf("%w: reconstruction qualification exceeded in partition %d", ErrContract, part)
		}
	}
	return r, ctx.Err()
}
