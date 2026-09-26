package alignment

import (
	"bytes"
	"context"
	"errors"
	"path/filepath"

	"github.com/tayi-ai/arandu-llama/training/fusioncache"
	"github.com/tayi-ai/arandu-llama/training/pipeline"
)

const EvidenceSchema = "alignment-evidence-v2"

// ReadBoundProjection reads one output resolved from a completed alignment
// receipt. The caller must resolve the selector with pipeline.ResolveArtifact
// and require EvidenceSchema. DurableRuntime verifies the producer's sources,
// reconstruction thresholds and receipt before supplying Completed. This reader
// rechecks the output bytes, producer identity and exact frozen matching plan;
// it does not refit or certify an independently supplied receipt.
func ReadBoundProjection(ctx context.Context, file pipeline.BoundArtifactFile, plan fusioncache.MatchingPlan, expectedPlanSHA string, limits fusioncache.ProjectionLimits, maxBytes int64) (fusioncache.Projection, error) {
	var empty fusioncache.Projection
	b := file.Binding
	if ctx == nil || maxBytes < 1 || maxBytes > 256<<20 || !artifactValid(b.Artifact, maxBytes) ||
		!identifier(b.StageID) || !validSHA(b.BindingSHA256) || !validSHA(b.ReceiptSHA256) ||
		!identifier(b.Identity.RunID) || !identifier(b.Identity.TenantID) || b.Identity.Generation == 0 ||
		!validSHA(b.Identity.RecipeSHA256) || !validSHA(b.Identity.TargetSHA256) || !validSHA(b.Identity.PlacementSHA256) ||
		!filepath.IsAbs(file.Path) || filepath.Clean(file.Path) != file.Path ||
		limits.MaxSamples < 1 || limits.MaxSamples > 1<<20 || limits.MaxDimension < 1 || limits.MaxDimension > 1<<20 || limits.MaxElements < 1 || limits.MaxElements > 1<<26 ||
		len(plan.Fit) < 1 || len(plan.Heldout) < 1 || len(plan.Fit) > limits.MaxSamples || len(plan.Heldout) > limits.MaxSamples-len(plan.Fit) ||
		!validModel(plan.SourceModel) || !validModel(plan.TargetModel) || !validSpec(plan.Source, limits.MaxDimension) || !validSpec(plan.Target, limits.MaxDimension) {
		return empty, ErrContract
	}
	if err := ctx.Err(); err != nil {
		return empty, err
	}
	planSHA, err := fusioncache.Digest(plan)
	if err != nil || planSHA != expectedPlanSHA || !validSHA(expectedPlanSHA) {
		return empty, errors.Join(ErrContract, err)
	}
	// Recover the root from the exact relative artifact path and verify every
	// component through os.Root, including nested paths and symlink refusal.
	directory := file.Path
	for name := b.Artifact.Path; name != "."; name = filepath.Dir(name) {
		directory = filepath.Dir(directory)
	}
	if filepath.Join(directory, b.Artifact.Path) != file.Path {
		return empty, ErrContract
	}
	root, err := rooted(directory)
	if err != nil {
		return empty, err
	}
	body, readErr := read(ctx, root, b.Artifact.Path, b.Artifact.Bytes, b.Artifact.SHA256)
	if err := errors.Join(readErr, root.Close()); err != nil {
		return empty, err
	}
	var e Evidence
	if err := decode(body, &e); err != nil {
		return empty, err
	}
	canonical, err := boundedJSON(e, maxBytes)
	if err != nil || !bytes.Equal(canonical, body) || e.Version != 2 || e.StageID != b.StageID || e.Step < 1 ||
		!sameIdentity(e.Identity, b.Identity) || e.Identity.Generation == 0 || e.Identity.Generation > b.Identity.Generation ||
		!validSHA(e.ProtocolSHA256) || !validSHA(e.FitSHA256) || !validSHA(e.ParentSHA256) || !artifactValid(e.Intent, maxBytes) ||
		e.Projection.SourceDimension != plan.Source.Dimension || e.Projection.TargetDimension != plan.Target.Dimension ||
		e.Report.PlanSHA256 != expectedPlanSHA || e.Report.Dual != e.Projection.Dual ||
		e.Report.FitValues != int64(len(plan.Fit))*int64(plan.Target.Dimension) || e.Report.HeldoutValues != int64(len(plan.Heldout))*int64(plan.Target.Dimension) ||
		!finite(e.Report.FitSquaredError) || e.Report.FitSquaredError < 0 || !finite(e.Report.HeldoutSquaredError) || e.Report.HeldoutSquaredError < 0 {
		return empty, errors.Join(ErrContract, err)
	}
	if err := e.Projection.Validate(expectedPlanSHA, limits); err != nil {
		return empty, err
	}
	return e.Projection, ctx.Err()
}
