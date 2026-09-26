package alignment

import (
	"context"
	"errors"
	"fmt"
	"os"
	"reflect"
	"slices"

	"github.com/tayi-ai/arandu-llama/training/fusioncache"
	"github.com/tayi-ai/arandu-llama/training/pipeline"
)

type state struct {
	generations []uint64
	artifacts   []pipeline.StageArtifact
}

func (s state) result(steps int, total int) pipeline.StageResult {
	return pipeline.StageResult{Steps: int64(steps), Complete: steps == total, Artifacts: slices.Clone(s.artifacts[:steps])}
}

func (h *Stage) scan(ctx context.Context, c pipeline.StageContext) (state, error) {
	var s state
	if _, err := h.identity(ctx, c); err != nil {
		return s, err
	}
	root, err := rooted(c.ArtifactDirectory)
	if err != nil {
		return s, err
	}
	defer root.Close()
	prior := ""
	var generation uint64
	var total int64
	gap := false
	for i := range h.c.Protocol.Fits {
		name := h.name(i + 1)
		body, err := read(ctx, root, name, h.c.Protocol.Limits.MaxArtifactBytes, "")
		if errors.Is(err, os.ErrNotExist) {
			gap = true
			continue
		}
		if err != nil {
			return s, err
		}
		if gap {
			return s, fmt.Errorf("%w: noncontiguous projection artifacts", ErrContract)
		}
		e, err := h.verifyEvidence(ctx, c, i+1, prior, generation, body)
		if err != nil {
			return s, err
		}
		if int64(len(body)) > h.c.Protocol.Limits.MaxTotalBytes-total {
			return s, ErrContract
		}
		total += int64(len(body))
		prior = bodySHA(body)
		generation = e.Identity.Generation
		s.generations = append(s.generations, e.Identity.Generation)
		s.artifacts = append(s.artifacts, pipeline.StageArtifact{Path: name, SHA256: prior, Bytes: int64(len(body))})
	}
	return s, ctx.Err()
}

func (h *Stage) receipt(ctx context.Context, c pipeline.StageContext, s state, r pipeline.StageReceipt) error {
	id, err := h.identity(ctx, c)
	steps := int(r.Result.Steps)
	if err != nil || r.Version != 1 || r.StageID != h.stage.ID || r.SHA256 != receiptSHA(r) || !sameIdentity(r.Identity, id) || r.Identity.Generation == 0 || r.Identity.Generation > id.Generation ||
		r.Result.Steps < 1 || r.Result.Steps > int64(len(s.artifacts)) || r.Identity.Generation < s.generations[steps-1] ||
		!reflect.DeepEqual(r.Result, s.result(steps, len(h.c.Protocol.Fits))) {
		return errors.Join(ErrContract, err)
	}
	return nil
}

// Verify rechecks immutable sources, coefficients, reports, thresholds and run
// lineage. It verifies recorded reconstruction, not a model capability gain.
func (h *Stage) Verify(ctx context.Context, c pipeline.StageContext, r pipeline.StageReceipt) error {
	s, err := h.scan(ctx, c)
	if err != nil {
		return err
	}
	return h.receipt(ctx, c, s, r)
}

func (h *Stage) enter(ctx context.Context) error {
	if h == nil || h.gate == nil || ctx == nil {
		return ErrContract
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

func (h *Stage) recover(ctx context.Context, c pipeline.StageContext, s state, commit func(context.Context, pipeline.StageResult) error) error {
	if commit == nil {
		return ErrContract
	}
	steps := 0
	if c.Previous != nil {
		if err := h.receipt(ctx, c, s, *c.Previous); err != nil {
			return err
		}
		steps = int(c.Previous.Result.Steps)
	}
	root, err := rooted(c.ArtifactDirectory)
	if err != nil {
		return err
	}
	defer root.Close()
	for i := steps; i < len(s.artifacts); i++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := syncArtifact(root, s.artifacts[i].Path); err != nil {
			return err
		}
		if err := commit(ctx, s.result(i+1, len(h.c.Protocol.Fits))); err != nil {
			return err
		}
	}
	return ctx.Err()
}

// Reconcile publishes only already persisted, reverified results. No fitting,
// feature capture or model loading occurs, including after a lost callback.
func (h *Stage) Reconcile(ctx context.Context, c pipeline.StageContext, commit func(context.Context, pipeline.StageResult) error) error {
	if err := h.enter(ctx); err != nil {
		return err
	}
	defer func() { <-h.gate }()
	s, err := h.scan(ctx, c)
	if err != nil {
		return err
	}
	return h.recover(ctx, c, s, commit)
}

// Run executes the existing deterministic ridge solver once per missing fit.
// It serializes all work, stops on callback errors, and never detaches compute.
func (h *Stage) Run(ctx context.Context, c pipeline.StageContext, commit func(context.Context, pipeline.StageResult) error) error {
	if err := h.enter(ctx); err != nil {
		return err
	}
	defer func() { <-h.gate }()
	s, err := h.scan(ctx, c)
	if err != nil {
		return err
	}
	if err := h.recover(ctx, c, s, commit); err != nil {
		return err
	}
	id, err := h.identity(ctx, c)
	if err != nil {
		return err
	}
	root, err := rooted(c.ArtifactDirectory)
	if err != nil {
		return err
	}
	defer root.Close()
	for i := len(s.artifacts); i < len(h.c.Protocol.Fits); i++ {
		f := h.c.Protocol.Fits[i]
		samples, err := h.source(ctx, c, f)
		if err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		projection, report, err := fusioncache.FitProjection(f.MatchingPlan, f.MatchingPlanSHA256, samples.Fit, samples.Heldout, h.c.Protocol.Limits.Projection)
		if err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		measured, err := h.evaluate(ctx, f, samples, projection)
		if err != nil || measured != report {
			return errors.Join(ErrContract, err)
		}
		fsha, _ := fusioncache.Digest(f)
		prior := ""
		if i > 0 {
			prior = s.artifacts[i-1].SHA256
		}
		e := Evidence{Version: 1, Identity: id, StageID: c.Stage.ID, Step: i + 1, ProtocolSHA256: h.c.ProtocolSHA256, FitSHA256: fsha, ParentSHA256: c.Parent.SHA256, PreviousSHA256: prior, Source: f.Source, Projection: projection, Report: report}
		body, err := boundedJSON(e, h.c.Protocol.Limits.MaxArtifactBytes)
		if err != nil {
			return err
		}
		name := h.name(i + 1)
		if err := writeAtomic(ctx, root, name, body); err != nil {
			return err
		}
		s.generations = append(s.generations, e.Identity.Generation)
		s.artifacts = append(s.artifacts, pipeline.StageArtifact{Path: name, SHA256: bodySHA(body), Bytes: int64(len(body))})
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := commit(ctx, s.result(i+1, len(h.c.Protocol.Fits))); err != nil {
			return err
		}
	}
	return ctx.Err()
}
