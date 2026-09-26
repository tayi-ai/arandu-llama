package teachers

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"

	"github.com/tayi-ai/arandu-llama/training/fusioncache"
	"github.com/tayi-ai/arandu-llama/training/pipeline"
)

// CacheStagePlan freezes bounded cache units and their external qualifications.
// Each unit is an indivisible recovery boundary. Splitting a teacher into units
// requires a separately configured factory for each exact expectation.
type CacheStagePlan struct {
	Version             int
	RecipeSHA256        string
	QualificationSHA256 string
	Placement           pipeline.Placement
	StageID             string
	Expectations        []fusioncache.Expectation
	Limits              fusioncache.Limits
}

// CacheStageConfig binds a frozen plan to installed, synchronous producers.
// MaxPlanBytes bounds the complete serialized plan before its owned snapshot.
// Factories must remain dedicated to this handler; admission launches no model.
// Run calls on this handler are serialized. Cross-process placement admission
// remains the responsibility of the caller's existing execution lease.
type CacheStageConfig struct {
	Plan         CacheStagePlan
	PlanSHA256   string
	MaxPlanBytes int64
	Factories    []fusioncache.Factory
}

// CacheStage materializes teacher-forced data through the durable phase runtime.
// It never updates the student and cannot complete other scientific phases.
type CacheStage struct {
	plan      CacheStagePlan
	factories []fusioncache.Factory
	runSlot   chan struct{}
}

var _ pipeline.StageHandler = (*CacheStage)(nil)

// NewCacheStage snapshots a pinned plan and checks all cache contracts first.
func NewCacheStage(config CacheStageConfig) (*CacheStage, error) {
	if config.MaxPlanBytes < 1 || config.MaxPlanBytes > 64<<20 || !validDigest(config.PlanSHA256) ||
		config.Plan.Limits.MaxBytes < 1 || config.Plan.Limits.MaxBytes > 1<<30 || len(config.Factories) == 0 || len(config.Factories) > 4096 || len(config.Factories) != len(config.Plan.Expectations) {
		return nil, rejected("invalid cache stage bounds")
	}
	p := config.Plan
	if p.Version != 1 || len(p.StageID) < 1 || len(p.StageID) > 128 || !validDigest(p.RecipeSHA256) || !validDigest(p.QualificationSHA256) ||
		p.Placement.QualificationSHA256 != p.QualificationSHA256 || !validDigest(p.Placement.RuntimeSHA256) ||
		len(p.Placement.Backend) < 1 || len(p.Placement.Backend) > 128 || len(p.Placement.Nodes) > 4096 || p.Placement.MemoryBytes <= 0 || p.Placement.MaxTokens < 2 {
		return nil, rejected("cache stage identity missing")
	}
	for _, node := range p.Placement.Nodes {
		if len(node) < 1 || len(node) > 128 {
			return nil, rejected("cache stage node identity exceeds bounds")
		}
	}
	// Validate bounded collections before encoding their snapshot.
	remaining := config.MaxPlanBytes
	for _, expected := range config.Plan.Expectations {
		if !rawFits(expected, NativeConfig{}, remaining) {
			return nil, rejected("cache stage input budget exceeded")
		}
		if err := fusioncache.ValidateExpectation(expected, config.Plan.Limits); err != nil {
			return nil, err
		}
		body, err := json.Marshal(expected)
		if err != nil || int64(len(body)) > remaining {
			return nil, rejected("cache stage encoding budget exceeded")
		}
		remaining -= int64(len(body))
	}
	body, err := json.Marshal(config.Plan)
	if err != nil || int64(len(body)) > config.MaxPlanBytes {
		return nil, rejected("cache stage plan budget exceeded")
	}
	sum := sha256.Sum256(body)
	if hex.EncodeToString(sum[:]) != config.PlanSHA256 {
		return nil, rejected("cache stage plan digest differs")
	}
	var plan CacheStagePlan
	if err := json.Unmarshal(body, &plan); err != nil {
		return nil, err
	}
	if plan.Version != 1 || plan.StageID == "" || !validDigest(plan.RecipeSHA256) || !validDigest(plan.QualificationSHA256) ||
		plan.Placement.QualificationSHA256 != plan.QualificationSHA256 || !validDigest(plan.Placement.RuntimeSHA256) ||
		plan.Placement.Backend == "" || plan.Placement.MemoryBytes <= 0 || plan.Placement.MaxTokens < 2 {
		return nil, rejected("cache stage identity missing")
	}
	for _, factory := range config.Factories {
		if factory == nil {
			return nil, rejected("cache stage factory missing")
		}
		value := reflect.ValueOf(factory)
		switch value.Kind() {
		case reflect.Pointer, reflect.Func, reflect.Chan, reflect.Interface, reflect.Map, reflect.Slice:
			if value.IsNil() {
				return nil, rejected("cache stage factory missing")
			}
		}
	}
	return &CacheStage{plan: plan, factories: append([]fusioncache.Factory(nil), config.Factories...), runSlot: make(chan struct{}, 1)}, nil
}

// Admit checks the exact method, data role, unit budget and qualified placement.
func (s *CacheStage) Admit(ctx context.Context, recipe pipeline.Recipe, stage pipeline.Stage, placement pipeline.Placement) error {
	if ctx == nil || s == nil || s.runSlot == nil {
		return rejected("cache stage context required")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	digest, err := recipe.Digest()
	if err != nil || digest != s.plan.RecipeSHA256 || !reflect.DeepEqual(placement, s.plan.Placement) ||
		stage.ID != s.plan.StageID || stage.Phase != pipeline.PhaseTeacherCache || stage.MaxSteps != len(s.plan.Expectations) {
		return rejected("cache stage method or qualification differs")
	}
	matchedStage := false
	for _, declared := range recipe.Stages {
		if reflect.DeepEqual(stage, declared) {
			matchedStage = true
		}
	}
	if !matchedStage {
		return rejected("cache stage not declared by recipe")
	}
	teachers := map[string]bool{}
	for _, ref := range recipe.Teachers {
		teachers[ref.SHA256] = true
	}
	observed := map[string]bool{}
	seenExamples := map[string]bool{}
	inputs := map[string]bool{}
	for _, ref := range stage.Inputs {
		inputs[ref.SHA256] = true
	}
	for _, expected := range s.plan.Expectations {
		if !teachers[expected.Teacher.WeightsSHA256] || !inputs[expected.Teacher.WeightsSHA256] || !inputs[recipe.Student.SHA256] || expected.Student.WeightsSHA256 != recipe.Student.SHA256 {
			return rejected("cache stage model differs from recipe")
		}
		observed[expected.Teacher.WeightsSHA256] = true
		for _, row := range expected.Examples {
			key := expected.Teacher.WeightsSHA256 + "/" + row.DatasetSHA256 + "/" + row.ID
			if seenExamples[key] {
				return rejected("cache stage duplicates a teacher example")
			}
			seenExamples[key] = true
			if row.Role != "train" || row.DatasetSHA256 != recipe.Training.SHA256 || len(row.TeacherTokens) > stage.MaxTokens || len(row.StudentTokens) > placement.MaxTokens {
				return rejected("cache stage dataset role or token budget differs")
			}
		}
	}
	if len(observed) != len(teachers) {
		return rejected("cache stage omits a declared teacher")
	}
	return nil
}

func cacheUnitPath(unit int) string { return fmt.Sprintf("cache-%06d.json", unit+1) }

func (s *CacheStage) readUnit(ctx context.Context, directory string, unit int) (pipeline.StageArtifact, error) {
	if err := ctx.Err(); err != nil {
		return pipeline.StageArtifact{}, err
	}
	path := filepath.Join(directory, cacheUnitPath(unit))
	info, err := os.Lstat(path)
	if err != nil {
		return pipeline.StageArtifact{}, err
	}
	if !info.Mode().IsRegular() || info.Size() < 1 || info.Size() > s.plan.Limits.MaxBytes {
		return pipeline.StageArtifact{}, rejected("cache checkpoint size or type differs")
	}
	file, err := os.Open(path)
	if err != nil {
		return pipeline.StageArtifact{}, err
	}
	hash := sha256.New()
	count, copyErr := io.Copy(hash, io.LimitReader(file, s.plan.Limits.MaxBytes+1))
	if err := errors.Join(copyErr, file.Close()); err != nil {
		return pipeline.StageArtifact{}, err
	}
	if count != info.Size() {
		return pipeline.StageArtifact{}, rejected("cache checkpoint changed during read")
	}
	digest := hex.EncodeToString(hash.Sum(nil))
	if _, err := fusioncache.Read(path, digest, s.plan.Expectations[unit], s.plan.Limits); err != nil {
		return pipeline.StageArtifact{}, err
	}
	return pipeline.StageArtifact{Path: cacheUnitPath(unit), SHA256: digest, Bytes: count}, ctx.Err()
}

// Verify validates the complete unit against its independent frozen expectation.
func (s *CacheStage) Verify(ctx context.Context, stage pipeline.StageContext, receipt pipeline.StageReceipt) error {
	if err := s.Admit(ctx, stage.Execution.Recipe, stage.Stage, stage.Execution.Placement); err != nil {
		return err
	}
	step := receipt.Result.Steps
	if receipt.StageID != stage.Stage.ID || step < 1 || step > int64(len(s.plan.Expectations)) || receipt.Result.Complete != (step == int64(len(s.plan.Expectations))) {
		return rejected("cache stage checkpoint boundary differs")
	}
	actual, err := s.result(ctx, stage.ArtifactDirectory, int(step-1))
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(actual, receipt.Result) {
		return rejected("cache stage artifact receipt differs")
	}
	return nil
}

func (s *CacheStage) result(ctx context.Context, directory string, unit int) (pipeline.StageResult, error) {
	result := pipeline.StageResult{Steps: int64(unit + 1), Complete: unit+1 == len(s.plan.Expectations)}
	first := unit
	if result.Complete {
		first = 0
	}
	for index := first; index <= unit; index++ {
		artifact, err := s.readUnit(ctx, directory, index)
		if err != nil {
			return pipeline.StageResult{}, err
		}
		result.Artifacts = append(result.Artifacts, artifact)
	}
	return result, nil
}

func syncCacheDirectory(directory string, artifacts []pipeline.StageArtifact) error {
	for _, artifact := range artifacts {
		file, err := os.Open(filepath.Join(directory, artifact.Path))
		if err != nil {
			return err
		}
		if err := errors.Join(file.Sync(), file.Close()); err != nil {
			return err
		}
	}
	dir, err := os.Open(directory)
	if err != nil {
		return err
	}
	return errors.Join(dir.Sync(), dir.Close())
}

func (s *CacheStage) position(ctx context.Context, stage pipeline.StageContext) (int, error) {
	if err := s.Admit(ctx, stage.Execution.Recipe, stage.Stage, stage.Execution.Placement); err != nil {
		return 0, err
	}
	if !filepath.IsAbs(stage.ArtifactDirectory) {
		return 0, rejected("cache stage artifact directory must be absolute")
	}
	if stage.Previous == nil {
		return 0, nil
	}
	if err := s.Verify(ctx, stage, *stage.Previous); err != nil {
		return 0, err
	}
	return int(stage.Previous.Result.Steps), nil
}

// Reconcile recovers only contiguous, already-materialized units after a crash.
// No model is opened. A gap before a later file is refused rather than replayed.
func (s *CacheStage) Reconcile(ctx context.Context, stage pipeline.StageContext, commit func(context.Context, pipeline.StageResult) error) error {
	position, err := s.position(ctx, stage)
	if err != nil {
		return err
	}
	if commit == nil {
		return rejected("cache stage commit required")
	}
	gap := false
	for unit := position; unit < len(s.plan.Expectations); unit++ {
		_, err := s.readUnit(ctx, stage.ArtifactDirectory, unit)
		if errors.Is(err, os.ErrNotExist) {
			gap = true
			continue
		}
		if err != nil {
			return err
		}
		if gap {
			return rejected("cache stage checkpoint gap")
		}
		result, err := s.result(ctx, stage.ArtifactDirectory, unit)
		if err != nil {
			return err
		}
		// A crash may have happened after link but before directory fsync.
		// Recover durability before making the application receipt durable.
		if err := syncCacheDirectory(stage.ArtifactDirectory, result.Artifacts); err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := commit(ctx, result); err != nil {
			return err
		}
	}
	return nil
}

type fixedCacheSink struct{ path string }

func (s fixedCacheSink) Store(ctx context.Context, cache fusioncache.Cache, expected fusioncache.Expectation, limits fusioncache.Limits) (fusioncache.Receipt, error) {
	if err := ctx.Err(); err != nil {
		return fusioncache.Receipt{}, err
	}
	return fusioncache.WriteAtomic(s.path, cache, expected, limits)
}

// Run produces each missing unit, closes its teacher, then commits its verified file.
// A failed commit stops immediately; Reconcile recovers the file on redelivery.
func (s *CacheStage) Run(ctx context.Context, stage pipeline.StageContext, commit func(context.Context, pipeline.StageResult) error) error {
	if ctx == nil || s == nil || s.runSlot == nil {
		return rejected("cache stage context required")
	}
	select {
	case s.runSlot <- struct{}{}:
		defer func() { <-s.runSlot }()
	case <-ctx.Done():
		return ctx.Err()
	}
	position, err := s.position(ctx, stage)
	if err != nil {
		return err
	}
	if commit == nil {
		return rejected("cache stage commit required")
	}
	for unit := position; unit < len(s.plan.Expectations); unit++ {
		path := filepath.Join(stage.ArtifactDirectory, cacheUnitPath(unit))
		if _, err := os.Lstat(path); !errors.Is(err, os.ErrNotExist) {
			return rejected("cache stage requires reconciliation before production")
		}
		_, err := fusioncache.ProduceSequential(ctx, []fusioncache.Expectation{s.plan.Expectations[unit]}, s.factories[unit], fixedCacheSink{path}, s.plan.Limits)
		if err != nil {
			return err
		}
		result, err := s.result(ctx, stage.ArtifactDirectory, unit)
		if err != nil {
			return err
		}
		if err := commit(ctx, result); err != nil {
			return err
		}
	}
	return nil
}
