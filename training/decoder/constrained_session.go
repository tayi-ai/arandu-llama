package decoder

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"sync"

	"github.com/tayi-ai/arandu-llama/training/alignment"
	"github.com/tayi-ai/arandu-llama/training/fusioncache"
	"github.com/tayi-ai/arandu-llama/training/pipeline"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

// ErrConstrainedSession reports refused native session wiring or source identity.
var ErrConstrainedSession = errors.New("decoder: constrained session refused")

// ConstrainedCache selects one exact source from the constrained update. Its
// expectation, weights and fitted projections are independently pinned by the
// factory configuration; paths arrive only through the validated stage request.
type ConstrainedCache struct {
	SourceIndex int
	Expectation fusioncache.Expectation
	Weight      float64
	Projections []pipeline.ProjectionInput
	// BoundProjections selects predecessor outputs in protocol version two.
	BoundProjections []ConstrainedProjection `json:"BoundProjections,omitempty"`
}

// ConstrainedProjection fixes matching semantics before fitted coefficients
// exist. SourceIndex selects an alignment-evidence-v2 predecessor artifact.
type ConstrainedProjection struct {
	SourceIndex int
	Plan        fusioncache.MatchingPlan
	PlanSHA256  string
	Weight      float64
	Limits      fusioncache.ProjectionLimits
	MaxBytes    int64
}

// ConstrainedRecovery selects one exact tokenized batch independently of teacher
// signals. Expectation identifies the original Recovery dataset and gold targets.
type ConstrainedRecovery struct {
	SourceIndex int
	Expectation RecoveryExpectation
	Limits      RecoveryLimits
}

// ConstrainedSignals fixes the full teacher and feature objective for one step.
// Other declared sources may hold qualification evidence; every source is hashed.
type ConstrainedSignals struct {
	Admission pipeline.SignalAdmission
	Caches    []ConstrainedCache
	Limits    pipeline.SignalLimits
}

// ConstrainedSessionRecipe pins one concrete native fusion or Master recovery
// stage. Version one uses existing fixed artifacts. Version two binds earlier
// outputs at execution, so construction does not require their bytes to exist.
// Initial supplies only restore budgets in version two; the actual checkpoint
// comes from the verified predecessor. Optimizer moments are never restored.
// MaxConfigBytes bounds the encoded configuration snapshot; resident native
// payload plus explicit copy/cache budgets must fit Placement.MemoryBytes.
// This check is not a peak-memory measurement: native workspaces and allocator
// overhead still require independent qualification for this exact placement.
type ConstrainedSessionRecipe struct {
	Recipe         pipeline.Recipe
	Placement      pipeline.Placement
	Protocol       pipeline.ConstrainedProtocol
	ProtocolSHA256 string
	Admission      TrainingAdmission
	Initializer    InitialAdapterSpec
	Initial        AdapterCheckpoint
	Rotary         RotarySpec
	Limits         Limits
	Signals        []ConstrainedSignals
	Recovery       []ConstrainedRecovery `json:"Recovery,omitempty"`
	MaxConfigBytes int64
	// StepWorkingBytes reserves explicit solver/gradient/logit payload separately
	// from activation checkpoints. Admission checks a conservative minimum for
	// every update; the caller must additionally qualify native peak memory.
	StepWorkingBytes int64
}

// ConstrainedFactoryConfig supplies the immutable assembly and its explicit I/O
// provider. SHA256 pins Recipe plus all private AssemblyPlan metadata. Shards
// performs only byte transport: no scientific callback or model discovery exists.
type ConstrainedFactoryConfig struct {
	Assembly *AssemblyPlan
	Shards   ShardProvider
	Recipe   ConstrainedSessionRecipe
	SHA256   string
}

type constrainedFactoryDocument struct {
	Recipe   ConstrainedSessionRecipe
	Geometry TextGeometry
	Limits   AssemblyLimits
	Summary  AssemblySummary
	Tensors  []AssemblyTensor
	LocalMPS bool
}

// Digest hashes the canonical typed configuration, including assembly placement.
// The provider implementation is not serialized; the loader verifies its tensor
// bytes against this exact assembly. A digest is identity, not qualification.
func (c ConstrainedFactoryConfig) Digest() (string, error) {
	if err := c.validate(); err != nil {
		return "", err
	}
	body, err := c.document()
	if err != nil {
		return "", err
	}
	return assemblyHash(body), nil
}

func (c ConstrainedFactoryConfig) document() ([]byte, error) {
	p := c.Assembly
	body, err := json.Marshal(constrainedFactoryDocument{Recipe: c.Recipe, Geometry: p.Geometry(), Limits: p.limits, Summary: p.Summary(), Tensors: p.Tensors(), LocalMPS: p.localMPS})
	if err != nil || int64(len(body)) > c.Recipe.MaxConfigBytes {
		return nil, errors.Join(ErrConstrainedSession, err)
	}
	return body, nil
}

func (c ConstrainedFactoryConfig) validate() error {
	p, r := c.Assembly, c.Recipe
	if p == nil || c.Shards == nil || r.MaxConfigBytes < 1 || r.MaxConfigBytes > 64<<20 ||
		r.Limits.MaxTokens < 2 || r.Limits.MaxTokens > 1<<20 || r.Limits.LogitRows < 1 || r.Limits.LogitRows > r.Limits.MaxTokens ||
		r.Limits.MaxCheckpointBytes < 1 || r.Limits.MaxCheckpointBytes > r.Placement.MemoryBytes || r.Limits.MaxTokens > int64(r.Placement.MaxTokens) ||
		r.Admission.Assembly != p.summary.Identity || r.Admission.Student.Vocabulary != int(p.geometry.Vocab) || r.Admission.Student.WeightsSHA256 != r.Recipe.Student.SHA256 ||
		r.Initializer.ExpectedSHA256 != p.summary.Identity.InitialAdapterSHA256 || r.Initial.MaxBytes < 1 || r.Initial.MaxWorkingBytes < 1 ||
		r.Rotary.Theta != p.geometry.RoPE.Theta || r.Rotary.Dimension != int(float64(p.geometry.Dimension)*p.geometry.RoPE.Partial) ||
		r.Rotary.MaxTokens < int(r.Limits.MaxTokens) || !validAssemblyHash(r.Rotary.ExpectedSHA256) || p.limits.Sequence.MaxTokens < r.Limits.MaxTokens ||
		len(p.adapters) != len(r.Protocol.Layout) {
		return ErrConstrainedSession
	}
	switch v := reflect.ValueOf(c.Shards); v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		if v.IsNil() {
			return ErrConstrainedSession
		}
	}
	protocolSHA, err := r.Protocol.Digest()
	if err != nil || protocolSHA != r.ProtocolSHA256 {
		return errors.Join(ErrConstrainedSession, err)
	}
	if r.Protocol.Version == 1 {
		if r.Initial.ParametersSHA256 != r.Protocol.InitialParametersSHA256 || !validAssemblyHash(r.Initial.FileSHA256) || r.Initial.Path == "" {
			return ErrConstrainedSession
		}
	} else if r.Initial.Path != "" || r.Initial.FileSHA256 != "" || r.Initial.ParametersSHA256 != "" || r.Initial.MaxBytes < r.Protocol.InitialSource.ByteLimit() {
		return ErrConstrainedSession
	}
	if err := c.memory(); err != nil {
		return err
	}
	recipeSHA, err := r.Recipe.Digest()
	placement, marshalErr := json.Marshal(r.Placement)
	if err != nil || marshalErr != nil || recipeSHA != r.Protocol.RecipeSHA256 || assemblyHash(placement) != r.Protocol.PlacementSHA256 || r.Protocol.FactoryQualificationSHA256 != r.Placement.QualificationSHA256 {
		return ErrConstrainedSession
	}
	var stage pipeline.Stage
	for _, value := range r.Recipe.Stages {
		if value.ID == r.Protocol.StageID {
			stage = value
		}
	}
	if stage.ID == "" || stage.MaxSteps != len(r.Protocol.Updates) || int64(stage.MaxTokens) > r.Limits.MaxTokens {
		return ErrConstrainedSession
	}
	switch stage.Phase {
	case pipeline.PhaseFusion:
		if len(r.Signals) != stage.MaxSteps || len(r.Recovery) != 0 {
			return ErrConstrainedSession
		}
	case pipeline.PhaseRecovery:
		if r.Protocol.Version != 2 || len(r.Recovery) != stage.MaxSteps || len(r.Signals) != 0 {
			return ErrConstrainedSession
		}
	default:
		return ErrConstrainedSession
	}
	if r.Protocol.Version == 2 {
		if err := r.Protocol.InitialSource.Validate(r.Recipe, stage, r.Protocol.Limits.MaxCheckpointBytes); err != nil {
			return errors.Join(ErrConstrainedSession, err)
		}
		wanted := pipeline.PhaseSFT
		if stage.Phase == pipeline.PhaseRecovery {
			wanted = pipeline.PhaseFusion
		}
		found := false
		for _, earlier := range r.Recipe.Stages {
			if earlier.ID == r.Protocol.InitialSource.Predecessor.StageID && earlier.Phase == wanted {
				found = true
			}
		}
		if !found {
			return ErrConstrainedSession
		}
		for _, update := range r.Protocol.Updates {
			for _, source := range update.Sources {
				if err := source.Binding.Validate(r.Recipe, stage, r.Protocol.Limits.MaxSourceBytes); err != nil {
					return errors.Join(ErrConstrainedSession, err)
				}
			}
		}
	}
	data := r.Recipe.Training
	teachers := make(map[string]bool, len(r.Recipe.Teachers))
	for _, teacher := range r.Recipe.Teachers {
		if teachers[teacher.SHA256] {
			if stage.Phase == pipeline.PhaseFusion {
				return ErrConstrainedSession
			}
		}
		teachers[teacher.SHA256] = true
	}
	var elements int64
	for i, adapter := range p.adapters {
		spec := r.Protocol.Layout[i]
		if adapter.ReferenceName != spec.Name || len(adapter.Shape) != len(spec.Shape) || !adapter.Trainable || adapter.DType != torch.Float32 {
			return ErrConstrainedSession
		}
		for axis, d := range adapter.Shape {
			if d < 1 || uint64(d) != spec.Shape[axis] {
				return ErrConstrainedSession
			}
		}
		elements += adapter.Bytes / 4
	}
	if elements != p.summary.AdapterElements || elements < 1 || elements > int64(r.Protocol.Limits.MaxParameters) {
		return ErrConstrainedSession
	}
	if len(r.Initializer.Projections)*2 != len(p.adapters) {
		return ErrConstrainedSession
	}
	for i, spec := range r.Initializer.Projections {
		a, b := p.adapters[i*2], p.adapters[i*2+1]
		if a.ReferenceName != spec.Name+".lora_A.default.weight" || b.ReferenceName != spec.Name+".lora_B.default.weight" || !sameShape(a.Shape, []int64{spec.Rank, spec.Input}) || !sameShape(b.Shape, []int64{spec.Output, spec.Rank}) {
			return ErrConstrainedSession
		}
	}
	for i, signals := range r.Signals {
		a := signals.Admission
		if a.Student != r.Admission.Student || a.Role != "train" || a.DatasetID != data.ID || a.Training.DatasetDigest != data.SHA256 ||
			len(a.Training.Tokens) < 2 || int64(len(a.Training.Tokens)) > r.Limits.MaxTokens || a.Training.PromptTokens < 1 || a.Training.PromptTokens >= len(a.Training.Tokens) ||
			int64(len(a.Training.Tokens)-a.Training.PromptTokens+1) > r.Limits.LogitRows || len(signals.Caches) < 1 || len(signals.Caches) > 8 || len(signals.Caches) != len(teachers) || signals.Limits.MaxBytes < 1 || signals.Limits.MaxBytes > r.Placement.MemoryBytes {
			return ErrConstrainedSession
		}
		seen := map[int]bool{}
		seenTeachers := map[string]bool{}
		boundCount := 0
		for _, cache := range signals.Caches {
			teacher := cache.Expectation.Teacher.WeightsSHA256
			if cache.SourceIndex < 0 || cache.SourceIndex >= len(r.Protocol.Updates[i].Sources) || seen[cache.SourceIndex] || cache.Expectation.Student != a.Student || !teachers[teacher] || seenTeachers[teacher] {
				return ErrConstrainedSession
			}
			seen[cache.SourceIndex] = true
			seenTeachers[teacher] = true
			if r.Protocol.Version == 1 {
				if len(cache.BoundProjections) != 0 {
					return ErrConstrainedSession
				}
			} else {
				if len(cache.Projections) != 0 || len(cache.BoundProjections) > signals.Limits.MaxFeatures {
					return ErrConstrainedSession
				}
				for _, projection := range cache.BoundProjections {
					if err := c.projection(projection, cache, i); err != nil {
						return err
					}
					boundCount++
				}
			}
		}
		if r.Protocol.Version == 2 && boundCount == 0 {
			return ErrConstrainedSession
		}
	}
	for i, recovery := range r.Recovery {
		if recovery.Expectation.validate(recovery.Limits) != nil || recovery.Expectation.Dataset != r.Recipe.Recovery || recovery.Expectation.Student != r.Admission.Student ||
			int64(recovery.Limits.MaxTokens) > r.Limits.MaxTokens || recovery.SourceIndex < 0 || recovery.SourceIndex >= len(r.Protocol.Updates[i].Sources) {
			return ErrConstrainedSession
		}
		source := r.Protocol.Updates[i].Sources[recovery.SourceIndex].Binding
		if source.ByteLimit() > recovery.Limits.MaxBytes || (source.Kind == pipeline.BindingDerivedInput && source.Input != r.Recipe.Recovery) ||
			(source.Kind == pipeline.BindingPredecessor && source.Predecessor.Schema != "recovery-batch-v1") {
			return ErrConstrainedSession
		}
	}
	for _, update := range r.Protocol.Updates {
		for _, pair := range update.Config.Protection {
			for _, row := range []pipeline.Example{pair.Positive, pair.Negative} {
				if row.DatasetDigest != r.Recipe.Protection.SHA256 || int64(len(row.Tokens)) > r.Limits.MaxTokens || int64(len(row.Tokens)-row.PromptTokens+1) > r.Limits.LogitRows {
					return ErrConstrainedSession
				}
			}
		}
	}
	return nil
}

func (c ConstrainedFactoryConfig) projection(p ConstrainedProjection, cache ConstrainedCache, step int) error {
	r := c.Recipe
	if p.SourceIndex < 0 || p.SourceIndex >= len(r.Protocol.Updates[step].Sources) || !finiteBackend(p.Weight) || p.Weight <= 0 ||
		p.MaxBytes < 1 || p.MaxBytes > r.Signals[step].Limits.MaxBytes || p.Limits.MaxSamples < 1 || p.Limits.MaxDimension < 1 || p.Limits.MaxElements < 1 ||
		p.Plan.SourceModel != cache.Expectation.Teacher || p.Plan.TargetModel != r.Admission.Student || p.Plan.TokenMappingSHA256 != cache.Expectation.MappingSHA256 ||
		p.Plan.Target.Tensor != "decoder_output" || p.Plan.Target.Dimension != int(c.Assembly.geometry.Hidden) || p.Plan.Target.Layer < 0 || p.Plan.Target.Layer >= int(c.Assembly.geometry.Layers) {
		return ErrConstrainedSession
	}
	digest, err := fusioncache.Digest(p.Plan)
	if err != nil || digest != p.PlanSHA256 {
		return errors.Join(ErrConstrainedSession, err)
	}
	binding := r.Protocol.Updates[step].Sources[p.SourceIndex].Binding
	if binding.Kind != pipeline.BindingPredecessor || binding.Predecessor.Schema != "alignment-evidence-v2" || binding.ByteLimit() > p.MaxBytes {
		return ErrConstrainedSession
	}
	for _, stage := range r.Recipe.Stages {
		if stage.ID == binding.Predecessor.StageID && stage.Phase == pipeline.PhaseAlignment {
			return nil
		}
	}
	return ErrConstrainedSession
}

// memory accounts for declared payload/copy budgets without overflow. The sum
// deliberately includes buffers from different phases, rather than assuming an
// allocator has returned them. Native peak qualification remains separate.
func (c ConstrainedFactoryConfig) memory() error {
	p, r := c.Assembly, c.Recipe
	remaining := r.Placement.MemoryBytes
	consume := func(n int64) bool {
		if n < 1 || n > remaining {
			return false
		}
		remaining -= n
		return true
	}
	if p.localMPS {
		if !consume(p.summary.LocalMPSBytes) {
			return ErrConstrainedSession
		}
	} else {
		if len(p.summary.PersistentBytes) == 0 {
			return ErrConstrainedSession
		}
		for _, n := range p.summary.PersistentBytes {
			if n < 0 || (n > 0 && !consume(n)) {
				return ErrConstrainedSession
			}
		}
		if remaining == r.Placement.MemoryBytes {
			return ErrConstrainedSession
		}
	}
	var signalBytes int64
	for _, signals := range r.Signals {
		budget := r.Placement.MemoryBytes
		if signals.Limits.MaxBytes < 1 || signals.Limits.MaxBytes > budget {
			return ErrConstrainedSession
		}
		budget -= signals.Limits.MaxBytes
		for _, cache := range signals.Caches {
			for _, projection := range cache.BoundProjections {
				if projection.MaxBytes < 1 || projection.MaxBytes > budget {
					return ErrConstrainedSession
				}
				budget -= projection.MaxBytes
				if projection.Limits.MaxElements < 1 || projection.Limits.MaxElements > budget/8 {
					return ErrConstrainedSession
				}
				budget -= projection.Limits.MaxElements * 8
			}
		}
		signalBytes = max(signalBytes, r.Placement.MemoryBytes-budget)
	}
	for _, batch := range r.Recovery {
		if !validRecoveryLimits(batch.Limits) {
			return ErrConstrainedSession
		}
		signalBytes = max(signalBytes, batch.Limits.WorkingBytes)
	}
	for _, update := range r.Protocol.Updates {
		working := r.StepWorkingBytes
		reserve := func(factors ...int64) bool {
			n := int64(1)
			for _, factor := range factors {
				if factor < 1 || n > working/factor {
					return false
				}
				n *= factor
			}
			working -= n
			return true
		}
		// Two FP64 Jacobian matrices cover returned rows plus defensive copies.
		// Thirty-two parameter vectors and eight constraint vectors reserve the
		// solver, candidate, parameter snapshots and gradient copies; four FP32
		// logit buffers cover forward values, copies, seeds and native cotangents.
		// These are explicit payload reservations, not native kernel/RSS bounds.
		n, m := p.summary.AdapterElements, int64(len(update.Config.Protection))
		if working < 1 || !reserve(n, m, 16) || !reserve(n, 32, 8) || !reserve(m, 8, 8) || !reserve(r.Limits.LogitRows, p.geometry.Vocab, 16) {
			return ErrConstrainedSession
		}
	}
	for _, n := range []int64{p.limits.TensorCopyBytes, r.Initial.MaxWorkingBytes, r.Limits.MaxCheckpointBytes, signalBytes, r.MaxConfigBytes, r.StepWorkingBytes} {
		if !consume(n) {
			return ErrConstrainedSession
		}
	}
	return nil
}

// NewConstrainedFactory wires the concrete loader, adapter restoration, admitted
// signals, rotary preparation and TrainingBackend. Each invocation owns its
// resources; Close releases them and the cancelable single-session gate. Failed
// cleanup poisons this factory rather than allowing another native allocation.
// Prior vector hashes are verified by ConstrainedStage; this factory validates
// exact geometry/finitude and installs that vector with ReplaceParameters.
// Fusion and version-two Master recovery have separate objective constructors.
// Variant recovery is refused until its quantized native backend is qualified.
func NewConstrainedFactory(config ConstrainedFactoryConfig) (pipeline.ConstrainedFactory, error) {
	return newConstrainedFactory(config, LoadTextAssembly)
}

type constrainedAssemblyLoader func(context.Context, *AssemblyPlan, ShardProvider, *InitialAdapter) (*LoadedTextModel, error)

func newConstrainedFactory(config ConstrainedFactoryConfig, load constrainedAssemblyLoader) (pipeline.ConstrainedFactory, error) {
	if err := config.validate(); err != nil {
		return nil, err
	}
	body, err := config.document()
	if err != nil || !validAssemblyHash(config.SHA256) || assemblyHash(body) != config.SHA256 {
		return nil, errors.Join(ErrConstrainedSession, err)
	}
	var document constrainedFactoryDocument
	if err := json.Unmarshal(body, &document); err != nil {
		return nil, err
	}
	config.Recipe = document.Recipe
	gate := make(chan struct{}, 1)
	var poisoned error
	return func(ctx context.Context, request pipeline.ConstrainedRequest) (result pipeline.ConstrainedSession, err error) {
		if ctx == nil {
			return result, ErrConstrainedSession
		}
		if err := ctx.Err(); err != nil {
			return result, err
		}
		select {
		case gate <- struct{}{}:
		case <-ctx.Done():
			return result, ctx.Err()
		}
		transferred := false
		defer func() {
			if !transferred {
				<-gate
			}
		}()
		if poisoned != nil {
			return result, errors.Join(ErrConstrainedSession, poisoned)
		}
		if err := config.request(request); err != nil {
			return result, err
		}
		request.Parameters = slices.Clone(request.Parameters)
		request.Sources = slices.Clone(request.Sources)
		for _, source := range request.Sources {
			if err := config.verifySource(ctx, request, source, config.Recipe.Protocol.Limits.MaxSourceBytes); err != nil {
				return result, err
			}
		}
		if config.Recipe.Protocol.Version == 2 {
			if err := config.verifySource(ctx, request, request.Initial.File, config.Recipe.Protocol.Limits.MaxCheckpointBytes); err != nil {
				return result, err
			}
		}
		var signals *pipeline.Signals
		var recovery *RecoveryBatch
		if request.Context.Stage.Phase == pipeline.PhaseRecovery {
			spec := config.Recipe.Recovery[request.Step-1]
			source := request.Sources[spec.SourceIndex]
			recovery, err = ReadRecoveryBatch(ctx, source.Path, source.Artifact.SHA256, source.Artifact.Bytes, spec.Expectation, spec.Limits)
			if err == nil {
				for _, row := range recovery.document.Examples {
					if int64(len(row.Tokens)-row.PromptTokens+1) > config.Recipe.Limits.LogitRows {
						err = ErrConstrainedSession
					}
				}
			}
		} else {
			signals, err = config.signals(ctx, request)
		}
		if err != nil {
			return result, err
		}
		initial, err := InitializeAdapter(ctx, config.Recipe.Initializer)
		if err != nil {
			return result, err
		}
		loaded, loadErr := load(ctx, config.Assembly, config.Shards, initial)
		initialErr := initial.Close()
		if loadErr != nil || initialErr != nil || loaded == nil {
			closeErr := loaded.Close()
			if initialErr != nil || closeErr != nil {
				poisoned = errors.Join(initialErr, closeErr)
			}
			return result, errors.Join(ErrConstrainedSession, loadErr, initialErr, closeErr)
		}
		keep := false
		defer func() {
			if !keep {
				closeErr := loaded.Close()
				err = errors.Join(err, closeErr)
				if closeErr != nil {
					poisoned = closeErr
				}
			}
		}()
		if loaded.Summary.Identity != config.Recipe.Admission.Assembly {
			return result, ErrConstrainedSession
		}
		if request.Step == 1 {
			checkpoint := config.Recipe.Initial
			if config.Recipe.Protocol.Version == 2 {
				checkpoint.Path = request.Initial.File.Path
				checkpoint.FileSHA256 = request.Initial.File.Artifact.SHA256
				checkpoint.ParametersSHA256 = request.Initial.ParametersSHA256
			}
			_, err = loaded.RestoreAdapter(ctx, checkpoint)
		} else {
			_, err = loaded.ReplaceParameters(ctx, request.Parameters)
		}
		if err != nil {
			return result, err
		}
		prepare := func(ctx context.Context, tokens int) (func() error, error) {
			return constrainedRotary(ctx, loaded.Model, tokens, config.Recipe.Rotary)
		}
		var backend pipeline.Model
		if recovery != nil {
			backend, err = NewRecoveryBackend(loaded, recovery, config.Recipe.Limits, prepare, config.Recipe.Admission)
		} else {
			backend, err = NewTrainingBackend(loaded, signals, config.Recipe.Limits, prepare, config.Recipe.Admission)
		}
		if err != nil {
			return result, err
		}
		if err := ctx.Err(); err != nil {
			return result, err
		}
		var once sync.Once
		var closeErr error
		result = pipeline.ConstrainedSession{Model: backend, Close: func() error {
			once.Do(func() {
				closeErr = loaded.Close()
				if closeErr != nil {
					poisoned = closeErr
				}
				<-gate
			})
			return closeErr
		}}
		keep, transferred = true, true
		return result, nil
	}, nil
}

func (c ConstrainedFactoryConfig) request(request pipeline.ConstrainedRequest) error {
	r := c.Recipe
	if request.Step < 1 || request.Step > len(r.Protocol.Updates) || request.Context.Execution.TargetSHA256 != r.Recipe.Student.SHA256 || !reflect.DeepEqual(request.Context.Execution.Recipe, r.Recipe) || !reflect.DeepEqual(request.Context.Execution.Placement, r.Placement) ||
		request.Context.Stage.ID != r.Protocol.StageID || len(request.Sources) != len(r.Protocol.Updates[request.Step-1].Sources) {
		return ErrConstrainedSession
	}
	var exact pipeline.Stage
	for _, stage := range r.Recipe.Stages {
		if stage.ID == r.Protocol.StageID {
			exact = stage
		}
	}
	if !reflect.DeepEqual(exact, request.Context.Stage) {
		return ErrConstrainedSession
	}
	if r.Protocol.Version == 1 {
		if request.Initial != (pipeline.ConstrainedInitialCheckpoint{}) {
			return ErrConstrainedSession
		}
	} else if request.Initial.File.Source != (pipeline.ConstrainedSource{Binding: r.Protocol.InitialSource}) || !validAssemblyHash(request.Initial.ParametersSHA256) || !filepath.IsAbs(request.Initial.File.Path) {
		return ErrConstrainedSession
	}
	if request.Step == 1 && request.Parameters != nil || request.Step > 1 && int64(len(request.Parameters)) != c.Assembly.summary.AdapterElements {
		return ErrConstrainedSession
	}
	for _, v := range request.Parameters {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return ErrConstrainedSession
		}
	}
	for i, source := range request.Sources {
		if source.Source != r.Protocol.Updates[request.Step-1].Sources[i] || !filepath.IsAbs(source.Path) {
			return ErrConstrainedSession
		}
	}
	return nil
}

func (c ConstrainedFactoryConfig) signals(ctx context.Context, request pipeline.ConstrainedRequest) (*pipeline.Signals, error) {
	spec := c.Recipe.Signals[request.Step-1]
	sources := make([]pipeline.CacheInput, 0, len(spec.Caches))
	for _, cache := range spec.Caches {
		source := request.Sources[cache.SourceIndex]
		projections := slices.Clone(cache.Projections)
		for _, spec := range cache.BoundProjections {
			file := request.Sources[spec.SourceIndex]
			projection, err := alignment.ReadBoundProjection(ctx, pipeline.BoundArtifactFile{Binding: file.Binding, Path: file.Path}, spec.Plan, spec.PlanSHA256, spec.Limits, spec.MaxBytes)
			if err != nil {
				return nil, err
			}
			projections = append(projections, pipeline.ProjectionInput{Plan: spec.Plan, PlanSHA256: spec.PlanSHA256, Projection: projection, SHA256: projection.SHA256, Weight: spec.Weight})
		}
		artifact := source.Artifact
		if c.Recipe.Protocol.Version == 1 {
			artifact = source.Source.Artifact
		}
		sources = append(sources, pipeline.CacheInput{Path: source.Path, SHA256: artifact.SHA256, Bytes: artifact.Bytes, Expectation: cache.Expectation, Weight: cache.Weight, Projections: projections})
	}
	signals, err := pipeline.NewSignals(spec.Admission, sources, spec.Limits)
	if err != nil {
		return nil, err
	}
	return signals, ctx.Err()
}

func (c ConstrainedFactoryConfig) verifySource(ctx context.Context, request pipeline.ConstrainedRequest, source pipeline.ConstrainedSourceFile, maxBytes int64) error {
	if c.Recipe.Protocol.Version == 1 {
		if source.Binding != (pipeline.BoundArtifact{}) || (source.Artifact != (pipeline.StageArtifact{}) && source.Artifact != source.Source.Artifact) {
			return ErrConstrainedSession
		}
		return verifyConstrainedSource(ctx, source)
	}
	// Predecessors resolve under the execution artifact root. For a pinned
	// external input, recover the installation root from its already-resolved
	// path and frozen relative path, then let the canonical resolver verify it.
	directory := request.Context.ArtifactDirectory
	if source.Source.Binding.Kind == pipeline.BindingDerivedInput {
		directory = source.Path
		for range strings.Split(filepath.ToSlash(source.Source.Binding.Artifact.Path), "/") {
			directory = filepath.Dir(directory)
		}
	}
	resolved, err := pipeline.ResolveArtifact(ctx, request.Context, directory, source.Source.Binding, maxBytes)
	if err != nil || resolved.Path != source.Path || resolved.Binding != source.Binding || resolved.Binding.Artifact != source.Artifact {
		return errors.Join(ErrConstrainedSession, err)
	}
	return nil
}

func verifyConstrainedSource(ctx context.Context, source pipeline.ConstrainedSourceFile) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	info, err := os.Lstat(source.Path)
	if err != nil || !info.Mode().IsRegular() || info.Size() != source.Source.Artifact.Bytes {
		return errors.Join(ErrConstrainedSession, err)
	}
	f, err := os.Open(source.Path)
	if err != nil {
		return err
	}
	defer f.Close()
	opened, err := f.Stat()
	if err != nil || !os.SameFile(info, opened) {
		return ErrConstrainedSession
	}
	digest := sha256.New()
	buffer := make([]byte, 64<<10)
	var read int64
	for read < info.Size() {
		if err := ctx.Err(); err != nil {
			return err
		}
		size := min(int64(len(buffer)), info.Size()-read)
		n, err := io.ReadFull(f, buffer[:size])
		if err != nil {
			return err
		}
		_, _ = digest.Write(buffer[:n])
		read += int64(n)
	}
	after, err := f.Stat()
	if err != nil || after.Size() != info.Size() || !after.ModTime().Equal(info.ModTime()) || hex.EncodeToString(digest.Sum(nil)) != source.Source.Artifact.SHA256 {
		return errors.Join(ErrConstrainedSession, err)
	}
	return ctx.Err()
}

func constrainedRotary(ctx context.Context, model *TextModel, tokens int, spec RotarySpec) (close func() error, err error) {
	var tables []*Rotary
	var indices []int
	cleanup := func() error {
		for _, index := range indices {
			model.Layers[index].Cosine, model.Layers[index].Sine = nil, nil
		}
		var failures []error
		for _, table := range tables {
			failures = append(failures, table.Close())
		}
		tables = nil
		indices = nil
		return errors.Join(failures...)
	}
	defer func() {
		if err != nil {
			err = errors.Join(err, cleanup())
		}
	}()
	byDevice := map[torch.Device]*Rotary{}
	for i := range model.Layers {
		layer := &model.Layers[i]
		if layer.Weights.Full == nil {
			continue
		}
		if layer.Cosine != nil || layer.Sine != nil {
			return nil, ErrConstrainedSession
		}
		table := byDevice[layer.Device]
		if table == nil {
			table, err = TextRotary(ctx, tokens, layer.Device, spec)
			if err != nil {
				return nil, err
			}
			tables = append(tables, table)
			byDevice[layer.Device] = table
		}
		layer.Cosine, layer.Sine = table.Cosine, table.Sine
		indices = append(indices, i)
	}
	return cleanup, ctx.Err()
}
