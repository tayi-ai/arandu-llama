//go:build libtorch && cgo

package decoder

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/tayi-ai/arandu-llama/checkpoint"
	"github.com/tayi-ai/arandu-llama/training/fusioncache"
	"github.com/tayi-ai/arandu-llama/training/layers"
	"github.com/tayi-ai/arandu-llama/training/pipeline"
	"github.com/tayi-ai/arandu-llama/training/protection"
	"github.com/tayi-ai/arandu-llama/training/sequence"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

type sessionNoShards struct{}

func (sessionNoShards) OpenShard(context.Context, string) (Shard, error) {
	return nil, errors.New("CPU fixture must not open production shards")
}

type sessionNilMap map[string]Shard
type sessionNilSlice []Shard
type sessionNilFunc func()
type sessionNilChan chan Shard

func (sessionNilMap) OpenShard(context.Context, string) (Shard, error)   { panic("nil provider") }
func (sessionNilSlice) OpenShard(context.Context, string) (Shard, error) { panic("nil provider") }
func (sessionNilFunc) OpenShard(context.Context, string) (Shard, error)  { panic("nil provider") }
func (sessionNilChan) OpenShard(context.Context, string) (Shard, error)  { panic("nil provider") }
func sessionJSONHash(v any) string                                       { body, _ := json.Marshal(v); return assemblyHash(body) }
func sessionReceipt(r pipeline.StageReceipt) pipeline.StageReceipt {
	r.SHA256 = ""
	r.SHA256 = sessionJSONHash(r)
	return r
}

func sessionInitializerHash(t *testing.T, s InitialAdapterSpec) string {
	t.Helper()
	g, err := torch.NewCPUGenerator(s.Seed)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	draw := func(shape []int64, low, high float64) []byte {
		v, e := g.Uniform(shape, low, high, torch.Float32)
		if e != nil {
			t.Fatal(e)
		}
		defer v.Close()
		b, e := v.Bytes()
		if e != nil {
			t.Fatal(e)
		}
		return b
	}
	h := sha256.New()
	for _, p := range s.Projections {
		a, b := 1/math.Sqrt(float64(p.Input)), 1/math.Sqrt(float64(p.Rank))
		draw([]int64{p.Rank, p.Input}, -a, a)
		draw([]int64{p.Output, p.Rank}, -b, b)
		h.Write([]byte(p.Name + ".lora_A.default.weight"))
		h.Write(draw([]int64{p.Rank, p.Input}, -a, a))
		h.Write([]byte(p.Name + ".lora_B.default.weight"))
		h.Write(make([]byte, p.Output*p.Rank*4))
	}
	return hex.EncodeToString(h.Sum(nil))
}

func sessionFixture(t *testing.T) (ConstrainedFactoryConfig, pipeline.ConstrainedRequest, constrainedAssemblyLoader, **LoadedTextModel) {
	t.Helper()
	ctx := context.Background()
	root := t.TempDir()
	prefix := "base_model.model.model.language_model.layers.0.self_attn."
	init := InitialAdapterSpec{Seed: 7, Projections: []InitialProjection{{Name: prefix + "q_proj", Input: 4, Output: 8, Rank: 1}, {Name: prefix + "v_proj", Input: 4, Output: 2, Rank: 1}}}
	init.ExpectedSHA256 = sessionInitializerHash(t, init)
	initial, err := InitializeAdapter(ctx, init)
	if err != nil {
		t.Fatal(err)
	}
	defer initial.Close()
	plan := &AssemblyPlan{geometry: TextGeometry{Hidden: 4, Vocab: 6, MLP: 5, Layers: 1, Heads: 2, KVHeads: 1, Dimension: 2, LayerTypes: []string{"full_attention"}, RoPE: RotaryGeometry{Theta: 10000, Partial: 1}},
		limits: AssemblyLimits{Sequence: sequence.SequenceLimits{MaxTokens: 3}, TensorCopyBytes: 4096}, summary: AssemblySummary{PersistentBytes: []int64{4096}, Identity: AssemblyIdentity{IndexSHA256: strings.Repeat("1", 64), ConfigSHA256: strings.Repeat("2", 64), ReferenceSHA256: strings.Repeat("3", 64), InitialAdapterSHA256: init.ExpectedSHA256}}}
	var tensors []checkpoint.Float32Tensor
	var layout []pipeline.ConstrainedTensor
	h := sha256.New()
	for _, p := range initial.Parameters {
		info, e := p.Value.Info()
		if e != nil {
			t.Fatal(e)
		}
		raw, e := p.Value.Float32Values()
		if e != nil {
			t.Fatal(e)
		}
		for i := range raw {
			raw[i] += .03
		}
		shape := []uint64{uint64(info.Shape[0]), uint64(info.Shape[1])}
		tensors = append(tensors, checkpoint.Float32Tensor{Name: p.Name, Shape: shape, Values: raw})
		layout = append(layout, pipeline.ConstrainedTensor{Name: p.Name, Shape: shape})
		plan.adapters = append(plan.adapters, AssemblyTensor{ReferenceName: p.Name, Shape: info.Shape, DType: torch.Float32, Device: torch.CPUDevice(), Bytes: info.Elements * 4, Trainable: true})
		plan.summary.AdapterElements += info.Elements
		h.Write([]byte(p.Name))
		var scalar [4]byte
		for _, v := range raw {
			binary.LittleEndian.PutUint32(scalar[:], math.Float32bits(v))
			h.Write(scalar[:])
		}
	}
	parameterSHA := hex.EncodeToString(h.Sum(nil))
	plan.summary.AdapterTensors = len(plan.adapters)
	var body bytes.Buffer
	limits := checkpoint.Limits{MaxHeaderBytes: 4096, MaxTensors: 8, MaxDimensions: 2, MaxMetadataEntries: 1, MaxChunkBytes: 64}
	write, err := checkpoint.WriteFloat32(ctx, &body, tensors, limits)
	if err != nil {
		t.Fatal(err)
	}
	adapterPath := filepath.Join(root, "initial.safetensors")
	if err := os.WriteFile(adapterPath, body.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	ref := func(name string) pipeline.ArtifactRef {
		return pipeline.ArtifactRef{ID: name, SHA256: assemblyHash([]byte(name))}
	}
	recipe := pipeline.Recipe{Version: 1, ID: "synthetic", Method: "generational-fusion-v1", Student: ref("student"), Teachers: []pipeline.ArtifactRef{ref("teacher")}, Training: ref("train"), Recovery: ref("recovery"), Calibration: ref("calibration"), Protection: ref("protection"), Heldout: ref("heldout")}
	phases := []pipeline.Phase{pipeline.PhaseSFT, pipeline.PhaseTeacherCache, pipeline.PhaseAlignment, pipeline.PhaseFusion, pipeline.PhaseRecovery, pipeline.PhaseMaster, pipeline.PhaseCalibration, pipeline.PhaseQuantize, pipeline.PhaseVariantRecovery, pipeline.PhaseQuantize, pipeline.PhaseVariantRecovery, pipeline.PhaseQuantize, pipeline.PhaseVariantRecovery}
	for i, phase := range phases {
		data := recipe.Training
		switch phase {
		case pipeline.PhaseRecovery, pipeline.PhaseVariantRecovery:
			data = recipe.Recovery
		case pipeline.PhaseMaster:
			data = recipe.Heldout
		case pipeline.PhaseCalibration, pipeline.PhaseQuantize:
			data = recipe.Calibration
		}
		s := pipeline.Stage{ID: fmt.Sprintf("stage-%d", i), Phase: phase, Inputs: []pipeline.ArtifactRef{data}, MaxSteps: 1, MaxTokens: 3, TimeoutSeconds: 60}
		if i > 0 {
			s.ParentStage = fmt.Sprintf("stage-%d", i-1)
		}
		if phase == pipeline.PhaseFusion || phase == pipeline.PhaseRecovery || phase == pipeline.PhaseVariantRecovery {
			s.Inputs = append(s.Inputs, recipe.Protection)
		}
		if i >= 7 {
			s.Format = []string{"Q8_0", "Q6_K", "Q4_K_M"}[(i-7)/2]
		}
		if phase == pipeline.PhaseQuantize {
			s.ParentStage = "stage-5"
		}
		recipe.Stages = append(recipe.Stages, s)
	}
	placement := pipeline.Placement{Backend: "fixture", MemoryBytes: 1 << 25, MaxTokens: 3, RuntimeSHA256: assemblyHash([]byte("runtime")), QualificationSHA256: assemblyHash([]byte("fixture qualification"))}
	rsha, err := recipe.Digest()
	if err != nil {
		t.Fatal(err)
	}
	model := func(name string) fusioncache.ModelIdentity {
		return fusioncache.ModelIdentity{Name: name, Revision: strings.Repeat("a", 40), WeightsSHA256: assemblyHash([]byte(name)), TokenizerSHA256: strings.Repeat("b", 64), TemplateSHA256: strings.Repeat("c", 64), RuntimeSHA256: strings.Repeat("d", 64), Vocabulary: 6}
	}
	student, teacher := model("student"), model("teacher")
	example := fusioncache.Example{DatasetID: recipe.Training.ID, DatasetSHA256: recipe.Training.SHA256, ID: "row", Role: "train", TeacherTokens: []int64{1, 2, 3}, StudentTokens: []int64{1, 2, 3}, PromptTokens: 1}
	mapping := fusioncache.TokenMapping{Identity: true, TeacherTokenizerSHA256: teacher.TokenizerSHA256, StudentTokenizerSHA256: student.TokenizerSHA256, EvidenceSHA256: strings.Repeat("e", 64)}
	mappingSHA, _ := fusioncache.Digest(mapping)
	feature := fusioncache.FeatureSpec{Layer: 0, Tensor: "source_output", DType: "float32", Dimension: 2}
	expectation := fusioncache.Expectation{Teacher: teacher, Student: student, Mapping: mapping, MappingSHA256: mappingSHA, Features: []fusioncache.FeatureSpec{feature}, Examples: []fusioncache.Example{example}}
	record := fusioncache.Record{Example: example}
	for position := 1; position < 3; position++ {
		record.Positions = append(record.Positions, fusioncache.Position{TargetIndex: position, TeacherPrefixSHA256: fusioncache.PrefixDigest(example.TeacherTokens[:position]), StudentPrefixSHA256: fusioncache.PrefixDigest(example.StudentTokens[:position]), RetainedMass: .8, Probabilities: []fusioncache.Probability{{TeacherTokenID: 2, StudentTokenID: 2, Probability: .3}, {TeacherTokenID: 4, StudentTokenID: 4, Probability: .5}}, Features: []fusioncache.Feature{{Spec: feature, Values: []float64{.1, .2}}}})
	}
	cache := fusioncache.Cache{Schema: 1, Mode: "teacher_forced", Teacher: teacher, Student: student, Mapping: mapping, MappingSHA256: mappingSHA, Features: []fusioncache.FeatureSpec{feature}, Records: []fusioncache.Record{record}}
	cacheBody, _ := json.Marshal(cache)
	cachePath := filepath.Join(root, "cache.json")
	if err := os.WriteFile(cachePath, cacheBody, 0600); err != nil {
		t.Fatal(err)
	}
	artifact := pipeline.StageArtifact{Path: "cache.json", SHA256: assemblyHash(cacheBody), Bytes: int64(len(cacheBody))}
	identity := pipeline.ReceiptIdentity{RunID: "run", TenantID: "tenant", RecipeSHA256: rsha, TargetSHA256: recipe.Student.SHA256, PlacementSHA256: sessionJSONHash(placement), Generation: 1}
	teacherReceipt := sessionReceipt(pipeline.StageReceipt{Version: 1, Identity: identity, StageID: "stage-1", Result: pipeline.StageResult{Steps: 1, Complete: true, Artifacts: []pipeline.StageArtifact{artifact}}})
	parent := sessionReceipt(pipeline.StageReceipt{Version: 1, Identity: identity, StageID: "stage-2", Result: pipeline.StageResult{Steps: 1, Complete: true}})
	source := pipeline.ConstrainedSource{Artifact: artifact, StageID: teacherReceipt.StageID, ReceiptSHA256: teacherReceipt.SHA256}
	positive := pipeline.Example{ID: "correct", DatasetDigest: recipe.Protection.SHA256, Tokens: []int64{1, 2}, PromptTokens: 1}
	negative := positive
	negative.ID = "wrong"
	negative.Tokens = []int64{1, 4}
	step := pipeline.StepConfig{Lambda: 100, MinimumGain: 0, Protection: []pipeline.ProtectedPair{{ID: "floor", Positive: positive, Negative: negative, Floor: -100}}, Solver: protection.DefaultConfig()}
	protocol := pipeline.ConstrainedProtocol{Version: 1, RecipeSHA256: rsha, PlacementSHA256: identity.PlacementSHA256, FactoryQualificationSHA256: placement.QualificationSHA256, StageID: "stage-3", InitialParametersSHA256: parameterSHA, Layout: layout, Updates: []pipeline.ConstrainedUpdate{{Config: step, Sources: []pipeline.ConstrainedSource{source}}}, Limits: pipeline.ConstrainedLimits{MaxProtocolBytes: 1 << 20, MaxReceiptBytes: 1 << 20, MaxCheckpointBytes: 1 << 20, MaxTotalBytes: 3 << 20, MaxSourceBytes: 1 << 20, MaxParameters: 100, MaxSteps: 1, Checkpoint: limits}}
	protocolSHA, err := protocol.Digest()
	if err != nil {
		t.Fatal(err)
	}
	sample := func(id string) fusioncache.SampleIdentity {
		return fusioncache.SampleIdentity{DatasetSHA256: recipe.Calibration.SHA256, ExampleID: id, Role: "calibration", TargetIndex: 1, TeacherPrefixSHA256: record.Positions[0].TeacherPrefixSHA256, StudentPrefixSHA256: record.Positions[0].StudentPrefixSHA256}
	}
	match := fusioncache.MatchingPlan{ID: "matching", SourceModel: teacher, TargetModel: student, TokenMappingSHA256: mappingSHA, Source: feature, Target: fusioncache.FeatureSpec{Layer: 0, Tensor: "decoder_output", DType: "float32", Dimension: 4}, Ridge: 1, Fit: []fusioncache.SampleIdentity{sample("fit-a"), sample("fit-b")}, Heldout: []fusioncache.SampleIdentity{sample("heldout")}}
	matchSHA, _ := fusioncache.Digest(match)
	projectionLimits := fusioncache.ProjectionLimits{MaxSamples: 8, MaxDimension: 8, MaxElements: 128}
	projection, _, err := fusioncache.FitProjection(match, matchSHA, []fusioncache.FeatureSample{{Identity: match.Fit[0], Source: []float64{1, 0}, Target: []float64{.1, .2, .3, .4}}, {Identity: match.Fit[1], Source: []float64{0, 1}, Target: []float64{.4, .3, .2, .1}}}, []fusioncache.FeatureSample{{Identity: match.Heldout[0], Source: []float64{1, 1}, Target: []float64{.5, .5, .5, .5}}}, projectionLimits)
	if err != nil {
		t.Fatal(err)
	}
	var one [4]byte
	binary.LittleEndian.PutUint32(one[:], math.Float32bits(1))
	policy := ConstrainedSessionRecipe{Recipe: recipe, Placement: placement, Protocol: protocol, ProtocolSHA256: protocolSHA, Admission: TrainingAdmission{Assembly: plan.summary.Identity, Student: student}, Initializer: init,
		Initial: AdapterCheckpoint{Path: adapterPath, FileSHA256: write.SHA256, ParametersSHA256: parameterSHA, MaxBytes: 1 << 20, MaxWorkingBytes: 4 << 20, Limits: limits}, Rotary: RotarySpec{Theta: 10000, Dimension: 2, MaxTokens: 3, ExpectedSHA256: assemblyHash(one[:])}, Limits: Limits{MaxTokens: 3, LogitRows: 3, MaxCheckpointBytes: 1 << 20}, MaxConfigBytes: 1 << 20, StepWorkingBytes: 1 << 20,
		Signals: []ConstrainedSignals{{Admission: pipeline.SignalAdmission{Student: student, DatasetID: recipe.Training.ID, Role: "train", Training: pipeline.Example{ID: example.ID, DatasetDigest: example.DatasetSHA256, Tokens: example.StudentTokens, PromptTokens: 1}}, Caches: []ConstrainedCache{{SourceIndex: 0, Expectation: expectation, Weight: .5, Projections: []pipeline.ProjectionInput{{Plan: match, PlanSHA256: matchSHA, Projection: projection, SHA256: projection.SHA256, Weight: .2}}}}, Limits: pipeline.SignalLimits{MaxBytes: 1 << 20, MaxFeatures: 16, MaxFeatureValues: 128, Cache: fusioncache.Limits{MaxBytes: 1 << 18, MaxExamples: 8, MaxTokens: 32, MaxPositions: 16, MaxTopK: 8, MaxFeatureValues: 128, MaxMappingPairs: 8}, Projection: projectionLimits}}}}
	c := ConstrainedFactoryConfig{Assembly: plan, Shards: sessionNoShards{}, Recipe: policy}
	c.SHA256, err = c.Digest()
	if err != nil {
		t.Fatal(err)
	}
	request := pipeline.ConstrainedRequest{Step: 1, Context: pipeline.StageContext{Execution: pipeline.Execution{RunID: identity.RunID, TenantID: identity.TenantID, Generation: 1, Recipe: recipe, TargetSHA256: identity.TargetSHA256, Placement: placement}, Stage: recipe.Stages[3], ArtifactDirectory: root, Parent: &parent, Completed: []pipeline.StageReceipt{teacherReceipt, parent}}, Sources: []pipeline.ConstrainedSourceFile{{Source: source, Path: cachePath}}}
	var last *LoadedTextModel
	loader := func(_ context.Context, p *AssemblyPlan, _ ShardProvider, a *InitialAdapter) (*LoadedTextModel, error) {
		last = sessionCPUModel(t, p, a)
		return last, nil
	}
	return c, request, loader, &last
}

func sessionCPUModel(t *testing.T, p *AssemblyPlan, initial *InitialAdapter) *LoadedTextModel {
	t.Helper()
	m := &LoadedTextModel{Model: &TextModel{Epsilon: 1e-6, Layers: make([]Layer, 1)}, Summary: p.Summary()}
	makeTensor := func(shape []int64, norm bool) *torch.Tensor {
		size := 1
		for _, d := range shape {
			size *= int(d)
		}
		values := make([]float32, size)
		for i := range values {
			values[i] = float32(.2 * math.Sin(float64(i+1)*.71))
			if norm {
				values[i] = 1
			}
		}
		value, err := torch.FromFloat32(values, shape, torch.CPUDevice(), false)
		if err != nil {
			t.Fatal(err)
		}
		m.owned = append(m.owned, value)
		return value
	}
	m.Model.Embedding = makeTensor([]int64{6, 4}, false)
	m.Model.Head = makeTensor([]int64{6, 4}, false)
	m.Model.FinalNorm = makeTensor([]int64{4}, true)
	w := layers.DecoderWeights{InputNorm: makeTensor([]int64{4}, true), PostAttentionNorm: makeTensor([]int64{4}, true), Gate: makeTensor([]int64{5, 4}, false), Up: makeTensor([]int64{5, 4}, false), Down: makeTensor([]int64{4, 5}, false)}
	w.Full = &layers.AttentionWeights{Query: makeTensor([]int64{8, 4}, false), Key: makeTensor([]int64{2, 4}, false), Value: makeTensor([]int64{2, 4}, false), Output: makeTensor([]int64{4, 4}, false), QueryNorm: makeTensor([]int64{2}, true), KeyNorm: makeTensor([]int64{2}, true)}
	for _, parameter := range initial.Parameters {
		info, err := parameter.Value.Info()
		if err != nil {
			t.Fatal(err)
		}
		values, err := parameter.Value.Float32Values()
		if err != nil {
			t.Fatal(err)
		}
		leaf, err := torch.FromFloat32(values, info.Shape, torch.CPUDevice(), true)
		if err != nil {
			t.Fatal(err)
		}
		m.owned = append(m.owned, leaf)
		m.Parameters = append(m.Parameters, InitialParameter{Name: parameter.Name, Value: leaf})
	}
	a := &layers.AttentionLoRA{QueryA: m.Parameters[0].Value, QueryB: m.Parameters[1].Value, ValueA: m.Parameters[2].Value, ValueB: m.Parameters[3].Value, Alpha: 2}
	m.Model.Layers[0] = Layer{Device: torch.CPUDevice(), Weights: w, Adapter: a, Config: layers.DecoderConfig{Epsilon: 1e-6, MaxInputElements: 12, Full: layers.AttentionConfig{Heads: 2, KVHeads: 1, HeadDimension: 2, RotaryDimension: 2, Epsilon: 1e-6, MaxScoreElements: 18}}}
	t.Cleanup(func() { _ = m.Close() })
	return m
}

func TestNativeConstrainedFactoryExecutesActualFusionStageCPU(t *testing.T) {
	c, request, loader, last := sessionFixture(t)
	factory, err := newConstrainedFactory(c, loader)
	if err != nil {
		t.Fatal(err)
	}
	session, err := factory(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	first, err := session.Model.Objective(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	second, err := session.Model.Objective(context.Background())
	if err != nil || !reflect.DeepEqual(first, second) || first.FeatureLoss <= 0 {
		t.Fatalf("unstable or missing objective: %+v %v", first, err)
	}
	if (*last).Model.Layers[0].Cosine != nil || (*last).Model.Layers[0].Sine != nil {
		t.Fatal("rotary tables survived objective")
	}
	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := session.Model.Parameters(context.Background()); err == nil {
		t.Fatal("closed backend remained usable")
	}
	h, err := pipeline.NewConstrainedStage(pipeline.ConstrainedConfig{Recipe: c.Recipe.Recipe, Placement: c.Recipe.Placement, Protocol: c.Recipe.Protocol, ProtocolSHA256: c.Recipe.ProtocolSHA256, SourceDirectory: request.Context.ArtifactDirectory, Factory: factory})
	if err != nil {
		t.Fatal(err)
	}
	var output pipeline.StageResult
	if err := h.Run(context.Background(), request.Context, func(_ context.Context, result pipeline.StageResult) error { output = result; return nil }); err != nil {
		t.Fatal(err)
	}
	if !output.Complete || output.Steps != 1 || (*last).Model != nil {
		t.Fatal("stage did not commit and release native model")
	}
	body, err := os.ReadFile(filepath.Join(request.Context.ArtifactDirectory, output.Artifacts[1].Path))
	if err != nil {
		t.Fatal(err)
	}
	var evidence pipeline.ConstrainedEvidence
	if err := json.Unmarshal(body, &evidence); err != nil {
		t.Fatal(err)
	}
	if evidence.Receipt.Objective.FeatureLoss <= 0 || !evidence.Receipt.Evaluation.Accepted || evidence.Receipt.CandidateObjective.Loss > evidence.Receipt.Objective.Loss || evidence.PriorParametersSHA256 == evidence.ParametersSHA256 {
		t.Fatal("native fusion did not produce an accepted changed candidate")
	}
}

func TestNativeConstrainedFactoryRefusesSourcesAndClosesFailedLoad(t *testing.T) {
	for _, mode := range []string{"context", "target", "source", "cache-bytes", "projection", "checkpoint", "cancel-after-load", "load-error"} {
		t.Run(mode, func(t *testing.T) {
			c, request, loader, last := sessionFixture(t)
			calls := 0
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			wrapped := func(ctx context.Context, p *AssemblyPlan, provider ShardProvider, a *InitialAdapter) (*LoadedTextModel, error) {
				calls++
				m, e := loader(ctx, p, provider, a)
				if mode == "cancel-after-load" {
					cancel()
				}
				if mode == "load-error" {
					return m, io.ErrUnexpectedEOF
				}
				return m, e
			}
			if mode == "projection" {
				c.Recipe.Signals[0].Caches[0].Projections[0].Projection.Coefficients[0][0] += 1
				c.SHA256, _ = c.Digest()
			}
			factory, err := newConstrainedFactory(c, wrapped)
			if err != nil {
				t.Fatal(err)
			}
			switch mode {
			case "context":
				request.Context.Stage.MaxTokens++
			case "target":
				request.Context.Execution.TargetSHA256 = strings.Repeat("0", 64)
			case "source":
				request.Sources[0].Source.Artifact.SHA256 = strings.Repeat("0", 64)
			case "cache-bytes":
				if err := os.WriteFile(request.Sources[0].Path, []byte("changed"), 0600); err != nil {
					t.Fatal(err)
				}
			case "checkpoint":
				if err := os.WriteFile(c.Recipe.Initial.Path, []byte("changed"), 0600); err != nil {
					t.Fatal(err)
				}
			}
			session, err := factory(ctx, request)
			if err == nil || session.Model != nil || session.Close != nil {
				t.Fatalf("bad session escaped: %v", err)
			}
			if mode == "checkpoint" || mode == "cancel-after-load" || mode == "load-error" {
				if calls != 1 || (*last).Model != nil || len((*last).owned) != 0 {
					t.Fatal("failed native session retained owner")
				}
			} else if calls != 0 {
				t.Fatal("bad source reached native loading")
			}
		})
	}
}

func TestNativeConstrainedFactoryPinsConfigurationAndCancelableOwnership(t *testing.T) {
	c, request, loader, _ := sessionFixture(t)
	bad := c
	bad.SHA256 = strings.Repeat("0", 64)
	if _, err := newConstrainedFactory(bad, loader); err == nil {
		t.Fatal("bad configuration digest accepted")
	}
	factory, err := newConstrainedFactory(c, loader)
	if err != nil {
		t.Fatal(err)
	}
	c.Recipe.Signals[0].Caches[0].Projections[0].Weight = 99
	session, err := factory(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := factory(ctx, request); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
	second, err := factory(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if err := second.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestNativeConstrainedFactoryAdmitsCompleteFusionAndMemory(t *testing.T) {
	for _, mode := range []string{"missing-teacher", "wrong-teacher", "duplicate-teacher", "wrong-student", "recovery", "persistent", "scratch", "solver-memory", "solver-total-memory", "local-mps", "overflow", "nil-map", "nil-slice", "nil-func", "nil-chan", "nil-pointer"} {
		t.Run(mode, func(t *testing.T) {
			c, _, _, _ := sessionFixture(t)
			switch mode {
			case "missing-teacher", "duplicate-teacher":
				c.Recipe.Recipe.Teachers = append(c.Recipe.Recipe.Teachers, pipeline.ArtifactRef{ID: "other-teacher", SHA256: assemblyHash([]byte("other-teacher"))})
				if mode == "duplicate-teacher" {
					duplicate := c.Recipe.Signals[0].Caches[0]
					duplicate.SourceIndex = 1
					c.Recipe.Signals[0].Caches = append(c.Recipe.Signals[0].Caches, duplicate)
					c.Recipe.Protocol.Updates[0].Sources = append(c.Recipe.Protocol.Updates[0].Sources, c.Recipe.Protocol.Updates[0].Sources[0])
					c.Recipe.Protocol.Updates[0].Sources[1].Artifact.Path = "second-cache.json"
				}
			case "wrong-teacher":
				c.Recipe.Signals[0].Caches[0].Expectation.Teacher.WeightsSHA256 = assemblyHash([]byte("not-admitted"))
			case "wrong-student":
				c.Recipe.Recipe.Student.SHA256 = assemblyHash([]byte("not-admitted"))
			case "recovery":
				c.Recipe.Protocol.StageID = "stage-4"
			case "persistent":
				c.Assembly.summary.PersistentBytes[0] = c.Recipe.Placement.MemoryBytes
			case "scratch":
				c.Recipe.Initial.MaxWorkingBytes = c.Recipe.Placement.MemoryBytes - 4096
			case "solver-memory":
				c.Recipe.StepWorkingBytes = c.Assembly.summary.AdapterElements * 8
			case "solver-total-memory":
				c.Recipe.StepWorkingBytes = c.Recipe.Placement.MemoryBytes
			case "local-mps":
				c.Assembly.localMPS = true
				c.Assembly.summary.LocalMPSBytes = c.Recipe.Placement.MemoryBytes
			case "overflow":
				c.Assembly.summary.PersistentBytes = []int64{math.MaxInt64, math.MaxInt64}
			case "nil-map":
				c.Shards = sessionNilMap(nil)
			case "nil-slice":
				c.Shards = sessionNilSlice(nil)
			case "nil-func":
				c.Shards = sessionNilFunc(nil)
			case "nil-chan":
				c.Shards = sessionNilChan(nil)
			case "nil-pointer":
				c.Shards = (*sessionNoShards)(nil)
			}
			var err error
			c.Recipe.Protocol.RecipeSHA256, err = c.Recipe.Recipe.Digest()
			if err != nil {
				t.Fatal(err)
			}
			c.Recipe.ProtocolSHA256, err = c.Recipe.Protocol.Digest()
			if err != nil {
				t.Fatal(err)
			}
			if _, err := c.Digest(); !errors.Is(err, ErrConstrainedSession) {
				t.Fatalf("invalid %s admitted: %v", mode, err)
			}
		})
	}
}

func TestNativeConstrainedFactoryRestoresPriorStepWithoutInitialCheckpoint(t *testing.T) {
	c, request, loader, _ := sessionFixture(t)
	c.Recipe.Recipe.Stages[3].MaxSteps = 2
	c.Recipe.Protocol.Limits.MaxSteps = 2
	c.Recipe.Protocol.Limits.MaxTotalBytes = 6 << 20
	c.Recipe.Protocol.Updates = append(c.Recipe.Protocol.Updates, c.Recipe.Protocol.Updates[0])
	c.Recipe.Signals = append(c.Recipe.Signals, c.Recipe.Signals[0])
	var err error
	c.Recipe.Protocol.RecipeSHA256, err = c.Recipe.Recipe.Digest()
	if err != nil {
		t.Fatal(err)
	}
	c.Recipe.ProtocolSHA256, err = c.Recipe.Protocol.Digest()
	if err != nil {
		t.Fatal(err)
	}
	c.SHA256, err = c.Digest()
	if err != nil {
		t.Fatal(err)
	}
	request.Context.Execution.Recipe = c.Recipe.Recipe
	request.Context.Stage = c.Recipe.Recipe.Stages[3]
	factory, err := newConstrainedFactory(c, loader)
	if err != nil {
		t.Fatal(err)
	}
	session, err := factory(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	parameters, err := session.Model.Parameters(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
	parameters[0] += .25
	request.Step, request.Parameters = 2, parameters
	if err := os.Remove(c.Recipe.Initial.Path); err != nil {
		t.Fatal(err)
	}
	session, err = factory(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	installed, err := session.Model.Parameters(context.Background())
	if err != nil || !reflect.DeepEqual(parameters, installed) {
		t.Fatalf("prior step not restored: %v", err)
	}
}
