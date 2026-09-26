package pipeline

import (
	"bytes"
	"context"
	"crypto/rand"
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
	"slices"
)

import "github.com/tayi-ai/arandu-llama/checkpoint"

// ConstrainedEvidence is an immutable scientific record of one accepted Step.
// Receipt retains objective gradients, KKT certificate and measured margins;
// Parameters is stored separately as safetensors, so Receipt.Parameters is nil.
// Verification establishes recorded consistency, not independent model execution.
type ConstrainedEvidence struct {
	Version               int             `json:"version"`
	Identity              ReceiptIdentity `json:"identity"`
	StageID               string          `json:"stage_id"`
	Step                  int             `json:"step"`
	ProtocolSHA256        string          `json:"protocol_sha256"`
	UpdateSHA256          string          `json:"update_sha256"`
	ParentSHA256          string          `json:"parent_sha256"`
	PreviousSHA256        string          `json:"previous_sha256"`
	PriorParametersSHA256 string          `json:"prior_parameters_sha256"`
	ParametersSHA256      string          `json:"parameters_sha256"`
	Parameters            StageArtifact   `json:"parameters"`
	Intent                StageArtifact   `json:"intent"`
	Prior                 []float32       `json:"prior"`
	Receipt               StepReceipt     `json:"receipt"`
}

type constrainedState struct {
	results    []StageResult
	last       ConstrainedEvidence
	lastSHA    string
	parameters []float32
	total      int64
}

type constrainedIntent struct {
	Version               int             `json:"version"`
	Identity              ReceiptIdentity `json:"identity"`
	StageID               string          `json:"stage_id"`
	Step                  int             `json:"step"`
	ProtocolSHA256        string          `json:"protocol_sha256"`
	UpdateSHA256          string          `json:"update_sha256"`
	ParentSHA256          string          `json:"parent_sha256"`
	PreviousSHA256        string          `json:"previous_sha256"`
	PriorParametersSHA256 string          `json:"prior_parameters_sha256"`
}

func constrainedRoot(path string) (*os.Root, error) {
	info, err := os.Lstat(path)
	if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return nil, errors.Join(ErrConstrained, err)
	}
	return os.OpenRoot(path)
}

func (h *ConstrainedStage) sourceFiles(ctx context.Context, c StageContext, step int) ([]ConstrainedSourceFile, error) {
	var files []ConstrainedSourceFile
	for _, source := range h.config.Protocol.Updates[step-1].Sources {
		directory := h.config.SourceDirectory
		if source.StageID != "" {
			directory = c.ArtifactDirectory
			found := false
			for _, receipt := range c.Completed {
				if receipt.StageID == source.StageID && receipt.Result.Complete && receipt.SHA256 == source.ReceiptSHA256 &&
					receiptSHA256(receipt) == receipt.SHA256 && slices.Contains(receipt.Result.Artifacts, source.Artifact) {
					identity, err := h.stageIdentity(c)
					if err != nil || !sameIdentity(identity, receipt.Identity) || receipt.Identity.Generation == 0 || receipt.Identity.Generation > identity.Generation {
						return nil, ErrConstrained
					}
					found = true
				}
			}
			if !found {
				return nil, fmt.Errorf("%w: admitted predecessor source absent", ErrConstrained)
			}
		}
		root, err := constrainedRoot(directory)
		if err != nil {
			return nil, err
		}
		_, err = constrainedRead(ctx, root, source.Artifact.Path, source.Artifact.Bytes, source.Artifact.SHA256, false)
		err = errors.Join(err, root.Close())
		if err != nil {
			return nil, err
		}
		files = append(files, ConstrainedSourceFile{Source: source, Path: filepath.Join(directory, source.Artifact.Path)})
	}
	return files, ctx.Err()
}

// constrainedRead checks immutable regular files before allocating. retain=false
// streams source hashes without retaining their potentially large contents.
func constrainedRead(ctx context.Context, root *os.Root, name string, limit int64, expected string, retain bool) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	f, err := openDurableRegular(root, name, os.O_RDONLY, 0)
	if err != nil {
		return nil, errors.Join(ErrConstrained, err)
	}
	defer f.Close()
	before, err := f.Stat()
	if err != nil || before.Size() < 1 || before.Size() > limit || before.Size() > int64(int(^uint(0)>>1)) || expected != "" && before.Size() != limit {
		return nil, ErrConstrained
	}
	var body []byte
	var destination io.Writer
	hash := sha256.New()
	if retain {
		body = make([]byte, int(before.Size()))
		for start := 0; start < len(body); {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			end := start + min(64<<10, len(body)-start)
			if _, err := io.ReadFull(f, body[start:end]); err != nil {
				return nil, err
			}
			_, _ = hash.Write(body[start:end])
			start = end
		}
	} else {
		destination = hash
		n, err := io.Copy(destination, io.LimitReader(contextReader{ctx: ctx, reader: f}, before.Size()+1))
		if err != nil || n != before.Size() {
			return nil, errors.Join(ErrConstrained, err)
		}
	}
	after, err := f.Stat()
	if err != nil || after.Size() != before.Size() || !after.ModTime().Equal(before.ModTime()) || expected != "" && hex.EncodeToString(hash.Sum(nil)) != expected {
		return nil, errors.Join(ErrConstrained, err)
	}
	return body, ctx.Err()
}

func (h *ConstrainedStage) namespace() string { return "constrained-" + h.stage.ID }
func (h *ConstrainedStage) stepDirectory(step int) string {
	return filepath.Join(h.namespace(), fmt.Sprintf("step-%06d", step))
}

func (h *ConstrainedStage) intentPath(step int) string {
	return filepath.Join(h.namespace(), fmt.Sprintf("attempt-%06d.json", step))
}

func (h *ConstrainedStage) scan(ctx context.Context, c StageContext) (constrainedState, error) {
	var state constrainedState
	identity, err := h.stageIdentity(c)
	if err != nil {
		return state, err
	}
	root, err := constrainedRoot(c.ArtifactDirectory)
	if err != nil {
		return state, err
	}
	defer root.Close()
	info, err := root.Lstat(h.namespace())
	if errors.Is(err, os.ErrNotExist) {
		return state, nil
	}
	if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return state, ErrConstrained
	}
	f, err := root.Open(h.namespace())
	if err != nil {
		return state, err
	}
	names, readErr := f.Readdirnames(2*len(h.config.Protocol.Updates) + 2)
	closeErr := f.Close()
	if readErr != nil && readErr != io.EOF || closeErr != nil || len(names) > 2*len(h.config.Protocol.Updates) {
		return state, ErrConstrained
	}
	entries := map[string]bool{}
	for _, name := range names {
		entries[name] = true
	}
	priorSHA := h.config.Protocol.InitialParametersSHA256
	var generation uint64
	for i := 0; i < len(h.config.Protocol.Updates); i++ {
		step := i + 1
		dir := h.stepDirectory(step)
		attempt := h.intentPath(step)
		hasUnit, hasIntent := entries[filepath.Base(dir)], entries[filepath.Base(attempt)]
		if !hasUnit && !hasIntent {
			break
		}
		if !hasIntent {
			return state, ErrConstrained
		}
		intentBody, err := constrainedRead(ctx, root, attempt, h.config.Protocol.Limits.MaxReceiptBytes, "", true)
		if err != nil {
			return state, err
		}
		var intent constrainedIntent
		d := json.NewDecoder(bytes.NewReader(intentBody))
		d.DisallowUnknownFields()
		if d.Decode(&intent) != nil || d.Decode(new(any)) != io.EOF {
			return state, ErrConstrained
		}
		intentCanonical, err := constrainedJSON(intent, h.config.Protocol.Limits.MaxReceiptBytes)
		if err != nil || !bytes.Equal(intentCanonical, intentBody) || intent.Version != 1 || !sameIdentity(identity, intent.Identity) || intent.Identity.Generation == 0 || intent.Identity.Generation > identity.Generation || intent.Identity.Generation < generation ||
			intent.StageID != h.stage.ID || intent.Step != step || intent.ProtocolSHA256 != h.config.ProtocolSHA256 || intent.UpdateSHA256 != jsonSHA256(h.config.Protocol.Updates[i]) || intent.ParentSHA256 != c.Parent.SHA256 || intent.PreviousSHA256 != state.lastSHA || intent.PriorParametersSHA256 != priorSHA {
			return state, ErrConstrained
		}
		if !hasUnit {
			return state, ErrConstrainedAttempt
		}
		intentArtifact := StageArtifact{Path: attempt, SHA256: constrainedSHA(intentBody), Bytes: int64(len(intentBody))}
		delete(entries, filepath.Base(dir))
		delete(entries, filepath.Base(attempt))
		if err := constrainedDirectoryContents(root, dir); err != nil {
			return state, err
		}
		body, err := constrainedRead(ctx, root, filepath.Join(dir, "evidence.json"), h.config.Protocol.Limits.MaxReceiptBytes, "", true)
		if err != nil {
			return state, err
		}
		var evidence ConstrainedEvidence
		decoder := json.NewDecoder(bytes.NewReader(body))
		decoder.DisallowUnknownFields()
		if decoder.Decode(&evidence) != nil || decoder.Decode(new(any)) != io.EOF {
			return state, ErrConstrained
		}
		canonical, err := constrainedJSON(evidence, h.config.Protocol.Limits.MaxReceiptBytes)
		if err != nil || !bytes.Equal(canonical, body) || evidence.Version != 1 || !sameIdentity(identity, evidence.Identity) || evidence.Identity.Generation == 0 || evidence.Identity.Generation > identity.Generation || evidence.Identity.Generation < generation ||
			evidence.StageID != h.stage.ID || evidence.Step != step || evidence.ProtocolSHA256 != h.config.ProtocolSHA256 || evidence.UpdateSHA256 != jsonSHA256(h.config.Protocol.Updates[i]) ||
			evidence.ParentSHA256 != c.Parent.SHA256 || evidence.PreviousSHA256 != state.lastSHA || evidence.PriorParametersSHA256 != priorSHA || !digest(evidence.ParametersSHA256) ||
			evidence.Parameters.Path != filepath.Join(dir, "parameters.safetensors") || !constrainedArtifact(evidence.Parameters, h.config.Protocol.Limits.MaxCheckpointBytes) || evidence.Intent != intentArtifact || evidence.Identity != intent.Identity {
			return state, ErrConstrained
		}
		parameters, err := h.readParameters(ctx, root, evidence.Parameters)
		if err != nil {
			return state, err
		}
		if err := h.checkEvidence(evidence, parameters); err != nil {
			return state, err
		}
		if _, err := h.sourceFiles(ctx, c, step); err != nil {
			return state, err
		}
		artifact := StageArtifact{Path: filepath.Join(dir, "evidence.json"), SHA256: constrainedSHA(body), Bytes: int64(len(body))}
		for _, a := range []StageArtifact{evidence.Parameters, artifact, intentArtifact} {
			if a.Bytes > h.config.Protocol.Limits.MaxTotalBytes-state.total {
				return state, ErrConstrained
			}
			state.total += a.Bytes
		}
		state.results = append(state.results, StageResult{Steps: int64(step), Complete: step == len(h.config.Protocol.Updates), Artifacts: []StageArtifact{evidence.Parameters, artifact, intentArtifact}})
		state.last, state.lastSHA, state.parameters = evidence, artifact.SHA256, parameters
		priorSHA, generation = evidence.ParametersSHA256, evidence.Identity.Generation
	}
	if len(entries) != 0 {
		return state, fmt.Errorf("%w: incomplete or unexpected checkpoint directory", ErrConstrained)
	}
	return state, ctx.Err()
}

func constrainedDirectoryContents(root *os.Root, name string) error {
	info, err := root.Lstat(name)
	if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return ErrConstrained
	}
	f, err := root.Open(name)
	if err != nil {
		return err
	}
	names, e := f.Readdirnames(3)
	closeErr := f.Close()
	slices.Sort(names)
	if e != nil && e != io.EOF || closeErr != nil || !slices.Equal(names, []string{"evidence.json", "parameters.safetensors"}) {
		return ErrConstrained
	}
	return nil
}

func (h *ConstrainedStage) readParameters(ctx context.Context, root *os.Root, artifact StageArtifact) ([]float32, error) {
	body, err := constrainedRead(ctx, root, artifact.Path, artifact.Bytes, artifact.SHA256, true)
	if err != nil {
		return nil, err
	}
	index, err := checkpoint.OpenSafetensors(bytes.NewReader(body), int64(len(body)), h.config.Protocol.Limits.Checkpoint)
	if err != nil || len(index.Tensors()) != len(h.config.Protocol.Layout) {
		return nil, errors.Join(ErrConstrained, err)
	}
	for _, spec := range h.config.Protocol.Layout {
		tensor, ok := index.Tensor(spec.Name)
		if !ok || tensor.DType != "F32" || !slices.Equal(tensor.Shape, spec.Shape) {
			return nil, ErrConstrained
		}
	}
	values := make([]float32, h.parameters)
	offset := 0
	for _, spec := range h.config.Protocol.Layout {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		reader, err := index.TensorReader(spec.Name)
		if err != nil {
			return nil, err
		}
		_, start, size := reader.Outer()
		for i := int64(0); i < size; i += 4 {
			if i&65535 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			value := math.Float32frombits(binary.LittleEndian.Uint32(body[start+i:]))
			if !finite(float64(value)) {
				return nil, ErrConstrained
			}
			values[offset] = value
			offset++
		}
	}
	if offset != len(values) {
		return nil, ErrConstrained
	}
	return values, nil
}

func (h *ConstrainedStage) checkEvidence(e ConstrainedEvidence, parameters []float32) error {
	prior, err := h.parameterSHA(e.Prior)
	if err != nil || prior != e.PriorParametersSHA256 {
		return ErrConstrained
	}
	next, err := h.parameterSHA(parameters)
	if err != nil || next != e.ParametersSHA256 {
		return ErrConstrained
	}
	r, cfg := e.Receipt, h.config.Protocol.Updates[e.Step-1].Config
	if len(r.Parameters) != 0 || !r.Evaluation.Accepted || r.Evaluation.Restored || len(r.Evaluation.Margins) != len(cfg.Protection) ||
		len(r.Objective.Gradient) != h.parameters || len(r.CandidateObjective.Gradient) != h.parameters || len(r.Solution.Step) != h.parameters || len(r.Solution.Multipliers) != len(cfg.Protection) ||
		!finite(r.Objective.Loss-r.CandidateObjective.Loss) || r.Objective.Loss-r.CandidateObjective.Loss < cfg.MinimumGain {
		return ErrConstrained
	}
	for _, o := range []Objective{r.Objective, r.CandidateObjective} {
		if !finite(o.Loss) || !finite(o.HardLoss) || !finite(o.FeatureLoss) {
			return ErrConstrained
		}
		for _, v := range o.Gradient {
			if !finite(v) {
				return ErrConstrained
			}
		}
	}
	for i, pair := range cfg.Protection {
		margin, multiplier := r.Evaluation.Margins[i], r.Solution.Multipliers[i]
		if margin.ID != pair.ID || !finite(margin.Value) || margin.Value < pair.Floor || !finite(multiplier) || multiplier < 0 {
			return ErrConstrained
		}
	}
	c := r.Solution.Certificate
	if !finite(c.PrimalViolation) || c.PrimalViolation < 0 || c.PrimalViolation > cfg.Solver.PrimalTolerance ||
		!finite(c.StationarityResidual) || c.StationarityResidual < 0 || c.StationarityResidual > cfg.Solver.StationarityTolerance ||
		!finite(c.ComplementarityResidual) || c.ComplementarityResidual < 0 || c.ComplementarityResidual > cfg.Solver.ComplementarityTolerance || !finite(c.Objective) ||
		c.Sweeps < 0 || c.Sweeps > cfg.Solver.MaxSweeps || c.Work == 0 || c.Work > cfg.Solver.MaxWork {
		return ErrConstrained
	}
	for i, delta := range r.Solution.Step {
		if !finite(delta) || math.Float32bits(float32(float64(e.Prior[i])+delta)) != math.Float32bits(parameters[i]) {
			return ErrConstrained
		}
	}
	return nil
}

type constrainedWriter struct {
	target    io.Writer
	ctx       context.Context
	remaining int64
}

func (w *constrainedWriter) Write(body []byte) (int, error) {
	if err := w.ctx.Err(); err != nil {
		return 0, err
	}
	if int64(len(body)) > w.remaining {
		return 0, ErrConstrained
	}
	n, err := w.target.Write(body)
	w.remaining -= int64(n)
	return n, err
}

func (h *ConstrainedStage) persist(ctx context.Context, c StageContext, e ConstrainedEvidence, parameters []float32) (StageResult, error) {
	root, err := constrainedRoot(c.ArtifactDirectory)
	if err != nil {
		return StageResult{}, err
	}
	defer root.Close()
	if err := root.Mkdir(h.namespace(), 0700); err != nil && !errors.Is(err, os.ErrExist) {
		return StageResult{}, err
	}
	info, err := root.Lstat(h.namespace())
	if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return StageResult{}, ErrConstrained
	}
	if err := syncDirectory(root); err != nil {
		return StageResult{}, err
	}
	dir, err := root.OpenRoot(h.namespace())
	if err != nil {
		return StageResult{}, err
	}
	defer dir.Close()
	final := filepath.Base(h.stepDirectory(e.Step))
	if _, err := dir.Lstat(final); !errors.Is(err, os.ErrNotExist) {
		return StageResult{}, fmt.Errorf("%w: checkpoint already exists", ErrConstrained)
	}
	var nonce [16]byte
	if _, err := rand.Read(nonce[:]); err != nil {
		return StageResult{}, err
	}
	temporary := ".pending-" + hex.EncodeToString(nonce[:])
	if err := dir.Mkdir(temporary, 0700); err != nil {
		return StageResult{}, err
	}
	// Pre-publication errors explicitly discard only this invocation's private
	// temporary directory. A process crash leaves it visible and reconciliation
	// refuses it; callers must quarantine that incomplete evidence explicitly.
	defer dir.RemoveAll(temporary)
	unit, err := dir.OpenRoot(temporary)
	if err != nil {
		return StageResult{}, err
	}
	defer unit.Close()
	file, err := openDurableRegular(unit, "parameters.safetensors", os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return StageResult{}, err
	}
	var tensors []checkpoint.Float32Tensor
	offset := 0
	for _, spec := range h.config.Protocol.Layout {
		size := 1
		for _, d := range spec.Shape {
			size *= int(d)
		}
		tensors = append(tensors, checkpoint.Float32Tensor{Name: spec.Name, Shape: slices.Clone(spec.Shape), Values: parameters[offset : offset+size]})
		offset += size
	}
	l := h.config.Protocol.Limits
	receipt, writeErr := checkpoint.WriteFloat32(ctx, &constrainedWriter{target: file, ctx: ctx, remaining: l.MaxCheckpointBytes}, tensors, l.Checkpoint)
	if err := errors.Join(writeErr, file.Sync(), file.Close()); err != nil {
		return StageResult{}, err
	}
	e.Parameters = StageArtifact{Path: filepath.Join(h.stepDirectory(e.Step), "parameters.safetensors"), SHA256: receipt.SHA256, Bytes: receipt.Bytes}
	readback := e.Parameters
	readback.Path = "parameters.safetensors"
	stored, err := h.readParameters(ctx, unit, readback)
	if err != nil {
		return StageResult{}, err
	}
	if err := h.checkEvidence(e, stored); err != nil {
		return StageResult{}, err
	}
	body, err := constrainedJSON(e, l.MaxReceiptBytes)
	if err != nil {
		return StageResult{}, err
	}
	file, err = openDurableRegular(unit, "evidence.json", os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return StageResult{}, err
	}
	_, writeErr = file.Write(body)
	if err := errors.Join(writeErr, file.Sync(), file.Close()); err != nil {
		return StageResult{}, err
	}
	if _, err := constrainedRead(ctx, unit, "evidence.json", int64(len(body)), constrainedSHA(body), false); err != nil {
		return StageResult{}, err
	}
	if err := syncDirectory(unit); err != nil {
		return StageResult{}, err
	}
	if err := ctx.Err(); err != nil {
		return StageResult{}, err
	}
	if err := dir.Rename(temporary, final); err != nil {
		return StageResult{}, err
	}
	if err := syncDirectory(dir); err != nil {
		return StageResult{}, err
	}
	return StageResult{Steps: int64(e.Step), Complete: e.Step == len(h.config.Protocol.Updates), Artifacts: []StageArtifact{e.Parameters, {Path: filepath.Join(h.stepDirectory(e.Step), "evidence.json"), SHA256: constrainedSHA(body), Bytes: int64(len(body))}, e.Intent}}, nil
}

func (h *ConstrainedStage) beginAttempt(ctx context.Context, c StageContext, step int, previousSHA, priorSHA string) (StageArtifact, error) {
	identity, err := h.stageIdentity(c)
	if err != nil {
		return StageArtifact{}, err
	}
	intent := constrainedIntent{Version: 1, Identity: identity, StageID: h.stage.ID, Step: step, ProtocolSHA256: h.config.ProtocolSHA256,
		UpdateSHA256: jsonSHA256(h.config.Protocol.Updates[step-1]), ParentSHA256: c.Parent.SHA256, PreviousSHA256: previousSHA, PriorParametersSHA256: priorSHA}
	body, err := constrainedJSON(intent, h.config.Protocol.Limits.MaxReceiptBytes)
	if err != nil {
		return StageArtifact{}, err
	}
	root, err := constrainedRoot(c.ArtifactDirectory)
	if err != nil {
		return StageArtifact{}, err
	}
	defer root.Close()
	if err := root.Mkdir(h.namespace(), 0700); err != nil && !errors.Is(err, os.ErrExist) {
		return StageArtifact{}, err
	}
	info, err := root.Lstat(h.namespace())
	if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return StageArtifact{}, ErrConstrained
	}
	if err := syncDirectory(root); err != nil {
		return StageArtifact{}, err
	}
	name := h.intentPath(step)
	file, err := openDurableRegular(root, name, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return StageArtifact{}, errors.Join(ErrConstrainedAttempt, err)
	}
	_, writeErr := file.Write(body)
	if err := errors.Join(writeErr, file.Sync(), file.Close()); err != nil {
		return StageArtifact{}, err
	}
	directory, err := root.OpenRoot(h.namespace())
	if err != nil {
		return StageArtifact{}, err
	}
	if err := errors.Join(syncDirectory(directory), directory.Close()); err != nil {
		return StageArtifact{}, err
	}
	if err := ctx.Err(); err != nil {
		return StageArtifact{}, err
	}
	return StageArtifact{Path: name, SHA256: constrainedSHA(body), Bytes: int64(len(body))}, nil
}

func constrainedSameResult(a, b StageResult) bool { return reflect.DeepEqual(a, b) }

// A unit may be visible after rename while its final directory entry was never
// fsynced. Recovery reestablishes the complete durability chain before any
// application receipt can become durable. No compute or mutation is replayed.
func (h *ConstrainedStage) syncRecovered(ctx context.Context, c StageContext, result StageResult) error {
	root, err := constrainedRoot(c.ArtifactDirectory)
	if err != nil {
		return err
	}
	defer root.Close()
	for _, artifact := range result.Artifacts {
		if err := ctx.Err(); err != nil {
			return err
		}
		file, err := openDurableRegular(root, artifact.Path, os.O_RDONLY, 0)
		if err != nil {
			return err
		}
		if err := errors.Join(file.Sync(), file.Close()); err != nil {
			return err
		}
	}
	for _, path := range []string{h.stepDirectory(int(result.Steps)), h.namespace()} {
		info, err := root.Lstat(path)
		if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
			return ErrConstrained
		}
		directory, err := root.OpenRoot(path)
		if err != nil {
			return err
		}
		if err := errors.Join(syncDirectory(directory), directory.Close()); err != nil {
			return err
		}
	}
	if err := syncDirectory(root); err != nil {
		return err
	}
	return ctx.Err()
}
