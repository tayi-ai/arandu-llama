package adapter

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
)

// The candidate: one constructor for the probe and for the step.
//
// A LoRA adapter is two factors per module, A and B, and the weight it adds is
// their product scaled by alpha over rank. The update a zeroth-order step
// accepts adds c*ZA to A and c*ZB to B, so the output moves along
// B*ZA + ZB*A. The first engine probed by loading the direction as a second
// adapter at scale c, and llama.cpp sums adapters in product space: what that
// measured was the derivative along ZB*ZA, the product of the direction's own
// factors. On the scalar oracle of the T02 addendum -- A = 2, B = 3, ZA = 5,
// ZB = -7, unit scale, eps = 0.001 -- the two derivatives are 1 and -35. The
// probe and the step were not moving along the same direction, and a training
// run built on that walks away from its own measurement.
//
// So there is one constructor. BuildCandidate takes an immutable snapshot S, a
// displacement Z over the same tensors, and a coefficient c, and returns a
// complete adapter whose factors are S + c*Z. The positive probe builds it with
// c = +eps, the negative with c = -eps, the accepted step with the optimiser's
// coefficient, and the engine applies a candidate only by loading the bytes
// that constructor wrote. The snapshot is never modified: a step that accepts
// a candidate replaces the snapshot with it, whole.
//
// Each probe writes a candidate file so its factor-space perturbation matches
// the factors an accepted update commits.

// AdapterHandle is a loaded adapter the engine can apply. *llama.Adapter is one.
type AdapterHandle interface {
	// Digest is the sha256 of the file the handle was loaded from. It is what
	// ties the handle back to the candidate the constructor wrote.
	Digest() string
	Close() error
}

// AdapterHost is the part of the engine that loads and applies adapters.
//
// It is the seam between the constructor, which is plain Go over bytes, and
// the cgo binding, which is only compiled with -tags llama. Everything the
// contract requires -- same constructor, same apply path, restore on failure
// -- lives above this interface and is tested without a card.
type AdapterHost interface {
	LoadAdapter(path string) (AdapterHandle, error)
	// SetAdapters replaces the whole applied set. An empty set clears it.
	SetAdapters(handles []AdapterHandle, scales []float32) error
}

// Displacement is a direction realised over the tensors of one snapshot: one
// slice per tensor, in header order, drawn from the standard normal.
//
// It is held in memory rather than written to a file because nothing loads it:
// the engine loads candidates, and a candidate is built from it.
type Displacement struct {
	Direction Direction
	Tensors   [][]float32
}

// Candidate is a complete adapter built as S + c*Z.
//
// It carries the bytes of a whole GGUF -- header copied from the snapshot, so
// names, dims, alpha and rank are preserved by construction -- and the digest
// the engine has to load back before the candidate counts as applied.
type Candidate struct {
	Seed        uint64
	Coefficient float64
	// Source is the digest of the snapshot the candidate was built from.
	Source string
	// Digest is the sha256 of the candidate's bytes.
	Digest string
	// Moved counts the elements whose stored value changed. An update below what
	// F32 resolves at the magnitude of the weight rounds away, and a chain of
	// those looks like convergence.
	Moved     int64
	container GGUF
	body      []byte
}

// Bytes returns a copy of the candidate's file.
func (c *Candidate) Bytes() []byte { return append([]byte(nil), c.body...) }

// Values decodes one tensor of the candidate by name.
func (c *Candidate) Values(name string) ([]float32, bool) {
	return tensorValues(c.container, c.body, name)
}

func tensorValues(container GGUF, body []byte, name string) ([]float32, bool) {
	for _, tensor := range container.Tensors {
		if tensor.Name != name {
			continue
		}
		start := container.DataOffset + tensor.Offset
		values := make([]float32, tensor.Elements)
		for i := range values {
			values[i] = math.Float32frombits(binary.LittleEndian.Uint32(body[start+int64(i)*4:]))
		}
		return values, true
	}
	return nil, false
}

// BuildCandidate returns S + coefficient*Z as an independent adapter.
//
// The arithmetic is element-wise over the same tensor table in the same order,
// which is why there is no transposition to get wrong: A is moved by ZA and B
// by ZB in place, and the header that says which is which is copied byte for
// byte. Each element is summed in float64 and stored once in float32.
func BuildCandidate(s *Snapshot, z *Displacement, coefficient float64) (*Candidate, error) {
	if math.IsNaN(coefficient) || math.IsInf(coefficient, 0) {
		return nil, fmt.Errorf("cluster: a candidate cannot be built with coefficient %v", coefficient)
	}
	if z == nil || len(z.Tensors) != len(s.container.Tensors) {
		return nil, fmt.Errorf("cluster: the displacement covers %d tensors and the snapshot has %d", displacementTensors(z), len(s.container.Tensors))
	}
	for i, tensor := range s.container.Tensors {
		if int64(len(z.Tensors[i])) != tensor.Elements {
			return nil, fmt.Errorf("cluster: the displacement has %d elements for %s, which has %d", len(z.Tensors[i]), tensor.Name, tensor.Elements)
		}
	}

	candidate := &Candidate{
		Seed:        z.Direction.Seed,
		Coefficient: coefficient,
		Source:      s.digest,
		container:   s.container,
		body:        append([]byte(nil), s.body...),
	}
	for i, tensor := range s.container.Tensors {
		start := s.container.DataOffset + tensor.Offset
		for k, step := range z.Tensors[i] {
			at := start + int64(k)*4
			before := math.Float32frombits(binary.LittleEndian.Uint32(candidate.body[at:]))
			after := float32(float64(before) + coefficient*float64(step))
			if after != before {
				candidate.Moved++
			}
			binary.LittleEndian.PutUint32(candidate.body[at:], math.Float32bits(after))
		}
	}
	candidate.Digest = Digest(candidate.body)
	return candidate, nil
}

func displacementTensors(z *Displacement) int {
	if z == nil {
		return 0
	}
	return len(z.Tensors)
}

// Snapshot is S: the policy adapter as a whole file, held immutable in memory
// and applied on the host at scale one.
//
// It owns the handle the host loaded for it and at most one candidate handle,
// the one currently applied instead of it. Everything that moves the engine
// goes through apply, and apply either leaves the requested handle applied or
// leaves the snapshot applied; there is no third state.
type Snapshot struct {
	host      AdapterHost
	path      string
	scratch   string
	container GGUF
	body      []byte
	digest    string
	handle    AdapterHandle
	candidate AdapterHandle
	drawn     *Displacement
}

// OpenSnapshot reads the policy, refuses what cannot be trained, loads it on
// the host and applies it at scale one.
//
// scratch is where the probe candidates are written before they are loaded;
// empty means beside the policy.
func OpenSnapshot(host AdapterHost, path, scratch string) (*Snapshot, error) {
	container, err := ReadGGUF(path)
	if err != nil {
		return nil, err
	}
	if len(container.Tensors) == 0 {
		return nil, fmt.Errorf("cluster: %s declares no tensors", filepath.Base(path))
	}
	body, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	// Refused for every tensor rather than the first: a file that mixes storage
	// types would otherwise pass on its first tensor and drift on the rest.
	for _, tensor := range container.Tensors {
		switch tensor.Type {
		case ggufF32:
		case ggufF16:
			return nil, fmt.Errorf("%w: %s in %s", ErrHalfPrecisionPolicy, tensor.Name, filepath.Base(path))
		default:
			return nil, fmt.Errorf("cluster: tensor %s has GGUF type %d; a candidate can only be built over F32", tensor.Name, tensor.Type)
		}
		if container.DataOffset+tensor.Offset+tensor.Elements*4 > int64(len(body)) {
			return nil, fmt.Errorf("cluster: tensor %s runs past the end of %s", tensor.Name, filepath.Base(path))
		}
	}
	if scratch == "" {
		scratch = filepath.Dir(path)
	}
	if err := os.MkdirAll(scratch, 0o755); err != nil {
		return nil, err
	}

	s := &Snapshot{host: host, path: path, scratch: scratch, container: container, body: body, digest: Digest(body)}
	handle, err := host.LoadAdapter(path)
	if err != nil {
		return nil, fmt.Errorf("cluster: loading the policy adapter: %w", err)
	}
	// The host hashes what it loaded. If that is not what was read here, the
	// engine and the constructor hold different policies and every candidate
	// would be built from one and applied beside the other.
	if handle.Digest() != s.digest {
		handle.Close()
		return nil, fmt.Errorf("cluster: the engine loaded %s as %s where the snapshot read %s", filepath.Base(path), handle.Digest(), s.digest)
	}
	if err := host.SetAdapters([]AdapterHandle{handle}, []float32{1}); err != nil {
		handle.Close()
		return nil, fmt.Errorf("cluster: applying the policy adapter: %w", err)
	}
	s.handle = handle
	return s, nil
}

// Digest is the sha256 of the snapshot's bytes.
func (s *Snapshot) Digest() string { return s.digest }

// Path is where the snapshot lives on disk.
func (s *Snapshot) Path() string { return s.path }

// Handle is what the host loaded for the snapshot.
func (s *Snapshot) Handle() AdapterHandle { return s.handle }

// Container is the tensor table of the snapshot.
func (s *Snapshot) Container() GGUF { return s.container }

// Bytes returns a copy of the snapshot's file.
func (s *Snapshot) Bytes() []byte { return append([]byte(nil), s.body...) }

// Values decodes one tensor of the snapshot by name.
func (s *Snapshot) Values(name string) ([]float32, bool) {
	return tensorValues(s.container, s.body, name)
}

// Draw realises a direction over the snapshot's tensors from its seed.
//
// The stream is the one Direction.Fill uses, consumed tensor by tensor in
// header order, so the vector a node rebuilds from the seed is the vector the
// estimator's message describes.
func (s *Snapshot) Draw(d Direction) *Displacement {
	source := d.stream()
	z := &Displacement{Direction: d, Tensors: make([][]float32, len(s.container.Tensors))}
	for i, tensor := range s.container.Tensors {
		values := make([]float32, tensor.Elements)
		for k := range values {
			values[k] = float32(source.NormFloat64())
		}
		z.Tensors[i] = values
	}
	return z
}

// Use hands the snapshot a displacement to use for its seed, in place of one
// drawn from the stream. It is how a test injects ZA = 5 and ZB = -7.
func (s *Snapshot) Use(z *Displacement) { s.drawn = z }

// displacement returns the realised direction for d, drawing it once per seed.
// Drawing the direction is paid once per step rather than three times.
func (s *Snapshot) displacement(d Direction) *Displacement {
	if s.drawn != nil && s.drawn.Direction.Seed == d.Seed {
		return s.drawn
	}
	s.drawn = s.Draw(d)
	return s.drawn
}

// Perturb applies the candidate S + sign*d.Scale*Z; a sign of zero re-applies
// the snapshot.
//
// The candidate is written to scratch, loaded by the host, checked against the
// digest the constructor computed, and only then applied. The file is removed
// once loaded: the seed and the coefficient reproduce it. Cancellation is
// honoured before anything is written; a restore is never refused, because a
// cancelled step still has to leave the snapshot applied.
func (s *Snapshot) Perturb(ctx context.Context, d Direction, sign float64) error {
	if sign == 0 {
		return s.apply(s.handle)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	candidate, err := BuildCandidate(s, s.displacement(d), sign*d.Scale)
	if err != nil {
		return err
	}
	loaded, path, err := s.load(candidate, s.scratch, ".candidate-")
	if err != nil {
		return err
	}
	// Removed as soon as it is loaded, whether or not the apply succeeds: the
	// seed and the coefficient reproduce it, and 69 MB a probe would fill the
	// scratch in a run.
	os.Remove(path)
	return s.apply(loaded)
}

// load writes a candidate into dir, has the host load it, and refuses a handle
// whose digest is not the candidate's. On success the caller owns the file at
// the returned path; on failure it is already gone.
func (s *Snapshot) load(candidate *Candidate, dir, prefix string) (AdapterHandle, string, error) {
	path, err := writeCandidate(candidate, dir, prefix)
	if err != nil {
		return nil, "", err
	}
	loaded, err := s.host.LoadAdapter(path)
	if err != nil {
		os.Remove(path)
		return nil, "", fmt.Errorf("cluster: loading the candidate: %w", err)
	}
	if loaded.Digest() != candidate.Digest {
		loaded.Close()
		os.Remove(path)
		return nil, "", fmt.Errorf("cluster: the engine loaded bytes that are not the candidate: %s where the constructor wrote %s", loaded.Digest(), candidate.Digest)
	}
	return loaded, path, nil
}

func writeCandidate(candidate *Candidate, dir, prefix string) (string, error) {
	temporary, err := os.CreateTemp(dir, prefix)
	if err != nil {
		return "", err
	}
	if _, err := temporary.Write(candidate.body); err != nil {
		temporary.Close()
		os.Remove(temporary.Name())
		return "", err
	}
	// Flushed to the device before the rename can promote it.
	//
	// Write plus rename is atomic against a process that dies: the reader sees
	// either name, never half a file. It is not atomic against the machine
	// losing power, because the rename can reach the disk before the bytes do
	// -- and then the policy is a name pointing at whatever the block held
	// before. That file still parses as GGUF and still loads; it is simply a
	// different adapter, which is the failure mode this package spends its
	// comments on.
	//
	// A probe candidate would not need this: it is removed at the end of the
	// step and regenerates from the snapshot, the seed and the coefficient.
	// One code path is cheaper to trust than two.
	if err := temporary.Sync(); err != nil {
		temporary.Close()
		os.Remove(temporary.Name())
		return "", err
	}
	if err := temporary.Close(); err != nil {
		os.Remove(temporary.Name())
		return "", err
	}
	return temporary.Name(), nil
}

// apply puts one handle on the host in place of whatever was applied.
//
// On failure the snapshot is re-applied and the rejected handle, if it was not
// the snapshot's own, is closed: the engine never stays on a candidate that
// was refused half way. On success the previously applied candidate is closed
// -- after the host has let go of it, because a context keeps the raw pointer
// and freeing an applied adapter leaves the next decode reading released
// memory and reporting a number rather than an error.
func (s *Snapshot) apply(next AdapterHandle) error {
	if err := s.host.SetAdapters([]AdapterHandle{next}, []float32{1}); err != nil {
		err = fmt.Errorf("cluster: applying the candidate: %w", err)
		// The snapshot goes back first; the handles are freed only once the host
		// has demonstrably let go of them. If even the restore fails, nothing is
		// freed: a leaked handle is a number, a freed applied one is a crash that
		// reports a number.
		if restore := s.host.SetAdapters([]AdapterHandle{s.handle}, []float32{1}); restore != nil {
			return errors.Join(err, fmt.Errorf("cluster: and the snapshot could not be re-applied: %w", restore))
		}
		if next != s.handle {
			next.Close()
		}
		if s.candidate != nil && s.candidate != next {
			s.candidate.Close()
		}
		s.candidate = nil
		return err
	}
	previous := s.candidate
	if next == s.handle {
		s.candidate = nil
	} else {
		s.candidate = next
	}
	if previous != nil && previous != next {
		previous.Close()
	}
	return nil
}

// Fold accepts S + coefficient*Z as the new snapshot.
//
// The candidate is built by the same constructor the probes used, written
// beside the policy, loaded, applied, and only then renamed over the policy;
// the old handle is closed after the host has let go of it. There is no state
// in which the file holds half a candidate, and none in which the file holds a
// policy the engine is not running.
//
// The apply comes before the rename, and the order is the point. A file on
// disk is not evidence that a policy is in effect -- the addendum says a hash
// of the changed file, on its own, does not prove the adapter was applied. Do
// it the other way round and an apply that fails leaves the disk holding the
// new policy whilst the engine, the digest and the body all still describe the
// old one; the next reader believes the file. This way an apply that fails
// removes the file it could not honour and restores the previous handle, so
// the three keep saying the same thing.
//
// The first version of the engine did not re-apply after a fold at all. A
// twenty-step run then measured 0.5086126903 twenty times, because every
// measurement read the base model with no adapter. The apply is the step, not
// a detail of it.
func (s *Snapshot) Fold(d Direction, coefficient float64) (int64, error) {
	candidate, err := BuildCandidate(s, s.displacement(d), coefficient)
	if err != nil {
		return 0, err
	}
	loaded, written, err := s.load(candidate, filepath.Dir(s.path), ".fold-")
	if err != nil {
		return 0, err
	}
	if err := s.apply(loaded); err != nil {
		loaded.Close()
		os.Remove(written)
		return 0, fmt.Errorf("cluster: the candidate was built but not applied, and the policy on disk is unchanged: %w", err)
	}
	if err := os.Rename(written, s.path); err != nil {
		// The engine is running the candidate and the disk is not. Put the
		// engine back on the policy the file still holds, so the two agree
		// again; a snapshot that outlived its file is worse than a failed fold.
		if restored := s.apply(s.handle); restored != nil {
			os.Remove(written)
			return 0, fmt.Errorf("cluster: the candidate could not be written (%v) and the previous policy could not be restored: %w", err, restored)
		}
		loaded.Close()
		os.Remove(written)
		return 0, err
	}
	old := s.handle
	s.handle = loaded
	s.candidate = nil
	s.body = candidate.body
	s.digest = candidate.Digest
	s.drawn = nil
	old.Close()
	return candidate.Moved, nil
}

// Close detaches the host from every handle before freeing any of them.
func (s *Snapshot) Close() error {
	err := s.host.SetAdapters(nil, nil)
	if s.candidate != nil {
		s.candidate.Close()
		s.candidate = nil
	}
	if s.handle != nil {
		s.handle.Close()
		s.handle = nil
	}
	return err
}
