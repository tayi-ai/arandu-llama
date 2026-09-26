package adapter

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
)

// The GGUF container, read far enough to place every tensor, and the direction
// writer that copies one.
//
// A zeroth-order step moves the parameters along a random direction and reads
// how the loss responded. The engine can only apply an adapter that exists as a
// file -- llama_adapter_lora_init takes a path, not a buffer -- so what it
// applies for a probe is a candidate: a copy of the policy with c times the
// direction added to its factors, built by BuildCandidate in Candidate.go and
// written per probe. The earlier design loaded the direction itself as a second
// adapter and rescaled it, which cost 0.4 ms a probe against 0.014 s a file;
// T02 measured that it probes along ZB*ZA while the step moves along
// B*ZA + ZB*A, so the cost of a file per probe is accepted until zo:probe
// measures it.
//
// A file is built by copying a real adapter and replacing its numbers. The
// alternative -- composing a GGUF from nothing -- would mean this code owning an
// opinion about tensor names, shapes and metadata, and an opinion that drifts
// from what PEFT actually exported is an adapter the engine loads happily and
// applies to the wrong weights.

// GGUF is what this package needs to know about the container: where the tensors
// are, how big they are, and nothing else.
type GGUF struct {
	// DataOffset is where the tensor data begins, after the header, the metadata
	// and the alignment padding.
	DataOffset int64
	Tensors    []GGUFTensor
}

// GGUFTensor is one tensor's placement inside the data section.
type GGUFTensor struct {
	Name string
	// Offset is relative to the file's DataOffset, which is how GGUF stores it.
	Offset int64
	// Dims are the dimensions in ggml order: ne[0] varies fastest. For a LoRA
	// factor they say which of A and B a tensor is -- lora_a is [n_in, r] and
	// lora_b is [r, n_out] -- which is what a test reads to check that a
	// candidate moved each factor by its own displacement.
	Dims []uint64
	// Elements is the product of the dimensions.
	Elements int64
	// Type is the GGUF type tag. Only F32 and F16 are handled here, because they
	// are what an adapter exported for training contains; a quantised tensor
	// cannot carry a perturbation drawn from a continuous distribution.
	Type uint32
}

// The two GGUF type tags an adapter is written in.
const (
	ggufF32 uint32 = 0
	ggufF16 uint32 = 1
)

// ggufMagic is "GGUF" little endian.
const ggufMagic uint32 = 0x46554747

// ReadGGUF parses enough of the container to locate every tensor.
//
// It reads the metadata only to step over it. Interpreting those keys is how a
// reader acquires opinions it then has to keep in sync with the writer, and this
// one has no reason to hold any: the file it reads was written by the exporter
// whose output is already admitted by AdmitAdapterConfig.
func ReadGGUF(path string) (GGUF, error) {
	file, err := os.Open(path)
	if err != nil {
		return GGUF{}, err
	}
	defer file.Close()
	reader := &ggufReader{r: file}

	if magic := reader.u32(); magic != ggufMagic {
		return GGUF{}, fmt.Errorf("cluster: %s does not start with the GGUF magic", filepath.Base(path))
	}
	version := reader.u32()
	if version != 3 {
		return GGUF{}, fmt.Errorf("cluster: %s is GGUF version %d; this reader was written against version 3 and refuses rather than guessing", filepath.Base(path), version)
	}
	tensorCount := reader.u64()
	kvCount := reader.u64()
	if reader.err != nil {
		return GGUF{}, reader.err
	}

	// The alignment is a metadata key with a default. It decides where the data
	// section starts, so it is the one key worth reading.
	alignment := int64(32)
	for i := uint64(0); i < kvCount; i++ {
		key := reader.str()
		value := reader.value()
		if key == "general.alignment" {
			if v, ok := value.(uint64); ok && v > 0 {
				alignment = int64(v)
			}
		}
		if reader.err != nil {
			return GGUF{}, fmt.Errorf("cluster: reading metadata key %d of %s: %w", i, filepath.Base(path), reader.err)
		}
	}

	tensors := make([]GGUFTensor, 0, tensorCount)
	for i := uint64(0); i < tensorCount; i++ {
		name := reader.str()
		count := reader.u32()
		if count > 4 {
			return GGUF{}, fmt.Errorf("cluster: tensor %d of %s declares %d dimensions; ggml allows four", i, filepath.Base(path), count)
		}
		dims := make([]uint64, count)
		elements := int64(1)
		for d := range dims {
			dims[d] = reader.u64()
			elements *= int64(dims[d])
		}
		kind := reader.u32()
		offset := int64(reader.u64())
		if reader.err != nil {
			return GGUF{}, fmt.Errorf("cluster: reading tensor %d of %s: %w", i, filepath.Base(path), reader.err)
		}
		tensors = append(tensors, GGUFTensor{Name: name, Offset: offset, Dims: dims, Elements: elements, Type: kind})
	}

	// The data section starts at the next multiple of the alignment.
	position := reader.position
	if remainder := position % alignment; remainder != 0 {
		position += alignment - remainder
	}
	return GGUF{DataOffset: position, Tensors: tensors}, nil
}

// ggufReader tracks how far it has read, because the data offset is defined by
// where the header ended rather than written down anywhere.
type ggufReader struct {
	r        io.Reader
	position int64
	err      error
}

func (g *ggufReader) read(into []byte) {
	if g.err != nil {
		return
	}
	n, err := io.ReadFull(g.r, into)
	g.position += int64(n)
	g.err = err
}

func (g *ggufReader) u32() uint32 {
	var b [4]byte
	g.read(b[:])
	return binary.LittleEndian.Uint32(b[:])
}

func (g *ggufReader) u64() uint64 {
	var b [8]byte
	g.read(b[:])
	return binary.LittleEndian.Uint64(b[:])
}

func (g *ggufReader) str() string {
	length := g.u64()
	if g.err != nil {
		return ""
	}
	// A length this large is a malformed file, and allocating on it is how a
	// parser turns a corrupt input into an out-of-memory kill.
	if length > 1<<20 {
		g.err = fmt.Errorf("cluster: a GGUF string of %d bytes is not credible", length)
		return ""
	}
	body := make([]byte, length)
	g.read(body)
	return string(body)
}

// value steps over one metadata value of any type, returning it only for the
// scalar cases the caller might care about.
func (g *ggufReader) value() any {
	kind := g.u32()
	return g.valueOf(kind)
}

func (g *ggufReader) valueOf(kind uint32) any {
	if g.err != nil {
		return nil
	}
	switch kind {
	case 0, 1: // uint8, int8
		var b [1]byte
		g.read(b[:])
		return uint64(b[0])
	case 2, 3: // uint16, int16
		var b [2]byte
		g.read(b[:])
		return uint64(binary.LittleEndian.Uint16(b[:]))
	case 4, 5: // uint32, int32
		return uint64(g.u32())
	case 6: // float32
		return uint64(g.u32())
	case 7: // bool
		var b [1]byte
		g.read(b[:])
		return uint64(b[0])
	case 8: // string
		return g.str()
	case 9: // array
		elementKind := g.u32()
		count := g.u64()
		for i := uint64(0); i < count && g.err == nil; i++ {
			g.valueOf(elementKind)
		}
		return nil
	case 10, 11, 12: // uint64, int64, float64
		return g.u64()
	default:
		g.err = fmt.Errorf("cluster: GGUF metadata type %d is not one this reader knows", kind)
		return nil
	}
}

// WriteDirection copies an adapter and replaces every number in it with a sample
// from the direction, producing a file the engine can load as a perturbation.
//
// The container is copied byte for byte and only the tensor payload is
// rewritten, so the result has the same tensor names, shapes and metadata as the
// adapter it came from. That is deliberate: the engine matches an adapter to the
// base model by those names and shapes, and a direction that disagreed with the
// adapter it perturbs would be applied to different weights.
//
// The perturbation is written at unit scale. Nothing in the training path
// loads such a file any more -- a probe applies a candidate built by
// BuildCandidate, not a direction -- and it stays as the way to look at a
// direction with the engine's own reader.
func WriteDirection(templatePath, outPath string, d Direction) error {
	container, err := ReadGGUF(templatePath)
	if err != nil {
		return err
	}
	if len(container.Tensors) == 0 {
		return fmt.Errorf("cluster: %s declares no tensors", filepath.Base(templatePath))
	}
	body, err := os.ReadFile(templatePath)
	if err != nil {
		return err
	}
	source := d.stream()

	written := int64(0)
	for _, tensor := range container.Tensors {
		start := container.DataOffset + tensor.Offset
		switch tensor.Type {
		case ggufF32:
			end := start + tensor.Elements*4
			if end > int64(len(body)) {
				return fmt.Errorf("cluster: tensor %s runs past the end of %s", tensor.Name, filepath.Base(templatePath))
			}
			for i := int64(0); i < tensor.Elements; i++ {
				binary.LittleEndian.PutUint32(body[start+i*4:], math.Float32bits(float32(source.NormFloat64())))
			}
		case ggufF16:
			end := start + tensor.Elements*2
			if end > int64(len(body)) {
				return fmt.Errorf("cluster: tensor %s runs past the end of %s", tensor.Name, filepath.Base(templatePath))
			}
			for i := int64(0); i < tensor.Elements; i++ {
				binary.LittleEndian.PutUint16(body[start+i*2:], float16Bits(float32(source.NormFloat64())))
			}
		default:
			// A quantised tensor cannot hold a sample from a continuous
			// distribution, and rounding one into a codebook would give a
			// direction whose distribution is not the one the estimator assumes.
			return fmt.Errorf("cluster: tensor %s has GGUF type %d; a direction can only be written into F32 or F16", tensor.Name, tensor.Type)
		}
		written += tensor.Elements
	}

	// Written to a temporary file and renamed, so a reader never opens a
	// half-written direction and perturbs along part of one.
	temporary, err := os.CreateTemp(filepath.Dir(outPath), ".direction-")
	if err != nil {
		return err
	}
	if _, err := temporary.Write(body); err != nil {
		temporary.Close()
		os.Remove(temporary.Name())
		return err
	}
	if err := temporary.Close(); err != nil {
		os.Remove(temporary.Name())
		return err
	}
	if err := os.Rename(temporary.Name(), outPath); err != nil {
		os.Remove(temporary.Name())
		return err
	}
	return nil
}

// ReadTensorValues decodes one tensor of an adapter by name, from the file.
//
// From the file and not from the engine, deliberately: the reference the engine
// is checked against has to be built out of something the engine did not supply,
// or the check is the engine agreeing with itself. The digest the snapshot
// verifies at load is what ties these bytes to the ones the engine read.
func ReadTensorValues(path, name string) ([]float32, error) {
	container, err := ReadGGUF(path)
	if err != nil {
		return nil, err
	}
	body, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	values, ok := tensorValues(container, body, name)
	if !ok {
		return nil, fmt.Errorf("cluster: %s holds no F32 tensor named %s", filepath.Base(path), name)
	}
	return values, nil
}

// StructuredCell names one number of an adapter by tensor and flat index.
//
// Index is in ggml order, where ne[0] varies fastest. For a lora_a of
// ne = [n_in, r], element (i, r) sits at r*n_in + i; for a lora_b of
// ne = [r, n_out], element (r, j) sits at j*r + r. Naming a cell by index
// rather than by coordinates keeps this type from owning an opinion about
// which dimension is which -- the template already says.
type StructuredCell struct {
	Tensor string
	Index  int64
	Value  float32
}

// WriteStructured copies an adapter, zeroes every number in it, and sets the
// named cells.
//
// It exists for one job: a positive control over the engine's LoRA arithmetic.
// The controls that leave B at zero prove only that a contribution of zero is
// applied as zero -- an engine computing 100*s*B*A passes every one of them.
// Catching that needs a contribution that is not zero and whose value is known,
// and knowing the value means owning every number in the file.
//
// Like WriteDirection, the container is copied byte for byte and only the
// payload is rewritten, so the result carries the template's tensor names,
// shapes and metadata. That is what makes the control a control: the engine
// matches an adapter to the base model by those names, and a fixture composed
// from nothing would be testing a layout this repository invented rather than
// the one the adapter under test actually has.
//
// A cell naming a tensor the template does not have, or an index past its end,
// is refused rather than skipped: a control that silently wrote nothing would
// report the engine as correct.
func WriteStructured(templatePath, outPath string, cells []StructuredCell) error {
	container, err := ReadGGUF(templatePath)
	if err != nil {
		return err
	}
	if len(container.Tensors) == 0 {
		return fmt.Errorf("cluster: %s declares no tensors", filepath.Base(templatePath))
	}
	body, err := os.ReadFile(templatePath)
	if err != nil {
		return err
	}

	at := make(map[string]GGUFTensor, len(container.Tensors))
	for _, tensor := range container.Tensors {
		if tensor.Type != ggufF32 {
			// F16 would round the cell values, and a control whose expectation is
			// rounded is not an expectation.
			return fmt.Errorf("cluster: tensor %s has GGUF type %d; a structured fixture can only be written into F32", tensor.Name, tensor.Type)
		}
		start := container.DataOffset + tensor.Offset
		if start+tensor.Elements*4 > int64(len(body)) {
			return fmt.Errorf("cluster: tensor %s runs past the end of %s", tensor.Name, filepath.Base(templatePath))
		}
		for i := int64(0); i < tensor.Elements; i++ {
			binary.LittleEndian.PutUint32(body[start+i*4:], 0)
		}
		at[tensor.Name] = tensor
	}

	for _, cell := range cells {
		tensor, ok := at[cell.Tensor]
		if !ok {
			return fmt.Errorf("cluster: %s has no tensor named %s", filepath.Base(templatePath), cell.Tensor)
		}
		if cell.Index < 0 || cell.Index >= tensor.Elements {
			return fmt.Errorf("cluster: index %d is outside %s, which holds %d elements", cell.Index, cell.Tensor, tensor.Elements)
		}
		offset := container.DataOffset + tensor.Offset + cell.Index*4
		binary.LittleEndian.PutUint32(body[offset:], math.Float32bits(cell.Value))
	}

	return writeAdapterAtomically(body, outPath, ".structured-")
}

// WriteAlpha copies an adapter and rewrites adapter.lora.alpha, leaving every
// number and every other key as it was.
//
// llama.cpp computes the applied scale as adapter_scale * alpha / rank, with
// rank read from the tensor rather than from metadata -- see
// llama_adapter_lora_weight::get_scale. Two adapters that differ only in alpha,
// with the factors compensating, therefore have to produce the same output. That
// equality is how the composition is tested without knowing what the projection
// is fed, which is not observable from outside the graph.
//
// The key is located by its own bytes rather than by re-walking the metadata
// block. It appears once, its value is an F32, and a diagnostic that patched the
// wrong four bytes would fail loudly at load.
func WriteAlpha(templatePath, outPath string, alpha float32) error {
	body, err := os.ReadFile(templatePath)
	if err != nil {
		return err
	}
	at, err := alphaOffset(body)
	if err != nil {
		return fmt.Errorf("cluster: %s: %w", filepath.Base(templatePath), err)
	}
	binary.LittleEndian.PutUint32(body[at:], math.Float32bits(alpha))

	return writeAdapterAtomically(body, outPath, ".alpha-")
}

// AlphaOf reads adapter.lora.alpha out of an adapter's bytes.
//
// It reads the file rather than asking the engine because a caller composing
// fixtures has to know the scale before a model is loaded, and because the
// engine's own reader is the thing under test when it is asked.
func AlphaOf(body []byte) (float32, bool) {
	at, err := alphaOffset(body)
	if err != nil {
		return 0, false
	}
	return math.Float32frombits(binary.LittleEndian.Uint32(body[at:])), true
}

// alphaOffset finds where adapter.lora.alpha's F32 value starts.
//
// The key is located by its own bytes rather than by walking the metadata
// block, because that walk already exists in ReadGGUF and duplicating it is how
// two readers drift. The key appears once and its value is an F32; both are
// checked, so a file that broke either assumption is refused rather than
// silently patched in the wrong place.
func alphaOffset(body []byte) (int, error) {
	const key = "adapter.lora.alpha"
	var prefix []byte
	prefix = binary.LittleEndian.AppendUint64(prefix, uint64(len(key)))
	prefix = append(prefix, key...)

	start := bytes.Index(body, prefix)
	if start < 0 {
		return 0, errors.New("declares no adapter.lora.alpha")
	}
	if bytes.Index(body[start+len(prefix):], prefix) >= 0 {
		return 0, errors.New("names adapter.lora.alpha more than once")
	}
	kind := start + len(prefix)
	if kind+8 > len(body) {
		return 0, errors.New("ends inside adapter.lora.alpha")
	}
	// 6 is the GGUF tag for F32. Anything else means the key is stored in a type
	// this would corrupt.
	if got := binary.LittleEndian.Uint32(body[kind:]); got != 6 {
		return 0, fmt.Errorf("adapter.lora.alpha has GGUF type %d, not F32", got)
	}
	return kind + 4, nil
}

// writeAdapterAtomically is the temp-then-rename WriteDirection uses, shared so
// that no caller here leaves a half-written adapter where a reader can open it.
func writeAdapterAtomically(body []byte, outPath, prefix string) error {
	temporary, err := os.CreateTemp(filepath.Dir(outPath), prefix)
	if err != nil {
		return err
	}
	if _, err := temporary.Write(body); err != nil {
		temporary.Close()
		os.Remove(temporary.Name())
		return err
	}
	if err := temporary.Close(); err != nil {
		os.Remove(temporary.Name())
		return err
	}
	if err := os.Rename(temporary.Name(), outPath); err != nil {
		os.Remove(temporary.Name())
		return err
	}
	return nil
}

// DirectionElements reports how many numbers a direction written from this
// template will carry.
//
// A count different from the admitted adapter identifies different parameters.
func DirectionElements(templatePath string) (int64, error) {
	container, err := ReadGGUF(templatePath)
	if err != nil {
		return 0, err
	}
	total := int64(0)
	for _, tensor := range container.Tensors {
		total += tensor.Elements
	}
	if total == 0 {
		return 0, errors.New("cluster: the template declares no elements")
	}
	return total, nil
}

// float16Bits converts to IEEE half precision.
//
// The cards are Tesla T10, compute capability 7.5, and they have no bfloat16.
// Half is what the adapter is exported in and what the engine reads, so the
// conversion happens here rather than being left to a library that might pick
// the other sixteen-bit format.
func float16Bits(value float32) uint16 {
	bits := math.Float32bits(value)
	sign := uint16((bits >> 16) & 0x8000)
	exponent := int32((bits>>23)&0xff) - 127 + 15
	mantissa := bits & 0x7fffff

	switch {
	case exponent >= 0x1f:
		// Overflow saturates to the largest finite half rather than to infinity:
		// an infinity in a perturbation direction propagates to a loss of nan,
		// and a nan estimate poisons every parameter the step touches.
		return sign | 0x7bff
	case exponent <= 0:
		// Subnormal, or too small to represent. A direction element of zero is a
		// coordinate the step does not move, which is correct behaviour for a
		// magnitude this far below what half precision resolves.
		if exponent < -10 {
			return sign
		}
		mantissa |= 0x800000
		shift := uint32(14 - exponent)
		return sign | uint16(mantissa>>shift)
	default:
		return sign | uint16(exponent)<<10 | uint16(mantissa>>13)
	}
}

// Float16BitsForTest exposes the conversion so a test can check it against the
// inverse. The cards decide this format, and a silent error in it would appear
// as a direction with the wrong spread rather than as a failure.
func Float16BitsForTest(value float32) uint16 { return float16Bits(value) }

// ErrHalfPrecisionPolicy is returned by OpenSnapshot when the policy is stored
// in half precision: an update cannot be folded into it, so it is refused at
// open rather than at the first step.
//
// Repeated small updates can be lost or biased by half-precision rounding.
// The candidate path therefore requires F32 factors rather than silently
// admitting an update with insufficient storage precision.
var ErrHalfPrecisionPolicy = errors.New("cluster: this adapter is stored in half precision, and folding an update into it drifts by more than the update; convert it with --outtype f32")
