package adapter

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
)

// LoRATarget is one base tensor geometry obtained from the admitted GGUF.
// Input and Output are in ggml order, matching lora_a [input, rank] and
// lora_b [rank, output].
type LoRATarget struct {
	Name   string
	Input  uint64
	Output uint64
}

// WriteInitialLoRA creates a deterministic F32 LoRA snapshot from measured
// base tensor geometry. A starts with a variance scaled by its input width and
// B starts at zero, so the initial adapter is an exact no-op while a symmetric
// factor-space perturbation has a non-zero first derivative through B.
func WriteInitialLoRA(path, architecture string, rank uint64, alpha float32, seed uint64, targets []LoRATarget) (string, error) {
	if path == "" || !filepath.IsAbs(path) {
		return "", errors.New("cluster: initial adapter path has to be absolute")
	}
	if architecture == "" || strings.ContainsAny(architecture, "\x00/\\") {
		return "", errors.New("cluster: initial adapter architecture is invalid")
	}
	if rank == 0 || rank > 256 || alpha <= 0 || math.IsNaN(float64(alpha)) || math.IsInf(float64(alpha), 0) {
		return "", errors.New("cluster: initial adapter rank and alpha are invalid")
	}
	if len(targets) == 0 || len(targets) > 64 {
		return "", errors.New("cluster: initial adapter needs between one and 64 targets")
	}

	tensors := make([]initialLoRATensor, 0, len(targets)*2)
	seen := make(map[string]bool, len(targets))
	stream := Direction{Seed: seed}.stream()
	for _, target := range targets {
		if target.Name == "" || !strings.HasSuffix(target.Name, ".weight") || strings.ContainsRune(target.Name, '\x00') {
			return "", fmt.Errorf("cluster: initial adapter target %q is invalid", target.Name)
		}
		if seen[target.Name] {
			return "", fmt.Errorf("cluster: initial adapter repeats target %s", target.Name)
		}
		seen[target.Name] = true
		if target.Input == 0 || target.Output == 0 || target.Input > 1<<20 || target.Output > 1<<20 {
			return "", fmt.Errorf("cluster: initial adapter target %s has invalid geometry", target.Name)
		}
		if target.Input > math.MaxUint64/rank || target.Output > math.MaxUint64/rank {
			return "", fmt.Errorf("cluster: initial adapter target %s is too large", target.Name)
		}
		if target.Input*rank+target.Output*rank > 64<<20 {
			return "", fmt.Errorf("cluster: initial adapter target %s exceeds the bounded factor size", target.Name)
		}
		a := make([]float32, target.Input*rank)
		stddev := 1 / math.Sqrt(float64(target.Input))
		for i := range a {
			a[i] = float32(stream.NormFloat64() * stddev)
		}
		tensors = append(tensors,
			initialLoRATensor{Name: target.Name + ".lora_a", Dims: []uint64{target.Input, rank}, Values: a},
			initialLoRATensor{Name: target.Name + ".lora_b", Dims: []uint64{rank, target.Output}, Values: make([]float32, rank*target.Output)},
		)
	}

	body, err := encodeInitialLoRA(architecture, alpha, tensors)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return "", err
	}
	if err := writeAdapterAtomically(body, path, ".initial-lora-"); err != nil {
		return "", err
	}
	return Digest(body), nil
}

type initialLoRATensor struct {
	Name   string
	Dims   []uint64
	Values []float32
}

func encodeInitialLoRA(architecture string, alpha float32, tensors []initialLoRATensor) ([]byte, error) {
	const alignment = uint64(32)
	var header bytes.Buffer
	putU32 := func(value uint32) error { return binary.Write(&header, binary.LittleEndian, value) }
	putU64 := func(value uint64) error { return binary.Write(&header, binary.LittleEndian, value) }
	putString := func(value string) error {
		if err := putU64(uint64(len(value))); err != nil {
			return err
		}
		_, err := header.WriteString(value)
		return err
	}
	putStringValue := func(key, value string) error {
		if err := putString(key); err != nil {
			return err
		}
		if err := putU32(8); err != nil {
			return err
		}
		return putString(value)
	}

	if err := putU32(ggufMagic); err != nil {
		return nil, err
	}
	if err := putU32(3); err != nil {
		return nil, err
	}
	if err := putU64(uint64(len(tensors))); err != nil {
		return nil, err
	}
	if err := putU64(5); err != nil {
		return nil, err
	}
	for _, pair := range [][2]string{{"general.type", "adapter"}, {"general.architecture", architecture}, {"adapter.type", "lora"}} {
		if err := putStringValue(pair[0], pair[1]); err != nil {
			return nil, err
		}
	}
	if err := putString("adapter.lora.alpha"); err != nil {
		return nil, err
	}
	if err := putU32(6); err != nil {
		return nil, err
	}
	if err := putU32(math.Float32bits(alpha)); err != nil {
		return nil, err
	}
	if err := putString("general.alignment"); err != nil {
		return nil, err
	}
	if err := putU32(4); err != nil {
		return nil, err
	}
	if err := putU32(uint32(alignment)); err != nil {
		return nil, err
	}

	var data bytes.Buffer
	for _, tensor := range tensors {
		if err := putString(tensor.Name); err != nil {
			return nil, err
		}
		if err := putU32(uint32(len(tensor.Dims))); err != nil {
			return nil, err
		}
		for _, dimension := range tensor.Dims {
			if err := putU64(dimension); err != nil {
				return nil, err
			}
		}
		if err := putU32(ggufF32); err != nil {
			return nil, err
		}
		if err := putU64(uint64(data.Len())); err != nil {
			return nil, err
		}
		for _, value := range tensor.Values {
			if err := binary.Write(&data, binary.LittleEndian, math.Float32bits(value)); err != nil {
				return nil, err
			}
		}
		for uint64(data.Len())%alignment != 0 {
			_ = data.WriteByte(0)
		}
	}
	for uint64(header.Len())%alignment != 0 {
		_ = header.WriteByte(0)
	}
	_, _ = header.Write(data.Bytes())
	return header.Bytes(), nil
}
