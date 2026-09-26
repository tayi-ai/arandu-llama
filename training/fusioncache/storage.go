package fusioncache

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

type boundedBuffer struct {
	bytes.Buffer
	maximum int64
}

func (b *boundedBuffer) Write(p []byte) (int, error) {
	if int64(len(p)) > b.maximum-int64(b.Len()) {
		return 0, fmt.Errorf("%w: JSON exceeds byte limit", ErrContract)
	}
	return b.Buffer.Write(p)
}

func encodeBounded(value any, maximum int64) ([]byte, error) {
	if maximum <= 0 {
		return nil, fmt.Errorf("%w: positive byte limit required", ErrContract)
	}
	b := boundedBuffer{maximum: maximum}
	if err := json.NewEncoder(&b).Encode(value); err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}

func decodeStrict(data []byte, value any) error {
	d := json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	if err := d.Decode(value); err != nil {
		return fmt.Errorf("%w: invalid JSON: %v", ErrContract, err)
	}
	var trailing any
	if err := d.Decode(&trailing); err != io.EOF {
		return fmt.Errorf("%w: trailing JSON", ErrContract)
	}
	return nil
}

func byteDigest(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// WriteAtomic writes a bounded, validated cache, syncs it and installs it without
// replacing any existing path. Repeating an identical write is idempotent.
// The caller owns directory creation and authorization for this filesystem path.
func WriteAtomic(path string, c Cache, e Expectation, limits Limits) (Receipt, error) {
	if err := ValidateAgainstStudent(c, e, limits); err != nil {
		return Receipt{}, err
	}
	encoded, err := encodeBounded(c, limits.MaxBytes)
	if err != nil {
		return Receipt{}, err
	}
	receipt := Receipt{Location: path, SHA256: byteDigest(encoded), Bytes: int64(len(encoded))}
	temporary, err := os.CreateTemp(filepath.Dir(path), ".fusioncache-*")
	if err != nil {
		return Receipt{}, err
	}
	name := temporary.Name()
	defer os.Remove(name)
	if _, err := temporary.Write(encoded); err != nil {
		return Receipt{}, errors.Join(err, temporary.Close())
	}
	if err := temporary.Sync(); err != nil {
		return Receipt{}, errors.Join(err, temporary.Close())
	}
	if err := temporary.Close(); err != nil {
		return Receipt{}, err
	}
	// A hard link installs a complete inode atomically and refuses replacement.
	if err := os.Link(name, path); err != nil {
		if !errors.Is(err, os.ErrExist) {
			return Receipt{}, err
		}
		if _, err := Read(path, receipt.SHA256, e, limits); err != nil {
			return Receipt{}, fmt.Errorf("%w: existing artifact differs: %v", ErrContract, err)
		}
	}
	directory, err := os.Open(filepath.Dir(path))
	if err != nil {
		return Receipt{}, err
	}
	if err := errors.Join(directory.Sync(), directory.Close()); err != nil {
		return Receipt{}, err
	}
	return receipt, nil
}

// Read verifies the expected external byte digest before decoding and validating
// the complete cache. A torn write, extra bytes or an altered prefix is refused.
func Read(path, expectedSHA256 string, e Expectation, limits Limits) (Cache, error) {
	if err := ValidateExpectation(e, limits); err != nil {
		return Cache{}, err
	}
	if !validSHA(expectedSHA256) {
		return Cache{}, fmt.Errorf("%w: expected SHA-256 required", ErrContract)
	}
	file, err := os.Open(path)
	if err != nil {
		return Cache{}, err
	}
	info, statErr := file.Stat()
	if statErr != nil || !info.Mode().IsRegular() || info.Size() > limits.MaxBytes {
		return Cache{}, errors.Join(fmt.Errorf("%w: file type or size invalid", ErrContract), statErr, file.Close())
	}
	data, readErr := io.ReadAll(io.LimitReader(file, limits.MaxBytes+1))
	if err := errors.Join(readErr, file.Close()); err != nil {
		return Cache{}, err
	}
	if int64(len(data)) > limits.MaxBytes || byteDigest(data) != expectedSHA256 {
		return Cache{}, fmt.Errorf("%w: artifact byte limit or SHA-256 differs", ErrContract)
	}
	var cache Cache
	if err := decodeStrict(data, &cache); err != nil {
		return Cache{}, err
	}
	if err := ValidateAgainstStudent(cache, e, limits); err != nil {
		return Cache{}, err
	}
	return cache, nil
}

// DirectorySink stores each cache under its content digest in an existing directory.
type DirectorySink struct{ Directory string }

// Store persists one immutable cache with cancellation checked before writing.
func (s DirectorySink) Store(ctx context.Context, c Cache, e Expectation, limits Limits) (Receipt, error) {
	if ctx == nil {
		return Receipt{}, fmt.Errorf("%w: context required", ErrContract)
	}
	if err := ctx.Err(); err != nil {
		return Receipt{}, err
	}
	if err := ValidateAgainstStudent(c, e, limits); err != nil {
		return Receipt{}, err
	}
	data, err := encodeBounded(c, limits.MaxBytes)
	if err != nil {
		return Receipt{}, err
	}
	return WriteAtomic(filepath.Join(s.Directory, byteDigest(data)+".json"), c, e, limits)
}
