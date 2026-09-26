package native

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// ValidateStart requires the whole admitted deadline to fit inside the lease.
// It deliberately does not apply to status or cancellation of an existing job.
func (a Admission) ValidateStart(now time.Time) error {
	if a.Deadline <= 0 || a.Deadline > 5400*time.Second || !a.ExpiresAt.After(a.NotBefore) ||
		now.Before(a.NotBefore) || !now.Before(a.ExpiresAt) || now.Add(a.Deadline).After(a.ExpiresAt) {
		return errors.New("experiment.train: job admission is stale, premature or shorter than its deadline")
	}
	return nil
}

func (a experimentAdmission) window() (time.Time, time.Time, error) {
	before, errBefore := time.Parse(time.RFC3339, a.NotBeforeUTC)
	expires, errExpires := time.Parse(time.RFC3339, a.ExpiresAtUTC)
	_, beforeOffset := before.Zone()
	_, expiryOffset := expires.Zone()
	if errBefore != nil || errExpires != nil || beforeOffset != 0 || expiryOffset != 0 || !expires.After(before) {
		return time.Time{}, time.Time{}, errors.New("invalid admission time window")
	}
	return before, expires, nil
}

func experimentSHA256(value string) bool {
	decoded, err := hex.DecodeString(value)
	return err == nil && len(decoded) == sha256.Size && value == strings.ToLower(value)
}

func experimentHash(value []byte) string {
	digest := sha256.Sum256(value)
	return hex.EncodeToString(digest[:])
}

func experimentWithin(root, path string) bool {
	relative, err := filepath.Rel(root, path)
	return err == nil && relative != ".." && !strings.HasPrefix(relative, ".."+string(filepath.Separator))
}

func experimentDirectory(path string) error {
	if !filepath.IsAbs(path) || filepath.Clean(path) != path {
		return errors.New("directory must be absolute and clean")
	}
	for current := path; ; current = filepath.Dir(current) {
		info, err := os.Lstat(current)
		if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
			return errors.New("directory contains a symlink or non-directory")
		}
		if filepath.Dir(current) == current {
			break
		}
	}
	return nil
}

func experimentReadFile(path string, limit int64) ([]byte, error) {
	info, err := os.Lstat(path)
	if err != nil || !info.Mode().IsRegular() || info.Size() > limit {
		return nil, errors.New("unsafe or oversized bundle file")
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	opened, err := file.Stat()
	if err != nil || !os.SameFile(info, opened) {
		return nil, errors.New("bundle file changed while opening")
	}
	contents, err := io.ReadAll(io.LimitReader(file, limit+1))
	if err != nil || int64(len(contents)) > limit {
		return nil, errors.New("cannot read bounded bundle file")
	}
	return contents, nil
}

// Reject duplicate keys as well as trailing JSON. Otherwise a manifest or
// request could have two meanings to independent readers.
func experimentDecode(contents []byte, target any, strict bool) error {
	scan := json.NewDecoder(bytes.NewReader(contents))
	var walk func(int) error
	walk = func(depth int) error {
		if depth > 64 {
			return errors.New("JSON nesting exceeds limit")
		}
		token, err := scan.Token()
		if err != nil {
			return err
		}
		delimiter, compound := token.(json.Delim)
		if !compound {
			return nil
		}
		if delimiter != '{' && delimiter != '[' {
			return errors.New("invalid JSON delimiter")
		}
		seen := make(map[string]bool)
		for scan.More() {
			if delimiter == '{' {
				key, err := scan.Token()
				if err != nil {
					return err
				}
				name, ok := key.(string)
				if !ok || seen[name] {
					return errors.New("duplicate or invalid JSON key")
				}
				seen[name] = true
			}
			if err := walk(depth + 1); err != nil {
				return err
			}
		}
		_, err = scan.Token()
		return err
	}
	if err := walk(0); err != nil {
		return err
	}
	if _, err := scan.Token(); err != io.EOF {
		return errors.New("trailing JSON")
	}
	decoder := json.NewDecoder(bytes.NewReader(contents))
	if strict {
		decoder.DisallowUnknownFields()
	}
	return decoder.Decode(target)
}

func readTrainingDocument(path, expected string) ([]byte, error) {
	if !filepath.IsAbs(path) || filepath.Clean(path) != path || !experimentSHA256(expected) {
		return nil, errors.New("native experiment: invalid installation document")
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil || !info.Mode().IsRegular() || info.Size() > 1<<20 {
		return nil, errors.New("native experiment: invalid installation document")
	}
	body, err := io.ReadAll(io.LimitReader(f, (1<<20)+1))
	if err != nil {
		return nil, err
	}
	sum := sha256.Sum256(body)
	if len(body) > 1<<20 || hex.EncodeToString(sum[:]) != expected {
		return nil, errors.New("training: installed document digest differs")
	}
	return body, nil
}
