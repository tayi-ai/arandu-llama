package pipeline

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"
)

type durableState struct {
	Version  int             `json:"version"`
	Identity ReceiptIdentity `json:"identity"`
	Receipts []StageReceipt  `json:"receipts"`
}

type durableSession struct {
	runtime   *DurableRuntime
	execution Execution
	directory string
	root      *os.Root
	lock      *os.File
	state     durableState
}

func (r *DurableRuntime) openExecution(ctx context.Context, x Execution) (_ *durableSession, err error) {
	if err := r.validate(ctx, x.Recipe, x.Placement); err != nil {
		return nil, err
	}
	if !identifier(x.RunID) || !identifier(x.TenantID) || x.Generation == 0 || !digest(x.TargetSHA256) {
		return nil, ErrDurableIdentity
	}
	x = cloneExecution(x)
	recipeDigest, _ := x.Recipe.Digest()
	identity := ReceiptIdentity{RunID: x.RunID, TenantID: x.TenantID, RecipeSHA256: recipeDigest, TargetSHA256: x.TargetSHA256, PlacementSHA256: jsonSHA256(x.Placement), Generation: x.Generation}
	if err := os.MkdirAll(r.config.Root, 0700); err != nil {
		return nil, err
	}
	if info, err := os.Lstat(r.config.Root); err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return nil, ErrDurableConfiguration
	}
	root, err := os.OpenRoot(r.config.Root)
	if err != nil {
		return nil, err
	}
	defer root.Close()
	name := jsonSHA256(struct{ Tenant, Run string }{x.TenantID, x.RunID})
	if err := root.Mkdir(name, 0700); err != nil && !errors.Is(err, os.ErrExist) {
		return nil, err
	}
	if err := syncDirectory(root); err != nil {
		return nil, err
	}
	if info, err := root.Lstat(name); err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return nil, ErrDurableReceipt
	}
	runRoot, err := root.OpenRoot(name)
	if err != nil {
		return nil, err
	}
	s := &durableSession{runtime: r, execution: x, directory: filepath.Join(r.config.Root, name), root: runRoot}
	defer func() {
		if err != nil {
			s.close()
		}
	}()
	s.lock, err = acquireDurableLock(runRoot)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := runRoot.Mkdir("artifacts", 0700); err != nil && !errors.Is(err, os.ErrExist) {
		return nil, err
	}
	if info, err := runRoot.Lstat("artifacts"); err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return nil, ErrDurableReceipt
	}
	body, err := readDurableFile(runRoot, "state.json", r.config.Limits.MaxStateBytes)
	if errors.Is(err, os.ErrNotExist) {
		// Existing output without its execution binding is never treated as a new run.
		files, openErr := runRoot.Open("artifacts")
		if openErr != nil {
			return nil, openErr
		}
		names, readErr := files.Readdirnames(1)
		closeErr := files.Close()
		if len(names) > 0 || readErr != io.EOF || closeErr != nil {
			return nil, ErrDurableReceipt
		}
		s.state = durableState{Version: 1, Identity: identity}
		if err := s.save(); err != nil {
			return nil, err
		}
		return s, nil
	}
	if err != nil {
		return nil, err
	}
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.DisallowUnknownFields()
	if decoder.Decode(&s.state) != nil || decoder.Decode(new(any)) != io.EOF {
		return nil, ErrDurableReceipt
	}
	canonical, _ := json.Marshal(s.state)
	if !bytes.Equal(canonical, body) || s.state.Version != 1 || s.state.Identity.Generation == 0 || len(s.state.Receipts) > r.config.Limits.MaxReceipts {
		return nil, ErrDurableReceipt
	}
	if !sameIdentity(identity, s.state.Identity) {
		return nil, ErrDurableIdentity
	}
	if identity.Generation < s.state.Identity.Generation {
		return nil, ErrDurableFence
	}
	if identity.Generation > s.state.Identity.Generation {
		s.state.Identity.Generation = identity.Generation
		if err := s.save(); err != nil {
			return nil, err
		}
	}
	return s, nil
}

func (s *durableSession) close() {
	if s.lock != nil {
		s.lock.Close() // Closing the descriptor releases the OS lock, even after a crash.
	}
	if s.root != nil {
		s.root.Close()
	}
}

func (s *durableSession) save() error {
	body, err := json.Marshal(s.state)
	if err != nil || int64(len(body)) > s.runtime.config.Limits.MaxStateBytes {
		return ErrDurableReceipt
	}
	var nonce [16]byte
	if _, err := rand.Read(nonce[:]); err != nil {
		return err
	}
	name := ".state-" + hex.EncodeToString(nonce[:])
	file, err := s.root.OpenFile(name, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	defer s.root.Remove(name)
	_, writeErr := file.Write(body)
	if err := errors.Join(writeErr, file.Sync(), file.Close()); err != nil {
		return err
	}
	if err := s.root.Rename(name, "state.json"); err != nil {
		return err
	}
	return syncDirectory(s.root)
}

func (s *durableSession) verify(ctx context.Context) error {
	if len(s.state.Receipts) > s.runtime.config.Limits.MaxReceipts {
		return ErrDurableReceipt
	}
	for index, receipt := range s.state.Receipts {
		if err := s.checkReceipt(ctx, receipt, s.state.Receipts[:index]); err != nil {
			return err
		}
	}
	return ctx.Err()
}

func (s *durableSession) commit(ctx context.Context, index int, result StageResult) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if index != s.nextStage() || len(s.state.Receipts) >= s.runtime.config.Limits.MaxReceipts {
		return ErrDurableReceipt
	}
	result.Artifacts = slices.Clone(result.Artifacts)
	receipt := StageReceipt{Version: 1, Identity: s.state.Identity, StageID: s.execution.Recipe.Stages[index].ID, Result: result}
	if n := len(s.state.Receipts); n > 0 {
		receipt.PreviousSHA256 = s.state.Receipts[n-1].SHA256
	}
	receipt.SHA256 = receiptSHA256(receipt)
	if err := s.checkReceipt(ctx, receipt, s.state.Receipts); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	s.state.Receipts = append(s.state.Receipts, receipt)
	// Any save error stops this invocation. A rename followed by fsync failure
	// has uncertain durability; the next locked reconciliation reads the disk.
	return s.save()
}

func (s *durableSession) checkReceipt(ctx context.Context, receipt StageReceipt, prior []StageReceipt) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if receipt.Version != 1 || !sameIdentity(receipt.Identity, s.state.Identity) || receipt.Identity.Generation == 0 || receipt.Identity.Generation > s.state.Identity.Generation || receipt.SHA256 != receiptSHA256(receipt) {
		return ErrDurableReceipt
	}
	index := 0
	var previousStep int64
	previousDigest := ""
	var previousGeneration uint64
	for _, previous := range prior {
		previousDigest = previous.SHA256
		previousGeneration = previous.Identity.Generation
		previousStep = previous.Result.Steps
		if previous.Result.Complete {
			index++
			previousStep = 0
		}
	}
	if index >= len(s.execution.Recipe.Stages) || receipt.PreviousSHA256 != previousDigest || receipt.Identity.Generation < previousGeneration {
		return ErrDurableReceipt
	}
	stage := s.execution.Recipe.Stages[index]
	v := receipt.Result
	if receipt.StageID != stage.ID || v.Steps <= 0 || v.Steps > int64(stage.MaxSteps) || v.Steps < previousStep || v.Steps == previousStep && !v.Complete || len(v.Artifacts) == 0 || len(v.Artifacts) > s.runtime.config.Limits.MaxArtifactsPerReceipt {
		return ErrDurableReceipt
	}
	if err := s.verifyArtifacts(ctx, v.Artifacts, prior); err != nil {
		return err
	}
	bounded, cancel := context.WithTimeout(ctx, time.Duration(stage.TimeoutSeconds)*time.Second)
	defer cancel()
	if err := s.runtime.config.Handlers[stage.Phase].Verify(bounded, s.stageContext(index, prior), cloneReceipt(receipt)); err != nil {
		return fmt.Errorf("%w: stage %s verification: %w", ErrDurableReceipt, stage.ID, err)
	}
	return bounded.Err()
}

func (s *durableSession) verifyArtifacts(ctx context.Context, artifacts []StageArtifact, prior []StageReceipt) error {
	limits := s.runtime.config.Limits
	known := map[string]StageArtifact{}
	var total int64
	for _, receipt := range prior {
		for _, artifact := range receipt.Result.Artifacts {
			if _, exists := known[artifact.Path]; !exists {
				if artifact.Bytes > limits.MaxTotalArtifactBytes-total {
					return ErrDurableReceipt
				}
				total += artifact.Bytes
				known[artifact.Path] = artifact
			}
		}
	}
	seen := map[string]bool{}
	for _, artifact := range artifacts {
		if !filepath.IsLocal(artifact.Path) || artifact.Path == "." || filepath.Clean(artifact.Path) != artifact.Path || strings.ContainsRune(artifact.Path, 0) || len(artifact.Path) > 1024 || !digest(artifact.SHA256) || artifact.Bytes <= 0 || artifact.Bytes > limits.MaxArtifactBytes || seen[artifact.Path] {
			return ErrDurableReceipt
		}
		seen[artifact.Path] = true
		if previous, exists := known[artifact.Path]; exists {
			if previous != artifact {
				return ErrDurableReceipt
			}
		} else {
			if artifact.Bytes > limits.MaxTotalArtifactBytes-total {
				return ErrDurableReceipt
			}
			total += artifact.Bytes
		}
		file, err := openDurableRegular(s.root, filepath.Join("artifacts", artifact.Path), os.O_RDONLY, 0)
		if err != nil {
			return errors.Join(ErrDurableReceipt, err)
		}
		info, err := file.Stat()
		if err != nil || info.Size() != artifact.Bytes {
			file.Close()
			return ErrDurableReceipt
		}
		hash := sha256.New()
		reader := contextReader{ctx: ctx, reader: file}
		n, copyErr := io.Copy(hash, io.LimitReader(reader, artifact.Bytes+1))
		closeErr := file.Close()
		if copyErr != nil || closeErr != nil || n != artifact.Bytes || hex.EncodeToString(hash.Sum(nil)) != artifact.SHA256 {
			return errors.Join(ErrDurableReceipt, copyErr, closeErr)
		}
	}
	return ctx.Err()
}

func sameIdentity(a, b ReceiptIdentity) bool { a.Generation, b.Generation = 0, 0; return a == b }

func syncDirectory(root *os.Root) error {
	directory, err := root.Open(".")
	if err != nil {
		return err
	}
	return errors.Join(directory.Sync(), directory.Close())
}

func readDurableFile(root *os.Root, name string, limit int64) ([]byte, error) {
	file, err := openDurableRegular(root, name, os.O_RDONLY, 0)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil || info.Size() < 1 || info.Size() > limit {
		return nil, ErrDurableReceipt
	}
	body, err := io.ReadAll(io.LimitReader(file, limit+1))
	if err != nil || int64(len(body)) > limit {
		return nil, ErrDurableReceipt
	}
	return body, nil
}

type contextReader struct {
	ctx    context.Context
	reader io.Reader
}

func (r contextReader) Read(p []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	return r.reader.Read(p)
}
