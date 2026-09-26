//go:build libtorch && cgo

package decoder_test

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"slices"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/decoder"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

func TestObservedForwardPreservesValuesAndReportsEveryPlacement(t *testing.T) {
	for _, dtype := range []torch.DType{torch.Float16, torch.Float32} {
		t.Run(map[torch.DType]string{torch.Float16: "fp16", torch.Float32: "fp32"}[dtype], func(t *testing.T) {
			f := newFixture(t, dtype)
			base := baseHash(t, f)
			original := forward(t, f)
			want := read(t, original.Logits)
			var observations []decoder.ForwardObservation
			observed, err := f.model.ForwardObserved(context.Background(), f.tokens, f.limits, func(_ context.Context, item decoder.ForwardObservation) error {
				copy := item
				copy.Shape = slices.Clone(item.Shape)
				observations = append(observations, copy)
				item.Shape[0] = 999 // Callback metadata cannot mutate the tensor.
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = observed.Close() })
			if !reflect.DeepEqual(read(t, observed.Logits), want) || baseHash(t, f) != base {
				t.Fatal("observation changed logits or frozen weights")
			}
			if len(observations) != 3*len(f.model.Layers)+3 {
				t.Fatalf("unexpected observations: %d", len(observations))
			}
			for _, item := range observations {
				if item.Device != torch.CPUDevice() || item.Shape[0] != 1 || !item.AllFinite || item.NonfiniteElements != 0 || item.NonzeroElements == 0 || item.Minimum == nil || item.Maximum == nil || *item.Minimum > *item.Maximum || item.ChunkBytes <= 0 || item.ChunkBytes > decoder.ObservationChunkBytes {
					t.Fatalf("invalid copied observation: %+v", item)
				}
				if _, err := json.Marshal(item); err != nil {
					t.Fatal(err)
				}
			}
			if observations[0].Stage != decoder.StageEmbedding || observations[0].Layer != -1 || observations[0].DType != dtype {
				t.Fatalf("embedding identity: %+v", observations[0])
			}
			previous := observations[0].SHA256
			for layer := range f.model.Layers {
				before, after, decoded := observations[1+3*layer], observations[2+3*layer], observations[3+3*layer]
				if before.Stage != decoder.StageBeforePlacement || after.Stage != decoder.StageAfterPlacement || decoded.Stage != decoder.StageAfterDecoder || before.Layer != layer || after.Layer != layer || decoded.Layer != layer || before.SHA256 != previous || after.SHA256 != before.SHA256 || before.DType != dtype || after.DType != dtype {
					t.Fatalf("layer %d placement attribution differs", layer)
				}
				previous = decoded.SHA256
			}
			norm, logits := observations[len(observations)-2], observations[len(observations)-1]
			if norm.Stage != decoder.StageFinalNorm || norm.Layer != -1 || !slices.Equal(norm.Shape, []int64{1, 3, 4}) || norm.DType != dtype || logits.Stage != decoder.StageLogits || logits.Layer != -1 || logits.DType != torch.Float32 || !slices.Equal(logits.Shape, []int64{1, 2, 6}) {
				t.Fatal("final normalization or logit identity differs")
			}
			body, err := observed.Logits.Bytes()
			if err != nil {
				t.Fatal(err)
			}
			digest := sha256.Sum256(body)
			if logits.SHA256 != hex.EncodeToString(digest[:]) || logits.Bytes != int64(len(body)) || logits.Elements != 12 {
				t.Fatal("observation does not hash exact raw logit bytes")
			}
			nilObserved, err := f.model.ForwardObserved(context.Background(), f.tokens, f.limits, nil)
			if err != nil {
				t.Fatal(err)
			}
			defer nilObserved.Close()
			if !reflect.DeepEqual(read(t, nilObserved.Logits), want) {
				t.Fatal("nil observer changed the original forward")
			}
		})
	}
}

func TestObserverFailureStopsAndLeavesModelReusable(t *testing.T) {
	for _, stage := range []decoder.ForwardStage{decoder.StageEmbedding, decoder.StageBeforePlacement, decoder.StageAfterPlacement, decoder.StageAfterDecoder, decoder.StageFinalNorm, decoder.StageLogits} {
		t.Run(string(stage), func(t *testing.T) {
			f := newFixture(t, torch.Float32)
			want := read(t, forward(t, f).Logits)
			base := baseHash(t, f)
			sentinel := errors.New("observer fixture refusal")
			failed := false
			snapshot, err := f.model.ForwardObserved(context.Background(), f.tokens, f.limits, func(_ context.Context, item decoder.ForwardObservation) error {
				if failed {
					t.Fatal("observer was called after refusal")
				}
				if item.Stage == stage {
					failed = true
					return sentinel
				}
				return nil
			})
			if snapshot != nil || !errors.Is(err, sentinel) || !failed {
				t.Fatalf("observer refusal: snapshot=%v error=%v", snapshot, err)
			}
			if baseHash(t, f) != base || !reflect.DeepEqual(read(t, forward(t, f).Logits), want) {
				t.Fatal("observer refusal invalidated model ownership or values")
			}
		})
	}
}

func TestObservedForwardCancellationStopsBeforeNextStage(t *testing.T) {
	f := newFixture(t, torch.Float32)
	ctx, cancel := context.WithCancel(context.Background())
	calls := 0
	snapshot, err := f.model.ForwardObserved(ctx, f.tokens, f.limits, func(_ context.Context, item decoder.ForwardObservation) error {
		calls++
		if item.Stage == decoder.StageAfterPlacement {
			cancel()
		}
		return nil
	})
	if snapshot != nil || !errors.Is(err, context.Canceled) || calls != 3 {
		t.Fatalf("cancellation: snapshot=%v error=%v calls=%d", snapshot, err, calls)
	}
	snapshot, err = f.model.ForwardObserved(ctx, f.tokens, f.limits, func(context.Context, decoder.ForwardObservation) error {
		t.Fatal("pre-canceled forward delivered observation")
		return nil
	})
	if snapshot != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("pre-canceled forward: %v", err)
	}
	_ = forward(t, f)
}

func TestObservationPreservesHalfBytesAndReportsNonfiniteValues(t *testing.T) {
	f := newFixture(t, torch.Float16)
	bits := []uint16{0, 0x8000, 0x3c00, 0xc000, 1, 0x7bff, 0x7c00, 0x7e01}
	body := make([]byte, 6*4*2)
	for index, value := range bits {
		binary.LittleEndian.PutUint16(body[2*index:], value)
	}
	value, err := torch.FromBytes(body, []int64{6, 4}, torch.Float16, torch.CPUDevice(), false)
	f.model.Embedding = own(t, value, err)
	stop := errors.New("stop after diagnostic observation")
	calls := 0
	snapshot, err := f.model.ForwardObserved(context.Background(), []int64{0, 1}, f.limits, func(_ context.Context, item decoder.ForwardObservation) error {
		calls++
		digest := sha256.Sum256(body[:16])
		if item.Stage != decoder.StageEmbedding || item.AllFinite || item.NonfiniteElements != 2 || item.NonzeroElements != 4 || item.Minimum == nil || *item.Minimum != -2 || item.Maximum == nil || *item.Maximum != 65504 || item.SHA256 != hex.EncodeToString(digest[:]) {
			t.Fatalf("half diagnostic differs: %+v", item)
		}
		if _, err := json.Marshal(item); err != nil {
			t.Fatalf("nonfinite observation is not JSON-safe: %v", err)
		}
		return stop
	})
	if snapshot != nil || !errors.Is(err, stop) || calls != 1 {
		t.Fatalf("diagnostic stop: snapshot=%v error=%v calls=%d", snapshot, err, calls)
	}
}

func TestObservationStreamsMoreThanOneChunkWithoutConvertingBytes(t *testing.T) {
	f := newFixture(t, torch.Float32)
	// The activation is just over four MiB; all weights remain the tiny fixture.
	tokens := make([]int64, decoder.ObservationChunkBytes/(4*4)+2)
	for index := range tokens {
		tokens[index] = 1
	}
	limits := decoder.Limits{MaxTokens: int64(len(tokens)), LogitRows: 2, MaxCheckpointBytes: int64(len(tokens)) * 4 * 4 * 34}
	embedding, err := f.model.Embedding.Bytes()
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.New()
	for range tokens {
		_, _ = digest.Write(embedding[16:32])
	}
	stop := errors.New("stop before decoder")
	snapshot, err := f.model.ForwardObserved(context.Background(), tokens, limits, func(_ context.Context, item decoder.ForwardObservation) error {
		if item.Stage != decoder.StageEmbedding || item.ChunkBytes != decoder.ObservationChunkBytes || item.Bytes != int64(len(tokens)*16) || item.SHA256 != hex.EncodeToString(digest.Sum(nil)) || !item.AllFinite || item.NonzeroElements != int64(len(tokens)*4) || item.Minimum == nil || math.IsNaN(*item.Minimum) {
			t.Fatalf("chunked observation differs: %+v", item)
		}
		return stop
	})
	if snapshot != nil || !errors.Is(err, stop) {
		t.Fatalf("chunked observation stop: %v", err)
	}
}
