package llama_test

import (
	"bufio"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/tayi-ai/arandu-llama"
)

// Capture test suite
//
// Tests the per-token final representation, covering:
// - The two digests being deterministic and never colliding across tokens,
//   positions, policy, scale sign, quantisation or version
// - PositionsOf and Capture.Row/Shape addressing rows by position index
// - Argument validation in Context.CaptureFinal before crossing into C
// - Shape, finiteness and identity of a capture against a real model
// - Two identical captures being bitwise identical
// - The capture being per token and causal: a changed token leaves the rows
//   before it untouched and moves every row from it on
// - Rows not depending on which positions were requested
// - Windows carrying the KV cache like one pass
// - A capture leaving the context able to score, on a plain context and on one
//   created with embeddings
// - The capture moving with an adapter and its digest, and back when cleared
// - The 2048-position cost figure the doc comment quotes
//
// Only the digest, PositionsOf and Row/Shape specs run with no model at all.
// The rest need TEST_CHAT_MODEL, and the adapter spec additionally needs
// TEST_LORA_ADAPTER pointing at a LoRA built for that model. When a
// declared.txt sits beside the model naming its size, a file of another size
// is still downloading and the specs skip rather than read a truncated GGUF.
// None of them need a GPU: models load with zero offloaded layers.
//
// The bitwise specs are evidence only when run with --flake-attempts 1; a
// retry would hide exactly the nondeterminism they exist to detect.

// captureModelPath returns TEST_CHAT_MODEL, or skips with the reason it cannot
// be used: unset, absent, or not yet the size declared.txt beside it declares.
func captureModelPath() string {
	modelPath := os.Getenv("TEST_CHAT_MODEL")
	if modelPath == "" {
		Skip("TEST_CHAT_MODEL not set - skipping integration test")
	}
	info, err := os.Stat(modelPath)
	if err != nil {
		Skip("TEST_CHAT_MODEL is not readable (" + err.Error() + ") - skipping integration test")
	}
	declared, err := os.Open(filepath.Join(filepath.Dir(modelPath), "declared.txt"))
	if err != nil {
		return modelPath
	}
	defer declared.Close()
	scanner := bufio.NewScanner(declared)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) < 3 || fields[0] != filepath.Base(modelPath) {
			continue
		}
		size, err := strconv.ParseInt(fields[2], 10, 64)
		if err != nil {
			continue
		}
		if info.Size() != size {
			Skip("TEST_CHAT_MODEL is " + strconv.FormatInt(info.Size(), 10) + " bytes and declared.txt says " +
				fields[2] + " - still downloading, skipping integration test")
		}
	}
	return modelPath
}

func bitwiseEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return false
		}
	}
	return true
}

func allFinite(values []float32) bool {
	for _, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return false
		}
	}
	return true
}

func allPositions(n int) []int32 {
	positions := make([]int32, n)
	for i := range positions {
		positions[i] = int32(i)
	}
	return positions
}

var _ = Describe("TokenDigest", func() {
	It("should be deterministic and injective over tokens, positions and lengths", Label("unit"), func() {
		tokens := []int32{1, 2, 3, 4}
		positions := []int32{1, 2}
		digest := llama.TokenDigest(tokens, positions)
		Expect(digest).To(HaveLen(64))
		Expect(llama.TokenDigest(tokens, positions)).To(Equal(digest))

		Expect(llama.TokenDigest([]int32{1, 2, 3, 5}, positions)).NotTo(Equal(digest))
		Expect(llama.TokenDigest(tokens, []int32{1, 3})).NotTo(Equal(digest))
		Expect(llama.TokenDigest(tokens, []int32{1, 2, 3})).NotTo(Equal(digest))
		// Length prefixes: the same integers split differently are different
		// requests
		Expect(llama.TokenDigest([]int32{1, 2}, []int32{0})).NotTo(Equal(llama.TokenDigest([]int32{1}, []int32{0, 1})))
		Expect(llama.TokenDigest(nil, nil)).To(HaveLen(64))
	})
})

var _ = Describe("SnapshotDigest", func() {
	It("should distinguish scale sign, policy, quantisation and version", Label("unit"), func() {
		q := "smollm3 3B Q4_K - Medium"
		d := "0123abcd"
		plus := llama.SnapshotDigest(q, d, []float32{1})
		Expect(plus).To(HaveLen(64))
		Expect(llama.SnapshotDigest(q, d, []float32{1})).To(Equal(plus))
		// A +eps and a -eps probe on one adapter are two policies
		Expect(llama.SnapshotDigest(q, d, []float32{-1})).NotTo(Equal(plus))
		Expect(llama.SnapshotDigest(q, "base", nil)).NotTo(Equal(plus))
		Expect(llama.SnapshotDigest("smollm3 3B Q2_K - Medium", d, []float32{1})).NotTo(Equal(plus))
		// Length-prefixed strings: moving a character across the join is a
		// different snapshot
		Expect(llama.SnapshotDigest("ab", "c", nil)).NotTo(Equal(llama.SnapshotDigest("a", "bc", nil)))
		// A bump of the pin is a deliberate edit of this line
		Expect(llama.CaptureVersion).To(HaveSuffix("llama.cpp.90c26fcd"))
	})
})

var _ = Describe("PositionsOf and Capture", func() {
	It("should address rows by position index", Label("unit"), func() {
		Expect(llama.PositionsOf([]bool{false, true, true, false, true})).To(Equal([]int32{1, 2, 4}))
		Expect(llama.PositionsOf([]bool{false, false})).To(BeEmpty())

		capture := &llama.Capture{
			Data:      []float32{0, 1, 2, 3, 4, 5},
			Positions: []int32{3, 7},
			Width:     3,
		}
		Expect(capture.Row(1)).To(Equal([]float32{3, 4, 5}))
		nEmbd, nPositions := capture.Shape()
		Expect(nEmbd).To(Equal(3))
		Expect(nPositions).To(Equal(2))
		Expect(func() { capture.Row(2) }).To(Panic())
	})
})

var _ = Describe("Context.CaptureFinal", func() {
	var (
		model *llama.Model
		ctx   *llama.Context
	)

	BeforeEach(func() {
		modelPath := captureModelPath()

		var err error
		model, err = llama.LoadModel(modelPath, llama.WithGPULayers(0))
		Expect(err).NotTo(HaveOccurred())
		Expect(model).NotTo(BeNil())

		ctx, err = model.NewContext(llama.WithContext(2048))
		Expect(err).NotTo(HaveOccurred())
		Expect(ctx).NotTo(BeNil())
	})

	AfterEach(func() {
		if ctx != nil {
			ctx.Close()
		}
		if model != nil {
			model.Close()
		}
	})

	Context("with arguments it cannot capture", func() {
		It("should refuse before crossing into C", Label("integration"), func() {
			tokens, err := ctx.Tokenize("The capital of France is Paris")
			Expect(err).NotTo(HaveOccurred())
			n := len(tokens)

			_, err = ctx.CaptureFinal(nil, []int32{0})
			Expect(err).To(MatchError(ContainSubstring("at least one token")))
			_, err = ctx.CaptureFinal(tokens, nil)
			Expect(err).To(MatchError(ContainSubstring("at least one position")))
			_, err = ctx.CaptureFinal([]int32{1, -1}, []int32{0})
			Expect(err).To(MatchError(ContainSubstring("invalid token -1")))
			_, err = ctx.CaptureFinal(tokens, []int32{int32(n)})
			Expect(err).To(MatchError(ContainSubstring("invalid position")))
			_, err = ctx.CaptureFinal(tokens, []int32{-1})
			Expect(err).To(MatchError(ContainSubstring("invalid position")))
			_, err = ctx.CaptureFinal(tokens, []int32{3, 3})
			Expect(err).To(MatchError(ContainSubstring("strictly increasing")))
			_, err = ctx.CaptureFinal(tokens, []int32{7, 3})
			Expect(err).To(MatchError(ContainSubstring("strictly increasing")))

			// A token outside the vocabulary is only known to C
			_, err = ctx.CaptureFinal([]int32{math.MaxInt32}, []int32{0})
			Expect(err).To(MatchError(ContainSubstring("outside the vocabulary")))
		})

		It("should refuse a closed context", Label("integration"), func() {
			ctx.Close()
			_, err := ctx.CaptureFinal([]int32{1, 2}, []int32{0})
			Expect(err).To(MatchError("context is closed"))
		})
	})

	Context("with a sentence", func() {
		It("should return n_positions rows of Width with the declared identity", Label("integration"), func() {
			tokens, err := ctx.Tokenize("The capital of France is Paris")
			Expect(err).NotTo(HaveOccurred())
			positions := allPositions(len(tokens))

			capture, err := ctx.CaptureFinal(tokens, positions)
			Expect(err).NotTo(HaveOccurred())

			Expect(capture.Width).To(BeNumerically(">", 0))
			Expect(capture.Data).To(HaveLen(len(positions) * capture.Width))
			nEmbd, nPositions := capture.Shape()
			Expect(nEmbd).To(Equal(capture.Width))
			Expect(nPositions).To(Equal(len(positions)))
			Expect(allFinite(capture.Data)).To(BeTrue())

			quantisation, err := model.Describe()
			Expect(err).NotTo(HaveOccurred())
			if strings.HasPrefix(quantisation, "smollm3") {
				Expect(capture.Width).To(Equal(2048))
			}
			Expect(capture.Tensor).To(Equal(llama.CaptureTensor))
			Expect(capture.Tensor).To(Equal("result_norm"))
			Expect(capture.DType).To(Equal("f32"))
			Expect(capture.Version).To(Equal(llama.CaptureVersion))
			Expect(capture.Policy).To(Equal("base"))
			Expect(capture.Scales).To(BeEmpty())
			Expect(capture.Quantisation).To(Equal(quantisation))
			Expect(capture.Positions).To(Equal(positions))
			Expect(capture.TokenDigest).To(Equal(llama.TokenDigest(tokens, positions)))
			Expect(capture.SnapshotDigest).To(Equal(llama.SnapshotDigest(quantisation, "base", nil)))
		})

		It("should capture two positions with shape n_embd x 2", Label("integration"), func() {
			tokens, err := ctx.Tokenize("The capital of France is Paris")
			Expect(err).NotTo(HaveOccurred())
			Expect(len(tokens)).To(BeNumerically(">=", 4))

			capture, err := ctx.CaptureFinal(tokens, []int32{1, int32(len(tokens) - 1)})
			Expect(err).NotTo(HaveOccurred())
			nEmbd, nPositions := capture.Shape()
			Expect(nEmbd).To(Equal(capture.Width))
			Expect(nPositions).To(Equal(2))
			Expect(capture.Data).To(HaveLen(2 * capture.Width))
			Expect(allFinite(capture.Data)).To(BeTrue())
			Expect(capture.Row(0)).To(HaveLen(capture.Width))
			Expect(capture.Row(1)).To(HaveLen(capture.Width))
		})

		It("should be bitwise identical across two captures and a fresh context", Label("integration"), func() {
			tokens, err := ctx.Tokenize("The capital of France is Paris")
			Expect(err).NotTo(HaveOccurred())
			positions := allPositions(len(tokens))

			first, err := ctx.CaptureFinal(tokens, positions)
			Expect(err).NotTo(HaveOccurred())
			second, err := ctx.CaptureFinal(tokens, positions)
			Expect(err).NotTo(HaveOccurred())
			// Identical to the last digit, or the two cannot be compared
			Expect(bitwiseEqual(first.Data, second.Data)).To(BeTrue())
			Expect(second.TokenDigest).To(Equal(first.TokenDigest))
			Expect(second.SnapshotDigest).To(Equal(first.SnapshotDigest))

			fresh, err := model.NewContext(llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())
			defer fresh.Close()
			third, err := fresh.CaptureFinal(tokens, positions)
			Expect(err).NotTo(HaveOccurred())
			Expect(bitwiseEqual(first.Data, third.Data)).To(BeTrue())
		})

		It("should be per token and causal", Label("integration"), func() {
			tokens, err := ctx.Tokenize("The capital of France is Paris and the capital of Italy is Rome")
			Expect(err).NotTo(HaveOccurred())
			Expect(len(tokens)).To(BeNumerically(">", 8))
			positions := allPositions(len(tokens))

			original, err := ctx.CaptureFinal(tokens, positions)
			Expect(err).NotTo(HaveOccurred())

			changed := append([]int32(nil), tokens...)
			for _, candidate := range tokens {
				if candidate != tokens[5] {
					changed[5] = candidate
					break
				}
			}
			Expect(changed[5]).NotTo(Equal(tokens[5]))
			moved, err := ctx.CaptureFinal(changed, positions)
			Expect(err).NotTo(HaveOccurred())
			Expect(moved.TokenDigest).NotTo(Equal(original.TokenDigest))

			// A pooled sequence embedding fails this: it has no rows before the
			// change to leave untouched
			for k := 0; k < 5; k++ {
				Expect(bitwiseEqual(original.Row(k), moved.Row(k))).To(BeTrue(), "row %d before the change moved", k)
			}
			for k := 5; k < len(tokens); k++ {
				Expect(bitwiseEqual(original.Row(k), moved.Row(k))).To(BeFalse(), "row %d after the change did not move", k)
			}
		})

		It("should not depend on which positions were requested", Label("integration"), func() {
			tokens, err := ctx.Tokenize("The capital of France is Paris")
			Expect(err).NotTo(HaveOccurred())
			full, err := ctx.CaptureFinal(tokens, allPositions(len(tokens)))
			Expect(err).NotTo(HaveOccurred())

			// Every window row is an output row either way; this proves the
			// row-to-position mapping
			for p := 0; p < len(tokens); p++ {
				single, err := ctx.CaptureFinal(tokens, []int32{int32(p)})
				Expect(err).NotTo(HaveOccurred())
				Expect(bitwiseEqual(full.Row(p), single.Row(0))).To(BeTrue(), "position %d", p)
			}
		})

		It("should carry the KV cache across windows like one pass", Label("integration"), func() {
			text := strings.Repeat("The capital of France is Paris and the capital of Italy is Rome. ", 4)
			tokens, err := ctx.Tokenize(text)
			Expect(err).NotTo(HaveOccurred())
			Expect(len(tokens)).To(BeNumerically(">=", 40))
			// Several windows of 8, including one with no requested position
			positions := []int32{0, 3, 7, 8, 9, 20, 30, 31, int32(len(tokens) - 1)}

			reference, err := ctx.CaptureFinal(tokens, positions)
			Expect(err).NotTo(HaveOccurred())

			windowed, err := model.NewContext(llama.WithContext(2048), llama.WithBatch(8))
			Expect(err).NotTo(HaveOccurred())
			defer windowed.Close()
			a, err := windowed.CaptureFinal(tokens, positions)
			Expect(err).NotTo(HaveOccurred())

			split, err := model.NewContext(llama.WithContext(2048), llama.WithBatch(8), llama.WithUBatch(4))
			Expect(err).NotTo(HaveOccurred())
			defer split.Close()
			b, err := split.CaptureFinal(tokens, positions)
			Expect(err).NotTo(HaveOccurred())

			// Measured bitwise on CPU. A GPU run that differs records the maximum
			// and decides; it does not loosen this silently
			Expect(bitwiseEqual(reference.Data, a.Data)).To(BeTrue())
			Expect(bitwiseEqual(reference.Data, b.Data)).To(BeTrue())
		})

		It("should leave the context able to score and restore the embeddings flag", Label("integration"), func() {
			tokens, err := ctx.Tokenize("The capital of France is Paris")
			Expect(err).NotTo(HaveOccurred())

			before, err := ctx.Score(tokens, 0)
			Expect(err).NotTo(HaveOccurred())
			_, err = ctx.CaptureFinal(tokens, allPositions(len(tokens)))
			Expect(err).NotTo(HaveOccurred())
			after, err := ctx.Score(tokens, 0)
			Expect(err).NotTo(HaveOccurred())
			Expect(after.Tokens).To(Equal(before.Tokens))
			Expect(after.SumNLL).To(Equal(before.SumNLL))

			embedding, err := model.NewContext(llama.WithContext(2048), llama.WithEmbeddings())
			Expect(err).NotTo(HaveOccurred())
			defer embedding.Close()
			capture, err := embedding.CaptureFinal(tokens, allPositions(len(tokens)))
			Expect(err).NotTo(HaveOccurred())
			Expect(allFinite(capture.Data)).To(BeTrue())
			vector, err := embedding.GetEmbeddings("The capital of France is Paris")
			Expect(err).NotTo(HaveOccurred())
			Expect(vector).NotTo(BeEmpty())
		})

		It("should capture 2048 positions of a 2048-token sequence", Label("integration"), func() {
			text := strings.Repeat("The capital of France is Paris and the capital of Italy is Rome. ", 200)
			tokens, err := ctx.Tokenize(text)
			Expect(err).NotTo(HaveOccurred())
			Expect(len(tokens)).To(BeNumerically(">=", 2048))
			tokens = tokens[:2048]

			started := time.Now()
			capture, err := ctx.CaptureFinal(tokens, allPositions(2048))
			Expect(err).NotTo(HaveOccurred())
			elapsed := time.Since(started)
			Expect(capture.Data).To(HaveLen(2048 * capture.Width))
			Expect(allFinite(capture.Data)).To(BeTrue())
			GinkgoWriter.Printf("captured 2048 x %d floats (%d bytes) in %s\n",
				capture.Width, 4*len(capture.Data), elapsed)
		})
	})

	Context("with an adapter", func() {
		It("should move with the adapter and its digest, and back when cleared", Label("integration"), func() {
			adapterPath := os.Getenv("TEST_LORA_ADAPTER")
			if adapterPath == "" {
				Skip("TEST_LORA_ADAPTER not set - skipping integration test")
			}

			tokens, err := ctx.Tokenize("The capital of France is Paris")
			Expect(err).NotTo(HaveOccurred())
			positions := allPositions(len(tokens))

			base, err := ctx.CaptureFinal(tokens, positions)
			Expect(err).NotTo(HaveOccurred())

			adapter, err := model.LoadAdapter(adapterPath)
			Expect(err).NotTo(HaveOccurred())
			defer adapter.Close()
			// LIFO: the adapter leaves the context before its tensors are freed.
			// A context keeps the raw handle rather than a reference this package
			// can see, so an adapter closed whilst still applied is exactly the
			// use-after-free adapter.go documents
			defer func() { Expect(ctx.ClearAdapters()).To(Succeed()) }()

			Expect(ctx.SetAdapters([]*llama.Adapter{adapter}, []float32{1})).To(Succeed())
			plus, err := ctx.CaptureFinal(tokens, positions)
			Expect(err).NotTo(HaveOccurred())
			Expect(bitwiseEqual(base.Data, plus.Data)).To(BeFalse())
			Expect(plus.Policy).To(Equal(adapter.Digest()))
			Expect(plus.Scales).To(Equal([]float32{1}))
			Expect(plus.SnapshotDigest).NotTo(Equal(base.SnapshotDigest))

			Expect(ctx.SetAdapters([]*llama.Adapter{adapter}, []float32{-1})).To(Succeed())
			minus, err := ctx.CaptureFinal(tokens, positions)
			Expect(err).NotTo(HaveOccurred())
			Expect(bitwiseEqual(plus.Data, minus.Data)).To(BeFalse())
			Expect(minus.Scales).To(Equal([]float32{-1}))
			Expect(minus.SnapshotDigest).NotTo(Equal(plus.SnapshotDigest))

			Expect(ctx.ClearAdapters()).To(Succeed())
			cleared, err := ctx.CaptureFinal(tokens, positions)
			Expect(err).NotTo(HaveOccurred())
			Expect(bitwiseEqual(base.Data, cleared.Data)).To(BeTrue())
			Expect(cleared.SnapshotDigest).To(Equal(base.SnapshotDigest))
		})
	})
})
