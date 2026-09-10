package llama_test

import (
	"math"
	"os"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/tayi-ai/arandu-llama"
)

// Adapter and scoring test suite
//
// Tests the LoRA adapter handles and the teacher-forced loss, covering:
// - Score.Mean returning zero rather than NaN when nothing was scored
// - Argument validation in Context.Score and Context.ScoreText
// - Closed-handle refusals for adapters, contexts and models
// - Adapter and scale lists of differing lengths
// - Adapters refused against a model they were not loaded against
// - Which positions a skip removes from the loss
// - Applying, removing and measuring the effect of an adapter
//
// Only the Score.Mean specs run with no model at all. The rest need
// TEST_CHAT_MODEL, and the adapter specs additionally need TEST_LORA_ADAPTER
// pointing at a LoRA built for that model. None of them need a GPU: models load
// with zero offloaded layers, because what these specs prove is refusal and
// arithmetic rather than throughput, and a spec that only runs beside 42 cards
// is a spec nobody runs while writing the code it guards.

var _ = Describe("Score.Mean", func() {
	Context("with no position scored", func() {
		It("should return zero rather than NaN for the zero value", Label("unit"), func() {
			// NaN here would survive the subtraction of two perturbed forward
			// passes and land in the parameter update unnoticed
			mean := llama.Score{}.Mean()
			Expect(math.IsNaN(mean)).To(BeFalse())
			Expect(mean).To(Equal(0.0))
		})

		It("should return zero rather than NaN when a sum is present without a count", Label("unit"), func() {
			mean := llama.Score{SumNLL: 12.5, Tokens: 0}.Mean()
			Expect(math.IsNaN(mean)).To(BeFalse())
			Expect(math.IsInf(mean, 0)).To(BeFalse())
			Expect(mean).To(Equal(0.0))
		})

		It("should report an empty score and a zero loss identically, distinguished only by Tokens", Label("unit"), func() {
			empty := llama.Score{}
			perfect := llama.Score{SumNLL: 0.0, Tokens: 4}

			Expect(empty.Mean()).To(Equal(perfect.Mean()))
			Expect(empty.Tokens).NotTo(Equal(perfect.Tokens))
		})
	})

	Context("with positions scored", func() {
		It("should divide the summed loss by the number of positions", Label("unit"), func() {
			score := llama.Score{SumNLL: 6.0, Tokens: 4}
			Expect(score.Mean()).To(BeNumerically("~", 1.5, 1e-12))
		})

		It("should leave a single-position score equal to its sum", Label("unit"), func() {
			// The literal is the reference mean quoted in README.tayi.md, used here
			// only as an arbitrary float64. Nothing in this spec measures it, and
			// this spec passing is not the reference being reproduced
			score := llama.Score{SumNLL: 0.3330187499523163, Tokens: 1}
			Expect(score.Mean()).To(Equal(0.3330187499523163))
		})

		It("should weight a batch by length when sums and counts are added before dividing", Label("unit"), func() {
			// The reason the API reports a sum and a count instead of a mean:
			// averaging the two means below gives 1.5, which is not the loss a
			// backpropagating trainer reports for the same six positions
			short := llama.Score{SumNLL: 2.0, Tokens: 1}
			long := llama.Score{SumNLL: 5.0, Tokens: 5}

			batch := llama.Score{
				SumNLL: short.SumNLL + long.SumNLL,
				Tokens: short.Tokens + long.Tokens,
			}

			Expect(batch.Mean()).To(BeNumerically("~", 7.0/6.0, 1e-12))
			Expect(batch.Mean()).NotTo(BeNumerically("~", (short.Mean()+long.Mean())/2, 1e-6))
		})
	})
})

var _ = Describe("Model.LoadAdapter", func() {
	var (
		model       *llama.Model
		modelPath   string
		adapterPath string
	)

	BeforeEach(func() {
		modelPath = os.Getenv("TEST_CHAT_MODEL")
		if modelPath == "" {
			Skip("TEST_CHAT_MODEL not set - skipping integration test")
		}

		var err error
		model, err = llama.LoadModel(modelPath, llama.WithGPULayers(0))
		Expect(err).NotTo(HaveOccurred())
		Expect(model).NotTo(BeNil())

		adapterPath = os.Getenv("TEST_LORA_ADAPTER")
	})

	AfterEach(func() {
		if model != nil {
			model.Close()
		}
	})

	Context("with an unusable path", func() {
		It("should refuse a path that does not exist", Label("integration"), func() {
			adapter, err := model.LoadAdapter("/nonexistent/adapter.gguf")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("Failed to load adapter from:"))
			Expect(adapter).To(BeNil())
		})

		It("should refuse an empty path", Label("integration"), func() {
			adapter, err := model.LoadAdapter("")
			Expect(err).To(HaveOccurred())
			Expect(adapter).To(BeNil())
		})

		It("should leave the model usable after a failed load", Label("integration"), func() {
			_, err := model.LoadAdapter("/nonexistent/adapter.gguf")
			Expect(err).To(HaveOccurred())

			ctx, err := model.NewContext(llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())
			defer ctx.Close()

			tokens, err := ctx.Tokenize("The capital of France is Paris")
			Expect(err).NotTo(HaveOccurred())
			Expect(len(tokens)).To(BeNumerically(">", 0))
		})
	})

	Context("when the model is closed", func() {
		It("should refuse to load an adapter", Label("integration"), func() {
			model.Close()

			adapter, err := model.LoadAdapter("/nonexistent/adapter.gguf")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(Equal("model is closed"))
			Expect(adapter).To(BeNil())
		})
	})

	Context("with a valid adapter", func() {
		It("should return a usable adapter handle", Label("integration"), func() {
			if adapterPath == "" {
				Skip("TEST_LORA_ADAPTER not set - skipping integration test")
			}

			adapter, err := model.LoadAdapter(adapterPath)
			Expect(err).NotTo(HaveOccurred())
			Expect(adapter).NotTo(BeNil())
			defer adapter.Close()

			ctx, err := model.NewContext(llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())
			defer ctx.Close()

			err = ctx.SetAdapters([]*llama.Adapter{adapter}, []float32{1.0})
			Expect(err).NotTo(HaveOccurred())
		})

		It("should hand out independent handles for the same file", Label("integration"), func() {
			if adapterPath == "" {
				Skip("TEST_LORA_ADAPTER not set - skipping integration test")
			}

			first, err := model.LoadAdapter(adapterPath)
			Expect(err).NotTo(HaveOccurred())
			defer first.Close()

			second, err := model.LoadAdapter(adapterPath)
			Expect(err).NotTo(HaveOccurred())
			defer second.Close()

			// Closing one must not free the other: a zeroth-order step holds an
			// adapter across two forward passes, and a handle freed underneath it
			// is a use-after-free in the middle of a measurement
			err = first.Close()
			Expect(err).NotTo(HaveOccurred())

			ctx, err := model.NewContext(llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())
			defer ctx.Close()

			err = ctx.SetAdapters([]*llama.Adapter{second}, []float32{1.0})
			Expect(err).NotTo(HaveOccurred())
		})
	})
})

var _ = Describe("Adapter.Close", func() {
	var (
		model       *llama.Model
		ctx         *llama.Context
		modelPath   string
		adapterPath string
	)

	BeforeEach(func() {
		modelPath = os.Getenv("TEST_CHAT_MODEL")
		if modelPath == "" {
			Skip("TEST_CHAT_MODEL not set - skipping integration test")
		}
		adapterPath = os.Getenv("TEST_LORA_ADAPTER")
		if adapterPath == "" {
			Skip("TEST_LORA_ADAPTER not set - skipping integration test")
		}

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

	Context("when called more than once", func() {
		It("should return nil on every call", Label("integration"), func() {
			adapter, err := model.LoadAdapter(adapterPath)
			Expect(err).NotTo(HaveOccurred())

			err = adapter.Close()
			Expect(err).To(BeNil())

			err = adapter.Close()
			Expect(err).To(BeNil())
		})

		It("should not panic on a double close", Label("integration"), func() {
			adapter, err := model.LoadAdapter(adapterPath)
			Expect(err).NotTo(HaveOccurred())

			Expect(func() {
				adapter.Close()
				adapter.Close()
			}).NotTo(Panic())
		})
	})

	Context("after the adapter is closed", func() {
		It("should refuse to apply the adapter to a context", Label("integration"), func() {
			adapter, err := model.LoadAdapter(adapterPath)
			Expect(err).NotTo(HaveOccurred())

			err = adapter.Close()
			Expect(err).NotTo(HaveOccurred())

			// The exact wording belongs to adapter.go; what is contractual is
			// that a freed adapter never reaches llama_set_adapters_lora
			err = ctx.SetAdapters([]*llama.Adapter{adapter}, []float32{1.0})
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("closed"))
		})
	})
})

var _ = Describe("Context.SetAdapters", func() {
	var (
		model       *llama.Model
		ctx         *llama.Context
		modelPath   string
		adapterPath string
	)

	BeforeEach(func() {
		modelPath = os.Getenv("TEST_CHAT_MODEL")
		if modelPath == "" {
			Skip("TEST_CHAT_MODEL not set - skipping integration test")
		}

		var err error
		model, err = llama.LoadModel(modelPath, llama.WithGPULayers(0))
		Expect(err).NotTo(HaveOccurred())
		Expect(model).NotTo(BeNil())

		ctx, err = model.NewContext(llama.WithContext(2048))
		Expect(err).NotTo(HaveOccurred())
		Expect(ctx).NotTo(BeNil())

		adapterPath = os.Getenv("TEST_LORA_ADAPTER")
	})

	AfterEach(func() {
		if ctx != nil {
			ctx.Close()
		}
		if model != nil {
			model.Close()
		}
	})

	Context("with mismatched arguments", func() {
		It("should refuse more scales than adapters", Label("integration"), func() {
			// Refused in Go: the C side reads n_adapters entries from both arrays,
			// so a scale list read past its end is silent corruption of the
			// perturbation, not a crash
			err := ctx.SetAdapters(nil, []float32{1.0})
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("scale"))
		})

		It("should refuse more adapters than scales", Label("integration"), func() {
			if adapterPath == "" {
				Skip("TEST_LORA_ADAPTER not set - skipping integration test")
			}

			adapter, err := model.LoadAdapter(adapterPath)
			Expect(err).NotTo(HaveOccurred())
			defer adapter.Close()

			err = ctx.SetAdapters([]*llama.Adapter{adapter}, []float32{})
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("scale"))
		})

		It("should leave the context unperturbed after a refusal", Label("integration"), func() {
			tokens, err := ctx.Tokenize("The capital of France is Paris")
			Expect(err).NotTo(HaveOccurred())

			before, err := ctx.Score(tokens, 0)
			Expect(err).NotTo(HaveOccurred())

			err = ctx.SetAdapters(nil, []float32{1.0})
			Expect(err).To(HaveOccurred())

			after, err := ctx.Score(tokens, 0)
			Expect(err).NotTo(HaveOccurred())
			Expect(after.SumNLL).To(BeNumerically("~", before.SumNLL, 1e-3))
		})
	})

	Context("with an adapter from another model", func() {
		It("should refuse an adapter belonging to a different model handle", Label("integration"), func() {
			if adapterPath == "" {
				Skip("TEST_LORA_ADAPTER not set - skipping integration test")
			}

			// Same file, different load: an adapter is read against the tensors
			// of one model, and applying it to another model's context measures a
			// perturbation of weights it was never built for
			other, err := llama.LoadModel(modelPath, llama.WithGPULayers(0))
			Expect(err).NotTo(HaveOccurred())
			defer other.Close()

			adapter, err := other.LoadAdapter(adapterPath)
			Expect(err).NotTo(HaveOccurred())
			defer adapter.Close()

			err = ctx.SetAdapters([]*llama.Adapter{adapter}, []float32{1.0})
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("model"))
		})
	})

	Context("when the context is closed", func() {
		It("should refuse to apply adapters", Label("integration"), func() {
			ctx.Close()

			err := ctx.SetAdapters(nil, nil)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(Equal("context is closed"))
		})
	})

	Context("with no adapters", func() {
		It("should succeed on a context that has none applied", Label("integration"), func() {
			err := ctx.SetAdapters(nil, nil)
			Expect(err).NotTo(HaveOccurred())
		})

		It("should remove an adapter that was applied", Label("integration"), func() {
			if adapterPath == "" {
				Skip("TEST_LORA_ADAPTER not set - skipping integration test")
			}

			tokens, err := ctx.Tokenize("The capital of France is Paris")
			Expect(err).NotTo(HaveOccurred())

			base, err := ctx.Score(tokens, 0)
			Expect(err).NotTo(HaveOccurred())

			adapter, err := model.LoadAdapter(adapterPath)
			Expect(err).NotTo(HaveOccurred())
			defer adapter.Close()

			err = ctx.SetAdapters([]*llama.Adapter{adapter}, []float32{1.0})
			Expect(err).NotTo(HaveOccurred())

			err = ctx.SetAdapters(nil, nil)
			Expect(err).NotTo(HaveOccurred())

			// An engine left perturbed is a model the next measurement reads
			// without anybody meaning to
			restored, err := ctx.Score(tokens, 0)
			Expect(err).NotTo(HaveOccurred())
			Expect(restored.SumNLL).To(BeNumerically("~", base.SumNLL, 1e-3))
		})
	})

	Context("with an adapter applied", func() {
		It("should leave the loss unchanged at scale zero", Label("integration"), func() {
			if adapterPath == "" {
				Skip("TEST_LORA_ADAPTER not set - skipping integration test")
			}

			tokens, err := ctx.Tokenize("The capital of France is Paris")
			Expect(err).NotTo(HaveOccurred())

			base, err := ctx.Score(tokens, 0)
			Expect(err).NotTo(HaveOccurred())

			adapter, err := model.LoadAdapter(adapterPath)
			Expect(err).NotTo(HaveOccurred())
			defer adapter.Close()
			// LIFO: the adapter leaves the context before its tensors are freed.
			// A context keeps the raw handle rather than a reference this package
			// can see, so an adapter closed whilst still applied is exactly the
			// use-after-free adapter.go documents
			defer func() { Expect(ctx.ClearAdapters()).To(Succeed()) }()

			// Scale zero is the identity, and a zeroth-order step reads the
			// difference between two scales; a scale that shifted the loss on its
			// own would put that shift in the gradient estimate
			err = ctx.SetAdapters([]*llama.Adapter{adapter}, []float32{0.0})
			Expect(err).NotTo(HaveOccurred())

			neutral, err := ctx.Score(tokens, 0)
			Expect(err).NotTo(HaveOccurred())
			Expect(neutral.SumNLL).To(BeNumerically("~", base.SumNLL, 1e-3))
		})

		It("should change the loss at a non-zero scale", Label("integration"), func() {
			if adapterPath == "" {
				Skip("TEST_LORA_ADAPTER not set - skipping integration test")
			}

			tokens, err := ctx.Tokenize("The capital of France is Paris")
			Expect(err).NotTo(HaveOccurred())

			base, err := ctx.Score(tokens, 0)
			Expect(err).NotTo(HaveOccurred())

			adapter, err := model.LoadAdapter(adapterPath)
			Expect(err).NotTo(HaveOccurred())
			defer adapter.Close()
			// Removed before the deferred Close frees it; see the scale-zero spec
			defer func() { Expect(ctx.ClearAdapters()).To(Succeed()) }()

			err = ctx.SetAdapters([]*llama.Adapter{adapter}, []float32{1.0})
			Expect(err).NotTo(HaveOccurred())

			// If this passes only because both numbers are identical, the adapter
			// never reached the forward pass and every step of training would
			// measure the base model twice
			perturbed, err := ctx.Score(tokens, 0)
			Expect(err).NotTo(HaveOccurred())
			Expect(perturbed.SumNLL).NotTo(BeNumerically("~", base.SumNLL, 1e-6))
		})

		It("should keep the same number of scored positions", Label("integration"), func() {
			if adapterPath == "" {
				Skip("TEST_LORA_ADAPTER not set - skipping integration test")
			}

			tokens, err := ctx.Tokenize("The capital of France is Paris")
			Expect(err).NotTo(HaveOccurred())

			base, err := ctx.Score(tokens, 0)
			Expect(err).NotTo(HaveOccurred())

			adapter, err := model.LoadAdapter(adapterPath)
			Expect(err).NotTo(HaveOccurred())
			defer adapter.Close()
			// Removed before the deferred Close frees it; see the scale-zero spec
			defer func() { Expect(ctx.ClearAdapters()).To(Succeed()) }()

			err = ctx.SetAdapters([]*llama.Adapter{adapter}, []float32{1.0})
			Expect(err).NotTo(HaveOccurred())

			perturbed, err := ctx.Score(tokens, 0)
			Expect(err).NotTo(HaveOccurred())
			Expect(perturbed.Tokens).To(Equal(base.Tokens))
		})
	})
})

var _ = Describe("Context.ClearAdapters", func() {
	var (
		model       *llama.Model
		ctx         *llama.Context
		modelPath   string
		adapterPath string
	)

	BeforeEach(func() {
		modelPath = os.Getenv("TEST_CHAT_MODEL")
		if modelPath == "" {
			Skip("TEST_CHAT_MODEL not set - skipping integration test")
		}

		var err error
		model, err = llama.LoadModel(modelPath, llama.WithGPULayers(0))
		Expect(err).NotTo(HaveOccurred())
		Expect(model).NotTo(BeNil())

		ctx, err = model.NewContext(llama.WithContext(2048))
		Expect(err).NotTo(HaveOccurred())
		Expect(ctx).NotTo(BeNil())

		adapterPath = os.Getenv("TEST_LORA_ADAPTER")
	})

	AfterEach(func() {
		if ctx != nil {
			ctx.Close()
		}
		if model != nil {
			model.Close()
		}
	})

	Context("on a context with no adapter applied", func() {
		It("should succeed", Label("integration"), func() {
			err := ctx.ClearAdapters()
			Expect(err).NotTo(HaveOccurred())
		})

		It("should be safe to call repeatedly", Label("integration"), func() {
			Expect(ctx.ClearAdapters()).To(Succeed())
			Expect(ctx.ClearAdapters()).To(Succeed())
		})
	})

	Context("on a context with an adapter applied", func() {
		It("should restore the loss measured before the adapter", Label("integration"), func() {
			if adapterPath == "" {
				Skip("TEST_LORA_ADAPTER not set - skipping integration test")
			}

			tokens, err := ctx.Tokenize("The capital of France is Paris")
			Expect(err).NotTo(HaveOccurred())

			base, err := ctx.Score(tokens, 0)
			Expect(err).NotTo(HaveOccurred())

			adapter, err := model.LoadAdapter(adapterPath)
			Expect(err).NotTo(HaveOccurred())
			defer adapter.Close()

			err = ctx.SetAdapters([]*llama.Adapter{adapter}, []float32{1.0})
			Expect(err).NotTo(HaveOccurred())

			err = ctx.ClearAdapters()
			Expect(err).NotTo(HaveOccurred())

			restored, err := ctx.Score(tokens, 0)
			Expect(err).NotTo(HaveOccurred())
			Expect(restored.SumNLL).To(BeNumerically("~", base.SumNLL, 1e-3))
		})
	})

	Context("when the context is closed", func() {
		It("should refuse to clear adapters", Label("integration"), func() {
			ctx.Close()

			err := ctx.ClearAdapters()
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(Equal("context is closed"))
		})
	})
})

var _ = Describe("Context.Score", func() {
	var (
		model     *llama.Model
		ctx       *llama.Context
		modelPath string
	)

	BeforeEach(func() {
		modelPath = os.Getenv("TEST_CHAT_MODEL")
		if modelPath == "" {
			Skip("TEST_CHAT_MODEL not set - skipping integration test")
		}

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

	Context("with a sequence too short to score", func() {
		It("should refuse a single token", Label("integration"), func() {
			// One token has nothing before it to condition on and nothing after
			// it to predict
			score, err := ctx.Score([]int32{1}, 0)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(Equal("at least two tokens required to score, got 1"))
			Expect(score.Tokens).To(Equal(0))
		})

		It("should refuse an empty sequence", Label("integration"), func() {
			score, err := ctx.Score(nil, 0)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(Equal("at least two tokens required to score, got 0"))
			Expect(score.SumNLL).To(Equal(0.0))
		})
	})

	Context("with an invalid skip", func() {
		It("should refuse a negative skip", Label("integration"), func() {
			_, err := ctx.Score([]int32{1, 2, 3}, -1)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(Equal("invalid skip: -1 for 3 tokens"))
		})

		It("should refuse a skip that leaves nothing to score", Label("integration"), func() {
			_, err := ctx.Score([]int32{1, 2, 3}, 3)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(Equal("invalid skip: 3 for 3 tokens"))
		})

		It("should refuse a skip beyond the end of the sequence", Label("integration"), func() {
			_, err := ctx.Score([]int32{1, 2, 3}, 9)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(Equal("invalid skip: 9 for 3 tokens"))
		})
	})

	Context("when the context is closed", func() {
		It("should refuse to score", Label("integration"), func() {
			ctx.Close()

			_, err := ctx.Score([]int32{1, 2, 3}, 0)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(Equal("context is closed"))
		})

		It("should refuse before reaching argument validation", Label("integration"), func() {
			ctx.Close()

			// A closed handle is reported as such even when the arguments are
			// also wrong: the caller's first problem is the freed context
			_, err := ctx.Score([]int32{1}, 0)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(Equal("context is closed"))
		})
	})

	Context("when the model is closed", func() {
		It("should refuse to score", Label("integration"), func() {
			model.Close()

			// The weights are gone while the context still points at them, so the
			// refusal has to happen in Go, before the tokens cross into C
			_, err := ctx.Score([]int32{1, 2, 3}, 0)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(Equal("model is closed"))
		})
	})

	Context("with a valid sequence", func() {
		It("should score every position but the first when nothing is skipped", Label("integration"), func() {
			tokens, err := ctx.Tokenize("The capital of France is Paris")
			Expect(err).NotTo(HaveOccurred())
			Expect(len(tokens)).To(BeNumerically(">=", 4))

			score, err := ctx.Score(tokens, 0)
			Expect(err).NotTo(HaveOccurred())
			Expect(score.Tokens).To(Equal(len(tokens) - 1))
		})

		It("should score the same positions for a skip of one as for none", Label("integration"), func() {
			tokens, err := ctx.Tokenize("The capital of France is Paris")
			Expect(err).NotTo(HaveOccurred())

			none, err := ctx.Score(tokens, 0)
			Expect(err).NotTo(HaveOccurred())

			one, err := ctx.Score(tokens, 1)
			Expect(err).NotTo(HaveOccurred())

			// Position zero is unscorable either way, so skipping it changes
			// nothing
			Expect(one.Tokens).To(Equal(none.Tokens))
			Expect(one.SumNLL).To(BeNumerically("~", none.SumNLL, 1e-6))
		})

		It("should drop one position per skipped token beyond the first", Label("integration"), func() {
			tokens, err := ctx.Tokenize("The capital of France is Paris")
			Expect(err).NotTo(HaveOccurred())
			Expect(len(tokens)).To(BeNumerically(">=", 4))

			score, err := ctx.Score(tokens, 3)
			Expect(err).NotTo(HaveOccurred())
			Expect(score.Tokens).To(Equal(len(tokens) - 3))
		})

		It("should report a positive summed negative log likelihood", Label("integration"), func() {
			tokens, err := ctx.Tokenize("The capital of France is Paris")
			Expect(err).NotTo(HaveOccurred())

			score, err := ctx.Score(tokens, 0)
			Expect(err).NotTo(HaveOccurred())
			Expect(score.SumNLL).To(BeNumerically(">", 0))
			Expect(math.IsNaN(score.SumNLL)).To(BeFalse())
			Expect(math.IsInf(score.SumNLL, 0)).To(BeFalse())
		})

		It("should report a mean equal to the sum over the count", Label("integration"), func() {
			tokens, err := ctx.Tokenize("The capital of France is Paris")
			Expect(err).NotTo(HaveOccurred())

			score, err := ctx.Score(tokens, 0)
			Expect(err).NotTo(HaveOccurred())
			Expect(score.Tokens).To(BeNumerically(">", 0))
			Expect(score.Mean()).To(BeNumerically("~", score.SumNLL/float64(score.Tokens), 1e-12))
		})

		It("should report a lower loss for text the model finds likely", Label("integration"), func() {
			likely, err := ctx.Tokenize("The capital of France is Paris")
			Expect(err).NotTo(HaveOccurred())

			unlikely, err := ctx.Tokenize("The capital of France is zzzz qqqq")
			Expect(err).NotTo(HaveOccurred())

			// Without this the loss could be any well-formed number; with it the
			// number is known to track the model's own predictions
			likelyScore, err := ctx.Score(likely, 0)
			Expect(err).NotTo(HaveOccurred())

			unlikelyScore, err := ctx.Score(unlikely, 0)
			Expect(err).NotTo(HaveOccurred())

			Expect(likelyScore.Mean()).To(BeNumerically("<", unlikelyScore.Mean()))
		})

		It("should return the same loss on repeated calls", Label("integration"), func() {
			tokens, err := ctx.Tokenize("The capital of France is Paris")
			Expect(err).NotTo(HaveOccurred())

			first, err := ctx.Score(tokens, 0)
			Expect(err).NotTo(HaveOccurred())

			second, err := ctx.Score(tokens, 0)
			Expect(err).NotTo(HaveOccurred())

			Expect(second.SumNLL).To(BeNumerically("~", first.SumNLL, 1e-6))
			Expect(second.Tokens).To(Equal(first.Tokens))
		})

		It("should return the same loss after generation has filled the KV cache", Label("integration"), func() {
			tokens, err := ctx.Tokenize("The capital of France is Paris")
			Expect(err).NotTo(HaveOccurred())

			before, err := ctx.Score(tokens, 0)
			Expect(err).NotTo(HaveOccurred())

			_, err = ctx.Generate("Something else entirely", llama.WithMaxTokens(8))
			Expect(err).NotTo(HaveOccurred())

			// A cache written under one set of weights and read under another
			// scores a mixture of two models, which is why scoring clears it
			after, err := ctx.Score(tokens, 0)
			Expect(err).NotTo(HaveOccurred())
			Expect(after.SumNLL).To(BeNumerically("~", before.SumNLL, 1e-3))
		})
	})
})

var _ = Describe("Context.ScoreText", func() {
	var (
		model     *llama.Model
		ctx       *llama.Context
		modelPath string
	)

	BeforeEach(func() {
		modelPath = os.Getenv("TEST_CHAT_MODEL")
		if modelPath == "" {
			Skip("TEST_CHAT_MODEL not set - skipping integration test")
		}

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

	Context("with a prompt and a completion", func() {
		It("should score only the tokens after the prompt", Label("integration"), func() {
			promptTokens, err := ctx.Tokenize("The capital of France is")
			Expect(err).NotTo(HaveOccurred())

			jointTokens, err := ctx.Tokenize("The capital of France is Paris")
			Expect(err).NotTo(HaveOccurred())
			Expect(len(jointTokens)).To(BeNumerically(">", len(promptTokens)))

			score, err := ctx.ScoreText("The capital of France is", " Paris")
			Expect(err).NotTo(HaveOccurred())
			Expect(score.Tokens).To(Equal(len(jointTokens) - len(promptTokens)))
		})

		It("should report the loss Score reports for the jointly tokenised sequence", Label("integration"), func() {
			promptTokens, err := ctx.Tokenize("The capital of France is")
			Expect(err).NotTo(HaveOccurred())

			jointTokens, err := ctx.Tokenize("The capital of France is Paris")
			Expect(err).NotTo(HaveOccurred())

			// Tokenising the two halves separately and concatenating would produce
			// a sequence the model never sees for that text, because the tokeniser
			// merges across the join
			direct, err := ctx.Score(jointTokens, len(promptTokens))
			Expect(err).NotTo(HaveOccurred())

			viaText, err := ctx.ScoreText("The capital of France is", " Paris")
			Expect(err).NotTo(HaveOccurred())

			Expect(viaText.Tokens).To(Equal(direct.Tokens))
			Expect(viaText.SumNLL).To(BeNumerically("~", direct.SumNLL, 1e-6))
		})

		It("should score fewer positions than the whole sequence carries", Label("integration"), func() {
			jointTokens, err := ctx.Tokenize("The capital of France is Paris")
			Expect(err).NotTo(HaveOccurred())

			score, err := ctx.ScoreText("The capital of France is", " Paris")
			Expect(err).NotTo(HaveOccurred())

			// A loss that included the prompt would reward a policy for predicting
			// text it was handed
			Expect(score.Tokens).To(BeNumerically("<", len(jointTokens)-1))
			Expect(score.Tokens).To(BeNumerically(">", 0))
		})

		It("should report a lower loss for a completion the model finds likely", Label("integration"), func() {
			likely, err := ctx.ScoreText("The capital of France is", " Paris")
			Expect(err).NotTo(HaveOccurred())

			unlikely, err := ctx.ScoreText("The capital of France is", " zzzz")
			Expect(err).NotTo(HaveOccurred())

			Expect(likely.Mean()).To(BeNumerically("<", unlikely.Mean()))
		})
	})

	Context("with an empty completion", func() {
		It("should refuse to score nothing", Label("integration"), func() {
			// The joint tokenisation equals the prompt, so the skip consumes the
			// whole sequence and the refusal arrives from Score
			_, err := ctx.ScoreText("The capital of France is", "")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("invalid skip"))
		})
	})

	Context("when the context is closed", func() {
		It("should refuse at tokenisation", Label("integration"), func() {
			ctx.Close()

			_, err := ctx.ScoreText("The capital of France is", " Paris")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(Equal("failed to tokenise prompt: context is closed"))
		})
	})
})
