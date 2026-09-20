//go:build libtorch && cgo

package initialadapter_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/ornith"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

const aggregate = "75185d68fcfdd09bb2dcb6dffe0902a35300f52d9fda716b408189991773aa7c"

// Preserved loader-reference/output/materialized-wrapped.json trainable tensor
// hashes, copied verbatim as test data. Tests never read a runtime artifact.
// Columns are qA,qB,vA,vB; rows are layers3,7,...31.
var reference = [8][4]string{
	{"c3aab779ad5e7a7051489c64e15b0da562e5588964d2b06b4058c277955babce", "fa43239bcee7b97ca62f007cc68487560a39e19f74f3dde7486db3f98df8e471", "dedae982be5a054f7101cd5cdddea29e9fc655c40b1662174b91893a27bf7c49", "4fe7b59af6de3b665b67788cc2f99892ab827efae3a467342b3bb4e3bc8e5bfe"},
	{"77f75126ce6f786d9612ac519f82ef2d8d5a37d89bc57bb83738a3e3589049ac", "fa43239bcee7b97ca62f007cc68487560a39e19f74f3dde7486db3f98df8e471", "b969c66deefa00617dc99ff54617674a941ab3f12d2bdcefec0a74cc25c1ff42", "4fe7b59af6de3b665b67788cc2f99892ab827efae3a467342b3bb4e3bc8e5bfe"},
	{"561da275d79c7a7ce455cf3b52a10c484cc2cf3dcf1e03ff16dd01d4be2b04d5", "fa43239bcee7b97ca62f007cc68487560a39e19f74f3dde7486db3f98df8e471", "91a09616e00269beea8441b8de8d69556dd8adc7893904ec64c22351be0aaab1", "4fe7b59af6de3b665b67788cc2f99892ab827efae3a467342b3bb4e3bc8e5bfe"},
	{"b1ebbdbdaca01d2cb278698526260460fb483965f8b22204d7400fa3e5d8e580", "fa43239bcee7b97ca62f007cc68487560a39e19f74f3dde7486db3f98df8e471", "9264197f3ca2579dc7370a4ab13690e01a7217dc600202223514de9363a325b8", "4fe7b59af6de3b665b67788cc2f99892ab827efae3a467342b3bb4e3bc8e5bfe"},
	{"7bf11888a574f018ee66b813ae6366272773a28e5e44774c072052915850308d", "fa43239bcee7b97ca62f007cc68487560a39e19f74f3dde7486db3f98df8e471", "aacd4a623aadc324de36f599e2ad4d5bec3af7f5673c00f3b6e75a6664f68e38", "4fe7b59af6de3b665b67788cc2f99892ab827efae3a467342b3bb4e3bc8e5bfe"},
	{"8dee004860f98fe1d0d2983dd29ceaa0064665916fb41f9a689456c742d7e222", "fa43239bcee7b97ca62f007cc68487560a39e19f74f3dde7486db3f98df8e471", "c3595251cb62f338aac08b33b7eb604c0f7f257aca0c5ea0b98d03c386db8895", "4fe7b59af6de3b665b67788cc2f99892ab827efae3a467342b3bb4e3bc8e5bfe"},
	{"ef7de86cf6677d5dfa41c1add9b9f4978f200ab5c27822d23fac340ccdaeed05", "fa43239bcee7b97ca62f007cc68487560a39e19f74f3dde7486db3f98df8e471", "5513fc3364339fd2e2a39cbb7c0b8f2acbfc8ed38736dfa6d87aafb3b6140c8e", "4fe7b59af6de3b665b67788cc2f99892ab827efae3a467342b3bb4e3bc8e5bfe"},
	{"24d1be290ed1f0da0339eade94f43047cef5fc479259b5902d81724a7164da02", "fa43239bcee7b97ca62f007cc68487560a39e19f74f3dde7486db3f98df8e471", "b621c861dc3a61b3974a3158b14d6772f5f9e648201224f6e5437f093aec7a92", "4fe7b59af6de3b665b67788cc2f99892ab827efae3a467342b3bb4e3bc8e5bfe"},
}

func spec() ornith.InitialAdapterSpec {
	return ornith.InitialAdapterSpec{Seed: 83, PreludeBlocks: 24, ExpectedSHA256: aggregate}
}

func initialize(t *testing.T) *ornith.InitialAdapter {
	t.Helper()
	result, err := ornith.InitializeAdapter(context.Background(), spec())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = result.Close() })
	return result
}

func TestExactReferenceHashesGeometryAndTrainableLeaves(t *testing.T) {
	result := initialize(t)
	if len(result.Parameters) != 32 || result.SHA256 != aggregate {
		t.Fatalf("unexpected receipt: %d %s", len(result.Parameters), result.SHA256)
	}
	hash := sha256.New()
	var elements int64
	for index, parameter := range result.Parameters {
		layer, column := index/4, index%4
		suffix := []string{"q_proj.lora_A.default.weight", "q_proj.lora_B.default.weight", "v_proj.lora_A.default.weight", "v_proj.lora_B.default.weight"}[column]
		expectedName := fmt.Sprintf("base_model.model.model.language_model.layers.%d.self_attn.%s", 3+layer*4, suffix)
		if parameter.Name != expectedName {
			t.Fatalf("parameter%d name mismatch", index)
		}
		info, err := parameter.Value.Info()
		if err != nil {
			t.Fatal(err)
		}
		shape := []int64{4, 4096}
		if column == 1 {
			shape = []int64{8192, 4}
		} else if column == 3 {
			shape = []int64{1024, 4}
		}
		if fmt.Sprint(info.Shape) != fmt.Sprint(shape) || info.DType != torch.Float32 || info.Device != torch.CPUDevice() || !info.RequiresGrad {
			t.Fatalf("parameter%d metadata: %+v", index, info)
		}
		elements += info.Elements
		finite, err := parameter.Value.AllFinite()
		if err != nil || !finite {
			t.Fatalf("parameter%d nonfinite: %v", index, err)
		}
		content, err := parameter.Value.Bytes()
		if err != nil {
			t.Fatal(err)
		}
		digest := sha256.Sum256(content)
		if hex.EncodeToString(digest[:]) != reference[layer][column] {
			t.Fatalf("parameter%d bytes differ: %x", index, digest)
		}
		if column%2 == 1 {
			for _, b := range content {
				if b != 0 {
					t.Fatal("B contains a nonzero byte")
				}
			}
		}
		_, _ = hash.Write([]byte(parameter.Name))
		_, _ = hash.Write(content)
	}
	if elements != 557056 || hex.EncodeToString(hash.Sum(nil)) != aggregate {
		t.Fatalf("aggregate or parameter count differs: %d %x", elements, hash.Sum(nil))
	}
	// Both random A and zero B must already be usable as independent leaves.
	for _, index := range []int{0, 1} {
		value := result.Parameters[index].Value
		sum, err := value.Sum(nil, false)
		if err != nil {
			t.Fatal(err)
		}
		seed, err := torch.FromFloat32([]float32{1}, nil, torch.CPUDevice(), false)
		if err != nil {
			_ = sum.Close()
			t.Fatal(err)
		}
		gradients, err := torch.Grad([]*torch.Tensor{sum}, []*torch.Tensor{value}, []*torch.Tensor{seed}, false, false)
		_ = seed.Close()
		_ = sum.Close()
		if err != nil {
			t.Fatal(err)
		}
		gradient, err := gradients[0].Float32Values()
		_ = gradients[0].Close()
		if err != nil {
			t.Fatal(err)
		}
		for _, entry := range gradient {
			if entry != 1 {
				t.Fatal("initializer returned a broken trainable leaf")
			}
		}
	}
	t.Logf("32/32 frozen reference hashes matched; %d FP32 CPU parameters; SHA256 %s", elements, result.SHA256)
}

func TestMandatoryIdentityRejectsChangedSeedPreludeAndDigest(t *testing.T) {
	for name, change := range map[string]func(*ornith.InitialAdapterSpec){
		"different seed":     func(s *ornith.InitialAdapterSpec) { s.Seed = 84 },
		"missing prelude":    func(s *ornith.InitialAdapterSpec) { s.PreludeBlocks = 0 },
		"short prelude":      func(s *ornith.InitialAdapterSpec) { s.PreludeBlocks = 23 },
		"different identity": func(s *ornith.InitialAdapterSpec) { s.ExpectedSHA256 = strings.Repeat("0", 64) },
	} {
		t.Run(name, func(t *testing.T) {
			s := spec()
			change(&s)
			result, err := ornith.InitializeAdapter(context.Background(), s)
			if result != nil {
				_ = result.Close()
				t.Fatal("mismatched identity returned parameters")
			}
			if !errors.Is(err, ornith.ErrInitialAdapterIdentity) {
				t.Fatalf("expected identity rejection, got%v", err)
			}
		})
	}
	for name, change := range map[string]func(*ornith.InitialAdapterSpec){
		"absent identity":    func(s *ornith.InitialAdapterSpec) { s.ExpectedSHA256 = "" },
		"short identity":     func(s *ornith.InitialAdapterSpec) { s.ExpectedSHA256 = aggregate[:63] },
		"uppercase identity": func(s *ornith.InitialAdapterSpec) { s.ExpectedSHA256 = strings.ToUpper(aggregate) },
		"malformed identity": func(s *ornith.InitialAdapterSpec) { s.ExpectedSHA256 = strings.Repeat("g", 64) },
		"negative prelude":   func(s *ornith.InitialAdapterSpec) { s.PreludeBlocks = -1 },
		"unbounded prelude":  func(s *ornith.InitialAdapterSpec) { s.PreludeBlocks = 25 },
	} {
		t.Run(name, func(t *testing.T) {
			s := spec()
			change(&s)
			result, err := ornith.InitializeAdapter(context.Background(), s)
			if result != nil {
				_ = result.Close()
				t.Fatal("invalid specification returned parameters")
			}
			if !errors.Is(err, ornith.ErrInitialAdapterSpec) {
				t.Fatalf("expected specification rejection, got%v", err)
			}
		})
	}
	if result, err := ornith.InitializeAdapter(nil, spec()); result != nil || !errors.Is(err, ornith.ErrInitialAdapterSpec) {
		_ = result.Close()
		t.Fatalf("nil context: %v", err)
	}
}

type checkingContext struct {
	context.Context
	cancel context.CancelFunc
	calls  atomic.Int32
}

func (c *checkingContext) Err() error {
	if c.calls.Add(1) >= 65 {
		c.cancel()
	}
	return c.Context.Err()
}

func TestCancellationReturnsNoPartialAdapterAndDoesNotPoisonNextCall(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if result, err := ornith.InitializeAdapter(ctx, spec()); result != nil || !errors.Is(err, context.Canceled) {
		_ = result.Close()
		t.Fatalf("pre-canceled initialization: %v", err)
	}
	ctx, cancel = context.WithCancel(context.Background())
	defer cancel()
	checking := &checkingContext{Context: ctx, cancel: cancel}
	result, err := ornith.InitializeAdapter(checking, spec())
	if result != nil || !errors.Is(err, context.Canceled) {
		_ = result.Close()
		t.Fatalf("mid-initialization cancellation: %v", err)
	}
	if checking.calls.Load() < 65 {
		t.Fatal("cancellation fixture did not reach partial construction")
	}
	if got := initialize(t); got.SHA256 != aggregate {
		t.Fatal("failed call changed the next stream")
	}
}

func TestIndependentResultsAndCompleteIdempotentClose(t *testing.T) {
	first, second := initialize(t), initialize(t)
	if first.SHA256 != second.SHA256 {
		t.Fatal("private streams diverged")
	}
	for index := range first.Parameters {
		if first.Parameters[index].Value == second.Parameters[index].Value {
			t.Fatal("results share a caller-owned tensor handle")
		}
	}
	handles := make([]*torch.Tensor, len(first.Parameters))
	for index, p := range first.Parameters {
		handles[index] = p.Value
	}
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}
	for _, value := range handles {
		if _, err := value.Info(); !errors.Is(err, torch.ErrClosed) {
			t.Fatalf("Close left a parameter alive: %v", err)
		}
	}
	for _, parameter := range second.Parameters {
		if _, err := parameter.Value.Info(); err != nil {
			t.Fatalf("closing first result invalidated second: %v", err)
		}
	}
	var absent *ornith.InitialAdapter
	if err := absent.Close(); err != nil {
		t.Fatal(err)
	}
}
