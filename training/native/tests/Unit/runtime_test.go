package native_test

import (
	"context"
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"

	services "github.com/tayi-ai/arandu-llama/training/native"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

// This check deliberately supplies no admitted bundle or model. A build lacking
// CUDA must stop at the capability boundary, before files, environment secrets,
// processes, tensor construction or a CPU fallback can be used by the runtime.
func TestNativeRuntimeRefusesUnsupportedBackendBeforeAdmission(t *testing.T) {
	if torch.CUDAEnabled() {
		t.Skip("requires a build without the CUDA bridge; this test never launches a GPU runtime")
	}
	t.Setenv("PATH", "")
	t.Setenv("TAYI_EXPERIMENT_COLLECTIVE_KEY", "unadmitted-sentinel")
	t.Setenv("TAYI_EXPERIMENT_RELEASE_SHA256", "invalid-release")
	t.Setenv("RANK", "not-a-rank")
	t.Setenv("WORLD_SIZE", "1")
	t.Setenv("CUDA_VISIBLE_DEVICES", "")
	root := t.TempDir()
	contract := filepath.Join(root, "missing-bundle", "contract.json")
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	for _, sample := range []struct {
		name string
		mode string
		ctx  context.Context
	}{
		{"canary", "canary", context.Background()},
		{"train", "train", context.Background()},
		{"unknown_mode", "unsupported", context.Background()},
		{"cancelled", "train", cancelled},
		{"nil_context", "canary", nil},
	} {
		t.Run(sample.name, func(t *testing.T) {
			output := filepath.Join(root, sample.name)
			if err := services.RunExperimentNative(testNativeExperiment(), sample.ctx, services.RuntimeConfig{Mode: sample.mode, ContractPath: contract, OutputDirectory: output}); !errors.Is(err, services.ErrExperimentNativeBackendUnqualified) {
				t.Fatalf("missing CUDA must refuse before admission, got %v", err)
			}
			if _, err := os.Lstat(output); !errors.Is(err, os.ErrNotExist) {
				t.Fatal("unsupported runtime created or accessed an output artifact")
			}
			if os.Getenv("TAYI_EXPERIMENT_COLLECTIVE_KEY") != "unadmitted-sentinel" {
				t.Fatal("unsupported runtime consumed a supervisor secret")
			}
		})
	}
	entries, err := os.ReadDir(root)
	if err != nil || len(entries) != 0 {
		t.Fatal("unsupported runtime wrote files before backend admission")
	}
}

func TestNativeRuntimeCannotUseExistingOutputAsCPUFallback(t *testing.T) {
	if torch.CUDAEnabled() {
		t.Skip("requires a build without the CUDA bridge; this test never launches a GPU runtime")
	}
	root := t.TempDir()
	output := filepath.Join(root, "existing-output")
	const content = "existing evidence must remain unchanged\n"
	if err := os.WriteFile(output, []byte(content), 0600); err != nil {
		t.Fatal(err)
	}
	if err := services.RunExperimentNative(testNativeExperiment(), context.Background(), services.RuntimeConfig{Mode: "train", OutputDirectory: output}); !errors.Is(err, services.ErrExperimentNativeBackendUnqualified) {
		t.Fatalf("unsupported runtime attempted another backend: %v", err)
	}
	body, err := os.ReadFile(output)
	if err != nil || string(body) != content {
		t.Fatal("unsupported runtime modified existing evidence")
	}
	entries, err := os.ReadDir(root)
	if err != nil || len(entries) != 1 {
		t.Fatal("unsupported runtime created additional artifacts")
	}
}

func TestNativeRuntimeMemoryAdmitsFrozenMaximumAndRejectsOverflow(t *testing.T) {
	// Independent frozen-corpus fixture: 1128 measured prompt tokens plus A,
	// a 248320x4096 FP16 largest tensor and explicit host/cgroup reserves.
	plan, err := services.ExperimentNativeMemory(1129)
	if err != nil {
		t.Fatal(err)
	}
	want := services.ExperimentNativeMemoryPlan{
		MaximumSequenceLength: 1129,
		LargestTensorBytes:    2034237440,
		TensorCopyBytes:       6102712320,
		HostOverheadBytes:     2147483648,
		HostReserveBytes:      6442450944,
		HostPreloadBytes:      14692646912,
		CheckpointBytes:       314458112,
		WorkingElements:       891134976,
		SequenceElements:      99215936,
	}
	if plan != want {
		t.Fatalf("frozen memory admission changed: got %+v, want %+v", plan, want)
	}
	for _, tokens := range []int64{math.MinInt64, -1, 0, 1, 1130, math.MaxInt64} {
		plan, err := services.ExperimentNativeMemory(tokens)
		if err == nil || plan != (services.ExperimentNativeMemoryPlan{}) {
			t.Fatalf("out-of-domain length %d produced an admission", tokens)
		}
	}
}

func TestNativeRuntimeAllocatorAdmitsT10ByteRoundedFraction(t *testing.T) {
	// Real CUDA-visible T10 total from the frozen loader-reference summary.
	// LibTorch 2.14 stores round(.80 * total) bytes and reports that / total.
	const total int64 = 16703356928
	const ceiling int64 = 13362685542
	stats := torch.CUDAMemoryStats{TotalBytes: total, AllocatorEnabled: true, AllocatorStatsValid: true,
		AllocatorFraction: float64(ceiling) / float64(total), PeakReservedBytes: ceiling}
	if stats.AllocatorFraction == .80 {
		t.Fatal("fixture must exercise the non-identical reported fraction")
	}
	if err := services.ValidateExperimentNativeAllocator(stats); err != nil {
		t.Fatalf("correct byte limit rejected: %v", err)
	}
	for _, sample := range []struct {
		name   string
		change func(*torch.CUDAMemoryStats)
	}{
		{"peak_one_byte_above", func(s *torch.CUDAMemoryStats) { s.PeakReservedBytes = ceiling + 1 }},
		{"allocator_one_byte_above", func(s *torch.CUDAMemoryStats) { s.AllocatorFraction = float64(ceiling+1) / float64(total) }},
		{"allocator_one_byte_below", func(s *torch.CUDAMemoryStats) { s.AllocatorFraction = float64(ceiling-1) / float64(total) }},
		{"disabled", func(s *torch.CUDAMemoryStats) { s.AllocatorEnabled = false }},
		{"invalid_stats", func(s *torch.CUDAMemoryStats) { s.AllocatorStatsValid = false }},
		{"zero_total", func(s *torch.CUDAMemoryStats) { s.TotalBytes = 0 }},
		{"inexact_total", func(s *torch.CUDAMemoryStats) { s.TotalBytes = 1<<53 + 1 }},
		{"nan_fraction", func(s *torch.CUDAMemoryStats) { s.AllocatorFraction = math.NaN() }},
		{"infinite_fraction", func(s *torch.CUDAMemoryStats) { s.AllocatorFraction = math.Inf(1) }},
		{"negative_peak", func(s *torch.CUDAMemoryStats) { s.PeakReservedBytes = -1 }},
	} {
		t.Run(sample.name, func(t *testing.T) {
			invalid := stats
			sample.change(&invalid)
			if err := services.ValidateExperimentNativeAllocator(invalid); err == nil {
				t.Fatal("invalid allocator admission succeeded")
			}
		})
	}
}

func TestNativeRuntimeAllocatorDoesNotRoundUpFrozenCeiling(t *testing.T) {
	// A fractional-byte remainder above .5 makes LibTorch round upward. The
	// application's stricter floor ceiling must still reject that extra byte.
	const total int64 = 16703356931
	const ceiling int64 = 13362685544
	stats := torch.CUDAMemoryStats{TotalBytes: total, AllocatorEnabled: true, AllocatorStatsValid: true,
		AllocatorFraction: float64(ceiling+1) / float64(total), PeakReservedBytes: ceiling}
	if err := services.ValidateExperimentNativeAllocator(stats); err == nil {
		t.Fatal("allocator limit above the floor ceiling was admitted")
	}
	stats.AllocatorFraction = float64(ceiling) / float64(total)
	if err := services.ValidateExperimentNativeAllocator(stats); err != nil {
		t.Fatalf("exact floor ceiling rejected: %v", err)
	}
}
