package native

import (
	"context"
	"errors"
	"io"
	"path/filepath"
	"slices"
	"time"

	"github.com/tayi-ai/arandu-llama/training/collective"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

// RuntimeAvailable reports native CUDA bridge capability without initializing
// a device or qualifying any model, recipe, artifact, or installed GPU.
func RuntimeAvailable() bool { return torch.Enabled() && torch.CUDAEnabled() }

// GPU describes independently measured driver capacity, before model loading.
type GPU struct {
	Index      int    `json:"index"`
	Name       string `json:"name"`
	Capability string `json:"capability"`
	TotalBytes int64  `json:"total_bytes"`
	FreeBytes  int64  `json:"free_bytes"`
	CapBytes   int64  `json:"cap_bytes"`
}

// GPUProbe records the result of the application's admitted driver probe.
type GPUProbe struct {
	Driver       string  `json:"driver"`
	VRAMFraction float64 `json:"vram_fraction"`
	GPUs         []GPU   `json:"gpus"`
}

// ExchangeRequest fixes the numerical wire contract. The application supplies
// transport, private addressing and authentication without changing this spec.
// Spec.Secret is empty; transport credentials never enter a recipe or result.
type ExchangeRequest struct {
	Spec   collective.Spec
	Update collective.Update
}

// RuntimeConfig supplies one authorized invocation and its host/transport seams.
// Callbacks must be concurrency-safe, bounded by context where supplied, and
// report actual measurements. They must not launch another training execution.
// Callers must not mutate this value during handoff. No environment is read.
type RuntimeConfig struct {
	Mode, ContractPath, OutputDirectory, ExecutablePath string
	Release                                             ExperimentNativeRelease
	Job                                                 Job
	ContractSHA256, RuntimeSHA256                       string
	NodeID                                              string
	Rank, World                                         int
	GPUIds                                              []int
	VRAMFraction                                        float64
	HostAvailableBytes                                  func() (int64, error)
	CgroupAvailableBytes                                func() (int64, error)
	ProbeGPUs                                           func(context.Context, float64) (GPUProbe, error)
	CreateExchange                                      func(context.Context, ExchangeRequest) (ExperimentGradientExchange, error)
	Output                                              io.Writer
}

func (c RuntimeConfig) validate(ctx context.Context) error {
	if ctx == nil {
		return errors.New("experiment native: context required")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if c.HostAvailableBytes == nil || c.CgroupAvailableBytes == nil || c.ProbeGPUs == nil || c.CreateExchange == nil || experimentNativeNil(c.Output) || !filepath.IsAbs(c.ExecutablePath) {
		return errors.New("experiment native: explicit resources, transport, output and executable required")
	}
	return nil
}

func (c RuntimeConfig) snapshot() RuntimeConfig {
	c.GPUIds = slices.Clone(c.GPUIds)
	c.Job.Request = slices.Clone(c.Job.Request)
	return c
}

func experimentNativeExchange(experiment *NativeExperiment, config RuntimeConfig, ctx context.Context, rank, world int, session, contract, layout string, steps, dimension uint32, update collective.Update) (ExperimentGradientExchange, error) {
	if experiment == nil {
		return nil, errors.New("native experiment: installation is not configured")
	}
	if world != len(ExperimentNativeNodeIDs(experiment)) || rank < 0 || rank >= world {
		return nil, errors.New("experiment native: collective topology differs from the admitted release")
	}
	if config.CreateExchange == nil {
		return nil, errors.New("experiment native: collective transport is not configured")
	}
	spec := collective.Spec{World: uint32(world), Rank: uint32(rank), Steps: steps, Dimension: dimension, SessionID: session, ContractSHA256: contract, TensorLayoutSHA256: layout, PhaseTimeout: 180 * time.Second, MaxFrameBytes: 4 << 20, MaxBufferedBytes: 64 << 20}
	exchange, err := config.CreateExchange(ctx, ExchangeRequest{Spec: spec, Update: update})
	if err == nil && experimentNativeNil(exchange) {
		return nil, errors.New("experiment native: collective transport returned no exchange")
	}
	return exchange, err
}
