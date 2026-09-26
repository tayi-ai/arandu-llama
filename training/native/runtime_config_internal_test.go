package native

import (
	"context"
	"errors"
	"io"
	"slices"
	"testing"
	"time"
)

func TestRuntimeSnapshotOwnsMutableInvocationIdentity(t *testing.T) {
	input := RuntimeConfig{GPUIds: []int{0, 1}, Job: Job{Request: []byte(`{"mode":"train"}`)}}
	owned := input.snapshot()
	input.GPUIds[0] = 7
	input.Job.Request[0] = '!'
	if !slices.Equal(owned.GPUIds, []int{0, 1}) || string(owned.Job.Request) != `{"mode":"train"}` {
		t.Fatal("runtime invocation aliases caller identity")
	}
}

func TestRuntimeValidationRequiresEveryExplicitSeamWithoutCallingIt(t *testing.T) {
	called := false
	resource := func() (int64, error) { called = true; return 0, errors.New("not a measured host") }
	c := RuntimeConfig{ExecutablePath: "/synthetic/runtime", Output: io.Discard, HostAvailableBytes: resource, CgroupAvailableBytes: resource,
		ProbeGPUs: func(context.Context, float64) (GPUProbe, error) {
			called = true
			return GPUProbe{}, errors.New("not a measured GPU")
		},
		CreateExchange: func(context.Context, ExchangeRequest) (ExperimentGradientExchange, error) {
			called = true
			return nil, errors.New("not a transport")
		}}
	if err := c.validate(context.Background()); err != nil || called {
		t.Fatalf("structural validation accessed runtime: %v", err)
	}
	for _, change := range []func(*RuntimeConfig){
		func(c *RuntimeConfig) { c.HostAvailableBytes = nil }, func(c *RuntimeConfig) { c.CgroupAvailableBytes = nil },
		func(c *RuntimeConfig) { c.ProbeGPUs = nil }, func(c *RuntimeConfig) { c.CreateExchange = nil },
		func(c *RuntimeConfig) { c.Output = nil }, func(c *RuntimeConfig) { c.ExecutablePath = "relative" },
	} {
		bad := c
		change(&bad)
		if bad.validate(context.Background()) == nil {
			t.Fatal("missing runtime seam accepted")
		}
	}
	if called {
		t.Fatal("validation invoked callback")
	}
}

func TestCollectiveBoundaryRetainsExactNumericalBudgetsAndNoCredentials(t *testing.T) {
	sentinel := errors.New("transport intentionally unopened")
	called := false
	c := RuntimeConfig{CreateExchange: func(_ context.Context, request ExchangeRequest) (ExperimentGradientExchange, error) {
		called = true
		s := request.Spec
		if s.World != 20 || s.Rank != 3 || s.Steps != 18 || s.Dimension != 557056 || s.SessionID != "run-gradient" || s.PhaseTimeout != 180*time.Second || s.MaxFrameBytes != 4<<20 || s.MaxBufferedBytes != 64<<20 || len(s.Secret) != 0 {
			t.Fatal("collective numerical contract or credential boundary changed")
		}
		return nil, sentinel
	}}
	_, err := experimentNativeExchange(testNativeExperiment(), c, context.Background(), 3, 20, "run-gradient", "contract", "layout", 18, 557056, nil)
	if !called || !errors.Is(err, sentinel) {
		t.Fatalf("transport result: %v", err)
	}
	called = false
	if _, err := experimentNativeExchange(testNativeExperiment(), c, context.Background(), 20, 20, "run-gradient", "contract", "layout", 18, 557056, nil); err == nil || called {
		t.Fatal("invalid topology reached transport")
	}
}
