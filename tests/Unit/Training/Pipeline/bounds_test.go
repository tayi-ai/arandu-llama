package pipeline_test

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/pipeline"
	"github.com/tayi-ai/arandu-llama/training/protection"
)

type admittedBudgetModel struct {
	scalarModel
	stop error
}

func (m *admittedBudgetModel) Objective(context.Context) (pipeline.Objective, error) {
	m.calls++
	return pipeline.Objective{}, m.stop
}

func completeProtectionPool(config pipeline.StepConfig) pipeline.StepConfig {
	pair := config.Protection[0]
	config.Protection = nil
	for i := range 33 {
		copy := pair
		copy.ID = fmt.Sprintf("protected-%d", i)
		config.Protection = append(config.Protection, copy)
	}
	return config
}

func TestStepHonorsExplicitBudgetBeforeComputingFullProtectionPool(t *testing.T) {
	const parameters = 557056
	stop := errors.New("stop before numerical work")
	m := &admittedBudgetModel{scalarModel: scalarModel{values: make([]float32, parameters)}, stop: stop}
	config := completeProtectionPool(stepConfig())
	if _, err := pipeline.Step(context.Background(), m, config); !errors.Is(err, protection.ErrLimit) || m.calls != 0 {
		t.Fatalf("default must refuse before objective: %v", err)
	}
	config.Solver.MaxCoefficients = parameters * len(config.Protection)
	if _, err := pipeline.Step(context.Background(), m, config); !errors.Is(err, stop) || m.calls != 1 {
		t.Fatalf("explicit full-pool budget was not honored: %v", err)
	}
	config.Solver.PrimalTolerance = 0
	if _, err := pipeline.Step(context.Background(), m, config); !errors.Is(err, protection.ErrProblem) || m.calls != 1 {
		t.Fatalf("invalid numerical policy reached objective: %v", err)
	}
}

func TestConstrainedAdmissionUsesSameExplicitSolverBudget(t *testing.T) {
	const parameters = 557056
	config, stageContext, tracker := constrainedFixture(t, 1)
	p := &config.Protocol
	p.Layout[0].Shape = []uint64{parameters}
	p.Limits.MaxParameters = parameters
	p.Limits.MaxCheckpointBytes = 4 << 20
	p.Limits.MaxTotalBytes = 8 << 20
	p.Updates[0].Config = completeProtectionPool(p.Updates[0].Config)
	if _, err := p.Digest(); err == nil {
		t.Fatal("default budget unexpectedly admits complete pool")
	}
	p.Updates[0].Config.Solver.MaxCoefficients = parameters * len(p.Updates[0].Config.Protection)
	constrainedPin(t, &config)
	h := constrainedHandler(t, config)
	if err := h.Admit(context.Background(), config.Recipe, stageContext.Stage, config.Placement); err != nil {
		t.Fatal(err)
	}
	if tracker.opens.Load() != 0 {
		t.Fatal("admission loaded a model")
	}
}
