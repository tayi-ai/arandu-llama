package pipeline

import "context"

// Placement declares a measured execution budget independently of model roles.
// Nodes are inventory IDs and contain no addresses or credentials.
type Placement struct {
	Backend             string
	Nodes               []string
	MemoryBytes         int64
	MaxTokens           int
	RuntimeSHA256       string
	QualificationSHA256 string
}

// Execution binds a backend operation to the application's durable fence.
// The runtime owns its configured artifact root; requests carry no paths.
type Execution struct {
	RunID        string
	TenantID     string
	Generation   uint64
	Recipe       Recipe
	TargetSHA256 string
	Placement    Placement
}

// Progress reports only a checkpoint already materialized and verified by the
// backend. CompletedSteps is cumulative across the ordered recipe stages.
type Progress struct {
	StageID        string
	CompletedSteps int64
	Checkpoint     string
}

// Runtime is the versioned module boundary used by both local queue workers and
// fleet agents. Admit must verify support for every stage and its independent
// qualification; it must not launch work. Reconcile fences prior work and reads
// its verified checkpoint. Run calls report after each durable commit and stops
// if report fails. Neither operation returns while its compute still runs.
// Implementations do not own an HTTP listener, application database or queue.
type Runtime interface {
	Admit(context.Context, Recipe, Placement) error
	Reconcile(context.Context, Execution) (Progress, error)
	Run(context.Context, Execution, func(context.Context, Progress) error) error
}
