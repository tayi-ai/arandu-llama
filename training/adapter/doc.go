// Package adapter implements bounded GGUF adapter operations and independent
// LoRA arithmetic. Callers own admission, authorization and persistent job state.
// A Snapshot is not goroutine-safe: serialize perturbation, folding and Close.
package adapter
