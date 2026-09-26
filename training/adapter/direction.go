package adapter

import "math/rand/v2"

// Direction identifies a reproducible standard-normal perturbation by seed.
// Scale is the declared epsilon used for the finite-difference probes.
type Direction struct {
	Seed  uint64  `json:"seed"`
	Scale float64 `json:"scale"`
}

// Fill writes the direction into the caller's buffer in stable element order.
func (d Direction) Fill(into []float32) {
	source := d.stream()
	for i := range into {
		into[i] = float32(source.NormFloat64())
	}
}

// stream preserves the same generator for adapter initialization and probes.
func (d Direction) stream() *rand.Rand {
	return rand.New(rand.NewPCG(d.Seed, d.Seed^0x9e3779b97f4a7c15))
}
