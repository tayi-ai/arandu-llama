package llama

import (
	"math"
	"testing"
)

func TestTeacherMassAcceptsOnlyBoundedSummationRoundoff(t *testing.T) {
	for _, mass := range []float64{0, 0.125, 0.875, 1} {
		got, err := boundedTeacherMass(mass)
		if err != nil || got != mass {
			t.Fatalf("mass %.17g changed to %.17g: %v", mass, got, err)
		}
	}
	// Different accumulation orders can exceed one even with a complete softmax.
	for _, mass := range []float64{math.Nextafter(1, 2), 1 + 5e-11} {
		got, err := boundedTeacherMass(mass)
		if err != nil || got != 1 {
			t.Fatalf("bounded rounding %.17g was not canonicalized: %v", mass, err)
		}
	}
	for _, mass := range []float64{-math.SmallestNonzeroFloat64, 1 + 2e-10, 2, math.NaN(), math.Inf(1), math.Inf(-1)} {
		if _, err := boundedTeacherMass(mass); err == nil {
			t.Fatalf("invalid mass %.17g accepted", mass)
		}
	}
}
