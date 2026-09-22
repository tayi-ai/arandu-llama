package support

import (
	"runtime"
	"strconv"
	"time"
)

// benchmarkFacade carries the measurement calls. It is reached through the
// [Benchmark] value rather than constructed, the way [Env] is.
type benchmarkFacade struct{}

// Benchmark measures how long a callback takes, in milliseconds.
var Benchmark benchmarkFacade

// Measure returns the average run of each callback, in milliseconds, in the
// order the callbacks were given. The variadic argument is the iteration count
// and defaults to 1; fewer than one iteration measures zero.
//
// A garbage collection is forced before every run, so the cost of collecting
// what an earlier run left behind is not charged to this one.
func (benchmarkFacade) Measure(benchmarkables []func(), iterations ...int) []float64 {
	count := firstOr(iterations, 1)
	results := make([]float64, 0, len(benchmarkables))
	for _, callback := range benchmarkables {
		if count < 1 {
			results = append(results, 0)
			continue
		}
		total := float64(0)
		for i := 0; i < count; i++ {
			runtime.GC()
			start := time.Now()
			callback()
			total += float64(time.Since(start)) / float64(time.Millisecond)
		}
		results = append(results, total/float64(count))
	}
	return results
}

// Value returns what the callback returned and how long it took, in
// milliseconds. The value keeps the type it went in with.
func Value[T any](callback func() T) (T, float64) {
	runtime.GC()
	start := time.Now()
	result := callback()
	return result, float64(time.Since(start)) / float64(time.Millisecond)
}

// Dd measures, writes the averages out with three decimals and an ms suffix,
// then ends the process.
func (b benchmarkFacade) Dd(benchmarkables []func(), iterations ...int) {
	measured := b.Measure(benchmarkables, iterations...)
	written := make([]string, 0, len(measured))
	for _, average := range measured {
		written = append(written, strconv.FormatFloat(average, 'f', 3, 64)+"ms")
	}
	Dd(written)
}
