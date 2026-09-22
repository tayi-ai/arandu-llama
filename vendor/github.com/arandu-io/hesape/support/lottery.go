package support

import (
	"crypto/rand"
	"errors"
	"math"
	"math/big"
	"sync"
)

// ErrFloatGreaterThanOne is returned by [NewLottery] and [Odds] for chances
// above one with no total to be out of.
var ErrFloatGreaterThanOne = errors.New("Float must not be greater than 1.")

// Lottery runs one callback or the other at the odds given, so a slow path
// runs on a fraction of the requests.
type Lottery struct {
	chances float64
	outOf   *int
	winner  func() any
	loser   func() any
}

var (
	lotteryMu            sync.Mutex
	lotteryResultFactory func(chances float64, outOf *int) bool
)

// NewLottery builds a lottery at the given chances. The variadic argument is
// the total to be out of; with no total the chances are read as a probability
// in [0, 1], and anything above one is [ErrFloatGreaterThanOne].
func NewLottery(chances float64, outOf ...int) (*Lottery, error) {
	l := &Lottery{chances: chances}
	if len(outOf) > 0 {
		total := outOf[0]
		l.outOf = &total
		return l, nil
	}
	if chances > 1 {
		return nil, ErrFloatGreaterThanOne
	}
	return l, nil
}

// Odds builds a lottery at the given chances, the same as [NewLottery].
func Odds(chances float64, outOf ...int) (*Lottery, error) {
	return NewLottery(chances, outOf...)
}

// Winner sets the callback run when the draw wins, and returns the lottery.
func (l *Lottery) Winner(callback func() any) *Lottery {
	l.winner = callback
	return l
}

// Loser sets the callback run when the draw loses, and returns the lottery.
func (l *Lottery) Loser(callback func() any) *Lottery {
	l.loser = callback
	return l
}

// Choose draws. With no argument it draws once and returns what the callback
// returned; with a count it draws that many times and returns a []any of the
// results.
func (l *Lottery) Choose(times ...int) any {
	if len(times) == 0 {
		return l.runCallback()
	}
	results := []any{}
	for i := 0; i < times[0]; i++ {
		results = append(results, l.runCallback())
	}
	return results
}

// runCallback draws once and runs the matching callback. A lottery carrying no
// callback of its own returns true when it wins and false when it loses.
func (l *Lottery) runCallback() any {
	if l.wins() {
		if l.winner == nil {
			return true
		}
		return l.winner()
	}
	if l.loser == nil {
		return false
	}
	return l.loser()
}

// wins draws once through the current result factory.
func (l *Lottery) wins() bool { return resultFactory()(l.chances, l.outOf) }

// resultFactory returns the factory a draw goes through, which is the one
// [SetResultFactory] installed, or the default when none is installed.
func resultFactory() func(chances float64, outOf *int) bool {
	lotteryMu.Lock()
	factory := lotteryResultFactory
	lotteryMu.Unlock()
	if factory != nil {
		return factory
	}
	return defaultResultFactory
}

// defaultResultFactory draws honestly: with no total, a random number in
// [0, 1] at or under the chances; with one, a draw from 1 to the total that
// lands at or under the chances. A total of zero or less never wins.
func defaultResultFactory(chances float64, outOf *int) bool {
	if outOf == nil {
		return randomFloat() <= chances
	}
	if *outOf <= 0 {
		return false
	}
	return float64(randomInt(1, *outOf)) <= chances
}

// randomFloat draws a number in [0, 1] from the cryptographic source. An
// unreadable source draws zero.
func randomFloat() float64 {
	drawn, err := rand.Int(rand.Reader, big.NewInt(math.MaxInt64))
	if err != nil {
		return 0
	}
	return float64(drawn.Int64()) / float64(math.MaxInt64)
}

// randomInt draws an integer between minimum and maximum, both ends included.
// A maximum at or below the minimum is the minimum.
func randomInt(minimum, maximum int) int {
	if maximum <= minimum {
		return minimum
	}
	drawn, err := rand.Int(rand.Reader, big.NewInt(int64(maximum-minimum+1)))
	if err != nil {
		return minimum
	}
	return minimum + int(drawn.Int64())
}

// AlwaysWin makes every later draw win. The variadic argument is a callback:
// given one, drawing goes back to normal once it has run.
func AlwaysWin(callback ...func()) {
	SetResultFactory(func(float64, *int) bool { return true })
	if len(callback) == 0 || callback[0] == nil {
		return
	}
	callback[0]()
	DetermineResultNormally()
}

// AlwaysLose makes every later draw lose. The variadic argument is a callback:
// given one, drawing goes back to normal once it has run.
func AlwaysLose(callback ...func()) {
	SetResultFactory(func(float64, *int) bool { return false })
	if len(callback) == 0 || callback[0] == nil {
		return
	}
	callback[0]()
	DetermineResultNormally()
}

// Fix pins the results a draw gives, the same as [ForceResultWithSequence].
func Fix(sequence []bool, whenMissing ...func(chances float64, outOf *int) bool) {
	ForceResultWithSequence(sequence, whenMissing...)
}

// ForceResultWithSequence pins the results a draw gives, in order, and says
// what to do once they run out.
//
// The variadic argument is the fallback: with none, drawing goes back to
// normal from there on.
func ForceResultWithSequence(sequence []bool, whenMissing ...func(chances float64, outOf *int) bool) {
	var mu sync.Mutex
	next := 0

	missing := firstOr(whenMissing, nil)
	if missing == nil {
		missing = func(chances float64, outOf *int) bool {
			result := defaultResultFactory(chances, outOf)
			next++
			return result
		}
	}

	SetResultFactory(func(chances float64, outOf *int) bool {
		mu.Lock()
		defer mu.Unlock()
		if next < len(sequence) {
			result := sequence[next]
			next++
			return result
		}
		return missing(chances, outOf)
	})
}

// DetermineResultsNormally drops any pinned result, the same as
// [DetermineResultNormally].
func DetermineResultsNormally() { DetermineResultNormally() }

// DetermineResultNormally drops any pinned result, so draws are random again.
func DetermineResultNormally() {
	lotteryMu.Lock()
	defer lotteryMu.Unlock()
	lotteryResultFactory = nil
}

// SetResultFactory installs the function every later draw goes through. It is
// process-wide, so a test that sets it must put it back with
// [DetermineResultNormally].
func SetResultFactory(factory func(chances float64, outOf *int) bool) {
	lotteryMu.Lock()
	defer lotteryMu.Unlock()
	lotteryResultFactory = factory
}
