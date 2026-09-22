package support

import (
	"errors"
	"sync"
	"testing"
	"time"
)

// ErrNoDurationSpecified is held on a [Sleep] when a unit is asked for and no
// number is pending, and handed back by [Sleep.Goodnight].
var ErrNoDurationSpecified = errors.New("No duration specified.")

// ErrUnknownDurationUnit is returned by [Sleep.Goodnight] when a number was
// given and no unit ever followed it.
var ErrUnknownDurationUnit = errors.New("Unknown duration unit.")

// Sleep is a duration built up unit by unit, which a test can capture instead
// of waiting out.
//
// Nothing is slept until [Sleep.Goodnight] or [Sleep.Then] is called: there is
// no destructor to sleep from, so the end of the chain is where the wait
// happens. Calling either twice sleeps once.
type Sleep struct {
	// Duration is how long the sleep has been built up to so far.
	Duration time.Duration

	// while keeps the sleep repeating for as long as it returns true. It is
	// not named While because [Sleep.While] already is, and a field and a
	// method cannot share one name in Go.
	while func() bool

	pending      *float64
	shouldSleep  bool
	alreadySlept bool
	err          error
}

var (
	sleepMu             sync.Mutex
	sleepFake           bool
	sleepSequence       []time.Duration
	sleepFakeCallbacks  []func(duration time.Duration)
	sleepSyncWithCarbon bool
)

// newSleep builds a sleep that will wait unless told otherwise.
func newSleep(duration any) *Sleep {
	s := &Sleep{shouldSleep: true}
	return s.duration(duration)
}

// For starts a sleep. A time.Duration is the whole duration; any other number
// is left pending until a unit method names it, as in For(2).Seconds().
func For(duration any) *Sleep { return newSleep(duration) }

// Until starts a sleep that runs until the given instant, which is a time.Time
// or a number of seconds since the epoch. An instant already past is no sleep
// at all.
func Until(timestamp any) *Sleep {
	var target time.Time
	switch t := timestamp.(type) {
	case time.Time:
		target = t
	case *time.Time:
		if t != nil {
			target = *t
		}
	default:
		target = CreateFromTimestamp(toSeconds(timestamp))
	}
	return newSleep(target.Sub(Now()))
}

// Usleep starts a sleep of the given number of microseconds.
func Usleep(duration int) *Sleep { return newSleep(duration).Microseconds() }

// duration sets the sleep from a time.Duration, or leaves a number pending for
// a unit method. A negative duration is clamped to nothing.
func (s *Sleep) duration(duration any) *Sleep {
	if d, ok := duration.(time.Duration); ok {
		if d < 0 {
			d = 0
		}
		s.Duration = d
		s.pending = nil
		return s
	}
	s.Duration = 0
	pending := float64(0)
	switch n := duration.(type) {
	case int:
		pending = float64(n)
	case int64:
		pending = float64(n)
	case float32:
		pending = float64(n)
	case float64:
		pending = n
	default:
		pending = toFloat(duration)
	}
	s.pending = &pending
	return s
}

// pullPending takes the pending number, clamped at zero, and clears it. With
// nothing pending it stops the sleep and holds [ErrNoDurationSpecified] on the
// instance, because a fluent call has no room to return an error.
func (s *Sleep) pullPending() float64 {
	if s.pending == nil {
		s.shouldNotSleep()
		if s.err == nil {
			s.err = ErrNoDurationSpecified
		}
		return 0
	}
	pending := *s.pending
	if pending < 0 {
		pending = 0
	}
	s.pending = nil
	return pending
}

func (s *Sleep) add(unit time.Duration) *Sleep {
	s.Duration += time.Duration(s.pullPending() * float64(unit))
	return s
}

// Minutes reads the pending number as minutes and adds it to the duration.
func (s *Sleep) Minutes() *Sleep { return s.add(time.Minute) }

// Minute is [Sleep.Minutes] under the singular name.
func (s *Sleep) Minute() *Sleep { return s.Minutes() }

// Seconds reads the pending number as seconds and adds it to the duration.
func (s *Sleep) Seconds() *Sleep { return s.add(time.Second) }

// Second is [Sleep.Seconds] under the singular name.
func (s *Sleep) Second() *Sleep { return s.Seconds() }

// Milliseconds reads the pending number as milliseconds and adds it to the
// duration.
func (s *Sleep) Milliseconds() *Sleep { return s.add(time.Millisecond) }

// Millisecond is [Sleep.Milliseconds] under the singular name.
func (s *Sleep) Millisecond() *Sleep { return s.Milliseconds() }

// Microseconds reads the pending number as microseconds and adds it to the
// duration.
func (s *Sleep) Microseconds() *Sleep { return s.add(time.Microsecond) }

// Microsecond is [Sleep.Microseconds] under the singular name.
func (s *Sleep) Microsecond() *Sleep { return s.Microseconds() }

// And leaves another number pending, waiting for its unit.
func (s *Sleep) And(duration any) *Sleep {
	pending := toFloat(duration)
	s.pending = &pending
	return s
}

// While keeps the sleep repeating for as long as the callback returns true.
func (s *Sleep) While(callback func() bool) *Sleep {
	s.while = callback
	return s
}

// When sleeps only when the condition holds. The condition is a bool, a
// func() bool or a func(*Sleep) bool; anything else is read for its truth.
func (s *Sleep) When(condition any) *Sleep {
	switch c := condition.(type) {
	case bool:
		s.shouldSleep = c
	case func() bool:
		s.shouldSleep = c()
	case func(*Sleep) bool:
		s.shouldSleep = c(s)
	default:
		s.shouldSleep = toBool(condition)
	}
	return s
}

// Unless sleeps only when the condition does not hold. It accepts the same
// shapes as [Sleep.When].
func (s *Sleep) Unless(condition any) *Sleep {
	switch c := condition.(type) {
	case bool:
		return s.When(!c)
	case func() bool:
		return s.When(!c())
	case func(*Sleep) bool:
		return s.When(!c(s))
	default:
		return s.When(!toBool(condition))
	}
}

// shouldNotSleep marks the sleep so that nothing is waited out.
func (s *Sleep) shouldNotSleep() *Sleep {
	s.shouldSleep = false
	return s
}

// Then sleeps, then runs the callback and hands back what it returned. A sleep
// that failed returns its error and the callback does not run.
func (s *Sleep) Then(then func() any) (any, error) {
	if err := s.Goodnight(); err != nil {
		return nil, err
	}
	s.alreadySlept = true
	return then(), nil
}

// Goodnight waits the duration out, and is what a caller writes at the end of
// the chain. Calling it twice sleeps once.
//
// A number left without a unit is [ErrUnknownDurationUnit]. While a test is
// faking, nothing is waited: the duration is recorded, every callback
// registered with [WhenFakingSleep] is run with it, and the pinned clock moves
// forward by it when [SyncWithCarbon] asked for that.
func (s *Sleep) Goodnight() error {
	if s.alreadySlept || !s.shouldSleep {
		return s.err
	}
	if s.err != nil {
		return s.err
	}
	if s.pending != nil {
		return ErrUnknownDurationUnit
	}
	s.alreadySlept = true

	sleepMu.Lock()
	faking := sleepFake
	syncWithCarbon := sleepSyncWithCarbon
	if faking {
		sleepSequence = append(sleepSequence, s.Duration)
	}
	callbacks := make([]func(time.Duration), len(sleepFakeCallbacks))
	copy(callbacks, sleepFakeCallbacks)
	sleepMu.Unlock()

	if faking {
		if syncWithCarbon {
			moved := Now().Add(s.Duration)
			SetTestNow(&moved)
		}
		for _, callback := range callbacks {
			callback(s.Duration)
		}
		return nil
	}

	remaining := s.Duration
	seconds := remaining / time.Second * time.Second

	while := s.while
	if while == nil {
		calls := 0
		while = func() bool {
			calls++
			return calls == 1
		}
	}

	for while() {
		if seconds > 0 {
			time.Sleep(seconds)
			remaining -= seconds
		}
		if remaining > 0 {
			time.Sleep(remaining)
		}
	}
	return nil
}

// Fake makes every later sleep record its duration instead of waiting it out,
// and clears whatever was recorded before.
//
// The variadic argument is whether to fake and whether to move the pinned
// clock forward by each sleep, in that order, defaulting to true and false.
// The setting is process-wide, so a test that calls it cannot run in parallel
// with one that does not.
func Fake(options ...bool) {
	sleepMu.Lock()
	defer sleepMu.Unlock()
	sleepFake = true
	if len(options) > 0 {
		sleepFake = options[0]
	}
	sleepSyncWithCarbon = len(options) > 1 && options[1]
	sleepSequence = nil
	sleepFakeCallbacks = nil
}

// SyncWithCarbon says whether a faked sleep moves the pinned "now" forward by
// its duration. The variadic argument defaults to true.
func SyncWithCarbon(value ...bool) {
	sleepMu.Lock()
	defer sleepMu.Unlock()
	sleepSyncWithCarbon = firstOr(value, true)
}

// WhenFakingSleep registers a callback run with every duration that was
// recorded instead of slept.
func WhenFakingSleep(callback func(duration time.Duration)) {
	sleepMu.Lock()
	defer sleepMu.Unlock()
	sleepFakeCallbacks = append(sleepFakeCallbacks, callback)
}

// sleepRecorded returns a copy of the recorded durations, taken under the
// lock.
func sleepRecorded() []time.Duration {
	sleepMu.Lock()
	defer sleepMu.Unlock()
	out := make([]time.Duration, len(sleepSequence))
	copy(out, sleepSequence)
	return out
}

// AssertSlept fails the test unless exactly the expected number of recorded
// durations pass the truth test. The variadic argument is that number and
// defaults to 1.
//
// The testing.TB is the first argument because there is no ambient running
// test to find: the caller hands its own in.
func AssertSlept(t testing.TB, expected func(duration time.Duration) bool, times ...int) {
	t.Helper()
	want := firstOr(times, 1)
	count := 0
	for _, duration := range sleepRecorded() {
		if expected(duration) {
			count++
		}
	}
	if count != want {
		t.Errorf("The expected sleep was found [%d] times instead of [%d].", count, want)
	}
}

// AssertSleptTimes fails the test unless exactly that many sleeps were
// recorded.
func AssertSleptTimes(t testing.TB, expected int) {
	t.Helper()
	count := len(sleepRecorded())
	if count != expected {
		t.Errorf("Expected [%d] sleeps but found [%d].", expected, count)
	}
}

// AssertSequence fails the test unless the recorded sleeps match the given
// ones, in order and in number. A nil entry skips the comparison at that
// position.
func AssertSequence(t testing.TB, sequence []*Sleep) {
	t.Helper()
	AssertSleptTimes(t, len(sequence))
	recorded := sleepRecorded()
	for i, expected := range sequence {
		if expected == nil || i >= len(recorded) {
			continue
		}
		expected.shouldNotSleep()
		if expected.Duration != recorded[i] {
			t.Errorf("Expected sleep duration of [%s] but actually slept for [%s].",
				expected.Duration, recorded[i])
		}
	}
}

// AssertNeverSlept fails the test unless no sleep was recorded at all.
func AssertNeverSlept(t testing.TB) {
	t.Helper()
	AssertSleptTimes(t, 0)
}

// AssertInsomniac fails the test unless every recorded sleep was of zero
// duration: sleeping was asked for, but nothing would have been waited.
func AssertInsomniac(t testing.TB) {
	t.Helper()
	for _, duration := range sleepRecorded() {
		if duration != 0 {
			t.Errorf("Unexpected sleep duration of [%s] found.", duration)
		}
	}
}
