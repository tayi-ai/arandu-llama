package support

import "time"

// Timebox runs a callback and does not return before the given number of
// microseconds has passed, so that the time a check took says nothing about
// its result.
type Timebox struct {
	// EarlyReturn lifts the wait: the box returns as soon as the callback
	// does.
	EarlyReturn bool
}

// NewTimebox returns a timebox that waits out its full duration.
func NewTimebox() *Timebox {
	return &Timebox{}
}

// Call runs the callback inside the timebox and returns what it returned. An
// error from the callback is held until the box has been waited out, so a
// failure does not come back any sooner than a success.
func (t *Timebox) Call(callback func(*Timebox) (any, error), microseconds int) (any, error) {
	start := time.Now()

	result, err := callback(t)

	remainder := time.Duration(microseconds)*time.Microsecond - time.Since(start)

	if !t.EarlyReturn && remainder > 0 {
		t.usleep(remainder)
	}

	if err != nil {
		return nil, err
	}
	return result, nil
}

// ReturnEarly lifts the wait, and returns the timebox.
func (t *Timebox) ReturnEarly() *Timebox {
	t.EarlyReturn = true
	return t
}

// DontReturnEarly restores the wait, and returns the timebox.
func (t *Timebox) DontReturnEarly() *Timebox {
	t.EarlyReturn = false
	return t
}

// usleep waits through [Usleep] rather than through the operating system
// directly, so a test that called [Fake] captures the wait instead of serving
// it.
func (t *Timebox) usleep(remainder time.Duration) {
	_ = Usleep(int(remainder / time.Microsecond)).Goodnight()
}
