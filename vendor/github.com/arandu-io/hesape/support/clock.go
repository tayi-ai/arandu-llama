package support

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Clock is the one place "now" comes from, so a test can replace it.
//
// There is no date type of this package's own: time.Time already is the value,
// and a second one would be a second way to hold an instant. The seam is this
// interface, and a test reaches it through [Use], [UseCallable], [SetTestNow],
// [Travel], [TravelTo], [TravelBack] and [FreezeTime].
type Clock interface {
	// Now returns the current instant.
	Now() time.Time
}

// SystemClock is the [Clock] the framework runs on outside a test: time.Now.
type SystemClock struct{}

// Now returns the current instant, read from the operating system.
func (SystemClock) Now() time.Time { return time.Now() }

var (
	clockMu sync.RWMutex
	clock   Clock = SystemClock{}
	testNow *time.Time
)

// Now returns the current instant, or the instant a test pinned with
// [SetTestNow]. It reads the [Clock] that [Use] set.
func Now() time.Time {
	clockMu.RLock()
	defer clockMu.RUnlock()
	if testNow != nil {
		return *testNow
	}
	return clock.Now()
}

// Today returns midnight of the current day, in the current location.
func Today() time.Time {
	n := Now()
	return time.Date(n.Year(), n.Month(), n.Day(), 0, 0, 0, 0, n.Location())
}

// Tomorrow returns midnight of the day after [Today].
func Tomorrow() time.Time { return Today().AddDate(0, 0, 1) }

// Yesterday returns midnight of the day before [Today].
func Yesterday() time.Time { return Today().AddDate(0, 0, -1) }

// parseLayouts are the layouts [Parse] tries, in order.
var parseLayouts = []string{
	time.RFC3339Nano,
	time.RFC3339,
	"2006-01-02T15:04:05",
	"2006-01-02 15:04:05.999999999",
	"2006-01-02 15:04:05",
	"2006-01-02 15:04",
	"2006-01-02",
	"2006/01/02",
	"15:04:05",
	"15:04",
	time.RFC1123Z,
	time.RFC1123,
}

// Parse reads a date out of a string, returning an error naming the string
// when no layout matches.
//
// The empty string and "now" are [Now]; "today", "tomorrow" and "yesterday"
// are the calls of those names. A time-only string is placed on the day [Now]
// reports. Everything else is read in the local location.
func Parse(value string) (time.Time, error) {
	value = strings.TrimSpace(value)
	if value == "" {
		return Now(), nil
	}
	if strings.EqualFold(value, "now") {
		return Now(), nil
	}
	if strings.EqualFold(value, "today") {
		return Today(), nil
	}
	if strings.EqualFold(value, "tomorrow") {
		return Tomorrow(), nil
	}
	if strings.EqualFold(value, "yesterday") {
		return Yesterday(), nil
	}
	for _, layout := range parseLayouts {
		parsed, err := time.ParseInLocation(layout, value, time.Local)
		if err != nil {
			continue
		}
		if parsed.Year() == 0 {
			n := Now()
			return time.Date(n.Year(), n.Month(), n.Day(),
				parsed.Hour(), parsed.Minute(), parsed.Second(), parsed.Nanosecond(), n.Location()), nil
		}
		return parsed, nil
	}
	return time.Time{}, fmt.Errorf("support: could not parse date [%s]", value)
}

// CreateFromFormat reads a date out of a string against the given layout, in
// the local location.
func CreateFromFormat(layout, value string) (time.Time, error) {
	return time.ParseInLocation(layout, value, time.Local)
}

// CreateFromTimestamp returns the instant the given number of seconds after
// the epoch.
func CreateFromTimestamp(timestamp int64) time.Time {
	return time.Unix(timestamp, 0)
}

// CreateFromTimestampMs returns the instant the given number of milliseconds
// after the epoch.
func CreateFromTimestampMs(timestamp int64) time.Time {
	return time.UnixMilli(timestamp)
}

// ErrNotAnOrderedID is returned by [CreateFromId] when the string is neither
// an ordered UUID nor a ULID.
var ErrNotAnOrderedID = errors.New("support: id is neither an ordered UUID nor a ULID")

// crockford is the alphabet a ULID is written in, which leaves out I, L, O and
// U so that no two characters can be misread for each other.
const crockford = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

// CreateFromId returns the instant an ordered UUID or a ULID was made at,
// which both carry in their first 48 bits. The milliseconds are read here
// rather than through a library, so this package carries no dependency of its
// own. A string that is neither is [ErrNotAnOrderedID].
func CreateFromId(id string) (time.Time, error) {
	if milliseconds, ok := ulidMilliseconds(id); ok {
		return time.UnixMilli(milliseconds), nil
	}
	if milliseconds, ok := orderedUUIDMilliseconds(id); ok {
		return time.UnixMilli(milliseconds), nil
	}
	return time.Time{}, ErrNotAnOrderedID
}

// ulidMilliseconds reads the timestamp out of a ULID: 26 Crockford
// characters, of which the first 10 carry the milliseconds.
func ulidMilliseconds(id string) (int64, bool) {
	if len(id) != 26 {
		return 0, false
	}
	milliseconds := int64(0)
	for i, r := range strings.ToUpper(id) {
		index := strings.IndexRune(crockford, r)
		if index < 0 {
			return 0, false
		}
		if i < 10 {
			milliseconds = milliseconds<<5 | int64(index)
		}
	}
	return milliseconds, true
}

// orderedUUIDMilliseconds reads the timestamp out of a time-ordered UUID: the
// first 48 bits carry the milliseconds, which is what version 7 puts there.
func orderedUUIDMilliseconds(id string) (int64, bool) {
	hex := strings.ReplaceAll(id, "-", "")
	if len(hex) != 32 {
		return 0, false
	}
	milliseconds, err := strconv.ParseInt(hex[:12], 16, 64)
	if err != nil {
		return 0, false
	}
	for _, r := range hex {
		if !strings.ContainsRune("0123456789abcdefABCDEF", r) {
			return 0, false
		}
	}
	return milliseconds, true
}

// SetTestNow pins the instant every later [Now] reports. A nil value unpins,
// handing time back to the [Clock].
func SetTestNow(value *time.Time) {
	clockMu.Lock()
	defer clockMu.Unlock()
	if value == nil {
		testNow = nil
		return
	}
	pinned := *value
	testNow = &pinned
}

// GetTestNow returns a copy of the pinned instant, or nil when nothing is
// pinned.
func GetTestNow() *time.Time {
	clockMu.RLock()
	defer clockMu.RUnlock()
	if testNow == nil {
		return nil
	}
	pinned := *testNow
	return &pinned
}

// HasTestNow reports whether an instant is pinned.
func HasTestNow() bool { return GetTestNow() != nil }

// WithTestNow runs the callback with the instant pinned, then puts back
// whatever was pinned before.
func WithTestNow(value time.Time, callback func()) {
	previous := GetTestNow()
	SetTestNow(&value)
	defer SetTestNow(previous)
	callback()
}

// Use sets the [Clock] every later [Now] reads. A nil clock restores
// [SystemClock].
func Use(c Clock) {
	clockMu.Lock()
	defer clockMu.Unlock()
	if c == nil {
		c = SystemClock{}
	}
	clock = c
}

// clockFunc lets a plain function be a [Clock].
type clockFunc func() time.Time

// Now returns the instant the function reports.
func (f clockFunc) Now() time.Time { return f() }

// UseCallable sets the function every later [Now] reads. A nil callable is
// [UseDefault].
func UseCallable(callable func() time.Time) {
	if callable == nil {
		UseDefault()
		return
	}
	Use(clockFunc(callable))
}

// UseDefault restores [SystemClock] and unpins whatever was pinned.
func UseDefault() {
	clockMu.Lock()
	defer clockMu.Unlock()
	clock = SystemClock{}
	testNow = nil
}

// Travel moves the pinned "now" by the given amount and returns it. A
// time.Duration already carries its unit, so a five-day jump is
// Travel(5 * 24 * time.Hour).
func Travel(d time.Duration) time.Time {
	return TravelTo(Now().Add(d))
}

// TravelTo pins "now" at the given instant and returns it. Given a callback,
// it runs the callback and then travels back.
func TravelTo(value time.Time, callback ...func()) time.Time {
	SetTestNow(&value)
	if len(callback) > 0 && callback[0] != nil {
		defer TravelBack()
		callback[0]()
	}
	return value
}

// TravelBack unpins "now".
func TravelBack() { SetTestNow(nil) }

// FreezeTime pins "now" where it already is, so nothing moves while the
// callback runs. With no callback it stays frozen until [TravelBack].
func FreezeTime(callback ...func()) time.Time {
	return TravelTo(Now(), callback...)
}

// FreezeSecond freezes on the start of the current second, so a comparison
// against a whole second holds.
func FreezeSecond(callback ...func()) time.Time {
	return TravelTo(Now().Truncate(time.Second), callback...)
}

// ErrUnknownDelay states the three shapes a delay may take: an instant
// (time.Time), a span (time.Duration) or a plain count of seconds. Anything
// else carries no duration and cannot be turned into one.
//
// [SecondsUntil], [AvailableAt] and [ParseDateInterval] never return it -- they
// take a value they cannot read as zero, so a delay of the wrong type schedules
// work for right now instead of failing. It is here for a caller that has to
// reject such a value rather than fall back, and it is the error to return.
var ErrUnknownDelay = errors.New("support: delay must be a time.Time, a time.Duration or a number of seconds")

// SecondsUntil returns how many seconds separate now from the delay. The delay
// is a time.Time, a time.Duration or a number of seconds; anything else reads
// as zero. An instant already past reads as zero rather than a negative count.
func SecondsUntil(delay any) int64 {
	switch d := delay.(type) {
	case time.Time:
		return max64(0, d.Unix()-CurrentTime())
	case *time.Time:
		if d == nil {
			return 0
		}
		return max64(0, d.Unix()-CurrentTime())
	case time.Duration:
		return int64(d / time.Second)
	default:
		return toSeconds(delay)
	}
}

// AvailableAt returns the UNIX timestamp the delay lands on. The delay is a
// time.Time, a time.Duration or a number of seconds; anything else lands on
// now.
func AvailableAt(delay any) int64 {
	switch d := delay.(type) {
	case time.Time:
		return d.Unix()
	case *time.Time:
		if d == nil {
			return CurrentTime()
		}
		return d.Unix()
	case time.Duration:
		return Now().Add(d).Unix()
	default:
		return Now().Add(time.Duration(toSeconds(delay)) * time.Second).Unix()
	}
}

// ParseDateInterval turns a time.Duration into the instant it lands on, and
// leaves every other value alone.
func ParseDateInterval(delay any) any {
	if d, ok := delay.(time.Duration); ok {
		return Now().Add(d)
	}
	return delay
}

// CurrentTime returns [Now] as a UNIX timestamp.
func CurrentTime() int64 { return Now().Unix() }

// RunTimeForHumans writes the span between the two instants the way a console
// line writes it. At one second or below it is milliseconds with two decimals;
// above it, a short cascade of units, as in "1s 250ms". An absent end is
// [Now], and an end before the start reads as zero.
func RunTimeForHumans(start time.Time, end ...time.Time) string {
	finish := Now()
	if len(end) > 0 && !end[0].IsZero() {
		finish = end[0]
	}
	runTime := finish.Sub(start)
	if runTime < 0 {
		runTime = 0
	}
	if runTime <= time.Second {
		return strconv.FormatFloat(float64(runTime)/float64(time.Millisecond), 'f', 2, 64) + "ms"
	}
	return forHumans(runTime)
}

// forHumans writes a duration as a short cascade of units, down to
// milliseconds, dropping the units it does not reach. Zero is "0ms".
func forHumans(d time.Duration) string {
	parts := []string{}
	units := []struct {
		suffix string
		size   time.Duration
	}{
		{"w", 7 * 24 * time.Hour},
		{"d", 24 * time.Hour},
		{"h", time.Hour},
		{"m", time.Minute},
		{"s", time.Second},
		{"ms", time.Millisecond},
	}
	for _, unit := range units {
		if d < unit.size {
			continue
		}
		count := d / unit.size
		d -= count * unit.size
		parts = append(parts, strconv.FormatInt(int64(count), 10)+unit.suffix)
	}
	if len(parts) == 0 {
		return "0ms"
	}
	return strings.Join(parts, " ")
}

// toSeconds reads any numeric shape, or a numeric string, as a count of
// seconds. Anything else is zero.
func toSeconds(delay any) int64 {
	switch n := delay.(type) {
	case nil:
		return 0
	case int:
		return int64(n)
	case int32:
		return int64(n)
	case int64:
		return n
	case float32:
		return int64(n)
	case float64:
		return int64(n)
	case string:
		parsed, err := strconv.ParseFloat(n, 64)
		if err != nil {
			return 0
		}
		return int64(parsed)
	default:
		return 0
	}
}

func max64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}
