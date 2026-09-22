package number

import (
	"cmp"
	"math"
	"strconv"
	"strings"
)

const (
	groupSeparator   = ','
	decimalSeparator = '.'

	// icuDefaultFractionDigits is what Format keeps when neither the precision
	// nor the maximum precision is given: at most three digits after the point,
	// with the trailing zeros dropped.
	icuDefaultFractionDigits = 3
)

// Format renders v with a group separator every three integer digits.
//
// The precision and the maximum precision are the two optional arguments, in
// that order, because Go has no named arguments:
//
//	Format(1234.5678)    // "1,234.568" -- neither given
//	Format(1234.5678, 2) // "1,234.57"  -- an exact digit count
//	Format(1234.5, 2, 4) // "1,234.5"   -- an upper bound
//
// A maximum precision drops the trailing zeros a precision would keep, which
// is the whole difference between them. A precision below zero is read as
// zero.
//
// The rendering is not locale-aware; see the package comment.
func Format(v float64, precision ...int) string {
	if s, ok := special(v); ok {
		return s
	}
	switch len(precision) {
	case 0:
		return group(trimTrailingZeros(strconv.FormatFloat(v, 'f', icuDefaultFractionDigits, 64)))
	case 1:
		return group(strconv.FormatFloat(v, 'f', max(precision[0], 0), 64))
	default:
		return group(trimTrailingZeros(strconv.FormatFloat(v, 'f', max(precision[1], 0), 64)))
	}
}

// Percentage renders v as a percentage.
//
// The value is taken as already being a percentage, so Percentage(12.5, 1) is
// "12.5%" and not "1250.0%".
//
// The optional arguments are the precision and the maximum precision, as on
// Format, except that the precision defaults to zero rather than to the
// default Format itself keeps.
func Percentage(v float64, precision ...int) string {
	if s, ok := special(v); ok {
		return s
	}
	if len(precision) == 0 {
		precision = []int{0}
	}
	return Format(v, precision...) + "%"
}

// Ordinal renders n in English ordinal form, with the same group separator
// Format uses.
//
//	Ordinal(1)    // "1st"
//	Ordinal(12)   // "12th"
//	Ordinal(1000) // "1,000th"
//
// The argument is an int because a fraction has no ordinal.
func Ordinal(n int) string {
	tens := n % 100
	if tens < 0 {
		tens = -tens
	}
	suffix := "th"
	if tens < 11 || tens > 13 {
		switch tens % 10 {
		case 1:
			suffix = "st"
		case 2:
			suffix = "nd"
		case 3:
			suffix = "rd"
		}
	}
	return group(strconv.Itoa(n)) + suffix
}

// abbreviateUnits is the abbreviated unit table, keyed by the power of ten
// each suffix stands for.
var abbreviateUnits = []unit{{3, "K"}, {6, "M"}, {9, "B"}, {12, "T"}, {15, "Q"}}

// humanUnits is the spelled-out table ForHumans uses when it is not
// abbreviating. Each suffix carries a leading space, and the trim at the end of
// summarize is what keeps it from showing on a value with no unit.
var humanUnits = []unit{
	{3, " thousand"}, {6, " million"}, {9, " billion"},
	{12, " trillion"}, {15, " quadrillion"},
}

type unit struct {
	exponent int
	suffix   string
}

// Abbreviate is ForHumans with the short suffixes K, M, B, T and Q.
//
// The optional arguments are the precision and the maximum precision.
//
//	Abbreviate(1500, 1)  // "1.5K"
//	Abbreviate(-2000000) // "-2M"
func Abbreviate(v float64, precision ...int) string {
	return ForHumans(v, argAt(precision, 0, 0), argAt(precision, 1, -1), true)
}

// ForHumans renders v against the largest thousand-scaled unit it reaches,
// spelled out unless abbreviate is set.
//
//	ForHumans(1500, 0, -1, false) // "2 thousand"
//	ForHumans(1500, 1, -1, true)  // "1.5K"
//
// All three arguments are required, because a boolean follows them, and a
// negative maxPrecision stands for the one that was not given.
//
// A value past a quadrillion is scaled by a quadrillion and rendered again,
// which is why ForHumans(1e18, 0, -1, true) is "1KQ" rather than "1,000Q".
func ForHumans(v float64, precision, maxPrecision int, abbreviate bool) string {
	units := humanUnits
	if abbreviate {
		units = abbreviateUnits
	}
	return summarize(v, precision, maxPrecision, units)
}

// summarize is the whole of ForHumans and Abbreviate: v rendered against the
// largest unit in the table it reaches.
func summarize(v float64, precision, maxPrecision int, units []unit) string {
	if s, ok := special(v); ok {
		return s
	}
	switch {
	case v == 0:
		if precision > 0 {
			return formatWith(0, precision, maxPrecision)
		}
		return "0"
	case v < 0:
		return "-" + summarize(math.Abs(v), precision, maxPrecision, units)
	case v >= 1e15:
		return summarize(v/1e15, precision, maxPrecision, units) + units[len(units)-1].suffix
	}

	numberExponent := int(math.Floor(math.Log10(v)))
	displayExponent := numberExponent - numberExponent%3
	v /= math.Pow(10, float64(displayExponent))

	return strings.TrimSpace(formatWith(v, precision, maxPrecision) + suffixFor(units, displayExponent))
}

// suffixFor looks a unit up by the power of ten it stands for: an exponent
// with no unit against it contributes nothing.
func suffixFor(units []unit, exponent int) string {
	for _, u := range units {
		if u.exponent == exponent {
			return u.suffix
		}
	}
	return ""
}

// Clamp is v held between minimum and maximum. It is generic over every
// ordered type, so nothing has to be coerced to reach it.
func Clamp[T cmp.Ordered](v, minimum, maximum T) T {
	return min(max(v, minimum), maximum)
}

// Pairs splits the range up to `to` into pairs of lower and upper bounds, `by`
// wide:
//
//	Pairs(10, 5)    // [[0 4] [5 9]]
//	Pairs(10, 5, 0) // the same, with the start spelled out
//
// The optional arguments are the start and the offset, which default to 0 and
// 1. A step of zero or less returns nothing, because it would otherwise be a
// loop that never advances, and no pair is a better answer than no answer at
// all.
func Pairs(to, by float64, startOffset ...float64) [][2]float64 {
	if by <= 0 {
		return nil
	}
	start := argAt(startOffset, 0, 0.0)
	offset := argAt(startOffset, 1, 1.0)

	var out [][2]float64
	for lower := start; lower < to; lower += by {
		upper := lower + by - offset
		if upper > to {
			upper = to
		}
		out = append(out, [2]float64{lower, upper})
	}
	return out
}

// Trim removes the trailing zero digits after the decimal point: Trim(12.30)
// is "12.3" and Trim(12.0) is "12".
//
// The change is to the way the number is written and not to the number itself,
// so what comes back is the writing. There is no group separator in it.
func Trim(v float64) string {
	if s, ok := special(v); ok {
		return s
	}
	return strconv.FormatFloat(v, 'f', -1, 64)
}

// formatWith calls Format, with a negative maxPrecision standing for the
// argument that was not given.
func formatWith(v float64, precision, maxPrecision int) string {
	if maxPrecision >= 0 {
		return Format(v, precision, maxPrecision)
	}
	return Format(v, precision)
}

// argAt reads an optional argument out of a variadic tail, or answers with the
// given fallback.
func argAt[T any](args []T, i int, fallback T) T {
	if i < len(args) {
		return args[i]
	}
	return fallback
}

// special reports the rendering of the float64 values that have no digits.
func special(v float64) (string, bool) {
	switch {
	case math.IsNaN(v):
		return "NaN", true
	case math.IsInf(v, 1):
		return "Inf", true
	case math.IsInf(v, -1):
		return "-Inf", true
	}
	return "", false
}

// trimTrailingZeros drops the zeros a fixed-digit rendering left behind, and
// the point with them when nothing survives it. It is what turns an exact digit
// count into an upper bound.
func trimTrailingZeros(s string) string {
	if !strings.ContainsRune(s, decimalSeparator) {
		return s
	}
	s = strings.TrimRight(s, "0")
	return strings.TrimSuffix(s, string(decimalSeparator))
}

// group inserts the group separator into the integer part of a decimal string
// as strconv writes it, and drops a sign that only survived rounding: a value
// of -0.004 at two digits is "0.00", never "-0.00".
func group(s string) string {
	negative := strings.HasPrefix(s, "-")
	if negative {
		s = s[1:]
	}
	integer, fraction := s, ""
	if i := strings.IndexByte(s, decimalSeparator); i >= 0 {
		integer, fraction = s[:i], s[i:]
	}
	if negative && isZero(integer) && isZero(fraction) {
		negative = false
	}

	grouped := groupDigits(integer, string(groupSeparator)) + fraction
	if negative {
		return "-" + grouped
	}
	return grouped
}

// groupDigits writes sep between each three integer digits, counting from the
// right. It takes the separator rather than reading groupSeparator so that it is
// the one place in the package that inserts one, which is what lets a locale
// change the separator without a second way of counting the groups.
//
// The digits arrive with no sign and no point in them.
func groupDigits(integer, sep string) string {
	if integer == "" {
		return ""
	}
	var b strings.Builder
	b.Grow(len(integer) + len(sep)*((len(integer)-1)/3))
	lead := (len(integer)-1)%3 + 1
	b.WriteString(integer[:lead])
	for i := lead; i < len(integer); i += 3 {
		b.WriteString(sep)
		b.WriteString(integer[i : i+3])
	}
	return b.String()
}

// isZero reports whether s carries no digit other than zero.
func isZero(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] >= '1' && s[i] <= '9' {
			return false
		}
	}
	return true
}
