package number

import (
	"math"
	"strconv"
	"strings"
)

// smallWords are the English cardinals below twenty, which no rule produces.
var smallWords = []string{
	"zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
	"nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
	"sixteen", "seventeen", "eighteen", "nineteen",
}

// tensWords are the English cardinals for the multiples of ten, indexed by the
// tens digit, so tensWords[2] is "twenty".
var tensWords = []string{"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"}

// scaleWords are the short-scale group names, indexed by the group: one group
// of three digits in, "thousand".
var scaleWords = []string{
	"", " thousand", " million", " billion", " trillion", " quadrillion",
	" quintillion",
}

// ordinalWords is every cardinal word whose ordinal is not simply the word with
// -th on it. What is not here takes -th, so "seventeen" becomes "seventeenth".
var ordinalWords = map[string]string{
	"zero": "zeroth", "one": "first", "two": "second", "three": "third",
	"four": "fourth", "five": "fifth", "six": "sixth", "seven": "seventh",
	"eight": "eighth", "nine": "ninth", "ten": "tenth", "eleven": "eleventh",
	"twelve": "twelfth", "twenty": "twentieth", "thirty": "thirtieth",
	"forty": "fortieth", "fifty": "fiftieth", "sixty": "sixtieth",
	"seventy": "seventieth", "eighty": "eightieth", "ninety": "ninetieth",
}

// Spell writes v out in words.
//
//	Spell(10)     // "ten"
//	Spell(-2)     // "minus two"
//	Spell(1234)   // "one thousand two hundred thirty-four"
//	Spell(1.25)   // "one point two five"
//
// The optional arguments bound the range that is spelled, in that order: a
// value at or below the first, or at or above the second, is written in digits
// by Format instead of in words. Passing neither spells every value.
//
//	Spell(9, 10)     // "9"  -- at or below the lower bound
//	Spell(11, 0, 10) // "11" -- at or above the upper bound
//
// The words are English; see the package comment.
func Spell(v float64, bounds ...int) string {
	if s, ok := special(v); ok {
		return s
	}
	if len(bounds) > 0 && v <= float64(bounds[0]) {
		return Format(v)
	}
	if len(bounds) > 1 && v >= float64(bounds[1]) {
		return Format(v)
	}

	sign := ""
	if math.Signbit(v) {
		sign, v = "minus ", math.Abs(v)
	}

	whole, fraction := splitDecimal(v)
	spelled := sign + spellInteger(whole)
	if fraction == "" {
		return spelled
	}
	var b strings.Builder
	b.WriteString(spelled)
	b.WriteString(" point")
	for _, digit := range fraction {
		b.WriteByte(' ')
		b.WriteString(smallWords[digit-'0'])
	}
	return b.String()
}

// SpellOrdinal writes v out in words in ordinal form.
//
//	SpellOrdinal(1)   // "first"
//	SpellOrdinal(21)  // "twenty-first"
//	SpellOrdinal(100) // "one hundredth"
//
// Only the last word changes. A fraction has no ordinal, so it is dropped. The
// words are English; see the package comment.
func SpellOrdinal(v float64) string {
	if s, ok := special(v); ok {
		return s
	}
	sign := ""
	if math.Signbit(v) {
		sign, v = "minus ", math.Abs(v)
	}
	return sign + ordinalize(spellInteger(uint64(math.Trunc(v))))
}

// ordinalize turns the last word of a spelled cardinal into its ordinal, which
// is the whole of the difference between the two forms.
func ordinalize(spelled string) string {
	i := strings.LastIndexAny(spelled, " -")
	head, last := "", spelled
	if i >= 0 {
		head, last = spelled[:i+1], spelled[i+1:]
	}
	if o, ok := ordinalWords[last]; ok {
		return head + o
	}
	return head + strings.TrimSuffix(last, "y") + "th"
}

// spellInteger writes a whole number out in words, in groups of three digits.
func spellInteger(n uint64) string {
	if n < 20 {
		return smallWords[n]
	}

	// The groups come out least significant first, so they are written into a
	// slice and read back the other way round.
	var groups []string
	for scale := 0; n > 0; scale++ {
		if g := n % 1000; g > 0 {
			groups = append(groups, spellGroup(uint(g))+scaleAt(scale))
		}
		n /= 1000
	}
	for i, j := 0, len(groups)-1; i < j; i, j = i+1, j-1 {
		groups[i], groups[j] = groups[j], groups[i]
	}
	return strings.Join(groups, " ")
}

// scaleAt is the group name, or the empty string once the number runs past the
// table -- which a uint64 cannot do, because it stops inside the quintillions.
func scaleAt(scale int) string {
	if scale < len(scaleWords) {
		return scaleWords[scale]
	}
	return ""
}

// spellGroup writes one group of three digits: "one hundred twenty-three".
func spellGroup(n uint) string {
	var parts []string
	if n >= 100 {
		parts = append(parts, smallWords[n/100]+" hundred")
		n %= 100
	}
	switch {
	case n == 0:
	case n < 20:
		parts = append(parts, smallWords[n])
	case n%10 == 0:
		parts = append(parts, tensWords[n/10])
	default:
		parts = append(parts, tensWords[n/10]+"-"+smallWords[n%10])
	}
	return strings.Join(parts, " ")
}

// splitDecimal cuts a non-negative value into its whole part and the digits
// after the point, with the trailing zeros gone. The shortest representation is
// what strconv writes, so 1.25 gives "25" and 1.0 gives "".
func splitDecimal(v float64) (uint64, string) {
	s := strconv.FormatFloat(v, 'f', -1, 64)
	whole, fraction, _ := strings.Cut(s, ".")
	n, err := strconv.ParseUint(whole, 10, 64)
	if err != nil {
		// Past what a uint64 holds there is no group table left to name the
		// value with, so the digits stand for themselves.
		return 0, ""
	}
	return n, strings.TrimRight(fraction, "0")
}
