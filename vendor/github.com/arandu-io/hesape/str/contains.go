package str

import (
	"regexp"
	"strings"
)

// Contains reports whether the haystack holds any one of the needles, comparing
// without regard to case when ignoreCase is set.
//
// An empty needle never matches: Contains("anything", []string{""}, false) is
// false, not true.
//
// The needles are a slice rather than a variadic tail because the case flag
// follows them.
func Contains(haystack string, needles []string, ignoreCase bool) bool {
	if ignoreCase {
		haystack = strings.ToLower(haystack)
	}
	for _, needle := range needles {
		if ignoreCase {
			needle = strings.ToLower(needle)
		}
		if needle != "" && strings.Contains(haystack, needle) {
			return true
		}
	}
	return false
}

// ContainsAll reports whether the haystack holds every one of the needles.
//
// An empty list of needles is true, because there is nothing that fails. A
// needle that is the empty string is false, because Contains refuses it.
func ContainsAll(haystack string, needles []string, ignoreCase bool) bool {
	for _, needle := range needles {
		if !Contains(haystack, []string{needle}, ignoreCase) {
			return false
		}
	}
	return true
}

// DoesntContain is Contains negated.
func DoesntContain(haystack string, needles []string, ignoreCase bool) bool {
	return !Contains(haystack, needles, ignoreCase)
}

// StartsWith reports whether the haystack begins with any one of the needles.
// An empty needle never matches.
func StartsWith(haystack string, needles ...string) bool {
	for _, needle := range needles {
		if needle != "" && strings.HasPrefix(haystack, needle) {
			return true
		}
	}
	return false
}

// DoesntStartWith is StartsWith negated.
func DoesntStartWith(haystack string, needles ...string) bool {
	return !StartsWith(haystack, needles...)
}

// EndsWith reports whether the haystack ends with any one of the needles. An
// empty needle never matches.
func EndsWith(haystack string, needles ...string) bool {
	for _, needle := range needles {
		if needle != "" && strings.HasSuffix(haystack, needle) {
			return true
		}
	}
	return false
}

// DoesntEndWith is EndsWith negated.
func DoesntEndWith(haystack string, needles ...string) bool {
	return !EndsWith(haystack, needles...)
}

// Wrap puts before in front of the value and after behind it; passing no after
// repeats before, so Wrap("name", `"`) is `"name"`.
func Wrap(value, before string, after ...string) string {
	return before + value + wrapAfter(before, after)
}

// Unwrap removes before from the front of the value and after from the back,
// each only if it is there; passing no after repeats before.
func Unwrap(value, before string, after ...string) string {
	tail := wrapAfter(before, after)
	if StartsWith(value, before) {
		value = Substr(value, Length(before))
	}
	if EndsWith(value, tail) {
		value = Substr(value, 0, -Length(tail))
	}
	return value
}

// wrapAfter is the closing wrapper, which falls back to the opening one.
func wrapAfter(before string, after []string) string {
	if len(after) > 0 {
		return after[0]
	}
	return before
}

// Deduplicate collapses a run of the given character to one occurrence of it;
// passing no character deduplicates spaces.
//
// The pattern is the quoted character followed by a plus, so a multi-character
// argument repeats only its last character: Deduplicate("a---b", "--") is
// "a--b". A character that is the empty string returns the value untouched.
func Deduplicate(value string, character ...string) string {
	char := " "
	if len(character) > 0 {
		char = character[0]
	}
	if char == "" {
		return value
	}
	re, err := regexp.Compile(regexp.QuoteMeta(char) + "+")
	if err != nil {
		return value
	}
	return re.ReplaceAllLiteralString(value, char)
}
