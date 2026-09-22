package str

import (
	"fmt"
	"strconv"
	"strings"
	"time"
	"unicode"
)

// phpDirname is the path with its last component gone: "." for a bare name,
// "/" for the root, and "" for the empty string.
func phpDirname(path string) string {
	if path == "" {
		return ""
	}
	trimmed := strings.TrimRight(path, "/")
	if trimmed == "" {
		return "/"
	}
	i := strings.LastIndexByte(trimmed, '/')
	switch {
	case i < 0:
		return "."
	case i == 0:
		return "/"
	default:
		return trimmed[:i]
	}
}

// chunkRunes cuts the string into pieces of the given number of characters,
// with the last one holding what is left. A size below one has no answer, so
// nothing comes back.
func chunkRunes(s string, size int) []string {
	if size < 1 {
		return nil
	}
	rs := []rune(s)
	if len(rs) == 0 {
		return nil
	}
	out := make([]string, 0, (len(rs)+size-1)/size)
	for i := 0; i < len(rs); i += size {
		out = append(out, string(rs[i:min(i+size, len(rs))]))
	}
	return out
}

// sscanf reads the fields a format names off the front of the subject and
// returns them as strings. The verbs it knows are listed on Stringable.Scan.
func sscanf(subject, format string) []string {
	var out []string
	s, f := []rune(subject), []rune(format)
	i := 0

	for j := 0; j < len(f); j++ {
		switch {
		case unicode.IsSpace(f[j]):
			for i < len(s) && unicode.IsSpace(s[i]) {
				i++
			}
		case f[j] != '%':
			if i >= len(s) || s[i] != f[j] {
				return out
			}
			i++
		default:
			j++
			if j >= len(f) {
				return out
			}
			if f[j] == '%' {
				if i >= len(s) || s[i] != '%' {
					return out
				}
				i++
				continue
			}
			for i < len(s) && unicode.IsSpace(s[i]) && f[j] != 'c' {
				i++
			}
			field, width := scanField(s[i:], f[j])
			if width == 0 {
				return out
			}
			out = append(out, field)
			i += width
		}
	}
	return out
}

// scanField reads one field of the given verb off the front of s and reports
// how many characters it took.
func scanField(s []rune, verb rune) (string, int) {
	switch verb {
	case 'c':
		if len(s) == 0 {
			return "", 0
		}
		return string(s[0]), 1
	case 's':
		n := 0
		for n < len(s) && !unicode.IsSpace(s[n]) {
			n++
		}
		return string(s[:n]), n
	case 'd', 'f', 'e', 'g':
		n := 0
		if n < len(s) && (s[n] == '+' || s[n] == '-') {
			n++
		}
		digits := n
		for n < len(s) && unicode.IsDigit(s[n]) {
			n++
		}
		if verb != 'd' && n < len(s) && s[n] == '.' {
			n++
			for n < len(s) && unicode.IsDigit(s[n]) {
				n++
			}
		}
		if n == digits {
			return "", 0
		}
		return string(s[:n]), n
	default:
		return "", 0
	}
}

// phpIntval is the longest integer at the front of the string in the given
// base, and zero when there is none.
func phpIntval(s string, base int) int {
	s = strings.TrimSpace(s)
	end := 0
	if end < len(s) && (s[end] == '+' || s[end] == '-') {
		end++
	}
	if base == 0 || base == 16 || base == 2 || base == 8 {
		// A prefix belongs to the number in the bases that write one.
		rest := strings.ToLower(s[end:])
		switch {
		case base != 8 && strings.HasPrefix(rest, "0x"):
			end += 2
		case base != 16 && strings.HasPrefix(rest, "0b"):
			end += 2
		}
	}
	digits := end
	for end < len(s) && digitValue(s[end]) >= 0 && (base == 0 || digitValue(s[end]) < base) {
		end++
	}
	if end == digits {
		return 0
	}
	v, err := strconv.ParseInt(s[:end], base, 64)
	if err != nil {
		return 0
	}
	return int(v)
}

// digitValue is the value of one digit in base 36, or -1 for a character that
// is not one.
func digitValue(c byte) int {
	switch {
	case c >= '0' && c <= '9':
		return int(c - '0')
	case c >= 'a' && c <= 'z':
		return int(c-'a') + 10
	case c >= 'A' && c <= 'Z':
		return int(c-'A') + 10
	default:
		return -1
	}
}

// phpFloatval is the longest decimal at the front of the string, and zero when
// there is none.
func phpFloatval(s string) float64 {
	s = strings.TrimSpace(s)
	end := 0
	if end < len(s) && (s[end] == '+' || s[end] == '-') {
		end++
	}
	digits := end
	for end < len(s) && s[end] >= '0' && s[end] <= '9' {
		end++
	}
	if end < len(s) && s[end] == '.' {
		end++
		for end < len(s) && s[end] >= '0' && s[end] <= '9' {
			end++
		}
	}
	if end == digits || (end == digits+1 && s[digits] == '.') {
		return 0
	}
	mantissa := end
	if end < len(s) && (s[end] == 'e' || s[end] == 'E') {
		end++
		if end < len(s) && (s[end] == '+' || s[end] == '-') {
			end++
		}
		exponent := end
		for end < len(s) && s[end] >= '0' && s[end] <= '9' {
			end++
		}
		if end == exponent {
			end = mantissa
		}
	}
	v, err := strconv.ParseFloat(s[:end], 64)
	if err != nil {
		return 0
	}
	return v
}

// phpDateTokens maps the letters a date format is written with to the reference
// times Go's layouts are written with. A letter not here has no Go equivalent
// and makes goLayout refuse the format.
var phpDateTokens = map[byte]string{
	'd': "02", 'j': "2", 'D': "Mon", 'l': "Monday",
	'm': "01", 'n': "1", 'M': "Jan", 'F': "January",
	'Y': "2006", 'y': "06",
	'H': "15", 'G': "15", 'h': "03", 'g': "3",
	'i': "04", 's': "05", 'v': "000", 'u': "000000",
	'A': "PM", 'a': "pm",
	'e': "MST", 'T': "MST", 'P': "-07:00", 'O': "-0700",
}

// goLayout translates a single-letter date format into the Go layout that reads
// the same text. A backslash escapes the letter behind it.
func goLayout(format string) (string, error) {
	var b strings.Builder
	b.Grow(len(format) * 2)
	for i := 0; i < len(format); i++ {
		c := format[i]
		if c == '\\' {
			if i+1 < len(format) {
				i++
				b.WriteByte(format[i])
			}
			continue
		}
		if layout, ok := phpDateTokens[c]; ok {
			b.WriteString(layout)
			continue
		}
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') {
			return "", fmt.Errorf("str: date format letter %q has no Go layout", string(c))
		}
		b.WriteByte(c)
	}
	return b.String(), nil
}

// parseLayouts are the writings of a time that a machine produces, tried in
// order when no format was given.
var parseLayouts = []string{
	time.RFC3339Nano,
	time.RFC3339,
	"2006-01-02T15:04:05",
	"2006-01-02 15:04:05.999999999",
	"2006-01-02 15:04:05",
	"2006-01-02 15:04",
	"2006-01-02",
	"2006/01/02",
	"02-01-2006",
	"15:04:05",
	"15:04",
	time.RFC1123Z,
	time.RFC1123,
	time.RFC822Z,
	time.RFC822,
	time.ANSIC,
}

// parseDate reads a time out of a string with no format given. A layout that
// carries no zone is read as UTC.
func parseDate(value string) (time.Time, error) {
	value = strings.TrimSpace(value)
	for _, layout := range parseLayouts {
		if t, err := time.Parse(layout, value); err == nil {
			return t, nil
		}
	}
	return time.Time{}, fmt.Errorf("str: %q is not a time this can read", value)
}
