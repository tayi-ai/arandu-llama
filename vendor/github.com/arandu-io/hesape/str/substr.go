package str

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// Length is the number of characters in the string, not the number of bytes, so
// Length("café") is 4.
func Length(value string) int { return runeLen(value) }

func runeLen(s string) int { return utf8.RuneCountInString(s) }

// Substr returns the portion of the string given by start and length, counted
// in characters.
//
// A negative start counts back from the end. A negative length stops that many
// characters short of the end. Passing no length runs to the end of the
// string.
func Substr(value string, start int, length ...int) string {
	return string(substrRunes([]rune(value), start, length...))
}

func substrRunes(rs []rune, start int, length ...int) []rune {
	n := len(rs)
	if start < 0 {
		start = max(n+start, 0)
	}
	if start > n {
		return nil
	}
	end := n
	if len(length) > 0 {
		if l := length[0]; l < 0 {
			end = n + l
		} else {
			end = start + l
		}
		end = min(end, n)
	}
	if end < start {
		return nil
	}
	return rs[start:end]
}

// Take returns the first limit characters of the string, or the last limit
// characters when limit is negative.
func Take(value string, limit int) string {
	if limit < 0 {
		return Substr(value, limit)
	}
	return Substr(value, 0, limit)
}

// CharAt returns the character at the given index, counting back from the end
// when the index is negative.
//
// The second result is false when the index is out of range, because the empty
// string is not distinguishable from a character on its own.
func CharAt(subject string, index int) (string, bool) {
	rs := []rune(subject)
	n := len(rs)
	if index < 0 {
		if index < -n {
			return "", false
		}
	} else if index > n-1 {
		return "", false
	}
	return string(substrRunes(rs, index, 1)), true
}

// Position is the character index of the first occurrence of needle at or after
// offset.
//
// It returns -1 when the needle is absent, the way strings.Index does, because
// a character index is never negative.
func Position(haystack, needle string, offset int) int {
	rs := []rune(haystack)
	n := len(rs)
	if offset < 0 {
		offset = max(n+offset, 0)
	}
	if offset > n {
		return -1
	}
	i := strings.Index(string(rs[offset:]), needle)
	if i < 0 {
		return -1
	}
	return offset + runeLen(string(rs[offset:])[:i])
}

// Limit cuts the string to at most limit display columns and appends end when
// it had to cut.
//
// Columns, not bytes: a full-width character counts as two. When preserveWords
// is true the cut is moved back to the last whitespace, and HTML tags and line
// breaks are removed first.
func Limit(value string, limit int, end string, preserveWords bool) string {
	if strWidth(value) <= limit {
		return value
	}
	if !preserveWords {
		return rtrimDefault(strimWidth(value, limit)) + end
	}

	value = strings.TrimSpace(collapseLineBreaks(stripTags(value)))

	trimmed := rtrimDefault(strimWidth(value, limit))

	if c, ok := CharAt(value, limit); ok && c == " " {
		return trimmed + end
	}
	if i := strings.LastIndexFunc(trimmed, unicode.IsSpace); i >= 0 {
		return trimmed[:i] + end
	}
	return trimmed + end
}

// Words cuts the string to at most words words and appends end when it had to
// cut. Whitespace inside what survives is left exactly as it was found.
//
// A words count of zero or less returns the string untouched, with no ending
// appended.
func Words(value string, words int, end string) string {
	if words <= 0 {
		return value
	}
	rs := []rune(value)
	i := 0
	for i < len(rs) && unicode.IsSpace(rs[i]) {
		i++
	}
	taken := 0
	for taken < words && i < len(rs) {
		for i < len(rs) && !unicode.IsSpace(rs[i]) {
			i++
		}
		for i < len(rs) && unicode.IsSpace(rs[i]) {
			i++
		}
		taken++
	}
	if i == len(rs) {
		return value
	}
	return rtrimDefault(string(rs[:i])) + end
}

// Squish trims both ends and collapses every run of whitespace inside the
// string to a single space.
//
// The two Hangul filler characters count as whitespace here, because they are
// invisible and a person pasting text carries them in without knowing.
func Squish(value string) string {
	var b strings.Builder
	b.Grow(len(value))
	pending := false
	for _, r := range Trim(value) {
		if isSquishSpace(r) {
			pending = b.Len() > 0
			continue
		}
		if pending {
			b.WriteByte(' ')
			pending = false
		}
		b.WriteRune(r)
	}
	return b.String()
}

func isSquishSpace(r rune) bool { return unicode.IsSpace(r) || r == 0x3164 || r == 0x1160 }

// Trim removes characters from both ends of the string. With no charlist it
// removes whitespace, the byte order mark, the zero width space and the
// left-to-right mark; with one it removes exactly the characters given.
func Trim(value string, charlist ...string) string {
	if len(charlist) == 0 {
		return strings.TrimFunc(value, isTrimDefault)
	}
	return strings.Trim(value, charlist[0])
}

// Ltrim is Trim on the left side only.
func Ltrim(value string, charlist ...string) string {
	if len(charlist) == 0 {
		return strings.TrimLeftFunc(value, isTrimDefault)
	}
	return strings.TrimLeft(value, charlist[0])
}

// Rtrim is Trim on the right side only.
func Rtrim(value string, charlist ...string) string {
	if len(charlist) == 0 {
		return strings.TrimRightFunc(value, isTrimDefault)
	}
	return strings.TrimRight(value, charlist[0])
}

func isTrimDefault(r rune) bool {
	return unicode.IsSpace(r) || r == 0 || r == 0xFEFF || r == 0x200B || r == 0x200E
}

// rtrimDefault trims spaces, tabs, line breaks, NULs and vertical tabs off the
// right end, which is what Limit and Words do before appending their ending.
func rtrimDefault(s string) string { return strings.TrimRight(s, " \t\n\r\x00\x0B") }

// After is everything past the first occurrence of search, or the whole string
// when search is empty or absent.
func After(subject, search string) string {
	if search == "" {
		return subject
	}
	i := strings.Index(subject, search)
	if i < 0 {
		return subject
	}
	return subject[i+len(search):]
}

// AfterLast is everything past the last occurrence of search, or the whole
// string when search is empty or absent.
func AfterLast(subject, search string) string {
	if search == "" {
		return subject
	}
	i := strings.LastIndex(subject, search)
	if i < 0 {
		return subject
	}
	return subject[i+len(search):]
}

// Before is everything up to the first occurrence of search, or the whole
// string when search is empty or absent.
func Before(subject, search string) string {
	if search == "" {
		return subject
	}
	i := strings.Index(subject, search)
	if i < 0 {
		return subject
	}
	return subject[:i]
}

// BeforeLast is everything up to the last occurrence of search, or the whole
// string when search is empty or absent.
func BeforeLast(subject, search string) string {
	if search == "" {
		return subject
	}
	i := strings.LastIndex(subject, search)
	if i < 0 {
		return subject
	}
	return subject[:i]
}

// Between is the largest portion sitting after the first from and before the
// last to. The whole string comes back when either delimiter is empty.
func Between(subject, from, to string) string {
	if from == "" || to == "" {
		return subject
	}
	return BeforeLast(After(subject, from), to)
}

// BetweenFirst is the smallest portion sitting after the first from and before
// the first to. The whole string comes back when either delimiter is empty.
func BetweenFirst(subject, from, to string) string {
	if from == "" || to == "" {
		return subject
	}
	return Before(After(subject, from), to)
}

// Start begins the string with a single instance of prefix, collapsing a prefix
// that is already there more than once: Start("//api", "/") is "/api".
func Start(value, prefix string) string {
	if prefix == "" {
		return value
	}
	for strings.HasPrefix(value, prefix) {
		value = value[len(prefix):]
	}
	return prefix + value
}

// Finish caps the string with a single instance of cap, collapsing a cap that
// is already there more than once: Finish("path//", "/") is "path/".
func Finish(value, cap string) string {
	if cap == "" {
		return value
	}
	for strings.HasSuffix(value, cap) {
		value = value[:len(value)-len(cap)]
	}
	return value + cap
}

// ChopStart removes the first of the given needles that the string starts with,
// and leaves the string alone when it starts with none of them.
func ChopStart(subject string, needle ...string) string {
	for _, n := range needle {
		if strings.HasPrefix(subject, n) {
			return subject[len(n):]
		}
	}
	return subject
}

// ChopEnd removes the first of the given needles that the string ends with, and
// leaves the string alone when it ends with none of them.
func ChopEnd(subject string, needle ...string) string {
	for _, n := range needle {
		if strings.HasSuffix(subject, n) {
			return subject[:len(subject)-len(n)]
		}
	}
	return subject
}

// Mask replaces a run of the string, starting at index, with the first
// character of character repeated:
// Mask("paulo@example.com", "*", 3, 8) is "pau********le.com".
//
// A negative index counts back from the end. Passing no length runs to the end
// of the string; a length of zero masks nothing and returns the string
// untouched.
//
// It is not redaction. The length of the answer still tells the reader how
// long the secret was, which for a password is more than it should learn.
func Mask(value, character string, index int, length ...int) string {
	if character == "" {
		return value
	}
	rs := []rune(value)
	segment := substrRunes(rs, index, length...)
	if len(segment) == 0 {
		return value
	}

	startIndex := index
	if index < 0 {
		if index < -len(rs) {
			startIndex = 0
		} else {
			startIndex = len(rs) + index
		}
	}

	mask := []rune(character)[0]
	for i := startIndex; i < startIndex+len(segment) && i < len(rs); i++ {
		rs[i] = mask
	}
	return string(rs)
}

// SubstrCount counts the non-overlapping occurrences of needle from offset
// onwards, in bytes.
//
// An empty needle returns zero, because there is no count that answers the
// question.
func SubstrCount(haystack, needle string, offset int, length ...int) int {
	if needle == "" {
		return 0
	}
	if offset < 0 {
		offset = max(len(haystack)+offset, 0)
	}
	if offset > len(haystack) {
		return 0
	}
	window := haystack[offset:]
	if len(length) > 0 {
		if l := length[0]; l < 0 {
			window = window[:max(len(window)+l, 0)]
		} else {
			window = window[:min(l, len(window))]
		}
	}
	return strings.Count(window, needle)
}

// SubstrReplace replaces the portion of the string given by offset and length,
// in bytes.
//
// A negative offset counts back from the end, a negative length stops that
// many bytes short of the end, and passing no length replaces everything from
// offset onwards.
func SubstrReplace(value, replace string, offset int, length ...int) string {
	n := len(value)
	if offset < 0 {
		offset = max(n+offset, 0)
	}
	offset = min(offset, n)

	end := n
	if len(length) > 0 {
		if l := length[0]; l < 0 {
			end = max(n+l, offset)
		} else {
			end = offset + l
		}
		end = min(end, n)
	}
	return value[:offset] + replace + value[end:]
}

// Excerpt cuts an excerpt out of the text around the first case-insensitive
// occurrence of phrase, keeping radius characters on each side and marking a
// cut end with omission.
//
// The second result is false when the phrase is not in the text.
func Excerpt(text, phrase string, radius int, omission string) (string, bool) {
	i := strings.Index(strings.ToLower(text), strings.ToLower(phrase))
	if i < 0 {
		return "", false
	}
	matched := text[i : i+len(phrase)]

	start := Ltrim(text[:i])
	startWithRadius := Ltrim(Substr(start, max(runeLen(start)-radius, 0), radius))
	if startWithRadius != start {
		startWithRadius = omission + startWithRadius
	}

	end := Rtrim(text[i+len(phrase):])
	endWithRadius := Rtrim(Substr(end, 0, radius))
	if endWithRadius != end {
		endWithRadius += omission
	}

	return startWithRadius + matched + endWithRadius, true
}

// strWidth is the display width of a string: a full-width character is two
// columns wide and everything else is one.
func strWidth(s string) int {
	w := 0
	for _, r := range s {
		w += runeWidth(r)
	}
	return w
}

func runeWidth(r rune) int {
	switch {
	case r >= 0x1100 && r <= 0x115F,
		r >= 0x2E80 && r <= 0xA4CF,
		r >= 0xAC00 && r <= 0xD7A3,
		r >= 0xF900 && r <= 0xFAFF,
		r >= 0xFE30 && r <= 0xFE6F,
		r >= 0xFF00 && r <= 0xFF60,
		r >= 0xFFE0 && r <= 0xFFE6,
		r >= 0x20000 && r <= 0x3FFFD:
		return 2
	default:
		return 1
	}
}

// strimWidth keeps the leading characters that fit in the given number of
// columns, with no marker in the place of what it dropped.
func strimWidth(s string, width int) string {
	used := 0
	for i, r := range s {
		w := runeWidth(r)
		if used+w > width {
			return s[:i]
		}
		used += w
	}
	return s
}

// collapseLineBreaks turns every run of line breaks into one space, which Limit
// does before it looks for a word boundary.
func collapseLineBreaks(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	pending := false
	for _, r := range s {
		if r == '\n' || r == '\r' {
			pending = true
			continue
		}
		if pending {
			b.WriteByte(' ')
			pending = false
		}
		b.WriteRune(r)
	}
	if pending {
		b.WriteByte(' ')
	}
	return b.String()
}

// stripTags removes everything between an opening angle bracket and the next
// closing one.
func stripTags(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	depth := 0
	for _, r := range s {
		switch {
		case r == '<':
			depth++
		case r == '>' && depth > 0:
			depth--
		case depth == 0:
			b.WriteRune(r)
		}
	}
	return b.String()
}
