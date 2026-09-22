package str

import "strings"

// PadLeft grows the string to the given number of characters by putting pad in
// front of it, repeating and cutting pad as needed; passing no pad uses a
// space.
//
// A length no greater than the string already has returns it untouched, so
// PadLeft("hello", 3) is "hello" and not "hel". An empty pad returns it
// untouched too.
func PadLeft(value string, length int, pad ...string) string {
	short, filler := padRoom(value, length, pad)
	if short <= 0 {
		return value
	}
	return padRepeat(filler, short) + value
}

// PadRight is PadLeft with the padding behind the string.
func PadRight(value string, length int, pad ...string) string {
	short, filler := padRoom(value, length, pad)
	if short <= 0 {
		return value
	}
	return value + padRepeat(filler, short)
}

// PadBoth splits the padding between the two ends, giving the odd character to
// the right side.
func PadBoth(value string, length int, pad ...string) string {
	short, filler := padRoom(value, length, pad)
	if short <= 0 {
		return value
	}
	left := short / 2
	return padRepeat(filler, left) + value + padRepeat(filler, short-left)
}

// padRoom is the number of characters the padding has to fill and the string to
// fill them with.
func padRoom(value string, length int, pad []string) (int, string) {
	filler := " "
	if len(pad) > 0 {
		filler = pad[0]
	}
	if filler == "" {
		return 0, filler
	}
	return length - Length(value), filler
}

// padRepeat writes exactly n characters of the filler, repeating and cutting it.
func padRepeat(filler string, n int) string {
	rs := []rune(filler)
	out := make([]rune, 0, n)
	for len(out) < n {
		out = append(out, rs[:min(len(rs), n-len(out))]...)
	}
	return string(out)
}

// WordCount counts the words in the string, where a word is a run of letters
// and of the extra characters given.
//
// An apostrophe and a hyphen are part of a word but cannot open one -- and a
// hyphen cannot close the string -- unless the caller names them in characters.
// Letters are the ASCII ones, because the counting reads bytes.
func WordCount(value string, characters ...string) int {
	extra := ""
	if len(characters) > 0 {
		extra = characters[0]
	}
	isWord := func(c byte) bool {
		return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
			c == '\'' || c == '-' || strings.IndexByte(extra, c) >= 0
	}

	start, end := 0, len(value)
	if end == 0 {
		return 0
	}
	if (value[start] == '\'' && strings.IndexByte(extra, '\'') < 0) ||
		(value[start] == '-' && strings.IndexByte(extra, '-') < 0) {
		start++
	}
	if value[end-1] == '-' && strings.IndexByte(extra, '-') < 0 {
		end--
	}

	count := 0
	for p := start; p < end; p++ {
		s := p
		for p < end && isWord(value[p]) {
			p++
		}
		if p > s {
			count++
		}
	}
	return count
}

// WordWrap breaks the string into lines of at most the given number of
// characters, breaking at spaces and writing brk where it breaks; a word longer
// than the line is left whole unless cutLongWords is set.
//
// It counts bytes rather than characters, and treats an occurrence of brk
// already in the string as a line that has ended. A width of zero or less
// returns the string untouched.
func WordWrap(value string, characters int, brk string, cutLongWords bool) string {
	if value == "" || brk == "" || characters <= 0 {
		return value
	}

	var b strings.Builder
	b.Grow(len(value) + len(value)/max(characters, 1)*len(brk))

	laststart, lastspace, current := 0, 0, 0
	for ; current < len(value); current++ {
		switch {
		case value[current] == brk[0] && current+len(brk) <= len(value) &&
			value[current:current+len(brk)] == brk:
			b.WriteString(value[laststart : current+len(brk)])
			laststart = current + len(brk)
			lastspace = laststart
			current += len(brk) - 1

		case value[current] == ' ':
			if current-laststart >= characters {
				b.WriteString(value[laststart:current])
				b.WriteString(brk)
				laststart = current + 1
			}
			lastspace = current

		case current-laststart >= characters && laststart >= lastspace && cutLongWords:
			b.WriteString(value[laststart:current])
			b.WriteString(brk)
			laststart = current
			lastspace = current

		case current-laststart >= characters && laststart < lastspace:
			b.WriteString(value[laststart:lastspace])
			b.WriteString(brk)
			laststart = lastspace + 1
			lastspace = laststart
		}
	}
	if laststart != current {
		b.WriteString(value[laststart:current])
	}
	return b.String()
}
