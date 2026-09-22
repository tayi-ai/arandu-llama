package str

import (
	"regexp"
	"strings"
)

var (
	// entityReference is an HTML entity, which passes through instead of
	// having its ampersand escaped.
	entityReference = regexp.MustCompile(`^&(#[0-9]{1,7}|#[xX][0-9a-fA-F]{1,6}|[A-Za-z][A-Za-z0-9]{1,31});`)

	// rawInlineTag is an HTML tag written inside a paragraph, which passes
	// through to the output untouched.
	rawInlineTag = regexp.MustCompile(`^<(/?[A-Za-z][A-Za-z0-9-]*(\s[^<>]*)?/?|!--.*?--)>`)

	// autolink is the <...> form, which needs a scheme or an email address.
	autolink = regexp.MustCompile(`^<([A-Za-z][A-Za-z0-9+.-]{1,31}:[^<>\x00-\x20]*|[^\s<>@]+@[^\s<>@]+\.[^\s<>@]+)>`)

	// bareURL is the GitHub flavour's autolinking of a URL nobody wrapped.
	bareURL = regexp.MustCompile(`^(https?://|www\.)[^\s<]*[^\s<.,:;"')\]]`)

	// asciiPunctuation is what a backslash may escape, per CommonMark.
	asciiPunctuation = `!"#$%&'()*+,-./:;<=>?@[\]^_` + "`" + `{|}~`
)

// renderInline renders the inline part of CommonMark: escapes, code spans,
// autolinks, raw HTML, images, links, emphasis, strikethrough and hard breaks.
func renderInline(s string) string {
	var b strings.Builder
	b.Grow(len(s) + len(s)/4)

	for i := 0; i < len(s); {
		switch c := s[i]; {
		case c == '\\' && i+1 < len(s) && strings.IndexByte(asciiPunctuation, s[i+1]) >= 0:
			b.WriteString(escapeHTML(string(s[i+1])))
			i += 2

		case c == '\\' && i+1 < len(s) && s[i+1] == '\n':
			b.WriteString("<br />\n")
			i += 2

		case c == '`':
			if code, width := codeSpan(s[i:]); width > 0 {
				b.WriteString(code)
				i += width
				continue
			}
			b.WriteString("`")
			i++

		case c == '<':
			if m := autolink.FindStringSubmatch(s[i:]); m != nil {
				href := m[1]
				if !strings.Contains(href, ":") {
					href = "mailto:" + href
				}
				b.WriteString(`<a href="` + escapeHTML(href) + `">` + escapeHTML(m[1]) + `</a>`)
				i += len(m[0])
				continue
			}
			if m := rawInlineTag.FindString(s[i:]); m != "" {
				b.WriteString(m)
				i += len(m)
				continue
			}
			b.WriteString("&lt;")
			i++

		case c == '&':
			if m := entityReference.FindString(s[i:]); m != "" {
				b.WriteString(m)
				i += len(m)
				continue
			}
			b.WriteString("&amp;")
			i++

		case c == '>':
			b.WriteString("&gt;")
			i++

		case c == '"':
			b.WriteString("&quot;")
			i++

		case c == '!' && i+1 < len(s) && s[i+1] == '[':
			if html, width := linkOrImage(s[i:], true); width > 0 {
				b.WriteString(html)
				i += width
				continue
			}
			b.WriteString("!")
			i++

		case c == '[':
			if html, width := linkOrImage(s[i:], false); width > 0 {
				b.WriteString(html)
				i += width
				continue
			}
			b.WriteString("[")
			i++

		case c == '*' || c == '_' || c == '~':
			if c == '_' && !startsWord(s, i) {
				b.WriteByte(c)
				i++
				continue
			}
			if html, width := emphasis(s[i:], c); width > 0 {
				b.WriteString(html)
				i += width
				continue
			}
			b.WriteByte(c)
			i++

		case c == '\n':
			if strings.HasSuffix(b.String(), "  ") {
				trimHardBreakSpaces(&b)
				b.WriteString("<br />\n")
			} else {
				b.WriteString("\n")
			}
			i++

		case c == 'h' || c == 'w':
			if m := bareURL.FindString(s[i:]); m != "" && startsWord(s, i) {
				href := m
				if strings.HasPrefix(m, "www.") {
					href = "http://" + m
				}
				b.WriteString(`<a href="` + escapeHTML(href) + `">` + escapeHTML(m) + `</a>`)
				i += len(m)
				continue
			}
			b.WriteByte(c)
			i++

		default:
			b.WriteByte(c)
			i++
		}
	}
	return b.String()
}

// startsWord reports whether the byte before i ends a word, so that the "www."
// inside "awww.no" is not read as a URL.
func startsWord(s string, i int) bool {
	if i == 0 {
		return true
	}
	c := s[i-1]
	return !isWordByte(c) && c != '/' && c != '.' && c != '@'
}

// isWordByte reports whether c can sit inside a word, which is what decides
// that an underscore between two letters is not an emphasis marker.
func isWordByte(c byte) bool {
	return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c >= 0x80
}

// trimHardBreakSpaces removes the two spaces that made a hard line break, which
// the builder has already been handed.
func trimHardBreakSpaces(b *strings.Builder) {
	kept := strings.TrimRight(b.String(), " ")
	b.Reset()
	b.WriteString(kept)
}

// codeSpan reads a backtick-delimited code span off the front of s and reports
// how much of it the span took.
func codeSpan(s string) (string, int) {
	open := 0
	for open < len(s) && s[open] == '`' {
		open++
	}
	fence := s[:open]
	rest := s[open:]

	for i := 0; i+open <= len(rest); i++ {
		if !strings.HasPrefix(rest[i:], fence) {
			continue
		}
		if i+open < len(rest) && rest[i+open] == '`' {
			continue
		}
		body := rest[:i]
		// One space on each side is stripped when the content is not all spaces.
		if len(body) >= 2 && body[0] == ' ' && body[len(body)-1] == ' ' && strings.TrimSpace(body) != "" {
			body = body[1 : len(body)-1]
		}
		return "<code>" + escapeHTML(strings.ReplaceAll(body, "\n", " ")) + "</code>", open + i + open
	}
	return "", 0
}

// linkOrImage reads an inline link or image off the front of s and reports how
// much of it it took. Reference links are not read: there is no link reference
// definition parser here.
func linkOrImage(s string, image bool) (string, int) {
	bracket := 0
	if image {
		bracket = 1
	}
	text, labelEnd := matchBrackets(s, bracket)
	if labelEnd < 0 || labelEnd >= len(s) || s[labelEnd] != '(' {
		return "", 0
	}
	destination, title, end := linkTarget(s[labelEnd:])
	if end < 0 {
		return "", 0
	}

	attributes := `href="` + escapeHTML(destination) + `"`
	if image {
		attributes = `src="` + escapeHTML(destination) + `" alt="` + escapeHTML(stripInline(text)) + `"`
	}
	if title != "" {
		attributes += ` title="` + escapeHTML(title) + `"`
	}
	if image {
		return "<img " + attributes + " />", labelEnd + end
	}
	return "<a " + attributes + ">" + renderInline(text) + "</a>", labelEnd + end
}

// matchBrackets finds the closing bracket that answers the one at open,
// counting the pairs in between, and reports the index just past it.
func matchBrackets(s string, open int) (string, int) {
	depth := 0
	for i := open; i < len(s); i++ {
		switch s[i] {
		case '\\':
			i++
		case '[':
			depth++
		case ']':
			depth--
			if depth == 0 {
				return s[open+1 : i], i + 1
			}
		}
	}
	return "", -1
}

// linkTarget reads the "(destination "title")" of a link and reports the index
// just past the closing parenthesis.
func linkTarget(s string) (destination, title string, end int) {
	if len(s) == 0 || s[0] != '(' {
		return "", "", -1
	}
	i := 1
	for i < len(s) && (s[i] == ' ' || s[i] == '\n') {
		i++
	}
	if i < len(s) && s[i] == '<' {
		j := strings.IndexByte(s[i:], '>')
		if j < 0 {
			return "", "", -1
		}
		destination = s[i+1 : i+j]
		i += j + 1
	} else {
		depth, start := 0, i
		for i < len(s) {
			c := s[i]
			if c == '\\' {
				i += 2
				continue
			}
			if c == '(' {
				depth++
			}
			if c == ')' {
				if depth == 0 {
					break
				}
				depth--
			}
			if c == ' ' || c == '\n' {
				break
			}
			i++
		}
		destination = unescapeMarkdown(s[start:i])
	}
	for i < len(s) && (s[i] == ' ' || s[i] == '\n') {
		i++
	}
	if i < len(s) && (s[i] == '"' || s[i] == '\'') {
		quote := s[i]
		j := strings.IndexByte(s[i+1:], quote)
		if j < 0 {
			return "", "", -1
		}
		title = unescapeMarkdown(s[i+1 : i+1+j])
		i += j + 2
	}
	for i < len(s) && (s[i] == ' ' || s[i] == '\n') {
		i++
	}
	if i >= len(s) || s[i] != ')' {
		return "", "", -1
	}
	return destination, title, i + 1
}

// unescapeMarkdown drops the backslashes that were escaping punctuation, which
// a destination and a title carry as their literal text.
func unescapeMarkdown(s string) string {
	var b strings.Builder
	for i := 0; i < len(s); i++ {
		if s[i] == '\\' && i+1 < len(s) && strings.IndexByte(asciiPunctuation, s[i+1]) >= 0 {
			i++
		}
		b.WriteByte(s[i])
	}
	return b.String()
}

// anyTag is every HTML tag, which stripInline drops to reach the text under it.
var anyTag = regexp.MustCompile(`<[^>]*>`)

// htmlUnescaper reads back what escapeHTML wrote.
var htmlUnescaper = strings.NewReplacer("&lt;", "<", "&gt;", ">", "&quot;", `"`, "&amp;", "&")

// stripInline is the plain text of a span, which is what an image alt carries.
func stripInline(s string) string {
	return htmlUnescaper.Replace(anyTag.ReplaceAllString(renderInline(s), ""))
}

// emphasis reads a run of emphasis markers and the span it closes, and reports
// how much of the string it took.
//
// Two markers are strong, one is emphasis, and two tildes are the GitHub
// flavour's strikethrough. An underscore inside a word opens nothing, which is
// what keeps snake_case_names whole.
func emphasis(s string, marker byte) (string, int) {
	run := 0
	for run < len(s) && s[run] == marker {
		run++
	}
	if marker == '~' && run != 2 {
		return "", 0
	}
	if run > 2 {
		run = 2
	}
	if run+1 > len(s) || s[run] == ' ' || s[run] == '\n' {
		return "", 0
	}

	open := s[:run]
	for i := run; i < len(s); i++ {
		if s[i] == '\\' {
			i++
			continue
		}
		if !strings.HasPrefix(s[i:], open) || s[i-1] == ' ' {
			continue
		}
		if i+run < len(s) && s[i+run] == marker {
			continue
		}
		if marker == '_' && i+run < len(s) && isWordByte(s[i+run]) {
			// An underscore inside a word closes nothing either, which is what
			// keeps snake_case_names whole.
			continue
		}
		inner := s[run:i]
		switch {
		case marker == '~':
			return "<del>" + renderInline(inner) + "</del>", i + run
		case run == 2:
			return "<strong>" + renderInline(inner) + "</strong>", i + run
		default:
			return "<em>" + renderInline(inner) + "</em>", i + run
		}
	}
	return "", 0
}
