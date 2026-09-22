package str

import (
	"strings"
	"sync"
	"unicode"
)

// The three casing caches. They are read on every call and cleared only by
// FlushCache.
var (
	caseCacheMu sync.RWMutex
	snakeCache  = map[string]string{}
	camelCache  = map[string]string{}
	studlyCache = map[string]string{}
)

func cachedCase(cache map[string]string, key string, compute func() string) string {
	caseCacheMu.RLock()
	v, ok := cache[key]
	caseCacheMu.RUnlock()
	if ok {
		return v
	}
	v = compute()
	caseCacheMu.Lock()
	cache[key] = v
	caseCacheMu.Unlock()
	return v
}

// FlushCache empties the snake, camel and studly caches.
func FlushCache() {
	caseCacheMu.Lock()
	clear(snakeCache)
	clear(camelCache)
	clear(studlyCache)
	caseCacheMu.Unlock()
}

// Snake converts a string to snake case with the given delimiter:
// Snake("PurchaseOrder", "_") is "purchase_order".
//
// A delimiter goes in before every capital, one per capital, so
// Snake("HTTPServer", "_") is "h_t_t_p_server". A string that is already all
// lowercase letters is returned untouched, so Snake("foo-bar", "_") is
// "foo-bar" and not "foo_bar".
func Snake(value, delimiter string) string {
	return cachedCase(snakeCache, value+"\x00"+delimiter, func() string {
		if ctypeLower(value) {
			return value
		}
		spaced := ucwords(value)
		var stripped strings.Builder
		stripped.Grow(len(spaced))
		for _, r := range spaced {
			if !unicode.IsSpace(r) {
				stripped.WriteRune(r)
			}
		}
		rs := []rune(stripped.String())
		var b strings.Builder
		b.Grow(len(value) + 8)
		for i, r := range rs {
			b.WriteRune(r)
			if i+1 < len(rs) && rs[i+1] >= 'A' && rs[i+1] <= 'Z' {
				b.WriteString(delimiter)
			}
		}
		return strings.ToLower(b.String())
	})
}

// Kebab is Snake with a hyphen: Kebab("WelcomeEmail") is "welcome-email".
func Kebab(value string) string { return Snake(value, "-") }

// Studly converts a string to studly caps:
// Studly("invoice_line") and Studly("invoice-line") are both "InvoiceLine".
//
// Only the first letter of each dash-, underscore- or space-separated part is
// touched, so Studly("HTTPServer") is "HTTPServer" and not "HttpServer".
func Studly(value string) string {
	return cachedCase(studlyCache, value, func() string {
		var b strings.Builder
		b.Grow(len(value))
		for _, word := range strings.Split(studlyBoundaries.Replace(value), " ") {
			b.WriteString(Ucfirst(word))
		}
		return b.String()
	})
}

var studlyBoundaries = strings.NewReplacer("-", " ", "_", " ")

// Pascal converts a string to Pascal case: Pascal("invoice_line") and
// Pascal("invoice-line") are both "InvoiceLine". Only the first letter of each
// dash-, underscore- or space-separated part is touched, so Pascal("HTTPServer")
// is "HTTPServer" and not "HttpServer".
//
// It is [Studly] under the other name for the same convention, and the two are
// interchangeable in every call.
func Pascal(value string) string { return Studly(value) }

// Camel is Studly with a lowercase initial: Camel("purchase_order") is
// "purchaseOrder".
func Camel(value string) string {
	return cachedCase(camelCache, value, func() string { return Lcfirst(Studly(value)) })
}

// Ucfirst uppercases the first character and leaves the rest of the string
// alone.
func Ucfirst(value string) string {
	if value == "" {
		return value
	}
	rs := []rune(value)
	return strings.ToUpper(string(rs[0])) + string(rs[1:])
}

// Lcfirst lowercases the first character and leaves the rest of the string
// alone.
func Lcfirst(value string) string {
	if value == "" {
		return value
	}
	rs := []rune(value)
	return strings.ToLower(string(rs[0])) + string(rs[1:])
}

// Ucsplit splits a string into pieces in front of every uppercase letter,
// dropping the empty pieces:
// Ucsplit("EmailNotificationSent") is ["Email", "Notification", "Sent"].
func Ucsplit(value string) []string {
	var out []string
	start := 0
	for i, r := range value {
		if i > start && unicode.IsUpper(r) {
			out = append(out, value[start:i])
			start = i
		}
	}
	if start < len(value) {
		out = append(out, value[start:])
	}
	return out
}

// Title uppercases the first letter of every word and lowercases the rest of
// each one: Title("welcome EMAIL") is "Welcome Email".
//
// A word starts after any character that is not a letter, which is what makes
// Title("hello-world") "Hello-World" and Title("o'neil") "O'Neil".
func Title(value string) string {
	var b strings.Builder
	b.Grow(len(value))
	boundary := true
	for _, r := range value {
		switch {
		case !unicode.IsLetter(r):
			boundary = true
			b.WriteRune(r)
		case boundary:
			boundary = false
			b.WriteRune(unicode.ToUpper(r))
		default:
			b.WriteRune(unicode.ToLower(r))
		}
	}
	return b.String()
}

// The case modes ConvertCase accepts.
const (
	CaseUpper = 0
	CaseLower = 1
	CaseTitle = 2
	CaseFold  = 3
)

// ConvertCase converts a string in the given mode, which is one of CaseUpper,
// CaseLower, CaseTitle or CaseFold. There is no default: name the one wanted.
//
// CaseFold is Unicode simple case folding, which for every alphabet this
// package can reach is lowercasing.
func ConvertCase(value string, mode int) string {
	switch mode {
	case CaseUpper:
		return strings.ToUpper(value)
	case CaseTitle:
		return Title(value)
	default:
		return strings.ToLower(value)
	}
}

// Headline converts a string to a title-cased sentence:
// Headline("EmailNotificationSent") and
// Headline("email_notification_sent") are both "Email Notification Sent".
func Headline(value string) string {
	parts := strings.Split(value, " ")
	if len(parts) == 1 {
		parts = Ucsplit(strings.Join(parts, "_"))
	}
	for i, p := range parts {
		parts[i] = Title(p)
	}
	collapsed := headlineBoundaries.Replace(strings.Join(parts, "_"))

	var kept []string
	for _, piece := range strings.Split(collapsed, "_") {
		// An empty piece is dropped, and so is a piece that is only "0".
		if piece != "" && piece != "0" {
			kept = append(kept, piece)
		}
	}
	return strings.Join(kept, " ")
}

var headlineBoundaries = strings.NewReplacer("-", "_", " ", "_")

// apaMinorWords are the words Apa leaves lowercase when they are short enough
// and not at the start of the title.
var apaMinorWords = map[string]bool{
	"and": true, "as": true, "but": true, "for": true, "if": true, "nor": true,
	"or": true, "so": true, "yet": true, "a": true, "an": true, "the": true,
	"at": true, "by": true, "in": true, "of": true, "off": true, "on": true,
	"per": true, "to": true, "up": true, "via": true, "et": true, "ou": true,
	"un": true, "une": true, "la": true, "le": true, "les": true, "de": true,
	"du": true, "des": true, "par": true, "à": true,
}

// apaEndPunctuation are the characters that end a sentence inside a title, so
// that the word after one of them is capitalized even when it is minor.
var apaEndPunctuation = map[string]bool{".": true, "!": true, "?": true, ":": true, "—": true, ",": true}

// Apa converts a string to APA-style title case, which leaves minor words of
// three letters or fewer lowercase unless they open the title or follow end
// punctuation.
//
// See https://apastyle.apa.org/style-grammar-guidelines/capitalization/title-case.
func Apa(value string) string {
	if strings.TrimSpace(value) == "" {
		return value
	}
	words := strings.FieldsFunc(value, unicode.IsSpace)

	for i := range words {
		lowercaseWord := strings.ToLower(words[i])

		if strings.Contains(lowercaseWord, "-") {
			hyphenated := strings.Split(lowercaseWord, "-")
			for j, part := range hyphenated {
				if apaMinorWords[part] && runeLen(part) <= 3 {
					hyphenated[j] = part
					continue
				}
				hyphenated[j] = Ucfirst(part)
			}
			words[i] = strings.Join(hyphenated, "-")
			continue
		}

		afterEnd := i == 0
		if i > 0 {
			prev := []rune(words[i-1])
			afterEnd = len(prev) > 0 && apaEndPunctuation[string(prev[len(prev)-1])]
		}
		if apaMinorWords[lowercaseWord] && runeLen(lowercaseWord) <= 3 && !afterEnd {
			words[i] = lowercaseWord
			continue
		}
		words[i] = Ucfirst(lowercaseWord)
	}

	return strings.Join(words, " ")
}

// ctypeLower reports whether every byte is an ASCII lowercase letter. The empty
// string is not.
func ctypeLower(s string) bool {
	if s == "" {
		return false
	}
	for i := range len(s) {
		if s[i] < 'a' || s[i] > 'z' {
			return false
		}
	}
	return true
}

// ucwords uppercases the first ASCII lowercase letter of the string and every
// one that follows a whitespace byte.
func ucwords(s string) string {
	b := []byte(s)
	boundary := true
	for i := range b {
		if boundary && b[i] >= 'a' && b[i] <= 'z' {
			b[i] -= 'a' - 'A'
		}
		switch b[i] {
		case ' ', '\t', '\r', '\n', '\f', '\v':
			boundary = true
		default:
			boundary = false
		}
	}
	return string(b)
}

// isSeparator reports whether r is one of the runes that ends a word without
// being part of one. It is what splitTail uses to find the last word of a
// compound name.
func isSeparator(r rune) bool {
	return r == '_' || r == '-' || r == '.' || unicode.IsSpace(r)
}
