package str

import (
	"strings"
	"unicode/utf8"
)

// Slug is the text as an address: Slug("Josefin Sans Café", "-") is
// "josefin-sans-cafe".
//
// The separator is required, as it is on Snake, because Go has no default
// arguments. Pass "" to run the words together.
//
// It folds to ASCII first, so an accented rune becomes the letter it is built
// on instead of disappearing: two names differing only in their accents would
// otherwise slug to one and the same address.
//
// An at sign becomes the word "at" before anything else happens, which is what
// keeps two addresses at different hosts from slugging to the same thing.
func Slug(s, separator string) string {
	folded := strings.ToLower(strings.ReplaceAll(ASCII(s), "@", " at "))

	var b strings.Builder
	b.Grow(len(folded))
	pending := false
	for _, r := range folded {
		if (r < 'a' || r > 'z') && (r < '0' || r > '9') {
			pending = b.Len() > 0
			continue
		}
		if pending {
			b.WriteString(separator)
			pending = false
		}
		b.WriteRune(r)
	}
	return b.String()
}

// ASCII folds a string to ASCII: "Ação" becomes "Acao", "Straße" becomes
// "Strasse".
//
// It is a table, not a transliterator: what it covers is Latin-1 Supplement and
// Latin Extended-A -- every language written in a Latin alphabet -- and a rune
// outside it is dropped rather than guessed at. Greek, Cyrillic and CJK come
// out empty, and a caller that slugs a title in one of those has to carry its
// own key.
func ASCII(s string) string {
	if isASCII(s) {
		return s
	}
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		switch {
		case r < utf8.RuneSelf:
			b.WriteRune(r)
		default:
			b.WriteString(asciiFolds[r])
		}
	}
	return b.String()
}

func isASCII(s string) bool {
	for i := range len(s) {
		if s[i] >= utf8.RuneSelf {
			return false
		}
	}
	return true
}

// asciiFolds is every rune ASCII knows how to answer for. A rune that is not in
// it folds to the empty string, which is the drop ASCII documents.
var asciiFolds = folds(
	"ÀÁÂÃÄÅĀĂĄ", "A",
	"àáâãäåāăą", "a",
	"Æ", "AE",
	"æ", "ae",
	"ÇĆĈĊČ", "C",
	"çćĉċč", "c",
	"ÐĎĐ", "D",
	"ðďđ", "d",
	"ÈÉÊËĒĔĖĘĚ", "E",
	"èéêëēĕėęě", "e",
	"ĜĞĠĢ", "G",
	"ĝğġģ", "g",
	"ĤĦ", "H",
	"ĥħ", "h",
	"ÌÍÎÏĨĪĬĮİ", "I",
	"ìíîïĩīĭįı", "i",
	"Ĳ", "IJ",
	"ĳ", "ij",
	"Ĵ", "J",
	"ĵ", "j",
	"Ķ", "K",
	"ķĸ", "k",
	"ĹĻĽĿŁ", "L",
	"ĺļľŀł", "l",
	"ÑŃŅŇŊ", "N",
	"ñńņňŉŋ", "n",
	"ÒÓÔÕÖØŌŎŐ", "O",
	"òóôõöøōŏő", "o",
	"Œ", "OE",
	"œ", "oe",
	"ŔŖŘ", "R",
	"ŕŗř", "r",
	"ŚŜŞŠ", "S",
	"śŝşšſ", "s",
	"ß", "ss",
	"ŢŤŦ", "T",
	"ţťŧ", "t",
	"Þ", "TH",
	"þ", "th",
	"ÙÚÛÜŨŪŬŮŰŲ", "U",
	"ùúûüũūŭůűų", "u",
	"Ŵ", "W",
	"ŵ", "w",
	"ÝŶŸ", "Y",
	"ýÿŷ", "y",
	"ŹŻŽ", "Z",
	"źżž", "z",
)

// folds builds the table from pairs of "these runes" and "this ASCII". Written
// as pairs because a map literal with three hundred entries is a table nobody
// checks.
func folds(pairs ...string) map[rune]string {
	table := make(map[rune]string, 256)
	for i := 0; i+1 < len(pairs); i += 2 {
		for _, r := range pairs[i] {
			table[r] = pairs[i+1]
		}
	}
	return table
}
