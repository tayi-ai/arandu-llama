package str

import (
	"strings"
	"unicode/utf8"
)

// Transliterate writes the string in ASCII, putting unknown in the place of
// every rune it has no ASCII spelling for.
//
//	Transliterate("Ação", "?", false)  // "Acao"
//	Transliterate("ⓐⓑⓒ", "?", false)   // "abc"
//	Transliterate("🎂", "H", false)    // "H"
//
// The unknown spelling is required, because Go has no default arguments. Pass
// the empty string to drop what cannot be spelled, which is what ASCII does.
//
// strict changes nothing: both modes answer from the tables in this package,
// and a script outside them -- Greek, Cyrillic, the CJK readings -- comes back
// as unknown either way.
func Transliterate(s, unknown string, strict bool) string {
	_ = strict
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		switch {
		case r < utf8.RuneSelf:
			b.WriteRune(r)
		default:
			if spelled, ok := transliterationFor(r); ok {
				b.WriteString(spelled)
			} else {
				b.WriteString(unknown)
			}
		}
	}
	return b.String()
}

// transliterationFor is the ASCII spelling of a rune, from the table ASCII
// folds with first and the wider table this file adds behind it.
func transliterationFor(r rune) (string, bool) {
	if s, ok := asciiFolds[r]; ok {
		return s, true
	}
	if s, ok := transliterations[r]; ok {
		return s, true
	}
	switch {
	// Fullwidth forms are ASCII a fixed distance up the code space.
	case r >= 0xFF01 && r <= 0xFF5E:
		return string(r - 0xFEE0), true
	// Circled and parenthesized Latin letters and the circled digits.
	case r >= 0x24B6 && r <= 0x24CF:
		return string('A' + (r - 0x24B6)), true
	case r >= 0x24D0 && r <= 0x24E9:
		return string('a' + (r - 0x24D0)), true
	case r >= 0x2460 && r <= 0x2468:
		return string('1' + (r - 0x2460)), true
	}
	return "", false
}

// transliterations is what Transliterate knows beyond the fold table: the
// punctuation a word processor substitutes, the spacing that is not a space,
// and the handful of symbols that have a settled ASCII spelling.
var transliterations = map[rune]string{
	'\u00a0': " ",
	'\u2002': " ",
	'\u2003': " ",
	'\u2007': " ",
	'\u2009': " ",
	'\u200a': " ",
	'\u202f': " ",
	'\u3000': " ",
	'‘':      "'",
	'’':      "'",
	'‚':      ",",
	'‛':      "'",
	'“':      `"`,
	'”':      `"`,
	'„':      `"`,
	'′':      "'",
	'″':      `"`,
	'‐':      "-",
	'‑':      "-",
	'‒':      "-",
	'–':      "-",
	'—':      "-",
	'―':      "-",
	'…':      "...",
	'•':      "*",
	'·':      "*",
	'«':      "<<",
	'»':      ">>",
	'‹':      "<",
	'›':      ">",
	'©':      "(c)",
	'®':      "(r)",
	'™':      "(tm)",
	'°':      "deg",
	'±':      "+/-",
	'×':      "x",
	'÷':      "/",
	'≠':      "!=",
	'≤':      "<=",
	'≥':      ">=",
	'←':      "<-",
	'→':      "->",
	'€':      "EUR",
	'£':      "GBP",
	'¥':      "JPY",
	'¢':      "c",
	'¼':      "1/4",
	'½':      "1/2",
	'¾':      "3/4",
	'¡':      "!",
	'¿':      "?",
	'µ':      "u",
	'⁄':      "/",
	'§':      "SS",
	'¶':      "P",
	'†':      "+",
	'‡':      "++",
	'‰':      "0/00",
	'¹':      "1",
	'²':      "2",
	'³':      "3",
}
