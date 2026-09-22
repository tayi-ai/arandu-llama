package cache

import "strings"

// htmlEntity is the HTML 4.01 named entity for a character.
//
// It exists for one caller, CleanRateLimiterKey, and it is a table rather than a
// call into a library because this collection carries no third-party
// dependency. It is the Latin-1 block, the Latin Extended-A characters HTML
// 4.01 names, the Greek letters, and the punctuation and symbols.
//
// The apostrophe is deliberately absent: it has no named entity, only the
// numeric &#039;, and the difference is visible in the answer. See
// CleanRateLimiterKey.
var htmlEntity = map[rune]string{
	'&': "amp", '<': "lt", '>': "gt", '"': "quot",

	// Latin-1.
	' ': "nbsp", '¡': "iexcl", '¢': "cent", '£': "pound", '¤': "curren",
	'¥': "yen", '¦': "brvbar", '§': "sect", '¨': "uml", '©': "copy",
	'ª': "ordf", '«': "laquo", '¬': "not", '­': "shy", '®': "reg",
	'¯': "macr", '°': "deg", '±': "plusmn", '²': "sup2", '³': "sup3",
	'´': "acute", 'µ': "micro", '¶': "para", '·': "middot", '¸': "cedil",
	'¹': "sup1", 'º': "ordm", '»': "raquo", '¼': "frac14", '½': "frac12",
	'¾': "frac34", '¿': "iquest",
	'À': "Agrave", 'Á': "Aacute", 'Â': "Acirc", 'Ã': "Atilde", 'Ä': "Auml",
	'Å': "Aring", 'Æ': "AElig", 'Ç': "Ccedil", 'È': "Egrave", 'É': "Eacute",
	'Ê': "Ecirc", 'Ë': "Euml", 'Ì': "Igrave", 'Í': "Iacute", 'Î': "Icirc",
	'Ï': "Iuml", 'Ð': "ETH", 'Ñ': "Ntilde", 'Ò': "Ograve", 'Ó': "Oacute",
	'Ô': "Ocirc", 'Õ': "Otilde", 'Ö': "Ouml", '×': "times", 'Ø': "Oslash",
	'Ù': "Ugrave", 'Ú': "Uacute", 'Û': "Ucirc", 'Ü': "Uuml", 'Ý': "Yacute",
	'Þ': "THORN", 'ß': "szlig",
	'à': "agrave", 'á': "aacute", 'â': "acirc", 'ã': "atilde", 'ä': "auml",
	'å': "aring", 'æ': "aelig", 'ç': "ccedil", 'è': "egrave", 'é': "eacute",
	'ê': "ecirc", 'ë': "euml", 'ì': "igrave", 'í': "iacute", 'î': "icirc",
	'ï': "iuml", 'ð': "eth", 'ñ': "ntilde", 'ò': "ograve", 'ó': "oacute",
	'ô': "ocirc", 'õ': "otilde", 'ö': "ouml", '÷': "divide", 'ø': "oslash",
	'ù': "ugrave", 'ú': "uacute", 'û': "ucirc", 'ü': "uuml", 'ý': "yacute",
	'þ': "thorn", 'ÿ': "yuml",

	// Latin Extended-A and the letter-like symbols HTML 4.01 names.
	'Œ': "OElig", 'œ': "oelig", 'Š': "Scaron", 'š': "scaron", 'Ÿ': "Yuml",
	'ƒ': "fnof",

	// Spacing modifier letters.
	'ˆ': "circ", '˜': "tilde",

	// Greek.
	'Α': "Alpha", 'Β': "Beta", 'Γ': "Gamma", 'Δ': "Delta", 'Ε': "Epsilon",
	'Ζ': "Zeta", 'Η': "Eta", 'Θ': "Theta", 'Ι': "Iota", 'Κ': "Kappa",
	'Λ': "Lambda", 'Μ': "Mu", 'Ν': "Nu", 'Ξ': "Xi", 'Ο': "Omicron",
	'Π': "Pi", 'Ρ': "Rho", 'Σ': "Sigma", 'Τ': "Tau", 'Υ': "Upsilon",
	'Φ': "Phi", 'Χ': "Chi", 'Ψ': "Psi", 'Ω': "Omega",
	'α': "alpha", 'β': "beta", 'γ': "gamma", 'δ': "delta", 'ε': "epsilon",
	'ζ': "zeta", 'η': "eta", 'θ': "theta", 'ι': "iota", 'κ': "kappa",
	'λ': "lambda", 'μ': "mu", 'ν': "nu", 'ξ': "xi", 'ο': "omicron",
	'π': "pi", 'ρ': "rho", 'ς': "sigmaf", 'σ': "sigma", 'τ': "tau",
	'υ': "upsilon", 'φ': "phi", 'χ': "chi", 'ψ': "psi", 'ω': "omega",
	'ϑ': "thetasym", 'ϒ': "upsih", 'ϖ': "piv",

	// General punctuation.
	' ': "ensp", ' ': "emsp", ' ': "thinsp",
	'‌': "zwnj", '‍': "zwj", '‎': "lrm", '‏': "rlm",
	'–': "ndash", '—': "mdash", '‘': "lsquo", '’': "rsquo", '‚': "sbquo",
	'“': "ldquo", '”': "rdquo", '„': "bdquo", '†': "dagger", '‡': "Dagger",
	'•': "bull", '…': "hellip", '‰': "permil", '′': "prime", '″': "Prime",
	'‹': "lsaquo", '›': "rsaquo", '‾': "oline", '⁄': "frasl", '€': "euro",

	// Letterlike symbols, arrows, mathematical operators and shapes.
	'ℑ': "image", '℘': "weierp", 'ℜ': "real", '™': "trade", 'ℵ': "alefsym",
	'←': "larr", '↑': "uarr", '→': "rarr", '↓': "darr", '↔': "harr",
	'↵': "crarr", '⇐': "lArr", '⇑': "uArr", '⇒': "rArr", '⇓': "dArr",
	'⇔': "hArr",
	'∀': "forall", '∂': "part", '∃': "exist", '∅': "empty", '∇': "nabla",
	'∈': "isin", '∉': "notin", '∋': "ni", '∏': "prod", '∑': "sum",
	'−': "minus", '∗': "lowast", '√': "radic", '∝': "prop", '∞': "infin",
	'∠': "ang", '∧': "and", '∨': "or", '∩': "cap", '∪': "cup",
	'∫': "int", '∴': "there4", '∼': "sim", '≅': "cong", '≈': "asymp",
	'≠': "ne", '≡': "equiv", '≤': "le", '≥': "ge", '⊂': "sub",
	'⊃': "sup", '⊄': "nsub", '⊆': "sube", '⊇': "supe", '⊕': "oplus",
	'⊗': "otimes", '⊥': "perp", '⋅': "sdot",
	'⌈': "lceil", '⌉': "rceil", '⌊': "lfloor", '⌋': "rfloor",
	'〈': "lang", '〉': "rang", '◊': "loz",
	'♠': "spades", '♣': "clubs", '♥': "hearts", '♦': "diams",
}

// CleanRateLimiterKey folds a key down to the characters a counter can be named
// by.
//
// It answers RateLimiter::cleanRateLimiterKey(), which is
// preg_replace('/&([a-z])[a-z]+;/i', '$1', htmlentities($key)) -- two steps that
// look like escaping and are not. What they do is fold an accented letter to its
// base one: htmlentities turns "é" into "&eacute;" and the pattern keeps the "e".
// A rate limit on a name spelled with and without its accent is one limit, which
// is the point.
//
// The details are worth stating, because they are the kind that get lost:
//
//   - "&" folds to "a", because its entity is "&amp;". So do "<", ">" and the
//     double quote, to "l", "g" and "q". They are not escaped, they are
//     replaced by a letter.
//   - The apostrophe does not fold. It has no named entity, only the numeric
//     "&#039;", and the pattern only matches letters -- so it survives as
//     "&#039;", six characters where there was one.
//   - Neither do "&sup2;", "&frac12;" and the other entities with a digit in the
//     name, for the same reason.
//   - A character with no entity in the table -- anything past Latin-1 that is
//     not one of the symbols HTML 4.01 names -- is left exactly as it was.
//
// It is not a security boundary and it is not an escape. It is a fold, and the
// only thing it protects is one counter from being two.
func (rl *RateLimiter) CleanRateLimiterKey(key string) string {
	var out strings.Builder
	out.Grow(len(key))

	for _, r := range key {
		if r == '\'' {
			// The apostrophe has no named entity, only the numeric &#039;, and
			// the pattern matches only letters, so it comes out the far side as
			// six characters.
			out.WriteString("&#039;")
			continue
		}

		name, ok := htmlEntity[r]
		if !ok {
			out.WriteRune(r)
			continue
		}
		if folded, ok := foldEntity(name); ok {
			out.WriteString(folded)
			continue
		}
		// The entity is written out in full, and the pattern leaves it alone.
		out.WriteByte('&')
		out.WriteString(name)
		out.WriteByte(';')
	}
	return out.String()
}

// foldEntity applies /&([a-z])[a-z]+;/i to one entity name: it matches when the
// name is two or more letters and nothing else, and what survives is the first
// of them.
func foldEntity(name string) (string, bool) {
	if len(name) < 2 {
		return "", false
	}
	for i := range len(name) {
		c := name[i]
		if (c < 'a' || c > 'z') && (c < 'A' || c > 'Z') {
			return "", false
		}
	}
	return name[:1], true
}
