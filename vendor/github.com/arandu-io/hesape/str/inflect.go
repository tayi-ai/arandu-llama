package str

import (
	"errors"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"unicode"

	"github.com/arandu-io/hesape/number"
)

// Plural is the plural of an English noun -- "purchase_order" becomes
// "purchase_orders", "person" becomes "people", "knife" becomes "knives".
//
// The optional count defaults to two. A count of one in either sign returns the
// word untouched, so Plural("rule", 1) is "rule".
//
// It pluralizes the last word and leaves the rest alone, so a compound name
// keeps its shape: "sales_person" is "sales_people", not "sales_persons". The
// case of that word is carried over -- "Invoice" is "Invoices" and "INVOICE" is
// "INVOICES" -- because this names a table in one call site and a heading in
// the next.
//
// A value that does not end in a letter, a digit or a non-ASCII rune is not a
// word and is returned untouched: "rule!" and "rule " stay as they are.
//
// English pluralization has hundreds of exceptions and this handles the ones a
// schema meets. Everything the tables in this file do not name takes -s or -es
// by rule. A word that comes out wrong is a line to add there, not a rule to
// change.
func Plural(s string, count ...int) string {
	if len(count) > 0 && (count[0] == 1 || count[0] == -1) {
		return s
	}
	if pluralizerUncountable[strings.ToLower(s)] || !endsInflectable(s) {
		return s
	}
	head, word := splitTail(s)
	if word == "" {
		return s
	}
	return head + matchCase(word, pluralize(strings.ToLower(word)))
}

// PluralStudly pluralizes the last word of a studly caps string and leaves
// everything in front of it alone: PluralStudly("VerifiedHuman") is
// "VerifiedHumans".
//
// That last word is found by splitting in front of every capital, which is why
// PluralStudly("HTTPServer") is "HTTPServers" and not "HTTPServer" with the run
// of capitals pluralized.
func PluralStudly(value string, count ...int) string {
	head, word := splitBeforeUpper(value)
	return head + Plural(word, count...)
}

// PluralPascal pluralizes the last word of a Pascal case string and leaves
// everything in front of it alone: PluralPascal("VerifiedHuman") is
// "VerifiedHumans". That last word is found by splitting in front of every
// capital, so PluralPascal("HTTPServer") is "HTTPServers". The optional count
// behaves as in [Plural]: a count of one returns the value untouched.
//
// It is [PluralStudly] under the other name for the same convention, and the
// two are interchangeable in every call.
func PluralPascal(value string, count ...int) string { return PluralStudly(value, count...) }

// Counted is the count and the noun agreeing with it: Counted("rule", 1) is
// "1 rule" and Counted("rule", 3) is "3 rules".
//
// The count goes through number.Format, so Counted("row", 1000) is
// "1,000 rows".
func Counted(value string, count int) string {
	return number.Format(float64(count)) + " " + Plural(value, count)
}

// PluralN is the old name of Counted, with its arguments the other way around.
//
// Deprecated: call Counted instead. It stays because callers outside this
// package still name it.
func PluralN(n int, singular string) string {
	return strconv.Itoa(n) + " " + Plural(singular, n)
}

// Singular is the singular of an English noun: "purchase_orders" becomes
// "purchase_order", "people" becomes "person", "knives" becomes "knife".
//
// It is Plural read backwards, with the same tables and the same treatment of
// compounds and case. English does not invert cleanly -- "axes" is the plural of
// both "axe" and "axis", and "-ses" is "-se" in "houses" and "-s" in "buses" --
// so where the ending is ambiguous this answers with the commoner word and the
// table in this file holds the corrections.
func Singular(s string) string {
	head, word := splitTail(s)
	if word == "" {
		return s
	}
	return head + matchCase(word, singularize(strings.ToLower(word)))
}

// pluralizerUncountable is the list of non-noun word forms the inflector must
// not touch. It is checked by Plural only.
var pluralizerUncountable = map[string]bool{
	"recommended": true,
	"related":     true,
}

// ErrLanguageNotSupported is returned by UseLanguage for a language this
// package carries no inflection rules for.
var ErrLanguageNotSupported = errors.New("str: unsupported inflector language")

// languageMu guards the inflector language, which UseLanguage may be called
// from any goroutine to set.
var languageMu sync.RWMutex

// language is the language the inflector works in, and it starts as "english".
var language = "english"

// UseLanguage names the language the inflector works in.
//
// English is the only language this package carries rules for, so it is the
// only one accepted, and anything else returns ErrLanguageNotSupported. Plural
// and Singular would otherwise keep answering in English under another
// language's name, which is worse than refusing.
func UseLanguage(lang string) error {
	if !strings.EqualFold(lang, "english") {
		return fmt.Errorf("%w: %q", ErrLanguageNotSupported, lang)
	}
	languageMu.Lock()
	language = strings.ToLower(lang)
	languageMu.Unlock()
	return nil
}

// Language reports the language UseLanguage last accepted.
func Language() string {
	languageMu.RLock()
	defer languageMu.RUnlock()
	return language
}

// inflectableTail is Plural's guard: a value that does not end in a letter, a
// digit or a rune past ASCII is not a word and is returned as it stands.
var inflectableTail = regexp.MustCompile(`[A-Za-z0-9\x{0080}-\x{FFFF}]$`)

func endsInflectable(s string) bool { return inflectableTail.MatchString(s) }

// splitBeforeUpper cuts a value at its last capital: everything in front of it,
// and the word starting at it.
func splitBeforeUpper(value string) (head, word string) {
	rs := []rune(value)
	for i := len(rs) - 1; i > 0; i-- {
		if unicode.IsUpper(rs[i]) {
			return string(rs[:i]), string(rs[i:])
		}
	}
	return "", value
}

// splitTail cuts a name into everything before the last word and the last word
// itself. Only the last word inflects: "purchase_order" pluralizes "order" and
// "PurchaseOrder" pluralizes "Order".
//
// The boundary is a separator or a change of case, which is what keeps the
// answer spelled the way the argument was -- PurchaseOrders, not
// Purchaseorders.
func splitTail(s string) (head, word string) {
	rs := []rune(s)
	for i := len(rs) - 1; i >= 0; i-- {
		if isSeparator(rs[i]) {
			return string(rs[:i+1]), string(rs[i+1:])
		}
		if i == 0 || !unicode.IsUpper(rs[i]) {
			continue
		}
		prevWord := unicode.IsLower(rs[i-1]) || unicode.IsDigit(rs[i-1])
		runEnd := unicode.IsUpper(rs[i-1]) && i+1 < len(rs) && unicode.IsLower(rs[i+1])
		if prevWord || runEnd {
			return string(rs[:i]), string(rs[i:])
		}
	}
	return "", s
}

// matchCase spells out the way the word it replaces was spelled: all capitals
// stay all capitals, an initial capital stays an initial capital.
func matchCase(was, now string) string {
	rs := []rune(was)
	if len(rs) == 0 || now == "" {
		return now
	}
	if !unicode.IsUpper(rs[0]) {
		return now
	}
	allUpper := true
	for _, r := range rs {
		if unicode.IsLetter(r) && !unicode.IsUpper(r) {
			allUpper = false
			break
		}
	}
	if allUpper && len(rs) > 1 {
		return strings.ToUpper(now)
	}
	return Ucfirst(now)
}

func pluralize(w string) string {
	if uncountable[w] {
		return w
	}
	if p, ok := irregularPlural[w]; ok {
		return p
	}
	if p, ok := fToVes[w]; ok {
		return p
	}
	if p, ok := oToOes[w]; ok {
		return p
	}
	switch {
	case hasAnySuffix(w, "s", "x", "z", "ch", "sh"):
		return w + "es"
	case strings.HasSuffix(w, "y") && len(w) > 1 && !isVowel(w[len(w)-2]):
		return w[:len(w)-1] + "ies"
	default:
		return w + "s"
	}
}

func singularize(w string) string {
	if uncountable[w] {
		return w
	}
	if s, ok := irregularSingular[w]; ok {
		return s
	}
	switch {
	case strings.HasSuffix(w, "ies") && len(w) > 4:
		return w[:len(w)-3] + "y"
	case hasAnySuffix(w, "sses", "xes", "zes", "ches", "shes"):
		return w[:len(w)-2]
	case strings.HasSuffix(w, "ses"):
		// "houses" over "buses": a word ending -se is the commoner shape, and
		// the ones that are not are entries in irregularSingular.
		return w[:len(w)-1]
	case strings.HasSuffix(w, "ss"), hasAnySuffix(w, "us", "is"):
		return w
	case strings.HasSuffix(w, "s") && len(w) > 1:
		return w[:len(w)-1]
	default:
		return w
	}
}

func hasAnySuffix(w string, suffixes ...string) bool {
	for _, s := range suffixes {
		if strings.HasSuffix(w, s) {
			return true
		}
	}
	return false
}

func isVowel(b byte) bool { return strings.IndexByte("aeiou", b) >= 0 }

// uncountable is a noun that is its own plural. A word here is never touched by
// either direction.
var uncountable = set(
	"aircraft", "audio", "bison", "chassis", "compensation", "coreopsis",
	"deer", "education", "emoji", "equipment", "evidence", "feedback",
	"firmware", "fish", "furniture", "gold", "hardware", "information",
	"jedi", "kin", "knowledge", "love", "metadata", "money", "moose", "news",
	"nutrition", "offspring", "police", "rain", "rice", "salmon", "series",
	"sheep", "software", "species", "staff", "swine", "traffic", "trout",
	"wheat",
)

// irregularPlural is the plural that no rule produces.
var irregularPlural = map[string]string{
	"analysis":   "analyses",
	"appendix":   "appendices",
	"axis":       "axes",
	"basis":      "bases",
	"cactus":     "cacti",
	"child":      "children",
	"crisis":     "crises",
	"criterion":  "criteria",
	"datum":      "data",
	"diagnosis":  "diagnoses",
	"die":        "dice",
	"focus":      "foci",
	"foot":       "feet",
	"fungus":     "fungi",
	"goose":      "geese",
	"index":      "indices",
	"louse":      "lice",
	"man":        "men",
	"matrix":     "matrices",
	"medium":     "media",
	"memorandum": "memoranda",
	"mouse":      "mice",
	"nucleus":    "nuclei",
	"ox":         "oxen",
	"person":     "people",
	"phenomenon": "phenomena",
	"quiz":       "quizzes",
	"radius":     "radii",
	"thesis":     "theses",
	"tooth":      "teeth",
	"vertex":     "vertices",
	"woman":      "women",
}

// fToVes is the -f and -fe words that take -ves. An allowlist and not a rule,
// because the rule turns "cafe" into "caves" and "roof" into "rooves".
var fToVes = map[string]string{
	"calf":  "calves",
	"elf":   "elves",
	"half":  "halves",
	"hoof":  "hooves",
	"knife": "knives",
	"leaf":  "leaves",
	"life":  "lives",
	"loaf":  "loaves",
	"self":  "selves",
	"sheaf": "sheaves",
	"shelf": "shelves",
	"thief": "thieves",
	"wharf": "wharves",
	"wife":  "wives",
	"wolf":  "wolves",
}

// oToOes is the -o words that take -oes. An allowlist for the same reason:
// "photo" and "piano" take -s, and they are the majority.
var oToOes = map[string]string{
	"buffalo":  "buffaloes",
	"echo":     "echoes",
	"embargo":  "embargoes",
	"hero":     "heroes",
	"mosquito": "mosquitoes",
	"potato":   "potatoes",
	"tomato":   "tomatoes",
	"torpedo":  "torpedoes",
	"veto":     "vetoes",
	"volcano":  "volcanoes",
}

// irregularSingular is every table above read backwards, plus the -es plurals
// whose singular ends in -s and would otherwise lose a letter.
var irregularSingular = reverse(
	irregularPlural, fToVes, oToOes,
	map[string]string{
		"atlas":  "atlases",
		"bias":   "biases",
		"bonus":  "bonuses",
		"bus":    "buses",
		"campus": "campuses",
		"gas":    "gases",
		"lens":   "lenses",
		"plus":   "pluses",
		"status": "statuses",
		"virus":  "viruses",
	},
)

func set(words ...string) map[string]bool {
	out := make(map[string]bool, len(words))
	for _, w := range words {
		out[w] = true
	}
	return out
}

func reverse(tables ...map[string]string) map[string]string {
	out := make(map[string]string)
	for _, t := range tables {
		for singular, plural := range t {
			out[plural] = singular
		}
	}
	return out
}
