package str

import (
	"fmt"
	"regexp"
	"strings"
	"time"
)

// Stringable is a string carried through a chain of the transformations in this
// package, where every step that produces a string produces another Stringable.
//
//	str.Of("  Purchase Order  ").Trim().Snake("_").ToString() // "purchase_order"
//
// It is a value, not a handle: every method returns a new Stringable and none
// of them changes the one it was called on, so a Stringable can be kept, passed
// and reused without anyone else's chain reaching it.
//
// The methods that end a chain -- Length, Contains, ToString and the rest --
// return the Go type the answer is, not a Stringable.
type Stringable struct {
	value string
}

// Of wraps a string as a Stringable, which is where a chain begins.
func Of(value string) Stringable { return Stringable{value: value} }

// String is the string itself. It makes Stringable an fmt.Stringer, which is
// what makes a Stringable print as its own text.
func (s Stringable) String() string { return s.value }

// ToString is the string itself.
func (s Stringable) ToString() string { return s.value }

// Value is the string itself. It is ToString under the other name.
func (s Stringable) Value() string { return s.value }

// After is everything past the first occurrence of search.
func (s Stringable) After(search string) Stringable { return Of(After(s.value, search)) }

// AfterLast is everything past the last occurrence of search.
func (s Stringable) AfterLast(search string) Stringable { return Of(AfterLast(s.value, search)) }

// Append puts the values on the end, in order.
func (s Stringable) Append(values ...string) Stringable {
	return Of(s.value + strings.Join(values, ""))
}

// Prepend puts the values on the front, in order.
func (s Stringable) Prepend(values ...string) Stringable {
	return Of(strings.Join(values, "") + s.value)
}

// NewLine appends count line breaks; passing none appends one.
//
// The break is "\n" on every platform.
func (s Stringable) NewLine(count ...int) Stringable {
	n := 1
	if len(count) > 0 {
		n = count[0]
	}
	return s.Append(Repeat("\n", n))
}

// ASCII folds the string to ASCII, dropping the runes the fold table cannot
// spell; see the comment on ASCII.
func (s Stringable) ASCII() Stringable { return Of(ASCII(s.value)) }

// Transliterate writes the string in ASCII, putting unknown in the place of
// every rune it has no ASCII spelling for.
func (s Stringable) Transliterate(unknown string, strict bool) Stringable {
	return Of(Transliterate(s.value, unknown, strict))
}

// Basename is the last component of a path, with the suffix removed when one is
// given and the component is more than the suffix itself.
func (s Stringable) Basename(suffix ...string) Stringable {
	name := strings.TrimRight(s.value, "/")
	if i := strings.LastIndexByte(name, '/'); i >= 0 {
		name = name[i+1:]
	}
	if len(suffix) > 0 && suffix[0] != "" && name != suffix[0] && strings.HasSuffix(name, suffix[0]) {
		name = name[:len(name)-len(suffix[0])]
	}
	return Of(name)
}

// Dirname is the path with its last component removed, levels times over;
// passing none removes one.
func (s Stringable) Dirname(levels ...int) Stringable {
	n := 1
	if len(levels) > 0 {
		n = levels[0]
	}
	dir := s.value
	for range n {
		dir = phpDirname(dir)
	}
	return Of(dir)
}

// ClassBasename is the part of a fully qualified class name after the last
// namespace separator, which is read as either a backslash or a slash.
func (s Stringable) ClassBasename() Stringable {
	return Of(AfterLast(strings.ReplaceAll(s.value, `\`, "/"), "/"))
}

// Before is everything up to the first occurrence of search.
func (s Stringable) Before(search string) Stringable { return Of(Before(s.value, search)) }

// BeforeLast is everything up to the last occurrence of search.
func (s Stringable) BeforeLast(search string) Stringable { return Of(BeforeLast(s.value, search)) }

// Between is the largest portion sitting after the first from and before the
// last to.
func (s Stringable) Between(from, to string) Stringable { return Of(Between(s.value, from, to)) }

// BetweenFirst is the smallest portion sitting after the first from and before
// the first to.
func (s Stringable) BetweenFirst(from, to string) Stringable {
	return Of(BetweenFirst(s.value, from, to))
}

// Camel converts the string to camel case.
func (s Stringable) Camel() Stringable { return Of(Camel(s.value)) }

// Kebab converts the string to kebab case.
func (s Stringable) Kebab() Stringable { return Of(Kebab(s.value)) }

// Snake converts the string to snake case with the given delimiter.
func (s Stringable) Snake(delimiter string) Stringable { return Of(Snake(s.value, delimiter)) }

// Studly converts the string to studly caps.
func (s Stringable) Studly() Stringable { return Of(Studly(s.value)) }

// Pascal converts the string to Pascal case. It is Studly under the other name.
func (s Stringable) Pascal() Stringable { return Of(Pascal(s.value)) }

// Title uppercases the first letter of every word and lowercases the rest of
// each one.
func (s Stringable) Title() Stringable { return Of(Title(s.value)) }

// Headline converts the string to a title-cased sentence.
func (s Stringable) Headline() Stringable { return Of(Headline(s.value)) }

// Apa converts the string to APA-style title case.
func (s Stringable) Apa() Stringable { return Of(Apa(s.value)) }

// Lower lowercases the whole string.
func (s Stringable) Lower() Stringable { return Of(strings.ToLower(s.value)) }

// Upper uppercases the whole string.
func (s Stringable) Upper() Stringable { return Of(strings.ToUpper(s.value)) }

// Ucfirst uppercases the first character and leaves the rest alone.
func (s Stringable) Ucfirst() Stringable { return Of(Ucfirst(s.value)) }

// Lcfirst lowercases the first character and leaves the rest alone.
func (s Stringable) Lcfirst() Stringable { return Of(Lcfirst(s.value)) }

// Ucwords uppercases the first letter of the string and every lowercase letter
// that follows one of the separators; passing none uses space, tab and the line
// breaks.
func (s Stringable) Ucwords(separators ...string) Stringable {
	return Of(Ucwords(s.value, separators...))
}

// Ucsplit splits the string in front of every uppercase letter, dropping the
// empty pieces.
func (s Stringable) Ucsplit() []string { return Ucsplit(s.value) }

// ConvertCase converts the string in the given mode, which is one of CaseUpper,
// CaseLower, CaseTitle or CaseFold.
func (s Stringable) ConvertCase(mode int) Stringable { return Of(ConvertCase(s.value, mode)) }

// CharAt is the character at the given index, counting back from the end when
// the index is negative. The second result is false for an index outside the
// string.
func (s Stringable) CharAt(index int) (string, bool) { return CharAt(s.value, index) }

// ChopStart removes the first of the given needles that the string starts with.
func (s Stringable) ChopStart(needle ...string) Stringable {
	return Of(ChopStart(s.value, needle...))
}

// ChopEnd removes the first of the given needles that the string ends with.
func (s Stringable) ChopEnd(needle ...string) Stringable { return Of(ChopEnd(s.value, needle...)) }

// Contains reports whether the string holds any one of the needles.
func (s Stringable) Contains(needles []string, ignoreCase bool) bool {
	return Contains(s.value, needles, ignoreCase)
}

// ContainsAll reports whether the string holds every one of the needles.
func (s Stringable) ContainsAll(needles []string, ignoreCase bool) bool {
	return ContainsAll(s.value, needles, ignoreCase)
}

// DoesntContain is Contains negated.
func (s Stringable) DoesntContain(needles []string, ignoreCase bool) bool {
	return DoesntContain(s.value, needles, ignoreCase)
}

// StartsWith reports whether the string begins with any one of the needles.
func (s Stringable) StartsWith(needles ...string) bool { return StartsWith(s.value, needles...) }

// EndsWith reports whether the string ends with any one of the needles.
func (s Stringable) EndsWith(needles ...string) bool { return EndsWith(s.value, needles...) }

// DoesntStartWith is StartsWith negated.
func (s Stringable) DoesntStartWith(needles ...string) bool {
	return DoesntStartWith(s.value, needles...)
}

// DoesntEndWith is EndsWith negated.
func (s Stringable) DoesntEndWith(needles ...string) bool {
	return DoesntEndWith(s.value, needles...)
}

// Deduplicate collapses a run of the given character to one occurrence of it;
// passing none deduplicates spaces.
func (s Stringable) Deduplicate(character ...string) Stringable {
	return Of(Deduplicate(s.value, character...))
}

// Exactly reports whether the string equals value.
func (s Stringable) Exactly(value string) bool { return s.value == value }

// Excerpt cuts an excerpt out of the string around the first case-insensitive
// occurrence of phrase, keeping radius characters on each side and marking a
// cut end with omission. The second result is false when the phrase is not
// there.
func (s Stringable) Excerpt(phrase string, radius int, omission string) (string, bool) {
	return Excerpt(s.value, phrase, radius, omission)
}

// Explode splits the string on the delimiter, and returns nothing when the
// delimiter is empty.
//
// A positive limit leaves the rest of the string in the last piece, a negative
// one drops that many pieces off the end, and zero is read as one.
func (s Stringable) Explode(delimiter string, limit ...int) []string {
	if delimiter == "" {
		return nil
	}
	if len(limit) == 0 {
		return strings.Split(s.value, delimiter)
	}
	n := limit[0]
	switch {
	case n > 0:
		return strings.SplitN(s.value, delimiter, n)
	case n == 0:
		return strings.SplitN(s.value, delimiter, 1)
	default:
		all := strings.Split(s.value, delimiter)
		if kept := len(all) + n; kept > 0 {
			return all[:kept]
		}
		return nil
	}
}

// Split cuts the string into pieces. An int pattern makes pieces that many
// characters wide, and a *regexp.Regexp or a string splits on the matches;
// anything else returns nothing.
//
// The optional limit is the greatest number of pieces to produce, and a value
// below one means no limit.
func (s Stringable) Split(pattern any, limit ...int) []string {
	n := -1
	if len(limit) > 0 && limit[0] > 0 {
		n = limit[0]
	}
	switch p := pattern.(type) {
	case int:
		return chunkRunes(s.value, p)
	case *regexp.Regexp:
		return p.Split(s.value, n)
	case string:
		re, err := regexp.Compile(p)
		if err != nil {
			return nil
		}
		return re.Split(s.value, n)
	default:
		return nil
	}
}

// Scan reads the fields a format names out of the string.
//
// The format takes %s for a run of non-space characters, %d for an integer, %f
// for a decimal, %c for one character, %% for a literal per cent, whitespace
// for any run of whitespace, and anything else for itself. A field that does
// not match ends the scan, and what was read up to there is the answer.
func (s Stringable) Scan(format string) []string { return sscanf(s.value, format) }

// Finish caps the string with a single instance of cap.
func (s Stringable) Finish(cap string) Stringable { return Of(Finish(s.value, cap)) }

// Start begins the string with a single instance of prefix.
func (s Stringable) Start(prefix string) Stringable { return Of(Start(s.value, prefix)) }

// Wrap puts before in front of the string and after behind it; passing no after
// repeats before.
func (s Stringable) Wrap(before string, after ...string) Stringable {
	return Of(Wrap(s.value, before, after...))
}

// Unwrap removes before from the front of the string and after from the back,
// each only if it is there; passing no after repeats before.
func (s Stringable) Unwrap(before string, after ...string) Stringable {
	return Of(Unwrap(s.value, before, after...))
}

// Is reports whether the string matches any one of the patterns, where *
// stands for any run of characters, including none.
func (s Stringable) Is(pattern []string, ignoreCase bool) bool {
	return Is(pattern, s.value, ignoreCase)
}

// IsASCII reports whether every byte of the string is below 128.
func (s Stringable) IsASCII() bool { return IsASCII(s.value) }

// IsJSON reports whether the string parses as JSON.
func (s Stringable) IsJSON() bool { return IsJSON(s.value) }

// IsURL reports whether the string is an absolute URL whose scheme is one of
// the protocols given.
func (s Stringable) IsURL(protocols ...string) bool { return IsURL(s.value, protocols...) }

// IsUUID reports whether the string has the shape of a hyphenated UUID.
func (s Stringable) IsUUID() bool { return IsUUID(s.value) }

// IsULID reports whether the string has the shape of a ULID.
func (s Stringable) IsULID() bool { return IsULID(s.value) }

// IsEmpty reports whether the string is empty.
func (s Stringable) IsEmpty() bool { return s.value == "" }

// IsNotEmpty is IsEmpty negated.
func (s Stringable) IsNotEmpty() bool { return !s.IsEmpty() }

// Length is the number of characters in the string, not the number of bytes.
func (s Stringable) Length() int { return Length(s.value) }

// Limit cuts the string to at most limit display columns and appends end when
// it had to cut.
func (s Stringable) Limit(limit int, end string, preserveWords bool) Stringable {
	return Of(Limit(s.value, limit, end, preserveWords))
}

// Words cuts the string to at most words words and appends end when it had to
// cut.
func (s Stringable) Words(words int, end string) Stringable {
	return Of(Words(s.value, words, end))
}

// WordCount counts the words in the string, where a word is a run of letters
// and of the extra characters given.
func (s Stringable) WordCount(characters ...string) int {
	return WordCount(s.value, characters...)
}

// WordWrap breaks the string into lines of at most the given number of
// characters, writing brk where it breaks.
func (s Stringable) WordWrap(characters int, brk string, cutLongWords bool) Stringable {
	return Of(WordWrap(s.value, characters, brk, cutLongWords))
}

// Markdown renders the string as HTML.
func (s Stringable) Markdown() Stringable { return Of(Markdown(s.value)) }

// InlineMarkdown renders only the inline part of the string as HTML.
func (s Stringable) InlineMarkdown() Stringable { return Of(InlineMarkdown(s.value)) }

// Mask replaces a run of the string, starting at index, with the first
// character of character repeated.
func (s Stringable) Mask(character string, index int, length ...int) Stringable {
	return Of(Mask(s.value, character, index, length...))
}

// Match is the first capture group of the first match, or the whole match when
// the pattern captures nothing.
func (s Stringable) Match(pattern *regexp.Regexp) Stringable {
	return Of(Match(pattern, s.value))
}

// MatchAll is the first capture group of every match, or every whole match when
// the pattern captures nothing.
func (s Stringable) MatchAll(pattern *regexp.Regexp) []string { return MatchAll(pattern, s.value) }

// IsMatch reports whether the string matches any one of the patterns.
func (s Stringable) IsMatch(patterns []*regexp.Regexp) bool { return IsMatch(patterns, s.value) }

// Test is IsMatch under the other name.
func (s Stringable) Test(patterns []*regexp.Regexp) bool { return s.IsMatch(patterns) }

// ReplaceMatches replaces at most limit matches of the pattern, and every one
// of them when limit is negative.
func (s Stringable) ReplaceMatches(pattern *regexp.Regexp, replace string, limit int) Stringable {
	return Of(ReplaceMatches(pattern, replace, s.value, limit))
}

// Numbers keeps the digits of the string and drops everything else.
func (s Stringable) Numbers() Stringable { return Of(Numbers(s.value)) }

// PadBoth grows the string to the given number of characters, splitting the
// padding between the two ends.
func (s Stringable) PadBoth(length int, pad ...string) Stringable {
	return Of(PadBoth(s.value, length, pad...))
}

// PadLeft grows the string to the given number of characters by putting pad in
// front of it.
func (s Stringable) PadLeft(length int, pad ...string) Stringable {
	return Of(PadLeft(s.value, length, pad...))
}

// PadRight grows the string to the given number of characters by putting pad
// behind it.
func (s Stringable) PadRight(length int, pad ...string) Stringable {
	return Of(PadRight(s.value, length, pad...))
}

// ParseCallback splits a "Class@method" string into its two halves, and answers
// the whole string with the fallback when there is no at sign in it.
func (s Stringable) ParseCallback(fallback string) (string, string) {
	return ParseCallback(s.value, fallback)
}

// Pipe hands the Stringable to the callback and wraps what comes back.
func (s Stringable) Pipe(callback func(Stringable) string) Stringable { return Of(callback(s)) }

// Plural is the plural of the English noun the string holds.
func (s Stringable) Plural(count ...int) Stringable { return Of(Plural(s.value, count...)) }

// PluralStudly pluralizes the last word of a studly caps string and leaves
// everything in front of it alone.
func (s Stringable) PluralStudly(count ...int) Stringable {
	return Of(PluralStudly(s.value, count...))
}

// PluralPascal pluralizes the last word of a Pascal case string. It is
// PluralStudly under the other name.
func (s Stringable) PluralPascal(count ...int) Stringable {
	return Of(PluralPascal(s.value, count...))
}

// Singular is the singular of the English noun the string holds.
func (s Stringable) Singular() Stringable { return Of(Singular(s.value)) }

// Counted is the count and the noun agreeing with it.
func (s Stringable) Counted(count int) Stringable { return Of(Counted(s.value, count)) }

// Position is the character index of the first occurrence of needle at or after
// offset, and -1 when the needle is absent.
func (s Stringable) Position(needle string, offset int) int {
	return Position(s.value, needle, offset)
}

// Remove deletes every occurrence of each search from the string.
func (s Stringable) Remove(search []string, caseSensitive bool) Stringable {
	return Of(Remove(search, s.value, caseSensitive))
}

// Reverse reverses the string character by character.
func (s Stringable) Reverse() Stringable { return Of(Reverse(s.value)) }

// Repeat is the string written times over.
func (s Stringable) Repeat(times int) Stringable { return Of(Repeat(s.value, times)) }

// Replace replaces every occurrence of each search with the replacement
// standing at the same index.
func (s Stringable) Replace(search, replace []string, caseSensitive bool) Stringable {
	return Of(Replace(search, replace, s.value, caseSensitive))
}

// ReplaceArray walks the occurrences of search from left to right and spends
// one replacement on each.
func (s Stringable) ReplaceArray(search string, replace []string) Stringable {
	return Of(ReplaceArray(search, replace, s.value))
}

// ReplaceFirst replaces the first occurrence of search.
func (s Stringable) ReplaceFirst(search, replace string) Stringable {
	return Of(ReplaceFirst(search, replace, s.value))
}

// ReplaceLast replaces the last occurrence of search.
func (s Stringable) ReplaceLast(search, replace string) Stringable {
	return Of(ReplaceLast(search, replace, s.value))
}

// ReplaceStart replaces search only where it begins the string.
func (s Stringable) ReplaceStart(search, replace string) Stringable {
	return Of(ReplaceStart(search, replace, s.value))
}

// ReplaceEnd replaces search only where it ends the string.
func (s Stringable) ReplaceEnd(search, replace string) Stringable {
	return Of(ReplaceEnd(search, replace, s.value))
}

// Slug is the string as an address, with separator between the words.
func (s Stringable) Slug(separator string) Stringable { return Of(Slug(s.value, separator)) }

// Squish trims both ends and collapses every run of whitespace inside the
// string to a single space.
func (s Stringable) Squish() Stringable { return Of(Squish(s.value)) }

// StripTags removes everything between an opening angle bracket and the next
// closing one. No tag is allowed through.
func (s Stringable) StripTags() Stringable { return Of(stripTags(s.value)) }

// Substr is the portion of the string given by start and length, counted in
// characters.
func (s Stringable) Substr(start int, length ...int) Stringable {
	return Of(Substr(s.value, start, length...))
}

// SubstrCount counts the non-overlapping occurrences of needle from offset
// onwards, in bytes.
func (s Stringable) SubstrCount(needle string, offset int, length ...int) int {
	return SubstrCount(s.value, needle, offset, length...)
}

// SubstrReplace replaces the portion of the string given by offset and length,
// in bytes.
func (s Stringable) SubstrReplace(replace string, offset int, length ...int) Stringable {
	return Of(SubstrReplace(s.value, replace, offset, length...))
}

// Swap applies a whole map of replacements in one left-to-right pass, so
// nothing a replacement produces is replaced again.
func (s Stringable) Swap(swap map[string]string) Stringable { return Of(Swap(swap, s.value)) }

// Take is the first limit characters of the string, or the last limit
// characters when limit is negative.
func (s Stringable) Take(limit int) Stringable { return Of(Take(s.value, limit)) }

// Trim removes the given characters from both ends; passing none removes
// whitespace and the invisible marks.
func (s Stringable) Trim(characters ...string) Stringable {
	return Of(Trim(s.value, characters...))
}

// Ltrim is Trim on the left side only.
func (s Stringable) Ltrim(characters ...string) Stringable {
	return Of(Ltrim(s.value, characters...))
}

// Rtrim is Trim on the right side only.
func (s Stringable) Rtrim(characters ...string) Stringable {
	return Of(Rtrim(s.value, characters...))
}

// ToBase64 encodes the string as standard base64 with padding.
func (s Stringable) ToBase64() Stringable { return Of(ToBase64(s.value)) }

// FromBase64 decodes standard base64. What does not decode becomes the empty
// string.
func (s Stringable) FromBase64(strict bool) Stringable {
	decoded, _ := FromBase64(s.value, strict)
	return Of(decoded)
}

// Initials keeps the first character of every whitespace-separated part.
func (s Stringable) Initials(capitalize bool) Stringable {
	return Of(Initials(s.value, capitalize))
}

// When runs the callback when the condition holds, and the optional second
// callback when it does not, and hands back whatever comes out.
func (s Stringable) When(condition bool, callback func(Stringable) Stringable, otherwise ...func(Stringable) Stringable) Stringable {
	if condition {
		if callback != nil {
			return callback(s)
		}
		return s
	}
	if len(otherwise) > 0 && otherwise[0] != nil {
		return otherwise[0](s)
	}
	return s
}

// Unless is When with the condition read the other way.
func (s Stringable) Unless(condition bool, callback func(Stringable) Stringable, otherwise ...func(Stringable) Stringable) Stringable {
	return s.When(!condition, callback, otherwise...)
}

// Tap hands the Stringable to the callback and returns it unchanged.
func (s Stringable) Tap(callback func(Stringable)) Stringable {
	if callback != nil {
		callback(s)
	}
	return s
}

// Dump writes the value out and returns the Stringable so a chain carries on
// through it.
func (s Stringable) Dump(args ...any) Stringable {
	fmt.Println(append([]any{s.value}, args...)...)
	return s
}

// WhenContains is When on whether the string holds any one of the needles.
func (s Stringable) WhenContains(needles []string, callback func(Stringable) Stringable, otherwise ...func(Stringable) Stringable) Stringable {
	return s.When(s.Contains(needles, false), callback, otherwise...)
}

// WhenContainsAll is When on whether the string holds every one of the needles.
func (s Stringable) WhenContainsAll(needles []string, callback func(Stringable) Stringable, otherwise ...func(Stringable) Stringable) Stringable {
	return s.When(s.ContainsAll(needles, false), callback, otherwise...)
}

// WhenEmpty is When on whether the string is empty.
func (s Stringable) WhenEmpty(callback func(Stringable) Stringable, otherwise ...func(Stringable) Stringable) Stringable {
	return s.When(s.IsEmpty(), callback, otherwise...)
}

// WhenNotEmpty is When on whether the string is not empty.
func (s Stringable) WhenNotEmpty(callback func(Stringable) Stringable, otherwise ...func(Stringable) Stringable) Stringable {
	return s.When(s.IsNotEmpty(), callback, otherwise...)
}

// WhenStartsWith is When on whether the string begins with any one of the
// needles.
func (s Stringable) WhenStartsWith(needles []string, callback func(Stringable) Stringable, otherwise ...func(Stringable) Stringable) Stringable {
	return s.When(s.StartsWith(needles...), callback, otherwise...)
}

// WhenEndsWith is When on whether the string ends with any one of the needles.
func (s Stringable) WhenEndsWith(needles []string, callback func(Stringable) Stringable, otherwise ...func(Stringable) Stringable) Stringable {
	return s.When(s.EndsWith(needles...), callback, otherwise...)
}

// WhenDoesntStartWith is When on whether the string begins with none of the
// needles.
func (s Stringable) WhenDoesntStartWith(needles []string, callback func(Stringable) Stringable, otherwise ...func(Stringable) Stringable) Stringable {
	return s.When(s.DoesntStartWith(needles...), callback, otherwise...)
}

// WhenDoesntEndWith is When on whether the string ends with none of the
// needles.
func (s Stringable) WhenDoesntEndWith(needles []string, callback func(Stringable) Stringable, otherwise ...func(Stringable) Stringable) Stringable {
	return s.When(s.DoesntEndWith(needles...), callback, otherwise...)
}

// WhenExactly is When on whether the string equals value.
func (s Stringable) WhenExactly(value string, callback func(Stringable) Stringable, otherwise ...func(Stringable) Stringable) Stringable {
	return s.When(s.Exactly(value), callback, otherwise...)
}

// WhenNotExactly is When on whether the string does not equal value.
func (s Stringable) WhenNotExactly(value string, callback func(Stringable) Stringable, otherwise ...func(Stringable) Stringable) Stringable {
	return s.When(!s.Exactly(value), callback, otherwise...)
}

// WhenIs is When on whether the string matches any one of the glob patterns.
func (s Stringable) WhenIs(pattern []string, callback func(Stringable) Stringable, otherwise ...func(Stringable) Stringable) Stringable {
	return s.When(s.Is(pattern, false), callback, otherwise...)
}

// WhenIsASCII is When on whether every byte of the string is below 128.
func (s Stringable) WhenIsASCII(callback func(Stringable) Stringable, otherwise ...func(Stringable) Stringable) Stringable {
	return s.When(s.IsASCII(), callback, otherwise...)
}

// WhenIsUUID is When on whether the string has the shape of a UUID.
func (s Stringable) WhenIsUUID(callback func(Stringable) Stringable, otherwise ...func(Stringable) Stringable) Stringable {
	return s.When(s.IsUUID(), callback, otherwise...)
}

// WhenIsULID is When on whether the string has the shape of a ULID.
func (s Stringable) WhenIsULID(callback func(Stringable) Stringable, otherwise ...func(Stringable) Stringable) Stringable {
	return s.When(s.IsULID(), callback, otherwise...)
}

// WhenTest is When on whether the string matches any one of the regular
// expressions.
func (s Stringable) WhenTest(patterns []*regexp.Regexp, callback func(Stringable) Stringable, otherwise ...func(Stringable) Stringable) Stringable {
	return s.When(s.Test(patterns), callback, otherwise...)
}

// ToInteger reads the longest integer at the front of the string and answers
// zero when there is none.
//
// The optional base defaults to ten. Base zero reads the prefix -- 0x for
// hexadecimal, 0b for binary, 0 for octal.
func (s Stringable) ToInteger(base ...int) int {
	b := 10
	if len(base) > 0 {
		b = base[0]
	}
	return phpIntval(s.value, b)
}

// ToFloat reads the longest decimal at the front of the string and answers zero
// when there is none.
func (s Stringable) ToFloat() float64 { return phpFloatval(s.value) }

// ToBoolean reads the string as a boolean: "1", "true", "on" and "yes" are
// true, in any case and with the ends trimmed, and everything else is false.
func (s Stringable) ToBoolean() bool {
	switch strings.ToLower(strings.TrimSpace(s.value)) {
	case "1", "true", "on", "yes":
		return true
	}
	return false
}

// ToDate reads the string as a time.
//
// With a format, the format is written in single-letter tokens -- "Y-m-d H:i:s"
// and the like -- and the string has to match it whole. With none it tries the
// layouts a machine writes, in turn, and a string none of them reads is an
// error.
//
// A layout that carries no zone is read as UTC, and a caller that wants another
// one converts what comes back.
func (s Stringable) ToDate(format ...string) (time.Time, error) {
	if len(format) > 0 {
		layout, err := goLayout(format[0])
		if err != nil {
			return time.Time{}, err
		}
		return time.Parse(layout, s.value)
	}
	return parseDate(s.value)
}
