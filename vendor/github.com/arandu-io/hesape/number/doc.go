// Package number formats, spells and parses numbers.
//
// It renders a value as a plain figure (Format), a percentage, an ordinal, a
// currency amount, a file size or a rounded summary (Abbreviate, ForHumans);
// it writes one out in words (Spell, SpellOrdinal); and it reads a formatted
// one back (Parse, ParseInt, ParseFloat).
//
// # Optional arguments
//
// Go has no default arguments, so the trailing optional digit counts arrive as
// a variadic tail in the order they are declared: Format(v) gives neither the
// precision nor the maximum precision, Format(v, 2) gives the precision, and
// Format(v, 2, 4) gives both. Where a boolean follows them -- FileSize and
// ForHumans -- every argument is spelled out instead, and a negative maximum
// precision stands for the one that was not given.
//
// # One set of conventions, and the money written outside it
//
// Locale-aware formatting needs CLDR data, and carrying it would mean a third
// party dependency this module does not take. This package therefore writes a
// single set of conventions, the ones of en-US: a comma every three digits, a
// dot before the fraction, English ordinal suffixes and English number words.
// Format, Parse, Ordinal, Percentage, FileSize, Spell and SpellOrdinal write
// those whatever locale is set.
//
// Money is the exception, because an amount written under the wrong convention
// is not merely foreign to read: R$1,234.50 and R$1.234,50 are both amounts,
// three orders of magnitude apart, and a reader checking a total cannot tell
// which one is meant. Currency and CurrencyFromCents therefore follow
// DefaultLocale for three things -- the group separator, the decimal separator,
// and which side of the amount the symbol goes on.
//
// The locales carrying conventions of their own are en and pt-BR. That list is
// short because each entry states what was checked rather than what was
// guessed; a locale not on it is written the en-US way. A tag is matched whole
// and then by its language, so en-GB is answered by en, while pt-PT is not
// answered by pt-BR.
//
// UseCurrency, DefaultCurrency and WithCurrency carry the default currency,
// which is the one Currency and CurrencyFromCents render an empty code in.
//
// # Amounts of money
//
// CurrencyFromCents takes an int64 count of a currency's minor units and never
// converts it to a float64. Currency takes a float64, for an amount that is
// fractional in its own right and has no exact count behind it. Which one to
// call is decided by the type the amount is already held in, and the two agree
// on everything after that, because they share the one assembly step.
//
// # Rounding
//
// Precision is the number of digits kept after the decimal point, and a value
// below zero is read as zero. Ties round to the nearest even digit, so
// Format(2.5, 0) is "2" and Format(3.5, 0) is "4".
//
// NaN and the infinities are returned as "NaN", "Inf" and "-Inf" by every
// function in this package, with no symbol, suffix or unit attached.
package number
