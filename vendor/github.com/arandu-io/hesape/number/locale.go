package number

import (
	"strings"
	"sync"
)

// The process-wide default locale and currency, and the lock that guards them
// against concurrent readers and writers.
var (
	defaultsMu sync.RWMutex
	locale     = "en"
	currency   = "USD"
)

// UseLocale sets the default locale.
//
// The locale is remembered and reported back by DefaultLocale, and it is what
// WithLocale swaps around a callback.
//
// It decides how Currency and CurrencyFromCents write an amount: what separates
// the groups of digits, what stands before the fraction, and which side of the
// amount the symbol goes on. It decides nothing else. Format, Parse, Ordinal
// and the spelled-out forms write the conventions of en-US whatever is set
// here, and the package comment says why.
//
// A locale this package carries no conventions for is written with the en-US
// ones, so setting an unknown locale changes nothing rather than failing.
func UseLocale(l string) {
	defaultsMu.Lock()
	locale = l
	defaultsMu.Unlock()
}

// DefaultLocale is the locale UseLocale last set, and "en" until one is.
func DefaultLocale() string {
	defaultsMu.RLock()
	defer defaultsMu.RUnlock()
	return locale
}

// UseCurrency sets the default currency, which is the one Currency renders an
// empty code in.
func UseCurrency(c string) {
	defaultsMu.Lock()
	currency = c
	defaultsMu.Unlock()
}

// DefaultCurrency is the currency UseCurrency last set, and "USD" until one is.
func DefaultCurrency() string {
	defaultsMu.RLock()
	defer defaultsMu.RUnlock()
	return currency
}

// WithLocale runs the callback with the given locale in force and puts the
// previous one back afterwards, whatever the callback does.
//
// The result is generic over the callback's own return type.
func WithLocale[T any](l string, callback func() T) T {
	previous := DefaultLocale()
	UseLocale(l)
	defer UseLocale(previous)
	return callback()
}

// WithCurrency runs the callback with the given currency in force and puts the
// previous one back afterwards, whatever the callback does.
//
// The result is generic over the callback's own return type.
func WithCurrency[T any](c string, callback func() T) T {
	previous := DefaultCurrency()
	UseCurrency(c)
	defer UseCurrency(previous)
	return callback()
}

// currencyFormat is how one locale writes an amount of money: what goes between
// groups of integer digits, what goes before the fraction, and where the symbol
// sits against the amount.
//
// A locale is four values rather than a rendering function of its own, so that
// carrying another one is filling in a row and cannot change what the rows
// already there produce.
type currencyFormat struct {
	// group separates each three integer digits, and decimal stands before
	// the fraction.
	group   string
	decimal string

	// symbolTrails puts the symbol after the amount rather than before it, and
	// symbolGap is what goes between the two. The sign leads the whole either
	// way, because a sign written after the amount is read as part of it.
	symbolTrails bool
	symbolGap    string
}

// currencyFormats holds the locales this package has been given money
// conventions for, keyed by the tag lowercased with its separator normalised to
// a hyphen.
//
// The table is short on purpose. A row states conventions that were checked
// against how the locale is actually written, and a locale filled in from a
// neighbour it shares a language with is wrong in a way nothing here reports:
// pt-BR and pt-PT do not write an amount the same way, so a row for one is not
// a row for the other.
var currencyFormats = map[string]currencyFormat{
	"en":    {group: ",", decimal: ".", symbolGap: ""},
	"pt-br": {group: ".", decimal: ",", symbolGap: " "},
}

// defaultCurrencyFormat writes an amount the way every other function in this
// package writes a number, and is what a locale with no row of its own gets.
var defaultCurrencyFormat = currencyFormat{
	group:     string(groupSeparator),
	decimal:   string(decimalSeparator),
	symbolGap: "",
}

// currencyFormatFor is the money conventions of a locale tag: the row keyed by
// the whole tag, then the row keyed by its language alone, then the default.
//
// The tag is matched whole before it is cut, so that a locale carrying a row of
// its own is never answered by the row of its language, and the language step
// is what lets one row serve every region that does write an amount alike.
func currencyFormatFor(l string) currencyFormat {
	tag := strings.ReplaceAll(strings.ToLower(strings.TrimSpace(l)), "_", "-")
	if f, ok := currencyFormats[tag]; ok {
		return f
	}
	if language, _, cut := strings.Cut(tag, "-"); cut {
		if f, ok := currencyFormats[language]; ok {
			return f
		}
	}
	return defaultCurrencyFormat
}
