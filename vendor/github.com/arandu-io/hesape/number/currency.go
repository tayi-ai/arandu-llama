package number

import (
	"strconv"
	"strings"
)

// currencySymbols holds the currencies whose symbol a reader of en-US is
// expected to recognise. A code that is not here is printed as itself, so the
// table can stay short and correct instead of long and half wrong.
var currencySymbols = map[string]string{
	"AUD": "A$",
	"BRL": "R$",
	"CAD": "CA$",
	"CNY": "CN¥",
	"EUR": "€",
	"GBP": "£",
	"ILS": "₪",
	"INR": "₹",
	"JPY": "¥",
	"KRW": "₩",
	"MXN": "MX$",
	"NGN": "₦",
	"PHP": "₱",
	"RUB": "₽",
	"THB": "฿",
	"TRY": "₺",
	"USD": "$",
	"VND": "₫",
}

// currencyDigits holds the ISO 4217 currencies that do not have two minor
// units. Anything absent from this table is rendered with two.
var currencyDigits = map[string]int{
	"BHD": 3,
	"BIF": 0,
	"CLP": 0,
	"DJF": 0,
	"GNF": 0,
	"IQD": 3,
	"ISK": 0,
	"JOD": 3,
	"JPY": 0,
	"KMF": 0,
	"KRW": 0,
	"KWD": 3,
	"LYD": 3,
	"OMR": 3,
	"PYG": 0,
	"RWF": 0,
	"TND": 3,
	"UGX": 0,
	"VND": 0,
	"VUV": 0,
	"XAF": 0,
	"XOF": 0,
	"XPF": 0,
}

// Currency renders v as an amount in the given ISO 4217 currency. The code is
// read case-insensitively and decides two things: the symbol, and how many
// digits the currency keeps after the decimal point -- two for most, none for
// the yen, three for the dinars.
//
// An empty currency falls back to DefaultCurrency. The optional argument is a
// precision, which overrides the digit count the currency itself carries.
//
// A code with no symbol in the table is printed in front of the amount as it
// was given. The sign leads, whichever side of the amount the symbol is on.
//
// The separators and the place of the symbol come from DefaultLocale; see the
// package comment for which locales carry conventions of their own.
//
//	Currency(1234.5, "USD")  // "$1,234.50"
//	Currency(1234.56, "JPY") // "¥1,235"
//	Currency(-99, "CHF")     // "-CHF 99.00"
//	Currency(1234.5, "USD", 0) // "$1,235"
//
// v is a float64, so an amount that reaches here as one has already been
// rounded once by the type that carries it. CurrencyFromCents is the variant
// for an amount held as an exact count of minor units, and it is the one to
// call when such a count exists: this one is for a value that is fractional in
// its own right -- a rate, an average, a converted total -- and has no exact
// count behind it.
func Currency(v float64, in string, precision ...int) string {
	if s, ok := special(v); ok {
		return s
	}
	code := currencyCode(in)

	digits := currencyFractionDigits(code)
	if len(precision) > 0 {
		digits = max(precision[0], 0)
	}

	fixed := strconv.FormatFloat(v, 'f', digits, 64)
	negative := strings.HasPrefix(fixed, "-")
	if negative {
		fixed = fixed[1:]
	}
	integer, fraction := fixed, ""
	if i := strings.IndexByte(fixed, decimalSeparator); i >= 0 {
		integer, fraction = fixed[:i], fixed[i+1:]
	}
	return renderCurrency(negative, integer, fraction, code, currencyFormatFor(DefaultLocale()))
}

// CurrencyFromCents renders an exact count of a currency's minor units as an
// amount in that currency. The code is read case-insensitively and says both
// what a minor unit is worth and what symbol stands for the currency:
//
//	CurrencyFromCents(123450, "USD") // "$1,234.50"
//	CurrencyFromCents(-99, "USD")    // "-$0.99"
//	CurrencyFromCents(1234, "JPY")   // "¥1,234"
//	CurrencyFromCents(1234500, "BHD") // "BHD 1,234.500"
//
// The yen has no minor unit, so its count is already whole units; the dinars
// have three digits rather than two, so a thousand of their minor units is one.
//
// No float64 appears anywhere on this path. An amount held as an integer count
// of minor units -- which is how an amount is held when it has to add up --
// reaches the page as the digits it was stored as, with no conversion in
// between where a unit could be rounded away. Call Currency instead when the
// amount is fractional in its own right and no exact count exists.
//
// There is no precision argument, because the currency's own digit count is
// what says where the integer splits. Overriding it would not round the amount:
// it would read the same integer as a different one.
//
// The separators and the place of the symbol come from DefaultLocale; see the
// package comment for which locales carry conventions of their own.
func CurrencyFromCents(cents int64, in string) string {
	code := currencyCode(in)
	digits := currencyFractionDigits(code)

	// The magnitude is taken in uint64 because the most negative int64 has no
	// positive counterpart to negate it into.
	negative := cents < 0
	magnitude := uint64(cents)
	if negative {
		magnitude = uint64(-(cents + 1)) + 1
	}

	whole := strconv.FormatUint(magnitude, 10)
	if pad := digits + 1 - len(whole); pad > 0 {
		whole = strings.Repeat("0", pad) + whole
	}
	integer, fraction := whole, ""
	if digits > 0 {
		integer, fraction = whole[:len(whole)-digits], whole[len(whole)-digits:]
	}
	return renderCurrency(negative, integer, fraction, code, currencyFormatFor(DefaultLocale()))
}

// renderCurrency is the assembly both currency functions end in: the digits
// either side of the point arrive with no sign and no separators in them, and
// the format says what separates them and where the symbol goes.
//
// Having one assembly is what keeps the two entry points from drifting into two
// answers for the same amount; what differs between them is the type they take
// the amount in, and nothing after that. The format is a parameter rather than
// a read of the default locale, so that what this produces depends on nothing
// but its arguments.
func renderCurrency(negative bool, integer, fraction, code string, f currencyFormat) string {
	if negative && isZero(integer) && isZero(fraction) {
		// A sign that only survived rounding is not a sign: an amount that
		// comes out as zero is written as zero.
		negative = false
	}
	sign := ""
	if negative {
		sign = "-"
	}

	amount := groupDigits(integer, f.group)
	if fraction != "" {
		amount += f.decimal + fraction
	}
	if code == "" {
		return sign + amount
	}

	symbol, gap := currencySymbols[code], f.symbolGap
	if symbol == "" {
		// A code standing in for a symbol always takes a space: "BRL1.234,50"
		// reads as one token, and "BRL 1.234,50" as the two it is.
		symbol, gap = code, " "
	}
	if f.symbolTrails {
		return sign + amount + gap + symbol
	}
	return sign + symbol + gap + amount
}

// currencyCode is the ISO 4217 code a caller gave, uppercased and trimmed so
// that the tables answer it, with an empty one standing for DefaultCurrency.
func currencyCode(in string) string {
	code := strings.ToUpper(strings.TrimSpace(in))
	if code == "" {
		code = strings.ToUpper(strings.TrimSpace(DefaultCurrency()))
	}
	return code
}

// currencyFractionDigits is how many digits a currency keeps after the point,
// which is two for every currency currencyDigits does not name.
func currencyFractionDigits(code string) int {
	if d, ok := currencyDigits[code]; ok {
		return d
	}
	return 2
}
