package validation

import (
	"context"
	"math"
	"math/big"
	"net/url"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/enum"
	"github.com/arandu-io/hesape/str"
)

// This file is one method per rule.
//
// Every one takes (attribute, value, parameters), including the rules that read
// only two of the three: one signature for all of them is what lets the rule
// table hold the method itself rather than a closure around it, and so what
// keeps the table from being a second place where a rule's behaviour is
// written.
//
// value is any, because a decoded body carries anything; parameters is []string,
// because a rule string carries text.

// numericRules are the three rules that make
// every size rule on the field measure the value rather than the characters.
var numericRules = []string{"numeric", "integer", "decimal"}

// implicitRules are the rules that run even
// when the value is blank, and whose failure stops the field.
var implicitRules = []string{
	"accepted", "accepted_if", "declined", "declined_if", "filled", "missing",
	"missing_if", "missing_unless", "missing_with", "missing_with_all",
	"present", "present_if", "present_unless", "present_with",
	"present_with_all", "required", "required_if", "required_if_accepted",
	"required_if_declined", "required_unless", "required_with",
	"required_with_all", "required_without", "required_without_all",
}

// excludeRules are the five whose failure
// removes the field from the validated data instead of putting a message on it.
var excludeRules = []string{"exclude", "exclude_if", "exclude_unless", "exclude_with", "exclude_without"}

// acceptable and declinable are the strings `accepted` and `declined` compare
// against. The comparison is by type as well as by value, so the boolean and the
// integer are separate cases in isAccepted rather than extra strings here.
var (
	acceptable = []string{"yes", "on", "1", "true"}
	declinable = []string{"no", "off", "0", "false"}
)

// phpExtensions are the upload extensions ShouldBlockPhpUpload refuses.
var phpExtensions = []string{"php", "php3", "php4", "php5", "php7", "php8", "phtml", "phar"}

// ---------------------------------------------------------------------------
// Accepted and declined.
// ---------------------------------------------------------------------------

// ValidateAccepted is `accepted`: the value is one of yes, on, 1 or true. It
// implies the attribute is required.
func (v *Validator) ValidateAccepted(attribute string, value any, parameters []string) bool {
	return v.ValidateRequired(attribute, value, nil) && isAccepted(value)
}

// ValidateAcceptedIf is `accepted_if`: accepted when the other attribute holds
// one of the given values.
func (v *Validator) ValidateAcceptedIf(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(2, parameters, "accepted_if") {
		return false
	}
	values, other := v.ParseDependentRuleParameters(parameters)
	if looseContains(values, other) {
		return v.ValidateRequired(attribute, value, nil) && isAccepted(value)
	}
	return true
}

// ValidateDeclined is `declined`: the value is one of no, off, 0 or false. It
// implies the attribute is required.
func (v *Validator) ValidateDeclined(attribute string, value any, parameters []string) bool {
	return v.ValidateRequired(attribute, value, nil) && isDeclined(value)
}

// ValidateDeclinedIf is `declined_if`: declined when the other attribute holds
// one of the given values.
func (v *Validator) ValidateDeclinedIf(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(2, parameters, "declined_if") {
		return false
	}
	values, other := v.ParseDependentRuleParameters(parameters)
	if looseContains(values, other) {
		return v.ValidateRequired(attribute, value, nil) && isDeclined(value)
	}
	return true
}

// isAccepted is what `accepted` accepts: the four strings, the integer 1 and the
// boolean true, and nothing that merely looks like them.
func isAccepted(value any) bool {
	switch n := value.(type) {
	case string:
		return slices.Contains(acceptable, n)
	case bool:
		return n
	case int:
		return n == 1
	case int64:
		return n == 1
	}
	return false
}

// isDeclined is what `declined` accepts: the four strings, the integer 0 and the
// boolean false.
func isDeclined(value any) bool {
	switch n := value.(type) {
	case string:
		return slices.Contains(declinable, n)
	case bool:
		return !n
	case int:
		return n == 0
	case int64:
		return n == 0
	}
	return false
}

// ---------------------------------------------------------------------------
// The network.
// ---------------------------------------------------------------------------

// ValidateActiveUrl is `active_url`: the host of the value has an A or an AAAA
// record.
//
// The name spells Url rather than URL because `url` is the rule somebody types,
// and ValidateUrl would then read differently from its neighbour here.
//
// This puts a DNS lookup on the request path. The deadline comes from the
// Validator's context, which is the request's.
func (v *Validator) ValidateActiveUrl(attribute string, value any, parameters []string) bool {
	s, ok := asString(value)
	if !ok {
		return false
	}
	parsed, err := url.Parse(s)
	if err != nil || parsed.Hostname() == "" {
		return false
	}
	records, err := v.GetDNSRecords(parsed.Hostname())
	return err == nil && len(records) > 0
}

// GetDNSRecords returns the addresses the hostname resolves to.
func (v *Validator) GetDNSRecords(hostname string) ([]string, error) {
	addrs, err := v.resolver().LookupIPAddr(v.Context(), hostname)
	if err != nil {
		return nil, err
	}
	records := make([]string, 0, len(addrs))
	for _, a := range addrs {
		records = append(records, a.IP.String())
	}
	return records, nil
}

// ---------------------------------------------------------------------------
// Shape of the text.
// ---------------------------------------------------------------------------

// ValidateAscii is `ascii`: every character is a 7-bit ASCII one.
func (v *Validator) ValidateAscii(attribute string, value any, parameters []string) bool {
	s, ok := asString(value)
	if !ok {
		return false
	}
	return str.IsASCII(s)
}

// ValidateBail is `bail`. It is a marker: Set.Validate reads it and stops the
// field at its first failure.
func (v *Validator) ValidateBail(attribute string, value any, parameters []string) bool { return true }

// ValidateAlpha is `alpha`: letters only. With the parameter "ascii" it is the
// ASCII letters only.
func (v *Validator) ValidateAlpha(attribute string, value any, parameters []string) bool {
	return v.matchesAlphabet(value, parameters, alphaUnicode, alphaASCII)
}

// ValidateAlphaDash is `alpha_dash`: letters, digits, dashes and underscores.
// With the parameter "ascii" it is the ASCII ones only.
func (v *Validator) ValidateAlphaDash(attribute string, value any, parameters []string) bool {
	return v.matchesAlphabet(value, parameters, dashUnicode, dashASCII)
}

// ValidateAlphaNum is `alpha_num`: letters and digits. With the parameter
// "ascii" it is the ASCII ones only.
func (v *Validator) ValidateAlphaNum(attribute string, value any, parameters []string) bool {
	return v.matchesAlphabet(value, parameters, numUnicode, numASCII)
}

func (v *Validator) matchesAlphabet(value any, parameters []string, unicoded, ascii *regexp.Regexp) bool {
	if !isStringOrNumber(value) {
		return false
	}
	if len(parameters) == 1 && parameters[0] == "ascii" {
		return ascii.MatchString(stringOf(value))
	}
	return unicoded.MatchString(stringOf(value))
}

// ValidateLowercase is `lowercase`: the value is already all lower case.
func (v *Validator) ValidateLowercase(attribute string, value any, parameters []string) bool {
	s := stringOf(value)
	return strings.ToLower(s) == s
}

// ValidateUppercase is `uppercase`: the value is already all upper case.
func (v *Validator) ValidateUppercase(attribute string, value any, parameters []string) bool {
	s := stringOf(value)
	return strings.ToUpper(s) == s
}

// ValidateStartsWith is `starts_with`: the value begins with one of the given
// prefixes.
func (v *Validator) ValidateStartsWith(attribute string, value any, parameters []string) bool {
	return anyAffix(parameters, strings.HasPrefix, stringOf(value))
}

// ValidateDoesntStartWith is `doesnt_start_with`: the value begins with none of
// the given prefixes.
func (v *Validator) ValidateDoesntStartWith(attribute string, value any, parameters []string) bool {
	return !anyAffix(parameters, strings.HasPrefix, stringOf(value))
}

// ValidateEndsWith is `ends_with`: the value ends with one of the given
// suffixes.
func (v *Validator) ValidateEndsWith(attribute string, value any, parameters []string) bool {
	return anyAffix(parameters, strings.HasSuffix, stringOf(value))
}

// ValidateDoesntEndWith is `doesnt_end_with`: the value ends with none of the
// given suffixes.
func (v *Validator) ValidateDoesntEndWith(attribute string, value any, parameters []string) bool {
	return !anyAffix(parameters, strings.HasSuffix, stringOf(value))
}

// ValidateString is `string`: the value is text.
//
// It is not the tautology it looks like. A JSON body carries numbers, booleans
// and lists, and this is the rule that refuses them where text was asked for.
func (v *Validator) ValidateString(attribute string, value any, parameters []string) bool {
	_, ok := asString(value)
	return ok
}

// ---------------------------------------------------------------------------
// Arrays.
// ---------------------------------------------------------------------------

// ValidateArray is `array`: the value is an array, and with parameters, an array
// with no key outside them.
func (v *Validator) ValidateArray(attribute string, value any, parameters []string) bool {
	if !isArray(value) {
		return false
	}
	if len(parameters) == 0 {
		return true
	}
	switch node := value.(type) {
	case Data:
		for key := range node {
			if !slices.Contains(parameters, key) {
				return false
			}
		}
	case map[string]any:
		for key := range node {
			if !slices.Contains(parameters, key) {
				return false
			}
		}
	case []any:
		for i := range node {
			if !slices.Contains(parameters, strconv.Itoa(i)) {
				return false
			}
		}
	}
	return true
}

// ValidateList is `list`: the value is an array with consecutive integer keys. A
// Data is an array with names, so it is not a list; a []any is.
func (v *Validator) ValidateList(attribute string, value any, parameters []string) bool {
	_, ok := asList(value)
	return ok
}

// ValidateRequiredArrayKeys is `required_array_keys`: the value is a keyed array
// holding every one of the named keys.
func (v *Validator) ValidateRequiredArrayKeys(attribute string, value any, parameters []string) bool {
	keyed, ok := value.(Data)
	if !ok {
		if m, isMap := value.(map[string]any); isMap {
			keyed = Data(m)
		} else {
			return false
		}
	}
	for _, key := range parameters {
		if _, exists := keyed[key]; !exists {
			return false
		}
	}
	return true
}

// ValidateContains is `contains`: the array holds every one of the given
// values.
func (v *Validator) ValidateContains(attribute string, value any, parameters []string) bool {
	list, ok := asList(value)
	if !ok {
		return false
	}
	for _, parameter := range parameters {
		found := false
		for _, member := range list {
			if stringOf(member) == parameter {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

// ValidateDistinct is `distinct`: no value repeats.
//
// There are no wildcards in a rule string, so the rule is asked of the whole
// list at once rather than once per member. The parameters are `ignore_case`,
// which compares case-insensitively, and `strict`, which compares types as well
// as values.
func (v *Validator) ValidateDistinct(attribute string, value any, parameters []string) bool {
	list, ok := asList(value)
	if !ok {
		return true
	}
	ignoreCase := slices.Contains(parameters, "ignore_case")
	strict := slices.Contains(parameters, "strict")

	for i, member := range list {
		for _, other := range list[i+1:] {
			switch {
			case ignoreCase:
				if strings.EqualFold(stringOf(member), stringOf(other)) {
					return false
				}
			case strict:
				if sameType(member, other) && sameValue(member, other) {
					return false
				}
			default:
				if stringOf(member) == stringOf(other) {
					return false
				}
			}
		}
	}
	return true
}

// GetDistinctValues returns the values `distinct` compares: the list held at the
// attribute, for the reason ValidateDistinct gives.
func (v *Validator) GetDistinctValues(attribute string) []any {
	return v.ExtractDistinctValues(attribute)
}

// ExtractDistinctValues reads the attribute and returns it as a list, or nothing
// when it holds no list.
func (v *Validator) ExtractDistinctValues(attribute string) []any {
	list, _ := asList(v.GetValue(attribute))
	return list
}

// ValidateInArray is `in_array`: the value is one of the values held by the
// other attribute.
func (v *Validator) ValidateInArray(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(1, parameters, "in_array") {
		return false
	}
	other := v.GetValue(parameters[0])
	list, ok := asList(other)
	if !ok {
		return false
	}
	for _, member := range list {
		if sameValue(member, value) || stringOf(member) == stringOf(value) {
			return true
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// Size, and the four comparisons.
// ---------------------------------------------------------------------------

// ValidateBetween is `between`: the size is within the two bounds, both of them
// inclusive.
func (v *Validator) ValidateBetween(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(2, parameters, "between") {
		return false
	}
	low, okLow := exactParameter(parameters[0])
	high, okHigh := exactParameter(parameters[1])
	size, okSize := v.GetSize(attribute, value)
	if !okLow || !okHigh || !okSize {
		return false
	}
	return size.Cmp(low) >= 0 && size.Cmp(high) <= 0
}

// ValidateMin is `min`: the size is at or above the bound.
func (v *Validator) ValidateMin(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(1, parameters, "min") {
		return false
	}
	low, okLow := exactParameter(parameters[0])
	size, okSize := v.GetSize(attribute, value)
	return okLow && okSize && size.Cmp(low) >= 0
}

// ValidateMax is `max`: the size is at or below the bound. An upload that did
// not finish fails before its size is asked for.
func (v *Validator) ValidateMax(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(1, parameters, "max") {
		return false
	}
	if up, ok := value.(UploadedFile); ok && !up.IsValid() {
		return false
	}
	high, okHigh := exactParameter(parameters[0])
	size, okSize := v.GetSize(attribute, value)
	return okHigh && okSize && size.Cmp(high) <= 0
}

// ValidateSize is `size`: the size is exactly the given one.
//
// The size is four things under one name, as GetSize measures it: the characters
// of a string, the value of a number when the field declares numeric, integer or
// decimal, the members of an array, and the KILOBYTES of a file.
func (v *Validator) ValidateSize(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(1, parameters, "size") {
		return false
	}
	want, okWant := exactParameter(parameters[0])
	size, okSize := v.GetSize(attribute, value)
	return okWant && okSize && size.Cmp(want) == 0
}

// ValidateGt is `gt`: greater than a literal bound, or than another attribute.
func (v *Validator) ValidateGt(attribute string, value any, parameters []string) bool {
	return v.compareToAnother(attribute, value, parameters, "gt", func(c int) bool { return c > 0 })
}

// ValidateGte is `gte`: greater than or equal to a literal bound, or to another
// attribute.
func (v *Validator) ValidateGte(attribute string, value any, parameters []string) bool {
	return v.compareToAnother(attribute, value, parameters, "gte", func(c int) bool { return c >= 0 })
}

// ValidateLt is `lt`: less than a literal bound, or than another attribute.
func (v *Validator) ValidateLt(attribute string, value any, parameters []string) bool {
	return v.compareToAnother(attribute, value, parameters, "lt", func(c int) bool { return c < 0 })
}

// ValidateLte is `lte`: less than or equal to a literal bound, or to another
// attribute.
func (v *Validator) ValidateLte(attribute string, value any, parameters []string) bool {
	return v.compareToAnother(attribute, value, parameters, "lte", func(c int) bool { return c <= 0 })
}

// compareToAnother is the body the four comparisons share, in this order: a
// literal bound when the parameter names no field, a numeric comparison when the
// field declares numeric, a refusal when the two values are not the same kind of
// thing, and a comparison of sizes otherwise.
func (v *Validator) compareToAnother(attribute string, value any, parameters []string, name string, ok func(int) bool) bool {
	if !v.RequireParameterCount(1, parameters, name) {
		return false
	}
	other := v.GetValue(parameters[0])

	bound, isBound := exactParameter(parameters[0])
	if other == nil && isBound {
		if _, numeric := numberOf(value); numeric {
			size, okSize := v.GetSize(attribute, value)
			return okSize && ok(size.Cmp(bound))
		}
	}
	if isBound {
		// The parameter is a number and a field of that name does hold a value.
		// Refuse rather than guess which of the two was meant.
		return false
	}
	if v.HasRule(attribute, numericRules) {
		mine, mineOK := exactNumber(value)
		theirs, theirsOK := exactNumber(other)
		if mineOK && theirsOK {
			return ok(mine.Cmp(theirs))
		}
	}
	if !sameType(value, other) {
		return false
	}
	mine, mineOK := v.GetSize(attribute, value)
	theirs, theirsOK := v.GetSize(parameters[0], other)
	return mineOK && theirsOK && ok(mine.Cmp(theirs))
}

// GetSize is what min, max, size, between and the four comparisons measure.
//
// A number is its own size when the field declares numeric, integer or decimal;
// an array is how many members it has; a file is its KILOBYTES; anything else is
// how many characters it has. Characters, never bytes: a limit in bytes rejects
// valid input in every language that needs more than one byte for a letter.
//
// The size is a *big.Rat and not a float64, because 9007199254740993 and
// 9007199254740992 are the SAME float64: "numeric|max:9007199254740992" would
// pass on the value 9007199254740993, and a monetary limit or a quota would be
// over-runnable by one unit of rounding. math/big is stdlib and the comparison
// is exact.
//
// The bool is false for an exponent outside the allowed range, which gives a
// value no size at all. Every caller fails the rule on it, which is the closed
// answer.
func (v *Validator) GetSize(attribute string, value any) (*big.Rat, bool) {
	if text, isNumber := numericText(value); isNumber && v.HasRule(attribute, numericRules) {
		if !v.exponentWithinAllowedRange(attribute, text, value) {
			return nil, false
		}
		return exactText(text)
	}
	if n, ok := countOf(value); ok {
		return new(big.Rat).SetInt64(int64(n)), true
	}
	if f, ok := asFile(value); ok {
		return new(big.Rat).SetFrac64(f.GetSize(), 1024), true
	}
	return new(big.Rat).SetInt64(int64(len([]rune(stringOf(value))))), true
}

// exponentWithinAllowedRange reports whether the value's exponent is one the
// package will size. Outside the range the value has no size at all, and the
// rule asking for it fails.
func (v *Validator) exponentWithinAllowedRange(attribute, text string, value any) bool {
	_, exponent, written := strings.Cut(strings.ToLower(text), "e")
	if !written {
		return true
	}
	scale, err := strconv.Atoi(strings.TrimPrefix(exponent, "+"))
	if err != nil {
		// Text that reads as no number at all is a zero exponent, which is in
		// range.
		scale = 0
	}
	return v.EnsureExponentWithinAllowedRange(scale, attribute, value)
}

// ValidateDigits is `digits`: the value is digits only, and exactly the given
// number of them.
func (v *Validator) ValidateDigits(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(1, parameters, "digits") {
		return false
	}
	want, ok := whole(parameters[0])
	s := stringOf(value)
	return ok && digitsOnly(s) && int64(len(s)) == want
}

// ValidateDigitsBetween is `digits_between`: the value is digits only, and how
// many of them is within the two bounds.
func (v *Validator) ValidateDigitsBetween(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(2, parameters, "digits_between") {
		return false
	}
	low, okLow := whole(parameters[0])
	high, okHigh := whole(parameters[1])
	s := stringOf(value)
	n := int64(len(s))
	return okLow && okHigh && digitsOnly(s) && n >= low && n <= high
}

// ValidateMaxDigits is `max_digits`: the value is digits only, and at most the
// given number of them.
func (v *Validator) ValidateMaxDigits(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(1, parameters, "max_digits") {
		return false
	}
	high, ok := whole(parameters[0])
	s := stringOf(value)
	return ok && digitsOnly(s) && int64(len(s)) <= high
}

// ValidateMinDigits is `min_digits`: the value is digits only, and at least the
// given number of them.
func (v *Validator) ValidateMinDigits(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(1, parameters, "min_digits") {
		return false
	}
	low, ok := whole(parameters[0])
	s := stringOf(value)
	return ok && digitsOnly(s) && int64(len(s)) >= low
}

// ValidateMultipleOf is `multiple_of`: the value divides by the parameter with
// no remainder.
//
// Zero on both sides fails, a zero numerator over a non-zero denominator passes,
// and a zero denominator fails -- three cases written out, because "is 0 a
// multiple of 0" has no answer worth guessing.
func (v *Validator) ValidateMultipleOf(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(1, parameters, "multiple_of") {
		return false
	}
	numerator, ok := numberOf(value)
	denominator, okDenominator := number(strings.TrimSpace(parameters[0]))
	if !ok || !okDenominator {
		return false
	}
	switch {
	case numerator == 0 && denominator == 0:
		return false
	case numerator == 0:
		return true
	case denominator == 0:
		return false
	}
	// Scaled to whole numbers before the remainder, so that 0.3 against 0.1 is
	// not the 0.09999999999999998 that float arithmetic answers.
	scale := math.Pow(10, float64(max(decimalsOf(stringOf(value)), decimalsOf(parameters[0]))))
	return math.Mod(math.Round(numerator*scale), math.Round(denominator*scale)) == 0
}

func decimalsOf(s string) int {
	if _, after, found := strings.Cut(s, "."); found {
		return len(after)
	}
	return 0
}

// ---------------------------------------------------------------------------
// Types.
// ---------------------------------------------------------------------------

// ValidateNumeric is `numeric`: the value reads as a number.
func (v *Validator) ValidateNumeric(attribute string, value any, parameters []string) bool {
	_, ok := numberOf(value)
	return ok
}

// ValidateInteger is `integer`: the value reads as a whole number.
func (v *Validator) ValidateInteger(attribute string, value any, parameters []string) bool {
	switch value.(type) {
	case int, int64:
		return true
	}
	_, ok := whole(strings.TrimSpace(stringOf(value)))
	return ok
}

// ValidateDecimal is `decimal`: the count of digits after the point is the given
// number, or between the two given numbers.
func (v *Validator) ValidateDecimal(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(1, parameters, "decimal") {
		return false
	}
	if !v.ValidateNumeric(attribute, value, nil) {
		return false
	}
	places, ok := decimalPlaces(stringOf(value))
	if !ok {
		return false
	}
	want, okWant := whole(parameters[0])
	if !okWant {
		return false
	}
	if len(parameters) < 2 {
		return int64(places) == want
	}
	high, okHigh := whole(parameters[1])
	return okHigh && int64(places) >= want && int64(places) <= high
}

// ValidateBoolean is `boolean`. The comparison is by type as well as by value,
// so exactly six pass: true, false, 0, 1, "0" and "1". A ticked checkbox sends
// "on", which is what `accepted` is for.
func (v *Validator) ValidateBoolean(attribute string, value any, parameters []string) bool {
	switch n := value.(type) {
	case bool:
		return true
	case int:
		return n == 0 || n == 1
	case int64:
		return n == 0 || n == 1
	case string:
		return n == "0" || n == "1"
	}
	return false
}

// ValidateJson is `json`: the value is text that parses as JSON.
func (v *Validator) ValidateJson(attribute string, value any, parameters []string) bool {
	s, ok := asString(value)
	return ok && str.IsJSON(s)
}

// ValidateNullable is `nullable`. It is a marker: Set.Validate reads it and stops
// the field when the value is null -- null, not empty, which is the whole of the
// difference from leaving `required` off.
func (v *Validator) ValidateNullable(attribute string, value any, parameters []string) bool {
	return true
}

// ValidateSometimes is `sometimes`. It is a marker: Set.Validate skips the field
// when its key was not sent at all.
func (v *Validator) ValidateSometimes(attribute string, value any, parameters []string) bool {
	return true
}

// ---------------------------------------------------------------------------
// Dates.
// ---------------------------------------------------------------------------

// ValidateDate is `date`: the value reads as a date.
func (v *Validator) ValidateDate(attribute string, value any, parameters []string) bool {
	_, ok := parseDate("", stringOf(value))
	return ok
}

// ValidateDateFormat is `date_format`: the value is a date written in one of the
// given layouts.
//
// The layout is a GO layout -- date_format:2006-01-02, never date_format:Y-m-d.
// compile.go refuses a layout it cannot read at boot, rather than accepting one
// that then rejects every date.
//
// More than one layout may be given, and the value passes when ANY of them
// matches. A numeric value is read as the text it prints as, so a JSON body that
// sent 20240301 unquoted is not refused for having no quotes.
func (v *Validator) ValidateDateFormat(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(1, parameters, "date_format") {
		return false
	}
	s, isString := asString(value)
	if !isString {
		// Neither text nor a number is refused; a number is read as the text it
		// prints as.
		if _, isNumber := numberOf(value); !isNumber {
			return false
		}
		s = stringOf(value)
	}
	for _, layout := range parameters {
		t, err := time.Parse(layout, s)
		// The round trip is what makes the layout exact: time.Parse accepts
		// "2006-1-2" for the layout "2006-01-02", and a format that accepts two
		// spellings of one day is not a format.
		if err == nil && t.Format(layout) == s {
			return true
		}
	}
	return false
}

// ValidateBefore is `before`: the date is earlier than the other moment.
func (v *Validator) ValidateBefore(attribute string, value any, parameters []string) bool {
	return v.CompareDates(attribute, value, parameters, "<")
}

// ValidateBeforeOrEqual is `before_or_equal`: the date is not later than the
// other moment.
func (v *Validator) ValidateBeforeOrEqual(attribute string, value any, parameters []string) bool {
	return v.CompareDates(attribute, value, parameters, "<=")
}

// ValidateAfter is `after`: the date is later than the other moment.
func (v *Validator) ValidateAfter(attribute string, value any, parameters []string) bool {
	return v.CompareDates(attribute, value, parameters, ">")
}

// ValidateAfterOrEqual is `after_or_equal`: the date is not earlier than the
// other moment.
func (v *Validator) ValidateAfterOrEqual(attribute string, value any, parameters []string) bool {
	return v.CompareDates(attribute, value, parameters, ">=")
}

// ValidateDateEquals is `date_equals`: the date is the other moment.
func (v *Validator) ValidateDateEquals(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(1, parameters, "date_equals") {
		return false
	}
	return v.CompareDates(attribute, value, parameters, "=")
}

// CompareDates reads both moments and answers what the operator says. A moment
// that cannot be read fails on either side -- a date rule that cannot read the
// date has not been passed.
//
// The other moment is a field this rule set declares, one of the three keywords
// today, tomorrow and yesterday, or a literal date. The keywords are those three
// and nothing else: a rule whose accepted set nobody can enumerate cannot be
// reasoned about.
func (v *Validator) CompareDates(attribute string, value any, parameters []string, operator string) bool {
	if len(parameters) == 0 {
		return false
	}
	if !isStringOrNumber(value) {
		return false
	}
	mine, ok := parseDate(v.GetDateFormat(attribute), stringOf(value))
	if !ok {
		return false
	}
	theirs, ok := v.moment(attribute, parameters[0])
	if !ok {
		return false
	}
	return v.Compare(mine.Compare(theirs), 0, operator)
}

// GetDateFormat returns the layout the field's date_format declares, and empty
// when it declares none.
func (v *Validator) GetDateFormat(attribute string) string {
	if f, ok := v.set.byName[attribute]; ok {
		return f.layout
	}
	return ""
}

func (v *Validator) moment(attribute, argument string) (time.Time, bool) {
	if at, isKeyword := keywords[argument]; isKeyword {
		return at(v.Now()), true
	}
	if other, declared := v.set.byName[argument]; declared {
		return parseDate(other.layout, stringOf(v.GetValue(argument)))
	}
	return parseDate(v.GetDateFormat(attribute), argument)
}

// Compare applies the operator to the result of a comparison. It takes what
// comparing the two values said rather than the values themselves, because there
// is no operator to apply to an any.
func (v *Validator) Compare(result, against int, operator string) bool {
	switch operator {
	case "<":
		return result < against
	case ">":
		return result > against
	case "<=":
		return result <= against
	case ">=":
		return result >= against
	case "=":
		return result == against
	}
	return false
}

// ValidateTimezone is `timezone`: the value names a zone the system knows.
func (v *Validator) ValidateTimezone(attribute string, value any, parameters []string) bool {
	s, ok := asString(value)
	if !ok {
		return false
	}
	// "Local" is Go's spelling of "whatever this machine is set to", which is
	// not a zone a form can mean: it would pass on the developer's laptop and
	// on nothing else.
	if s == "Local" {
		return false
	}
	_, err := time.LoadLocation(s)
	return err == nil
}

// ---------------------------------------------------------------------------
// Identity and shape.
// ---------------------------------------------------------------------------

// ValidateEmail is `email`: the value has the shape of an address.
//
// Shape is all it checks by default. Deliverability is proven by sending mail;
// the parameter `dns` asks whether the domain resolves at all, which is as far
// as a lookup can go.
func (v *Validator) ValidateEmail(attribute string, value any, parameters []string) bool {
	s, ok := asString(value)
	if !ok {
		return false
	}
	if !emailShape(s) {
		return false
	}
	for _, validation := range parameters {
		switch validation {
		case "rfc", "strict", "filter", "filter_unicode":
			// The shape check above is all four of them: this package has one
			// answer to "is this an address" and does not keep four that drift.
		case "dns":
			at := strings.LastIndexByte(s, '@')
			records, err := v.GetDNSRecords(s[at+1:])
			if err != nil || len(records) == 0 {
				return false
			}
		}
	}
	return true
}

// ValidateUrl is `url`. The parameters are the schemes allowed, and http and
// https are the default.
func (v *Validator) ValidateUrl(attribute string, value any, parameters []string) bool {
	s, ok := asString(value)
	return ok && str.IsURL(s, parameters...)
}

// ValidateUuid is `uuid`: the value is a UUID.
func (v *Validator) ValidateUuid(attribute string, value any, parameters []string) bool {
	s, ok := asString(value)
	return ok && str.IsUUID(s)
}

// ValidateUlid is `ulid`: the value is a ULID.
func (v *Validator) ValidateUlid(attribute string, value any, parameters []string) bool {
	s, ok := asString(value)
	return ok && str.IsULID(s)
}

// ValidateIp is `ip`: the value is an IP address of either version.
func (v *Validator) ValidateIp(attribute string, value any, parameters []string) bool {
	_, ok := address(stringOf(value))
	return ok
}

// ValidateIpv4 is `ipv4`: the value is an IPv4 address.
func (v *Validator) ValidateIpv4(attribute string, value any, parameters []string) bool {
	a, ok := address(stringOf(value))
	return ok && a.Is4()
}

// ValidateIpv6 is `ipv6`: the value is an IPv6 address.
func (v *Validator) ValidateIpv6(attribute string, value any, parameters []string) bool {
	a, ok := address(stringOf(value))
	return ok && a.Is6()
}

// ValidateMacAddress is `mac_address`: the value is a MAC address.
func (v *Validator) ValidateMacAddress(attribute string, value any, parameters []string) bool {
	return macShape.MatchString(stringOf(value))
}

// ValidateHexColor is `hex_color`: the value is a hexadecimal colour.
func (v *Validator) ValidateHexColor(attribute string, value any, parameters []string) bool {
	return hexColorShape.MatchString(stringOf(value))
}

// ValidateRegex is `regex`: the value matches the pattern.
//
// The pattern is a GO pattern: RE2, with no backreference and no lookaround, and
// with no delimiters around it. compile.go compiles it at boot, so an unclosed
// group is a boot failure naming the field rather than a 500 on the first form
// somebody submits.
func (v *Validator) ValidateRegex(attribute string, value any, parameters []string) bool {
	if !isStringOrNumber(value) {
		return false
	}
	if !v.RequireParameterCount(1, parameters, "regex") {
		return false
	}
	re := v.pattern(parameters[0])
	return re != nil && re.MatchString(stringOf(value))
}

// ValidateNotRegex is `not_regex`: the value does not match the pattern, which
// is a GO pattern for the reason ValidateRegex gives.
func (v *Validator) ValidateNotRegex(attribute string, value any, parameters []string) bool {
	if !isStringOrNumber(value) {
		return false
	}
	if !v.RequireParameterCount(1, parameters, "not_regex") {
		return false
	}
	re := v.pattern(parameters[0])
	return re != nil && !re.MatchString(stringOf(value))
}

// pattern is the boot-compiled pattern of the rule being run, so a request
// compiles nothing.
func (v *Validator) pattern(source string) *regexp.Regexp {
	if v.currentRule != nil && v.currentRule.re != nil {
		return v.currentRule.re
	}
	re, err := regexp.Compile(source)
	if err != nil {
		return nil
	}
	return re
}

// ValidateIn is `in`: the value is one of the parameters.
//
// A value that is an array on a field that also declares `array` passes when
// every member is one of the parameters, which is how a multi-select is
// validated in one rule.
func (v *Validator) ValidateIn(attribute string, value any, parameters []string) bool {
	if list, ok := asList(value); ok && v.HasRule(attribute, []string{"array"}) {
		for _, member := range list {
			if isArray(member) {
				return false
			}
			if !slices.Contains(parameters, stringOf(member)) {
				return false
			}
		}
		return true
	}
	if isArray(value) {
		return false
	}
	return slices.Contains(parameters, stringOf(value))
}

// ValidateNotIn is `not_in`: the value is none of the parameters.
func (v *Validator) ValidateNotIn(attribute string, value any, parameters []string) bool {
	return !v.ValidateIn(attribute, value, parameters)
}

// ValidateEnum is `enum`: the value is one of the cases.
//
// A value that carries its own type -- a generated enum -- is asked, and the
// parameters are not consulted for membership. The list in a rule string is a
// second copy of a set the type already holds, and a second copy can disagree
// with it: a case added to the type and not to the rule used to reject a value
// the column accepts, and the two were never compared by anything.
//
// The parameters are still read, for exactly that disagreement. A case named in
// the rule that the type does not declare fails the rule outright, whatever the
// value was, because a rule listing a case nothing can produce was written
// against a different type -- and passing the field would hide it until the
// case that is missing arrives.
//
// Written against an untyped value -- a string straight off a form, which is
// what a field that has not been typed yet carries -- the parameters are the
// only set there is, and the rule reads them. Without either, there is nothing
// to decide against and the value is refused rather than waved through.
func (v *Validator) ValidateEnum(attribute string, value any, parameters []string) bool {
	if value == nil || isArray(value) {
		return false
	}

	if cases, typed := enum.From(value); typed {
		if unknown, comparable := enum.Unknown(cases, parameters); comparable && len(unknown) > 0 {
			return false
		}
		return cases.Valid()
	}

	if !v.RequireParameterCount(1, parameters, "enum") {
		return false
	}
	return slices.Contains(parameters, stringOf(value))
}

// ---------------------------------------------------------------------------
// Presence.
// ---------------------------------------------------------------------------

// ValidateRequired is `required`: the attribute holds an answer.
//
// Absent is one of four things, and the four are the whole rule: null, a string
// that is nothing but whitespace, an array with no members, and a file that was
// never written. A form sending three spaces has not answered the question.
func (v *Validator) ValidateRequired(attribute string, value any, parameters []string) bool {
	switch {
	case value == nil:
		return false
	case isString(value):
		return strings.TrimSpace(value.(string)) != ""
	}
	if n, ok := countOf(value); ok {
		return n >= 1
	}
	if f, ok := asFile(value); ok {
		return f.GetPath() != ""
	}
	return true
}

// ValidatePresent is `present`: the key was sent, whatever it holds.
func (v *Validator) ValidatePresent(attribute string, value any, parameters []string) bool {
	return v.data.Has(attribute)
}

// ValidateFilled is `filled`: a key that was sent holds an answer. A key that was
// not sent passes, which is the whole difference from required.
func (v *Validator) ValidateFilled(attribute string, value any, parameters []string) bool {
	if v.data.Has(attribute) {
		return v.ValidateRequired(attribute, value, nil)
	}
	return true
}

// ValidateMissing is `missing`: the key was not sent at all.
func (v *Validator) ValidateMissing(attribute string, value any, parameters []string) bool {
	return !v.data.Has(attribute)
}

// ValidateProhibited is `prohibited`: the attribute holds no answer.
func (v *Validator) ValidateProhibited(attribute string, value any, parameters []string) bool {
	return !v.ValidateRequired(attribute, value, nil)
}

// ValidateRequiredIf is `required_if`: required when the other attribute holds
// one of the given values.
//
// A key the form did not send at all passes: a form that does not carry the
// other field cannot be answering it.
func (v *Validator) ValidateRequiredIf(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(2, parameters, "required_if") {
		return false
	}
	if !v.data.Has(parameters[0]) {
		return true
	}
	values, other := v.ParseDependentRuleParameters(parameters)
	if looseContains(values, other) {
		return v.ValidateRequired(attribute, value, nil)
	}
	return true
}

// ValidateRequiredUnless is `required_unless`: required unless the other
// attribute holds one of the given values.
func (v *Validator) ValidateRequiredUnless(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(2, parameters, "required_unless") {
		return false
	}
	values, other := v.ParseDependentRuleParameters(parameters)
	if !looseContains(values, other) {
		return v.ValidateRequired(attribute, value, nil)
	}
	return true
}

// ValidateRequiredIfAccepted is `required_if_accepted`: required when the other
// attribute is accepted.
func (v *Validator) ValidateRequiredIfAccepted(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(1, parameters, "required_if_accepted") {
		return false
	}
	if v.ValidateAccepted(parameters[0], v.GetValue(parameters[0]), nil) {
		return v.ValidateRequired(attribute, value, nil)
	}
	return true
}

// ValidateRequiredIfDeclined is `required_if_declined`: required when the other
// attribute is declined.
func (v *Validator) ValidateRequiredIfDeclined(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(1, parameters, "required_if_declined") {
		return false
	}
	if v.ValidateDeclined(parameters[0], v.GetValue(parameters[0]), nil) {
		return v.ValidateRequired(attribute, value, nil)
	}
	return true
}

// ValidateRequiredWith is `required_with`: required when ANY of the named
// attributes is present.
func (v *Validator) ValidateRequiredWith(attribute string, value any, parameters []string) bool {
	if !v.AllFailingRequired(parameters) {
		return v.ValidateRequired(attribute, value, nil)
	}
	return true
}

// ValidateRequiredWithAll is `required_with_all`: required when ALL of the named
// attributes are present.
func (v *Validator) ValidateRequiredWithAll(attribute string, value any, parameters []string) bool {
	if !v.AnyFailingRequired(parameters) {
		return v.ValidateRequired(attribute, value, nil)
	}
	return true
}

// ValidateRequiredWithout is `required_without`: required when ANY of the named
// attributes is missing.
func (v *Validator) ValidateRequiredWithout(attribute string, value any, parameters []string) bool {
	if v.AnyFailingRequired(parameters) {
		return v.ValidateRequired(attribute, value, nil)
	}
	return true
}

// ValidateRequiredWithoutAll is `required_without_all`: required when ALL of the
// named attributes are missing.
func (v *Validator) ValidateRequiredWithoutAll(attribute string, value any, parameters []string) bool {
	if v.AllFailingRequired(parameters) {
		return v.ValidateRequired(attribute, value, nil)
	}
	return true
}

// AnyFailingRequired reports whether ANY of the attributes holds no answer.
func (v *Validator) AnyFailingRequired(attributes []string) bool {
	for _, key := range attributes {
		if !v.ValidateRequired(key, v.GetValue(key), nil) {
			return true
		}
	}
	return false
}

// AllFailingRequired reports whether ALL of the attributes hold no answer.
func (v *Validator) AllFailingRequired(attributes []string) bool {
	for _, key := range attributes {
		if v.ValidateRequired(key, v.GetValue(key), nil) {
			return false
		}
	}
	return true
}

// ValidatePresentIf is `present_if`: the key was sent when the other attribute
// holds one of the given values.
func (v *Validator) ValidatePresentIf(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(2, parameters, "present_if") {
		return false
	}
	values, other := v.ParseDependentRuleParameters(parameters)
	if looseContains(values, other) {
		return v.ValidatePresent(attribute, value, nil)
	}
	return true
}

// ValidatePresentUnless is `present_unless`: the key was sent unless the other
// attribute holds one of the given values.
func (v *Validator) ValidatePresentUnless(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(2, parameters, "present_unless") {
		return false
	}
	values, other := v.ParseDependentRuleParameters(parameters)
	if !looseContains(values, other) {
		return v.ValidatePresent(attribute, value, nil)
	}
	return true
}

// ValidatePresentWith is `present_with`: the key was sent when ANY of the named
// attributes was.
func (v *Validator) ValidatePresentWith(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(1, parameters, "present_with") {
		return false
	}
	if v.data.HasAny(parameters) {
		return v.ValidatePresent(attribute, value, nil)
	}
	return true
}

// ValidatePresentWithAll is `present_with_all`: the key was sent when ALL of the
// named attributes were.
func (v *Validator) ValidatePresentWithAll(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(1, parameters, "present_with_all") {
		return false
	}
	if v.data.HasAll(parameters) {
		return v.ValidatePresent(attribute, value, nil)
	}
	return true
}

// ValidateMissingIf is `missing_if`: the key was not sent when the other
// attribute holds one of the given values.
func (v *Validator) ValidateMissingIf(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(2, parameters, "missing_if") {
		return false
	}
	values, other := v.ParseDependentRuleParameters(parameters)
	if looseContains(values, other) {
		return v.ValidateMissing(attribute, value, parameters)
	}
	return true
}

// ValidateMissingUnless is `missing_unless`: the key was not sent unless the
// other attribute holds one of the given values.
func (v *Validator) ValidateMissingUnless(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(2, parameters, "missing_unless") {
		return false
	}
	values, other := v.ParseDependentRuleParameters(parameters)
	if !looseContains(values, other) {
		return v.ValidateMissing(attribute, value, parameters)
	}
	return true
}

// ValidateMissingWith is `missing_with`: the key was not sent when ANY of the
// named attributes was.
func (v *Validator) ValidateMissingWith(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(1, parameters, "missing_with") {
		return false
	}
	if v.data.HasAny(parameters) {
		return v.ValidateMissing(attribute, value, parameters)
	}
	return true
}

// ValidateMissingWithAll is `missing_with_all`: the key was not sent when ALL of
// the named attributes were.
func (v *Validator) ValidateMissingWithAll(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(1, parameters, "missing_with_all") {
		return false
	}
	if v.data.HasAll(parameters) {
		return v.ValidateMissing(attribute, value, parameters)
	}
	return true
}

// ValidateProhibitedIf is `prohibited_if`: prohibited when the other attribute
// holds one of the given values.
func (v *Validator) ValidateProhibitedIf(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(2, parameters, "prohibited_if") {
		return false
	}
	values, other := v.ParseDependentRuleParameters(parameters)
	if looseContains(values, other) {
		return !v.ValidateRequired(attribute, value, nil)
	}
	return true
}

// ValidateProhibitedIfAccepted is `prohibited_if_accepted`: prohibited when the
// other attribute is accepted.
func (v *Validator) ValidateProhibitedIfAccepted(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(1, parameters, "prohibited_if_accepted") {
		return false
	}
	if v.ValidateAccepted(parameters[0], v.GetValue(parameters[0]), nil) {
		return v.ValidateProhibited(attribute, value, nil)
	}
	return true
}

// ValidateProhibitedIfDeclined is `prohibited_if_declined`: prohibited when the
// other attribute is declined.
func (v *Validator) ValidateProhibitedIfDeclined(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(1, parameters, "prohibited_if_declined") {
		return false
	}
	if v.ValidateDeclined(parameters[0], v.GetValue(parameters[0]), nil) {
		return v.ValidateProhibited(attribute, value, nil)
	}
	return true
}

// ValidateProhibitedUnless is `prohibited_unless`: prohibited unless the other
// attribute holds one of the given values.
func (v *Validator) ValidateProhibitedUnless(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(2, parameters, "prohibited_unless") {
		return false
	}
	values, other := v.ParseDependentRuleParameters(parameters)
	if !looseContains(values, other) {
		return !v.ValidateRequired(attribute, value, nil)
	}
	return true
}

// ValidateProhibits is `prohibits`: this attribute being present forbids the
// named ones.
func (v *Validator) ValidateProhibits(attribute string, value any, parameters []string) bool {
	if v.ValidateRequired(attribute, value, nil) {
		for _, parameter := range parameters {
			if v.ValidateRequired(parameter, v.GetValue(parameter), nil) {
				return false
			}
		}
	}
	return true
}

// ---------------------------------------------------------------------------
// Exclusion. These five never put a message on a field: their failure removes
// the field from the validated data instead. See Validator.AddFailure.
// ---------------------------------------------------------------------------

// ValidateExclude is `exclude`: always excluded.
func (v *Validator) ValidateExclude(attribute string, value any, parameters []string) bool {
	return false
}

// ValidateExcludeIf is `exclude_if`: excluded when the other attribute holds one
// of the given values.
func (v *Validator) ValidateExcludeIf(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(2, parameters, "exclude_if") {
		return false
	}
	if !v.data.Has(parameters[0]) {
		return true
	}
	values, other := v.ParseDependentRuleParameters(parameters)
	return !looseContains(values, other)
}

// ValidateExcludeUnless is `exclude_unless`: excluded unless the other attribute
// holds one of the given values.
func (v *Validator) ValidateExcludeUnless(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(2, parameters, "exclude_unless") {
		return false
	}
	values, other := v.ParseDependentRuleParameters(parameters)
	return looseContains(values, other)
}

// ValidateExcludeWith is `exclude_with`: excluded when the named attribute was
// sent.
func (v *Validator) ValidateExcludeWith(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(1, parameters, "exclude_with") {
		return false
	}
	return !v.data.Has(parameters[0])
}

// ValidateExcludeWithout is `exclude_without`: excluded when the named attributes
// hold no answer.
func (v *Validator) ValidateExcludeWithout(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(1, parameters, "exclude_without") {
		return false
	}
	return !v.AnyFailingRequired(parameters)
}

// ---------------------------------------------------------------------------
// Cross-field.
// ---------------------------------------------------------------------------

// ValidateSame is `same`: this attribute and the named one hold the same value.
func (v *Validator) ValidateSame(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(1, parameters, "same") {
		return false
	}
	return sameValue(value, v.GetValue(parameters[0]))
}

// ValidateDifferent is `different`: this attribute holds none of the values the
// named ones hold. An attribute the form did not send is not compared: a field
// that is not in the form is not the same as this one.
func (v *Validator) ValidateDifferent(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(1, parameters, "different") {
		return false
	}
	for _, parameter := range parameters {
		if v.data.Has(parameter) && sameValue(value, v.GetValue(parameter)) {
			return false
		}
	}
	return true
}

// ValidateConfirmed is `confirmed`: the value is repeated by
// <attribute>_confirmation, or by the attribute the parameter names.
func (v *Validator) ValidateConfirmed(attribute string, value any, parameters []string) bool {
	other := attribute + "_confirmation"
	if len(parameters) > 0 {
		other = parameters[0]
	}
	return v.ValidateSame(attribute, value, []string{other})
}

// ValidateCurrentPassword is `current_password`: the value is the password of
// whoever is signed in.
//
// The question is asked of the CurrentPasswordChecker given to the Validator
// with WithCurrentPassword. Without one the rule fails closed, because a
// password check that passes for want of a checker is the worst outcome
// available. The parameter, when given, names the guard.
func (v *Validator) ValidateCurrentPassword(attribute string, value any, parameters []string) bool {
	if v.passwords == nil {
		return false
	}
	guard := ""
	if len(parameters) > 0 {
		guard = parameters[0]
	}
	return v.passwords.CheckCurrentPassword(v.Context(), v.grant, guard, stringOf(value))
}

// ---------------------------------------------------------------------------
// Uploads.
// ---------------------------------------------------------------------------

// ValidateFile is `file`: the value is an upload that finished.
func (v *Validator) ValidateFile(attribute string, value any, parameters []string) bool {
	return v.IsValidFileInstance(value)
}

// IsValidFileInstance reports whether the value is a File, and an upload that
// finished.
func (v *Validator) IsValidFileInstance(value any) bool {
	if up, ok := value.(UploadedFile); ok && !up.IsValid() {
		return false
	}
	_, ok := asFile(value)
	return ok
}

// ValidateMimes is `mimes`: the extension GUESSED FROM THE CONTENT is one of the
// given ones, which is why a .png renamed to .jpg does not pass. jpg and jpeg
// each stand for the other.
func (v *Validator) ValidateMimes(attribute string, value any, parameters []string) bool {
	if !v.IsValidFileInstance(value) {
		return false
	}
	allowed := parameters
	if slices.Contains(parameters, "jpg") || slices.Contains(parameters, "jpeg") {
		allowed = append(slices.Clone(parameters), "jpg", "jpeg")
	}
	f, _ := asFile(value)
	return f.GetPath() != "" && slices.Contains(allowed, f.GuessExtension())
}

// ValidateMimetypes is `mimetypes`: the media type of the content is one of the
// given ones. A parameter of the form "image/*" matches every type in the
// group.
func (v *Validator) ValidateMimetypes(attribute string, value any, parameters []string) bool {
	if !v.IsValidFileInstance(value) {
		return false
	}
	f, _ := asFile(value)
	mime := f.GetMimeType()
	group, _, _ := strings.Cut(mime, "/")
	return f.GetPath() != "" &&
		(slices.Contains(parameters, mime) || slices.Contains(parameters, group+"/*"))
}

// ValidateExtensions is `extensions`: the extension THE CLIENT SENT is one of the
// given ones. It is the counterpart of mimes, which asks the content instead.
func (v *Validator) ValidateExtensions(attribute string, value any, parameters []string) bool {
	if !v.IsValidFileInstance(value) {
		return false
	}
	if v.ShouldBlockPhpUpload(value, parameters) {
		return false
	}
	return slices.Contains(parameters, strings.ToLower(clientExtension(value)))
}

// ValidateImage is `image`: a complete PNG, JPEG or GIF decoded under fixed
// bounds, plus a well-formed SVG when the parameters carry allow_svg. Formats
// without an audited decoder fail closed even when their MIME can be detected.
func (v *Validator) ValidateImage(attribute string, value any, parameters []string) bool {
	if !v.IsValidFileInstance(value) {
		return false
	}
	f, _ := asFile(value)
	return validImage(f, slices.Contains(parameters, "allow_svg"))
}

// ShouldBlockPhpUpload reports whether the explicit `extensions` rule refuses
// the client-announced name: one of phpExtensions is, unless the rule asked for
// php by name. Content-based rules deliberately do not consult this metadata.
func (v *Validator) ShouldBlockPhpUpload(value any, parameters []string) bool {
	if slices.Contains(parameters, "php") {
		return false
	}
	return slices.Contains(phpExtensions, strings.TrimSpace(strings.ToLower(clientExtension(value))))
}

// clientExtension is the extension the upload came in with for an UploadedFile,
// and the file's own extension for anything else.
func clientExtension(value any) string {
	if up, ok := value.(UploadedFile); ok {
		return up.GetClientOriginalExtension()
	}
	if f, ok := asFile(value); ok {
		return f.GetExtension()
	}
	return ""
}

// ValidateDimensions is `dimensions`: the image measures what the named
// parameters ask for.
//
// An SVG passes without being measured once its document is validated: it has
// no pixels to count. Validating it means reading its bytes, so a file that
// announces an SVG it will not let anyone read fails instead of passing.
// Supported raster bytes are decoded under fixed input and pixel bounds before
// their dimensions are compared.
func (v *Validator) ValidateDimensions(attribute string, value any, parameters []string) bool {
	if v.IsValidFileInstance(value) {
		if f, _ := asFile(value); f.GetMimeType() == "image/svg+xml" || f.GetMimeType() == "image/svg" {
			return validSVGDocument(f)
		}
	}
	if !v.IsValidFileInstance(value) {
		return false
	}
	f, _ := asFile(value)
	width, height, ok := imageDimensions(f)
	if !ok {
		return false
	}
	if !v.RequireParameterCount(1, parameters, "dimensions") {
		return false
	}
	named := v.ParseNamedParameters(parameters)
	return !(v.FailsBasicDimensionChecks(named, width, height) ||
		v.FailsRatioCheck(named, width, height) ||
		v.FailsMinRatioCheck(named, width, height) ||
		v.FailsMaxRatioCheck(named, width, height))
}

// FailsBasicDimensionChecks reports whether the image misses any of width,
// min_width, max_width, height, min_height and max_height.
func (v *Validator) FailsBasicDimensionChecks(parameters map[string]string, width, height int) bool {
	fails := func(key string, against int, worse func(a, b int) bool) bool {
		raw, given := parameters[key]
		if !given {
			return false
		}
		n, err := strconv.Atoi(raw)
		if err != nil {
			return true
		}
		return worse(n, against)
	}
	notEqual := func(a, b int) bool { return a != b }
	above := func(a, b int) bool { return a > b }
	below := func(a, b int) bool { return a < b }

	return fails("width", width, notEqual) ||
		fails("min_width", width, above) ||
		fails("max_width", width, below) ||
		fails("height", height, notEqual) ||
		fails("min_height", height, above) ||
		fails("max_height", height, below)
}

// FailsRatioCheck reports whether the image misses the ratio. The tolerance
// widens with the image, so that a 3/2 photograph of 1201 by 800 is still 3/2.
func (v *Validator) FailsRatioCheck(parameters map[string]string, width, height int) bool {
	raw, given := parameters["ratio"]
	if !given {
		return false
	}
	want, ok := parseRatio(raw)
	if !ok || height == 0 {
		return true
	}
	precision := 1 / (max(float64(width+height)/2, float64(height)) + 1)
	return math.Abs(want-float64(width)/float64(height)) > precision
}

// FailsMinRatioCheck reports whether the image is wider than min_ratio allows.
func (v *Validator) FailsMinRatioCheck(parameters map[string]string, width, height int) bool {
	raw, given := parameters["min_ratio"]
	if !given {
		return false
	}
	want, ok := parseRatio(raw)
	if !ok || height == 0 {
		return true
	}
	return float64(width)/float64(height) > want
}

// FailsMaxRatioCheck reports whether the image is taller than max_ratio allows.
func (v *Validator) FailsMaxRatioCheck(parameters map[string]string, width, height int) bool {
	raw, given := parameters["max_ratio"]
	if !given {
		return false
	}
	want, ok := parseRatio(raw)
	if !ok || height == 0 {
		return true
	}
	return float64(width)/float64(height) < want
}

// parseRatio reads the two spellings of a ratio, "3/2" and "1.5".
func parseRatio(raw string) (float64, bool) {
	numerator, denominator, divided := strings.Cut(raw, "/")
	n, err := strconv.ParseFloat(strings.TrimSpace(numerator), 64)
	if err != nil {
		return 0, false
	}
	if !divided {
		return n, true
	}
	d, err := strconv.ParseFloat(strings.TrimSpace(denominator), 64)
	if err != nil || d == 0 {
		return 0, false
	}
	return n / d, true
}

// ParseNamedParameters reads parameters of the form "min_width=100" into a
// map.
func (v *Validator) ParseNamedParameters(parameters []string) map[string]string {
	named := make(map[string]string, len(parameters))
	for _, item := range parameters {
		key, value, _ := strings.Cut(item, "=")
		named[key] = value
	}
	return named
}

// ---------------------------------------------------------------------------
// The database. Both rules take the Grant, and the query is filtered by its
// tenant.
//
// auth.Grant is a struct and cannot be compared to nil, so what both rules check
// is the tenant on it. That is the stronger check and the one that was meant: a
// Grant carrying no tenant cannot scope the count, and a count that is not
// scoped answers whether SOMEBODY holds the value.
// ---------------------------------------------------------------------------

// ValidateExists is `exists`: the table holds a row with this value.
//
// The verifier arrives with the request, through WithPresence, together with the
// Grant -- because a rule that reads a table to answer "does this exist" is a
// read, and a read is authorized like any other. Without a verifier the rule
// fails closed.
func (v *Validator) ValidateExists(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(1, parameters, "exists") {
		return false
	}
	if v.presence == nil || auth.Tenant(v.grant) == "" {
		return false
	}
	_, table, _ := v.ParseTable(parameters[0])
	column := v.GetQueryColumn(parameters, attribute)
	extra := v.GetExtraConditions(sliceFrom(parameters, 2))

	if list, ok := asList(value); ok {
		unique := uniqueValues(list)
		if len(unique) == 0 {
			return true
		}
		count, err := v.presence.GetMultiCount(v.Context(), v.grant, table, column, unique, extra)
		return err == nil && count >= len(unique)
	}
	count, err := v.presence.GetCount(v.Context(), v.grant, table, column, value, nil, "", extra)
	return err == nil && count >= 1
}

// ValidateUnique is `unique`: the table holds no row with this value.
//
// The parameters, in order:
// unique:table,column,exceptID,idColumn,extraColumn,extraValue...
func (v *Validator) ValidateUnique(attribute string, value any, parameters []string) bool {
	if !v.RequireParameterCount(1, parameters, "unique") {
		return false
	}
	if v.presence == nil || auth.Tenant(v.grant) == "" {
		return false
	}
	_, table, idColumn := v.ParseTable(parameters[0])
	column := v.GetQueryColumn(parameters, attribute)

	var id any
	if len(parameters) > 2 {
		idColumn, id = v.GetUniqueIds(idColumn, parameters)
	}
	count, err := v.presence.GetCount(v.Context(), v.grant, table, column, value, id, idColumn, v.GetUniqueExtra(parameters))
	return err == nil && count == 0
}

// GetUniqueIds returns the column and the value of the row the rule must ignore,
// which is the row being edited.
func (v *Validator) GetUniqueIds(idColumn string, parameters []string) (string, any) {
	if idColumn == "" {
		idColumn = "id"
		if len(parameters) > 3 && parameters[3] != "" {
			idColumn = parameters[3]
		}
	}
	return idColumn, v.PrepareUniqueId(parameters[2])
}

// PrepareUniqueId reads the ignored row's id: "[field]" takes it out of the
// submitted data, "null" is null, and a whole number is a number.
func (v *Validator) PrepareUniqueId(id string) any {
	if match := bracketed.FindStringSubmatch(id); match != nil {
		return v.GetValue(match[1])
	}
	if strings.EqualFold(id, "null") {
		return nil
	}
	if n, ok := whole(id); ok {
		return n
	}
	return id
}

var bracketed = regexp.MustCompile(`\[(.*)\]`)

// GetUniqueExtra returns the extra conditions of a `unique` rule, which are its
// parameters past the fourth.
func (v *Validator) GetUniqueExtra(parameters []string) map[string]string {
	if len(parameters) > 4 {
		return v.GetExtraConditions(parameters[4:])
	}
	return nil
}

// GetExtraConditions reads the trailing parameters in pairs, column then
// value.
func (v *Validator) GetExtraConditions(segments []string) map[string]string {
	extra := make(map[string]string, len(segments)/2)
	for i := 0; i+1 < len(segments); i += 2 {
		extra[segments[i]] = segments[i+1]
	}
	if len(extra) == 0 {
		return nil
	}
	return extra
}

// ParseTable splits "connection.table", and reads a bare name as the table. A
// rule names a table, never a model: there are no models.
func (v *Validator) ParseTable(table string) (connection, name, idColumn string) {
	if before, after, found := strings.Cut(table, "."); found {
		return before, after, ""
	}
	return "", table, ""
}

// GetQueryColumn returns the column the rule queries: the second parameter when
// it names one, and the guess otherwise.
func (v *Validator) GetQueryColumn(parameters []string, attribute string) string {
	if len(parameters) > 1 && parameters[1] != "" && parameters[1] != "NULL" {
		return parameters[1]
	}
	return v.GuessColumnForQuery(attribute)
}

// GuessColumnForQuery returns the last segment of a dotted attribute, and the
// attribute itself otherwise.
func (v *Validator) GuessColumnForQuery(attribute string) string {
	if i := strings.LastIndexByte(attribute, '.'); i >= 0 {
		last := attribute[i+1:]
		if _, isNumber := whole(last); !isNumber {
			return last
		}
	}
	return attribute
}

func sliceFrom(list []string, at int) []string {
	if at >= len(list) {
		return nil
	}
	return list[at:]
}

func uniqueValues(list []any) []any {
	seen := make(map[string]struct{}, len(list))
	out := make([]any, 0, len(list))
	for _, item := range list {
		key := stringOf(item)
		if _, repeated := seen[key]; repeated {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, item)
	}
	return out
}

// ---------------------------------------------------------------------------
// The shared helpers of the trait.
// ---------------------------------------------------------------------------

// ParseDependentRuleParameters returns the values a dependent rule compares
// against, and the other attribute's value.
//
// Nothing is converted here. Reading "true", "false" and "null" as what they
// name is the comparison's job, so that one place decides what "the other field
// says yes" means.
func (v *Validator) ParseDependentRuleParameters(parameters []string) ([]string, any) {
	if len(parameters) == 0 {
		return nil, nil
	}
	return parameters[1:], v.GetValue(parameters[0])
}

// RequireParameterCount reports whether the rule was given the parameters it
// needs, and the rule that asked returns false when it was not.
//
// The count is already proven at boot by compile.go, so this is the second lock
// on a door the first one closed -- and a rule invoked directly, outside a
// compiled set, still cannot read past the end of its parameters.
func (v *Validator) RequireParameterCount(count int, parameters []string, rule string) bool {
	return len(parameters) >= count
}

// isStringOrNumber is the guard the rules that read text open with: a value that
// is neither text nor a number has nothing for them to read.
func isStringOrNumber(value any) bool {
	if _, ok := asString(value); ok {
		return true
	}
	_, ok := numberOf(value)
	return ok
}

func isString(value any) bool {
	_, ok := asString(value)
	return ok
}

// sameType reports whether the two values are the same kind of thing.
func sameType(first, second any) bool { return phpType(first) == phpType(second) }

func phpType(value any) string {
	switch value.(type) {
	case nil:
		return "NULL"
	case bool:
		return "boolean"
	case int, int64:
		return "integer"
	case float32, float64:
		return "double"
	case string:
		return "string"
	case []any, Data, map[string]any:
		return "array"
	}
	return "object"
}

// Context is the context every rule that leaves the process runs under: the
// request's, so a DNS lookup or a count query carries the request's deadline.
func (v *Validator) Context() context.Context {
	if v.ctx == nil {
		return context.Background()
	}
	return v.ctx
}

// Now is the clock the relative date keywords read. It is settable so that a
// test of `after:today` does not have to be run before midnight.
func (v *Validator) Now() time.Time {
	if v.now != nil {
		return v.now()
	}
	return time.Now()
}

func (v *Validator) resolver() Resolver {
	if v.dns != nil {
		return v.dns
	}
	return defaultResolver
}
