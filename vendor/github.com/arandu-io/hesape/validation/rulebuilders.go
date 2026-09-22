package validation

import (
	"strconv"
	"strings"
	"unicode"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/enum"
)

// This file is the rule builders: a typed way to write a rule that a caller
// could also have typed by hand.
//
// Most of them build exactly that rule string, and String is what renders it:
//
//	validation.MustCompile(validation.Rules{
//		"role":  rules.In("admin", "editor").String(),
//		"price": NewNumeric().Min(0).Max(1000).String(),
//	})
//
// Three carry a Rule suffix because the plain name is already taken: ArrayRule
// because array is a type, FileRule because File is the upload interface, and
// EmailRule because Email is the one-value helper.

// ---------------------------------------------------------------------------
// The membership rules.
// ---------------------------------------------------------------------------

// In builds the "in" rule -- the value must be one of a fixed list -- without
// the caller spelling the rule string by hand. String renders every value
// quoted, so one containing a comma survives the parameter list instead of
// splitting into two allowed values. Build one with NewIn.
type In struct {
	rule   string
	values []string
}

// NewIn returns an `in` rule over the given values.
func NewIn(values ...string) *In { return &In{rule: "in", values: values} }

// String renders the rule, every value quoted.
func (r *In) String() string { return r.rule + ":" + quotedList(r.values) }

// NotIn builds the "not_in" rule -- the value must be none of a fixed list --
// with the same quoting In uses, so a listed value containing a comma is still
// one value. Build one with NewNotIn.
type NotIn struct {
	rule   string
	values []string
}

// NewNotIn returns a `not_in` rule over the given values.
func NewNotIn(values ...string) *NotIn { return &NotIn{rule: "not_in", values: values} }

// String renders the rule, every value quoted.
func (r *NotIn) String() string { return r.rule + ":" + quotedList(r.values) }

// quotedList is the escaping both do: every value quoted, and a quote inside one
// doubled, so that a value containing a comma survives the parameter list.
func quotedList(values []string) string {
	quoted := make([]string, len(values))
	for i, value := range values {
		quoted[i] = `"` + strings.ReplaceAll(value, `"`, `""`) + `"`
	}
	return strings.Join(quoted, ",")
}

// ---------------------------------------------------------------------------
// The array rule.
// ---------------------------------------------------------------------------

// ArrayRule builds the "array" rule: the value must be an array, and when keys
// are given it may hold no key outside that list, which is how a nested object
// arriving from a form is kept to the shape it was meant to have. With no keys
// it renders the bare "array". The name carries a suffix because array is a
// type.
type ArrayRule struct{ keys []string }

// NewArrayRule returns an `array` rule, restricted to the given keys when any
// are named.
func NewArrayRule(keys ...string) *ArrayRule { return &ArrayRule{keys: keys} }

// String renders the rule, with the keys when there are any.
func (r *ArrayRule) String() string {
	if len(r.keys) == 0 {
		return "array"
	}
	return "array:" + strings.Join(r.keys, ",")
}

// ---------------------------------------------------------------------------
// The three rules settled before the request is read.
// ---------------------------------------------------------------------------

// ExcludeIf renders the "exclude" rule when its condition holds and an empty
// string when it does not, so a field can be dropped from a rule set without
// branching around the set itself. An excluded attribute is removed from the
// validated data rather than reported: it is not part of this request, which is
// a different thing from being wrong, and no message is put on it. Build one
// with NewExcludeIf.
type ExcludeIf struct{ Condition bool }

// NewExcludeIf returns an ExcludeIf over a condition already settled. A caller
// holding a closure calls it: the condition is a bool, so nothing else can be
// passed by mistake.
func NewExcludeIf(condition bool) *ExcludeIf { return &ExcludeIf{Condition: condition} }

// String renders `exclude` when the condition holds, and nothing at all when it
// does not.
func (r *ExcludeIf) String() string {
	if r.Condition {
		return "exclude"
	}
	return ""
}

// ProhibitedIf renders the "prohibited" rule when its condition holds and an
// empty string when it does not. A prohibited attribute fails whenever it
// arrived with a value at all, which refuses a field that must not accompany
// this request instead of quietly ignoring it. Build one with NewProhibitedIf.
type ProhibitedIf struct{ Condition bool }

// NewProhibitedIf returns a ProhibitedIf over a condition already settled.
func NewProhibitedIf(condition bool) *ProhibitedIf { return &ProhibitedIf{Condition: condition} }

// String renders `prohibited` when the condition holds, and nothing at all when
// it does not.
func (r *ProhibitedIf) String() string {
	if r.Condition {
		return "prohibited"
	}
	return ""
}

// RequiredIf renders the "required" rule when its condition holds and an empty
// string when it does not, so a field is demanded only in the case the caller
// decides. The condition is a bool settled before the rule set is built; a
// condition that has to look at the request is required_if, which is a rule name
// and takes the other field as its parameter. Build one with NewRequiredIf.
type RequiredIf struct{ Condition bool }

// NewRequiredIf returns a RequiredIf over a condition already settled.
func NewRequiredIf(condition bool) *RequiredIf { return &RequiredIf{Condition: condition} }

// String renders `required` when the condition holds, and nothing at all when it
// does not.
func (r *RequiredIf) String() string {
	if r.Condition {
		return "required"
	}
	return ""
}

// ---------------------------------------------------------------------------
// The date rules.
// ---------------------------------------------------------------------------

// Date builds the date rules of one field, without the caller remembering their
// names. Build one with NewDate.
type Date struct {
	format      string
	constraints []string
}

// NewDate returns a Date carrying no constraint yet.
func NewDate() *Date { return &Date{} }

// Format sets the layout the value has to be written in, which makes the rule
// date_format rather than date. It is a GO layout -- Format("2006-01-02"),
// never "Y-m-d".
func (r *Date) Format(format string) *Date {
	r.format = format

	return r
}

// BeforeToday adds before:today.
func (r *Date) BeforeToday() *Date { return r.Before("today") }

// AfterToday adds after:today.
func (r *Date) AfterToday() *Date { return r.After("today") }

// TodayOrBefore adds before_or_equal:today.
func (r *Date) TodayOrBefore() *Date { return r.BeforeOrEqual("today") }

// TodayOrAfter adds after_or_equal:today.
func (r *Date) TodayOrAfter() *Date { return r.AfterOrEqual("today") }

// Before adds before, against the given moment.
func (r *Date) Before(date string) *Date { return r.addRule("before:" + date) }

// After adds after, against the given moment.
func (r *Date) After(date string) *Date { return r.addRule("after:" + date) }

// BeforeOrEqual adds before_or_equal, against the given moment.
func (r *Date) BeforeOrEqual(date string) *Date { return r.addRule("before_or_equal:" + date) }

// AfterOrEqual adds after_or_equal, against the given moment.
func (r *Date) AfterOrEqual(date string) *Date { return r.addRule("after_or_equal:" + date) }

// Between adds after and before, so both ends are exclusive.
func (r *Date) Between(from, to string) *Date { return r.After(from).Before(to) }

// BetweenOrEqual adds after_or_equal and before_or_equal, so both ends are
// inclusive.
func (r *Date) BetweenOrEqual(from, to string) *Date {
	return r.AfterOrEqual(from).BeforeOrEqual(to)
}

func (r *Date) addRule(rules string) *Date {
	r.constraints = append(r.constraints, rules)

	return r
}

// String renders the chain, starting with date or with date_format.
func (r *Date) String() string {
	head := "date"
	if r.format != "" {
		head = "date_format:" + r.format
	}
	return strings.Join(append([]string{head}, r.constraints...), "|")
}

// ---------------------------------------------------------------------------
// The numeric rules.
// ---------------------------------------------------------------------------

// Numeric builds a whole chain of numeric rules for one field -- bounds, digit
// counts, decimal places, comparisons against another field -- without the
// caller remembering any of their names. It always begins with "numeric", and
// String renders the chain with repeats dropped, because several of the methods
// add "integer" of their own. Build one with NewNumeric.
type Numeric struct{ constraints []string }

// NewNumeric returns a Numeric carrying `numeric` and nothing else yet.
func NewNumeric() *Numeric { return &Numeric{constraints: []string{"numeric"}} }

// Between adds between, both bounds inclusive.
func (r *Numeric) Between(min, max float64) *Numeric {
	return r.addRule("between:" + number64(min) + "," + number64(max))
}

// Decimal adds decimal: exactly min places, or between min and max of them.
func (r *Numeric) Decimal(min int, max ...int) *Numeric {
	rule := "decimal:" + strconv.Itoa(min)
	if len(max) > 0 {
		rule += "," + strconv.Itoa(max[0])
	}
	return r.addRule(rule)
}

// Different adds different, against the named field.
func (r *Numeric) Different(field string) *Numeric { return r.addRule("different:" + field) }

// Digits adds integer and digits: exactly this many of them.
func (r *Numeric) Digits(length int) *Numeric {
	return r.Integer().addRule("digits:" + strconv.Itoa(length))
}

// DigitsBetween adds integer and digits_between.
func (r *Numeric) DigitsBetween(min, max int) *Numeric {
	return r.Integer().addRule("digits_between:" + strconv.Itoa(min) + "," + strconv.Itoa(max))
}

// GreaterThan adds gt, against the named field.
func (r *Numeric) GreaterThan(field string) *Numeric { return r.addRule("gt:" + field) }

// GreaterThanOrEqualTo adds gte, against the named field.
func (r *Numeric) GreaterThanOrEqualTo(field string) *Numeric { return r.addRule("gte:" + field) }

// Integer adds integer.
func (r *Numeric) Integer() *Numeric { return r.addRule("integer") }

// LessThan adds lt, against the named field.
func (r *Numeric) LessThan(field string) *Numeric { return r.addRule("lt:" + field) }

// LessThanOrEqualTo adds lte, against the named field.
func (r *Numeric) LessThanOrEqualTo(field string) *Numeric { return r.addRule("lte:" + field) }

// Max adds max.
func (r *Numeric) Max(value float64) *Numeric { return r.addRule("max:" + number64(value)) }

// MaxDigits adds max_digits.
func (r *Numeric) MaxDigits(value int) *Numeric {
	return r.addRule("max_digits:" + strconv.Itoa(value))
}

// Min adds min.
func (r *Numeric) Min(value float64) *Numeric { return r.addRule("min:" + number64(value)) }

// MinDigits adds min_digits.
func (r *Numeric) MinDigits(value int) *Numeric {
	return r.addRule("min_digits:" + strconv.Itoa(value))
}

// MultipleOf adds multiple_of.
func (r *Numeric) MultipleOf(value float64) *Numeric {
	return r.addRule("multiple_of:" + number64(value))
}

// Same adds same, against the named field.
func (r *Numeric) Same(field string) *Numeric { return r.addRule("same:" + field) }

// Exactly adds integer and size.
func (r *Numeric) Exactly(value int) *Numeric {
	return r.Integer().addRule("size:" + strconv.Itoa(value))
}

func (r *Numeric) addRule(rules string) *Numeric {
	r.constraints = append(r.constraints, rules)

	return r
}

// String renders the chain, with the repeats dropped -- Digits and Exactly both
// add `integer`.
func (r *Numeric) String() string {
	seen := make(map[string]struct{}, len(r.constraints))
	out := make([]string, 0, len(r.constraints))
	for _, constraint := range r.constraints {
		if _, repeated := seen[constraint]; repeated {
			continue
		}
		seen[constraint] = struct{}{}
		out = append(out, constraint)
	}
	return strings.Join(out, "|")
}

// number64 renders a bound without a trailing zero.
func number64(value float64) string { return strconv.FormatFloat(value, 'f', -1, 64) }

// ---------------------------------------------------------------------------
// The image dimension rules.
// ---------------------------------------------------------------------------

// Dimensions builds the "dimensions" rule: the pixel width and height an
// uploaded image must have or stay within, and the aspect ratio it must match.
// Each method sets one constraint, and String renders them all as a single rule
// in a fixed order rather than the order they were set, since a Go map remembers
// none. Build one with NewDimensions.
type Dimensions struct{ constraints map[string]string }

// NewDimensions returns a Dimensions carrying no constraint yet.
func NewDimensions() *Dimensions { return &Dimensions{constraints: map[string]string{}} }

// Width sets the exact width, in pixels.
func (r *Dimensions) Width(value int) *Dimensions { return r.set("width", value) }

// Height sets the exact height, in pixels.
func (r *Dimensions) Height(value int) *Dimensions { return r.set("height", value) }

// MinWidth sets the lowest width, in pixels.
func (r *Dimensions) MinWidth(value int) *Dimensions { return r.set("min_width", value) }

// MinHeight sets the lowest height, in pixels.
func (r *Dimensions) MinHeight(value int) *Dimensions { return r.set("min_height", value) }

// MaxWidth sets the highest width, in pixels.
func (r *Dimensions) MaxWidth(value int) *Dimensions { return r.set("max_width", value) }

// MaxHeight sets the highest height, in pixels.
func (r *Dimensions) MaxHeight(value int) *Dimensions { return r.set("max_height", value) }

// Ratio sets the exact aspect ratio, width over height.
func (r *Dimensions) Ratio(value float64) *Dimensions { return r.setRatio("ratio", value) }

// MinRatio sets the lowest aspect ratio.
func (r *Dimensions) MinRatio(value float64) *Dimensions { return r.setRatio("min_ratio", value) }

// MaxRatio sets the highest aspect ratio.
func (r *Dimensions) MaxRatio(value float64) *Dimensions { return r.setRatio("max_ratio", value) }

// RatioBetween sets both ends of the aspect ratio.
func (r *Dimensions) RatioBetween(min, max float64) *Dimensions {
	return r.setRatio("min_ratio", min).setRatio("max_ratio", max)
}

func (r *Dimensions) set(key string, value int) *Dimensions {
	r.constraints[key] = strconv.Itoa(value)

	return r
}

func (r *Dimensions) setRatio(key string, value float64) *Dimensions {
	r.constraints[key] = number64(value)

	return r
}

// dimensionOrder is the order the constraints are written in. A Go map remembers
// no order, so the spelling is fixed here rather than left to the map.
var dimensionOrder = []string{
	"width", "height", "min_width", "min_height",
	"max_width", "max_height", "ratio", "min_ratio", "max_ratio",
}

// String renders the constraints as one rule, in dimensionOrder.
func (r *Dimensions) String() string {
	pairs := make([]string, 0, len(r.constraints))
	for _, key := range dimensionOrder {
		if value, set := r.constraints[key]; set {
			pairs = append(pairs, key+"="+value)
		}
	}
	return "dimensions:" + strings.Join(pairs, ",")
}

// ---------------------------------------------------------------------------
// The upload rules.
// ---------------------------------------------------------------------------

// FileRule builds the rules an upload has to pass: what it may be, and how big
// it may be. It carries the suffix because File is already the upload interface.
// Build one with NewFileRule, or with NewImageFile.
type FileRule struct {
	allowedMimetypes  []string
	allowedExtensions []string
	minimumFileSize   *int
	maximumFileSize   *int
	customRules       []string
	image             bool
	allowSvg          bool
}

// NewFileRule returns a FileRule that asks only for an upload that finished.
func NewFileRule() *FileRule { return &FileRule{} }

// NewImageFile returns a FileRule that asks for an image, and for an SVG too
// when allowSvg is true.
func NewImageFile(allowSvg ...bool) *FileRule {
	r := &FileRule{image: true}
	if len(allowSvg) > 0 {
		r.allowSvg = allowSvg[0]
	}
	return r
}

// Types returns a FileRule over the media types or extensions the upload may
// be.
func Types(mimetypes ...string) *FileRule {
	return &FileRule{allowedMimetypes: mimetypes}
}

// Extensions sets the extensions the upload may carry. It is the extension the
// BROWSER sent, which is not the same question the content asks.
func (r *FileRule) Extensions(extensions ...string) *FileRule {
	r.allowedExtensions = extensions

	return r
}

// Size sets the exact size, in kilobytes.
func (r *FileRule) Size(size int) *FileRule {
	r.minimumFileSize, r.maximumFileSize = &size, &size

	return r
}

// Between sets both ends of the size, in kilobytes.
func (r *FileRule) Between(minSize, maxSize int) *FileRule {
	r.minimumFileSize, r.maximumFileSize = &minSize, &maxSize

	return r
}

// Min sets the smallest size, in kilobytes.
func (r *FileRule) Min(size int) *FileRule {
	r.minimumFileSize = &size

	return r
}

// Max sets the largest size, in kilobytes.
func (r *FileRule) Max(size int) *FileRule {
	r.maximumFileSize = &size

	return r
}

// Dimensions adds the dimension rule the builder rendered.
func (r *FileRule) Dimensions(dimensions *Dimensions) *FileRule {
	return r.Rules(dimensions.String())
}

// Rules merges more rules into the ones this builds.
func (r *FileRule) Rules(rules ...string) *FileRule {
	r.customRules = append(r.customRules, rules...)

	return r
}

// ToKilobytes reads a size written with a suffix: "2mb" is 2000 kilobytes -- a
// thousand, not 1024.
//
// The bool is false for a suffix it does not know, and for a number written with
// no suffix at all.
func ToKilobytes(size string) (int, bool) {
	size = strings.ToLower(strings.TrimSpace(size))

	for suffix, multiplier := range map[string]float64{
		"kb": 1, "mb": 1_000, "gb": 1_000_000, "tb": 1_000_000_000,
	} {
		rest, matched := strings.CutSuffix(size, suffix)
		if !matched {
			continue
		}
		value, err := strconv.ParseFloat(strings.TrimSpace(rest), 64)
		if err != nil {
			return 0, false
		}
		return int(value*multiplier + 0.5), true
	}
	return 0, false
}

// String renders the chain: what the upload may be, then how big it may be, then
// whatever Rules added.
func (r *FileRule) String() string {
	head := "file"
	if r.image {
		head = "image"
		if r.allowSvg {
			head = "image:allow_svg"
		}
	}
	rules := []string{head}

	rules = append(rules, r.buildMimetypes()...)

	if len(r.allowedExtensions) > 0 {
		lowered := make([]string, len(r.allowedExtensions))
		for i, extension := range r.allowedExtensions {
			lowered[i] = strings.ToLower(extension)
		}
		rules = append(rules, "extensions:"+strings.Join(lowered, ","))
	}

	switch {
	case r.minimumFileSize == nil && r.maximumFileSize == nil:
	case r.maximumFileSize == nil:
		rules = append(rules, "min:"+strconv.Itoa(*r.minimumFileSize))
	case r.minimumFileSize == nil:
		rules = append(rules, "max:"+strconv.Itoa(*r.maximumFileSize))
	case *r.minimumFileSize != *r.maximumFileSize:
		rules = append(rules, "between:"+strconv.Itoa(*r.minimumFileSize)+","+strconv.Itoa(*r.maximumFileSize))
	default:
		rules = append(rules, "size:"+strconv.Itoa(*r.minimumFileSize))
	}

	return strings.Join(append(rules, r.customRules...), "|")
}

// buildMimetypes splits what Types was given: a name with a slash is a media type
// and everything else is an extension, so the two become two rules.
func (r *FileRule) buildMimetypes() []string {
	if len(r.allowedMimetypes) == 0 {
		return nil
	}

	var mimetypes, mimes []string
	for _, allowed := range r.allowedMimetypes {
		if strings.Contains(allowed, "/") {
			mimetypes = append(mimetypes, allowed)
			continue
		}
		mimes = append(mimes, allowed)
	}

	var rules []string
	if len(mimetypes) > 0 {
		rules = append(rules, "mimetypes:"+strings.Join(mimetypes, ","))
	}
	if len(mimes) > 0 {
		rules = append(rules, "mimes:"+strings.Join(mimes, ","))
	}
	return rules
}

// ---------------------------------------------------------------------------
// The email rule.
// ---------------------------------------------------------------------------

// EmailRule builds the "email" rule together with the list of checks it should
// run: the RFC reading, a stricter one, an MX record lookup on the domain, the
// spoofing check, and the native one. With nothing asked for it renders the bare
// "email", which is the shape check alone. Build one with NewEmailRule. It
// carries the suffix because Email is already the one-value helper here.
type EmailRule struct{ validations []string }

// NewEmailRule returns an EmailRule asking for the shape check alone.
func NewEmailRule() *EmailRule { return &EmailRule{} }

// RfcCompliant asks for the RFC reading, or for the stricter one when strict is
// true.
func (r *EmailRule) RfcCompliant(strict ...bool) *EmailRule {
	if len(strict) > 0 && strict[0] {
		return r.add("strict")
	}
	return r.add("rfc")
}

// Strict asks for the stricter reading.
func (r *EmailRule) Strict() *EmailRule { return r.add("strict") }

// ValidateMxRecord asks for the lookup on the domain.
func (r *EmailRule) ValidateMxRecord() *EmailRule { return r.add("dns") }

// PreventSpoofing asks for the spoofing check.
func (r *EmailRule) PreventSpoofing() *EmailRule { return r.add("spoof") }

// WithNativeValidation asks for the native check, allowing unicode when told
// to.
func (r *EmailRule) WithNativeValidation(allowUnicode ...bool) *EmailRule {
	if len(allowUnicode) > 0 && allowUnicode[0] {
		return r.add("filter_unicode")
	}
	return r.add("filter")
}

// Rules merges more checks into the ones this asks for.
func (r *EmailRule) Rules(rules ...string) *EmailRule {
	r.validations = append(r.validations, rules...)

	return r
}

func (r *EmailRule) add(validation string) *EmailRule {
	r.validations = append(r.validations, validation)

	return r
}

// String renders the chain. With nothing asked for it is the bare `email`, which
// is the shape check.
func (r *EmailRule) String() string {
	if len(r.validations) == 0 {
		return "email"
	}
	return "email:" + strings.Join(r.validations, ",")
}

// ---------------------------------------------------------------------------
// The two database rules, over what they share.
// ---------------------------------------------------------------------------

// DatabaseRule holds what the two database rules have in common: the table and
// the column to look in, the extra conditions that narrow the search, and the
// query callbacks registered through Using. Unique and Exists embed it rather
// than repeat it, so the Where methods below are written once and read as
// methods on either; FormatWheres renders the conditions into the parameter list
// each of them builds. It is not useful on its own.
type DatabaseRule struct {
	table  string
	column string
	wheres [][2]string

	// using is the query callbacks Using registers.
	using []func(query any)
}

// ResolveTableName returns the table a rule queries. There are no models here, so
// a table name is already the table name.
func (r *DatabaseRule) ResolveTableName(table string) string { return table }

// Where narrows the search to rows holding value in column.
func (r *DatabaseRule) Where(column, value string) *DatabaseRule {
	r.wheres = append(r.wheres, [2]string{column, value})

	return r
}

// WhereNot narrows the search to rows NOT holding value in column.
func (r *DatabaseRule) WhereNot(column, value string) *DatabaseRule {
	return r.Where(column, "!"+value)
}

// WhereNull narrows the search to rows holding null in column.
func (r *DatabaseRule) WhereNull(column string) *DatabaseRule { return r.Where(column, "NULL") }

// WhereNotNull narrows the search to rows holding anything but null in column.
func (r *DatabaseRule) WhereNotNull(column string) *DatabaseRule {
	return r.Where(column, "NOT_NULL")
}

// WhereIn narrows the search to rows holding one of the values in column.
func (r *DatabaseRule) WhereIn(column string, values ...string) *DatabaseRule {
	return r.Where(column, strings.Join(values, ","))
}

// WhereNotIn narrows the search to rows holding none of the values in column.
func (r *DatabaseRule) WhereNotIn(column string, values ...string) *DatabaseRule {
	return r.Where(column, "!"+strings.Join(values, ","))
}

// WithoutTrashed narrows the search to rows that are not soft deleted. The
// column defaults to deleted_at.
func (r *DatabaseRule) WithoutTrashed(deletedAtColumn ...string) *DatabaseRule {
	return r.WhereNull(columnOr(deletedAtColumn, "deleted_at"))
}

// OnlyTrashed narrows the search to rows that are soft deleted. The column
// defaults to deleted_at.
func (r *DatabaseRule) OnlyTrashed(deletedAtColumn ...string) *DatabaseRule {
	return r.WhereNotNull(columnOr(deletedAtColumn, "deleted_at"))
}

// FormatWheres renders the conditions into the trailing parameters of the rule,
// column then value, every value quoted.
func (r *DatabaseRule) FormatWheres() string {
	pairs := make([]string, len(r.wheres))
	for i, where := range r.wheres {
		pairs[i] = where[0] + `,"` + strings.ReplaceAll(where[1], `"`, `""`) + `"`
	}
	return strings.Join(pairs, ",")
}

func columnOr(given []string, def string) string {
	if len(given) > 0 && given[0] != "" {
		return given[0]
	}
	return def
}

// Unique builds the "unique" rule: no row in the table may already hold this
// value in the column. Ignore names the one row allowed to hold it, which is how
// a form that edits an existing record does not collide with itself. The check
// is a read of the table, so it runs only with a Grant carrying a tenant and
// counts only that tenant's rows. Build one with NewUnique.
type Unique struct {
	DatabaseRule
	ignore   string
	idColumn string
}

// NewUnique returns a `unique` rule over the table. The column defaults to
// "NULL", which the rule reads as the attribute's own name.
func NewUnique(table string, column ...string) *Unique {
	return &Unique{
		DatabaseRule: DatabaseRule{table: table, column: columnOr(column, "NULL")},
		idColumn:     "id",
	}
}

// Ignore names the row this check is allowed to find, which is the row being
// edited.
func (r *Unique) Ignore(id string, idColumn ...string) *Unique {
	r.ignore = id
	r.idColumn = columnOr(idColumn, "id")

	return r
}

// String renders the rule, with the ignored row and the conditions.
func (r *Unique) String() string {
	ignore := "NULL"
	if r.ignore != "" {
		ignore = `"` + r.ignore + `"`
	}
	return strings.TrimRight(strings.Join([]string{
		"unique:" + r.table, r.column, ignore, r.idColumn, r.FormatWheres(),
	}, ","), ",")
}

// Exists builds the "exists" rule: some row in the table must already hold this
// value in the column, which is how an identifier arriving from a form is
// checked before anything is written against it. The check is a read of the
// table, so it runs only with a Grant carrying a tenant and counts only that
// tenant's rows -- an identifier belonging to somebody else does not exist as
// far as this rule is concerned. Build one with NewExists.
type Exists struct{ DatabaseRule }

// NewExists returns an `exists` rule over the table. The column defaults to
// "NULL", which the rule reads as the attribute's own name.
func NewExists(table string, column ...string) *Exists {
	return &Exists{DatabaseRule{table: table, column: columnOr(column, "NULL")}}
}

// String renders the rule, with the conditions.
func (r *Exists) String() string {
	return strings.TrimRight(strings.Join([]string{
		"exists:" + r.table, r.column, r.FormatWheres(),
	}, ","), ",")
}

// ---------------------------------------------------------------------------
// The password rule.
// ---------------------------------------------------------------------------

// UncompromisedVerifier answers whether a password has appeared in a data leak.
//
// Asking a breach service is a network call and belongs with whoever owns the
// HTTP client, so what is here is only the question.
type UncompromisedVerifier interface {
	// Verify reports false when the value has appeared more times than the
	// threshold allows.
	Verify(value string, threshold int) bool
}

// Password is the password policy of one field: a length, the kinds of character
// it must carry, and whether it has appeared in a leak.
//
// It cannot be a rule string: it holds a verifier, and it says four different
// things depending on which part failed. It runs through Validator.After:
//
//	v.After(func(v *validation.Validator) {
//		v.ValidateUsingCustomRule("password", v.GetValue("password"), rule)
//	})
type Password struct {
	min           int
	max           int
	mixedCase     bool
	letters       bool
	numbers       bool
	symbols       bool
	uncompromised bool

	compromisedThreshold int
	verifier             UncompromisedVerifier

	customRules []string
	messages    []string

	validator *Validator
	data      Data
}

// defaultPasswordCallback is what PasswordDefaults registered, and what
// PasswordDefault answers with.
var defaultPasswordCallback func() *Password

// NewPassword returns a policy asking for a minimum length. A minimum below one
// is raised to one.
func NewPassword(min int) *Password {
	return &Password{min: max(min, 1)}
}

// PasswordMin returns a policy asking for a minimum length.
func PasswordMin(size int) *Password { return NewPassword(size) }

// PasswordDefaults registers the policy every later PasswordDefault answers
// with.
func PasswordDefaults(callback func() *Password) { defaultPasswordCallback = callback }

// PasswordDefault returns the registered policy, or a minimum of eight when none
// was registered.
func PasswordDefault() *Password {
	if defaultPasswordCallback == nil {
		return PasswordMin(8)
	}
	if configured := defaultPasswordCallback(); configured != nil {
		return configured
	}
	return PasswordMin(8)
}

// PasswordRequired returns the `required` rule string and the default policy,
// which are the two things a required password field takes.
func PasswordRequired() (string, *Password) { return "required", PasswordDefault() }

// PasswordSometimes returns the `sometimes` rule string and the default policy.
func PasswordSometimes() (string, *Password) { return "sometimes", PasswordDefault() }

// Max sets the longest the password may be.
func (p *Password) Max(size int) *Password {
	p.max = size

	return p
}

// Uncompromised refuses a password that has appeared in a data leak more times
// than the threshold allows.
//
// The verifier is a parameter because nothing here resolves one. A nil verifier
// FAILS the check rather than passing it: "we could not ask" is not "it is
// safe".
func (p *Password) Uncompromised(verifier UncompromisedVerifier, threshold ...int) *Password {
	p.uncompromised = true
	p.verifier = verifier
	if len(threshold) > 0 {
		p.compromisedThreshold = threshold[0]
	}

	return p
}

// MixedCase asks for at least one letter of each case.
func (p *Password) MixedCase() *Password {
	p.mixedCase = true

	return p
}

// Letters asks for at least one letter.
func (p *Password) Letters() *Password {
	p.letters = true

	return p
}

// Numbers asks for at least one number.
func (p *Password) Numbers() *Password {
	p.numbers = true

	return p
}

// Symbols asks for at least one symbol.
func (p *Password) Symbols() *Password {
	p.symbols = true

	return p
}

// Rules merges more rules into the ones this policy enforces.
func (p *Password) Rules(rules ...string) *Password {
	p.customRules = append(p.customRules, rules...)

	return p
}

// AppliedRules returns what this policy is currently asking for, which is what a
// "your password must" list on the form is drawn from.
func (p *Password) AppliedRules() map[string]any {
	return map[string]any{
		"min":                  p.min,
		"max":                  p.max,
		"mixedCase":            p.mixedCase,
		"letters":              p.letters,
		"numbers":              p.numbers,
		"symbols":              p.symbols,
		"uncompromised":        p.uncompromised,
		"compromisedThreshold": p.compromisedThreshold,
		"customRules":          p.customRules,
	}
}

// SetValidator hands the rule the validator running it, so that its messages
// come from the same translator.
func (p *Password) SetValidator(validator *Validator) { p.validator = validator }

// SetData hands the rule the data being validated.
func (p *Password) SetData(data Data) { p.data = data }

// Passes reports whether the value satisfies the policy, and Message says what
// failed when it does not.
//
// The length and the merged rules are checked by a sibling validator compiled
// from the same chain; the four character checks run after it.
func (p *Password) Passes(attribute string, value any) bool {
	p.messages = nil

	chain := "string|min:" + strconv.Itoa(p.min)
	if p.max > 0 {
		chain += "|max:" + strconv.Itoa(p.max)
	}
	for _, rule := range p.customRules {
		if rule != "" {
			chain += "|" + rule
		}
	}

	data := p.data
	if data == nil {
		data = Data{}
	}
	data = data.Clone()
	data[attribute] = value

	set, err := Compile(Rules{attribute: chain})
	if err != nil {
		return p.fail(err.Error())
	}

	sibling := Make(data, set, p.siblingOptions()...)

	if sibling.Fails() {
		return p.fail(sibling.Errors().All()...)
	}

	text, isString := value.(string)
	if !isString {
		return true
	}

	if p.mixedCase && !hasMixedCase(text) {
		p.fail(p.line("validation.password.mixed",
			"The :attribute field must contain at least one uppercase and one lowercase letter."))
	}
	if p.letters && !containsRune(text, unicode.IsLetter) {
		p.fail(p.line("validation.password.letters",
			"The :attribute field must contain at least one letter."))
	}
	if p.symbols && !containsRune(text, isSymbolRune) {
		p.fail(p.line("validation.password.symbols",
			"The :attribute field must contain at least one symbol."))
	}
	if p.numbers && !containsRune(text, unicode.IsNumber) {
		p.fail(p.line("validation.password.numbers",
			"The :attribute field must contain at least one number."))
	}

	if len(p.messages) > 0 {
		return false
	}

	if p.uncompromised && (p.verifier == nil || !p.verifier.Verify(text, p.compromisedThreshold)) {
		return p.fail(p.line("validation.password.uncompromised",
			"The given :attribute has appeared in a data leak. Please choose a different :attribute."))
	}

	return true
}

// siblingOptions carries the overrides of the validator this rule belongs to
// into the sibling, so that both say the same things.
func (p *Password) siblingOptions() []ValidatorOption {
	if p.validator == nil {
		return nil
	}
	return []ValidatorOption{
		WithTranslator(p.validator.trans),
		WithCustomMessages(p.validator.CustomMessages()),
		WithCustomAttributes(p.validator.CustomAttributes()),
	}
}

// line reads one of the password lines out of the translator, falling back to
// the English sentence -- which is what Rules\Enum, Rules\Can and Rules\AnyOf do
// with their own line.
func (p *Password) line(key, fallback string) string {
	if p.validator == nil || p.validator.trans == nil {
		return fallback
	}
	if sentence := line(p.validator.GetTranslator().Get(key, nil, ""), key); sentence != key {
		return sentence
	}
	return fallback
}

// Message returns what the last Passes refused, one sentence per failed check.
func (p *Password) Message() []string { return p.messages }

// fail records the messages and reports false, so a check can return it
// directly.
func (p *Password) fail(messages ...string) bool {
	p.messages = append(p.messages, messages...)

	return false
}

// hasMixedCase reports at least one letter of each case, in either order.
func hasMixedCase(value string) bool {
	var lower, upper bool
	for _, r := range value {
		if unicode.IsLower(r) {
			lower = true
		}
		if unicode.IsUpper(r) {
			upper = true
		}
	}
	return lower && upper
}

// isSymbolRune reports a separator, a symbol or a punctuation mark.
func isSymbolRune(r rune) bool {
	return unicode.IsSpace(r) || unicode.Is(unicode.Z, r) ||
		unicode.IsSymbol(r) || unicode.IsPunct(r)
}

func containsRune(value string, is func(rune) bool) bool {
	for _, r := range value {
		if is(r) {
			return true
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// The three rules that need more than a rule string.
// ---------------------------------------------------------------------------

// Enum is the rule that the value must be one of the cases of a type.
//
// Only and Except then narrow them. Build it with EnumOf, which reads the cases
// off the type, or with NewEnum when there is no type to read -- a set that
// lives in a column and nowhere else.
type Enum struct {
	cases  []string
	only   []string
	except []string

	validator *Validator
}

// NewEnum returns an Enum over the given cases.
//
// The cases are text, so the rule can only compare text. Prefer EnumOf wherever
// a generated type exists: a list written beside a type is a second copy of the
// set, and the two disagree the first time a case is added to one of them.
func NewEnum(cases ...string) *Enum { return &Enum{cases: cases} }

// EnumOf returns an Enum over the cases of a generated enum type.
//
//	validation.EnumOf(enums.InvoiceStatusValues()...)
//
// The cases are derived rather than written, so the rule cannot disagree with
// the type -- which is the failure a hand-written list produces and nothing
// reports. It takes the values because the generated list of them is a package
// function, the one place that knows the declaration order.
func EnumOf[E enum.Enum](values ...E) *Enum { return &Enum{cases: enum.Names(values...)} }

// Only narrows the cases to these, and nothing else passes.
func (r *Enum) Only(values ...string) *Enum {
	r.only = values

	return r
}

// Except narrows the cases by removing these.
func (r *Enum) Except(values ...string) *Enum {
	r.except = values

	return r
}

// Passes reports whether the value is one of the cases, after Only and Except
// have narrowed them.
//
// A value of the enum type is asked rather than compared: stringOf renders a
// named type as nothing, so comparing text used to refuse every typed value the
// rule was built for. Only and Except still narrow, and they narrow on the
// shown spelling, which is what the caller wrote them in.
func (r *Enum) Passes(attribute string, value any) bool {
	if value == nil {
		return false
	}

	if cases, typed := enum.From(value); typed {
		if !cases.Valid() {
			return false
		}
		return r.isDesirable(cases.String())
	}

	// An untyped value -- a string off a form -- is compared against the cases,
	// which EnumOf derived from the type and cannot have got wrong.
	text := stringOf(value)

	var known bool
	for _, c := range r.cases {
		if c == text {
			known = true
			break
		}
	}
	if !known {
		return false
	}
	return r.isDesirable(text)
}

// isDesirable applies Only, then Except, to a value already known to be a case.
func (r *Enum) isDesirable(value string) bool {
	switch {
	case len(r.only) > 0:
		return containsString(r.only, value)
	case len(r.except) > 0:
		return !containsString(r.except, value)
	}
	return true
}

// Message returns the sentence for a value outside the cases.
func (r *Enum) Message() []string {
	return []string{translatedOr(r.validator, "validation.enum", "The selected :attribute is invalid.")}
}

// SetValidator hands the rule the validator running it, so that its message
// comes from the same translator.
func (r *Enum) SetValidator(validator *Validator) { r.validator = validator }

// Can is the rule that the value is one the current subject is allowed to
// choose.
//
// The question is asked through the callback the caller gives, against the
// Grant the validator carries. Build one with NewCan.
type Can struct {
	ability   string
	arguments []string
	allows    func(g auth.Grant, ability string, arguments []string, value any) bool

	validator *Validator
}

// NewCan returns a Can that asks allows about the ability.
func NewCan(allows func(g auth.Grant, ability string, arguments []string, value any) bool, ability string, arguments ...string) *Can {
	return &Can{ability: ability, arguments: arguments, allows: allows}
}

// Passes reports what the callback says about the Grant, the ability and the
// value.
//
// A missing callback FAILS rather than passes: "nobody wired the authorizer" is
// not "everybody is allowed".
func (r *Can) Passes(attribute string, value any) bool {
	if r.allows == nil || r.validator == nil {
		return false
	}
	return r.allows(r.validator.grant, r.ability, r.arguments, value)
}

// Message returns the sentence for a value the subject may not choose.
func (r *Can) Message() []string {
	return []string{translatedOr(r.validator, "validation.can",
		"The :attribute field contains an unauthorized value.")}
}

// SetValidator hands the rule the validator running it, which is where the Grant
// and the translator come from.
func (r *Can) SetValidator(validator *Validator) { r.validator = validator }

// AnyOf is the rule that the value passes when it passes any one of the given
// rule sets. Build one with NewAnyOf.
type AnyOf struct {
	sets []*Set

	validator *Validator
}

// NewAnyOf returns an AnyOf over the given sets.
func NewAnyOf(sets ...*Set) *AnyOf { return &AnyOf{sets: sets} }

// Passes reports whether the value passes any one of the sets, each run over the
// validator's data with this attribute replaced.
func (r *AnyOf) Passes(attribute string, value any) bool {
	for _, set := range r.sets {
		if set == nil {
			continue
		}
		data := Data{}
		if r.validator != nil {
			data = r.validator.GetData().Clone()
		}
		data[attribute] = value

		if Make(data, set).Passes() {
			return true
		}
	}
	return false
}

// Message returns the sentence for a value that passed none of the sets.
func (r *AnyOf) Message() []string {
	return []string{translatedOr(r.validator, "validation.any_of", "The :attribute field is invalid.")}
}

// SetValidator hands the rule the validator running it, which is where the data
// and the translator come from.
func (r *AnyOf) SetValidator(validator *Validator) { r.validator = validator }

// translatedOr reads the line out of the translator, falling back to the English
// sentence when there is no translator or no line under the key.
func translatedOr(v *Validator, key, fallback string) string {
	if v == nil || v.trans == nil {
		return fallback
	}
	if sentence := line(v.GetTranslator().Get(key, nil, ""), key); sentence != key {
		return sentence
	}
	return fallback
}

func containsString(list []string, value string) bool {
	for _, item := range list {
		if item == value {
			return true
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// The three that build no rule string of their own.
// ---------------------------------------------------------------------------

// When returns the rules an attribute gets when a condition holds, and the ones
// it gets when it does not.
func When(condition func(Data) bool, rules string, defaultRules ...string) *ConditionalRules {
	return NewConditionalRules(condition, rules, defaultRules...)
}

// Unless is When with the condition turned around.
func Unless(condition func(Data) bool, rules string, defaultRules ...string) *ConditionalRules {
	return NewConditionalRules(func(data Data) bool {
		return condition == nil || !condition(data)
	}, rules, defaultRules...)
}

// ForEach returns rules for one member of an array, decided by looking at that
// member.
func ForEach(callback func(value any, attribute string, data Data) Rules) *NestedRules {
	return NewNestedRules(callback)
}

// Using registers a query callback the rule runs instead of the conditions it
// would otherwise build.
//
// The query is any because the builder belongs to hesape/database and this
// package does not import it -- a rule carries the callback and the repository
// that runs it knows what it holds.
func (r *DatabaseRule) Using(callback func(query any)) *DatabaseRule {
	r.using = append(r.using, callback)

	return r
}

// QueryCallbacks returns the callbacks Using registered.
func (r *DatabaseRule) QueryCallbacks() []func(query any) { return r.using }

// IgnoreModel names the row being edited by its key rather than by the record
// itself. There are no records to pass here, so it is Ignore under a second
// name.
func (r *Unique) IgnoreModel(key string, idColumn ...string) *Unique {
	return r.Ignore(key, idColumn...)
}
