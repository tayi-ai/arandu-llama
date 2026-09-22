package validation

import (
	"fmt"
	"math/big"
	"net/netip"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/arandu-io/hesape/str"
)

// evaluator is the signature every ValidateX method has, so an entry below holds
// the method itself rather than a closure around it: there is one place a rule's
// behaviour is written, and it is the method.
type evaluator = func(v *Validator, attribute string, value any, parameters []string) bool

// spec is one rule name: how many arguments it takes, whether it runs on a
// blank value, what it checks at boot, what it answers on a request and what it
// says when it fails.
//
// Everything about a rule lives in its entry. Adding one is a single edit, and
// reading one does not mean opening four files to find the message, the arity
// and the boot check separately -- which is how a rule ends up accepting an
// argument nobody validates.
type spec struct {
	// minArgs and maxArgs bound the argument list. A maxArgs of -1 is
	// unbounded; both zero means the rule takes no argument, and giving it one
	// is a boot failure rather than a silently ignored string.
	minArgs, maxArgs int

	// implicit marks a rule that runs even when the value is blank, and whose
	// failure stops every later rule on the field.
	implicit bool

	// sizeIsValue marks numeric, integer and decimal: the three rules that make
	// min, max, size, between and the four comparisons measure the VALUE rather
	// than the number of characters.
	sizeIsValue bool

	// refs are the argument positions that name another field of the same set.
	// Each is checked at boot: a rule pointing at a field name that does not
	// exist never fires, and nothing says so.
	refs []int

	// check runs at boot, after every field's flags are known. It rejects a
	// malformed argument and parses whatever the request should not parse
	// again.
	check func(c *checkCtx) error

	// eval answers whether the value passes. It never allocates a message.
	eval evaluator

	// message is the sentence a failure puts on the field. It is written to
	// read after a humanised field name and to be drawn without one -- see
	// Page.ErrorSummary and components.Field.
	message func(f *field, r *rule) string
}

// specs is the whole catalogue: every rule there is, at the spelling somebody
// types into a rule string, pointing at the method that runs it.
//
// It is closed. A name that is not in here and not in refused is a boot
// failure, which is what makes "a rule set that boots is a rule set whose names
// are all real" true.
var specs = map[string]*spec{
	// ---------------------------------------------------------------------
	// Presence and flow.
	// ---------------------------------------------------------------------
	"required": {
		implicit: true,
		eval:     (*Validator).ValidateRequired,
		message:  func(f *field, r *rule) string { return "is required" },
	},
	"sometimes": {
		eval:    (*Validator).ValidateSometimes,
		message: func(f *field, r *rule) string { return "" },
	},
	"bail": {
		eval:    (*Validator).ValidateBail,
		message: func(f *field, r *rule) string { return "" },
	},
	"nullable": {
		eval:    (*Validator).ValidateNullable,
		message: func(f *field, r *rule) string { return "" },
	},
	"filled": {
		implicit: true,
		eval:     (*Validator).ValidateFilled,
		message:  func(f *field, r *rule) string { return "must have a value" },
	},
	"present": {
		implicit: true,
		eval:     (*Validator).ValidatePresent,
		message:  func(f *field, r *rule) string { return "must be present" },
	},
	"present_if": {
		minArgs: 2, maxArgs: -1, implicit: true, refs: []int{0},
		eval: (*Validator).ValidatePresentIf,
		message: func(f *field, r *rule) string {
			return fmt.Sprintf("must be present when %s is %s", str.Headline(r.args[0]), or(r.args[1:]))
		},
	},
	"present_unless": {
		minArgs: 2, maxArgs: -1, implicit: true, refs: []int{0},
		eval: (*Validator).ValidatePresentUnless,
		message: func(f *field, r *rule) string {
			return fmt.Sprintf("must be present unless %s is %s", str.Headline(r.args[0]), or(r.args[1:]))
		},
	},
	"present_with": {
		minArgs: 1, maxArgs: -1, implicit: true,
		eval: (*Validator).ValidatePresentWith,
		message: func(f *field, r *rule) string {
			return fmt.Sprintf("must be present when %s is present", andHeadlined(r.args))
		},
	},
	"present_with_all": {
		minArgs: 1, maxArgs: -1, implicit: true,
		eval: (*Validator).ValidatePresentWithAll,
		message: func(f *field, r *rule) string {
			return fmt.Sprintf("must be present when %s are present", andHeadlined(r.args))
		},
	},
	"missing": {
		implicit: true,
		eval:     (*Validator).ValidateMissing,
		message:  func(f *field, r *rule) string { return "must not be sent" },
	},
	"missing_if": {
		minArgs: 2, maxArgs: -1, implicit: true, refs: []int{0},
		eval: (*Validator).ValidateMissingIf,
		message: func(f *field, r *rule) string {
			return fmt.Sprintf("must not be sent when %s is %s", str.Headline(r.args[0]), or(r.args[1:]))
		},
	},
	"missing_unless": {
		minArgs: 2, maxArgs: -1, implicit: true, refs: []int{0},
		eval: (*Validator).ValidateMissingUnless,
		message: func(f *field, r *rule) string {
			return fmt.Sprintf("must not be sent unless %s is %s", str.Headline(r.args[0]), or(r.args[1:]))
		},
	},
	"missing_with": {
		minArgs: 1, maxArgs: -1, implicit: true,
		eval: (*Validator).ValidateMissingWith,
		message: func(f *field, r *rule) string {
			return fmt.Sprintf("must not be sent when %s is present", andHeadlined(r.args))
		},
	},
	"missing_with_all": {
		minArgs: 1, maxArgs: -1, implicit: true,
		eval: (*Validator).ValidateMissingWithAll,
		message: func(f *field, r *rule) string {
			return fmt.Sprintf("must not be sent when %s are present", andHeadlined(r.args))
		},
	},
	"prohibited": {
		eval:    (*Validator).ValidateProhibited,
		message: func(f *field, r *rule) string { return "is not allowed" },
	},
	"prohibited_if": {
		minArgs: 2, maxArgs: -1, refs: []int{0},
		eval: (*Validator).ValidateProhibitedIf,
		message: func(f *field, r *rule) string {
			return fmt.Sprintf("is not allowed when %s is %s", str.Headline(r.args[0]), or(r.args[1:]))
		},
	},
	"prohibited_if_accepted": {
		minArgs: 1, maxArgs: -1, refs: []int{0},
		eval: (*Validator).ValidateProhibitedIfAccepted,
		message: func(f *field, r *rule) string {
			return "is not allowed when " + str.Headline(r.args[0]) + " is accepted"
		},
	},
	"prohibited_if_declined": {
		minArgs: 1, maxArgs: -1, refs: []int{0},
		eval: (*Validator).ValidateProhibitedIfDeclined,
		message: func(f *field, r *rule) string {
			return "is not allowed when " + str.Headline(r.args[0]) + " is declined"
		},
	},
	"prohibited_unless": {
		minArgs: 2, maxArgs: -1, refs: []int{0},
		eval: (*Validator).ValidateProhibitedUnless,
		message: func(f *field, r *rule) string {
			return fmt.Sprintf("is not allowed unless %s is %s", str.Headline(r.args[0]), or(r.args[1:]))
		},
	},
	"prohibits": {
		minArgs: 1, maxArgs: -1,
		eval: (*Validator).ValidateProhibits,
		message: func(f *field, r *rule) string {
			return "cannot be sent together with " + andHeadlined(r.args)
		},
	},
	"required_if": {
		minArgs: 2, maxArgs: -1, implicit: true, refs: []int{0},
		eval: (*Validator).ValidateRequiredIf,
		message: func(f *field, r *rule) string {
			return fmt.Sprintf("is required when %s is %s", str.Headline(r.args[0]), or(r.args[1:]))
		},
	},
	"required_if_accepted": {
		minArgs: 1, maxArgs: -1, implicit: true, refs: []int{0},
		eval: (*Validator).ValidateRequiredIfAccepted,
		message: func(f *field, r *rule) string {
			return "is required when " + str.Headline(r.args[0]) + " is accepted"
		},
	},
	"required_if_declined": {
		minArgs: 1, maxArgs: -1, implicit: true, refs: []int{0},
		eval: (*Validator).ValidateRequiredIfDeclined,
		message: func(f *field, r *rule) string {
			return "is required when " + str.Headline(r.args[0]) + " is declined"
		},
	},
	"required_unless": {
		minArgs: 2, maxArgs: -1, implicit: true, refs: []int{0},
		eval: (*Validator).ValidateRequiredUnless,
		message: func(f *field, r *rule) string {
			return fmt.Sprintf("is required unless %s is %s", str.Headline(r.args[0]), or(r.args[1:]))
		},
	},
	"required_with": {
		minArgs: 1, maxArgs: -1, implicit: true,
		eval: (*Validator).ValidateRequiredWith,
		message: func(f *field, r *rule) string {
			return fmt.Sprintf("is required when %s is present", andHeadlined(r.args))
		},
	},
	"required_with_all": {
		minArgs: 1, maxArgs: -1, implicit: true,
		eval: (*Validator).ValidateRequiredWithAll,
		message: func(f *field, r *rule) string {
			return fmt.Sprintf("is required when %s are present", andHeadlined(r.args))
		},
	},
	"required_without": {
		minArgs: 1, maxArgs: -1, implicit: true,
		eval: (*Validator).ValidateRequiredWithout,
		message: func(f *field, r *rule) string {
			return fmt.Sprintf("is required when %s is not present", andHeadlined(r.args))
		},
	},
	"required_without_all": {
		minArgs: 1, maxArgs: -1, implicit: true,
		eval: (*Validator).ValidateRequiredWithoutAll,
		message: func(f *field, r *rule) string {
			return fmt.Sprintf("is required when none of %s are present", andHeadlined(r.args))
		},
	},
	"required_array_keys": {
		minArgs: 1, maxArgs: -1,
		eval: (*Validator).ValidateRequiredArrayKeys,
		message: func(f *field, r *rule) string {
			return "must contain entries for " + and(r.args)
		},
	},

	// ---------------------------------------------------------------------
	// Exclusion. These five never put a message on the field: their failure
	// takes the field out of the validated data instead.
	// ---------------------------------------------------------------------
	"exclude": {
		eval:    (*Validator).ValidateExclude,
		message: func(f *field, r *rule) string { return "" },
	},
	"exclude_if": {
		minArgs: 2, maxArgs: -1, refs: []int{0},
		eval:    (*Validator).ValidateExcludeIf,
		message: func(f *field, r *rule) string { return "" },
	},
	"exclude_unless": {
		minArgs: 2, maxArgs: -1, refs: []int{0},
		eval:    (*Validator).ValidateExcludeUnless,
		message: func(f *field, r *rule) string { return "" },
	},
	"exclude_with": {
		minArgs: 1, maxArgs: 1, refs: []int{0},
		eval:    (*Validator).ValidateExcludeWith,
		message: func(f *field, r *rule) string { return "" },
	},
	"exclude_without": {
		minArgs: 1, maxArgs: -1, refs: []int{0},
		eval:    (*Validator).ValidateExcludeWithout,
		message: func(f *field, r *rule) string { return "" },
	},

	// ---------------------------------------------------------------------
	// Cross-field and consent.
	// ---------------------------------------------------------------------
	"confirmed": {
		maxArgs: 1, refs: []int{0},
		eval: (*Validator).ValidateConfirmed,
		// The message goes on the field rather than on the confirmation, and it
		// says what is wrong rather than which box to change: a form that
		// reports "does not match" next to the first box gets the first box
		// edited, and fails again.
		message: func(f *field, r *rule) string { return "does not match" },
	},
	"same": {
		minArgs: 1, maxArgs: 1, refs: []int{0},
		eval:    (*Validator).ValidateSame,
		message: func(f *field, r *rule) string { return "must match " + str.Headline(r.args[0]) },
	},
	"different": {
		minArgs: 1, maxArgs: -1, refs: []int{0},
		eval:    (*Validator).ValidateDifferent,
		message: func(f *field, r *rule) string { return "must be different from " + andHeadlined(r.args) },
	},
	"accepted": {
		implicit: true,
		eval:     (*Validator).ValidateAccepted,
		message:  func(f *field, r *rule) string { return "must be accepted" },
	},
	"accepted_if": {
		minArgs: 2, maxArgs: -1, implicit: true, refs: []int{0},
		eval: (*Validator).ValidateAcceptedIf,
		message: func(f *field, r *rule) string {
			return fmt.Sprintf("must be accepted when %s is %s", str.Headline(r.args[0]), or(r.args[1:]))
		},
	},
	"declined": {
		implicit: true,
		eval:     (*Validator).ValidateDeclined,
		message:  func(f *field, r *rule) string { return "must be declined" },
	},
	"declined_if": {
		minArgs: 2, maxArgs: -1, implicit: true, refs: []int{0},
		eval: (*Validator).ValidateDeclinedIf,
		message: func(f *field, r *rule) string {
			return fmt.Sprintf("must be declined when %s is %s", str.Headline(r.args[0]), or(r.args[1:]))
		},
	},
	"current_password": {
		maxArgs: 1,
		eval:    (*Validator).ValidateCurrentPassword,
		message: func(f *field, r *rule) string { return "is incorrect" },
	},

	// ---------------------------------------------------------------------
	// Size. One name for four measurements: characters in a string, the value
	// in a number, members in an array, KILOBYTES in a file.
	// ---------------------------------------------------------------------
	"min": {
		minArgs: 1, maxArgs: 1, check: needSizes,
		eval:    (*Validator).ValidateMin,
		message: func(f *field, r *rule) string { return "must be at least " + measure(f, r.args[0]) },
	},
	"max": {
		minArgs: 1, maxArgs: 1, check: needSizes,
		eval:    (*Validator).ValidateMax,
		message: func(f *field, r *rule) string { return "must be at most " + measure(f, r.args[0]) },
	},
	"size": {
		minArgs: 1, maxArgs: 1, check: needSizes,
		eval:    (*Validator).ValidateSize,
		message: func(f *field, r *rule) string { return "must be exactly " + measure(f, r.args[0]) },
	},
	"between": {
		minArgs: 2, maxArgs: 2, check: chain(needSizes, ascending),
		eval: (*Validator).ValidateBetween,
		message: func(f *field, r *rule) string {
			return fmt.Sprintf("must be between %s and %s", r.args[0], measure(f, r.args[1]))
		},
	},
	"digits": {
		minArgs: 1, maxArgs: 1, check: needWholeNumbers,
		eval:    (*Validator).ValidateDigits,
		message: func(f *field, r *rule) string { return "must be " + r.args[0] + " digits" },
	},
	"digits_between": {
		minArgs: 2, maxArgs: 2, check: chain(needWholeNumbers, ascending),
		eval: (*Validator).ValidateDigitsBetween,
		message: func(f *field, r *rule) string {
			return fmt.Sprintf("must be between %s and %s digits", r.args[0], r.args[1])
		},
	},
	"max_digits": {
		minArgs: 1, maxArgs: 1, check: needWholeNumbers,
		eval:    (*Validator).ValidateMaxDigits,
		message: func(f *field, r *rule) string { return "must have at most " + r.args[0] + " digits" },
	},
	"min_digits": {
		minArgs: 1, maxArgs: 1, check: needWholeNumbers,
		eval:    (*Validator).ValidateMinDigits,
		message: func(f *field, r *rule) string { return "must have at least " + r.args[0] + " digits" },
	},
	"multiple_of": {
		minArgs: 1, maxArgs: 1, check: needNumbers,
		eval:    (*Validator).ValidateMultipleOf,
		message: func(f *field, r *rule) string { return "must be a multiple of " + r.args[0] },
	},
	"gt": {
		minArgs: 1, maxArgs: 1, refs: []int{0},
		eval:    (*Validator).ValidateGt,
		message: func(f *field, r *rule) string { return "must be greater than " + bound(r.args[0]) },
	},
	"gte": {
		minArgs: 1, maxArgs: 1, refs: []int{0},
		eval: (*Validator).ValidateGte,
		message: func(f *field, r *rule) string {
			return "must be greater than or equal to " + bound(r.args[0])
		},
	},
	"lt": {
		minArgs: 1, maxArgs: 1, refs: []int{0},
		eval:    (*Validator).ValidateLt,
		message: func(f *field, r *rule) string { return "must be less than " + bound(r.args[0]) },
	},
	"lte": {
		minArgs: 1, maxArgs: 1, refs: []int{0},
		eval: (*Validator).ValidateLte,
		message: func(f *field, r *rule) string {
			return "must be less than or equal to " + bound(r.args[0])
		},
	},

	// ---------------------------------------------------------------------
	// Type.
	// ---------------------------------------------------------------------
	"numeric": {
		sizeIsValue: true,
		eval:        (*Validator).ValidateNumeric,
		message:     func(f *field, r *rule) string { return "must be a number" },
	},
	"integer": {
		sizeIsValue: true,
		eval:        (*Validator).ValidateInteger,
		message:     func(f *field, r *rule) string { return "must be a whole number" },
	},
	"decimal": {
		minArgs: 1, maxArgs: 2, sizeIsValue: true, check: chain(needWholeNumbers, ascending),
		eval: (*Validator).ValidateDecimal,
		message: func(f *field, r *rule) string {
			if len(r.args) == 1 {
				return "must have " + r.args[0] + " decimal places"
			}
			return fmt.Sprintf("must have between %s and %s decimal places", r.args[0], r.args[1])
		},
	},
	"boolean": {
		eval:    (*Validator).ValidateBoolean,
		message: func(f *field, r *rule) string { return "must be true or false" },
	},
	"string": {
		eval:    (*Validator).ValidateString,
		message: func(f *field, r *rule) string { return "must be text" },
	},
	"ascii": {
		eval:    (*Validator).ValidateAscii,
		message: func(f *field, r *rule) string { return "must contain only ASCII characters" },
	},
	"json": {
		eval:    (*Validator).ValidateJson,
		message: func(f *field, r *rule) string { return "must be valid JSON" },
	},
	"array": {
		maxArgs: -1,
		eval:    (*Validator).ValidateArray,
		message: func(f *field, r *rule) string {
			if len(r.args) == 0 {
				return "must be a list of values"
			}
			return "may only contain " + and(r.args)
		},
	},
	"list": {
		eval:    (*Validator).ValidateList,
		message: func(f *field, r *rule) string { return "must be a list of values" },
	},
	"distinct": {
		maxArgs: -1, check: checkDistinct,
		eval:    (*Validator).ValidateDistinct,
		message: func(f *field, r *rule) string { return "has a duplicate value" },
	},
	"contains": {
		minArgs: 1, maxArgs: -1,
		eval:    (*Validator).ValidateContains,
		message: func(f *field, r *rule) string { return "must contain " + and(r.args) },
	},
	"in_array": {
		minArgs: 1, maxArgs: 1, refs: []int{0},
		eval:    (*Validator).ValidateInArray,
		message: func(f *field, r *rule) string { return "must be one of the values of " + str.Headline(r.args[0]) },
	},

	// ---------------------------------------------------------------------
	// Dates. The layout is a Go layout; see checkLayout.
	// ---------------------------------------------------------------------
	"date": {
		eval:    (*Validator).ValidateDate,
		message: func(f *field, r *rule) string { return "is not a valid date" },
	},
	"date_format": {
		minArgs: 1, maxArgs: -1, check: checkLayout,
		eval:    (*Validator).ValidateDateFormat,
		message: func(f *field, r *rule) string { return "must match the format " + or(r.args) },
	},
	"date_equals": {
		minArgs: 1, maxArgs: 1, check: checkMoment,
		eval:    (*Validator).ValidateDateEquals,
		message: func(f *field, r *rule) string { return "must be the date " + moment(r.args[0]) },
	},
	"after": {
		minArgs: 1, maxArgs: 1, check: checkMoment,
		eval:    (*Validator).ValidateAfter,
		message: func(f *field, r *rule) string { return "must be a date after " + moment(r.args[0]) },
	},
	"after_or_equal": {
		minArgs: 1, maxArgs: 1, check: checkMoment,
		eval:    (*Validator).ValidateAfterOrEqual,
		message: func(f *field, r *rule) string { return "must be a date on or after " + moment(r.args[0]) },
	},
	"before": {
		minArgs: 1, maxArgs: 1, check: checkMoment,
		eval:    (*Validator).ValidateBefore,
		message: func(f *field, r *rule) string { return "must be a date before " + moment(r.args[0]) },
	},
	"before_or_equal": {
		minArgs: 1, maxArgs: 1, check: checkMoment,
		eval:    (*Validator).ValidateBeforeOrEqual,
		message: func(f *field, r *rule) string { return "must be a date on or before " + moment(r.args[0]) },
	},
	"timezone": {
		eval:    (*Validator).ValidateTimezone,
		message: func(f *field, r *rule) string { return "is not a valid time zone" },
	},

	// ---------------------------------------------------------------------
	// String shape.
	// ---------------------------------------------------------------------
	"email": {
		maxArgs: -1, check: checkEmailValidations,
		eval:    (*Validator).ValidateEmail,
		message: func(f *field, r *rule) string { return "is not a valid email address" },
	},
	"url": {
		maxArgs: -1,
		eval:    (*Validator).ValidateUrl,
		message: func(f *field, r *rule) string { return "is not a valid URL" },
	},
	"active_url": {
		eval:    (*Validator).ValidateActiveUrl,
		message: func(f *field, r *rule) string { return "is not an active URL" },
	},
	"uuid": {
		maxArgs: 1,
		eval:    (*Validator).ValidateUuid,
		message: func(f *field, r *rule) string { return "is not a valid UUID" },
	},
	"ulid": {
		eval:    (*Validator).ValidateUlid,
		message: func(f *field, r *rule) string { return "is not a valid ULID" },
	},
	"alpha": {
		maxArgs: 1, check: checkAscii,
		eval:    (*Validator).ValidateAlpha,
		message: func(f *field, r *rule) string { return "must contain only letters" },
	},
	"alpha_dash": {
		maxArgs: 1, check: checkAscii,
		eval: (*Validator).ValidateAlphaDash,
		message: func(f *field, r *rule) string {
			return "must contain only letters, numbers, dashes and underscores"
		},
	},
	"alpha_num": {
		maxArgs: 1, check: checkAscii,
		eval:    (*Validator).ValidateAlphaNum,
		message: func(f *field, r *rule) string { return "must contain only letters and numbers" },
	},
	"lowercase": {
		eval:    (*Validator).ValidateLowercase,
		message: func(f *field, r *rule) string { return "must be lowercase" },
	},
	"uppercase": {
		eval:    (*Validator).ValidateUppercase,
		message: func(f *field, r *rule) string { return "must be uppercase" },
	},
	"starts_with": {
		minArgs: 1, maxArgs: -1,
		eval:    (*Validator).ValidateStartsWith,
		message: func(f *field, r *rule) string { return "must start with " + or(r.args) },
	},
	"ends_with": {
		minArgs: 1, maxArgs: -1,
		eval:    (*Validator).ValidateEndsWith,
		message: func(f *field, r *rule) string { return "must end with " + or(r.args) },
	},
	"doesnt_start_with": {
		minArgs: 1, maxArgs: -1,
		eval:    (*Validator).ValidateDoesntStartWith,
		message: func(f *field, r *rule) string { return "must not start with " + or(r.args) },
	},
	"doesnt_end_with": {
		minArgs: 1, maxArgs: -1,
		eval:    (*Validator).ValidateDoesntEndWith,
		message: func(f *field, r *rule) string { return "must not end with " + or(r.args) },
	},

	// ---------------------------------------------------------------------
	// Membership and pattern.
	// ---------------------------------------------------------------------
	"in": {
		minArgs: 1, maxArgs: -1,
		eval:    (*Validator).ValidateIn,
		message: func(f *field, r *rule) string { return "is not one of the allowed values" },
	},
	"not_in": {
		minArgs: 1, maxArgs: -1,
		eval:    (*Validator).ValidateNotIn,
		message: func(f *field, r *rule) string { return "is not one of the allowed values" },
	},
	// The cases are optional because a typed value carries them: `enum` on a
	// field holding a generated enum reads the set off the type, and a list
	// written beside it can only disagree.
	"enum": {
		minArgs: 0, maxArgs: -1,
		eval:    (*Validator).ValidateEnum,
		message: func(f *field, r *rule) string { return "is not one of the allowed values" },
	},
	"regex": {
		minArgs: 1, maxArgs: 1, check: compilePattern,
		eval:    (*Validator).ValidateRegex,
		message: func(f *field, r *rule) string { return "is not in the expected format" },
	},
	"not_regex": {
		minArgs: 1, maxArgs: 1, check: compilePattern,
		eval:    (*Validator).ValidateNotRegex,
		message: func(f *field, r *rule) string { return "is not in the expected format" },
	},

	// ---------------------------------------------------------------------
	// Network and colour shape.
	// ---------------------------------------------------------------------
	"ip": {
		eval:    (*Validator).ValidateIp,
		message: func(f *field, r *rule) string { return "is not a valid IP address" },
	},
	"ipv4": {
		eval:    (*Validator).ValidateIpv4,
		message: func(f *field, r *rule) string { return "is not a valid IPv4 address" },
	},
	"ipv6": {
		eval:    (*Validator).ValidateIpv6,
		message: func(f *field, r *rule) string { return "is not a valid IPv6 address" },
	},
	"mac_address": {
		eval:    (*Validator).ValidateMacAddress,
		message: func(f *field, r *rule) string { return "is not a valid MAC address" },
	},
	"hex_color": {
		eval:    (*Validator).ValidateHexColor,
		message: func(f *field, r *rule) string { return "is not a valid hex color" },
	},

	// ---------------------------------------------------------------------
	// Uploads. The value is a File, not a string, and a size on one of these
	// fields is measured in kilobytes.
	// ---------------------------------------------------------------------
	"file": {
		eval:    (*Validator).ValidateFile,
		message: func(f *field, r *rule) string { return "must be a file" },
	},
	"image": {
		maxArgs: -1, check: checkImage,
		eval:    (*Validator).ValidateImage,
		message: func(f *field, r *rule) string { return "must be an image" },
	},
	"mimes": {
		minArgs: 1, maxArgs: -1,
		eval:    (*Validator).ValidateMimes,
		message: func(f *field, r *rule) string { return "must be a file of type " + or(r.args) },
	},
	"mimetypes": {
		minArgs: 1, maxArgs: -1,
		eval:    (*Validator).ValidateMimetypes,
		message: func(f *field, r *rule) string { return "must be a file of type " + or(r.args) },
	},
	"extensions": {
		minArgs: 1, maxArgs: -1,
		eval:    (*Validator).ValidateExtensions,
		message: func(f *field, r *rule) string { return "must be a file of type " + or(r.args) },
	},
	"dimensions": {
		minArgs: 1, maxArgs: -1, check: checkDimensions,
		eval:    (*Validator).ValidateDimensions,
		message: func(f *field, r *rule) string { return "does not have the required image dimensions" },
	},

	// ---------------------------------------------------------------------
	// The database. Both take the Grant and both fail closed without one: a
	// read is authorized like any other. See WithPresence.
	// ---------------------------------------------------------------------
	"exists": {
		minArgs: 1, maxArgs: -1,
		eval:    (*Validator).ValidateExists,
		message: func(f *field, r *rule) string { return "does not exist" },
	},
	"unique": {
		minArgs: 1, maxArgs: -1,
		eval:    (*Validator).ValidateUnique,
		message: func(f *field, r *rule) string { return "has already been taken" },
	},
}

var (
	// numericShape is what `numeric` accepts. strconv.ParseFloat alone would
	// take "Inf", "NaN" and hex floats, none of which a person types into a
	// form.
	numericShape = regexp.MustCompile(`^[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?$`)

	// integerShape refuses a leading zero.
	integerShape = regexp.MustCompile(`^[+-]?(0|[1-9][0-9]*)$`)

	digitShape = regexp.MustCompile(`^[0-9]+$`)

	// decimalShape refuses exponent notation on purpose: "1e2" has no decimal
	// places to count.
	decimalShape = regexp.MustCompile(`^[+-]?[0-9]*\.?([0-9]*)$`)

	// macShape is the three spellings of a MAC address that are accepted.
	macShape = regexp.MustCompile(`^([0-9a-fA-F]{2}[:-]){5}[0-9a-fA-F]{2}$|^([0-9a-fA-F]{4}\.){2}[0-9a-fA-F]{4}$`)

	hexColorShape = regexp.MustCompile(`^#(([0-9a-fA-F]{3}){1,2}|([0-9a-fA-F]{4}){1,2})$`)

	alphaUnicode = regexp.MustCompile(`^[\p{L}\p{M}]+$`)
	alphaASCII   = regexp.MustCompile(`^[a-zA-Z]+$`)
	dashUnicode  = regexp.MustCompile(`^[\p{L}\p{M}\p{N}_-]+$`)
	dashASCII    = regexp.MustCompile(`^[a-zA-Z0-9_-]+$`)
	numUnicode   = regexp.MustCompile(`^[\p{L}\p{M}\p{N}]+$`)
	numASCII     = regexp.MustCompile(`^[a-zA-Z0-9]+$`)
)

// dateLayouts are the layouts `date`, `after` and `before` read when the field
// declares no date_format of its own.
//
// The list is short and stated rather than permissive: a rule whose accepted set
// nobody can enumerate cannot be reasoned about. A form that needs another
// spelling declares date_format.
var dateLayouts = []string{
	time.RFC3339,
	"2006-01-02T15:04:05",
	"2006-01-02 15:04:05",
	"2006-01-02 15:04",
	"2006-01-02",
}

// number reads a form value as a number, and reports whether it was one.
func number(v string) (float64, bool) {
	if !numericShape.MatchString(v) {
		return 0, false
	}
	n, err := strconv.ParseFloat(v, 64)
	if err != nil {
		return 0, false
	}
	return n, true
}

// numericText returns the TEXT of a value that is a number, trimmed, which is
// what GetSize measures. A float64 read back out of a form has already lost the
// digits this exists to keep, so the text is what travels.
func numericText(v any) (string, bool) {
	if s, isString := asString(v); isString {
		s = strings.TrimSpace(s)
		if _, isNumber := number(s); !isNumber {
			return "", false
		}
		return s, true
	}
	if _, isNumber := numberOf(v); !isNumber {
		return "", false
	}
	return stringOf(v), true
}

// exactText reads numeric text at arbitrary precision, which is what
// BigNumber::of does with the same string.
//
// big.Rat reads a fraction as well ("1/2"), and no caller reaches here with
// one: every path asks number() first, and is_numeric has no spelling for a
// fraction.
func exactText(s string) (*big.Rat, bool) {
	r, ok := new(big.Rat).SetString(s)
	return r, ok
}

// exactNumber is exactText for a value rather than a string.
func exactNumber(v any) (*big.Rat, bool) {
	text, isNumber := numericText(v)
	if !isNumber {
		return nil, false
	}
	return exactText(text)
}

// exactParameter reads the bound a size rule was written with. Text that is not
// a number fails the rule, which is what every other malformed parameter already
// does.
func exactParameter(p string) (*big.Rat, bool) {
	p = strings.TrimSpace(p)
	if _, isNumber := number(p); !isNumber {
		return nil, false
	}
	return exactText(p)
}

// sizeText renders a size into a message. An integer prints as one; anything
// else prints as a decimal, never as the "3/2" a big.Rat spells by default --
// that is not a number anybody typed into a form.
func sizeText(size *big.Rat) string {
	if size == nil {
		return ""
	}
	if size.IsInt() {
		return size.Num().String()
	}
	f, _ := size.Float64()
	return strconv.FormatFloat(f, 'f', -1, 64)
}

// whole is FILTER_VALIDATE_INT for a form value.
func whole(v string) (int64, bool) {
	if !integerShape.MatchString(v) {
		return 0, false
	}
	n, err := strconv.ParseInt(v, 10, 64)
	if err != nil {
		return 0, false
	}
	return n, true
}

func digitsOnly(v string) bool { return digitShape.MatchString(v) }

// decimalPlaces counts the digits after the point, which is what `decimal`
// bounds. It answers false for anything that is not a plain decimal number.
func decimalPlaces(v string) (int, bool) {
	if _, ok := number(v); !ok {
		return 0, false
	}
	m := decimalShape.FindStringSubmatch(v)
	if m == nil {
		return 0, false
	}
	return len(m[1]), true
}

// emailShape is the shape check `email` and the Email helper share, so there is
// one answer to "is this an address" rather than two that drift.
//
// Whitespace is rejected rather than trimmed: an address with a space in it is
// almost always a paste accident, and silently trimming input hides the mistake
// from the person who made it.
func emailShape(value string) bool {
	if strings.ContainsAny(value, " \t\r\n") {
		return false
	}
	at := strings.IndexByte(value, '@')
	return at > 0 && at != len(value)-1 && strings.Contains(value[at:], ".")
}

// address parses an IP and refuses a zone: "fe80::1%eth0" names an interface on
// one machine.
func address(v string) (netip.Addr, bool) {
	a, err := netip.ParseAddr(v)
	if err != nil || a.Zone() != "" {
		return netip.Addr{}, false
	}
	return a, true
}

// parseDate reads a value with the field's declared layout first, then with the
// short list of layouts a form can send.
func parseDate(layout, v string) (time.Time, bool) {
	if layout != "" {
		if t, err := time.Parse(layout, v); err == nil {
			return t, true
		}
	}
	for _, l := range dateLayouts {
		if t, err := time.Parse(l, v); err == nil {
			return t, true
		}
	}
	return time.Time{}, false
}

// keywords are the three relative moments after and before accept, so a rule
// that means "not in the past" does not have to be regenerated every morning.
var keywords = map[string]func(time.Time) time.Time{
	"today":     func(now time.Time) time.Time { return startOfDay(now) },
	"tomorrow":  func(now time.Time) time.Time { return startOfDay(now).AddDate(0, 0, 1) },
	"yesterday": func(now time.Time) time.Time { return startOfDay(now).AddDate(0, 0, -1) },
}

func startOfDay(t time.Time) time.Time {
	return time.Date(t.Year(), t.Month(), t.Day(), 0, 0, 0, 0, t.Location())
}

// moment renders the argument of after/before for a message.
func moment(arg string) string {
	if _, isKeyword := keywords[arg]; isKeyword {
		return arg
	}
	if _, isDate := parseDate("", arg); isDate {
		return arg
	}
	return str.Headline(arg)
}

// bound renders the argument of gt/gte/lt/lte for a message: a number is a
// literal bound and anything else names another field.
func bound(arg string) string {
	if _, isNumber := number(arg); isNumber {
		return arg
	}
	return str.Headline(arg)
}

// measure renders a size limit with its unit. A limit on a string is a number
// of characters; a limit on a field that declares numeric, integer or decimal
// is the number itself, and saying "characters" there is wrong in the one place
// a person is reading carefully.
func measure(f *field, arg string) string {
	if f.numeric {
		return arg
	}
	if f.file {
		return arg + " kilobytes"
	}
	return arg + " characters"
}

func or(list []string) string  { return joinWith(list, "or") }
func and(list []string) string { return joinWith(list, "and") }

func andHeadlined(list []string) string {
	headlined := make([]string, len(list))
	for i, item := range list {
		headlined[i] = str.Headline(item)
	}
	return and(headlined)
}

func joinWith(list []string, conjunction string) string {
	switch len(list) {
	case 0:
		return ""
	case 1:
		return list[0]
	}
	return strings.Join(list[:len(list)-1], ", ") + " " + conjunction + " " + list[len(list)-1]
}
