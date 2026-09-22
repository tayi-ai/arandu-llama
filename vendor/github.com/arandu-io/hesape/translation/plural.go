package translation

import "strings"

// GetPluralIndex reports which segment of a line a count selects in a locale,
// counting from zero: English
// answers 0 for one and 1 for anything else, Japanese always answers 0, Russian
// has three forms and Arabic six.
//
// A locale whose rule is not carried answers 0: that is the right answer for a
// language with no plural distinction and the safe one for a language nobody
// has written a rule for.
//
// The plural rules are derived from code of the Zend Framework (2010-09-25),
// which is subject to the new BSD license.
//
// The locale is matched on the language subtag alone, so pt-BR and pt_BR and pt
// resolve the same rule: no region changes the plural forms of its language,
// and an Accept-Language header carries a hyphenated tag. The count is read by
// magnitude, so -2 selects the form 2 does.
func (MessageSelector) GetPluralIndex(locale string, number int) int {
	if number < 0 {
		number = -number
	}
	if rule, ok := pluralRules[language(locale)]; ok {
		return rule(number)
	}
	return 0
}

// language reduces a locale to its lowercase language subtag.
func language(locale string) string {
	if i := strings.IndexAny(locale, "-_"); i >= 0 {
		locale = locale[:i]
	}
	return strings.ToLower(locale)
}

// The fifteen rules. Each is named for a language that has it, and the table
// below says which languages share it.
var (
	// one form: the language does not inflect for number.
	pluralSingle = func(n int) int { return 0 }

	// two forms, the singular being exactly one.
	pluralEnglish = func(n int) int {
		if n == 1 {
			return 0
		}
		return 1
	}

	// two forms, the singular covering zero as well.
	pluralFrench = func(n int) int {
		if n == 0 || n == 1 {
			return 0
		}
		return 1
	}

	// two forms, the singular being anything ending in one.
	pluralMacedonian = func(n int) int {
		if n%10 == 1 {
			return 0
		}
		return 1
	}

	pluralRussian = func(n int) int {
		switch {
		case n%10 == 1 && n%100 != 11:
			return 0
		case n%10 >= 2 && n%10 <= 4 && (n%100 < 10 || n%100 >= 20):
			return 1
		default:
			return 2
		}
	}

	pluralCzech = func(n int) int {
		switch {
		case n == 1:
			return 0
		case n >= 2 && n <= 4:
			return 1
		default:
			return 2
		}
	}

	pluralIrish = func(n int) int {
		switch n {
		case 1:
			return 0
		case 2:
			return 1
		default:
			return 2
		}
	}

	pluralLithuanian = func(n int) int {
		switch {
		case n%10 == 1 && n%100 != 11:
			return 0
		case n%10 >= 2 && (n%100 < 10 || n%100 >= 20):
			return 1
		default:
			return 2
		}
	}

	pluralSlovenian = func(n int) int {
		switch {
		case n%100 == 1:
			return 0
		case n%100 == 2:
			return 1
		case n%100 == 3 || n%100 == 4:
			return 2
		default:
			return 3
		}
	}

	pluralMaltese = func(n int) int {
		switch {
		case n == 1:
			return 0
		case n == 0 || (n%100 > 1 && n%100 < 11):
			return 1
		case n%100 > 10 && n%100 < 20:
			return 2
		default:
			return 3
		}
	}

	pluralLatvian = func(n int) int {
		switch {
		case n == 0:
			return 0
		case n%10 == 1 && n%100 != 11:
			return 1
		default:
			return 2
		}
	}

	pluralPolish = func(n int) int {
		switch {
		case n == 1:
			return 0
		case n%10 >= 2 && n%10 <= 4 && (n%100 < 12 || n%100 > 14):
			return 1
		default:
			return 2
		}
	}

	pluralWelsh = func(n int) int {
		switch {
		case n == 1:
			return 0
		case n == 2:
			return 1
		case n == 8 || n == 11:
			return 2
		default:
			return 3
		}
	}

	pluralRomanian = func(n int) int {
		switch {
		case n == 1:
			return 0
		case n == 0 || (n%100 > 0 && n%100 < 20):
			return 1
		default:
			return 2
		}
	}

	pluralArabic = func(n int) int {
		switch {
		case n == 0:
			return 0
		case n == 1:
			return 1
		case n == 2:
			return 2
		case n%100 >= 3 && n%100 <= 10:
			return 3
		case n%100 >= 11 && n%100 <= 99:
			return 4
		default:
			return 5
		}
	}
)

// pluralRules maps a language subtag to its rule. A language that is not here
// is read as having one form, which is what an unknown language can be assumed
// to need and never renders a wrong segment for a catalogue that carries only
// one.
var pluralRules = buildPluralRules()

func buildPluralRules() map[string]func(int) int {
	families := []struct {
		rule      func(int) int
		languages string
	}{
		{pluralSingle, "az bo dz id ja jv ka km kn ko ms th tr vi zh"},
		{pluralEnglish, "af bg bn ca da de el en eo es et eu fa fi fo fur fy gl gu ha he hu is it ku lb ml mn mr nah nb ne nl nn no om or pa pap ps pt so sq sv sw ta te tk ur zu"},
		{pluralFrench, "am bh fil fr gun hi hy ln mg nso ti wa xbr"},
		{pluralMacedonian, "mk"},
		{pluralRussian, "be bs hr ru sr uk"},
		{pluralCzech, "cs sk"},
		{pluralIrish, "ga"},
		{pluralLithuanian, "lt"},
		{pluralSlovenian, "sl"},
		{pluralMaltese, "mt"},
		{pluralLatvian, "lv"},
		{pluralPolish, "pl"},
		{pluralWelsh, "cy"},
		{pluralRomanian, "ro"},
		{pluralArabic, "ar"},
	}

	rules := make(map[string]func(int) int)
	for _, f := range families {
		for _, language := range strings.Fields(f.languages) {
			rules[language] = f.rule
		}
	}
	return rules
}
