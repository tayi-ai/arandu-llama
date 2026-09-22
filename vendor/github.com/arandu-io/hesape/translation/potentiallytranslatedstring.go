package translation

// PotentiallyTranslatedString is a message that is a key if the catalogue
// carries it and the message itself if it does not.
//
// It is what a validation rule hands back when it fails. The rule writes
// "The :attribute must be a working URL." or "validation.custom.url", calls
// [PotentiallyTranslatedString.Translate], and the caller reads
// [PotentiallyTranslatedString.ToString] without having to know which of the
// two it was given.
type PotentiallyTranslatedString struct {
	string      string
	translation string
	translator  *Translator
}

// NewPotentiallyTranslatedString answers
// PotentiallyTranslatedString::__construct().
func NewPotentiallyTranslatedString(s string, translator *Translator) *PotentiallyTranslatedString {
	return &PotentiallyTranslatedString{string: s, translator: translator}
}

// Translate resolves the string through the translator and keeps the result.
//
// An empty locale means the translator's own. It returns the same value so that
// calls chain.
func (s *PotentiallyTranslatedString) Translate(replace Replace, locale string) *PotentiallyTranslatedString {
	if s.translator == nil {
		return s
	}
	s.translation = s.translator.Get(locale, s.string, replace)
	return s
}

// TranslateChoice answers PotentiallyTranslatedString::translateChoice(): it
// resolves the string as a pluralised line and keeps the result.
func (s *PotentiallyTranslatedString) TranslateChoice(number int, replace Replace, locale string) *PotentiallyTranslatedString {
	if s.translator == nil {
		return s
	}
	s.translation = s.translator.Choice(locale, s.string, number, replace)
	return s
}

// Original answers PotentiallyTranslatedString::original(): the string as it
// was given, whether or not it has since been translated.
func (s *PotentiallyTranslatedString) Original() string { return s.string }

// ToString answers PotentiallyTranslatedString::toString(): the translation
// when there is one, and the original string when there is not.
func (s *PotentiallyTranslatedString) ToString() string {
	if s.translation != "" {
		return s.translation
	}
	return s.string
}

// String satisfies fmt.Stringer. It is
// [PotentiallyTranslatedString.ToString].
func (s *PotentiallyTranslatedString) String() string { return s.ToString() }
