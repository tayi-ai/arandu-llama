package translation

import (
	"fmt"
	"maps"
	"reflect"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"
)

// Replace holds the arguments a line interpolates, keyed by the placeholder
// name without its colon: Replace{"attribute": "email"} fills ":attribute",
// ":Attribute" and ":ATTRIBUTE".
//
// Values are rendered with fmt.Sprint, so a count may be given as an int and a
// duration as a time.Duration without the caller formatting it first. A value
// of type func(string) string is applied differently: it replaces the text
// between <name> and </name> rather than a placeholder.
type Replace map[string]any

// Translator resolves a key into a sentence.
//
// One instance answers every request. It caches the groups it has loaded, and
// the setters below are callable at any time, so every field is behind a lock.
type Translator struct {
	mu sync.RWMutex

	// loader is what the application was built with, and what GetLoader
	// returns. lookup is that loader followed by the English lines that ship
	// with this package.
	loader Loader
	lookup Loader

	locale   string
	fallback string

	// loaded is namespace -> group -> locale -> lines: the groups this
	// translator has already asked the loader for, plus whatever AddLines
	// wrote straight into it.
	loaded map[string]map[string]map[string]Lines

	selector              Selector
	determineLocalesUsing func(locales []string) []string
	stringableHandlers    map[reflect.Type]func(any) string

	missingTranslationKeyCallback func(key string, replace Replace, locale string, fallback bool) string
	handleMissingTranslationKeys  bool
}

// New returns a translator over a loader.
//
// locale is the locale used when a caller passes none, and fallback the one
// consulted when a key is missing from the locale asked for. The English lines
// that ship with this package are answered after both, so auth, validation,
// passwords and pagination resolve with no catalogue at all -- l may be nil.
//
// They are answered by a second loader rather than a second path, so that a
// translator built over an [ArrayLoader] still answers them.
func New(l Loader, locale, fallback string) *Translator {
	t := &Translator{
		loader:                       l,
		fallback:                     fallback,
		loaded:                       make(map[string]map[string]map[string]Lines),
		selector:                     MessageSelector{},
		stringableHandlers:           make(map[reflect.Type]func(any) string),
		handleMissingTranslationKeys: true,
	}
	if l == nil {
		t.loader = bundled
		t.lookup = bundled
	} else {
		t.lookup = chain{l, bundled}
	}
	// A locale holding a path separator is rejected here as it is in
	// SetLocale. There is no error to return from a constructor that yields
	// one value, so a rejected locale becomes the empty one, which every
	// method below reads as "no locale given".
	if err := t.SetLocale(locale); err != nil {
		t.locale = ""
	}
	return t
}

// Get is the line stored under key, with its placeholders replaced.
//
// The locale comes first and the replacements last. An empty locale means the
// translator's own.
//
// A key no catalogue carries is returned unchanged, after the callback
// registered by [Translator.HandleMissingKeysUsing] has seen it, so a wrong key
// shows as itself on the page instead of as a blank.
//
// A key naming a group and no item -- "messages" rather than
// "messages.welcome" -- returns the key, because the result is a sentence and
// a group is not one.
func (t *Translator) Get(locale, key string, replace Replace) string {
	return t.get(locale, key, replace, true)
}

func (t *Translator) get(locale, key string, replace Replace, fallback bool) string {
	locale = t.localeOr(locale)

	// For JSON translations there is one file per locale, so it is loaded and
	// the key looked up in it directly -- these are one level deep.
	t.Load(AppNamespace, JSONGroup, locale)

	t.mu.RLock()
	line, found := t.loaded[AppNamespace][JSONGroup][locale][key]
	t.mu.RUnlock()

	if !found {
		namespace, group, item := t.ParseKey(key)
		if item == "" {
			return t.makeReplacements(t.handleMissingTranslationKey(key, replace, locale, fallback), replace)
		}

		locales := []string{locale}
		if fallback {
			locales = t.localeArray(locale)
		}
		for _, at := range locales {
			if l, ok := t.getLine(namespace, group, at, item, replace); ok {
				return l
			}
		}

		key = t.handleMissingTranslationKey(key, replace, locale, fallback)
	}

	// A line that is there but empty reads as missing, because an empty
	// sentence on the page is indistinguishable from a bug.
	if line == "" {
		line = key
	}
	return t.makeReplacements(line, replace)
}

// Choice answers Translator::choice(): the segment of a line that a count
// selects, with its placeholders replaced.
//
// The segments of a line are divided by "|" and may open with an explicit
// condition, "{0}" for a single count and "[1,19]" or "[20,*]" for a range. The
// first condition that matches wins; with no condition, the plural rule of the
// locale chooses. See [MessageSelector.Choose].
//
// ":count" is filled with count unless replace already carries it. The count
// is the number itself; use len at the call site for a collection.
func (t *Translator) Choice(locale, key string, number int, replace Replace) string {
	locale = t.localeForChoice(key, locale)

	line := t.Get(locale, key, nil)

	if _, taken := replace["count"]; !taken {
		replace = maps.Clone(replace)
		if replace == nil {
			replace = make(Replace, 1)
		}
		replace["count"] = number
	}

	return t.makeReplacements(t.GetSelector().Choose(line, number, locale), replace)
}

// localeForChoice answers Translator::localeForChoice(). A line that only
// exists in the fallback pluralises by the rule of the fallback: picking the
// segment by the rule of a language the line is not written in selects the
// wrong one as soon as the two rules differ.
func (t *Translator) localeForChoice(key, locale string) string {
	locale = t.localeOr(locale)
	if t.HasForLocale(locale, key) {
		return locale
	}
	return t.GetFallback()
}

// Has answers Translator::has(): whether a line exists for key, in the given
// locale or in the fallback.
func (t *Translator) Has(locale, key string) bool {
	return t.has(locale, key, true)
}

// HasForLocale answers Translator::hasForLocale(): whether a line exists for
// key in the given locale alone, with no fall through to the fallback locale.
func (t *Translator) HasForLocale(locale, key string) bool {
	return t.has(locale, key, false)
}

func (t *Translator) has(locale, key string, fallback bool) bool {
	locale = t.localeOr(locale)

	// The handling of missing keys is disabled while the existence check runs,
	// and put back to what it was afterwards: asking whether a line exists must
	// not report it as missing.
	t.mu.Lock()
	handle := t.handleMissingTranslationKeys
	t.handleMissingTranslationKeys = false
	t.mu.Unlock()

	line := t.get(locale, key, nil, fallback)

	t.mu.Lock()
	t.handleMissingTranslationKeys = handle
	t.mu.Unlock()

	// For JSON translations the loaded file holds the line under the key
	// itself, so the key coming back is not proof of anything.
	t.mu.RLock()
	_, json := t.loaded[AppNamespace][JSONGroup][locale][key]
	t.mu.RUnlock()
	if json {
		return true
	}

	return line != key
}

// getLine is one line out of the loaded catalogue, with its replacements made.
//
// A group is not a sentence (see [Translator.Get]), so this answers one line
// and never a whole group.
func (t *Translator) getLine(namespace, group, locale, item string, replace Replace) (string, bool) {
	t.Load(namespace, group, locale)

	t.mu.RLock()
	line, ok := t.loaded[namespace][group][locale][item]
	t.mu.RUnlock()
	if !ok || line == "" {
		return "", false
	}
	return t.makeReplacements(line, replace), true
}

// makeReplacements fills the placeholders of a line.
//
// Three spellings of every name are replaced: ":name" with the value, ":Name"
// with the value's first letter uppercased, and ":NAME" with the value
// uppercased. A form is offered so that
// "The :attribute field is required." and ":Attribute is required." can share
// one argument.
//
// A placeholder no argument names is left as it stands, and the longest name
// that does match wins at each position, so ":value" does not eat the front of
// ":values" when both are given, and does eat it when only ":value" is.
//
// An argument whose value is a func(string) string replaces the text between
// <name> and </name>, which is how a sentence hands part of itself to the
// caller to wrap.
func (t *Translator) makeReplacements(line string, replace Replace) string {
	if len(replace) == 0 {
		return line
	}

	for key, value := range replace {
		wrap, ok := value.(func(string) string)
		if !ok {
			continue
		}
		line = replaceWrapped(line, key, wrap)
	}

	if !strings.ContainsRune(line, ':') {
		return line
	}

	var b strings.Builder
	for i := 0; i < len(line); {
		if line[i] != ':' {
			b.WriteByte(line[i])
			i++
			continue
		}
		name := placeholder(line[i+1:])
		written := false
		for n := len(name); n > 0; n-- {
			value, form, ok := t.replacement(replace, name[:n])
			if !ok {
				continue
			}
			b.WriteString(form(value))
			i += 1 + n
			written = true
			break
		}
		if !written {
			b.WriteByte(':')
			i++
		}
	}
	return b.String()
}

// replacement finds the argument one spelling of a placeholder names, and the
// case that spelling asks for. ":attribute" is the value, ":Attribute" is it
// with the first rune uppercased, and ":ATTRIBUTE" is it uppercased.
func (t *Translator) replacement(replace Replace, name string) (string, func(string) string, bool) {
	if value, ok := replace[name]; ok {
		if _, wrap := value.(func(string) string); wrap {
			// A function argument names a pair of tags, not a placeholder, so
			// it is skipped here rather than printed.
			return "", nil, false
		}
		return t.stringify(value), func(s string) string { return s }, true
	}
	lower := strings.ToLower(name)
	if lower == name {
		return "", nil, false
	}
	value, ok := replace[lower]
	if !ok {
		return "", nil, false
	}
	if _, wrap := value.(func(string) string); wrap {
		return "", nil, false
	}
	if name == strings.ToUpper(name) {
		return t.stringify(value), strings.ToUpper, true
	}
	if name == ucfirst(lower) {
		return t.stringify(value), ucfirst, true
	}
	return "", nil, false
}

// stringify renders one argument. A type a handler was registered for through
// [Translator.Stringable] goes through it; anything else goes through
// fmt.Sprint.
func (t *Translator) stringify(value any) string {
	if value == nil {
		return ""
	}
	t.mu.RLock()
	handler, ok := t.stringableHandlers[reflect.TypeOf(value)]
	t.mu.RUnlock()
	if ok {
		return handler(value)
	}
	return fmt.Sprint(value)
}

// replaceWrapped is the function branch of makeReplacements: every
// <name>text</name> in the line, with text passed through the function.
//
// The text does not cross a newline.
func replaceWrapped(line, name string, wrap func(string) string) string {
	opening, closing := "<"+name+">", "</"+name+">"
	var b strings.Builder
	for {
		start := strings.Index(line, opening)
		if start < 0 {
			break
		}
		rest := line[start+len(opening):]
		end := strings.Index(rest, closing)
		if end < 0 {
			break
		}
		inner := rest[:end]
		if strings.ContainsRune(inner, '\n') {
			// A newline between the tags is beyond what a match spans, so the
			// tags stay on the page as written.
			b.WriteString(line[:start+len(opening)])
			line = rest
			continue
		}
		b.WriteString(line[:start])
		b.WriteString(wrap(inner))
		line = rest[end+len(closing):]
	}
	b.WriteString(line)
	return b.String()
}

// AddLines answers Translator::addLines(): translation lines written straight
// into the loaded array, under keys of the form "group.item".
//
// An empty namespace means [AppNamespace]. A group written this way is marked
// loaded, so the loader is never asked for it -- which is how a test, or a
// module registering its own text at boot, replaces a group rather than merging
// into one.
func (t *Translator) AddLines(lines map[string]string, locale, namespace string) {
	if namespace == "" {
		namespace = AppNamespace
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	for key, value := range lines {
		group, item, found := strings.Cut(key, ".")
		if !found {
			continue
		}
		t.set(namespace, group, locale, item, value)
	}
}

// set writes one item into the loaded array, creating the levels above it. The
// caller holds the write lock.
func (t *Translator) set(namespace, group, locale, item, value string) {
	if t.loaded[namespace] == nil {
		t.loaded[namespace] = make(map[string]map[string]Lines)
	}
	if t.loaded[namespace][group] == nil {
		t.loaded[namespace][group] = make(map[string]Lines)
	}
	if t.loaded[namespace][group][locale] == nil {
		t.loaded[namespace][group][locale] = make(Lines)
	}
	t.loaded[namespace][group][locale][item] = value
}

// Load answers Translator::load(): it asks the loader for one group of one
// locale and keeps what comes back.
//
// A group that is already loaded is not loaded again, which is what makes
// [Translator.AddLines] and [Translator.SetLoaded] stick.
func (t *Translator) Load(namespace, group, locale string) {
	if t.isLoaded(namespace, group, locale) {
		return
	}

	lines := t.lookup.Load(locale, group, namespace)

	t.mu.Lock()
	defer t.mu.Unlock()
	if t.isLoadedLocked(namespace, group, locale) {
		// Another request loaded it while this one was reading the files.
		return
	}
	if t.loaded[namespace] == nil {
		t.loaded[namespace] = make(map[string]map[string]Lines)
	}
	if t.loaded[namespace][group] == nil {
		t.loaded[namespace][group] = make(map[string]Lines)
	}
	t.loaded[namespace][group][locale] = lines
}

// isLoaded answers Translator::isLoaded().
func (t *Translator) isLoaded(namespace, group, locale string) bool {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.isLoadedLocked(namespace, group, locale)
}

func (t *Translator) isLoadedLocked(namespace, group, locale string) bool {
	groups, ok := t.loaded[namespace]
	if !ok {
		return false
	}
	locales, ok := groups[group]
	if !ok {
		return false
	}
	_, ok = locales[locale]
	return ok
}

// handleMissingTranslationKey answers
// Translator::handleMissingTranslationKey(). It hands the key to the callback
// registered by [Translator.HandleMissingKeysUsing] and returns whatever that
// answers, or the key itself.
func (t *Translator) handleMissingTranslationKey(key string, replace Replace, locale string, fallback bool) string {
	t.mu.Lock()
	callback := t.missingTranslationKeyCallback
	if !t.handleMissingTranslationKeys || callback == nil {
		t.mu.Unlock()
		return key
	}
	// Prevent infinite loops: a callback that translates is not reported for
	// the key it is reporting.
	t.handleMissingTranslationKeys = false
	t.mu.Unlock()

	answered := callback(key, replace, locale, fallback)

	t.mu.Lock()
	t.handleMissingTranslationKeys = true
	t.mu.Unlock()

	if answered == "" {
		return key
	}
	return answered
}

// HandleMissingKeysUsing answers Translator::handleMissingKeysUsing(): it
// registers what runs when no catalogue carries a key.
//
// The translator still returns the key; this is the hook that reports it, to a
// logger in development or to a counter in production, and it may answer with a
// sentence of its own. A nil callback removes the one registered.
func (t *Translator) HandleMissingKeysUsing(callback func(key string, replace Replace, locale string, fallback bool) string) *Translator {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.missingTranslationKeyCallback = callback
	return t
}

// AddNamespace answers Translator::addNamespace(): it points a namespace at the
// place its lines are loaded from, by forwarding to the loader.
func (t *Translator) AddNamespace(namespace, hint string) {
	t.lookup.AddNamespace(namespace, hint)
}

// AddPath registers another directory of locale directories for the loader to
// read. A loader with no notion of a path ignores it.
func (t *Translator) AddPath(path string) {
	if adder, ok := t.lookup.(interface{ AddPath(string) }); ok {
		adder.AddPath(path)
	}
}

// AddJSONPath answers Translator::addJsonPath(): another directory of per
// locale JSON catalogues for the loader to read.
func (t *Translator) AddJSONPath(path string) {
	t.lookup.AddJSONPath(path)
}

// ParseKey answers Translator::parseKey(): a key split into namespace, group
// and item.
//
// "messages.welcome" is the item "welcome" of the group "messages" in the
// application namespace; "shop::orders.title" names the namespace "shop"; and
// "validation.min.string" is the item "min.string", because everything after
// the first dot is the item. A key with no dot names a group and no item, and
// the item comes back empty.
//
// A key with no namespace resolves in [AppNamespace].
func (t *Translator) ParseKey(key string) (namespace, group, item string) {
	if i := strings.Index(key, "::"); i >= 0 {
		namespace, key = key[:i], key[i+2:]
	}
	if namespace == "" {
		namespace = AppNamespace
	}
	group, item, _ = strings.Cut(key, ".")
	return namespace, group, item
}

// localeArray answers Translator::localeArray(): the locales to be checked, in
// order, being the one asked for and then the fallback.
func (t *Translator) localeArray(locale string) []string {
	locales := make([]string, 0, 2)
	for _, l := range []string{t.localeOr(locale), t.GetFallback()} {
		if l != "" && (len(locales) == 0 || locales[0] != l) {
			locales = append(locales, l)
		}
	}

	t.mu.RLock()
	determine := t.determineLocalesUsing
	t.mu.RUnlock()
	if determine != nil {
		return determine(locales)
	}
	return locales
}

// DetermineLocalesUsing answers Translator::determineLocalesUsing(): it
// registers a callback that says which locales a key is looked for in, given
// the locale asked for and the fallback.
//
// It is what a regional catalogue needs: a translator asked for pt-BR can
// answer out of pt before it reaches the fallback.
func (t *Translator) DetermineLocalesUsing(callback func(locales []string) []string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.determineLocalesUsing = callback
}

// GetSelector answers Translator::getSelector(): the thing that picks a segment
// of a pluralised line.
func (t *Translator) GetSelector() Selector {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.selector
}

// SetSelector answers Translator::setSelector(). A nil selector is ignored: a
// translator with no selector cannot answer Choice at all.
func (t *Translator) SetSelector(selector Selector) {
	if selector == nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.selector = selector
}

// GetLoader answers Translator::getLoader(): the loader this translator was
// built with.
//
// It is not the loader the lookups go through -- that one is this loader
// followed by the English lines that ship with the package. Returning the pair
// would hand out a way to write into the bundled catalogue.
func (t *Translator) GetLoader() Loader {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.loader
}

// Locale is the locale used when a caller passes none. It is
// [Translator.GetLocale] under the name that reads better in a sentence.
func (t *Translator) Locale() string { return t.GetLocale() }

// GetLocale answers Translator::getLocale(): the locale used when a caller
// passes none.
func (t *Translator) GetLocale() string {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.locale
}

// SetLocale answers Translator::setLocale(): it sets the locale used when a
// caller passes none.
//
// A locale holding "/" or "\" is refused, because a locale reaches the
// filesystem as a directory name and one carrying a separator reads a catalogue
// somewhere else.
func (t *Translator) SetLocale(locale string) error {
	if strings.ContainsAny(locale, `/\`) {
		return fmt.Errorf("translation: invalid characters present in locale %q", locale)
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.locale = locale
	return nil
}

// GetFallback answers Translator::getFallback(): the locale consulted when a
// key is missing from the one asked for.
func (t *Translator) GetFallback() string {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.fallback
}

// SetFallback answers Translator::setFallback().
func (t *Translator) SetFallback(fallback string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.fallback = fallback
}

// SetLoaded answers Translator::setLoaded(): it replaces the loaded groups
// wholesale, keyed by namespace, then group, then locale.
//
// Passing nil empties it, which is how a long lived process drops a catalogue
// it has finished with.
func (t *Translator) SetLoaded(loaded map[string]map[string]map[string]Lines) {
	copied := make(map[string]map[string]map[string]Lines, len(loaded))
	for namespace, groups := range loaded {
		copied[namespace] = make(map[string]map[string]Lines, len(groups))
		for group, locales := range groups {
			copied[namespace][group] = make(map[string]Lines, len(locales))
			for locale, lines := range locales {
				copied[namespace][group][locale] = maps.Clone(lines)
			}
		}
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.loaded = copied
}

// Stringable registers how one type is rendered when it is passed as a
// replacement.
//
// The type is named by a zero value of it: pass money.Amount{} to say how a
// money.Amount renders. The handler receives the value that was passed to
// [Translator.Get] and must assert it back.
//
// A nil handler removes the one registered for the type.
func (t *Translator) Stringable(class any, handler func(any) string) {
	if class == nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if handler == nil {
		delete(t.stringableHandlers, reflect.TypeOf(class))
		return
	}
	t.stringableHandlers[reflect.TypeOf(class)] = handler
}

func (t *Translator) localeOr(locale string) string {
	if locale == "" {
		return t.GetLocale()
	}
	return locale
}

// placeholder reads the name that follows a colon: letters, digits and
// underscores, which is every placeholder the catalogue uses.
func placeholder(s string) string {
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '_' {
			continue
		}
		return s[:i]
	}
	return s
}

// ucfirst uppercases the first rune, and leaves the rest alone.
func ucfirst(s string) string {
	if s == "" {
		return s
	}
	first, size := utf8.DecodeRuneInString(s)
	return string(unicode.ToUpper(first)) + s[size:]
}
