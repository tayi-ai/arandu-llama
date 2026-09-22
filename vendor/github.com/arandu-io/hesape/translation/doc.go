// Package translation resolves a key into a sentence in a locale.
//
// It holds [Translator], [MessageSelector], [Selector], [ArrayLoader],
// [FileLoader], [Loader], [Lines], [Replace], [PotentiallyTranslatedString],
// the four English groups the framework produces sentences from, and
// [Negotiate], [Middleware], [InLocale] and [Locale] for the locale of a
// request.
//
// # The translator is a value
//
// [Translator] is built once where the application is wired and passed to
// whatever renders a sentence. There is no package level translator and no set
// of free functions that read one.
//
// It is safe for concurrent use: every field is behind a lock, so one instance
// answers every request at once while [Translator.AddLines] or
// [Translator.SetLocale] is called on it.
//
// The locale comes first on [Translator.Get], [Translator.Choice] and
// [Translator.Has]. An empty locale means the translator's own, so a caller
// that has no request still works.
//
// # Two catalogue formats
//
// A group key is "group.item": "auth.failed" is the item "failed" of the group
// "auth", and "validation.min.string" is the item "min.string", because
// everything after the first dot is the item. [Translator.ParseKey] is what
// splits one, and a "namespace::" prefix names a module's own catalogue.
//
// A JSON key is the sentence itself, read out of one file per locale --
// lang/pt-BR.json holding {"Save changes": "Salvar alterações"}.
// [Translator.Get] checks it before the groups, because a key with no dot names
// no item and would otherwise resolve to nothing.
//
// [FileLoader] reads both, out of an fs.FS: a group at
// <path>/<locale>/<group>.json, a JSON catalogue at <path>/<locale>.json, and
// an override an application published for a module at
// <path>/vendor/<namespace>/<locale>/<group>.json. A language file is data, and
// nested objects flatten to dotted items, so the four size shapes of a
// validation message stay grouped in the file and are read as "min.string".
//
// It parses every file under its initial paths before returning and reports a
// malformed one as an error there: a language file that cannot be parsed is a
// boot failure, not a sentence that goes missing on the one request that needed
// it. [ArrayLoader] is the same catalogue written in Go, which is what a test
// uses.
//
// The English lines for validation, auth, passwords and pagination are embedded
// here and answered after the application catalogue and its fallback locale. An
// application overrides one of them by defining the same key; it never has to
// publish a file to get a sentence.
//
// # Where the application's own catalogue goes
//
// The embedded English is a fallback, not the only path. An application carries
// its own catalogue, in as many locales as it has, and this package holds none
// of them: the sentences a product says are the product's, and a language
// shipped here would be a second place they are written.
//
// The catalogue is a directory of locale directories, which the application
// owns and names -- lang at the root of the project is the usual place. Inside
// it, a group is lang/<locale>/<group>.json and the sentence-keyed catalogue of
// a locale is lang/<locale>.json:
//
//	lang/pt-BR/validation.json
//	lang/pt-BR/auth.json
//	lang/pt-BR.json
//	lang/en/auth.json
//
// It is read by a [FileLoader] over any fs.FS -- os.DirFS for a directory on
// disk, embed.FS for one compiled into the binary -- and that loader is what
// [New] is built with, together with the locale to answer in and the locale to
// fall back to:
//
//	loader, err := translation.NewFileLoader(os.DirFS("."), "lang")
//	tr := translation.New(loader, "pt-BR", "en")
//
// Three things answer a key, in this order: the application's catalogue in the
// locale asked for, then its catalogue in the fallback locale, then the English
// embedded here. So a project that has translated forty validation lines into a
// locale gets those forty and English for the rest, which is what makes a
// catalogue worth shipping before it is finished.
//
// Overriding is per key, not per file: an auth.json carrying one item replaces
// that item and leaves the rest of the group answering from here. A language
// this package has never heard of needs no change to it -- [New] takes the
// locale as a string, and the catalogue is files.
//
// Files are not the only way in: [ArrayLoader] is the same catalogue written in
// Go, and [Translator.AddLines] writes lines into a translator already built,
// which is what a module registering its own text at boot uses.
//
// # Plurals
//
// [Translator.Choice] takes the count and hands the line to the [Selector],
// which is [MessageSelector] unless [Translator.SetSelector] replaced it. A
// segment may open with an explicit condition -- "{0} none|[1,19] some|[20,*] many" --
// and the first one that matches the count wins. With no condition, the count
// chooses by the plural rule of the locale: two forms in English, one in
// Japanese, three in Russian, six in Arabic.
//
// Only the segment a condition selected is trimmed, which is worth knowing
// before writing "one | :count many": the second segment renders with its
// leading space, because the plural rule is what chose it.
//
// [MessageSelector.GetPluralIndex] carries the rule table. It is keyed by the
// language subtag alone, because no region changes the plural forms of its
// language, and because an Accept-Language header carries a hyphenated tag.
//
// # Replacements
//
// [Replace] fills the placeholders of a line. Three spellings of every name are
// replaced: ":name" with the value, ":Name" with its first letter uppercased
// and ":NAME" with it uppercased, so one argument serves "The :attribute field
// is required." and ":Attribute is required.". A value of type
// func(string) string is applied differently: it replaces the text between
// <name> and </name> rather than a placeholder.
//
// A placeholder no argument names is left as it stands, and the longest name
// that does match wins at each position.
//
// # The locale of a request
//
// The locale is a parameter of [Translator.Get], [Translator.Choice] and
// [Translator.Has] rather than something the caller sets on the translator,
// because the locale belongs to the request and one shared translator serves
// all of them. [Middleware] negotiates it once from Accept-Language and puts it
// in the request context, and [InLocale] puts there a locale the route already
// decided; [Locale] reads back whichever of them ran. An application uses one or
// the other, because a language decided twice is a page with two addresses.
// [Translator.SetLocale] sets
// the default for a caller that passes none -- a console command, or a queued
// job -- and returns an error for a locale holding a path separator.
//
// [PotentiallyTranslatedString] carries a validation message that may or may
// not need translating. [PotentiallyTranslatedString.ToString] is what writes
// it back, and the validator calls it explicitly: nothing here runs when the
// value falls out of scope.
package translation
