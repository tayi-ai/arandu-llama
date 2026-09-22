// Package str transforms strings.
//
// It holds the string work a generator, a router and a validator all need and
// none of them owns: Slug, Snake, Studly, Camel, Kebab, Headline, Plural,
// Singular, Limit, Squish, Mask, Is, UUID, ULID and Random, among some eighty
// others.
//
// # Functions and the chain
//
// Every transformation is a package-level function:
//
//	str.Studly(str.Singular(name))
//
// Of returns a Stringable, which is the same transformations read the other way
// round:
//
//	str.Of(name).Singular().Studly().ToString()
//
// They are one implementation with two spellings of the call, not two
// implementations: every Stringable method forwards to the function of the same
// name.
//
// # Optional arguments
//
// Go has no default arguments. A trailing run of optional arguments of one type
// arrives as a variadic tail -- Substr(value, start), Substr(value, start,
// length) -- and where a boolean or a second type follows them, every argument
// is spelled out instead, as on Limit and Replace.
//
// # What it does not do
//
// Nothing here is locale-aware, because locale-aware text needs CLDR data and
// this package carries no dependency to hold it. ASCII folds a table of
// Latin-1 and Latin Extended-A runes and drops what it does not know, and
// Transliterate puts a stand-in in the place of what that table cannot spell;
// Plural and Singular are English, and every irregular they get right is a
// line in a table in inflect.go, not a rule. UseLanguage refuses any language
// but English rather than answer in English under another one's name.
// Locale-correct pluralization belongs to the translation package, not this
// one.
//
// Markdown and InlineMarkdown are this package's own renderer, so that no
// third-party dependency is needed to render a document. They cover CommonMark
// and the GitHub flavour's strikethrough, task lists, tables and bare-URL
// autolinking, and they take no options.
package str
