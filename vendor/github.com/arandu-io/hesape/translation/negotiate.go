package translation

import (
	"cmp"
	"context"
	"net/http"
	"slices"
	"strconv"
	"strings"
)

type ctxLocaleKey struct{}

// WithLocale returns a context carrying the locale of the request.
func WithLocale(ctx context.Context, locale string) context.Context {
	return context.WithValue(ctx, ctxLocaleKey{}, locale)
}

// Locale reads the locale [Middleware] negotiated for the request, and returns
// the empty string when there is none.
//
// The empty string is what [Translator.T] reads as "the default locale", so a
// caller passes this straight through without checking it.
func Locale(ctx context.Context) string {
	locale, _ := ctx.Value(ctxLocaleKey{}).(string)
	return locale
}

// Middleware negotiates the locale of every request from its Accept-Language
// header and puts it in the context, where [Locale] reads it.
//
// This is one of the two ways an application decides a language, and [InLocale]
// is the other. Here the header decides and one address serves every language;
// there the path decides and each language has its own address. An application
// picks one of them: carrying both means one page has two addresses and a reader
// can be handed either, which is the thing that goes wrong, not the path itself.
//
// The header is the only input on this side. A locale read from a query
// parameter or a cookie on top of it is a third answer to a question already
// answered, and each of them is another cache key for one page.
//
// It answers with Content-Language, and adds Accept-Language to Vary: a page
// whose text depends on a request header and does not say so is a page a shared
// cache will serve in the wrong language. Vary is the truth about a negotiated
// response and belongs on every response this wraps -- which is also why it must
// not be reached for when nothing was negotiated, and why [InLocale] exists
// rather than a flag here.
func Middleware(supported []string, fallback string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			locale := Negotiate(r.Header.Get("Accept-Language"), supported, fallback)
			w.Header().Add("Vary", "Accept-Language")
			w.Header().Set("Content-Language", locale)
			next.ServeHTTP(w, r.WithContext(WithLocale(r.Context(), locale)))
		})
	}
}

// InLocale puts a locale the route already decided on every request that reaches
// it, and says so in the response.
//
// It goes on the route group of one language: the group under /es carries
// InLocale("es"), and the group with no prefix carries whichever language the
// unprefixed addresses are written in. The path is the whole input -- no header
// is read here, and no cookie and no query parameter, because each of those
// would be a second answer to a question the address already answered.
//
// It writes Content-Language and does not write Vary. What these pages say is a
// function of the path and of nothing else: one address is in one language for
// everybody who asks for it, so a shared cache in front of the application may
// keep one copy of it. Vary here would be false, and it would be paid for on
// every response rather than on the one that negotiated.
//
// [Locale] reads the locale back, spelled the way it was passed, so it indexes
// the catalogue directly. An application that has more to say about a language
// than its catalogue key -- a BCP 47 tag for hreflang, a label for a selector --
// keeps that beside its own list of languages and passes the key here.
func InLocale(locale string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Language", locale)
			next.ServeHTTP(w, r.WithContext(WithLocale(r.Context(), locale)))
		})
	}
}

// Negotiate picks the locale of an Accept-Language header out of the ones the
// application supports, and returns fallback when none of them fit.
//
// Tags are read in the order of their quality value, highest first, and ties
// keep the order the client wrote. A tag matches a supported locale exactly,
// case insensitively and whichever separator either spells it with; failing
// that it matches by language, so a client asking for pt-BR is served pt and a
// client asking for pt is served the first pt-* on offer. "*" takes the first
// supported locale. A tag at q=0 is refused rather than matched.
//
// The answer is always spelled the way the application spells it, so it indexes
// the catalogue directly.
func Negotiate(header string, supported []string, fallback string) string {
	if len(supported) == 0 {
		return fallback
	}

	for _, tag := range accepted(header) {
		if tag == "*" {
			return supported[0]
		}
		for _, locale := range supported {
			if strings.EqualFold(canonical(tag), canonical(locale)) {
				return locale
			}
		}
		for _, locale := range supported {
			if language(tag) == language(locale) {
				return locale
			}
		}
	}
	return fallback
}

// canonical spells a tag with one separator, so pt_BR and pt-BR compare equal.
func canonical(tag string) string { return strings.ReplaceAll(tag, "_", "-") }

// accepted reads the tags of an Accept-Language header, best first.
func accepted(header string) []string {
	type candidate struct {
		tag     string
		quality float64
	}

	var candidates []candidate
	for _, part := range strings.Split(header, ",") {
		tag, params, _ := strings.Cut(part, ";")
		tag = strings.TrimSpace(tag)
		if tag == "" {
			continue
		}
		quality := 1.0
		for _, param := range strings.Split(params, ";") {
			name, value, ok := strings.Cut(param, "=")
			if !ok || strings.ToLower(strings.TrimSpace(name)) != "q" {
				continue
			}
			q, err := strconv.ParseFloat(strings.TrimSpace(value), 64)
			if err != nil {
				quality = 0
				break
			}
			quality = q
		}
		if quality <= 0 {
			continue
		}
		candidates = append(candidates, candidate{tag, quality})
	}

	slices.SortStableFunc(candidates, func(a, b candidate) int {
		return cmp.Compare(b.quality, a.quality)
	})

	tags := make([]string, len(candidates))
	for i, c := range candidates {
		tags[i] = c.tag
	}
	return tags
}
