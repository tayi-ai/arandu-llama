package arr

import (
	"reflect"
	"sort"
	"strconv"
	"strings"
)

// Query renders the map as a query string.
//
// The separator is "&", a nested container becomes name[key], and both sides
// are encoded per RFC 3986 -- a space is %20 and never a plus, and "-_.~" are
// left alone. A nil value is dropped, and a bool is "1" or "0".
//
// The pairs come out in ascending key order, so the result is reproducible
// from the value alone.
func Query(array map[string]any) string {
	return strings.Join(queryPairs("", array), "&")
}

func queryPairs(prefix string, node any) []string {
	out := []string{}
	for _, key := range keys(node) {
		value, ok := index(node, key)
		if !ok || value == nil {
			continue
		}
		name := key
		if prefix != "" {
			name = prefix + "[" + key + "]"
		}
		if _, nested := container(value); nested {
			out = append(out, queryPairs(name, value)...)
			continue
		}
		out = append(out, rawURLEncode(name)+"="+rawURLEncode(queryValue(value)))
	}
	return out
}

// queryValue renders a scalar as the string a query pair carries.
func queryValue(value any) string {
	switch typed := value.(type) {
	case bool:
		if typed {
			return "1"
		}
		return "0"
	case string:
		return typed
	case float64:
		return strconv.FormatFloat(typed, 'g', -1, 64)
	case float32:
		return strconv.FormatFloat(float64(typed), 'g', -1, 32)
	default:
		return keyString(value)
	}
}

// rawURLEncode percent-encodes everything but the unreserved set of RFC 3986.
// It is written out because net/url's escapers follow different rules --
// QueryEscape turns a space into a plus.
func rawURLEncode(s string) string {
	const hex = "0123456789ABCDEF"
	var b strings.Builder
	b.Grow(len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case c >= 'A' && c <= 'Z', c >= 'a' && c <= 'z', c >= '0' && c <= '9',
			c == '-', c == '_', c == '.', c == '~':
			b.WriteByte(c)
		default:
			b.WriteByte('%')
			b.WriteByte(hex[c>>4])
			b.WriteByte(hex[c&0x0f])
		}
	}
	return b.String()
}

// ToCssClasses builds a space-separated class list from entries that are
// either unconditional or conditional.
//
// Each argument is one entry, and it is either:
//
//	a string          the class, always
//	map[string]bool   the keys whose value is true, in ascending key order
//
// A nil argument contributes nothing; anything else is rendered as a string
// and taken as a class.
func ToCssClasses(array ...any) string {
	return strings.Join(conditionalList(array, ""), " ")
}

// ToCssStyles is ToCssClasses with every entry finished with a semicolon. An
// entry that already ends in one does not get a second.
func ToCssStyles(array ...any) string {
	return strings.Join(conditionalList(array, ";"), " ")
}

func conditionalList(entries []any, suffix string) []string {
	out := []string{}
	for _, entry := range entries {
		switch typed := entry.(type) {
		case string:
			out = append(out, finish(typed, suffix))
		case map[string]bool:
			names := make([]string, 0, len(typed))
			for name := range typed {
				names = append(names, name)
			}
			sort.Strings(names)
			for _, name := range names {
				if typed[name] {
					out = append(out, finish(name, suffix))
				}
			}
		case nil:
			continue
		default:
			out = append(out, finish(keyString(entry), suffix))
		}
	}
	return out
}

// finish ends the string with cap, unless it already does.
func finish(s, cap string) string {
	if cap == "" || strings.HasSuffix(s, cap) {
		return s
	}
	return s + cap
}

// SortRecursive sorts every nested list, all the way down.
//
// A map holds no order to establish, so it is walked and rebuilt with its
// values sorted recursively and its keys left as they are -- which is the
// whole of what "sorted" can mean for a map. Lists are sorted.
//
// The result is built fresh, so the argument is untouched. It is a
// map[string]any or a []any whatever the concrete types going in were, because
// the recursion has to hold both.
//
// The optional descending reverses the order; only the first one is used.
// Comparison is numeric between numbers, lexicographic between strings, false
// before true between bools, and otherwise by the order nil, bool, number,
// string, container.
func SortRecursive(array any, descending ...bool) any {
	desc := false
	if len(descending) > 0 {
		desc = descending[0]
	}
	return sortRecursive(array, desc)
}

// SortRecursiveDesc is SortRecursive with the order reversed.
func SortRecursiveDesc(array any) any { return SortRecursive(array, true) }

func sortRecursive(value any, descending bool) any {
	rv, ok := container(value)
	if !ok {
		return value
	}
	if rv.Kind() == reflect.Map {
		out := make(map[string]any, rv.Len())
		for _, key := range keys(value) {
			inner, _ := index(value, key)
			out[key] = sortRecursive(inner, descending)
		}
		return out
	}
	out := make([]any, rv.Len())
	for i := range out {
		out[i] = sortRecursive(rv.Index(i).Interface(), descending)
	}
	sort.SliceStable(out, func(i, j int) bool {
		if descending {
			return compareValues(out[i], out[j]) > 0
		}
		return compareValues(out[i], out[j]) < 0
	})
	return out
}

// compareValues orders two values of any type: numbers numerically, strings
// lexicographically, false before true, and values of different kinds by the
// order nil, bool, number, string, container.
func compareValues(a, b any) int {
	ra, rb := valueRank(a), valueRank(b)
	if ra != rb {
		if ra < rb {
			return -1
		}
		return 1
	}
	switch ra {
	case rankBool:
		left, right := a.(bool), b.(bool)
		switch {
		case left == right:
			return 0
		case !left:
			return -1
		default:
			return 1
		}
	case rankNumber:
		left, right := asFloat(a), asFloat(b)
		switch {
		case left < right:
			return -1
		case left > right:
			return 1
		default:
			return 0
		}
	case rankString:
		return strings.Compare(keyString(a), keyString(b))
	default:
		return 0
	}
}

const (
	rankNil = iota
	rankBool
	rankNumber
	rankString
	rankArray
)

func valueRank(value any) int {
	if value == nil {
		return rankNil
	}
	switch reflect.ValueOf(value).Kind() {
	case reflect.Bool:
		return rankBool
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr,
		reflect.Float32, reflect.Float64:
		return rankNumber
	case reflect.String:
		return rankString
	case reflect.Map, reflect.Slice, reflect.Array:
		return rankArray
	default:
		return rankString
	}
}

func asFloat(value any) float64 {
	rv := reflect.ValueOf(value)
	switch rv.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return float64(rv.Int())
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return float64(rv.Uint())
	case reflect.Float32, reflect.Float64:
		return rv.Float()
	default:
		return 0
	}
}
