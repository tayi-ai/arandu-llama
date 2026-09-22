package routing

import "strings"

// Merge folds a group's attributes into the ones it is nested inside, and
// returns the combined set.
//
// Prefix concatenates with a slash, name with nothing (the dot is the
// caller's), namespace with a backslash, where unions with the new winning.
// Domain and controller replace rather than merge: a nested group that names
// a domain is declaring one, not adding to one.
//
// prependExistingPrefix defaults to true. False puts the new prefix in front,
// which is how a group registered under a versioned API root keeps the version
// segment outermost.
func Merge(attributes, old map[string]any, prependExistingPrefix ...bool) map[string]any {
	prepend := len(prependExistingPrefix) == 0 || prependExistingPrefix[0]

	out := map[string]any{}
	for k, v := range old {
		switch k {
		case "namespace", "prefix", "where", "as":
			// Replaced below by the formatted value.
		default:
			out[k] = v
		}
	}

	if _, declared := attributes["domain"]; declared {
		delete(out, "domain")
	}
	if _, declared := attributes["controller"]; declared {
		delete(out, "controller")
	}

	for k, v := range attributes {
		switch k {
		case "namespace", "prefix", "where", "as":
		default:
			out[k] = v
		}
	}

	if ns := formatNamespace(attributes, old); ns != "" {
		out["namespace"] = ns
	}
	if prefix := formatPrefix(attributes, old, prepend); prefix != "" {
		out["prefix"] = prefix
	}
	if where := formatWhere(attributes, old); len(where) > 0 {
		out["where"] = where
	}
	if as := formatAs(attributes, old); as != "" {
		out["as"] = as
	}

	return out
}

// formatNamespace merges a group's namespace with the enclosing one.
func formatNamespace(attributes, old map[string]any) string {
	oldNamespace := getString(old, "namespace")

	newNamespace, declared := attributes["namespace"].(string)
	if !declared {
		return oldNamespace
	}
	if oldNamespace != "" && !strings.HasPrefix(newNamespace, `\`) {
		return strings.Trim(oldNamespace, `\`) + `\` + strings.Trim(newNamespace, `\`)
	}
	return strings.Trim(newNamespace, `\`)
}

// formatPrefix merges a group's prefix with the enclosing one.
func formatPrefix(attributes, old map[string]any, prependExistingPrefix bool) string {
	oldPrefix := getString(old, "prefix")

	newPrefix, declared := attributes["prefix"].(string)
	if !declared {
		return oldPrefix
	}
	if prependExistingPrefix {
		return strings.Trim(strings.Trim(oldPrefix, "/")+"/"+strings.Trim(newPrefix, "/"), "/")
	}
	return strings.Trim(strings.Trim(newPrefix, "/")+"/"+strings.Trim(oldPrefix, "/"), "/")
}

// formatWhere merges a group's where constraints with the enclosing one's.
func formatWhere(attributes, old map[string]any) map[string]string {
	out := map[string]string{}
	if existing, ok := old["where"].(map[string]string); ok {
		for k, v := range existing {
			out[k] = v
		}
	}
	if incoming, ok := attributes["where"].(map[string]string); ok {
		for k, v := range incoming {
			out[k] = v
		}
	}
	return out
}

// formatAs concatenates a group's name prefix with the enclosing one's,
// literally: the dot is the caller's to remember here. Route.Name is where
// the dot is joined for you.
func formatAs(attributes, old map[string]any) string {
	newAs := getString(attributes, "as")
	if oldAs := getString(old, "as"); oldAs != "" {
		return oldAs + newAs
	}
	return newAs
}

// getString reads a string field from a map, or empty.
func getString(m map[string]any, key string) string {
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}
