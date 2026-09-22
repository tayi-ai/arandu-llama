package support

import (
	"strings"
	"sync"
)

// NamespacedItemResolver turns a key of the form "package::group.item" into
// its three parts. Configuration keys and translation keys are both read this
// way. It is safe for concurrent use.
type NamespacedItemResolver struct {
	mu     sync.Mutex
	parsed map[string]parsedKey
}

// parsedKey is the three parts a key breaks into.
type parsedKey struct {
	namespace string
	group     string
	item      string
}

// NewNamespacedItemResolver returns a resolver with an empty cache.
func NewNamespacedItemResolver() *NamespacedItemResolver {
	return &NamespacedItemResolver{parsed: map[string]parsedKey{}}
}

// ParseKey returns the namespace, the group and the item of a key, in that
// order. A key with no namespace, and a group asked for whole, give the empty
// string for the part they lack. Every key parsed is cached.
func (r *NamespacedItemResolver) ParseKey(key string) (namespace, group, item string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.parsed == nil {
		r.parsed = map[string]parsedKey{}
	}
	if held, ok := r.parsed[key]; ok {
		return held.namespace, held.group, held.item
	}

	var parsed parsedKey
	if !strings.Contains(key, "::") {
		parsed = parseBasicSegments(strings.Split(key, "."))
	} else {
		parsed = parseNamespacedSegments(key)
	}

	r.parsed[key] = parsed
	return parsed.namespace, parsed.group, parsed.item
}

// parseBasicSegments reads a key carrying no namespace: the first segment is
// the group, and the rest, rejoined by dots, is the item.
func parseBasicSegments(segments []string) parsedKey {
	parsed := parsedKey{group: segments[0]}
	if len(segments) > 1 {
		parsed.item = strings.Join(segments[1:], ".")
	}
	return parsed
}

// parseNamespacedSegments splits the namespace off at "::" and reads what
// follows as a key carrying no namespace.
func parseNamespacedSegments(key string) parsedKey {
	namespace, item, _ := strings.Cut(key, "::")
	parsed := parseBasicSegments(strings.Split(item, "."))
	parsed.namespace = namespace
	return parsed
}

// SetParsedKey caches the three parts of a key, so
// [NamespacedItemResolver.ParseKey] hands them back without reading it.
func (r *NamespacedItemResolver) SetParsedKey(key, namespace, group, item string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.parsed == nil {
		r.parsed = map[string]parsedKey{}
	}
	r.parsed[key] = parsedKey{namespace: namespace, group: group, item: item}
}

// FlushParsedKeys empties the cache.
func (r *NamespacedItemResolver) FlushParsedKeys() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.parsed = map[string]parsedKey{}
}
