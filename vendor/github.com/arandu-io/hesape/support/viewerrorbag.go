package support

import "sort"

// DefaultErrorBag is the key a [ViewErrorBag] reads when no bag is named.
const DefaultErrorBag = "default"

// ViewErrorBag holds the named [MessageBag] values a request carries, one of
// them under [DefaultErrorBag].
//
// The read side of MessageBag is repeated on this type, each method reading
// the default bag, so a view does not have to name it.
type ViewErrorBag struct {
	bags map[string]*MessageBag
}

// NewViewErrorBag returns a ViewErrorBag holding no bags.
func NewViewErrorBag() *ViewErrorBag {
	return &ViewErrorBag{bags: map[string]*MessageBag{}}
}

// HasBag reports whether a bag is held under the key. An empty key means
// [DefaultErrorBag].
func (v *ViewErrorBag) HasBag(key string) bool {
	if key == "" {
		key = DefaultErrorBag
	}
	_, ok := v.bags[key]
	return ok
}

// GetBag returns the bag under the key, or an empty [MessageBag] when there is
// none. It is never nil, so a view can call First on it unguarded. An empty
// key means [DefaultErrorBag].
func (v *ViewErrorBag) GetBag(key string) *MessageBag {
	if key == "" {
		key = DefaultErrorBag
	}
	if bag, ok := v.bags[key]; ok && bag != nil {
		return bag
	}
	return NewMessageBag(nil)
}

// GetBags returns a copy of the map of bags, keyed by name.
func (v *ViewErrorBag) GetBags() map[string]*MessageBag {
	out := make(map[string]*MessageBag, len(v.bags))
	for k, bag := range v.bags {
		out[k] = bag
	}
	return out
}

// Put stores a bag under the key and returns the ViewErrorBag. An empty key
// means [DefaultErrorBag].
func (v *ViewErrorBag) Put(key string, bag *MessageBag) *ViewErrorBag {
	if v.bags == nil {
		v.bags = map[string]*MessageBag{}
	}
	if key == "" {
		key = DefaultErrorBag
	}
	v.bags[key] = bag
	return v
}

// Any reports whether the default bag holds any message.
func (v *ViewErrorBag) Any() bool { return v.Count() > 0 }

// Count returns how many messages the default bag holds.
func (v *ViewErrorBag) Count() int { return v.GetBag(DefaultErrorBag).Count() }

// String returns the default bag as JSON, so ViewErrorBag satisfies
// fmt.Stringer.
func (v *ViewErrorBag) String() string { return v.GetBag(DefaultErrorBag).String() }

// Names lists the bag names, sorted. A map has no order of its own, so sorted
// is the only order that can be promised.
func (v *ViewErrorBag) Names() []string {
	out := make([]string, 0, len(v.bags))
	for k := range v.bags {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// First returns the first message under the key in the default bag.
func (v *ViewErrorBag) First(key string, format ...string) string {
	return v.GetBag(DefaultErrorBag).First(key, format...)
}

// Get returns the messages under the key in the default bag.
func (v *ViewErrorBag) Get(key string, format ...string) []string {
	return v.GetBag(DefaultErrorBag).Get(key, format...)
}

// All returns every message in the default bag.
func (v *ViewErrorBag) All(format ...string) []string {
	return v.GetBag(DefaultErrorBag).All(format...)
}

// Unique returns every message in the default bag, with duplicates dropped.
func (v *ViewErrorBag) Unique(format ...string) []string {
	return v.GetBag(DefaultErrorBag).Unique(format...)
}

// Has reports whether the default bag holds a message for every key given.
func (v *ViewErrorBag) Has(keys ...string) bool {
	return v.GetBag(DefaultErrorBag).Has(keys...)
}

// HasAny reports whether the default bag holds a message for any key given.
func (v *ViewErrorBag) HasAny(keys ...string) bool {
	return v.GetBag(DefaultErrorBag).HasAny(keys...)
}

// Missing reports whether the default bag holds no message for any of the
// keys.
func (v *ViewErrorBag) Missing(keys ...string) bool {
	return v.GetBag(DefaultErrorBag).Missing(keys...)
}

// Keys lists the keys the default bag holds messages under.
func (v *ViewErrorBag) Keys() []string { return v.GetBag(DefaultErrorBag).Keys() }

// Messages returns the default bag's messages, keyed by field.
func (v *ViewErrorBag) Messages() map[string][]string {
	return v.GetBag(DefaultErrorBag).Messages()
}

// ToArray returns the default bag's messages keyed by field, which is what a
// view writes out when it dumps the errors.
func (v *ViewErrorBag) ToArray() map[string][]string {
	return v.GetBag(DefaultErrorBag).ToArray()
}

// ToJson encodes the default bag's messages as JSON.
func (v *ViewErrorBag) ToJson() (string, error) {
	return v.GetBag(DefaultErrorBag).ToJson()
}

// IsEmpty reports whether the default bag holds no message.
func (v *ViewErrorBag) IsEmpty() bool { return v.GetBag(DefaultErrorBag).IsEmpty() }

// IsNotEmpty reports whether the default bag holds any message.
func (v *ViewErrorBag) IsNotEmpty() bool { return v.GetBag(DefaultErrorBag).IsNotEmpty() }
