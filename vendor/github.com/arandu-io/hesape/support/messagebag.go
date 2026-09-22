package support

import (
	"encoding/json"
	"sort"
	"strings"
)

// DefaultMessageFormat is what a [MessageBag] wraps its messages in until
// [MessageBag.SetFormat] says otherwise: nothing at all, the message on its own.
//
// A format is a template read on the way out, in [MessageBag.Get],
// [MessageBag.First] and [MessageBag.All]: ":message" is replaced by the
// message and ":key" by the key it is filed under, so a bag set to
// "<li>:message</li>" hands every message back already wrapped. The messages
// held in the bag are never changed, only the copies it returns.
const DefaultMessageFormat = ":message"

// MessageProvider is implemented by a value that carries validation messages
// and can hand them over as a [MessageBag]. Taking the interface instead of the
// bag lets a caller pass whatever produced the errors and leave it to decide
// which messages to expose.
//
// *MessageBag implements it by returning itself, so any type that holds a bag
// satisfies it by delegating one method.
type MessageProvider interface {
	// GetMessageBag returns the messages the value carries.
	GetMessageBag() *MessageBag
}

// MessageBag is the keyed list of messages a validator hands to a view.
//
// A map has no order of its own, so the bag carries the key order beside it:
// [MessageBag.All], [MessageBag.Keys] and [MessageBag.Unique] walk the keys in
// the order they were first added.
type MessageBag struct {
	messages map[string][]string
	order    []string
	format   string
}

// NewMessageBag builds a bag over the given messages. Each list is
// deduplicated, keeping the first occurrence.
//
// A map has no order, so the initial keys are sorted to make the bag
// deterministic. Keys added later with [MessageBag.Add] keep their insertion
// order.
func NewMessageBag(messages map[string][]string) *MessageBag {
	b := &MessageBag{
		messages: make(map[string][]string, len(messages)),
		format:   DefaultMessageFormat,
	}
	keys := make([]string, 0, len(messages))
	for k := range messages {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		b.messages[k] = unique(messages[k])
		b.order = append(b.order, k)
	}
	return b
}

func unique(values []string) []string {
	seen := make(map[string]struct{}, len(values))
	out := make([]string, 0, len(values))
	for _, v := range values {
		if _, ok := seen[v]; ok {
			continue
		}
		seen[v] = struct{}{}
		out = append(out, v)
	}
	return out
}

// Keys returns the keys the bag holds messages under, in order.
func (b *MessageBag) Keys() []string {
	out := make([]string, len(b.order))
	copy(out, b.order)
	return out
}

// Add files a message under a key and returns the bag. A key and message pair
// the bag already holds is not added twice.
func (b *MessageBag) Add(key, message string) *MessageBag {
	if !b.isUnique(key, message) {
		return b
	}
	if b.messages == nil {
		b.messages = map[string][]string{}
	}
	if _, ok := b.messages[key]; !ok {
		b.order = append(b.order, key)
	}
	b.messages[key] = append(b.messages[key], message)
	return b
}

// AddIf adds the message only when the condition holds, and returns the bag.
func (b *MessageBag) AddIf(boolean bool, key, message string) *MessageBag {
	if boolean {
		return b.Add(key, message)
	}
	return b
}

func (b *MessageBag) isUnique(key, message string) bool {
	existing, ok := b.messages[key]
	if !ok {
		return true
	}
	for _, m := range existing {
		if m == message {
			return false
		}
	}
	return true
}

// Merge appends the given messages and returns the bag, so a key present on
// both sides ends up carrying both lists, duplicates included.
//
// To merge a [MessageProvider], pass provider.GetMessageBag().GetMessages().
func (b *MessageBag) Merge(messages map[string][]string) *MessageBag {
	if b.messages == nil {
		b.messages = map[string][]string{}
	}
	keys := make([]string, 0, len(messages))
	for k := range messages {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		if _, ok := b.messages[k]; !ok {
			b.order = append(b.order, k)
		}
		b.messages[k] = append(b.messages[k], messages[k]...)
	}
	return b
}

// Has reports whether the bag holds a message for every key given. With no key
// it reports whether the bag holds anything at all.
func (b *MessageBag) Has(keys ...string) bool {
	if b.IsEmpty() {
		return false
	}
	if len(keys) == 0 {
		return b.Any()
	}
	for _, key := range keys {
		if b.First(key) == "" {
			return false
		}
	}
	return true
}

// HasAny reports whether the bag holds a message for any key given. With no
// key it is false.
func (b *MessageBag) HasAny(keys ...string) bool {
	if b.IsEmpty() {
		return false
	}
	for _, key := range keys {
		if b.Has(key) {
			return true
		}
	}
	return false
}

// Missing reports whether the bag holds no message for any of the keys.
func (b *MessageBag) Missing(keys ...string) bool {
	return !b.HasAny(keys...)
}

// First returns the first message under the key, or the empty string when
// there is none. An empty key takes the first message in the whole bag. The
// variadic argument overrides the bag's format; only the first is read.
func (b *MessageBag) First(key string, format ...string) string {
	var messages []string
	if key == "" {
		messages = b.All(format...)
	} else {
		messages = b.Get(key, format...)
	}
	if len(messages) == 0 {
		return ""
	}
	return messages[0]
}

// Get returns the messages under the key, formatted. A key holding a * is a
// pattern: * stands for any run of characters, including none, and every other
// rune is literal. The matches come back flattened, in key order.
func (b *MessageBag) Get(key string, format ...string) []string {
	f := b.checkFormat(format...)
	if messages, ok := b.messages[key]; ok {
		return b.transform(messages, f, key)
	}
	if strings.Contains(key, "*") {
		return b.getMessagesForWildcardKey(key, f)
	}
	return []string{}
}

func (b *MessageBag) getMessagesForWildcardKey(key, format string) []string {
	out := []string{}
	for _, k := range b.order {
		if wildcardIs(key, k) {
			out = append(out, b.transform(b.messages[k], format, k)...)
		}
	}
	return out
}

// wildcardIs matches a pattern against a value: * stands for any run of
// characters, and every other rune is literal. It is kept here so the message
// bag carries no dependency of its own.
func wildcardIs(pattern, value string) bool {
	if pattern == value {
		return true
	}
	parts := strings.Split(pattern, "*")
	if len(parts) == 1 {
		return false
	}
	if !strings.HasPrefix(value, parts[0]) {
		return false
	}
	rest := value[len(parts[0]):]
	for _, part := range parts[1 : len(parts)-1] {
		i := strings.Index(rest, part)
		if i < 0 {
			return false
		}
		rest = rest[i+len(part):]
	}
	return strings.HasSuffix(rest, parts[len(parts)-1])
}

// All returns every message in the bag, in key order, formatted.
func (b *MessageBag) All(format ...string) []string {
	f := b.checkFormat(format...)
	all := []string{}
	for _, k := range b.order {
		all = append(all, b.transform(b.messages[k], f, k)...)
	}
	return all
}

// Unique returns every message in the bag with duplicates dropped, keeping the
// first occurrence.
func (b *MessageBag) Unique(format ...string) []string {
	return unique(b.All(format...))
}

// Forget drops every message under the key and returns the bag.
func (b *MessageBag) Forget(key string) *MessageBag {
	if _, ok := b.messages[key]; !ok {
		return b
	}
	delete(b.messages, key)
	for i, k := range b.order {
		if k == key {
			b.order = append(b.order[:i], b.order[i+1:]...)
			break
		}
	}
	return b
}

func (b *MessageBag) transform(messages []string, format, messageKey string) []string {
	if format == DefaultMessageFormat {
		out := make([]string, len(messages))
		copy(out, messages)
		return out
	}
	out := make([]string, 0, len(messages))
	for _, message := range messages {
		s := strings.ReplaceAll(format, ":message", message)
		out = append(out, strings.ReplaceAll(s, ":key", messageKey))
	}
	return out
}

func (b *MessageBag) checkFormat(format ...string) string {
	if len(format) > 0 && format[0] != "" {
		return format[0]
	}
	return b.GetFormat()
}

// Messages returns a copy of the messages, keyed by field, unformatted.
func (b *MessageBag) Messages() map[string][]string {
	out := make(map[string][]string, len(b.messages))
	for k, v := range b.messages {
		copied := make([]string, len(v))
		copy(copied, v)
		out[k] = copied
	}
	return out
}

// GetMessages returns a copy of the messages, the same as
// [MessageBag.Messages].
func (b *MessageBag) GetMessages() map[string][]string { return b.Messages() }

// GetMessageBag returns the bag itself, so *MessageBag satisfies
// [MessageProvider].
func (b *MessageBag) GetMessageBag() *MessageBag { return b }

// GetFormat returns the format messages are wrapped in, which is
// [DefaultMessageFormat] when none was set.
func (b *MessageBag) GetFormat() string {
	if b.format == "" {
		return DefaultMessageFormat
	}
	return b.format
}

// SetFormat sets the format messages are wrapped in and returns the bag. An
// empty format means [DefaultMessageFormat].
func (b *MessageBag) SetFormat(format string) *MessageBag {
	if format == "" {
		format = DefaultMessageFormat
	}
	b.format = format
	return b
}

// IsEmpty reports whether the bag holds no message.
func (b *MessageBag) IsEmpty() bool { return !b.Any() }

// IsNotEmpty reports whether the bag holds any message.
func (b *MessageBag) IsNotEmpty() bool { return b.Any() }

// Any reports whether the bag holds any message.
func (b *MessageBag) Any() bool { return b.Count() > 0 }

// Count returns how many messages the bag holds, counting messages and not
// keys.
func (b *MessageBag) Count() int {
	n := 0
	for _, v := range b.messages {
		n += len(v)
	}
	return n
}

// ToArray returns a copy of the messages, keyed by field.
func (b *MessageBag) ToArray() map[string][]string { return b.GetMessages() }

// ToJson encodes the messages as JSON, or returns the error encoding raised.
func (b *MessageBag) ToJson() (string, error) {
	raw, err := json.Marshal(b.jsonValue())
	if err != nil {
		return "", err
	}
	return string(raw), nil
}

// MarshalJSON encodes the messages, so a MessageBag nests inside another
// encoded value. An empty bag encodes as {}.
func (b *MessageBag) MarshalJSON() ([]byte, error) {
	return json.Marshal(b.jsonValue())
}

func (b *MessageBag) jsonValue() map[string][]string {
	out := b.ToArray()
	if out == nil {
		return map[string][]string{}
	}
	return out
}

// String returns the messages as JSON, or "{}" when they cannot be encoded, so
// MessageBag satisfies fmt.Stringer.
func (b *MessageBag) String() string {
	s, err := b.ToJson()
	if err != nil {
		return "{}"
	}
	return s
}
