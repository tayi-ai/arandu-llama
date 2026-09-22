package collections

import "encoding/json"

// FromJSON builds a collection by decoding a JSON array.
//
// Malformed input returns the decoding error, and a JSON null builds an empty
// collection rather than a nil one. A Collection[T] is a list, so a JSON
// object is a decoding error here; decode it into a map with encoding/json
// directly.
func FromJSON[T any](text string) (Collection[T], error) {
	var items []T
	if err := json.Unmarshal([]byte(text), &items); err != nil {
		return nil, err
	}
	if items == nil {
		return Collection[T]{}, nil
	}
	return Collection[T](items), nil
}

// ToArray returns the elements as a []any, with each element that knows how to
// become an array turned into one.
//
// An element with a ToArray method returning []any or map[string]any is
// converted through it; anything else is passed through as it is.
//
// It differs from All: All hands back the elements with their own type,
// ToArray hands back []any with the convertible ones already converted.
func (c Collection[T]) ToArray() []any {
	out := make([]any, len(c))
	for i, item := range c {
		switch value := any(item).(type) {
		case interface{ ToArray() []any }:
			out[i] = value.ToArray()
		case interface{ ToArray() map[string]any }:
			out[i] = value.ToArray()
		default:
			out[i] = item
		}
	}
	return out
}

// ToJSON encodes the elements as a JSON array, and returns the marshalling
// error when an element cannot be encoded. For indented output use
// ToPrettyJSON.
func (c Collection[T]) ToJSON() (string, error) {
	encoded, err := json.Marshal(c.All())
	if err != nil {
		return "", err
	}
	return string(encoded), nil
}

// ToPrettyJSON is ToJSON with the output indented by four spaces.
func (c Collection[T]) ToPrettyJSON() (string, error) {
	encoded, err := json.MarshalIndent(c.All(), "", "    ")
	if err != nil {
		return "", err
	}
	return string(encoded), nil
}

// Head returns the first element of a slice. The second result is false when
// the slice is empty.
//
// Over a collection, First with a nil callback is the same thing.
func Head[T any](array []T) (T, bool) {
	if len(array) == 0 {
		var zero T
		return zero, false
	}
	return array[0], true
}
