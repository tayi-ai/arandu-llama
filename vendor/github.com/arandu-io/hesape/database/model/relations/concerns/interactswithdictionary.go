package concerns

import (
	"fmt"
	"strconv"
)

// GetDictionaryKey answers InteractsWithDictionary::getDictionaryKey.
//
// A trait with one protected method is a function here: there is no state to
// carry and nothing to override, so a struct to embed would be ceremony around
// a call.
//
// The PHP returns the attribute unchanged for scalars and casts objects through
// __toString or the enum's value. This returns a string in every case, and that
// is not tidiness -- it is the behaviour PHP gets for free and Go does not. An
// array key in PHP coerces: $dictionary[1] and $dictionary["1"] are the same
// bucket. A Go map keyed by any does not coerce, so a parent whose key the
// driver returned as int64(1) would miss a child whose foreign key came back as
// "1", and the relation would come out empty with every value on screen looking
// right. Rendering both to "1" is what keeps the dictionary matching the way
// the PHP one matches.
//
// It carries an error where the PHP throws InvalidArgumentException, for the
// same case: a value that is neither scalar nor stringable cannot key anything.
func GetDictionaryKey(attribute any) (string, error) {
	switch value := attribute.(type) {
	case nil:
		return "", nil
	case string:
		return value, nil
	case []byte:
		return string(value), nil
	case fmt.Stringer:
		return value.String(), nil
	case bool:
		return strconv.FormatBool(value), nil
	case int:
		return strconv.FormatInt(int64(value), 10), nil
	case int8:
		return strconv.FormatInt(int64(value), 10), nil
	case int16:
		return strconv.FormatInt(int64(value), 10), nil
	case int32:
		return strconv.FormatInt(int64(value), 10), nil
	case int64:
		return strconv.FormatInt(value, 10), nil
	case uint:
		return strconv.FormatUint(uint64(value), 10), nil
	case uint8:
		return strconv.FormatUint(uint64(value), 10), nil
	case uint16:
		return strconv.FormatUint(uint64(value), 10), nil
	case uint32:
		return strconv.FormatUint(uint64(value), 10), nil
	case uint64:
		return strconv.FormatUint(value, 10), nil
	case float32:
		return strconv.FormatFloat(float64(value), 'f', -1, 32), nil
	case float64:
		return strconv.FormatFloat(value, 'f', -1, 64), nil
	default:
		return "", fmt.Errorf("relations: a model attribute of type %T cannot key a relation dictionary: it is not a scalar and does not answer String()", attribute)
	}
}
