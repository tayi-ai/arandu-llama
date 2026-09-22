package database

import (
	"fmt"
	"strings"

	"github.com/arandu-io/hesape/database/query"
)

// Reading a value off a normalised record.
//
// A driver answers what it answers -- an int64 here, a []byte there, a string
// somewhere else -- and the processors normalise the column NAMES rather than
// the value types. These four read a value of the shape the schema struct
// declares and answer the zero value rather than panicking, because a column a
// driver did not report is an ordinary absence and not a failure.

func text(row query.Record, key string) string {
	switch v := row[key].(type) {
	case string:
		return v
	case []byte:
		return string(v)
	case nil:
		return ""
	default:
		return fmt.Sprint(v)
	}
}

func number(row query.Record, key string) int64 {
	switch v := row[key].(type) {
	case int64:
		return v
	case int:
		return int64(v)
	case int32:
		return int64(v)
	case float64:
		return int64(v)
	default:
		return 0
	}
}

func flag(row query.Record, key string) bool {
	switch v := row[key].(type) {
	case bool:
		return v
	case int64:
		return v != 0
	case int:
		return v != 0
	case string:
		return v == "1" || strings.EqualFold(v, "yes") || strings.EqualFold(v, "true") ||
			strings.EqualFold(v, "on") || strings.EqualFold(v, "t")
	case []byte:
		return flag(query.Record{key: string(v)}, key)
	default:
		return false
	}
}

// list reads a column list, which a driver reports either already split or as
// one comma-separated string.
func list(row query.Record, key string) []string {
	switch v := row[key].(type) {
	case []string:
		return v
	case []any:
		out := make([]string, 0, len(v))
		for _, one := range v {
			out = append(out, fmt.Sprint(one))
		}
		return out
	case string:
		if v == "" {
			return nil
		}
		out := strings.Split(v, ",")
		for i := range out {
			out[i] = strings.TrimSpace(out[i])
		}
		return out
	case []byte:
		return list(query.Record{key: string(v)}, key)
	default:
		return nil
	}
}
