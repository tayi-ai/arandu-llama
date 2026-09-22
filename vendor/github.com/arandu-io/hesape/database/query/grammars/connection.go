package grammars

import (
	"strconv"
	"strings"
)

// A handful of grammar decisions are not decisions about SQL: they are questions
// about the server on the other end. MySQL emits a different group limit before
// 8.0.11 and SQLite before 3.25.0, because window functions did not exist yet,
// and MySQL's upsert spells the excluded row differently depending on a
// connection option.
//
// The connection a builder holds here is query.connection, which is
// narrowed to running statements, so these questions are asked through
// optional interfaces instead: a connection that can answer implements one,
// and a connection that cannot gets the modern answer, which is the one
// every supported server gives.

// ServerVersionConnection is implemented by a connection that can report
// its server version string.
type ServerVersionConnection interface {
	// GetServerVersion returns the version string the server reports.
	GetServerVersion() string
}

// ConfigConnection is implemented by a connection that can report one
// option of its configuration.
type ConfigConnection interface {
	// GetConfig returns one option of the connection's configuration.
	GetConfig(option string) any
}

// serverVersion asks the connection its version, and returns false when there
// is nobody to ask.
func serverVersion(connection any) (string, bool) {
	if versioned, ok := connection.(ServerVersionConnection); ok && versioned != nil {
		version := versioned.GetServerVersion()
		return version, version != ""
	}
	return "", false
}

// config asks the connection for an option, and returns nil when there is
// nobody to ask.
func config(connection any, option string) any {
	if configured, ok := connection.(ConfigConnection); ok && configured != nil {
		return configured.GetConfig(option)
	}
	return nil
}

// configBool reads an option as a flag: a boolean, a non-empty and
// non-"0"/"false" string, or a non-zero int.
func configBool(connection any, option string) bool {
	switch value := config(connection, option).(type) {
	case bool:
		return value
	case string:
		return value != "" && value != "0" && !strings.EqualFold(value, "false")
	case int:
		return value != 0
	default:
		return false
	}
}

// versionLess reports whether version sorts before other, comparing only
// their leading numeric parts.
//
// Only the leading numeric parts are compared, so "8.0.11-log" and
// "3.45.0" are read the way the servers mean them. A version that cannot be
// read at all compares as not less, which keeps an unrecognisable server on
// the modern statement rather than on the compatibility one.
func versionLess(version, other string) bool {
	left, right := versionParts(version), versionParts(other)

	for i := 0; i < len(left) || i < len(right); i++ {
		var a, b int
		if i < len(left) {
			a = left[i]
		}
		if i < len(right) {
			b = right[i]
		}
		if a != b {
			return a < b
		}
	}

	return false
}

func versionParts(version string) []int {
	fields := strings.FieldsFunc(version, func(r rune) bool {
		return r == '.' || r == '-' || r == '+' || r == ' '
	})

	out := make([]int, 0, len(fields))
	for _, field := range fields {
		number, err := strconv.Atoi(field)
		if err != nil {
			break
		}
		out = append(out, number)
	}

	return out
}
