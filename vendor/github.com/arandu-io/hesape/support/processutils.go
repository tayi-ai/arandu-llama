package support

import "strings"

// processUtilsFacade carries the process helpers, reached through the
// [ProcessUtils] value rather than constructed.
type processUtilsFacade struct{}

// ProcessUtils holds the helpers for handing arguments to a shell.
var ProcessUtils processUtilsFacade

// EscapeArgument returns the string made safe to hand a shell as one argument.
//
// The quoting rules differ by platform, so this branches on [Windows_os].
// Everywhere but Windows the argument is single quoted and every single quote
// inside it is closed, escaped and reopened, which no shell can read as the
// end of the argument. On Windows the argument is double quoted, an embedded
// quote is escaped, a trailing backslash is doubled so it cannot escape the
// closing quote, and a run wrapped in percent signs is broken up so the shell
// does not expand it as an environment variable.
func (processUtilsFacade) EscapeArgument(argument string) string {
	if !Windows_os() {
		return "'" + strings.ReplaceAll(argument, "'", `'\''`) + "'"
	}
	if argument == "" {
		return `""`
	}

	escaped := strings.Builder{}
	quote := false
	for _, part := range splitKeepingQuotes(argument) {
		switch {
		case part == `"`:
			escaped.WriteString(`\"`)
		case isSurroundedBy(part, '%'):
			// Avoid environment variable expansion.
			escaped.WriteString(`^%"` + part[1:len(part)-1] + `"^%`)
		default:
			// Escape a trailing backslash, which would otherwise escape the
			// closing quote.
			if strings.HasSuffix(part, `\`) {
				part += `\`
			}
			quote = true
			escaped.WriteString(part)
		}
	}
	if quote {
		return `"` + escaped.String() + `"`
	}
	return escaped.String()
}

// splitKeepingQuotes returns the runs between double quotes, with the quotes
// themselves kept as parts of their own and nothing empty.
func splitKeepingQuotes(argument string) []string {
	parts := []string{}
	current := strings.Builder{}
	for i := 0; i < len(argument); i++ {
		if argument[i] != '"' {
			current.WriteByte(argument[i])
			continue
		}
		if current.Len() > 0 {
			parts = append(parts, current.String())
			current.Reset()
		}
		parts = append(parts, `"`)
	}
	if current.Len() > 0 {
		parts = append(parts, current.String())
	}
	return parts
}

// isSurroundedBy reports whether the argument opens and closes with char and
// has something between the two.
func isSurroundedBy(argument string, char byte) bool {
	return len(argument) > 2 && argument[0] == char && argument[len(argument)-1] == char
}
