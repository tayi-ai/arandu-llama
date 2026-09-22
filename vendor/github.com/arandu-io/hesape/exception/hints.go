package exception

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"
)

const (
	nPlusOneThreshold = 5
	slowQuery         = 200 * time.Millisecond
)

// hints is the part that copies the best idea in Ignition: do not just show the
// error, suggest the likely cause. Every pattern recognized here saves an hour
// of support later, and the list grows with each bug we hit ourselves.
func hints(d viewData) []string {
	var out []string

	if len(d.NPlusOne) > 0 {
		out = append(out, fmt.Sprintf("Likely N+1: the same statement ran %d or more times in this request. Load it in one batch inside the repository.", nPlusOneThreshold))
	}
	for _, q := range d.Queries {
		if q.Duration >= slowQuery {
			out = append(out, "A query took "+q.Duration.String()+". Check the index for: "+truncate(q.SQL, 60))
			break
		}
	}

	// A failed query is usually the panic itself: the repository returns the
	// driver error, the action returns it, and the handler panics with it.
	// Printing it again under Diagnosis said the same sentence twice, once as
	// the headline and once as a finding, which reads as two problems. Say it
	// only when the headline is about something else -- a query that failed, was
	// handled, and caused a later panic is exactly the case worth naming.
	for _, q := range d.Queries {
		if q.Err != nil && !strings.Contains(d.Message, q.Err.Error()) {
			out = append(out, "A query failed before this panic: "+q.Err.Error())
			break
		}
	}

	if h, ok := messageHint(d.Message); ok {
		out = append(out, h)
	}
	return out
}

// messageHint reads the failure message and returns at most one sentence.
//
// At most one, and narrowest first. These are ordered by how much of the message
// they require, because a broad substring above a narrow one answers the narrow
// failure with the wrong fix.
//
// That is not hypothetical. The branch that used to be here matched "CSRF", so
// it also matched "CSRFToken" -- and the one message reaching this page that
// contains that word is the view engine saying "@csrf needs the page data to
// provide the token. Add a CSRFToken() string method". Somebody whose page
// struct was missing a method was told to go edit hx-headers on their base
// template. It fired on any panic naming the CSRF type too, and the token
// mismatch itself never arrives here: it is classified as 419 and answered with
// a status page, so the branch had no correct trigger at all.
//
// A pattern earns its place by three tests: it names the failure in the words of
// the reader's own project, it names the one command or line that fixes it, and
// it cannot fire on a different problem. The third is the expensive one -- a
// hint that guesses wrong costs more than no hint, because the reader spends the
// next hour in the file it named.
func messageHint(msg string) (string, bool) {
	// A service was called with the zero auth.Subject, because the session was
	// never loaded or the store's error was dropped. Authorize refuses before it
	// consults a policy, so the policy for this action never ran -- and the
	// developer opens it, edits it, and nothing they change has any effect.
	//
	// The anchor carries "not authorized: " on purpose. Authorize's other branch
	// interpolates the application's own denial text verbatim, so a policy that
	// wrote "the anonymous subject on this blog may not comment" would trip a
	// bare match -- and there the policy did run, which is the opposite of what
	// this says.
	if strings.Contains(msg, "not authorized: anonymous subject on") {
		return "No policy ran. auth.Authorize refuses a Subject with no ID before it consults one, so the policy for this action was never called and editing it changes nothing. Give the action a subject: auth.Guest(tenant) when the page is meant to be readable without an account, or the Subject from the session with its error handled.", true
	}

	if strings.Contains(msg, "missing grant") {
		return "You reached a repository without going through auth.Authorize. Authorize the action first: the Grant is what proves a policy ran.", true
	}

	if strings.Contains(msg, "grant issued for") {
		return "The Grant was issued for one action and used on another. Check the action constant in the repository method.", true
	}

	// The statement names a relation the database cannot resolve: most often a
	// project whose migrations have never run against this database, which is
	// the highest-value case here and the one this page had nothing to say
	// about.
	//
	// It states what is true and offers the check rather than asserting the
	// cause, because the same text has more than one cause and none of them is
	// separable from the string: a missing sequence, a misspelled CTE, a trigger
	// body naming a missing table and a table in another schema all render
	// identically.
	if name, ok := relationNotFound(msg); ok {
		return "Nothing in this database resolves the name " + strconv.Quote(name) + ". If the migrations have never run against it, `aru migrate` creates it. If `aru migrate` answers \"nothing to migrate\", no migration in this project declares that name, and the statement above is naming something else: a CTE, a sequence, or a table in another schema.", true
	}

	return "", false
}

// relationNotFound reports the relation a driver said does not exist, for the
// three engines, and false when the message is about something else.
//
// The anchors are long deliberately. PostgreSQL's SQLSTATE 42P01 is the whole
// undefined_table class, and an alias typo lives in it too: `SELECT p.id FROM
// posts` answers `missing FROM-clause entry for table "p" (SQLSTATE 42P01)`,
// where `aru migrate` is not the fix and the name captured would be a query
// alias. A framework whose data path is hand-written SQL sees that far more
// often than a missing migration.
func relationNotFound(msg string) (string, bool) {
	for _, re := range relationPatterns {
		if m := re.FindStringSubmatch(msg); m != nil {
			return m[1], true
		}
	}
	return "", false
}

var relationPatterns = []*regexp.Regexp{
	// PostgreSQL: relation "posts" does not exist (SQLSTATE 42P01)
	regexp.MustCompile(`relation "([^"]+)" does not exist`),
	// MySQL 1146. 1051 ("Unknown table", from DROP) is deliberately excluded:
	// it is a different failure with a different fix.
	regexp.MustCompile(`Error 1146 \(42S02\): Table '(?:[^.']*\.)?([^']+)' doesn't exist`),
	// SQLite
	regexp.MustCompile(`no such table: ([A-Za-z_][A-Za-z0-9_.]*)`),
}

func truncate(s string, n int) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
