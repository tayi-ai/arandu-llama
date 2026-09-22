package pagination

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/arandu-io/hesape/encryption"
)

// pointsToNextItemsKey is the key the direction travels under inside an encoded
// cursor. It is fixed byte for byte, because the token travels in URLs and in
// API payloads and has to read the same on both ends.
const pointsToNextItemsKey = "_pointsToNextItems"

// cursorPurpose is what a cursor token is signed for. A token issued to page a
// list does not work as a password reset, and the reverse, because the purpose
// is part of the signature.
const cursorPurpose = "pagination.cursor"

// DefaultCursorTTL is how long a token stays valid when [NewCursorSigner] is
// given no lifetime of its own.
//
// It is a day and not forever because the token is a bearer of a position in a
// result set: it lands in browser history, in proxy logs and in links people
// share, and every one of those outlives the page it was made for. It is a day
// and not an hour because a reader who leaves a list open over lunch should
// still be able to press "next".
const DefaultCursorTTL = 24 * time.Hour

// ErrCursor is what [CursorSigner.FromEncoded] returns for a cursor it cannot
// read: a string that is not a token, a token this application did not write,
// one that has run out, or one whose payload is not a JSON object.
//
// A failed check wraps the error the signature returned, so a caller that wants
// to tell "it has expired" from "it was forged" can still ask.
//
// Callers that page a public list should treat it as "start from the
// beginning", the way ResolveCurrentCursor does: a mangled cursor is a
// truncated link in an e-mail client, not an attack worth a 400.
var ErrCursor = errors.New("pagination: malformed cursor")

// Cursor is a position in an ordered result set: the values of the columns the
// query orders by for the row at one edge of a page, plus the direction the next
// query walks in.
//
// It is what keyset pagination replaces OFFSET with. OFFSET makes the database
// count and discard every row it skips, so page 500 costs five hundred pages of
// work, and it reads the page boundary at query time, so a row inserted while
// somebody reads page 3 pushes a row from page 3 onto page 4 -- a row that was
// never shown, and never will be. A cursor names the boundary by value, so the
// scan starts at the index entry and the pages hold still.
//
// The parameters are strings because they land in a URL and come back from one.
// A repository formats its ordering columns into strings on the way out and
// parses them on the way back in, and formatting is where it decides that a
// timestamp keeps its subsecond digits -- a cursor rounded to the second walks
// past every row that shares it.
//
// The fields are unexported because they are read through Parameter,
// Parameters, PointsToNextItems and PointsToPreviousItems, and exporting them
// too would offer two ways to ask the same question.
type Cursor struct {
	parameters        map[string]string
	pointsToNextItems bool
}

// NewCursor builds a cursor over the ordering columns of a boundary row.
//
// parameters maps the name of each ordering column to the value it has in the
// boundary row. The set must match the ORDER BY exactly: a cursor over
// (created_at, id) read by a query ordered by (email, id) computes the boundary
// on one ordering and returns the rows in another, which skips rows and repeats
// rows in silence.
//
// pointsToNextItems reports which side of the boundary the next query reads.
// True is forward, the ordinary "next page". False is backward, and a backward
// query returns its rows in reverse order; CursorPaginate turns them around
// again.
//
// The map is copied, so a cursor cannot be changed through the map its caller
// kept.
func NewCursor(parameters map[string]string, pointsToNextItems bool) Cursor {
	copied := make(map[string]string, len(parameters))
	for name, value := range parameters {
		copied[name] = value
	}
	return Cursor{parameters: copied, pointsToNextItems: pointsToNextItems}
}

// Parameter returns the value the boundary row has in the named ordering
// column.
//
// A name the cursor does not carry is an error. It is not an empty string,
// because an empty string
// is a legitimate value for a nullable column and telling the two apart is the
// difference between reading the next page and reading the first one again.
func (c Cursor) Parameter(parameterName string) (string, error) {
	value, ok := c.parameters[parameterName]
	if !ok {
		return "", fmt.Errorf("pagination: unable to find parameter [%s] in pagination item", parameterName)
	}
	return value, nil
}

// Parameters returns the values of the named ordering columns, in the order
// asked for, and fails on the first name the cursor does not carry.
func (c Cursor) Parameters(parameterNames []string) ([]string, error) {
	values := make([]string, 0, len(parameterNames))
	for _, name := range parameterNames {
		value, err := c.Parameter(name)
		if err != nil {
			return nil, err
		}
		values = append(values, value)
	}
	return values, nil
}

// PointsToNextItems reports whether the query reading from this cursor walks
// forward.
func (c Cursor) PointsToNextItems() bool { return c.pointsToNextItems }

// PointsToPreviousItems reports whether the query reading from this cursor
// walks backward, and so returns its rows in reverse order.
func (c Cursor) PointsToPreviousItems() bool { return !c.pointsToNextItems }

// ToArray is the parameters with the direction merged in under
// _pointsToNextItems, which is the shape [CursorSigner.Encode] signs.
//
// An ordering column actually named _pointsToNextItems would collide with the
// direction. It is not defended against here: doing so would change the
// encoding, and a token no other reader of the same format can parse is a worse
// fault than a column name nobody uses.
func (c Cursor) ToArray() map[string]any {
	out := make(map[string]any, len(c.parameters)+1)
	for name, value := range c.parameters {
		out[name] = value
	}
	out[pointsToNextItemsKey] = c.pointsToNextItems
	return out
}

// CursorSigner turns a Cursor into the token it travels as, and reads one back.
//
// It holds the application key because a cursor comes back from the client. The
// token names the boundary row of a page -- the value of every column the query
// orders by -- and a client that can rewrite it moves that boundary anywhere the
// encoding allows: onto a row that was never shown, or onto a column name the
// repository then has to defend against. Base64 of JSON stops neither, because
// base64 is transport and not a signature.
//
// What the signature buys is that the token is unchanged, not that it is
// secret. The parameters stay readable to anyone holding one, exactly as they
// were; what changes is that only this application can write one.
//
// It is the only way a Cursor becomes a string, which is why Cursor itself has
// no Encode: a second, keyless one would be the one that gets called.
type CursorSigner struct {
	signer *encryption.Signer
	ttl    time.Duration
}

// NewCursorSigner returns a signer whose tokens are valid for ttl.
//
// A ttl of zero or less means DefaultCursorTTL.
//
// A nil signer panics rather than yielding something that writes unsigned
// tokens: there is no cursor worth issuing that nobody checked.
func NewCursorSigner(s *encryption.Signer, ttl time.Duration) *CursorSigner {
	if s == nil {
		panic("pagination: NewCursorSigner needs a signer over the application key")
	}
	if ttl <= 0 {
		ttl = DefaultCursorTTL
	}
	return &CursorSigner{signer: s, ttl: ttl}
}

// Encode renders the cursor as one URL-safe token: the JSON of ToArray, signed
// with the application key and stamped with the expiry the signer was built
// with. Every character of the result is unreserved in a URL, so it needs no
// escaping in a query string or in an e-mail.
//
// The token is not stable over time. The expiry is part of what is signed, so
// the same page yields a different token as the clock moves; it is not a cache
// key, and two of them are not worth comparing.
//
// A cursor holding a name or a value that is not valid UTF-8 has no token, and
// the empty string is what says so. JSON strings are UTF-8, and encoding one
// replaces every byte that is not with U+FFFD rather than failing -- so the
// alternative to refusing is a token that decodes to a different boundary than
// the one it was made from, and a query reading from it walks past rows and
// repeats rows in silence. A page whose next link is missing is a bug someone
// reports; a page of the wrong rows is not. A boundary value that is bytes
// rather than text -- a binary key, a column read out as a blob -- is formatted
// into text by the same key function that names it, the way a timestamp is.
func (cs *CursorSigner) Encode(c Cursor) string {
	fields := c.ToArray()
	for name, value := range fields {
		text, ok := value.(string)
		if !utf8.ValidString(name) || (ok && !utf8.ValidString(text)) {
			return ""
		}
	}
	// A map of strings and a bool always marshal, so the error is unreachable
	// rather than ignored.
	payload, err := json.Marshal(fields)
	if err != nil {
		return ""
	}
	return cs.signer.Sign(cursorPurpose, string(payload), cs.ttl)
}

// FromEncoded reads a cursor back out of the token Encode wrote, and refuses
// one this application did not write.
//
// Padding is tolerated, so a cursor that travelled through something that pads
// base64 still parses. Anything else -- empty, not a token, signed with another
// key, run out, not a JSON object -- is ErrCursor.
//
// A token carrying no signature is one of those, and there is no window in
// which it is accepted: stripping the signature is exactly what forging one
// produces, so accepting an unsigned token accepts every forged token with it.
// Read through ResolveCurrentCursor it reads as the first page, which is what a
// reader following a link older than the key sees.
//
// Numbers keep the digits they were written with rather than going through a
// float, so a cursor over a 64-bit key survives the round trip. A missing
// direction reads as backward.
func (cs *CursorSigner) FromEncoded(encodedString string) (Cursor, error) {
	if encodedString == "" {
		return Cursor{}, fmt.Errorf("%w: empty", ErrCursor)
	}
	payload, err := cs.signer.Verify(cursorPurpose, strings.TrimRight(encodedString, "="))
	if err != nil {
		return Cursor{}, fmt.Errorf("%w: %w", ErrCursor, err)
	}

	decoder := json.NewDecoder(strings.NewReader(payload))
	decoder.UseNumber()
	var wire map[string]any
	if err := decoder.Decode(&wire); err != nil {
		return Cursor{}, fmt.Errorf("%w: %v", ErrCursor, err)
	}
	if wire == nil {
		return Cursor{}, fmt.Errorf("%w: null", ErrCursor)
	}

	cursor := Cursor{parameters: make(map[string]string, len(wire))}
	for name, value := range wire {
		if name == pointsToNextItemsKey {
			cursor.pointsToNextItems, _ = value.(bool)
			continue
		}
		text, ok := cursorParameterString(value)
		if !ok {
			return Cursor{}, fmt.Errorf("%w: parameter [%s] is not a scalar", ErrCursor, name)
		}
		cursor.parameters[name] = text
	}
	return cursor, nil
}

// cursorParameterString renders one decoded JSON scalar as the string the
// parameters map holds. A JSON null becomes the empty string, matching what a
// nullable column formats to on the way out.
func cursorParameterString(value any) (string, bool) {
	switch v := value.(type) {
	case nil:
		return "", true
	case string:
		return v, true
	case json.Number:
		return v.String(), true
	case bool:
		return strconv.FormatBool(v), true
	default:
		return "", false
	}
}

// ResolveCurrentCursor reads the cursor out of the URL of the request being
// served, checking its signature before reading it.
//
// The URL is passed in rather than reached for. A cursor that is absent, does
// not parse, or was not signed by this application is nil, which every
// constructor reads as "the first page". A reader whose link has run out gets
// the top of the list rather than an error page, and a client that rewrote one
// gets the same thing it would have got by sending nothing.
//
// An empty cursorName means DefaultCursorName. A nil signer panics: there is no
// reading of a cursor that skips the check.
func ResolveCurrentCursor(cs *CursorSigner, u *url.URL, cursorName string) *Cursor {
	if cs == nil {
		panic("pagination: ResolveCurrentCursor needs the signer the cursor was written with")
	}
	if cursorName == "" {
		cursorName = DefaultCursorName
	}
	if u == nil {
		return nil
	}
	raw := u.Query().Get(cursorName)
	if raw == "" {
		return nil
	}
	cursor, err := cs.FromEncoded(raw)
	if err != nil {
		return nil
	}
	return &cursor
}
