package str

import (
	"crypto/rand"
	"encoding/hex"
	"regexp"
	"time"
)

// UUID is a random version 4 UUID in the canonical 8-4-4-4-12 hyphenated
// form.
//
// Random, which means unordered: a table with one of these as its primary key
// writes to a random leaf of the index on every insert. UUID7 is the one to
// reach for when the value is a key. This one is for a correlation id, a
// one-time token or anything a person may paste.
//
// CreateUUIDsUsing and FreezeUUIDs stand in front of it, so a test can pin what
// it returns.
//
// There is no error to return. Since Go 1.24 crypto/rand.Read cannot fail; when
// the system source is unavailable the program crashes rather than handing back
// a value that is not random.
func UUID() string {
	if s, ok := fromUUIDFactory(); ok {
		return s
	}
	var b [16]byte
	rand.Read(b[:])
	b[6] = b[6]&0x0f | 0x40 // version 4
	b[8] = b[8]&0x3f | 0x80 // RFC 4122 variant
	return formatUUID(b)
}

// UUID7 is a version 7 UUID: 48 bits of Unix milliseconds followed by 74 bits
// of randomness, in the same hyphenated form.
//
// Sortable by generation time, which is what makes it usable as a primary key
// and as a cursor. It also tells anyone holding it when the row was created, so
// it is not the identifier for something whose age is private.
//
// It reads the clock, and a caller that has to pin the value uses
// CreateUUIDsUsing, which stands in front of this the same way.
func UUID7() string {
	if s, ok := fromUUIDFactory(); ok {
		return s
	}
	var b [16]byte
	rand.Read(b[:])
	ms := time.Now().UnixMilli()
	for i := range 6 {
		b[i] = byte(ms >> (40 - 8*i))
	}
	b[6] = b[6]&0x0f | 0x70 // version 7
	b[8] = b[8]&0x3f | 0x80 // RFC 4122 variant
	return formatUUID(b)
}

// OrderedUUID is a time-ordered UUID: 48 bits of Unix milliseconds in front,
// the rest random, and the version nibble still reading 4.
//
// It exists for the same reason UUID7 does -- an index that is written in order
// -- for a schema whose column is documented as holding a version 4 UUID.
func OrderedUUID() string {
	if s, ok := fromUUIDFactory(); ok {
		return s
	}
	var b [16]byte
	rand.Read(b[:])
	ms := time.Now().UnixMilli()
	for i := range 6 {
		b[i] = byte(ms >> (40 - 8*i))
	}
	b[6] = b[6]&0x0f | 0x40 // version 4
	b[8] = b[8]&0x3f | 0x80 // RFC 4122 variant
	return formatUUID(b)
}

// fromUUIDFactory reads the factory CreateUUIDsUsing set, if there is one.
func fromUUIDFactory() (string, bool) {
	factoryMu.RLock()
	factory := uuidFactory
	factoryMu.RUnlock()
	if factory == nil {
		return "", false
	}
	return factory(), true
}

// ULID is a 26-character Crockford base32 identifier: 48 bits of Unix
// milliseconds followed by 80 bits of randomness.
//
// Sortable like UUID7 and shorter, with an alphabet that has no I, L, O or U in
// it, so it survives being read aloud and cannot spell a word. It is what to
// use where the value shows up in a URL.
//
// CreateULIDsUsing and FreezeULIDs stand in front of it. It reads the clock,
// and a caller that has to pin the value uses those.
func ULID() string {
	factoryMu.RLock()
	factory := ulidFactory
	factoryMu.RUnlock()
	if factory != nil {
		return factory()
	}

	var rnd [10]byte
	rand.Read(rnd[:])

	out := make([]byte, 26)
	ms := uint64(time.Now().UnixMilli())
	for i := 9; i >= 0; i-- {
		out[i] = crockford[ms&31]
		ms >>= 5
	}
	// 10 bytes are 80 bits, and 80 bits are exactly 16 base32 characters, so
	// this consumes the buffer with nothing left over and no padding.
	acc, bits, pos := uint64(0), 0, 10
	for _, by := range rnd {
		acc = acc<<8 | uint64(by)
		bits += 8
		for bits >= 5 {
			bits -= 5
			out[pos] = crockford[(acc>>bits)&31]
			pos++
		}
	}
	return string(out)
}

// Random is n random alphanumeric characters, from the system source.
//
// Use it for anything that has to be unguessable -- a token, a nonce, a
// password a machine invented. It is not a hash and not an identifier; UUID7 or
// ULID is what names a row.
//
// CreateRandomStringsUsing stands in front of it, so a test can pin what it
// returns.
func Random(n int) string {
	factoryMu.RLock()
	factory := randomFactory
	factoryMu.RUnlock()
	if factory != nil {
		return factory(n)
	}
	if n <= 0 {
		return ""
	}
	out := make([]byte, 0, n)
	// A byte masked to six bits lands outside the 62-character alphabet about
	// one time in sixteen, and a rejected byte is drawn again rather than folded
	// back in with a modulo -- the modulo is what makes the first two characters
	// of the alphabet twice as likely as the rest.
	buf := make([]byte, n+n/4+8)
	for len(out) < n {
		rand.Read(buf)
		for _, b := range buf {
			v := int(b & 63)
			if v >= len(alphanumeric) {
				continue
			}
			out = append(out, alphanumeric[v])
			if len(out) == n {
				break
			}
		}
	}
	return string(out)
}

// IsUUID reports whether s has the shape of a hyphenated UUID. Shape only: it
// says nothing about the version or the variant, because a caller checking an
// identifier off a request wants to know whether it can be looked up, not who
// made it.
func IsUUID(s string) bool { return uuidShape.MatchString(s) }

// IsULID reports whether s has the shape of a ULID.
func IsULID(s string) bool { return ulidShape.MatchString(s) }

const (
	// crockford is base32 without I, L, O and U -- the four that are read as
	// something else.
	crockford = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

	alphanumeric = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)

var (
	uuidShape = regexp.MustCompile(`^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$`)

	// ulidShape is Crockford base32 without I, L, O and U, and a first character
	// no greater than 7 so the 48-bit timestamp cannot overflow.
	ulidShape = regexp.MustCompile(`^[0-7][0-9A-HJKMNP-TV-Za-hjkmnp-tv-z]{25}$`)
)

// formatUUID writes the sixteen bytes in the 8-4-4-4-12 hyphenated form.
func formatUUID(b [16]byte) string {
	var out [36]byte
	hex.Encode(out[0:8], b[0:4])
	out[8] = '-'
	hex.Encode(out[9:13], b[4:6])
	out[13] = '-'
	hex.Encode(out[14:18], b[6:8])
	out[18] = '-'
	hex.Encode(out[19:23], b[8:10])
	out[23] = '-'
	hex.Encode(out[24:36], b[10:16])
	return string(out[:])
}
