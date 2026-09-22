package str

import "crypto/rand"

// The four character sets Password draws from.
const (
	passwordLetters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
	passwordNumbers = "0123456789"
	passwordSymbols = `~!#$%^&*()-_.,<>?/\{}[]|:;`
	passwordSpaces  = " "
)

// Password is a random password of the given length, drawn from the sets that
// are switched on.
//
//	Password(32, true, true, true, false)
//
// All five arguments are required, because Go has no default arguments.
//
// Every set that is on contributes one character up front, so a password that
// asks for symbols has a symbol in it. Those characters count against the
// length and the rest is drawn from the sets together, then the whole thing is
// shuffled -- which is why a length shorter than the number of sets still comes
// back with one character from each and is longer than asked for.
//
// With every set switched off there is nothing to draw from and the answer is
// empty.
func Password(length int, letters, numbers, symbols, spaces bool) string {
	var alphabet string
	var password []byte
	for _, set := range []struct {
		on    bool
		chars string
	}{
		{letters, passwordLetters},
		{numbers, passwordNumbers},
		{symbols, passwordSymbols},
		{spaces, passwordSpaces},
	} {
		if !set.on {
			continue
		}
		alphabet += set.chars
		password = append(password, set.chars[randomIndex(len(set.chars))])
	}
	if alphabet == "" {
		return ""
	}
	for range length - len(password) {
		password = append(password, alphabet[randomIndex(len(alphabet))])
	}
	shuffle(password)
	return string(password)
}

// randomIndex is a uniform index below n, from the system source, with the
// modulo bias rejected rather than folded in.
func randomIndex(n int) int {
	if n <= 1 {
		return 0
	}
	limit := ^uint32(0) - ^uint32(0)%uint32(n) - 1
	var b [4]byte
	for {
		rand.Read(b[:])
		v := uint32(b[0])<<24 | uint32(b[1])<<16 | uint32(b[2])<<8 | uint32(b[3])
		if v <= limit {
			return int(v % uint32(n))
		}
	}
}

// shuffle is a Fisher-Yates shuffle, drawing every swap from the system
// source.
func shuffle(b []byte) {
	for i := len(b) - 1; i > 0; i-- {
		j := randomIndex(i + 1)
		b[i], b[j] = b[j], b[i]
	}
}
