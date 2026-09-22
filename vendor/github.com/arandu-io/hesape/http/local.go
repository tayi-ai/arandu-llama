package http

// maxAddress is the longest address worth carrying.
//
// A browser drops a cookie over about four kilobytes, silently, and the signed
// form of an address is a third longer than the address plus the signature. Past
// this the cookie would be discarded by the browser rather than by us, which is
// the same outcome arrived at without a decision.
const maxAddress = 1024

// LocalPath reports whether an address stays inside this application, and
// returns it when it does.
//
// It is the open-redirect defence, and it is one function in the collection
// rather than one per caller: [Reject] calls it on the address a rejected form
// is sent back to, which comes off the Referer header and is therefore the
// visitor's to choose, and [Intended] calls it twice around a signed cookie. Two
// copies of this would be two lists of refused shapes, and the second list is
// always the shorter one.
//
// A destination is accepted only as an absolute path on this origin. That is
// narrow on purpose: "is this URL one of ours" answered by comparing hosts is a
// question with a decade of published bypasses in it, and the only answer that
// has none is not to accept a host at all.
//
// The four shapes it refuses, and what each one does:
//
//   - anything not starting with "/": "https://evil.example/login" and
//     "javascript:..." are both Location headers a browser will act on.
//   - "//evil.example/x": a protocol-relative URL. It arrives here looking like
//     a path -- net/http parses a request target beginning with "//" into
//     URL.Path, host and all -- and a browser resolves it to another origin.
//   - "/\evil.example/x": browsers normalise a backslash to a slash in the
//     authority position, so this is the line above with one character changed,
//     and it is the form that gets past a check written as "starts with a single
//     slash".
//   - a control character or a space anywhere in it: header injection, and the
//     browsers that strip such characters before resolving, which turns a
//     rejected address into an accepted one.
func LocalPath(raw string) (string, bool) {
	if len(raw) == 0 || len(raw) > maxAddress {
		return "", false
	}
	if raw[0] != '/' {
		return "", false
	}
	if len(raw) > 1 && (raw[1] == '/' || raw[1] == '\\') {
		return "", false
	}
	for i := 0; i < len(raw); i++ {
		if raw[i] <= ' ' || raw[i] == 0x7f {
			return "", false
		}
	}
	return raw, true
}
