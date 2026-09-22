package session

import (
	"errors"
	"fmt"
	"net/http"
	"strings"
)

// ErrUnsafeConfig is what every refusal from [Config.Check] wraps, so a caller
// can tell an unsafe cookie configuration from a missing driver with errors.Is
// instead of matching on the message.
var ErrUnsafeConfig = errors.New("session: unsafe cookie configuration")

// ConfigError is the refusal [Config.Check] returns: which field is unsafe, and
// what it has to be instead.
//
// The field is a separate string rather than a sentence, because the caller that
// prints this is usually a boot sequence with one line to spend, and "SameSite"
// is the part a person greps their configuration for.
type ConfigError struct {
	// Field is the name of the [Config] field that is unsafe.
	Field string

	// Want says what the field has to be, and why.
	Want string
}

// Error renders the refusal, naming the field first.
func (e *ConfigError) Error() string {
	return fmt.Sprintf("%s: %s %s", ErrUnsafeConfig.Error(), e.Field, e.Want)
}

// Unwrap returns [ErrUnsafeConfig].
func (e *ConfigError) Unwrap() error { return ErrUnsafeConfig }

// Check refuses a cookie configuration whose values would put the session id
// somewhere it can be read or replayed. It returns a *[ConfigError] naming the
// first field that is unsafe, and nil when there is nothing to say.
//
// It exists because three of these fields are booleans and one is an enum, and
// all four have an unsafe zero value. A struct that corrected them itself would
// accept an explicit HTTPOnly: false and ignore it, which is worse than either
// honouring it or refusing it. So the values stay as written and this refuses
// the combination, out loud, before a cookie carries it to a browser.
//
// What it checks:
//
//   - HTTPOnly must be true. A session id a script can read is a session id a
//     script can take, and no deployment needs the id in JavaScript.
//   - SameSite must state a policy. The zero value and http.SameSiteDefaultMode
//     both emit no attribute at all and leave the decision to whichever browser
//     is asking, so two clients get two policies from one configuration.
//   - SameSite None requires Secure, because a browser drops a None cookie that
//     is not secure and the session then never persists.
//   - Secure must be true when Domain names a host that is not loopback.
//
// What it cannot check: the address the process is actually served on. An empty
// Domain makes the cookie host-only for whatever host answered, and that host is
// not in this struct, so the Secure rule has nothing to read and does not fire.
// The rule sees a configured domain or it sees nothing.
func (c Config) Check() error {
	if !c.HTTPOnly {
		return &ConfigError{
			Field: "HTTPOnly",
			Want:  "must be true: a session id a script can read is a session id a script can take",
		}
	}

	if c.SameSite == 0 || c.SameSite == http.SameSiteDefaultMode {
		return &ConfigError{
			Field: "SameSite",
			Want: "must be http.SameSiteLaxMode, http.SameSiteStrictMode or http.SameSiteNoneMode: " +
				"the zero value and DefaultMode both emit no attribute, which leaves the policy to the browser",
		}
	}

	if c.SameSite == http.SameSiteNoneMode && !c.Secure {
		return &ConfigError{
			Field: "Secure",
			Want:  "must be true when SameSite is None: a browser drops a None cookie that is not secure",
		}
	}

	if !c.Secure && c.Domain != "" && !isLoopbackDomain(c.Domain) {
		return &ConfigError{
			Field: "Secure",
			Want: fmt.Sprintf("must be true for domain %q: without it the session id travels in the "+
				"clear and anything on the path can replay it", c.Domain),
		}
	}

	return nil
}

// isLoopbackDomain reports whether domain names the machine the browser is
// running on, which is the one place a cookie without Secure goes nowhere a
// third party can read it.
//
// A leading dot is a cookie domain written the old way and means the same host,
// so it is trimmed before the comparison.
func isLoopbackDomain(domain string) bool {
	switch strings.ToLower(strings.TrimPrefix(domain, ".")) {
	case "localhost", "127.0.0.1", "::1", "[::1]":
		return true
	}
	return false
}
