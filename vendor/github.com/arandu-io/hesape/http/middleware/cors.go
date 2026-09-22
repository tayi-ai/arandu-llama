package middleware

import "time"

// CorsConfig is the CORS policy a project configures.
//
// It is declared here, in the package that reads it: without a container
// nothing looks a value up by key, so the middleware is handed its settings
// and the compiler checks the field.
//
// It exists as a type and not as five parameters because of what the five
// permit in combination. See [CorsConfig.Valid].
type CorsConfig struct {
	// AllowedOrigins is the set of origins that may read a response.
	//
	// "*" means any, and it means it literally: the header sent is "*", not the
	// caller's own origin echoed back. Those two are not the same thing, and the
	// difference is the whole of Valid below.
	AllowedOrigins []string

	// AllowedMethods and AllowedHeaders answer a preflight. Empty means the
	// header is not sent at all, which is not the same as sending it empty.
	AllowedMethods []string
	AllowedHeaders []string

	// ExposedHeaders is what a browser lets the calling script read off the
	// response. Everything not listed is invisible to it, including headers the
	// response plainly carries.
	ExposedHeaders []string

	// MaxAge is how long a browser may cache a preflight. Zero means the header
	// is not sent and the browser preflights every time.
	MaxAge time.Duration

	// AllowCredentials lets the browser send cookies and read the response of a
	// cross-origin request that carried them.
	//
	// This is the field that turns a permissive origin list into a way for any
	// site to read an authenticated response. Valid refuses the combination.
	AllowCredentials bool
}

// Valid reports whether the configuration is safe to serve, and says what is
// wrong when it is not.
//
// # The combination it exists to refuse
//
// "*" together with AllowCredentials is forbidden by the CORS specification,
// and every browser refuses it -- a response carrying both
// Access-Control-Allow-Origin: * and Access-Control-Allow-Credentials: true is
// rejected by the client.
//
// The dangerous shape is the workaround: instead of sending "*", echo back
// whatever Origin the request carried. That passes the browser's check, because
// the header now names one specific origin, and it means every origin is
// allowed to read authenticated responses. It is the classic CORS leak, and it
// is what this middleware used to do -- the allow list was checked with
// `o == "*" || o == origin` and the header was then set to the request's own
// origin either way, so configuring "*" with credentials handed every site on
// the internet the ability to read any signed-in user's data.
//
// Refusing at boot rather than at request time is deliberate. A wrong CORS
// policy does not fail: it works, quietly, for the attacker.
func (c CorsConfig) Valid() error {
	if !c.AllowCredentials {
		return nil
	}
	for _, o := range c.AllowedOrigins {
		if o == "*" {
			return errCorsWildcardWithCredentials
		}
	}
	return nil
}

// DefaultCorsConfig is what a project gets without configuring anything: no
// origin allowed.
//
// An allow-all default would be unsafe here: this framework's requests carry
// a session cookie by design, and a permissive origin list combined with
// credentials is exactly the leak Valid refuses.
//
// An empty list means the middleware adds no header and the browser refuses the
// cross-origin read, which is the correct answer for an application that has
// not decided who may call it.
func DefaultCorsConfig() CorsConfig {
	return CorsConfig{
		AllowedMethods: []string{"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"},
		AllowedHeaders: []string{"Content-Type", "X-CSRF-Token", "X-Requested-With"},
		MaxAge:         24 * time.Hour,
	}
}

// errCorsWildcardWithCredentials is the one refusal Valid makes.
var errCorsWildcardWithCredentials = corsError(
	"cors: allowed_origins contains \"*\" and supports_credentials is on. " +
		"A browser refuses a response carrying both, and the usual workaround -- " +
		"echoing the caller's own Origin back instead of \"*\" -- lets every site " +
		"read the responses of anybody signed in. Name the origins that may call, " +
		"or turn credentials off.")

// corsError is a string error, declared here so this file takes no import for
// one value.
type corsError string

func (e corsError) Error() string { return string(e) }
