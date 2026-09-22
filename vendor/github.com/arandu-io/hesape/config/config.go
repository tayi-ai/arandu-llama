package config

import (
	"fmt"
	"net/url"
	"strings"
	"time"

	"github.com/arandu-io/hesape/encryption"
)

// Env is the deployment environment. It gates everything that must never run
// outside development, starting with the debug error page.
type Env string

// Supported environments.
const (
	EnvDev     Env = "dev"
	EnvStaging Env = "staging"
	EnvProd    Env = "prod"
)

// Is reports whether the environment is want.
//
// It exists so a caller reads as a sentence -- cfg.Env.Is(config.EnvStaging) --
// and so the comparison happens in one place if Env ever stops being a bare
// string.
func (e Env) Is(want Env) bool { return e == want }

// IsProduction reports whether this is the environment where a mistake is
// visible to a customer. It is the question asked most often, and spelling it
// out is what keeps EnvProd from being compared against a literal "production"
// -- a typo the type system cannot catch and this method removes the occasion
// for.
func (e Env) IsProduction() bool { return e == EnvProd }

// App is the identity of the application. It is a struct, not a map, and every
// field is validated at boot.
type App struct {
	// Name appears in the page title, in the log and in outgoing mail.
	Name string

	// Env gates everything that must never run outside development.
	Env Env

	// Debug turns on the debug error page -- the one that prints a stack trace,
	// the request and the environment. Validate refuses it in production, where
	// that page is a disclosure and not a convenience.
	Debug bool

	// URL is the canonical address of the application, parsed. It is what builds
	// an absolute link from a job or a scheduled task, where there is no request
	// to read a host from.
	URL *url.URL

	// HTTPAddr is what the server listens on.
	HTTPAddr string

	// Timezone is the location dates are rendered in, resolved. Storage is
	// always UTC; this is presentation only.
	Timezone *time.Location

	// Locale is the default language tag.
	Locale string

	// Key signs session cookies, CSRF tokens and signed URLs. Exactly
	// [encryption.KeySize] bytes.
	Key []byte

	// PreviousKeys are keys retired but still accepted on verification, newest
	// first. They are what makes rotating Key a deploy rather than an outage:
	// signatures issued under the old key keep verifying until they expire.
	// Nothing is ever signed with one.
	PreviousKeys [][]byte
}

// Load reads the environment and validates it. It fails at boot, not on the
// first request.
//
// A.env in the working directory is read first, and it only fills variables
// the environment does not already define -- see [LoadDotenv]. It is read here
// rather than by the application, because a step the application has to
// remember is a step that gets forgotten: this one was, and `aru migrate`
// failed on every new project for it.
func Load() (App, error) {
	if err := LoadDotenv(); err != nil {
		return App{}, err
	}

	// The key is parsed by encryption, not here. It is the same key encryption
	// signs with, and a second reader of the same variable is a second answer to
	// "is this key valid". Naming the variable is this package's part: the error
	// has to send a reader to a line of a.env file.
	key, err := encryption.ParseKey(String("APP_KEY", ""))
	if err != nil {
		return App{}, fmt.Errorf("APP_KEY: %w", err)
	}
	previous, err := parsePreviousKeys(String("APP_PREVIOUS_KEYS", ""))
	if err != nil {
		return App{}, err
	}

	env := Env(String("APP_ENV", string(EnvDev)))

	raw := String("APP_URL", "http://localhost:8080")
	parsed, err := url.Parse(raw)
	if err != nil {
		return App{}, fmt.Errorf("APP_URL is not a URL: %w", err)
	}

	zone := String("APP_TIMEZONE", "UTC")
	location, err := time.LoadLocation(zone)
	if err != nil {
		return App{}, fmt.Errorf("APP_TIMEZONE %q is not a time zone: %w", zone, err)
	}

	app := App{
		Name: String("APP_NAME", "arandu-app"),
		Env:  env,
		// The default is the environment, not false: a developer who has to set
		// a variable to see the error page will read a blank 500 instead, and a
		// default of true would carry the page into production on the first
		// deploy that forgets to unset it. Validate refuses that combination
		// either way.
		Debug:        Bool("APP_DEBUG", env == EnvDev),
		URL:          parsed,
		HTTPAddr:     String("HTTP_ADDR", ":8080"),
		Timezone:     location,
		Locale:       String("APP_LOCALE", "en"),
		Key:          key,
		PreviousKeys: previous,
	}
	return app, app.Validate()
}

// Validate reports the first configuration error, with the command that fixes
// it whenever one exists.
func (a App) Validate() error {
	switch a.Env {
	case EnvDev, EnvStaging, EnvProd:
	default:
		return fmt.Errorf("invalid APP_ENV: %q (expected dev, staging or prod)", a.Env)
	}
	if len(a.Key) != encryption.KeySize {
		return fmt.Errorf("APP_KEY must be %d bytes, got %d (run `aru key:generate`)", encryption.KeySize, len(a.Key))
	}
	for i, k := range a.PreviousKeys {
		if len(k) != encryption.KeySize {
			return fmt.Errorf("APP_PREVIOUS_KEYS entry %d must be %d bytes, got %d", i+1, encryption.KeySize, len(k))
		}
	}
	if a.Debug && a.Env.IsProduction() {
		return fmt.Errorf("APP_DEBUG is forbidden in production: the debug page prints the stack, the request and the environment")
	}
	if a.URL == nil || a.URL.Scheme == "" || a.URL.Host == "" {
		return fmt.Errorf("APP_URL must be absolute, with a scheme and a host (for example https://billing.example.com)")
	}
	if a.Timezone == nil {
		return fmt.Errorf("APP_TIMEZONE is required")
	}
	if !a.Env.Is(EnvDev) && a.HTTPAddr == "" {
		return fmt.Errorf("HTTP_ADDR is required outside development")
	}
	return nil
}

// parsePreviousKeys reads the retired keys, newest first, separated by commas.
// Each one is in the format [encryption.ParseKey] accepts, which is the format
// APP_KEY is in: a rotated key is a key that used to be APP_KEY.
//
// Empty entries are skipped rather than rejected, because the value is written
// by prepending: "new,old" is typed as ",old" first often enough that failing
// on it would cost a deploy for nothing.
func parsePreviousKeys(v string) ([][]byte, error) {
	if v == "" {
		return nil, nil
	}
	var keys [][]byte
	for _, part := range strings.Split(v, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		key, err := encryption.ParseKey(part)
		if err != nil {
			return nil, fmt.Errorf("APP_PREVIOUS_KEYS entry %d: %w", len(keys)+1, err)
		}
		keys = append(keys, key)
	}
	return keys, nil
}
