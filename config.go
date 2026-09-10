package llama

import (
	"fmt"
	"net/http"

	"github.com/arandu-io/framework/security"
)

// The defaults for the optional settings. They are constants rather than
// literals inside Config.withDefaults, so the value a reader finds here is the
// value the package uses.
const (
	// DefaultPrefix is where the routes are mounted when Config leaves Prefix
	// empty.
	DefaultPrefix = "/llama"
	// DefaultPageSize is how many records one page answers with when Config
	// leaves PageSize at zero.
	DefaultPageSize = 25
	// MaxPageSize is the ceiling PageSize is refused above. A page nobody
	// bounded is a page that reads the whole table on the day the table is
	// large.
	MaxPageSize = 200
)

// Config is what the application passes when it wires this package.
//
// A typed struct rather than a map: a misspelled key in a map is a setting that
// silently keeps its default, and the failure shows up as behaviour nobody
// asked for rather than as an error. Here a field that does not exist does not
// compile.
type Config struct {
	// Tenant is the customer a visitor with no session is read as.
	//
	// It is required, and it comes from the application's own configuration --
	// never from the request. A tenant a visitor could name is a visitor who
	// chooses whose rows they read. Everywhere there is a session, the tenant
	// comes from the Grant instead, and this value is not consulted at all.
	Tenant string

	// Prefix is the path the routes are mounted under. Empty means
	// DefaultPrefix.
	Prefix string

	// PageSize is how many records one page answers with. Zero means
	// DefaultPageSize, and anything above MaxPageSize is refused rather than
	// clamped: a number somebody wrote and did not get is worse than a number
	// somebody wrote and was told about.
	PageSize int
}

// Validate reports what the configuration cannot be used with.
//
// It is called by New, so an application with a setting that cannot work fails
// where it is wired rather than on the first request that needed it.
func (c Config) Validate() error {
	if c.Tenant == "" {
		return fmt.Errorf("llama: Config.Tenant is required: a visitor with no session has to be read as some customer, and it cannot be one the request names")
	}
	// The same rule the framework applies to every tenant it accepts. A tenant
	// is concatenated into a storage path, a cache key and a lock name, so one
	// carrying a separator lands in another tenant's namespace.
	if !security.ValidTenant(c.Tenant) {
		return fmt.Errorf("llama: Config.Tenant is %q, which cannot be a tenant: lowercase letters, digits, - and _, up to 64 characters", c.Tenant)
	}
	if c.Prefix != "" && c.Prefix[0] != '/' {
		return fmt.Errorf("llama: Config.Prefix is %q and has to start with /", c.Prefix)
	}
	if c.Prefix != "" {
		if err := validateRoutePrefix(c.Prefix); err != nil {
			return err
		}
	}
	if c.PageSize < 0 || c.PageSize > MaxPageSize {
		return fmt.Errorf("llama: Config.PageSize is %d, and has to be between 0 and %d, where 0 means %d", c.PageSize, MaxPageSize, DefaultPageSize)
	}
	return nil
}

// validateRoutePrefix asks the standard library to parse the exact patterns
// the module will register. Its parser is not exported and reports invalid
// patterns by panic, so the throwaway mux turns that boot-time panic into the
// configuration error New promises.
func validateRoutePrefix(prefix string) (err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("llama: Config.Prefix %q cannot be registered as a route path", prefix)
		}
	}()

	handler := http.NotFoundHandler()
	mux := http.NewServeMux()
	mux.Handle(http.MethodGet+" "+prefix, handler)
	mux.Handle(http.MethodGet+" "+prefix+"/{id}", handler)
	mux.Handle(http.MethodPost+" "+prefix, handler)
	return nil
}

// withDefaults returns the configuration with the optional fields filled in.
//
// It runs after Validate and never before: filling a default in first would
// hide the value somebody actually wrote from the check that would have refused
// it.
func (c Config) withDefaults() Config {
	if c.Prefix == "" {
		c.Prefix = DefaultPrefix
	}
	if c.PageSize == 0 {
		c.PageSize = DefaultPageSize
	}
	return c
}
