package support

import (
	"encoding/json"
	"errors"
	"net/url"
	"regexp"
	"strings"
	"sync"

	"github.com/arandu-io/hesape/collections/arr"
)

// ErrMalformedConfigurationURL is returned when the configuration URL cannot
// be read.
var ErrMalformedConfigurationURL = errors.New("The database configuration URL is malformed.")

// ConfigurationUrlParser turns a single database URL into the connection
// options a driver wants.
type ConfigurationUrlParser struct{}

// NewConfigurationUrlParser returns a parser. It carries no state, so the zero
// value works just as well.
func NewConfigurationUrlParser() *ConfigurationUrlParser { return &ConfigurationUrlParser{} }

var (
	driverAliasesMu sync.RWMutex

	// driverAliases maps a URL scheme to the driver it selects.
	driverAliases = map[string]string{
		"mssql":      "sqlsrv",
		"mysql2":     "mysql", // RDS
		"postgres":   "pgsql",
		"postgresql": "pgsql",
		"sqlite3":    "sqlite",
		"redis":      "tcp",
		"rediss":     "tls",
	}
)

// sqliteTripleSlash matches a hostless sqlite URL, whose path would otherwise
// be eaten by the parser.
var sqliteTripleSlash = regexp.MustCompile(`^(sqlite3?):///`)

// ParseConfiguration returns the configuration with its url key read out and
// spread over driver, database, host, port, username, password and whatever
// the query string carried.
//
// The configuration is a map, or a bare string taken as the URL itself; nil
// and any other type give an empty map. A URL that cannot be read is
// [ErrMalformedConfigurationURL].
func (p *ConfigurationUrlParser) ParseConfiguration(config any) (map[string]any, error) {
	settings := map[string]any{}
	switch c := config.(type) {
	case nil:
		return settings, nil
	case string:
		settings["url"] = c
	case map[string]any:
		for key, held := range c {
			settings[key] = held
		}
	default:
		return settings, nil
	}

	pulled, _ := arr.Pull(settings, "url")
	raw := toString(pulled)
	if raw == "" {
		return settings, nil
	}

	parsed, err := parseConfigurationURL(raw)
	if err != nil {
		return nil, err
	}

	for key, held := range p.getPrimaryOptions(parsed) {
		settings[key] = held
	}
	for key, held := range p.getQueryOptions(parsed) {
		settings[key] = held
	}
	return settings, nil
}

// urlComponents holds the parts of a URL the parser reads.
type urlComponents struct {
	scheme string
	host   string
	port   string
	user   string
	pass   string
	path   string
	query  string
}

// parseConfigurationURL splits a URL into its parts. A hostless sqlite URL is
// given a placeholder host first, so its path survives parsing.
func parseConfigurationURL(raw string) (urlComponents, error) {
	raw = sqliteTripleSlash.ReplaceAllString(raw, "$1://null/")
	parsed, err := url.Parse(raw)
	if err != nil {
		return urlComponents{}, ErrMalformedConfigurationURL
	}
	components := urlComponents{
		scheme: parsed.Scheme,
		host:   parsed.Hostname(),
		port:   parsed.Port(),
		path:   parsed.Path,
		query:  parsed.RawQuery,
	}
	if parsed.User != nil {
		components.user = parsed.User.Username()
		components.pass, _ = parsed.User.Password()
	}
	return components, nil
}

// getPrimaryOptions reads driver, database, host, port, username and password
// out of the URL. A component the URL does not carry is left out entirely, so
// it does not write over a value the configuration already held.
func (p *ConfigurationUrlParser) getPrimaryOptions(parsed urlComponents) map[string]any {
	options := map[string]any{}
	if driver := p.getDriver(parsed); driver != "" {
		options["driver"] = driver
	}
	if database := p.getDatabase(parsed); database != "" {
		options["database"] = parseStringToNativeType(rawURLDecode(database))
	}
	if parsed.host != "" {
		options["host"] = parseStringToNativeType(rawURLDecode(parsed.host))
	}
	if parsed.port != "" {
		options["port"] = parseStringToNativeType(parsed.port)
	}
	if parsed.user != "" {
		options["username"] = parseStringToNativeType(rawURLDecode(parsed.user))
	}
	if parsed.pass != "" {
		options["password"] = parseStringToNativeType(rawURLDecode(parsed.pass))
	}
	return options
}

// getDriver returns the driver the scheme selects, through the alias table
// when the scheme has one, and the scheme itself otherwise.
func (p *ConfigurationUrlParser) getDriver(parsed urlComponents) string {
	if parsed.scheme == "" {
		return ""
	}
	driverAliasesMu.RLock()
	defer driverAliasesMu.RUnlock()
	if alias, ok := driverAliases[parsed.scheme]; ok {
		return alias
	}
	return parsed.scheme
}

// getDatabase returns the path with its leading slash off, and nothing at all
// when there is no path.
func (p *ConfigurationUrlParser) getDatabase(parsed urlComponents) string {
	if parsed.path == "" || parsed.path == "/" {
		return ""
	}
	return strings.TrimPrefix(parsed.path, "/")
}

// getQueryOptions reads the query string into a map, each value converted to
// the type its text names.
func (p *ConfigurationUrlParser) getQueryOptions(parsed urlComponents) map[string]any {
	if parsed.query == "" {
		return map[string]any{}
	}
	return parseStringsToNativeTypes(parseQueryString(parsed.query)).(map[string]any)
}

// parseStringsToNativeTypes walks a value and converts every string it reaches
// with [parseStringToNativeType], descending into maps and slices.
func parseStringsToNativeTypes(v any) any {
	switch typed := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(typed))
		for key, held := range typed {
			out[key] = parseStringsToNativeTypes(held)
		}
		return out
	case []any:
		out := make([]any, 0, len(typed))
		for _, held := range typed {
			out = append(out, parseStringsToNativeTypes(held))
		}
		return out
	case string:
		return parseStringToNativeType(typed)
	default:
		return v
	}
}

// parseStringToNativeType converts a string to the type its text names:
// "true" is a bool, "5432" is a number, and anything the JSON decoder refuses
// stays the string it was.
//
// The decoder reads every number as a float64, so an integral one is handed
// back as an int.
func parseStringToNativeType(v string) any {
	var decoded any
	if err := json.Unmarshal([]byte(v), &decoded); err != nil {
		return v
	}
	if number, ok := decoded.(float64); ok && number == float64(int(number)) {
		return int(number)
	}
	return decoded
}

// GetDriverAliases returns a copy of the table mapping a URL scheme to the
// driver it selects.
func GetDriverAliases() map[string]string {
	driverAliasesMu.RLock()
	defer driverAliasesMu.RUnlock()
	out := make(map[string]string, len(driverAliases))
	for alias, driver := range driverAliases {
		out[alias] = driver
	}
	return out
}

// AddDriverAlias registers a URL scheme and the driver it selects, replacing
// whatever the scheme mapped to before.
func AddDriverAlias(alias, driver string) {
	driverAliasesMu.Lock()
	defer driverAliasesMu.Unlock()
	driverAliases[alias] = driver
}
