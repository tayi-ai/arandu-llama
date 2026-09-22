package database

import (
	"database/sql"
	"fmt"
	"net/url"
	"strings"

	"github.com/arandu-io/hesape/support"
)

// ConnectionFactory turns a configuration into an open connection, picking the
// driver by name.
//
// It lives in the root package rather than in database/connectors, and that is
// the whole point of it. The connectors are separate Go modules -- pgx, mysql
// and sqlite each with their own go.mod -- because Go has no optional
// dependency, and a factory that imported all three to choose between them
// would put all three in the go.sum of every project. So the choice is made by
// registration instead:
//
//	import (
//	    "github.com/arandu-io/hesape/database"
//	    _ "github.com/arandu-io/hesape/database/connectors/pgx"
//	)
//
// The blank import runs the connector's init, which calls database.Register
// with the dialect it supports and the database/sql driver name it linked.
// The factory reads that registry, and never names a driver package.
//
// # How a project registers its own
//
// A project that speaks an engine this framework does not ship writes the same
// four lines the shipped connectors write:
//
//	package clickhouse
//
//	import (
//	    "github.com/arandu-io/hesape/database"
//	    _ "github.com/ClickHouse/clickhouse-go/v2"
//	)
//
//	type Connector struct{}
//
//	func (Connector) Dialect() database.Dialect { return "clickhouse" }
//	func (Connector) DriverName() string        { return "clickhouse" }
//
//	func init() { database.Register(Connector{}) }
//
// and blank-imports it from main. Nothing in this package changes, and nothing
// in this package learns the name. If the connection also needs its own SQL
// spelling, it registers a grammar through DefaultQueryGrammar and a connection
// through ResolverFor, which are the same inversion one layer down.
type ConnectionFactory struct{}

// NewConnectionFactory builds a ConnectionFactory. It takes nothing: what it
// needs is the driver registry, which is package state.
func NewConnectionFactory() *ConnectionFactory { return &ConnectionFactory{} }

// Make opens one connection, or a read/write pair when the configuration
// has a "read" key.
func (f *ConnectionFactory) Make(config map[string]any, name string) (*Connection, error) {
	parsed, err := f.parseConfig(config, name)
	if err != nil {
		return nil, err
	}

	if _, hasRead := parsed["read"]; hasRead {
		return f.createReadWriteConnection(parsed)
	}
	return f.createSingleConnection(parsed)
}

// parseConfig expands a configuration's url, if it has one, and fills in
// "prefix" and "name" when absent -- one place rather than two, because a
// configuration that carries a url and is read by the factory directly is
// the same configuration.
func (f *ConnectionFactory) parseConfig(config map[string]any, name string) (map[string]any, error) {
	parsed, err := support.NewConfigurationUrlParser().ParseConfiguration(config)
	if err != nil {
		return nil, err
	}
	if _, set := parsed["prefix"]; !set {
		parsed["prefix"] = ""
	}
	if _, set := parsed["name"]; !set {
		parsed["name"] = name
	}
	return parsed, nil
}

// createSingleConnection opens one connection from config.
func (f *ConnectionFactory) createSingleConnection(config map[string]any) (*Connection, error) {
	pool, err := f.connect(config)
	if err != nil {
		return nil, err
	}

	driver, _ := config["driver"].(string)
	database, _ := config["database"].(string)
	prefix, _ := config["prefix"].(string)

	return f.CreateConnection(driver, pool, database, prefix, config)
}

// createReadWriteConnection opens the write connection from config, then
// opens a read pool from the read-side configuration and attaches it.
func (f *ConnectionFactory) createReadWriteConnection(config map[string]any) (*Connection, error) {
	connection, err := f.createSingleConnection(f.getWriteConfig(config))
	if err != nil {
		return nil, err
	}

	readConfig := f.getReadConfig(config)

	readPool, err := f.connect(readConfig)
	if err != nil {
		return nil, err
	}

	return connection.SetReadPDO(readPool).SetReadPDOConfig(readConfig), nil
}

// getReadConfig returns the merged configuration for the read side.
func (f *ConnectionFactory) getReadConfig(config map[string]any) map[string]any {
	return f.mergeReadWriteConfig(config, f.getReadWriteConfig(config, "read"))
}

// getWriteConfig returns the merged configuration for the write side.
func (f *ConnectionFactory) getWriteConfig(config map[string]any) map[string]any {
	return f.mergeReadWriteConfig(config, f.getReadWriteConfig(config, "write"))
}

// getReadWriteConfig returns the first entry of config[typ], whether it is
// given as a single map or a list of them.
//
// A list of replicas picks the first rather than a random one, and the
// difference is deliberate: picking at random spreads load, but it also
// makes an error message name a different host every time somebody runs the
// command. Load spreading belongs to whatever sits in front of the
// replicas, and every managed platform has one.
func (f *ConnectionFactory) getReadWriteConfig(config map[string]any, typ string) map[string]any {
	switch value := config[typ].(type) {
	case map[string]any:
		return value
	case []map[string]any:
		if len(value) > 0 {
			return value[0]
		}
	case []any:
		if len(value) > 0 {
			if first, ok := value[0].(map[string]any); ok {
				return first
			}
		}
	}
	return map[string]any{}
}

// mergeReadWriteConfig merges merge over config, dropping the "read" and
// "write" keys that named the source of merge in the first place.
func (f *ConnectionFactory) mergeReadWriteConfig(config, merge map[string]any) map[string]any {
	out := make(map[string]any, len(config)+len(merge))
	for key, value := range config {
		if key == "read" || key == "write" {
			continue
		}
		out[key] = value
	}
	for key, value := range merge {
		out[key] = value
	}
	return out
}

// CreateConnector returns the connector registered for this configuration's
// driver.
//
// The match is a registry lookup rather than a name comparison against
// every known driver, because the alternative is importing every driver
// package just to be able to name them.
func (f *ConnectionFactory) CreateConnector(config map[string]any) (Connector, error) {
	driver, _ := config["driver"].(string)
	if driver == "" {
		return nil, fmt.Errorf("A driver must be specified.")
	}

	dialect, err := ParseDialect(driver)
	if err != nil {
		return nil, fmt.Errorf("Unsupported driver [%s].", driver)
	}

	name, err := driverFor(dialect)
	if err != nil {
		return nil, err
	}
	return registeredConnector{dialect: dialect, driverName: name}, nil
}

// registeredConnector is the pair a connector's init put in the registry,
// returned by CreateConnector.
type registeredConnector struct {
	dialect    Dialect
	driverName string
}

// Dialect returns the dialect this connector was registered for.
func (c registeredConnector) Dialect() Dialect { return c.dialect }

// DriverName returns the database/sql driver name this connector linked.
func (c registeredConnector) DriverName() string { return c.driverName }

// connect opens the pool for config, eagerly.
//
// It does not defer opening into a closure for an unused connection to skip:
// sql.Open does not talk to the server either, so a lazy connect only moves
// the first failure into the first request, which is inside somebody's page
// load.
func (f *ConnectionFactory) connect(config map[string]any) (*sql.DB, error) {
	connector, err := f.CreateConnector(config)
	if err != nil {
		return nil, err
	}

	dsn, err := f.dsn(config)
	if err != nil {
		return nil, err
	}

	pool, err := sqlOpen(connector.DriverName(), dsn)
	if err != nil {
		return nil, fmt.Errorf("opening %s: %w", config["driver"], err)
	}
	return pool, nil
}

// dsn turns the configuration array into the connection string the driver
// takes, by way of Config -- which is where every DSN in this package is
// spelled, so there is one of them.
func (f *ConnectionFactory) dsn(config map[string]any) (string, error) {
	driver, _ := config["driver"].(string)
	dialect, err := ParseDialect(driver)
	if err != nil {
		return "", fmt.Errorf("Unsupported driver [%s].", driver)
	}

	cfg := Config{
		Connection: dialect,
		Database:   asText(config["database"]),
		Host:       hostOf(config["host"]),
		Port:       asText(config["port"]),
		Username:   asText(config["username"]),
		Password:   asText(config["password"]),
		Options:    url.Values{},
	}

	for key, value := range config {
		switch key {
		case "driver", "database", "host", "port", "username", "password",
			"prefix", "name", "read", "write", "url", "options":
			continue
		}
		cfg.Options.Set(key, asText(value))
	}

	return cfg.DSN(), nil
}

// hostOf reads the host, allowing it to be a list of replicas, and returns
// the first one.
func hostOf(value any) string {
	switch host := value.(type) {
	case string:
		return host
	case []string:
		if len(host) > 0 {
			return host[0]
		}
	case []any:
		if len(host) > 0 {
			return asText(host[0])
		}
	}
	return ""
}

// CreateConnection returns the Connection for a driver, through the
// resolver a connector registered, or the plain constructor otherwise.
//
// It is exported because a connector in another module registers through
// ResolverFor and a project building a connection by hand has no other door.
func (f *ConnectionFactory) CreateConnection(driver string, pool *sql.DB, database, prefix string, config map[string]any) (*Connection, error) {
	if resolver := GetResolver(driver); resolver != nil {
		return resolver(pool, database, prefix, config), nil
	}

	if _, err := ParseDialect(driver); err != nil {
		return nil, fmt.Errorf("Unsupported driver [%s].", driver)
	}

	return NewConnection(pool, database, prefix, config), nil
}

// SupportedDrivers answers the dialects this package can open.
//
// SQL Server is not on it and will not be, and mariadb is spoken by the MySQL
// connector rather than a fourth one.
func SupportedDrivers() []string {
	return []string{"mysql", "mariadb", "pgsql", "sqlite"}
}

// AvailableDrivers returns the supported drivers that this binary can
// actually reach: the intersection of SupportedDrivers and the connector
// registry, which reports what main.go imported -- the same question asked
// of the thing that decides it here.
func AvailableDrivers() []string {
	linked := map[Dialect]bool{}
	for _, dialect := range Registered() {
		linked[dialect] = true
	}

	var out []string
	for _, driver := range SupportedDrivers() {
		dialect, err := ParseDialect(driver)
		if err != nil {
			continue
		}
		if linked[dialect] {
			out = append(out, driver)
		}
	}
	return out
}

// driverTitles is what GetDriverTitle answers per driver, so a console table
// reads the way a person names the engine.
var driverTitles = map[string]string{
	"mysql":   "MySQL",
	"mariadb": "MariaDB",
	"pgsql":   "PostgreSQL",
	"sqlite":  "SQLite",
}

// DriverTitle answers the human name of a driver, and the driver itself for one
// nobody named.
func DriverTitle(driver string) string {
	if title, known := driverTitles[strings.ToLower(driver)]; known {
		return title
	}
	return driver
}
