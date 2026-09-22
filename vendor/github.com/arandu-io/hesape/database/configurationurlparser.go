package database

import "github.com/arandu-io/hesape/support"

// ConfigurationUrlParser reads a database URL into its parts.
//
// It is an alias of support.ConfigurationUrlParser, so a caller in this package
// can name it without importing that one. There is exactly one implementation.
type ConfigurationUrlParser = support.ConfigurationUrlParser

// NewConfigurationUrlParser creates a ConfigurationUrlParser.
func NewConfigurationUrlParser() *ConfigurationUrlParser {
	return support.NewConfigurationUrlParser()
}
