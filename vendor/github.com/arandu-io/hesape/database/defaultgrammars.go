package database

import (
	"github.com/arandu-io/hesape/database/query"
	"github.com/arandu-io/hesape/database/query/grammars"
	"github.com/arandu-io/hesape/database/query/processors"
)

// init registers the built-in grammars and processors as the defaults.
//
// It goes through DefaultQueryGrammar and DefaultPostProcessor rather than
// around them, so the shipped dialects register the way a project's own driver
// registers: one door, exercised by the drivers that ship.
//
// The nil these two variables held before was never a choice anybody made. It
// was a registration nobody performed, and every Connection built by
// NewConnection carried a nil grammar and a nil processor -- so Query returned
// a builder that dereferenced nil on its first statement, on every dialect.
//
// The inversion point is unchanged: a package init runs before the init of any
// package that imports it, so a connector assigning either variable overwrites
// what is set here. What changes is where a project starts from.
func init() {
	DefaultQueryGrammar = builtinQueryGrammar
	DefaultPostProcessor = builtinPostProcessor
}

// builtinQueryGrammar returns the shipped grammar for a dialect.
//
// SQLite is the fallback rather than nil, for the same reason Connection.dialect
// falls back to it: a grammar is dereferenced on the first statement, so an
// unknown dialect returning nothing here is a panic later instead of an answer
// now. MariaDB is spoken by the MySQL connector, so grammars.MariaDBGrammar has
// no dialect of its own to be selected by.
func builtinQueryGrammar(dialect Dialect) query.Grammar {
	switch dialect {
	case DialectPostgres:
		return grammars.NewPostgresGrammar()
	case DialectMySQL:
		return grammars.NewMySQLGrammar()
	default:
		return grammars.NewSQLiteGrammar()
	}
}

// builtinPostProcessor returns the shipped processor for a dialect.
//
// It takes the dialect because the processors differ by it, and by exactly the
// thing that is hardest to notice when it is wrong: Postgres reads an inserted
// identifier out of a returning clause and MySQL reads it out of band, so a
// connection given the wrong one gets a wrong identifier rather than an error.
func builtinPostProcessor(dialect Dialect) query.Processor {
	switch dialect {
	case DialectPostgres:
		return processors.NewPostgresProcessor()
	case DialectMySQL:
		return processors.NewMySQLProcessor()
	default:
		return processors.NewSQLiteProcessor()
	}
}
