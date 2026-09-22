package database

// Connector links one engine's database/sql driver into the binary.
//
// It is declared here rather than in the connectors package one level down: it
// names a Dialect, and a connectors package holding the interface would have to
// import this one while this one imports it back. Go refuses that cycle, and
// the interface is what moves.
//
// A connector is a package with an init() and nothing to call. The project
// blank-imports the engines it speaks:
//
//	import (
//	    "github.com/arandu-io/hesape/database"
//	    _ "github.com/arandu-io/hesape/database/connectors/pgx"
//	    _ "github.com/arandu-io/hesape/database/connectors/sqlite"
//	)
//
// and Open resolves the rest from DATABASE_URL. There is no second way to open
// a connection: a connector says which driver it linked, and never opens one
// itself.
type Connector interface {
	// Dialect reports the connection dialect this connector supports.
	Dialect() Dialect

	// DriverName is the name the driver registered with database/sql. It is
	// read back from here rather than guessed, so a connector that switches
	// implementations does not need Open to be edited with it.
	DriverName() string
}
