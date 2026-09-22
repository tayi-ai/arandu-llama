package database

import (
	"errors"
	"fmt"
	"strings"

	"github.com/arandu-io/hesape/database/concerns"
)

// QueryException reports that a statement failed, and says which one, on which
// connection, with which values.
//
// The message is assembled in three parts: the driver's own sentence, then the
// connection with its host, port and database, then the SQL with the bindings
// written into the placeholders. That last part is why this type
// exists at all -- "syntax error at or near $3" names nothing a person can act
// on, and the same error with the values in it usually names the mistake.
type QueryException struct {
	// ConnectionName is the connection's name.
	ConnectionName string

	// SQL is the query as it was issued.
	SQL string

	// Bindings are the values that went with it, already through
	// PrepareBindings.
	Bindings []any

	// ConnectionDetails is driver, name, host, port, database, unix_socket.
	ConnectionDetails map[string]any

	// ReadWriteType is "read", "write", or empty.
	ReadWriteType string

	// Previous is the driver error this exception wraps.
	Previous error
}

// NewQueryException creates a QueryException.
func NewQueryException(connectionName, sql string, bindings []any, previous error, connectionDetails map[string]any, readWriteType string) *QueryException {
	return &QueryException{
		ConnectionName:    connectionName,
		SQL:               sql,
		Bindings:          bindings,
		ConnectionDetails: connectionDetails,
		ReadWriteType:     readWriteType,
		Previous:          previous,
	}
}

// Error formats the driver's message, the connection and the query with its
// bindings written in, in that order.
func (e *QueryException) Error() string {
	message := ""
	if e.Previous != nil {
		message = e.Previous.Error()
	}
	return message + " (Connection: " + e.ConnectionName + e.formatConnectionDetails() +
		", SQL: " + substituteBindings(e.SQL, e.Bindings) + ")"
}

// formatConnectionDetails renders the connection's host, port and database
// -- or its socket, for a driver that uses one -- for Error's message.
func (e *QueryException) formatConnectionDetails() string {
	if len(e.ConnectionDetails) == 0 {
		return ""
	}

	driver, _ := e.ConnectionDetails["driver"].(string)

	var segments []string
	if driver != "sqlite" {
		if socket, _ := e.ConnectionDetails["unix_socket"].(string); socket != "" {
			segments = append(segments, "Socket: "+socket)
		} else {
			segments = append(segments,
				"Host: "+detail(e.ConnectionDetails, "host"),
				"Port: "+detail(e.ConnectionDetails, "port"))
		}
	}
	segments = append(segments, "Database: "+detail(e.ConnectionDetails, "database"))

	return ", " + strings.Join(segments, ", ")
}

func detail(details map[string]any, key string) string {
	switch v := details[key].(type) {
	case nil:
		return ""
	case string:
		return v
	case []string:
		return strings.Join(v, ", ")
	default:
		return fmt.Sprint(v)
	}
}

// Unwrap makes errors.Is and errors.As reach the driver error.
func (e *QueryException) Unwrap() error { return e.Previous }

// GetConnectionName returns the connection's name.
func (e *QueryException) GetConnectionName() string { return e.ConnectionName }

// GetSQL returns the query as it was issued.
func (e *QueryException) GetSQL() string { return e.SQL }

// GetBindings returns the values that went with the query.
func (e *QueryException) GetBindings() []any { return e.Bindings }

// GetConnectionDetails returns the driver, name, host, port, database and
// socket the connection reported.
func (e *QueryException) GetConnectionDetails() map[string]any { return e.ConnectionDetails }

// GetRawSQL answers the statement with its bindings written in.
//
// Nothing has to be looked up to build it: the bindings are already on the
// exception.
func (e *QueryException) GetRawSQL() string { return substituteBindings(e.SQL, e.Bindings) }

// UniqueConstraintViolationException is a QueryException, which it embeds,
// narrowed to a violated unique constraint.
//
// It exists so firstOrCreate and its neighbours can catch the one failure they
// expect -- two requests inserting the same row at the same time -- without
// catching every other statement error with it.
type UniqueConstraintViolationException struct{ QueryException }

// NewUniqueConstraintViolationException answers its constructor, which is
// QueryException's.
func NewUniqueConstraintViolationException(connectionName, sql string, bindings []any, previous error, connectionDetails map[string]any, readWriteType string) *UniqueConstraintViolationException {
	return &UniqueConstraintViolationException{
		*NewQueryException(connectionName, sql, bindings, previous, connectionDetails, readWriteType),
	}
}

// DeadlockException reports that the engine chose this transaction as the
// deadlock victim.
//
// It is an alias rather than a declaration because ManagesTransactions is what
// raises it and lives in the concerns package, which this one imports -- so
// declaring the type here would close the cycle. An alias is one type under two
// names, so errors.As works whichever name a caller reaches for.
type DeadlockException = concerns.DeadlockError

// NewDeadlockException answers `new DeadlockException(...)`.
func NewDeadlockException(previous error) *DeadlockException {
	return concerns.NewDeadlockError(previous)
}

// LostConnectionException reports that the connection went away and there was
// no reconnector to get it back.
type LostConnectionException struct {
	// Message is the sentence NewLostConnectionException was given.
	Message string
}

// NewLostConnectionException answers `new LostConnectionException($message)`.
func NewLostConnectionException(message string) *LostConnectionException {
	return &LostConnectionException{Message: message}
}

// Error is the message.
func (e *LostConnectionException) Error() string { return e.Message }

// ErrRecordNotFound is what FirstOrFail returns when the query matched no row.
//
// It is the same value as concerns.ErrRecordNotFound, re-exported: FirstOrFail
// lives there, and errors.Is has to keep working across both names, which it
// does because there is only one.
var ErrRecordNotFound = concerns.ErrRecordNotFound

// ErrRecordsNotFound is what Sole returns when the query matched no row at all.
//
// It is the same value as concerns.ErrRecordsNotFound, re-exported for the same
// reason ErrRecordNotFound is: Sole lives in that package, and errors.Is has to
// give the same answer under either name.
var ErrRecordsNotFound = concerns.ErrRecordsNotFound

// MultipleRecordsFoundException is what Sole returns when the query matched
// more than one row, and it carries how many it saw.
//
// It is an alias of concerns.MultipleRecordsFoundError, not a second type: a
// value built by either package satisfies errors.As under both names. See that
// type for what the count means.
type MultipleRecordsFoundException = concerns.MultipleRecordsFoundError

// NewMultipleRecordsFoundException answers its constructor.
func NewMultipleRecordsFoundException(count int) *MultipleRecordsFoundException {
	return concerns.NewMultipleRecordsFoundError(count)
}

// ErrMultipleColumnsSelected is what Scalar raises when the row it read has
// more than one column.
var ErrMultipleColumnsSelected = errors.New("database: more than one column was selected")

// ErrSQLiteDatabaseDoesNotExist reports that the SQLite file named by the
// configuration is not there.
//
// The path is wrapped into the message rather than carried in a field, because
// a caller that has the error has nothing to do with the path except print it.
var ErrSQLiteDatabaseDoesNotExist = errors.New("database file does not exist")

// NewSQLiteDatabaseDoesNotExistException answers its constructor.
func NewSQLiteDatabaseDoesNotExistException(path string) error {
	return fmt.Errorf("%w: %s. Ensure this is an absolute path to the database", ErrSQLiteDatabaseDoesNotExist, path)
}

// substituteBindings writes the bindings into the placeholders, for a
// message a person reads.
//
// It is only ever used to build a message: nothing here executes what it
// produces, and nothing should. A string built this way is the injection
// this framework exists to make impossible -- it is safe here precisely
// because it goes to a log and not to a server.
func substituteBindings(sql string, bindings []any) string {
	var b strings.Builder
	next := 0

	for i := 0; i < len(sql); i++ {
		if sql[i] != '?' || next >= len(bindings) {
			b.WriteByte(sql[i])
			continue
		}
		b.WriteString(formatBinding(bindings[next]))
		next++
	}
	return b.String()
}

// formatBinding writes one value the way a log wants to read it.
func formatBinding(v any) string {
	switch value := v.(type) {
	case nil:
		return "null"
	case string:
		return "'" + strings.ReplaceAll(value, "'", "''") + "'"
	case bool:
		if value {
			return "1"
		}
		return "0"
	case []byte:
		return "'" + strings.ReplaceAll(string(value), "'", "''") + "'"
	default:
		return fmt.Sprint(value)
	}
}
