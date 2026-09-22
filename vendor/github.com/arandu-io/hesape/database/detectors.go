package database

import "strings"

// LostConnectionDetector reads a driver error and says whether the connection is
// gone.
//
// It is a list of substrings and nothing cleverer, because every driver reports
// this differently and several report it more than one way. The list is
// deliberately wide: an entry that never matches costs one string comparison,
// and a missed one costs a request.
type LostConnectionDetector struct{}

// NewLostConnectionDetector creates a LostConnectionDetector.
func NewLostConnectionDetector() *LostConnectionDetector { return &LostConnectionDetector{} }

// CausedByLostConnection reports whether err indicates that the connection is
// gone, checking both this detector's message list and the messages the Go
// drivers use.
func (d *LostConnectionDetector) CausedByLostConnection(err error) bool {
	if err == nil {
		return false
	}
	message := err.Error()
	return containsAny(message, lostConnectionMessages) ||
		containsAny(message, goLostConnectionMessages)
}

// goLostConnectionMessages is what the Go drivers say when the connection is
// gone. lostConnectionMessages holds messages in the SQLSTATE-prefixed,
// capitalized style a wrapped C client library reports -- for example "Broken
// pipe" -- while pgx, go-sql-driver and modernc/sqlite write their own, in
// Go's house style: lower case, no SQLSTATE prefix, so "broken pipe" is what a
// Go program actually reads.
//
// The consequence of getting this wrong is not cosmetic: a lost connection that
// goes unrecognised is a query that is not retried and a transaction level that
// is never reset, so the next request on that connection begins inside a
// transaction that no longer exists.
var goLostConnectionMessages = []string{
	// database/sql's own signal that the pool handed out a dead connection.
	"driver: bad connection",
	"sql: connection is already closed",
	"sql: database is closed",

	// The net package, which is where all three drivers' failures end up.
	"broken pipe",
	"connection reset by peer",
	"use of closed network connection",
	"connection refused",
	"i/o timeout",
	"no route to host",
	"network is unreachable",
	"unexpected EOF",

	// pgx and go-sql-driver, in their own words.
	"conn closed",
	"invalid connection",
	"unexpected read from socket",
	"failed to connect to `host=",
	"terminating connection due to administrator command",
}

// CausedByLostConnection reports whether an error means the connection is gone.
//
// It reads the package's own LostConnectionDetector; there is no detector to
// register and none to swap.
func CausedByLostConnection(err error) bool {
	return (&LostConnectionDetector{}).CausedByLostConnection(err)
}

// ConcurrencyErrorDetector reads a driver error and says whether it was a
// deadlock or a serialization failure.
type ConcurrencyErrorDetector struct{}

// NewConcurrencyErrorDetector creates a ConcurrencyErrorDetector.
func NewConcurrencyErrorDetector() *ConcurrencyErrorDetector { return &ConcurrencyErrorDetector{} }

// CausedByConcurrencyError reports whether err indicates a deadlock or a
// serialization failure.
//
// database/sql carries no error code, and 40001 is the SQLSTATE for a
// serialization failure, so the check looks for that string in the message
// text -- which is where pgx and go-sql-driver both put it.
func (d *ConcurrencyErrorDetector) CausedByConcurrencyError(err error) bool {
	if err == nil {
		return false
	}
	message := err.Error()
	if strings.Contains(message, "40001") {
		return true
	}
	return containsAny(message, concurrencyErrorMessages)
}

// CausedByConcurrencyError reports whether an error means a deadlock or a
// serialization failure.
//
// It reads the package's own ConcurrencyErrorDetector; there is no detector to
// register and none to swap.
func CausedByConcurrencyError(err error) bool {
	return (&ConcurrencyErrorDetector{}).CausedByConcurrencyError(err)
}

func containsAny(haystack string, needles []string) bool {
	for _, needle := range needles {
		if strings.Contains(haystack, needle) {
			return true
		}
	}
	return false
}

// concurrencyErrorMessages is ConcurrencyErrorDetector's own list.
var concurrencyErrorMessages = []string{
	"Deadlock found when trying to get lock",
	"deadlock detected",
	"The database file is locked",
	"database is locked",
	"database table is locked",
	"A table in the database is locked",
	"has been chosen as the deadlock victim",
	"Lock wait timeout exceeded; try restarting transaction",
	"WSREP detected deadlock/conflict and aborted the transaction. Try restarting the transaction",
	"Record has changed since last read in table",
}

// lostConnectionMessages is LostConnectionDetector's own list.
var lostConnectionMessages = []string{
	"server has gone away",
	"Server has gone away",
	"no connection to the server",
	"Lost connection",
	"is dead or not enabled",
	"Error while sending",
	"decryption failed or bad record mac",
	"server closed the connection unexpectedly",
	"SSL connection has been closed unexpectedly",
	"Error writing data to the connection",
	"Resource deadlock avoided",
	"Transaction() on null",
	"child connection forced to terminate due to client_idle_limit",
	"query_wait_timeout",
	"reset by peer",
	"Physical connection is not usable",
	"TCP Provider: Error code 0x68",
	"ORA-03114",
	"Packets out of order. Expected",
	"Adaptive Server connection failed",
	"Communication link failure",
	"connection is no longer usable",
	"Login timeout expired",
	"SQLSTATE[HY000] [2002] Connection refused",
	"running with the --read-only option so it cannot execute this statement",
	"The connection is broken and recovery is not possible. The connection is marked by the client driver as unrecoverable. No attempt was made to restore the connection.",
	"SQLSTATE[HY000] [2002] php_network_getaddresses: getaddrinfo failed: Try again",
	"SQLSTATE[HY000] [2002] php_network_getaddresses: getaddrinfo failed: Name or service not known",
	"SQLSTATE[HY000] [2002] php_network_getaddresses: getaddrinfo for",
	"SQLSTATE[HY000]: General error: 7 SSL SYSCALL error: EOF detected",
	"SSL error: unexpected eof",
	"SQLSTATE[HY000] [2002] Connection timed out",
	"SSL: Connection timed out",
	"SQLSTATE[HY000]: General error: 1105 The last transaction was aborted due to Seamless Scaling. Please retry.",
	"Temporary failure in name resolution",
	"SQLSTATE[08S01]: Communication link failure",
	"SQLSTATE[08006] [7] could not connect to server: Connection refused Is the server running on host",
	"SQLSTATE[HY000]: General error: 7 SSL SYSCALL error: No route to host",
	"The client was disconnected by the server because of inactivity. See wait_timeout and interactive_timeout for configuring this behavior.",
	"SQLSTATE[08006] [7] could not translate host name",
	"TCP Provider: Error code 0x274C",
	"SQLSTATE[HY000] [2002] No such file or directory",
	"SSL: Operation timed out",
	"Reason: Server is in script upgrade mode. Only administrator can connect at this time.",
	"Unknown $curl_error_code: 77",
	"SSL: Handshake timed out",
	"SSL error: sslv3 alert unexpected message",
	"unrecognized SSL error code:",
	"SQLSTATE[HY000] [1045] Access denied for user",
	"SQLSTATE[HY000] [2002] No connection could be made because the target machine actively refused it",
	"SQLSTATE[HY000] [2002] A connection attempt failed because the connected party did not properly respond after a period of time, or established connection failed because connected host has failed to respond",
	"SQLSTATE[HY000] [2002] Network is unreachable",
	"SQLSTATE[HY000] [2002] The requested address is not valid in its context",
	"SQLSTATE[HY000] [2002] A socket operation was attempted to an unreachable network",
	"SQLSTATE[HY000] [2002] Operation now in progress",
	"SQLSTATE[HY000] [2002] Operation in progress",
	"SQLSTATE[HY000]: General error: 3989",
	"went away",
	"No such file or directory",
	"server is shutting down",
	"failed to connect to",
	"Channel connection is closed",
	"Connection lost",
	"Broken pipe",
	"SQLSTATE[25006]: Read only sql transaction: 7",
	"vtgate connection error: no healthy endpoints",
	"primary is not serving, there may be a reparent operation in progress",
	"current keyspace is being resharded",
	"no healthy tablet available",
	"transaction pool connection limit exceeded",
	"SSL operation failed with code 5",
}
