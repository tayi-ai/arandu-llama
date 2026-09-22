package concerns

import (
	"errors"
	"fmt"
	"strconv"
)

// TransactionDriver is what ManagesTransactions drives.
//
// Connection reaches straight for its pool, its query grammar and its own
// lost-connection detection. Those calls are written out here as an
// interface, which is the Go spelling of "this embedded type may only be
// used by a type that has these".
//
// A Connection satisfies it; nothing else is meant to.
type TransactionDriver interface {
	// GetName returns the connection's name. The transactions manager keys
	// its bookkeeping by it.
	GetName() string

	// ExecuteBeginTransactionStatement issues a BEGIN, pinning a connection
	// for the life of the transaction.
	ExecuteBeginTransactionStatement() error

	// CommitTransactionStatement issues a COMMIT.
	CommitTransactionStatement() error

	// RollBackTransactionStatement issues a ROLLBACK, and reports false when
	// there was no transaction open to roll back.
	RollBackTransactionStatement() (bool, error)

	// ExecuteSavepointStatement runs the statement the savepoint paths build.
	ExecuteSavepointStatement(sql string) error

	// SupportsSavepoints reports whether the query grammar supports
	// savepoints.
	SupportsSavepoints() bool

	// CompileSavepoint returns the statement that creates a savepoint.
	CompileSavepoint(name string) string

	// CompileSavepointRollBack returns the statement that rolls back to a
	// savepoint.
	CompileSavepointRollBack(name string) string

	// FireConnectionEvent dispatches the transaction event named by event:
	// one of "beganTransaction", "committing", "committed" or "rollingBack".
	FireConnectionEvent(event string)

	// CausedByConcurrencyError reports whether err means a deadlock or a
	// serialization failure.
	CausedByConcurrencyError(err error) bool

	// CausedByLostConnection reports whether err means the connection is
	// gone.
	CausedByLostConnection(err error) bool

	// Reconnect replaces the pool.
	Reconnect() error

	// ReconnectIfMissingConnection reconnects when the connection has no
	// pool yet.
	ReconnectIfMissingConnection() error
}

// TransactionsManager is what ManagesTransactions reports to.
//
// It is narrowed to the five methods this file calls. It is an interface here
// and a concrete type in the database package for the usual reason: that package
// imports this one.
type TransactionsManager interface {
	// Begin records that a new transaction level opened on connection.
	Begin(connection string, level int)

	// Commit records that a transaction level committed on connection.
	Commit(connection string, levelBeingCommitted, newTransactionLevel int)

	// Rollback records that connection rolled back to newTransactionLevel.
	Rollback(connection string, newTransactionLevel int)

	// AddCallback registers a callback to run after the outermost commit.
	AddCallback(callback func())

	// AddCallbackForRollback registers a callback to run after a rollback.
	AddCallbackForRollback(callback func())
}

// DeadlockError reports that a nested transaction hit a concurrency error, and
// the whole transaction is gone.
//
// It is declared here because ManagesTransactions is what raises it and the
// database package imports this one, so declaring it there would close the
// cycle. database.DeadlockException is an alias of this type -- one type under
// two names.
//
// It is not retried, and that is the point of having its own type: on a
// deadlock the engine has already rolled the whole transaction back, so
// re-running the statement runs it outside the transaction the caller thinks
// it is in.
type DeadlockError struct {
	// Err is the driver error the engine reported.
	Err error
}

// NewDeadlockError wraps the driver error that reported a deadlock.
func NewDeadlockError(err error) *DeadlockError { return &DeadlockError{Err: err} }

// Error carries the driver's own message through unchanged.
func (e *DeadlockError) Error() string {
	if e.Err == nil {
		return "deadlock"
	}
	return e.Err.Error()
}

// Unwrap makes errors.Is and errors.As reach the driver error.
func (e *DeadlockError) Unwrap() error { return e.Err }

// ErrNoTransactionsManager is the error AfterCommit and
// AfterRollBack raise when no manager was set.
var ErrNoTransactionsManager = errors.New("Transactions Manager has not been set.")

// ManagesTransactions is the transaction half of a connection: the nesting
// level, the transactions manager, and the hooks that run before one starts.
//
// It is a struct Connection embeds, so the state it owns is the connection's
// state.
//
// # Savepoints exist here and are not the framework's transaction story
//
// The nested transaction below opens a savepoint. That is not the path an
// Arandu application takes: database.Transaction joins the outer transaction
// rather than nesting, on the grounds that partial rollback is a second failure
// mode for one operation. The generated repository uses database.Transaction.
type ManagesTransactions struct {
	// driver is the connection this trait was used by.
	driver TransactionDriver

	// transactions is the nesting level.
	transactions int

	// transactionsManager records transaction lifecycle events.
	transactionsManager TransactionsManager

	// beforeStartingTransaction is the hooks that run just before a
	// transaction opens.
	beforeStartingTransaction []func()
}

// UseTransactions wires ManagesTransactions to the connection that embeds it.
//
// A Go struct has to be told which driver it belongs to, so the connection
// calls this once from its constructor.
func (m *ManagesTransactions) UseTransactions(driver TransactionDriver) {
	m.driver = driver
}

// Transaction runs callback inside a transaction, committing when it returns
// nil and rolling back when it returns an error.
//
// attempts bounds the retry count, and it retries only what is worth
// retrying: a concurrency error at the outermost level. Anything else is
// returned on the first try, because re-running a statement that failed on a
// constraint just fails again, slower.
//
// A Go method cannot be generic, so the callback returns only an error and
// carries its result out through the closure -- which is the shape
// database.Transaction already has.
func (m *ManagesTransactions) Transaction(callback func() error, attempts int) error {
	if attempts < 1 {
		attempts = 1
	}

	for currentAttempt := 1; currentAttempt <= attempts; currentAttempt++ {
		if err := m.BeginTransaction(); err != nil {
			return err
		}

		// An error from the callback is handled below, and a retryable one
		// continues the loop.
		if err := callback(); err != nil {
			if retry, handled := m.handleTransactionException(err, currentAttempt, attempts); !retry {
				return handled
			}
			continue
		}

		levelBeingCommitted := m.transactions

		if err := m.commitCurrent(); err != nil {
			if retry, handled := m.handleCommitTransactionException(err, currentAttempt, attempts); !retry {
				return handled
			}
			continue
		}

		if m.transactionsManager != nil {
			m.transactionsManager.Commit(m.driver.GetName(), levelBeingCommitted, m.transactions)
		}

		m.driver.FireConnectionEvent("committed")

		return nil
	}

	// Every attempt asked for a retry and the loop ran out. Falling off the
	// end here and returning nil would be a bug nobody could explain, so this
	// says what happened instead.
	return fmt.Errorf("the transaction was retried %d times and every attempt hit a concurrency error", attempts)
}

// commitCurrent is the commit half of the transaction loop: the COMMIT
// statement at level one, and the level decrement at every level.
func (m *ManagesTransactions) commitCurrent() error {
	if m.transactions == 1 {
		m.driver.FireConnectionEvent("committing")
		if err := m.driver.CommitTransactionStatement(); err != nil {
			return err
		}
	}
	m.transactions = max(0, m.transactions-1)
	return nil
}

// handleTransactionException decides what to do after callback returned err.
//
// It returns (retry, err): retry true means the loop tries again, and err is
// what to return when it does not.
func (m *ManagesTransactions) handleTransactionException(err error, currentAttempt, maxAttempts int) (bool, error) {
	// On a deadlock the engine has rolled the entire transaction back, so a
	// nested level cannot be retried in place: it has to reach the caller.
	if m.driver.CausedByConcurrencyError(err) && m.transactions > 1 {
		m.transactions--

		if m.transactionsManager != nil {
			m.transactionsManager.Rollback(m.driver.GetName(), m.transactions)
		}

		return false, NewDeadlockError(err)
	}

	if rollbackErr := m.RollBack(nil); rollbackErr != nil {
		return false, rollbackErr
	}

	if m.driver.CausedByConcurrencyError(err) && currentAttempt < maxAttempts {
		return true, nil
	}

	return false, err
}

// handleCommitTransactionException decides what to do after a commit failed
// with err, the same way handleTransactionException does for the callback.
func (m *ManagesTransactions) handleCommitTransactionException(err error, currentAttempt, maxAttempts int) (bool, error) {
	m.transactions = max(0, m.transactions-1)

	if m.driver.CausedByConcurrencyError(err) && currentAttempt < maxAttempts {
		return true, nil
	}

	if m.driver.CausedByLostConnection(err) {
		// The connection is gone, so the level it was counting is gone with it.
		m.transactions = 0
	}

	return false, err
}

// BeginTransaction opens a new transaction, or a nested savepoint if one is
// already open.
func (m *ManagesTransactions) BeginTransaction() error {
	for _, callback := range m.beforeStartingTransaction {
		callback()
	}

	if err := m.createTransaction(); err != nil {
		return err
	}

	m.transactions++

	if m.transactionsManager != nil {
		m.transactionsManager.Begin(m.driver.GetName(), m.transactions)
	}

	m.driver.FireConnectionEvent("beganTransaction")

	return nil
}

// createTransaction opens a real transaction at level zero, and a savepoint
// above it.
func (m *ManagesTransactions) createTransaction() error {
	if m.transactions == 0 {
		if err := m.driver.ReconnectIfMissingConnection(); err != nil {
			return err
		}
		if err := m.driver.ExecuteBeginTransactionStatement(); err != nil {
			return m.handleBeginTransactionException(err)
		}
		return nil
	}

	if m.transactions >= 1 && m.driver.SupportsSavepoints() {
		return m.createSavepoint()
	}
	return nil
}

// createSavepoint issues a savepoint for the level about to be entered.
//
// The name is "trans" plus that level, which is what makes performRollBack
// able to name the one it wants.
func (m *ManagesTransactions) createSavepoint() error {
	return m.driver.ExecuteSavepointStatement(
		m.driver.CompileSavepoint("trans" + strconv.Itoa(m.transactions+1)),
	)
}

// handleBeginTransactionException reconnects and reissues the BEGIN once
// when err was caused by a lost connection; anything else is returned.
func (m *ManagesTransactions) handleBeginTransactionException(err error) error {
	if m.driver.CausedByLostConnection(err) {
		if reconnectErr := m.driver.Reconnect(); reconnectErr != nil {
			return reconnectErr
		}
		return m.driver.ExecuteBeginTransactionStatement()
	}
	return err
}

// Commit commits the current transaction level, for a transaction somebody
// began by hand rather than through Transaction.
func (m *ManagesTransactions) Commit() error {
	if m.TransactionLevel() == 1 {
		m.driver.FireConnectionEvent("committing")
		if err := m.driver.CommitTransactionStatement(); err != nil {
			return err
		}
	}

	levelBeingCommitted := m.transactions
	m.transactions = max(0, m.transactions-1)

	if m.transactionsManager != nil {
		m.transactionsManager.Commit(m.driver.GetName(), levelBeingCommitted, m.transactions)
	}

	m.driver.FireConnectionEvent("committed")

	return nil
}

// RollBack rolls the transaction back to toLevel, or back one level when
// toLevel is nil. A level outside the open range is ignored rather than
// refused: rolling back to a level that does not exist is a no-op, not a
// failure.
func (m *ManagesTransactions) RollBack(toLevel *int) error {
	level := m.transactions - 1
	if toLevel != nil {
		level = *toLevel
	}

	if level < 0 || level >= m.transactions {
		return nil
	}

	if err := m.performRollBack(level); err != nil {
		return m.handleRollBackException(err)
	}

	m.transactions = level

	if m.transactionsManager != nil {
		m.transactionsManager.Rollback(m.driver.GetName(), m.transactions)
	}

	m.driver.FireConnectionEvent("rollingBack")

	return nil
}

// performRollBack issues the ROLLBACK or the savepoint rollback that reaches
// toLevel.
func (m *ManagesTransactions) performRollBack(toLevel int) error {
	if toLevel == 0 {
		_, err := m.driver.RollBackTransactionStatement()
		return err
	}
	if m.driver.SupportsSavepoints() {
		return m.driver.ExecuteSavepointStatement(
			m.driver.CompileSavepointRollBack("trans" + strconv.Itoa(toLevel+1)),
		)
	}
	return nil
}

// handleRollBackException resets the transaction level to zero when err was
// caused by a lost connection: a rollback that failed because the connection
// went away leaves nothing to roll back to.
func (m *ManagesTransactions) handleRollBackException(err error) error {
	if m.driver.CausedByLostConnection(err) {
		m.transactions = 0

		if m.transactionsManager != nil {
			m.transactionsManager.Rollback(m.driver.GetName(), m.transactions)
		}
	}
	return err
}

// TransactionLevel reports how many transactions are open, counting
// savepoints.
func (m *ManagesTransactions) TransactionLevel() int { return m.transactions }

// AfterCommit runs callback once the outermost transaction commits, or now
// when there is none open.
func (m *ManagesTransactions) AfterCommit(callback func()) error {
	if m.transactionsManager != nil {
		m.transactionsManager.AddCallback(callback)
		return nil
	}
	return ErrNoTransactionsManager
}

// AfterRollBack runs callback when the current transaction rolls back.
func (m *ManagesTransactions) AfterRollBack(callback func()) error {
	if m.transactionsManager != nil {
		m.transactionsManager.AddCallbackForRollback(callback)
		return nil
	}
	return ErrNoTransactionsManager
}

// BeforeStartingTransaction registers a hook to run just before a
// transaction opens.
//
// It lives here rather than on the connection because the slice it appends
// to is read by BeginTransaction, which is here too.
func (m *ManagesTransactions) BeforeStartingTransaction(callback func()) {
	m.beforeStartingTransaction = append(m.beforeStartingTransaction, callback)
}

// SetTransactionManager sets the manager that records transaction lifecycle
// events.
func (m *ManagesTransactions) SetTransactionManager(manager TransactionsManager) {
	m.transactionsManager = manager
}

// UnsetTransactionManager clears the transactions manager.
func (m *ManagesTransactions) UnsetTransactionManager() { m.transactionsManager = nil }

// ResetTransactionLevel sets the nesting level back to zero.
//
// Replacing the handle throws away whatever transaction was open on the old
// one, and a level that outlived its connection is a rollback aimed at nothing.
func (m *ManagesTransactions) ResetTransactionLevel() { m.transactions = 0 }
