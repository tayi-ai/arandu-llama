package database

import (
	"sync"

	"github.com/arandu-io/hesape/database/concerns"
	dbevents "github.com/arandu-io/hesape/database/events"
)

// newConnectionEstablished builds the event the manager dispatches. It is a
// function rather than a call at the site so the events package's Connection
// interface is satisfied in one place.
func newConnectionEstablished(connection *Connection) *dbevents.ConnectionEstablished {
	return dbevents.NewConnectionEstablished(connection)
}

// DatabaseTransactionRecord is one open transaction, and the callbacks waiting
// on how it ends.
type DatabaseTransactionRecord struct {
	// Connection is the connection's name.
	Connection string

	// Level is the transaction's nesting level.
	Level int

	// Parent is the transaction this one was opened inside.
	Parent *DatabaseTransactionRecord

	callbacks            []func()
	callbacksForRollback []func()
}

// NewDatabaseTransactionRecord creates a DatabaseTransactionRecord.
func NewDatabaseTransactionRecord(connection string, level int, parent *DatabaseTransactionRecord) *DatabaseTransactionRecord {
	return &DatabaseTransactionRecord{Connection: connection, Level: level, Parent: parent}
}

// AddCallback registers callback to run after this transaction commits.
func (r *DatabaseTransactionRecord) AddCallback(callback func()) {
	r.callbacks = append(r.callbacks, callback)
}

// AddCallbackForRollback registers callback to run after this transaction
// rolls back.
func (r *DatabaseTransactionRecord) AddCallbackForRollback(callback func()) {
	r.callbacksForRollback = append(r.callbacksForRollback, callback)
}

// ExecuteCallbacks runs every callback registered with AddCallback.
func (r *DatabaseTransactionRecord) ExecuteCallbacks() {
	for _, callback := range r.callbacks {
		callback()
	}
}

// ExecuteCallbacksForRollback runs every callback registered with
// AddCallbackForRollback.
func (r *DatabaseTransactionRecord) ExecuteCallbacksForRollback() {
	for _, callback := range r.callbacksForRollback {
		callback()
	}
}

// GetCallbacks returns the callbacks registered with AddCallback.
func (r *DatabaseTransactionRecord) GetCallbacks() []func() { return r.callbacks }

// GetCallbacksForRollback returns the callbacks registered with
// AddCallbackForRollback.
func (r *DatabaseTransactionRecord) GetCallbacksForRollback() []func() { return r.callbacksForRollback }

// DatabaseTransactionsManager remembers which transactions are open on which
// connection, and runs the callbacks that were waiting for the outermost one to
// commit.
//
// It is what makes AfterCommit mean what it says. A queue job dispatched inside
// a transaction that then rolls back is a job about a row that does not exist,
// and it is the classic way a background worker starts failing on data nobody
// can find.
type DatabaseTransactionsManager struct {
	mu sync.Mutex

	// committedTransactions is the records staged for their callbacks to run
	// once the outermost transaction commits.
	committedTransactions []*DatabaseTransactionRecord

	// pendingTransactions is every open transaction record, across every
	// connection.
	pendingTransactions []*DatabaseTransactionRecord

	// currentTransaction is the innermost open transaction, keyed by
	// connection.
	currentTransaction map[string]*DatabaseTransactionRecord
}

// NewDatabaseTransactionsManager creates a DatabaseTransactionsManager.
func NewDatabaseTransactionsManager() *DatabaseTransactionsManager {
	return &DatabaseTransactionsManager{currentTransaction: map[string]*DatabaseTransactionRecord{}}
}

// Begin records that a new transaction level opened on connection.
func (m *DatabaseTransactionsManager) Begin(connection string, level int) {
	m.mu.Lock()
	defer m.mu.Unlock()

	record := NewDatabaseTransactionRecord(connection, level, m.currentTransaction[connection])
	m.pendingTransactions = append(m.pendingTransactions, record)
	m.currentTransaction[connection] = record
}

// Commit records that a transaction level committed on connection, and runs
// the callbacks staged for it once the outermost transaction is the one
// committing.
//
// The callbacks run only then, which is the whole guarantee: a savepoint
// that commits inside a transaction that later rolls back has not committed
// anything.
func (m *DatabaseTransactionsManager) Commit(connection string, levelBeingCommitted, newTransactionLevel int) {
	m.mu.Lock()

	m.stageTransactionsLocked(connection, levelBeingCommitted)

	if current := m.currentTransaction[connection]; current != nil {
		m.currentTransaction[connection] = current.Parent
	}

	if !m.AfterCommitCallbacksShouldBeExecuted(newTransactionLevel) && newTransactionLevel != 0 {
		m.mu.Unlock()
		return
	}

	m.pendingTransactions = rejectRecords(m.pendingTransactions, func(r *DatabaseTransactionRecord) bool {
		return r.Connection == connection && r.Level >= levelBeingCommitted
	})

	var forThisConnection, forOtherConnections []*DatabaseTransactionRecord
	for _, record := range m.committedTransactions {
		if record.Connection == connection {
			forThisConnection = append(forThisConnection, record)
			continue
		}
		forOtherConnections = append(forOtherConnections, record)
	}
	m.committedTransactions = forOtherConnections

	m.mu.Unlock()

	// Outside the lock: a callback that touches the database would otherwise
	// deadlock against the manager it is being called from.
	for _, record := range forThisConnection {
		record.ExecuteCallbacks()
	}
}

// StageTransactions moves the pending records of a connection at or above a
// level into the committed list.
func (m *DatabaseTransactionsManager) StageTransactions(connection string, levelBeingCommitted int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.stageTransactionsLocked(connection, levelBeingCommitted)
}

func (m *DatabaseTransactionsManager) stageTransactionsLocked(connection string, levelBeingCommitted int) {
	for _, record := range m.pendingTransactions {
		if record.Connection == connection && record.Level >= levelBeingCommitted {
			m.committedTransactions = append(m.committedTransactions, record)
		}
	}
	m.pendingTransactions = rejectRecords(m.pendingTransactions, func(r *DatabaseTransactionRecord) bool {
		return r.Connection == connection && r.Level >= levelBeingCommitted
	})
}

// Rollback records that connection rolled back to newTransactionLevel, and
// runs the rollback callbacks for every level undone.
func (m *DatabaseTransactionsManager) Rollback(connection string, newTransactionLevel int) {
	m.mu.Lock()

	var toRun []*DatabaseTransactionRecord

	if newTransactionLevel == 0 {
		for record := m.currentTransaction[connection]; record != nil; record = record.Parent {
			toRun = append(toRun, record)
		}
		m.currentTransaction[connection] = nil

		m.pendingTransactions = rejectRecords(m.pendingTransactions, func(r *DatabaseTransactionRecord) bool {
			return r.Connection == connection
		})
		m.committedTransactions = rejectRecords(m.committedTransactions, func(r *DatabaseTransactionRecord) bool {
			return r.Connection == connection
		})
	} else {
		m.pendingTransactions = rejectRecords(m.pendingTransactions, func(r *DatabaseTransactionRecord) bool {
			return r.Connection == connection && r.Level > newTransactionLevel
		})

		for current := m.currentTransaction[connection]; current != nil; current = m.currentTransaction[connection] {
			m.removeCommittedTransactionsThatAreChildrenOf(current)
			toRun = append(toRun, current)
			m.currentTransaction[connection] = current.Parent

			if next := m.currentTransaction[connection]; next == nil || next.Level <= newTransactionLevel {
				break
			}
		}
	}

	m.mu.Unlock()

	for _, record := range toRun {
		record.ExecuteCallbacksForRollback()
	}
}

// removeCommittedTransactionsThatAreChildrenOf answers the protected method of
// the same name. The caller holds the lock.
func (m *DatabaseTransactionsManager) removeCommittedTransactionsThatAreChildrenOf(transaction *DatabaseTransactionRecord) {
	var removed, kept []*DatabaseTransactionRecord
	for _, committed := range m.committedTransactions {
		if committed.Connection == transaction.Connection && committed.Parent == transaction {
			removed = append(removed, committed)
			continue
		}
		kept = append(kept, committed)
	}
	m.committedTransactions = kept

	for _, child := range removed {
		m.removeCommittedTransactionsThatAreChildrenOf(child)
	}
}

// AddCallback runs callback after the outermost transaction commits, or now
// when there is none.
func (m *DatabaseTransactionsManager) AddCallback(callback func()) {
	m.mu.Lock()
	current := lastRecord(m.pendingTransactions)
	m.mu.Unlock()

	if current != nil {
		current.AddCallback(callback)
		return
	}
	callback()
}

// AddCallbackForRollback runs callback if the current transaction rolls
// back.
//
// Outside a transaction it does nothing, and that is right: there is no
// rollback coming for a statement that already committed.
func (m *DatabaseTransactionsManager) AddCallbackForRollback(callback func()) {
	m.mu.Lock()
	current := lastRecord(m.pendingTransactions)
	m.mu.Unlock()

	if current != nil {
		current.AddCallbackForRollback(callback)
	}
}

// CallbackApplicableTransactions returns a copy of the pending transaction
// records a callback could be added to.
func (m *DatabaseTransactionsManager) CallbackApplicableTransactions() []*DatabaseTransactionRecord {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]*DatabaseTransactionRecord(nil), m.pendingTransactions...)
}

// AfterCommitCallbacksShouldBeExecuted reports whether level is the
// outermost transaction level, the only one whose commit runs the staged
// callbacks.
func (m *DatabaseTransactionsManager) AfterCommitCallbacksShouldBeExecuted(level int) bool {
	return level == 0
}

// GetPendingTransactions returns a copy of every open transaction record.
func (m *DatabaseTransactionsManager) GetPendingTransactions() []*DatabaseTransactionRecord {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]*DatabaseTransactionRecord(nil), m.pendingTransactions...)
}

// GetCommittedTransactions returns a copy of the records staged to run their
// callbacks.
func (m *DatabaseTransactionsManager) GetCommittedTransactions() []*DatabaseTransactionRecord {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]*DatabaseTransactionRecord(nil), m.committedTransactions...)
}

func rejectRecords(records []*DatabaseTransactionRecord, reject func(*DatabaseTransactionRecord) bool) []*DatabaseTransactionRecord {
	var out []*DatabaseTransactionRecord
	for _, record := range records {
		if !reject(record) {
			out = append(out, record)
		}
	}
	return out
}

func lastRecord(records []*DatabaseTransactionRecord) *DatabaseTransactionRecord {
	if len(records) == 0 {
		return nil
	}
	return records[len(records)-1]
}

// DatabaseTransactionsManager is what ManagesTransactions reports to, and the
// compiler says so here rather than at the call site.
var _ concerns.TransactionsManager = (*DatabaseTransactionsManager)(nil)
