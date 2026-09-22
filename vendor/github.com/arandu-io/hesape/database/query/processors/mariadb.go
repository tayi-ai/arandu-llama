package processors

import "github.com/arandu-io/hesape/database/query"

// MariaDBProcessor is the Processor for MariaDB.
//
// It adds nothing to MySQL's: MariaDB differs from MySQL in what it accepts,
// not in what it returns. It exists so that a
// connection wires a grammar and a processor by the same name, and so that the
// place to put the first difference already exists.
type MariaDBProcessor struct {
	MySQLProcessor
}

var _ query.Processor = (*MariaDBProcessor)(nil)

// NewMariaDBProcessor creates a MariaDBProcessor.
func NewMariaDBProcessor() *MariaDBProcessor { return &MariaDBProcessor{} }
