package arr

import "github.com/arandu-io/hesape/collections"

// The sentinels are the ones the parent package already declares, not copies
// of them. Sole here and Sole there fail for the same reason, and a caller
// reaching for errors.Is should not have to know which of the two produced the
// error. Assigning the value rather than restating it keeps errors.Is matching
// across both names.
var (
	// ErrInvalidArgument reports an argument this package cannot honour:
	// Random, Array, From and the typed readers return it.
	ErrInvalidArgument = collections.ErrInvalidArgument

	// ErrItemNotFound is returned by Sole when nothing matches.
	ErrItemNotFound = collections.ErrItemNotFound

	// ErrMultipleItemsFound reports that more than one element matched where
	// only one was admitted. Sole returns a
	// collections.MultipleItemsFoundError, which unwraps to this sentinel.
	ErrMultipleItemsFound = collections.ErrMultipleItemsFound
)
