package collections

import (
	"errors"
	"fmt"
)

// ErrItemNotFound is returned by Sole and FirstOrFail when no item passes the
// filter.
var ErrItemNotFound = errors.New("collections: item not found")

// ErrMultipleItemsFound reports that more than one item passed a filter that
// admits only one. Sole returns a MultipleItemsFoundError, which unwraps to
// this sentinel and carries how many were found.
var ErrMultipleItemsFound = errors.New("collections: multiple items found")

// MultipleItemsFoundError is the error Sole returns when more than one item
// passes the filter, carrying how many did.
//
// It unwraps to ErrMultipleItemsFound, so errors.Is keeps working on the
// sentinel and errors.As reaches the count.
type MultipleItemsFoundError struct {
	// Count is how many items passed the filter.
	Count int
}

// GetCount returns how many items passed the filter.
func (e *MultipleItemsFoundError) GetCount() int { return e.Count }

// Error renders the sentinel's message followed by the count.
func (e *MultipleItemsFoundError) Error() string {
	return fmt.Sprintf("%v: %d items were found", ErrMultipleItemsFound, e.Count)
}

// Unwrap reports ErrMultipleItemsFound, so that errors.Is matches the sentinel.
func (e *MultipleItemsFoundError) Unwrap() error { return ErrMultipleItemsFound }

// ErrInvalidArgument reports an argument a collection operation cannot honour:
// Nth, Split, SplitIn, Sliding and Random return it.
var ErrInvalidArgument = errors.New("collections: invalid argument")

// ErrUnexpectedValue reports a callback that returned something the operation
// cannot use: ReduceSpread and Ensure return it.
var ErrUnexpectedValue = errors.New("collections: unexpected value")
