package failed

import (
	"context"
	"fmt"
	"time"

	"github.com/arandu-io/hesape/auth"
)

// NullFailedJobProvider accepts every failure and keeps none of them.
//
// It is what an application wires when the dead letter list lives somewhere
// else entirely --
// an error tracker, a log pipeline -- and keeping a second copy in the database
// would be a table nobody reads.
//
// It is a value and not a pointer, for the reason NullQueue is: a nil pointer
// that silently swallows every failure is a worse mistake than the one this
// type exists to make cheap.
type NullFailedJobProvider struct{}

var (
	_ FailedJobProvider          = NullFailedJobProvider{}
	_ CountableFailedJobProvider = NullFailedJobProvider{}
)

// Log discards the failure and returns no id.
func (NullFailedJobProvider) Log(_ context.Context, g auth.Grant, _ FailedJob) (string, error) {
	if _, err := tenantOf(g); err != nil {
		return "", err
	}
	return "", nil
}

// IDs is empty.
func (NullFailedJobProvider) IDs(context.Context, auth.Grant, string) ([]string, error) {
	return nil, nil
}

// All is empty.
func (NullFailedJobProvider) All(context.Context, auth.Grant) ([]FailedJob, error) {
	return nil, nil
}

// Find never finds anything.
func (NullFailedJobProvider) Find(_ context.Context, _ auth.Grant, id string) (FailedJob, error) {
	return FailedJob{}, fmt.Errorf("%w: %s", ErrNotFound, id)
}

// Forget has nothing to forget, and reports true: the caller asked for the job
// to be gone, and it is.
func (NullFailedJobProvider) Forget(context.Context, auth.Grant, string) (bool, error) {
	return true, nil
}

// Flush has nothing to flush.
func (NullFailedJobProvider) Flush(context.Context, auth.Grant, time.Duration) error { return nil }

// Count is zero.
func (NullFailedJobProvider) Count(context.Context, auth.Grant, string, string) (int, error) {
	return 0, nil
}
