package events

// InvokeQueuedClosure runs a QueuedClosure once the queue picks the job up.
//
// It is the listener class the job names: the closure travels as the job's
// data, and this is what calls it.
type InvokeQueuedClosure struct{}

// Handle calls the closure with the arguments the event was dispatched with.
func (InvokeQueuedClosure) Handle(closure any, arguments []any) any {
	return callFunc(closure, arguments)
}

// Failed handles a job failure, calling every catch callback with the arguments
// and, last, the failure.
//
// The first parameter is the closure the job carries, which this never reads: it
// is taken because the job's data is positional and the closure is the first of
// it.
func (InvokeQueuedClosure) Failed(_ any, arguments []any, catchCallbacks []any, err error) {
	arguments = append(append([]any(nil), arguments...), err)

	for _, callback := range catchCallbacks {
		callFunc(callback, arguments)
	}
}
