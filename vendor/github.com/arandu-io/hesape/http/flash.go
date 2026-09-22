package http

// Old is the value that was typed into the form that was rejected, so the
// page it was sent back to can fill the boxes again. With no key, returns
// all of the old input; with a key, returns the value at that key or the
// default.
//
// It is what makes a form come back pre-filled after a rejection, and its
// pair with [RedirectResponse.WithInput] is the two ends of one thing:
// WithInput writes into the session's old input, and Old reads it. If the
// two diverge, the form comes back blank, which is the bug this exists to
// not have.
//
// Returns the default when no session is set.
func (r *Request) Old(key string, def ...any) any {
	if r.session == nil {
		if len(def) > 0 {
			return def[0]
		}
		return nil
	}
	value := r.session.GetOldInput(key, def...)
	if value == nil && len(def) > 0 {
		return def[0]
	}
	return value
}

// Flash flashes the request's input to the session, so the next request can
// read it as old input.
//
// Panics when no session is set. A handler that calls this is inside a flow
// that started with a session, and a missing one there is a wiring error the
// caller should see immediately.
func (r *Request) Flash() {
	r.Session().FlashInput(r.inputMap())
}

// FlashOnly flashes only some of the input.
func (r *Request) FlashOnly(keys ...string) {
	r.Session().FlashInput(r.Only(keys...))
}

// FlashExcept flashes the input except the given keys.
//
// The keys are the form's own reason to drop a field -- a long body already
// saved elsewhere, a step the next page recomputes. A credential is not that
// reason and does not need naming here: the session store drops the secret
// fields whatever the caller passes.
func (r *Request) FlashExcept(keys ...string) {
	r.Session().FlashInput(r.Except(keys...))
}

// Flush removes all of the old input from the session.
func (r *Request) Flush() {
	r.Session().FlashInput(map[string]any{})
}
