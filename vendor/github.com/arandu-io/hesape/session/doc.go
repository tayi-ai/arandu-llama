// Package session issues sessions, carries the flash across the redirect that
// follows a rejected form, and mints the CSRF token bound to the session.
//
// # What is here
//
//   - [Store]: one session, loaded for one request, holding a bag of keys.
//     The flash, the old input, the CSRF token and the previous URL live in
//     it, which is to say everything a page needs to redisplay the old
//     input and validation errors of a rejected form.
//   - [SessionHandler] and the six handlers that implement it:
//     [ArraySessionHandler], [NullSessionHandler], [FileSessionHandler],
//     [CookieSessionHandler], [CacheBasedSessionHandler] and
//     [DatabaseSessionHandler].
//   - [EncryptedStore], the same session sealed before it reaches the handler.
//   - [SessionManager], which builds one from configuration.
//   - [RecordStore], the other kind of session store -- see below.
//   - [Flash], the one-shot signed cookie that carries the messages and the
//     typed input of a rejected request even when there is no session at all.
//   - [CSRF], the double-submit token bound to the session id.
//
// The middleware is github.com/arandu-io/hesape/session/middleware:
// StartSession and AuthenticateSession. The generator is
// github.com/arandu-io/hesape/session/console: SessionTableCommand.
//
// [ErrTokenMismatch] is a sentinel error: no fields, no methods, checked
// with errors.Is.
//
// # Two stores, and which one to reach for
//
// [Store] holds string keys in dot notation with values of any type, plus
// flash and old input. Reach for it for anything a page draws.
//
// [RecordStore] is the store for signing the cookie, minting ids, reading
// one typed [Record] back, and ending every session of a subject at once.
// It is generic over the payload, so what auth keeps about who is signed in
// is checked by the compiler rather than asserted out of a map. They share
// [Handler] and [Record]; nothing else is duplicated.
//
// # Two flashes, and which one to reach for
//
// [Store.Flash] is the session one: it needs a session, and it survives exactly
// one request because [Store.Save] ages it.
//
// [Flash] is a signed cookie with no session behind it, and it exists for the
// three screens that need the messages most -- sign in, sign up, password reset
// -- which are submitted by somebody who has no session yet. Cleared on the read
// rather than aged, so a message cannot appear on a page nobody submitted.
package session
