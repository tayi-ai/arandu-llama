// Package exceptions holds the error types a route, a signed URL or a
// streamed response can fail with: an invalid signature, a missing rate
// limiter, a route URL that is missing required parameters, a backed-enum
// parameter that matched no case, and a streamed response that failed after
// the headers were already sent.
package exceptions
