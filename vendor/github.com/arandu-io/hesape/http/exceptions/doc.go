// Package exceptions holds the exception types an HTTP response can carry:
// HttpResponseException, MalformedUrlException, OriginMismatchException,
// PostTooLargeException and ThrottleRequestsException.
//
// # Constructors, and the shared struct that stands in for a hierarchy
//
// Each type has a NewX function rather than a constructor: Go has no
// constructors, and the zero value of these types would be a 0 status,
// which is not an answer.
//
// PostTooLargeException, ThrottleRequestsException and MalformedUrlException
// all carry an HTTP status the way a common base class would. Go has no
// class inheritance, so [HTTPError] is a struct they embed instead, and
// errors.As reaches it through any of them.
package exceptions
