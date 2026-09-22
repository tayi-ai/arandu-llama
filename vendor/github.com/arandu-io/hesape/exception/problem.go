package exception

import (
	"encoding/json"
	"net/http"
)

// ProblemContentType is the media type a problem document is served as.
//
// It is not application/json. A client that understands problem documents
// recognises this one and knows every member below without being told, and one
// that does not still parses it, because the "+json" suffix says how.
const ProblemContentType = "application/problem+json"

// blankProblemType is the type of a problem whose status code is the whole of
// what went wrong.
//
// A type URI is a public promise: name one and it becomes a string clients
// match on, which cannot change without breaking them. Nothing here has earned
// one, so nothing here claims one, and Problem.Type stays open for an
// application that wants to name its own.
const blankProblemType = "about:blank"

// Problem is the body of an error response: the problem details document of
// RFC 9457.
//
// It is one shape for every failure answered as JSON, which is what a client
// gets to rely on. A client reads Status to decide what to do, Title to know
// which failure it is, Detail to show somebody, and RequestID to quote when it
// asks what happened.
//
// The members are the ones the RFC names, plus RequestID, which it allows as an
// extension. Detail is the only one written for a person, and it is the one
// that carries nothing the caller was not allowed to see.
type Problem struct {
	// Type names the class of failure as a URI, and is "about:blank" when the
	// status code says all there is to say. A client that matches on it gets
	// the same answer for every problem written here, which is why it should
	// match on Status instead.
	Type string `json:"type"`

	// Title is the short, stable name of the failure: "Not Found", "Page
	// Expired". It does not change between occurrences of the same status, so
	// it is the member to group by.
	Title string `json:"title"`

	// Status is the HTTP status, repeated in the body so that a document
	// separated from its response -- logged, forwarded, stored -- still says
	// what it was.
	Status int `json:"status"`

	// Detail is the sentence written for the person reading it, and it is
	// specific to this occurrence. It is empty when there is nothing to add to
	// the title.
	Detail string `json:"detail,omitempty"`

	// Instance is the address the failure happened at, as a URI reference.
	Instance string `json:"instance,omitempty"`

	// RequestID ties the document to the log line that holds the cause. It is
	// the extension member, and it is the one thing here worth quoting in a
	// support conversation: the detail is deliberately vague, and this is not.
	RequestID string `json:"request_id,omitempty"`
}

// WriteProblem answers the request with a problem document, and is the only
// shape a JSON failure leaves this application in.
//
// detail is shown to the caller, so it carries what was written for them and
// never what was written for the log: a driver's text, a policy's reason, the
// contents of a dereference. The response is marked not to be cached, because a
// refusal is one person's.
//
// It writes the status and the body, so nothing may write to w afterwards.
func WriteProblem(w http.ResponseWriter, r *http.Request, status int, detail string) {
	body := Problem{
		Type:      blankProblemType,
		Title:     statusTitle(status),
		Status:    status,
		Detail:    detail,
		Instance:  instanceOf(r),
		RequestID: requestID(w, r),
	}

	w.Header().Set("Content-Type", ProblemContentType)
	// A refusal is one person's, and must never be held by a cache shared
	// between people.
	w.Header().Set("Cache-Control", "no-store, private")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}

// instanceOf is the address the failure happened at: the escaped path of the
// request, and nothing after it.
//
// The query string is left out. It is written by the client, it is the half of
// the address that carries values, and a body that echoes it hands those values
// to whoever reads the response -- which, for a link somebody shared, is not
// who sent them.
func instanceOf(r *http.Request) string {
	if r.URL == nil {
		return ""
	}
	return r.URL.EscapedPath()
}
