package filesystem

import (
	"errors"
	"fmt"
	"path"
	"strings"

	"github.com/arandu-io/hesape/auth"
)

// ErrNoTenant is returned when the Grant carries no tenant, or carries one that
// cannot be a path segment.
var ErrNoTenant = errors.New("filesystem: the Grant carries no tenant, and a file without one belongs to everybody")

// ErrBadKey is returned for a key that would escape its tenant's prefix.
var ErrBadKey = errors.New("filesystem: the key escapes the tenant prefix")

// Key builds the stored path for a key: <tenant>/<key>.
//
// Every [Disk] method calls it, which is what makes tenant isolation a property
// of this package rather than of each [Adapter] implementation remembering. An
// Adapter is handed the result and never sees the Grant, so there is no driver
// in which the prefix can be forgotten.
//
// The zero Grant carries no tenant, so it reaches no file. That is the same
// answer auth.Grant.Check gives, arrived at by a different route: a caller who
// authorized nothing has nothing to build a path out of.
func Key(g auth.Grant, key string) (string, error) {
	tenant := auth.Tenant(g)
	if tenant == "" {
		return "", ErrNoTenant
	}
	// The key is checked below, and for a long time the tenant was not -- so
	// tenant "acme/reports" with key "q1.pdf" and tenant "acme" with key
	// "reports/q1.pdf" resolved to the same object, each holding a valid Grant.
	// No Policy was violated: the path is built after the Policy runs.
	//
	// auth.SystemGrant refuses an invalid tenant now, but a Grant built by
	// auth.Authorize carries whatever the session holds, so the check belongs
	// here too -- this is the line that turns a tenant into a namespace.
	if !auth.ValidTenant(tenant) {
		return "", fmt.Errorf("%w: %q cannot be a path segment", ErrNoTenant, tenant)
	}
	clean, err := CleanKey(key)
	if err != nil {
		return "", err
	}
	return tenant + "/" + clean, nil
}

// CleanKey normalizes a key and refuses one that would escape.
//
// This is the check that matters. A key is often a filename that came from an
// upload, and "../../../etc/passwd" is what an upload form eventually receives.
// Rejecting rather than sanitizing: a key that had to be rewritten to be safe is
// a key the caller did not mean, and silently storing it somewhere else is worse
// than an error.
func CleanKey(key string) (string, error) {
	if key == "" {
		return "", ErrBadKey
	}
	// A leading slash would make path.Clean produce an absolute path, which a
	// driver would then join into the wrong place.
	trimmed := strings.TrimPrefix(key, "/")
	clean := path.Clean(trimmed)

	if clean == "." || clean == "/" || strings.HasPrefix(clean, "../") || clean == ".." {
		return "", ErrBadKey
	}
	if strings.HasPrefix(clean, "/") {
		return "", ErrBadKey
	}
	// A NUL byte truncates a path in every syscall that takes one, so a key
	// carrying it can name a different file than it appears to.
	if strings.ContainsRune(clean, 0) {
		return "", ErrBadKey
	}
	return clean, nil
}

// prefix builds the stored path for a listing prefix.
//
// It differs from [Key] in one place: the empty prefix is legal and means
// "everything this tenant has", which CleanKey refuses because an empty key
// names no file. Listing and naming are not the same question, and answering
// the first with the rules of the second forced callers to pass "." -- a key
// that then had to be special-cased in every driver.
func prefixPath(g auth.Grant, p string) (string, error) {
	tenant := auth.Tenant(g)
	if tenant == "" {
		return "", ErrNoTenant
	}
	if !auth.ValidTenant(tenant) {
		return "", fmt.Errorf("%w: %q cannot be a path segment", ErrNoTenant, tenant)
	}
	if p == "" || p == "/" {
		return tenant + "/", nil
	}
	clean, err := CleanKey(p)
	if err != nil {
		return "", err
	}
	// The trailing slash is kept when the caller wrote one, because "invoice"
	// and "invoice/" select different sets and only the caller knows which was
	// meant.
	if strings.HasSuffix(p, "/") {
		clean += "/"
	}
	return tenant + "/" + clean, nil
}
