package filesystem

import (
	"fmt"
	"io"
	"mime"
	"net/http"
	"path"
	"strconv"

	"github.com/arandu-io/hesape/auth"
)

// ServeOptions is how a file is offered.
type ServeOptions struct {
	// Download asks the browser to save the file instead of rendering it.
	//
	// It is one field rather than two functions, because two functions that
	// differ by a header is two places to fix the header.
	Download bool
	// Filename is what to call it on the way out. Empty means the last segment
	// of the key.
	Filename string
	// CacheControl is the header to send. Empty means "private, max-age=0,
	// must-revalidate", which is the only safe default for a file that a Policy
	// decided about: a shared cache holding one tenant's file is the same leak
	// as a missing prefix, arriving later.
	CacheControl string
}

// Serve writes a stored file to an HTTP response.
//
// The range-aware read, the attachment and the inline render are one function,
// and which of the three it is comes from [ServeOptions].
//
// This is the half that makes "no symlink into a document root" a real answer
// rather than a refusal. A file is served by a route, the route runs a Policy
// like any other, and the Grant that Policy produced is what reaches the disk.
//
// It writes nothing on error, so the caller decides the status -- ErrNotFound
// is a 404 and auth.ErrForbidden is a 403, and both are decisions the exception
// handler already knows how to make.
//
// Three headers are not negotiable. The content type comes from the stored
// metadata, which came from the key and never from what an upload announced;
// X-Content-Type-Options is nosniff, so a browser cannot decide a .txt is HTML;
// and the disposition filename is escaped, because a filename with a quote and
// a semicolon in it is a header injection with a friendly name.
func Serve(w http.ResponseWriter, r *http.Request, g auth.Grant, d *Disk, key string, opt ServeOptions) error {
	f, err := d.Get(r.Context(), g, key)
	if err != nil {
		return err
	}
	defer f.Body.Close()

	name := opt.Filename
	if name == "" {
		name = path.Base(f.Key)
	}
	disposition := "inline"
	if opt.Download {
		disposition = "attachment"
	}
	cache := opt.CacheControl
	if cache == "" {
		cache = "private, max-age=0, must-revalidate"
	}

	contentType := f.ContentType
	if contentType == "" {
		contentType = TypeOf(key)
	}

	h := w.Header()
	h.Set("Content-Type", contentType)
	h.Set("X-Content-Type-Options", "nosniff")
	h.Set("Content-Disposition", mime.FormatMediaType(disposition, map[string]string{"filename": name}))
	h.Set("Cache-Control", cache)

	// http.ServeContent is what handles a Range request, a conditional GET and
	// the 304 that follows, and it needs to seek. A driver that streams cannot,
	// and gets a plain copy -- correct, just without resumable downloads.
	if seeker, ok := f.Body.(io.ReadSeeker); ok {
		http.ServeContent(w, r, name, f.ModifiedAt, seeker)
		return nil
	}
	if f.Size > 0 {
		h.Set("Content-Length", strconv.FormatInt(f.Size, 10))
	}
	if r.Method == http.MethodHead {
		w.WriteHeader(http.StatusOK)
		return nil
	}
	if _, err := io.Copy(w, f.Body); err != nil {
		// The status is already written by now, so this cannot become a 500.
		// It is returned so the caller can log a truncated response rather than
		// record a success.
		return fmt.Errorf("filesystem: sending %s: %w", key, err)
	}
	return nil
}
