package filesystem

import (
	"errors"
	"net/http"
	"strings"

	"github.com/arandu-io/hesape/auth"
)

// ServeCallback is what [Disk.ServeUsing] installs: the whole of how a file on
// this disk reaches a browser.
//
// It replaces [Serve] for this disk. An application installs one when the
// response needs something this package does not do -- a redirect to a CDN, an
// extra audit line, a watermark -- and it is a callback rather than a subclass
// because a Disk is a value here and not a class to extend.
type ServeCallback func(w http.ResponseWriter, r *http.Request, g auth.Grant, key string, opt ServeOptions) error

// ServeUsing installs the callback that [Disk.Response] and [Disk.Download]
// hand the work to. Passing nil puts [Serve] back.
func (d *Disk) ServeUsing(callback ServeCallback) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.serveUsing = callback
}

// Response writes a stored file into the response for the browser to render.
//
// name is what to call the file, and empty means the last segment of the key;
// headers are added to the
// response before the file's own are set, so the three headers [Serve] insists
// on -- the stored content type, nosniff, and the escaped disposition -- cannot
// be overridden by a caller passing them in. That is deliberate: they are what
// stops an upload becoming stored XSS, and a header map is exactly where
// somebody would turn them off by accident.
func (d *Disk) Response(w http.ResponseWriter, r *http.Request, g auth.Grant, key, name string, headers http.Header) error {
	return d.respond(w, r, g, key, ServeOptions{Filename: name}, headers)
}

// Download writes a stored file into the response as an attachment, so the
// browser saves it instead of rendering it.
func (d *Disk) Download(w http.ResponseWriter, r *http.Request, g auth.Grant, key, name string, headers http.Header) error {
	return d.respond(w, r, g, key, ServeOptions{Download: true, Filename: name}, headers)
}

// respond is the one path both take, so the callback and the extra headers
// cannot apply to one of them and not the other.
func (d *Disk) respond(w http.ResponseWriter, r *http.Request, g auth.Grant, key string, opt ServeOptions, headers http.Header) error {
	for name, values := range headers {
		for _, value := range values {
			w.Header().Add(name, value)
		}
	}
	d.mu.RLock()
	callback := d.serveUsing
	d.mu.RUnlock()
	if callback != nil {
		return callback(w, r, g, key, opt)
	}
	return Serve(w, r, g, d, key, opt)
}

// DiskName records the name this adapter's disk is registered under.
//
// It is the name [ServeFile] needs to find the disk again from a link.
// [Disks.Build] sets it; setting it by hand is for a disk built without the
// manager.
func (a *LocalFilesystemAdapter) DiskName(disk string) *LocalFilesystemAdapter {
	a.diskName = disk
	return a
}

// ShouldServeSignedUrls records that this disk's files reach a browser through
// the signed route rather than a public address.
//
// [ServeFile] reads it. There is no "serve without a signature" here: a route
// that hands out a stored file with no proof attached is a read with no
// authorization, which is the one thing this collection does not have a way to
// write. Off
// therefore means this disk is not served by that route at all, not that it is
// served openly.
func (a *LocalFilesystemAdapter) ShouldServeSignedUrls(serve bool) *LocalFilesystemAdapter {
	a.serveSigned = serve
	return a
}

// ServesSignedURLs reports what [LocalFilesystemAdapter.ShouldServeSignedUrls]
// was told.
func (a *LocalFilesystemAdapter) ServesSignedURLs() bool { return a.serveSigned }

// ServeFile is the route that turns a link from [URLSigner.TemporaryURL] back
// into the file it names.
//
// It is the half that makes "no symlink into a document root" a real answer:
// the file is served by a route,
// and the route's proof that a Policy said yes is the signature on the link.
//
// Mount it under the same base [NewURLSigner] was given, with the token as the
// last segment:
//
//	mux.Handle("GET /files/", http.StripPrefix("/files/", serveFile))
type ServeFile struct {
	disks  *Disks
	signer *URLSigner
	// shouldServeSignedUrls is what the disk was configured with. See
	// [LocalFilesystemAdapter.ShouldServeSignedUrls] for why off means "not
	// served here" rather than "served openly".
	shouldServeSignedUrls bool
}

// NewServeFile returns the route.
//
// shouldServeSignedUrls is what the disk was configured with; pass false and
// every request is refused, which is what a disk that is not meant to be
// reachable by link wants.
func NewServeFile(disks *Disks, signer *URLSigner, shouldServeSignedUrls bool) *ServeFile {
	return &ServeFile{disks: disks, signer: signer, shouldServeSignedUrls: shouldServeSignedUrls}
}

// ServeHTTP redeems the token in the path and writes the file it names.
//
// The status codes are the ones the rest of the collection uses: 403 for a link
// that is not valid -- forged, expired, or minted for something else -- and 404
// for a valid link to a file that is no longer there. They are different on
// purpose: an expired link is something the person can be told to ask for again,
// and a missing file is not.
func (s *ServeFile) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !s.shouldServeSignedUrls {
		http.Error(w, "Forbidden", http.StatusForbidden)
		return
	}

	token := strings.TrimPrefix(r.URL.Path, "/")
	if token == "" {
		http.Error(w, "Forbidden", http.StatusForbidden)
		return
	}

	g, diskName, key, err := s.signer.Redeem(token)
	if err != nil {
		http.Error(w, "Forbidden", http.StatusForbidden)
		return
	}
	disk, err := s.disks.Disk(diskName)
	if err != nil {
		http.Error(w, "Not Found", http.StatusNotFound)
		return
	}

	if err := Serve(w, r, g, disk, key, ServeOptions{}); err != nil {
		if errors.Is(err, ErrNotFound) {
			http.Error(w, "Not Found", http.StatusNotFound)
			return
		}
		// Anything else already has a status on the wire or is a failure the
		// caller cannot fix by retrying; the body is not extended with a reason
		// a stranger holding a link has no business reading.
		http.Error(w, "Not Found", http.StatusNotFound)
	}
}
