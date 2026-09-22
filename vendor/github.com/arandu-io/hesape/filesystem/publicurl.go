package filesystem

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/arandu-io/hesape/auth"
)

// ErrNoURL is returned by [Disk.URL], [Disk.TemporaryURL] and
// [Disk.TemporaryUploadURL] on a disk that cannot produce the address asked
// for.
var ErrNoURL = errors.New("filesystem: this disk cannot produce that address")

// TemporaryURLCallback is what [Disk.BuildTemporaryURLsUsing] installs.
type TemporaryURLCallback func(ctx context.Context, g auth.Grant, key string, ttl time.Duration) (string, error)

// TemporaryUploadURLCallback is what [Disk.BuildTemporaryUploadURLsUsing]
// installs. The headers are the ones the client must send with the upload.
type TemporaryUploadURLCallback func(ctx context.Context, g auth.Grant, key string, ttl time.Duration) (string, http.Header, error)

// PublicURLGenerator is the optional half of an [Adapter] that can name a file
// at a permanent public address.
//
// Flysystem calls it PublicUrlGenerator, and this is the same idea under the Go
// spelling of the initialism. It is optional for the reason [Presigner] is: a
// directory on disk has no such address, and a method returning an error nobody
// can act on is worse than an interface a driver either satisfies or does not.
type PublicURLGenerator interface {
	PublicURL(ctx context.Context, storedPath string) (string, error)
}

// PresignPutter is the optional half of an [Adapter] that can hand out a URL a
// browser uploads directly TO.
//
// It is the write half of [Presigner], and it is what makes a large upload
// possible without the bytes passing through the application at all. The
// headers it returns are the ones the client must repeat, because a presigned
// PUT is signed over them.
type PresignPutter interface {
	PresignPut(ctx context.Context, storedPath string, ttl time.Duration) (string, http.Header, error)
}

// URL returns the permanent public address of a file.
//
// # Read this before using it
//
// The address carries no authorization. Anybody who has it has the file, for as
// long as the file exists -- no session, no Policy, no expiry. That is what a
// public disk IS, and it is the correct shape for exactly one kind of content:
// something every visitor is meant to see anyway, like a logo. For anything a
// tenant uploaded, the answer is [Disk.TemporaryURL], which expires, or a route
// that runs a Policy and calls [Serve].
//
// It takes a Grant anyway: handing out the address of a file is a
// read, and the caller has to have been allowed to reach the file to be allowed
// to publish where it lives.
//
// It answers [ErrNoURL] when the disk has no public address, which is the
// default -- [Config.URL] is empty and the driver generates none.
func (d *Disk) URL(ctx context.Context, g auth.Grant, key string) (string, error) {
	full, err := Key(g, key)
	if err != nil {
		return "", err
	}
	if generator, ok := d.adapter.(PublicURLGenerator); ok {
		link, err := generator.PublicURL(ctx, full)
		if err != nil {
			return "", d.wrap("url", key, err)
		}
		return link, nil
	}
	if base := d.config.URL; base != "" {
		return strings.TrimSuffix(base, "/") + "/" + full, nil
	}
	return "", fmt.Errorf("%w: disk %q has no public address. Use TemporaryURL, which expires", ErrNoURL, d.name)
}

// ProvidesTemporaryURLs reports whether [Disk.TemporaryURL] will answer with
// one rather than an error.
func (d *Disk) ProvidesTemporaryURLs() bool {
	d.mu.RLock()
	installed := d.temporaryURL != nil
	d.mu.RUnlock()
	if installed {
		return true
	}
	_, ok := d.adapter.(Presigner)
	return ok
}

// ProvidesTemporaryUploadURLs reports whether [Disk.TemporaryUploadURL] will.
func (d *Disk) ProvidesTemporaryUploadURLs() bool {
	d.mu.RLock()
	installed := d.temporaryUploadURL != nil
	d.mu.RUnlock()
	if installed {
		return true
	}
	_, ok := d.adapter.(PresignPutter)
	return ok
}

// BuildTemporaryURLsUsing installs the callback that answers
// [Disk.TemporaryURL] on a disk whose driver cannot presign.
//
// It is how a local disk gets temporary URLs: the application hands it a
// closure that calls [URLSigner.TemporaryURL], and every caller of
// Disk.TemporaryURL then works the same on the directory and on the bucket.
// Wire it once, at boot, next to where the disk is registered.
func (d *Disk) BuildTemporaryURLsUsing(callback TemporaryURLCallback) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.temporaryURL = callback
}

// BuildTemporaryUploadURLsUsing installs the callback that answers
// [Disk.TemporaryUploadURL].
func (d *Disk) BuildTemporaryUploadURLsUsing(callback TemporaryUploadURLCallback) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.temporaryUploadURL = callback
}

// TemporaryURL returns a URL that serves one file for ttl and then stops.
//
// The callback installed by [Disk.BuildTemporaryURLsUsing] wins, so an
// application can route every temporary link through its own signer. Otherwise
// the driver presigns, and the bytes never pass through the application.
//
// The link is a bearer credential for one file. It names the file, it expires,
// and it is proof that a Policy said yes at the moment it was made -- which is
// why it takes a Grant to mint and none to redeem.
func (d *Disk) TemporaryURL(ctx context.Context, g auth.Grant, key string, ttl time.Duration) (string, error) {
	if ttl <= 0 {
		return "", fmt.Errorf("filesystem: a temporary URL needs a lifetime, and %s is not one", ttl)
	}
	d.mu.RLock()
	callback := d.temporaryURL
	d.mu.RUnlock()
	if callback != nil {
		return callback(ctx, g, key, ttl)
	}

	full, err := Key(g, key)
	if err != nil {
		return "", err
	}
	presigner, ok := d.adapter.(Presigner)
	if !ok {
		return "", fmt.Errorf("%w: disk %q cannot presign. Install one with BuildTemporaryURLsUsing, or use URLSigner", ErrNoURL, d.name)
	}
	link, err := presigner.PresignGet(ctx, full, ttl)
	if err != nil {
		return "", d.wrap("presign", key, err)
	}
	return link, nil
}

// TemporaryUploadURL returns a URL a client may upload one file to for ttl, and
// the headers it has to repeat.
//
// It is the answer to a file too large to pass through the application: the
// browser sends the bytes straight to the store. The Grant is what says the
// caller may write that key, and the signature carries that decision -- the
// store will not check it again, so the ttl is the whole of the exposure and
// should be minutes.
func (d *Disk) TemporaryUploadURL(ctx context.Context, g auth.Grant, key string, ttl time.Duration) (string, http.Header, error) {
	if ttl <= 0 {
		return "", nil, fmt.Errorf("filesystem: a temporary upload URL needs a lifetime, and %s is not one", ttl)
	}
	d.mu.RLock()
	callback := d.temporaryUploadURL
	d.mu.RUnlock()
	if callback != nil {
		return callback(ctx, g, key, ttl)
	}

	full, err := Key(g, key)
	if err != nil {
		return "", nil, err
	}
	putter, ok := d.adapter.(PresignPutter)
	if !ok {
		return "", nil, fmt.Errorf("%w: disk %q cannot presign an upload. Install one with BuildTemporaryUploadURLsUsing", ErrNoURL, d.name)
	}
	link, headers, err := putter.PresignPut(ctx, full, ttl)
	if err != nil {
		return "", nil, d.wrap("presign upload", key, err)
	}
	return link, headers, nil
}
