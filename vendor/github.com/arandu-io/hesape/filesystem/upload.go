package filesystem

import (
	"context"
	"crypto/rand"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"path"
	"strings"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/internal/filetype"
)

// ErrRefusedUpload is what every [UploadRules] failure unwraps to, so a handler
// answers "we cannot take this file" once instead of switching on four reasons.
var ErrRefusedUpload = errors.New("filesystem: the upload was refused")

// Upload is a file that arrived on a request.
//
// It is the "own contract" that validation points at: a file is not in
// url.Values, and validating one inside the form pipeline would need a second
// input shape and a second set of rules, which is a second way to validate. An
// upload is checked here, against [UploadRules], and stored with
// [Disk.Put].
//
// Everything on it except Size came from the client and is a string somebody
// chose. Name is not a key: it is what to call the file if it is ever offered
// back, and using it as a key is how "../../etc/passwd" gets stored -- which
// [CleanKey] then refuses, but the habit is the bug. Size is counted by the
// server as the body is read, so it is the one field worth checking.
type Upload struct {
	// Field is the form field the file arrived in.
	Field string
	// Name is the filename the client announced, with any directory stripped.
	Name string
	// Size is how many bytes arrived.
	Size int64
	// ContentType is the type the client announced. It is recorded and never
	// trusted: [Disk.Put] infers the stored type from the key instead.
	ContentType string
	// Open reads the content. The caller closes what it returns, and may call
	// Open more than once.
	Open func() (io.ReadCloser, error)
}

// FromMultipart builds an Upload out of a parsed multipart part.
//
// It is the one place multipart is understood, so the HTTP layer above hands
// over a *multipart.FileHeader and gets back something with rules attached.
func FromMultipart(field string, h *multipart.FileHeader) Upload {
	return Upload{
		Field: field,
		// A browser sends a bare filename and a script sends whatever it likes.
		// path.Base and the backslash trim between them leave nothing that
		// reads as a directory on either kind of host.
		Name:        path.Base(strings.ReplaceAll(h.Filename, `\`, "/")),
		Size:        h.Size,
		ContentType: h.Header.Get("Content-Type"),
		Open: func() (io.ReadCloser, error) {
			f, err := h.Open()
			if err != nil {
				return nil, fmt.Errorf("filesystem: opening upload %q: %w", h.Filename, err)
			}
			return f, nil
		},
	}
}

// Extension returns the lowercased extension of the announced name, with the
// dot, or "" when there is none.
func (u Upload) Extension() string { return strings.ToLower(path.Ext(u.Name)) }

// UploadRules is what an upload has to satisfy.
//
// Both fields are required, and both fail closed. A rules value that named no
// maximum would accept a four gigabyte file, and one that named no extension
// would accept an .exe -- and the moment those are the defaults, the rule that
// was forgotten looks exactly like the rule that was written.
//
// There is no client content-type rule. The announced type and filename are
// metadata the client wrote; Extensions is compared to the extension derived
// from a bounded read of the bytes.
type UploadRules struct {
	// MaxBytes is the largest file accepted. It must be positive.
	MaxBytes int64
	// Extensions is the accepted set, lowercased and with the dot: ".pdf".
	// It must not be empty.
	Extensions []string
}

// Check answers whether this upload satisfies the rules.
//
// Every failure unwraps to [ErrRefusedUpload] and every message is one a person
// can act on, because most of them are shown to one.
func (u Upload) Check(r UploadRules) error {
	if r.MaxBytes <= 0 {
		return fmt.Errorf("%w: UploadRules.MaxBytes is not set, and a size limit that defaults to unlimited is the one nobody notices is missing", ErrRefusedUpload)
	}
	if len(r.Extensions) == 0 {
		return fmt.Errorf("%w: UploadRules.Extensions is empty, and an accepted set that defaults to everything accepts an executable", ErrRefusedUpload)
	}
	if u.Open == nil {
		return fmt.Errorf("%w: it carries no content", ErrRefusedUpload)
	}
	if u.Size <= 0 {
		return fmt.Errorf("%w: the file is empty", ErrRefusedUpload)
	}
	if u.Size > r.MaxBytes {
		return fmt.Errorf("%w: it is %d bytes and the limit is %d", ErrRefusedUpload, u.Size, r.MaxBytes)
	}
	ext, ok := u.contentExtension()
	if !ok {
		return fmt.Errorf("%w: its content could not be classified", ErrRefusedUpload)
	}
	for _, want := range r.Extensions {
		if sameUploadExtension(ext, strings.ToLower(want)) {
			return nil
		}
	}
	return fmt.Errorf("%w: content detected as %s is not accepted, only %s", ErrRefusedUpload, ext, strings.Join(r.Extensions, ", "))
}

func (u Upload) contentExtension() (string, bool) {
	_, extension, ok := filetype.Detect(u.Open)
	if !ok || extension == "" {
		return "", false
	}
	return "." + extension, true
}

func sameUploadExtension(detected, allowed string) bool {
	if detected == allowed {
		return true
	}
	return detected == ".jpeg" && allowed == ".jpg" || detected == ".jpg" && allowed == ".jpeg"
}

// PutFile stores an upload under a directory and returns the key it landed on.
//
// The name is drawn at random and keeps only the extension detected from the
// content, which is the point of the call: the announced filename is a string
// the client chose, and storing under it means two people uploading "scan.pdf"
// overwrite each other -- and that somebody who guesses a filename guesses a
// key or controls its executable suffix.
//
// It does not check [UploadRules]. Checking is [Upload.Check], called where the
// answer can be shown to the person who picked the file; folding it in here
// would make an unchecked Put impossible to write and a checked one impossible
// to see.
func (d *Disk) PutFile(ctx context.Context, g auth.Grant, directory string, u Upload) (string, error) {
	extension, ok := u.contentExtension()
	if !ok {
		return "", fmt.Errorf("%w: its content could not be classified", ErrRefusedUpload)
	}
	return d.PutFileAs(ctx, g, directory, u, randomName()+extension)
}

// randomName draws the forty random characters a stored file is named with.
//
// It does not call str.Random, which is the same forty characters, because that
// one answers to a test hook that makes the value deterministic process-wide. A
// deterministic name is fine for a token in
// a fixture and is not fine here: unpredictability is half of why this call
// exists, and a global that can switch it off is a global somebody will switch
// off in a test helper that also runs in staging.
func randomName() string {
	const alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	b := make([]byte, 40)
	// crypto/rand.Read is documented never to fail, and panics internally if the
	// operating system's source ever does. There is no error to handle.
	rand.Read(b)
	for i, v := range b {
		// 62 does not divide 256, so a modulo makes the first eight letters very
		// slightly likelier. Over forty characters that is not a weakness worth
		// a rejection loop, and it is written down rather than hidden.
		b[i] = alphabet[int(v)%len(alphabet)]
	}
	return string(b)
}

// PutFileAs stores an upload under a directory with the name you give it.
//
// The name is a name: one with a separator in it is refused rather than cleaned,
// because a caller that meant a subdirectory should say so in directory, and a
// caller that did not mean one has been handed something by a client.
func (d *Disk) PutFileAs(ctx context.Context, g auth.Grant, directory string, u Upload, name string) (string, error) {
	if u.Open == nil {
		return "", fmt.Errorf("%w: it carries no content", ErrRefusedUpload)
	}
	if name == "" || name == "." || name == ".." ||
		strings.ContainsAny(name, `/\`) || strings.ContainsRune(name, 0) {
		return "", fmt.Errorf("%w: %q is not a filename", ErrBadKey, name)
	}
	key := name
	if dir := strings.Trim(directory, "/"); dir != "" {
		key = dir + "/" + name
	}

	body, err := u.Open()
	if err != nil {
		return "", err
	}
	defer body.Close()

	// The empty content type is on purpose: Put infers it from the key, and the
	// one the client announced is a string an attacker chose.
	if err := d.Put(ctx, g, key, body, ""); err != nil {
		return "", err
	}
	return key, nil
}
