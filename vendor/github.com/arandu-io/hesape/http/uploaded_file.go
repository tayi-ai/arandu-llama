package http

import (
	"context"
	"errors"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"os"
	"path"
	"path/filepath"
	"strings"
	"sync"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/filesystem"
	"github.com/arandu-io/hesape/internal/filetype"
)

// ErrFileNotFound is returned when the file behind an upload is gone.
var ErrFileNotFound = errors.New("http: the uploaded file is not readable")

// BaseFile is a file on disk.
//
// It is BaseFile and not File because [File] is already the function that
// pulls an upload off a Context, and a package cannot have both.
type BaseFile struct {
	// pathname is the location on disk.
	pathname string
	// hashName caches the name HashName drew, so two calls to HashName on
	// one file return the same name.
	hashName string
}

// NewBaseFile builds a BaseFile at a path.
func NewBaseFile(pathname string) *BaseFile { return &BaseFile{pathname: pathname} }

// Path is the fully-qualified path to the file.
func (f *BaseFile) Path() string { return f.pathname }

// GetRealPath resolves symlinks in the path; a path that cannot be resolved
// is returned as it stands, because the caller wanted a path and not an
// error about one.
func (f *BaseFile) GetRealPath() string {
	resolved, err := filepath.EvalSymlinks(f.pathname)
	if err != nil {
		return f.pathname
	}
	return resolved
}

// GetPathname is an alias for Path.
func (f *BaseFile) GetPathname() string { return f.pathname }

// Extension is an alias for GuessExtension.
func (f *BaseFile) Extension() string { return f.GuessExtension() }

// GuessExtension is the canonical extension implied by the file's content,
// without the dot. It is empty when the content cannot be read or classified.
func (f *BaseFile) GuessExtension() string {
	_, extension, ok := filetype.Detect(func() (io.ReadCloser, error) { return os.Open(f.pathname) })
	if !ok {
		return ""
	}
	return extension
}

// HashName is a random 40-character name with the file's extension, under
// an optional directory.
//
// The name is drawn once per file and cached, so a caller that asks twice
// stores and links the same thing. The variadic argument is an optional
// directory prefix.
func (f *BaseFile) HashName(dir ...string) string {
	prefix := ""
	if len(dir) > 0 && dir[0] != "" {
		prefix = strings.TrimRight(dir[0], "/") + "/"
	}
	if f.hashName == "" {
		f.hashName = strRandom(40)
	}
	extension := f.GuessExtension()
	if extension != "" {
		extension = "." + extension
	}
	return prefix + f.hashName + extension
}

// Dimensions is the width and height of the image, and whether it is one.
//
// Returns (width, height, ok): ok is false when the file is not a
// decodable image.
func (f *BaseFile) Dimensions() (int, int, bool) {
	return filetype.Image(func() (io.ReadCloser, error) { return os.Open(f.GetRealPath()) }, false)
}

// GetSize is the file's size in bytes.
func (f *BaseFile) GetSize() int64 {
	info, err := os.Stat(f.pathname)
	if err != nil {
		return 0
	}
	return info.Size()
}

// GetMimeType is the normalized MIME type detected from a bounded read of the
// content. It is empty when the content cannot be opened or is empty.
func (f *BaseFile) GetMimeType() string {
	mediaType, _, ok := filetype.Detect(func() (io.ReadCloser, error) { return os.Open(f.pathname) })
	if !ok {
		return ""
	}
	return mediaType
}

// GetPath returns the location that identifies the file to validation.
func (f *BaseFile) GetPath() string { return f.pathname }

// GetExtension returns the extension of the file's name, without the dot.
func (f *BaseFile) GetExtension() string {
	return strings.TrimPrefix(strings.ToLower(path.Ext(f.pathname)), ".")
}

// UploadedFile is one file that arrived in a form field, with the
// file-info methods and the four store variants.
//
// It embeds BaseFile, so the two share the same methods on the same fields.
//
// # Storing one needs a Grant
//
// The store methods take a context and an auth.Grant, because
// [filesystem.Disk.PutFileAs] does: a file is written under a tenant, the
// tenant comes off the Grant a policy minted, and there is no path from a
// form field to a storage prefix.
type UploadedFile struct {
	BaseFile

	// header is the multipart part this was read from, which is what Open and
	// the client-announced values come off.
	header *multipart.FileHeader
	// originalName is the name the client announced, with any directory
	// stripped.
	originalName string
	// mimeType is the type the client announced. It is recorded and never
	// trusted.
	mimeType string
	// err is the upload error code, 0 when there is none.
	err int
	// test is whether this file was made by a factory rather than by a
	// browser, which is what makes IsValid true without a real upload
	// behind it.
	test bool
	// inspection is shared by every content-derived answer, so a validation
	// chain decodes untrusted bytes once rather than once per rule.
	inspection *uploadInspection
}

type uploadInspection struct {
	once   sync.Once
	result filetype.Inspection
}

// NewUploadedFile builds an UploadedFile from the multipart part a browser
// sent.
//
// This is the one constructor: a second package-level CreateFromBase for
// this type would collide with Request's, since Go namespaces by package
// rather than by type.
func NewUploadedFile(header *multipart.FileHeader, field string) *UploadedFile {
	name := ""
	announced := ""
	if header != nil {
		name = path.Base(strings.ReplaceAll(header.Filename, `\`, "/"))
		announced = header.Header.Get("Content-Type")
	}
	return &UploadedFile{
		BaseFile:     BaseFile{pathname: name},
		header:       header,
		originalName: name,
		mimeType:     announced,
		inspection:   &uploadInspection{},
	}
}

// NewUploadedFileFromPath builds an UploadedFile standing in for one that
// arrived, from a file that is already on disk -- the shape a testing
// factory uses. The test flag is what IsValid checks instead of looking for
// a real multipart part.
func NewUploadedFileFromPath(pathname, originalName, mimeType string, test bool) *UploadedFile {
	return &UploadedFile{
		BaseFile:     BaseFile{pathname: pathname},
		originalName: originalName,
		mimeType:     mimeType,
		test:         test,
		inspection:   &uploadInspection{},
	}
}

// GetClientOriginalName is the name the client announced.
func (f *UploadedFile) GetClientOriginalName() string { return f.originalName }

// GetClientOriginalExtension is the extension of the name the client
// announced.
func (f *UploadedFile) GetClientOriginalExtension() string {
	return strings.TrimPrefix(strings.ToLower(path.Ext(f.originalName)), ".")
}

// GetClientMimeType is the type the client announced, which is a header
// somebody wrote.
func (f *UploadedFile) GetClientMimeType() string { return f.mimeType }

// ClientExtension is an alias for GuessClientExtension.
func (f *UploadedFile) ClientExtension() string { return f.GuessClientExtension() }

// GuessClientExtension is the extension guessed from the type the client
// announced.
func (f *UploadedFile) GuessClientExtension() string {
	extensions, err := mime.ExtensionsByType(f.mimeType)
	if err != nil || len(extensions) == 0 {
		return f.GetClientOriginalExtension()
	}
	return strings.TrimPrefix(extensions[0], ".")
}

// GetError is the upload error code, 0 when there is none.
func (f *UploadedFile) GetError() int { return f.err }

// IsValid reports whether the upload completed and there are bytes behind
// it.
func (f *UploadedFile) IsValid() bool {
	if f.err != 0 {
		return false
	}
	if f.test {
		_, err := os.Stat(f.pathname)
		return err == nil
	}
	return f.header != nil && f.header.Size > 0
}

// GetSize is the byte count the server counted as the body arrived, which
// is the one value on an upload that did not come from the client.
func (f *UploadedFile) GetSize() int64 {
	if f.header != nil {
		return f.header.Size
	}
	return f.BaseFile.GetSize()
}

// GuessExtension is the canonical extension implied by the uploaded bytes.
// It never falls back to the announced name or Content-Type.
func (f *UploadedFile) GuessExtension() string {
	result := f.inspect()
	if !result.OK {
		return ""
	}
	return result.Extension
}

// HashName is redeclared here so that the extension comes from the
// upload's own GuessExtension rather than the embedded BaseFile's.
func (f *UploadedFile) HashName(dir ...string) string {
	prefix := ""
	if len(dir) > 0 && dir[0] != "" {
		prefix = strings.TrimRight(dir[0], "/") + "/"
	}
	if f.hashName == "" {
		f.hashName = strRandom(40)
	}
	extension := f.GuessExtension()
	if extension != "" {
		extension = "." + extension
	}
	return prefix + f.hashName + extension
}

// Extension is an alias for GuessExtension.
func (f *UploadedFile) Extension() string { return f.GuessExtension() }

// GetMimeType is the normalized MIME type detected from a bounded read of the
// uploaded bytes. It never falls back to client metadata.
func (f *UploadedFile) GetMimeType() string {
	result := f.inspect()
	if !result.OK {
		return ""
	}
	return result.MediaType
}

// Dimensions returns the dimensions of a structurally valid raster image.
func (f *UploadedFile) Dimensions() (int, int, bool) {
	result := f.inspect()
	return result.Width, result.Height, result.OK && result.Image && !result.SVG
}

func (f *UploadedFile) inspect() filetype.Inspection {
	if f.inspection == nil {
		// A zero-value UploadedFile has no useful stream and is not cacheable.
		// Keeping this branch mutation-free also makes concurrent failure safe.
		return filetype.Inspect(f.Open)
	}
	f.inspection.once.Do(func() {
		f.inspection.result = filetype.Inspect(f.Open)
	})
	return f.inspection.result
}

// Open returns a reader over the bytes. It is what Get and the store methods
// read through, and it may be called more than once.
func (f *UploadedFile) Open() (io.ReadCloser, error) {
	if f.header != nil {
		handle, err := f.header.Open()
		if err != nil {
			return nil, fmt.Errorf("%w: %v", ErrFileNotFound, err)
		}
		return handle, nil
	}
	handle, err := os.Open(f.pathname)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrFileNotFound, err)
	}
	return handle, nil
}

// Get is the contents of the uploaded file.
//
// Returns ([]byte, error): the error unwraps to [ErrFileNotFound] when the
// file is not valid.
func (f *UploadedFile) Get() ([]byte, error) {
	if !f.IsValid() {
		return nil, fmt.Errorf("%w: %s", ErrFileNotFound, f.pathname)
	}
	handle, err := f.Open()
	if err != nil {
		return nil, err
	}
	defer handle.Close()
	return io.ReadAll(handle)
}

// Upload converts this file into the [filesystem.Upload] the storage layer
// takes. It is the one place the two vocabularies meet, so that the conversion
// is written once.
func (f *UploadedFile) Upload(field string) filesystem.Upload {
	if f.header != nil {
		return filesystem.FromMultipart(field, f.header)
	}
	return filesystem.Upload{
		Field:       field,
		Name:        f.originalName,
		Size:        f.GetSize(),
		ContentType: f.mimeType,
		Open:        func() (io.ReadCloser, error) { return f.Open() },
	}
}

// StoreOptions is what Store and its three siblings take to say where and
// how to store a file: a struct says this without a caller having to know a
// set of map keys by name.
type StoreOptions struct {
	// Disk is the disk to write to. The zero value is the default disk.
	Disk *filesystem.Disk
	// Visibility sets the stored file's visibility. StorePublicly and
	// StorePubliclyAs set it to "public".
	Visibility string
}

// Store writes the file under a directory with a random name, and returns
// the key it landed on.
//
// Returns (string, error). The context and the Grant are what
// [filesystem.Disk.PutFileAs] requires: a file is stored under the tenant a
// policy authorised, never under one read off the request.
func (f *UploadedFile) Store(ctx context.Context, g auth.Grant, directory string, options StoreOptions) (string, error) {
	return f.StoreAs(ctx, g, directory, f.HashName(), options)
}

// StorePublicly is Store with the visibility set to public.
func (f *UploadedFile) StorePublicly(ctx context.Context, g auth.Grant, directory string, options StoreOptions) (string, error) {
	options.Visibility = filesystem.VisibilityPublic
	return f.StoreAs(ctx, g, directory, f.HashName(), options)
}

// StorePubliclyAs is StoreAs with the visibility set to public.
func (f *UploadedFile) StorePubliclyAs(ctx context.Context, g auth.Grant, directory, name string, options StoreOptions) (string, error) {
	options.Visibility = filesystem.VisibilityPublic
	return f.StoreAs(ctx, g, directory, name, options)
}

// StoreAs writes the file under a directory with the name given, and
// returns the key it landed on.
//
// Returns (string, error).
func (f *UploadedFile) StoreAs(ctx context.Context, g auth.Grant, directory, name string, options StoreOptions) (string, error) {
	if options.Disk == nil {
		return "", errors.New("http: storing an upload needs a disk: set StoreOptions.Disk to the filesystem.Disk it belongs on")
	}
	if !f.IsValid() {
		return "", fmt.Errorf("%w: %s", ErrFileNotFound, f.pathname)
	}
	return options.Disk.PutFileAs(ctx, g, directory, f.Upload(f.originalName), name)
}
