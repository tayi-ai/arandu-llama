package image

import (
	"context"
	"encoding/base64"
	"io"
	"net/http"
	"os"
	"sync"

	"github.com/arandu-io/hesape/auth"
)

// StdDriverName is the name the driver this package carries is registered
// under, and what [ImageManager.GetDefaultDriver] returns until somebody
// changes it.
//
// It drives the standard library, and it says so.
const StdDriverName = "std"

// HTTPClient is the slice of an HTTP client that [ImageManager.FromURL] needs.
//
// *http.Client is one. It is a parameter rather than something this package
// builds because the client is where the timeout lives, and a package that
// invented its own would be downloading from a stranger's URL with no deadline.
type HTTPClient interface {
	Do(req *http.Request) (*http.Response, error)
}

// ImageManager is where images come from, and which driver processes them.
//
// An application builds one and keeps it: it is cheap, it holds no per-image
// state, and every method on it is safe to call from several goroutines at
// once.
//
//	images := image.NewImageManager()
//	img, err := images.FromPath("photo.jpg")
type ImageManager struct {
	mu            sync.RWMutex
	defaultDriver string
	drivers       map[string]Driver
	creators      map[string]func() (Driver, error)
	handlers      map[string]map[string]TransformationHandler
	// maxPixels is the ceiling handed to every driver this manager builds.
	// Zero means nobody set one, and each driver keeps its own default.
	maxPixels int
}

// NewImageManager returns a manager with the one driver this package carries
// already registered under [StdDriverName].
func NewImageManager() *ImageManager {
	m := &ImageManager{
		defaultDriver: StdDriverName,
		drivers:       map[string]Driver{},
		creators:      map[string]func() (Driver, error){},
		handlers:      map[string]map[string]TransformationHandler{},
	}
	m.creators[StdDriverName] = func() (Driver, error) { return newStdDriver(), nil }
	return m
}

// GetDefaultDriver returns the name of the driver used when nobody names one:
// whatever [ImageManager.SetDefaultDriver] was told, and [StdDriverName]
// until then.
func (m *ImageManager) GetDefaultDriver() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.defaultDriver
}

// SetDefaultDriver names the driver every image uses unless [Image.Using]
// says otherwise.
func (m *ImageManager) SetDefaultDriver(name string) *ImageManager {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.defaultDriver = name
	return m
}

// Extend registers a driver under a name.
//
// The creator runs at most once, the first time somebody asks for that
// driver, and lazily for the same reason it always should -- a driver nobody
// uses should cost nothing.
func (m *ImageManager) Extend(name string, creator func() (Driver, error)) *ImageManager {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.creators[name] = creator
	delete(m.drivers, name)
	return m
}

// Driver returns the driver by name, or the default when no name is given,
// and an error naming it for a name nobody registered.
func (m *ImageManager) Driver(name ...string) (Driver, error) {
	want := optionalString(name)
	if want == "" {
		want = m.GetDefaultDriver()
	}

	m.mu.RLock()
	if d, ok := m.drivers[want]; ok {
		m.mu.RUnlock()
		return d, nil
	}
	m.mu.RUnlock()

	m.mu.Lock()
	defer m.mu.Unlock()
	if d, ok := m.drivers[want]; ok {
		return d, nil
	}
	creator, ok := m.creators[want]
	if !ok {
		return nil, fail("image driver [%s] is not supported", want)
	}
	d, err := creator()
	if err != nil {
		return nil, failWith(err, "creating the [%s] image driver", want)
	}
	for transformation, handler := range m.handlers[want] {
		d.TransformUsing(transformation, handler)
	}
	if m.maxPixels != 0 {
		d.MaxPixels(m.maxPixels)
	}
	m.drivers[want] = d
	return d, nil
}

// MaxPixels sets how many pixels an image may declare before the drivers of
// this manager refuse to decode it. A value of zero or less restores
// [DefaultMaxPixels].
//
// The ceiling is read out of the header, so an image past it is refused before
// its pixels are allocated -- which is the only moment where refusing it saves
// anything, since the memory a crafted file asks for is spent by the decode:
//
//	images := image.NewImageManager()
//	images.MaxPixels(8_000_000)    // nothing over eight megapixels is decoded
//
// There is no unlimited setting. An image with no bound on it is one whoever
// sent it decides the size of.
//
// It reaches a driver already built immediately, and every driver built after
// it, so the ceiling is one number for the manager rather than one per driver.
func (m *ImageManager) MaxPixels(pixels int) *ImageManager {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.maxPixels = pixels
	for _, d := range m.drivers {
		d.MaxPixels(pixels)
	}
	return m
}

// TransformUsing registers how one driver should carry out one
// transformation, keyed by the transformation's TransformationName.
//
// A handler registered for a driver already built reaches it immediately.
func (m *ImageManager) TransformUsing(driver, transformation string, handler TransformationHandler) *ImageManager {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.handlers[driver] == nil {
		m.handlers[driver] = map[string]TransformationHandler{}
	}
	m.handlers[driver][transformation] = handler
	if d, ok := m.drivers[driver]; ok {
		d.TransformUsing(transformation, handler)
	}
	return m
}

// newImage is the one place an [Image] is built, so that every entry point
// below hands out the same thing.
func (m *ImageManager) newImage(contents []byte, file UploadedFile) *Image {
	return &Image{manager: m, contents: contents, file: file, pipeline: NewImagePipeline()}
}

// FromBytes builds an [Image] from bytes already in memory.
func (m *ImageManager) FromBytes(contents []byte) *Image {
	return m.newImage(contents, nil)
}

// FromStream builds an [Image] by reading stream fully, now rather than
// keeping it for later.
//
// A stream held for later is a file descriptor held for later, and a
// deferred read would put the read somewhere the caller cannot see it fail.
func (m *ImageManager) FromStream(stream io.Reader) (*Image, error) {
	contents, err := io.ReadAll(stream)
	if err != nil {
		return nil, failWith(err, "invalid stream image data")
	}
	return m.newImage(contents, nil), nil
}

// FromBase64 builds an [Image] by decoding a base64 string.
func (m *ImageManager) FromBase64(encoded string) (*Image, error) {
	contents, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, failWith(err, "invalid base64 image data")
	}
	return m.newImage(contents, nil), nil
}

// FromPath builds an [Image] from a file of the application's own, read from
// the local filesystem.
//
// It is not the way in for something a customer sent. An upload has no path
// until somebody stores it, and storing it goes through [Image.Store], which
// takes an auth.Grant and a tenant with it.
func (m *ImageManager) FromPath(path string) (*Image, error) {
	contents, err := os.ReadFile(path)
	if err != nil {
		return nil, failWith(err, "reading the image at %s", path)
	}
	return m.newImage(contents, nil), nil
}

// FromStorage builds an [Image] read back off a storage disk.
//
// The disk is an argument, and the read takes an auth.Grant: a read is a path
// to somebody else's data exactly as a write is, and there is no query path
// without a policy. The key is resolved under the Grant's tenant by the disk
// itself.
func (m *ImageManager) FromStorage(ctx context.Context, g auth.Grant, disk DiskReader, path string) (*Image, error) {
	body, err := disk.Get(ctx, g, path)
	if err != nil {
		return nil, failWith(err, "reading the image at %s", path)
	}
	defer body.Close()

	contents, err := io.ReadAll(body)
	if err != nil {
		return nil, failWith(err, "reading the image at %s", path)
	}
	return m.newImage(contents, nil), nil
}

// FromUpload builds an [Image] from an uploaded file.
//
// The upload is kept on the image, which is what makes [Image.File] return
// something, and is the only entry point where it does.
func (m *ImageManager) FromUpload(file UploadedFile) (*Image, error) {
	contents, err := file.GetContent()
	if err != nil {
		return nil, failWith(err, "reading the uploaded image")
	}
	return m.newImage(contents, file), nil
}

// FromURL builds an [Image] by downloading url with client.
//
// It takes a client because the client carries the timeout and the
// transport, and both are decisions this package must not make for a request
// to an address somebody else supplied.
func (m *ImageManager) FromURL(ctx context.Context, client HTTPClient, url string) (*Image, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, failWith(err, "requesting %s", url)
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, failWith(err, "requesting %s", url)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return nil, fail("requesting %s: the server answered %s", url, resp.Status)
	}
	contents, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, failWith(err, "reading the image from %s", url)
	}
	return m.newImage(contents, nil), nil
}
