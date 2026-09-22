package image

import (
	"bytes"
	"context"
	"encoding/base64"
	"io"
	"net/http"
	"strconv"
	"strings"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/image/transformations"
	"github.com/arandu-io/hesape/str"
)

// Image is one picture, and everything that has been asked of it but not yet
// done.
//
// Nothing here decodes anything. Every call adds to an [ImagePipeline] and
// hands back a new Image -- the receiver is left exactly as it was, which is
// why a thumbnail built from a source image does not disturb the full-size
// one:
//
//	thumb := original.Cover(200, 200).Blur(3)
//	full, _ := original.ToBytes()   // still the original
//
// The pixels are touched once, by the [Driver], when somebody asks for bytes --
// [Image.ToBytes], [Image.Store], [Image.ToResponse], [Image.Width]. Ask twice
// and the work happens once: the processed bytes replace the originals on the
// instance.
//
// Every one of those methods takes a context, because that one call is where
// all of the work is: resampling a photograph is seconds of arithmetic, and a
// caller who has stopped waiting for the answer needs a way to say so. The
// context is not held on the Image -- each call carries its own, and an Image
// built during one request can be asked for bytes during another.
//
// An Image is not safe for use from several goroutines at once. Two
// goroutines sharing one is two goroutines processing the same pipeline; give
// each the clone that a fluent call already returns.
type Image struct {
	manager  *ImageManager
	contents []byte
	file     UploadedFile
	pipeline *ImagePipeline

	// driver is set by [Image.Using]. Empty means the manager's default.
	driver string
	// processed is whether contents is the output of the pipeline rather than
	// what arrived.
	processed bool
	// hashName caches what [Image.HashName] invented, so two calls on one
	// image return the same name.
	hashName string
	// mimeType and dimensions cache what the driver already computed, so
	// asking twice costs once.
	mimeType string
	width    int
	height   int
}

// UploadedFile is the slice of an uploaded file that this package needs.
//
// It is an interface and not an import for the reason mail.UploadedFile is
// one: an uploaded file belongs to the HTTP layer, and an image is underneath
// it. [ImageManager.FromUpload] calls the first method; [Image.File] hands
// the whole thing back so that a caller who wants the name the browser sent
// still has it.
type UploadedFile interface {
	// GetContent returns the file's contents.
	GetContent() ([]byte, error)
	// GetClientOriginalName returns the filename the client announced. It is
	// not a key and not a path: it is a string a stranger chose.
	GetClientOriginalName() string
}

// Disk is the slice of a storage disk that [Image.Store] needs.
//
// The two methods are filesystem.Disk's, spelled the same, so a
// *filesystem.Disk is one of these without an adapter. Both take an
// auth.Grant: the tenant a stored image lands under comes off the Grant and
// from nowhere else.
type Disk interface {
	Put(ctx context.Context, g auth.Grant, key string, body io.Reader, contentType string) error
	SetVisibility(ctx context.Context, g auth.Grant, key, visibility string) error
}

// DiskReader is the slice of a storage disk that [ImageManager.FromStorage]
// needs: one read, authorized, tenant-scoped.
type DiskReader interface {
	Get(ctx context.Context, g auth.Grant, key string) (io.ReadCloser, error)
}

// visibilityPublic is the string filesystem.VisibilityPublic holds. It is
// repeated rather than imported for the reason [Disk] is an interface.
const visibilityPublic = "public"

// newClone returns a copy with its own pipeline, ready to be added to.
//
// The remembered media type and dimensions do not come along: they describe
// the image as the receiver would produce it, and the clone is about to be
// given another transformation -- a thumbnail that inherited its source's
// width would report 4000 for a picture 200 across. The invented name does
// come along, because it is the file's name and not a fact about its pixels.
func (i *Image) newClone() *Image {
	out := *i
	out.pipeline = i.pipeline.clone()
	out.processed = false
	out.mimeType, out.width, out.height = "", 0, 0
	return &out
}

// withClone returns a clone with apply run on it.
func (i *Image) withClone(apply func(*Image)) *Image {
	clone := i.newClone()
	apply(clone)
	return clone
}

// withOutput returns a clone with apply run on its output options.
func (i *Image) withOutput(apply func(*ImageOutputOptions)) *Image {
	return i.withClone(func(clone *Image) { apply(&clone.pipeline.Output) })
}

// Cover fills the box edge to edge, aspect ratio kept, the overflow cropped
// from the centre.
func (i *Image) Cover(width, height int) *Image {
	return i.Transform(transformations.Cover{Width: maxInt(1, width), Height: maxInt(1, height)})
}

// Contain scales the whole image to fit the box and pads it to fill the rest.
//
// background is an optional hex colour, the sentinel "dominant" for the
// image's own average colour, or nothing at all, which pads with white.
func (i *Image) Contain(width, height int, background ...string) *Image {
	return i.Transform(transformations.Contain{
		Width:      maxInt(1, width),
		Height:     maxInt(1, height),
		Background: optionalString(background),
	})
}

// Crop cuts a rectangle out of the image.
//
// offset is x and y, in that order, and both default to zero -- the top left
// corner.
func (i *Image) Crop(width, height int, offset ...int) *Image {
	x, y := 0, 0
	if len(offset) > 0 {
		x = offset[0]
	}
	if len(offset) > 1 {
		y = offset[1]
	}
	return i.Transform(transformations.Crop{Width: maxInt(1, width), Height: maxInt(1, height), X: x, Y: y})
}

// Resize sets the image to exactly these dimensions; the aspect ratio is not
// kept.
//
// A zero leaves that axis as it is. Both zero is refused with an error: a
// resize that named no dimension is a call that meant something and said
// nothing.
func (i *Image) Resize(width, height int) (*Image, error) {
	if width <= 0 && height <= 0 {
		return nil, fail("at least one resize dimension must be specified")
	}
	return i.Transform(transformations.Resize{Width: width, Height: height}), nil
}

// Rotate turns the image clockwise by the given angle, with the corners it
// opens up filled.
//
// background takes the same spellings as [Image.Contain]'s.
func (i *Image) Rotate(angle float64, background ...string) *Image {
	return i.Transform(transformations.Rotate{Angle: angle, Background: optionalString(background)})
}

// Scale fits the image inside the box, aspect ratio kept, and never scales it
// up.
//
// A zero drops that constraint; both zero is refused with an error, as on
// [Image.Resize].
func (i *Image) Scale(width, height int) (*Image, error) {
	if width <= 0 && height <= 0 {
		return nil, fail("at least one scale dimension must be specified")
	}
	return i.Transform(transformations.Scale{Width: width, Height: height}), nil
}

// Orient reads the EXIF orientation and applies it, so a photograph taken
// sideways is stored upright.
func (i *Image) Orient() *Image {
	return i.Transform(transformations.Orient{})
}

// Blur softens the image. amount is optional, 0 to 100, and defaults to 5.
func (i *Image) Blur(amount ...int) *Image {
	return i.Transform(transformations.Blur{Amount: clampRange(optionalInt(amount, 5), 0, 100)})
}

// Grayscale drops the colour and keeps the brightness.
func (i *Image) Grayscale() *Image {
	return i.Transform(transformations.Grayscale{})
}

// Sharpen makes edges read as crisper. amount is optional, 0 to 100, and
// defaults to 10.
func (i *Image) Sharpen(amount ...int) *Image {
	return i.Transform(transformations.Sharpen{Amount: clampRange(optionalInt(amount, 10), 0, 100)})
}

// FlipVertically flips the image top to bottom.
func (i *Image) FlipVertically() *Image {
	return i.Transform(transformations.FlipVertically{})
}

// FlipHorizontally flips the image left to right.
func (i *Image) FlipHorizontally() *Image {
	return i.Transform(transformations.FlipHorizontally{})
}

// Flip is the shorter name for [Image.FlipVertically].
func (i *Image) Flip() *Image { return i.FlipVertically() }

// Flop is the shorter name for [Image.FlipHorizontally].
func (i *Image) Flop() *Image { return i.FlipHorizontally() }

// Transform adds any transformation, including one this package did not
// write, onto the pipeline.
func (i *Image) Transform(t transformations.Transformation) *Image {
	return i.withClone(func(clone *Image) { clone.pipeline.Add(t) })
}

// Optimize sets a format and a quality in one call.
//
// Go has no default arguments and the two are of different types, so both
// are spelled out here; passing zero for the quality is what leaving it out
// would mean.
func (i *Image) Optimize(format string, quality int) (*Image, error) {
	out, err := i.ToFormat(format)
	if err != nil {
		return nil, err
	}
	return out.Quality(quality), nil
}

// Quality sets the output quality, 1 to 100, clamped to that range. Zero
// leaves it at [DefaultQuality].
func (i *Image) Quality(quality int) *Image {
	return i.withOutput(func(o *ImageOutputOptions) {
		if quality == 0 {
			o.Quality = 0
			return
		}
		o.Quality = clampRange(quality, 1, 100)
	})
}

// ToWebp sets the output format to WebP.
func (i *Image) ToWebp() *Image { return i.format("webp") }

// ToJpg sets the output format to JPEG.
func (i *Image) ToJpg() *Image { return i.format("jpg") }

// ToJpeg is the other spelling of [Image.ToJpg].
func (i *Image) ToJpeg() *Image { return i.ToJpg() }

// ToPng sets the output format to PNG.
func (i *Image) ToPng() *Image { return i.format("png") }

// ToGif sets the output format to GIF.
func (i *Image) ToGif() *Image { return i.format("gif") }

// ToAvif sets the output format to AVIF.
func (i *Image) ToAvif() *Image { return i.format("avif") }

// ToHeic sets the output format to HEIC.
func (i *Image) ToHeic() *Image { return i.format("heic") }

// ToBmp sets the output format to BMP.
func (i *Image) ToBmp() *Image { return i.format("bmp") }

// format is the eight named converters' shared body. They cannot fail --
// the name is a constant in this file and always passes [Image.ToFormat]'s
// check -- so they do not return the error that ToFormat does.
func (i *Image) format(name string) *Image {
	return i.withOutput(func(o *ImageOutputOptions) { o.Format = name })
}

// ToFormat sets the output format by name.
//
// "heif" is folded to "heic". Whether this package can write the format it
// accepted is a different question, settled at a different point: encoding
// to webp, avif, heic or bmp fails with a message naming the format, because
// there is no encoder for them in the standard library and a dependency for
// one is not a trade this module makes.
func (i *Image) ToFormat(format string) (*Image, error) {
	format = strings.ToLower(strings.TrimSpace(format))
	found := false
	for _, f := range outputFormats {
		if f == format {
			found = true
			break
		}
	}
	if !found {
		return nil, fail("the [%s] format is not supported", format)
	}
	if format == "heif" {
		format = "heic"
	}
	return i.format(format), nil
}

// Store writes the processed image to a disk under an invented name.
//
// The disk arrives as an argument, and it takes an auth.Grant because
// everything a customer stores does. The returned path is the key the image
// landed under, which a caller records in a row.
func (i *Image) Store(ctx context.Context, g auth.Grant, disk Disk, path string) (string, error) {
	name, err := i.HashName(ctx)
	if err != nil {
		return "", err
	}
	return i.StoreAs(ctx, g, disk, path, name)
}

// StorePublicly is [Image.Store] with the file made public afterwards.
//
// Public is about the store, never about authorization: a public file is one
// the object store will hand to anybody holding the URL, and the Policy that
// decided this caller may write it still ran.
func (i *Image) StorePublicly(ctx context.Context, g auth.Grant, disk Disk, path string) (string, error) {
	name, err := i.HashName(ctx)
	if err != nil {
		return "", err
	}
	return i.StorePubliclyAs(ctx, g, disk, path, name)
}

// StoreAs writes the processed image under a name the caller chose.
//
// An empty name means the path is used as the name.
func (i *Image) StoreAs(ctx context.Context, g auth.Grant, disk Disk, path, name string) (string, error) {
	body, err := i.ToBytes(ctx)
	if err != nil {
		return "", err
	}
	mediaType, err := i.MimeType(ctx)
	if err != nil {
		return "", err
	}
	key := joinKey(path, name)
	if err := disk.Put(ctx, g, key, bytes.NewReader(body), mediaType); err != nil {
		return "", failWith(err, "storing the image at %s", key)
	}
	return key, nil
}

// StorePubliclyAs is [Image.StoreAs] with the file made public afterwards.
func (i *Image) StorePubliclyAs(ctx context.Context, g auth.Grant, disk Disk, path, name string) (string, error) {
	key, err := i.StoreAs(ctx, g, disk, path, name)
	if err != nil {
		return "", err
	}
	if err := disk.SetVisibility(ctx, g, key, visibilityPublic); err != nil {
		return "", failWith(err, "making %s public", key)
	}
	return key, nil
}

// joinKey joins path and name with a slash, trimming any extra slashes at
// the edges and between them.
func joinKey(path, name string) string {
	if name == "" {
		path, name = "", path
	}
	return strings.Trim(strings.Trim(path, "/")+"/"+strings.Trim(name, "/"), "/")
}

// HashName returns a random forty-character name with the right extension on
// it, invented once and then remembered.
//
// path is an optional prefix. The extension comes from the processed bytes,
// which is why this can fail and why it takes a context: naming the file
// requires knowing what it turned out to be, and that runs the pipeline.
func (i *Image) HashName(ctx context.Context, path ...string) (string, error) {
	if i.hashName == "" {
		i.hashName = str.Random(40)
	}
	extension, err := i.Extension(ctx)
	if err != nil {
		return "", err
	}
	name := i.hashName + "." + extension
	if prefix := optionalString(path); prefix != "" {
		return prefix + "/" + name, nil
	}
	return name, nil
}

// ToBytes returns the image, processed.
//
// A pipeline with nothing in it is not run at all -- the bytes that arrived are
// the bytes that leave, undecoded and unre-encoded, which is what makes an
// upload that needed no work cost nothing, and why a picture too large to
// decode can still be stored untouched. A pipeline with something in it runs
// once, and what it produced replaces what arrived.
//
// The context bounds that run. A pipeline stopped part of the way through
// leaves the image as it was: nothing is written back, so the same call can be
// made again with a context that lasts.
func (i *Image) ToBytes(ctx context.Context) ([]byte, error) {
	if !i.pipeline.HasChanges() || i.processed {
		return i.contents, nil
	}
	driver, err := i.resolveDriver()
	if err != nil {
		return nil, err
	}
	out, err := driver.Process(ctx, i.contents, i.pipeline)
	if err != nil {
		return nil, err
	}
	i.contents = out
	i.pipeline = NewImagePipeline()
	i.processed = true
	i.mimeType, i.width, i.height = "", 0, 0
	return i.contents, nil
}

// ToBase64 returns the processed image, base64-encoded.
func (i *Image) ToBase64(ctx context.Context) (string, error) {
	body, err := i.ToBytes(ctx)
	if err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(body), nil
}

// ToDataURI returns the processed image as a data URI.
func (i *Image) ToDataURI(ctx context.Context) (string, error) {
	mediaType, err := i.MimeType(ctx)
	if err != nil {
		return "", err
	}
	encoded, err := i.ToBase64(ctx)
	if err != nil {
		return "", err
	}
	return "data:" + mediaType + ";base64," + encoded, nil
}

// ToString returns the data URI, the same as [Image.ToDataURI].
//
// This is not named String(), and Image does not implement fmt.Stringer:
// String() cannot return the error that processing the image can produce,
// and an image that printed an empty string because the encode failed would
// be worse than one that would not print.
func (i *Image) ToString(ctx context.Context) (string, error) { return i.ToDataURI(ctx) }

// Extension returns the file extension the processed image should carry,
// from what it actually is rather than from what it was called.
func (i *Image) Extension(ctx context.Context) (string, error) {
	mediaType, err := i.MimeType(ctx)
	if err != nil {
		return "", err
	}
	return extensionFor(mediaType), nil
}

// MimeType returns the media type of the processed image, read out of its
// bytes.
func (i *Image) MimeType(ctx context.Context) (string, error) {
	if i.mimeType != "" {
		return i.mimeType, nil
	}
	body, err := i.ToBytes(ctx)
	if err != nil {
		return "", err
	}
	i.mimeType = sniffMimeType(body)
	return i.mimeType, nil
}

// Dimensions returns the width and height of the processed image.
//
// It is read out of the header rather than out of the pixels, but the pipeline
// runs first -- the answer is about the image as it will be, not as it arrived
// -- which is what the context bounds.
func (i *Image) Dimensions(ctx context.Context) (int, int, error) {
	if i.width != 0 && i.height != 0 {
		return i.width, i.height, nil
	}
	body, err := i.ToBytes(ctx)
	if err != nil {
		return 0, 0, err
	}
	driver, err := i.resolveDriver()
	if err != nil {
		return 0, 0, err
	}
	w, h, err := driver.Dimensions(body)
	if err != nil {
		return 0, 0, err
	}
	i.width, i.height = w, h
	return w, h, nil
}

// Width returns the width of the processed image.
func (i *Image) Width(ctx context.Context) (int, error) {
	w, _, err := i.Dimensions(ctx)
	return w, err
}

// Height returns the height of the processed image.
func (i *Image) Height(ctx context.Context) (int, error) {
	_, h, err := i.Dimensions(ctx)
	return h, err
}

// DominantColor returns the average colour as "#rrggbb".
//
// The pipeline runs over a copy before sampling, so that the colour reports
// for the image as it will be and not as it arrived, and leaves the image
// itself unprocessed.
func (i *Image) DominantColor(ctx context.Context) (string, error) {
	driver, err := i.resolveDriver()
	if err != nil {
		return "", err
	}
	body := i.contents
	if i.pipeline.HasChanges() && !i.processed {
		sample := i.newClone()
		if body, err = sample.ToBytes(ctx); err != nil {
			return "", err
		}
	}
	return driver.DominantColor(ctx, body)
}

// Using sets the driver, by the name it was registered under, for this image
// only.
func (i *Image) Using(driver string) *Image {
	clone := i.newClone()
	clone.driver = driver
	return clone
}

// File returns the upload this image was made from, or nil when it came from
// anywhere else.
func (i *Image) File() UploadedFile { return i.file }

// ToResponse writes the processed image to w: its media type, its length,
// and its bytes. Nothing else is set -- caching and disposition are the
// caller's to decide, and guessing them here would be a second place where
// they are decided.
//
// This is the one place that takes no context of its own: the request carries
// one already, and it is the right one -- a browser that goes away while the
// thumbnail is being resampled cancels it, and the resize stops. A nil request
// means there is nobody to go away, and the work runs unbounded.
func (i *Image) ToResponse(w http.ResponseWriter, r *http.Request) error {
	ctx := context.Background()
	if r != nil {
		ctx = r.Context()
	}
	body, err := i.ToBytes(ctx)
	if err != nil {
		return err
	}
	mediaType, err := i.MimeType(ctx)
	if err != nil {
		return err
	}
	w.Header().Set("Content-Type", mediaType)
	w.Header().Set("Content-Length", strconv.Itoa(len(body)))
	w.WriteHeader(http.StatusOK)
	if r != nil && r.Method == http.MethodHead {
		return nil
	}
	if _, err := w.Write(body); err != nil {
		return failWith(err, "writing the image to the response")
	}
	return nil
}

// resolveDriver returns the manager's driver for this image, by name.
func (i *Image) resolveDriver() (Driver, error) {
	if i.manager == nil {
		return nil, fail("this image has no manager, so it has no driver (build images with ImageManager)")
	}
	return i.manager.Driver(i.driver)
}

func optionalString(values []string) string {
	if len(values) > 0 {
		return values[0]
	}
	return ""
}

func optionalInt(values []int, fallback int) int {
	if len(values) > 0 {
		return values[0]
	}
	return fallback
}

func clampRange(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
