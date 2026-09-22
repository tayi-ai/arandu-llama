package image

import (
	"bytes"
	"context"
	stdimage "image"
	"image/color"
	"math"
	"sync"

	"github.com/arandu-io/hesape/image/transformations"
)

// TransformationHandler is the callable a driver registers to handle one
// transformation: it is handed the working canvas and the transformation that
// asked for it, and returns the canvas that replaces it.
//
// The canvas is the standard library's *image.RGBA, alpha premultiplied. A
// handler may return the canvas it was given, modified in place, or a new one
// of a different size.
//
// The context is the one the caller asked for bytes with, and a handler whose
// work is measured in pixels is expected to stop when it is done: the pixel
// loops either side of the handler do, and a pipeline is only as cancellable as
// its slowest step.
type TransformationHandler func(ctx context.Context, canvas *stdimage.RGBA, t transformations.Transformation) (*stdimage.RGBA, error)

// Driver is the thing that actually touches pixels.
//
// This package's own driver does the work, in Go, out of image/jpeg,
// image/png and image/gif in the standard library. The interface stays
// because registering a transformation handler and [ImageManager.Extend]
// both need something to be an implementation of.
type Driver interface {
	// Process runs the pipeline over the bytes and returns the encoded result.
	// The context bounds the pixel work: a transformation the caller stopped
	// waiting for returns the context's error rather than finishing.
	Process(ctx context.Context, contents []byte, pipeline *ImagePipeline) ([]byte, error)
	// Dimensions returns the width and height of the image in the bytes. It
	// takes no context because it reads the header and never the pixels, which
	// is a bounded read of a few dozen bytes.
	Dimensions(contents []byte) (int, int, error)
	// DominantColor returns the average colour as "#rrggbb". It reads every
	// pixel, so it takes a context as [Driver.Process] does.
	DominantColor(ctx context.Context, contents []byte) (string, error)
	// TransformUsing registers a handler for one transformation, named by its
	// TransformationName. It returns the driver, so registrations chain.
	TransformUsing(transformation string, handler TransformationHandler) Driver
	// MaxPixels sets how many pixels an image may declare before this driver
	// refuses to decode it, and returns the driver so calls chain. Zero or less
	// means the driver's own default.
	//
	// An implementation is expected to read the declared dimensions out of the
	// header and refuse past this ceiling before it allocates the pixels --
	// which is the only point at which refusing them is worth anything, since
	// the memory a crafted file asks for is spent by the decode itself.
	MaxPixels(pixels int) Driver
}

// stdDriver is the one driver this package carries.
//
// It is unexported on purpose: there is no natural name to give it, and a
// caller needs no name for it either. What a caller needs is reachable
// without one: [NewImageManager] installs it, [ImageManager.Driver] hands it
// back, and [ImageManager.Extend] is how a different one gets in.
type stdDriver struct {
	mu        sync.RWMutex
	handlers  map[string]TransformationHandler
	maxPixels int
}

func newStdDriver() *stdDriver {
	return &stdDriver{handlers: map[string]TransformationHandler{}}
}

// TransformUsing implements [Driver.TransformUsing].
func (d *stdDriver) TransformUsing(transformation string, handler TransformationHandler) Driver {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.handlers == nil {
		d.handlers = map[string]TransformationHandler{}
	}
	d.handlers[transformation] = handler
	return d
}

// MaxPixels implements [Driver.MaxPixels].
func (d *stdDriver) MaxPixels(pixels int) Driver {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.maxPixels = pixels
	return d
}

// pixelLimit is the ceiling this driver decodes under, with the default filled
// in here rather than at the setter: a driver nobody configured and one told to
// use the default are the same driver, and reading it in one place is what
// makes them so.
func (d *stdDriver) pixelLimit() int {
	d.mu.RLock()
	defer d.mu.RUnlock()
	if d.maxPixels <= 0 {
		return DefaultMaxPixels
	}
	return d.maxPixels
}

// handlerFor returns the handler registered for name, or nil when none was.
func (d *stdDriver) handlerFor(name string) TransformationHandler {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.handlers[name]
}

// Process implements [Driver.Process].
func (d *stdDriver) Process(ctx context.Context, contents []byte, pipeline *ImagePipeline) ([]byte, error) {
	if err := interrupted(ctx); err != nil {
		return nil, err
	}

	mediaType := sniffMimeType(contents)
	if !supportedInput(mediaType) {
		return nil, fail("the image format [%s] is not supported", mediaType)
	}

	canvas, sourceFormat, err := decodeCanvas(contents, d.pixelLimit())
	if err != nil {
		return nil, err
	}

	// The orientation is read from the bytes as they arrived, before anything
	// is applied, because re-encoding drops the EXIF block: a pipeline that
	// resized before it oriented would have nothing left to read.
	orientation := exifOrientation(contents)

	for _, t := range pipeline.Transformations {
		if handler := d.handlerFor(t.TransformationName()); handler != nil {
			canvas, err = handler(ctx, canvas, t)
			if err != nil {
				return nil, failWith(err, "the [%s] transformation failed", t.TransformationName())
			}
			continue
		}
		canvas, err = d.apply(ctx, canvas, t, orientation)
		if err != nil {
			return nil, err
		}
	}

	// Checked once more before the encode, which is the one long step left and
	// the one that has no place to check inside it: a pipeline abandoned during
	// its last transformation should not go on to compress the result of it.
	if err := interrupted(ctx); err != nil {
		return nil, err
	}

	format := pipeline.Output.Format
	if format == "" {
		format = sourceFormat
	}
	return encodeCanvas(canvas, format, pipeline.Output.quality())
}

// apply runs one transformation onto the canvas, or returns an error naming
// the one nobody taught this driver.
func (d *stdDriver) apply(ctx context.Context, canvas *stdimage.RGBA, t transformations.Transformation, orientation int) (*stdimage.RGBA, error) {
	switch v := t.(type) {
	case transformations.Orient:
		return applyOrientation(canvas, orientation), nil
	case transformations.Cover:
		return cover(ctx, canvas, v.Width, v.Height)
	case transformations.Contain:
		bg, err := resolveBackground(ctx, canvas, v.Background)
		if err != nil {
			return nil, err
		}
		return contain(ctx, canvas, v.Width, v.Height, bg)
	case transformations.Crop:
		return crop(canvas, v.X, v.Y, v.Width, v.Height), nil
	case transformations.Resize:
		return resize(ctx, canvas, v.Width, v.Height)
	case transformations.Rotate:
		bg, err := resolveBackground(ctx, canvas, v.Background)
		if err != nil {
			return nil, err
		}
		return rotate(ctx, canvas, v.Angle, bg)
	case transformations.Scale:
		return scaleDown(ctx, canvas, v.Width, v.Height)
	case transformations.Blur:
		return blur(ctx, canvas, blurRadius(v.Amount))
	case transformations.Grayscale:
		return grayscale(ctx, canvas)
	case transformations.Sharpen:
		return sharpen(ctx, canvas, v.Amount)
	case transformations.FlipVertically:
		return flipVertical(canvas), nil
	case transformations.FlipHorizontally:
		return flipHorizontal(canvas), nil
	default:
		return nil, fail("the image transformation [%s] is not supported", t.TransformationName())
	}
}

// Dimensions implements [Driver.Dimensions].
func (d *stdDriver) Dimensions(contents []byte) (int, int, error) {
	cfg, _, err := stdimage.DecodeConfig(bytes.NewReader(contents))
	if err != nil {
		return 0, 0, failWith(err, "unable to determine the dimensions of the image")
	}
	return cfg.Width, cfg.Height, nil
}

// DominantColor implements [Driver.DominantColor].
//
// Averaging every pixel directly reaches the same result as resizing the
// image down to a single pixel and reading it, without the intermediate
// canvas or the rounding the resize would add.
func (d *stdDriver) DominantColor(ctx context.Context, contents []byte) (string, error) {
	if err := interrupted(ctx); err != nil {
		return "", err
	}
	canvas, _, err := decodeCanvas(contents, d.pixelLimit())
	if err != nil {
		return "", err
	}
	average, err := averageColor(ctx, canvas)
	if err != nil {
		return "", err
	}
	return hex(average), nil
}

// resolveBackground resolves a background spelling to a colour, including the
// "dominant" sentinel that stands for the image's own average colour.
//
// An empty background defaults to white rather than transparency: a rotate
// that left the corners clear would look right in a PNG and black in a JPEG,
// and white is the choice that looks the same in both.
func resolveBackground(ctx context.Context, canvas *stdimage.RGBA, background string) (color.RGBA, error) {
	switch background {
	case "":
		return color.RGBA{R: 255, G: 255, B: 255, A: 255}, nil
	case "dominant":
		return averageColor(ctx, canvas)
	default:
		return parseHexColor(background)
	}
}

// cover fills the box edge to edge, keeping the aspect ratio and cropping what
// overflows from the centre.
func cover(ctx context.Context, src *stdimage.RGBA, w, h int) (*stdimage.RGBA, error) {
	sw, sh := src.Rect.Dx(), src.Rect.Dy()
	if sw == 0 || sh == 0 || w < 1 || h < 1 {
		return src, nil
	}
	scale := math.Max(float64(w)/float64(sw), float64(h)/float64(sh))
	rw := maxInt(1, int(math.Round(float64(sw)*scale)))
	rh := maxInt(1, int(math.Round(float64(sh)*scale)))
	resized, err := resample(ctx, src, rw, rh)
	if err != nil {
		return nil, err
	}
	return crop(resized, (rw-w)/2, (rh-h)/2, w, h), nil
}

// contain fits the whole image inside the box, keeping the aspect ratio, and
// pads the rest with bg so the result is exactly w by h.
func contain(ctx context.Context, src *stdimage.RGBA, w, h int, bg color.RGBA) (*stdimage.RGBA, error) {
	sw, sh := src.Rect.Dx(), src.Rect.Dy()
	if sw == 0 || sh == 0 || w < 1 || h < 1 {
		return src, nil
	}
	scale := math.Min(float64(w)/float64(sw), float64(h)/float64(sh))
	rw := maxInt(1, int(math.Round(float64(sw)*scale)))
	rh := maxInt(1, int(math.Round(float64(sh)*scale)))
	resized, err := resample(ctx, src, rw, rh)
	if err != nil {
		return nil, err
	}

	dst := stdimage.NewRGBA(stdimage.Rect(0, 0, w, h))
	fill(dst, bg)
	at := stdimage.Rect((w-rw)/2, (h-rh)/2, (w-rw)/2+rw, (h-rh)/2+rh)
	drawOver(dst, at, resized)
	return dst, nil
}

// resize sets the canvas to exactly the dimensions given, aspect ratio not
// kept; an axis left at zero keeps the size it had.
func resize(ctx context.Context, src *stdimage.RGBA, w, h int) (*stdimage.RGBA, error) {
	sw, sh := src.Rect.Dx(), src.Rect.Dy()
	if w < 1 {
		w = sw
	}
	if h < 1 {
		h = sh
	}
	return resample(ctx, src, w, h)
}

// scaleDown fits the canvas inside the box: the aspect ratio is kept, the
// result fits inside the box, and an image already smaller than the box is
// left alone.
func scaleDown(ctx context.Context, src *stdimage.RGBA, w, h int) (*stdimage.RGBA, error) {
	sw, sh := src.Rect.Dx(), src.Rect.Dy()
	if sw == 0 || sh == 0 {
		return src, nil
	}
	scale := 1.0
	if w > 0 {
		scale = math.Min(scale, float64(w)/float64(sw))
	}
	if h > 0 {
		scale = math.Min(scale, float64(h)/float64(sh))
	}
	if scale >= 1 {
		return src, nil
	}
	return resample(ctx, src,
		maxInt(1, int(math.Round(float64(sw)*scale))),
		maxInt(1, int(math.Round(float64(sh)*scale))))
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
