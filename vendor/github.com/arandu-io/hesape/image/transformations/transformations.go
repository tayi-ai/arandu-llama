package transformations

// Transformation is the interface every transformation type implements.
//
// An empty interface in Go is `any`, which every type satisfies, so a marker
// interface needs at least one method -- and the method it got is the one the
// driver needs anyway.
//
// TransformationName is the key a per-driver handler is registered under with
// ImageManager.TransformUsing, and the tag the driver switches on:
//
//	manager.TransformUsing("std", "Blur", handler)
//
// A type outside this package may implement it, and that is how a custom
// transformation reaches a driver that was taught to handle it. A driver that
// meets a name it does not know returns an error rather than dropping the
// transformation on the floor -- a resize that silently did not happen is worse
// than one that failed.
type Transformation interface {
	// TransformationName is the string that names this transformation, such
	// as "Blur", "Cover", or "Crop".
	TransformationName() string
}

// Blur softens the whole canvas: every pixel is averaged with the ones around
// it, so noise and fine detail go and the broad shapes stay. There is no way to
// blur part of an image -- crop first if that is what you want.
//
// Amount is a strength, not a distance in pixels, and the driver decides how
// far it reaches. Zero is a no-op: the canvas comes out of the pipeline
// untouched rather than rejected.
type Blur struct {
	// Amount is 0 to 100. Image.Blur clamps a caller's value to that range
	// before building this transformation.
	Amount int
}

// TransformationName returns "Blur".
func (Blur) TransformationName() string { return "Blur" }

// Contain scales the whole image to fit inside the box, padded with
// Background to exactly Width by Height.
type Contain struct {
	Width  int
	Height int
	// Background is a CSS-style hex colour ("#ffffff", "fff", "ffffffff") or
	// the sentinel "dominant", which expands to the image's own dominant
	// colour. Empty means opaque white.
	Background string
}

// TransformationName returns "Contain".
func (Contain) TransformationName() string { return "Contain" }

// Cover fills the box edge to edge, aspect ratio kept, whatever overflows
// cropped from the centre.
type Cover struct {
	Width  int
	Height int
}

// TransformationName returns "Cover".
func (Cover) TransformationName() string { return "Cover" }

// Crop cuts a rectangle out of the image and throws the rest away. Nothing is
// scaled: the result is Width by Height at the original resolution, taken from
// the source starting at X, Y, which are counted from the top left corner.
//
// The rectangle is not required to fit inside the source. Whatever part of it
// hangs over an edge has no pixels to take, and that part of the result stays
// transparent.
type Crop struct {
	Width  int
	Height int
	X      int
	Y      int
}

// TransformationName returns "Crop".
func (Crop) TransformationName() string { return "Crop" }

// FlipHorizontally mirrors the image across its vertical centre line: the
// leftmost column of pixels becomes the rightmost. The dimensions do not
// change, no pixel is interpolated, and applying it twice gives the original
// back.
type FlipHorizontally struct{}

// TransformationName returns "FlipHorizontally".
func (FlipHorizontally) TransformationName() string { return "FlipHorizontally" }

// FlipVertically flips the image top to bottom.
type FlipVertically struct{}

// TransformationName returns "FlipVertically".
func (FlipVertically) TransformationName() string { return "FlipVertically" }

// Grayscale drops the colour and keeps the brightness: every pixel becomes the
// grey of its own luminance. Transparency survives untouched, and the image
// stays in a colour format -- this makes the picture grey, it does not make the
// file smaller by storing one channel instead of three.
//
// It is not reversible. The colour is gone from the pipeline's output, so an
// image needed in both forms has to be derived twice from the original.
type Grayscale struct{}

// TransformationName returns "Grayscale".
func (Grayscale) TransformationName() string { return "Grayscale" }

// Orient reads the EXIF orientation tag and applies it, so a photograph taken
// with the phone on its side is stored the way it was seen.
type Orient struct{}

// TransformationName returns "Orient".
func (Orient) TransformationName() string { return "Orient" }

// Resize sets the image to exactly Width by Height; the aspect ratio is not
// kept.
type Resize struct {
	// Width zero leaves that axis as it is, and the other one moves alone.
	// Image.Resize refuses the call where both are absent.
	Width int
	// Height zero leaves that axis as it is.
	Height int
}

// TransformationName returns "Resize".
func (Resize) TransformationName() string { return "Resize" }

// Rotate turns the image around its centre. The canvas grows to the bounding
// box of the turned rectangle, so no corner is cut off -- a square rotated by
// 45 degrees comes back wider and taller than it went in, and only a multiple
// of 90 degrees keeps the dimensions (swapped, for the odd multiples).
//
// The turn opens up triangles of empty canvas at the corners, and Background is
// what goes in them. A right angle is a straight copy of the pixels; any other
// angle samples between them, so repeated rotation of the same image softens
// it.
type Rotate struct {
	// Angle is degrees clockwise.
	Angle float64
	// Background fills the corners the rotation leaves empty. Same spelling as
	// [Contain.Background], including the "dominant" sentinel; empty means
	// transparent.
	Background string
}

// TransformationName returns "Rotate".
func (Rotate) TransformationName() string { return "Rotate" }

// Scale fits the image inside the box, aspect ratio kept, and never scales up.
//
// Never scaling up is the whole difference from Resize.
type Scale struct {
	// Width zero means that axis does not constrain the result.
	Width int
	// Height zero means that axis does not constrain the result.
	Height int
}

// TransformationName returns "Scale".
func (Scale) TransformationName() string { return "Scale" }

// Sharpen makes edges read as crisper by pushing the contrast up where the
// image already changes fast. It adds no detail that was not there: what it
// works on is the difference between the image and a softened copy of itself,
// so it recovers the look of definition a resize cost and nothing more.
//
// Amount is how much of that difference is added back, and a high value shows
// as a bright halo tracing every edge. Zero is a no-op.
type Sharpen struct {
	// Amount ranges from 0 to 100.
	Amount int
}

// TransformationName returns "Sharpen".
func (Sharpen) TransformationName() string { return "Sharpen" }
