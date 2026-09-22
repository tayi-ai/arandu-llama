package image

import (
	"bytes"
	stdimage "image"
	"image/color"
	"image/gif"
	"image/jpeg"
	"image/png"
	"strings"
)

// outputFormats are the names [Image.ToFormat] accepts, in order. Accepting a
// name here is not a promise that this driver can write it -- see
// [encodeCanvas], which says so at the point where it would have to.
var outputFormats = []string{"webp", "jpg", "jpeg", "png", "gif", "avif", "heic", "heif", "bmp"}

// inputTypes are the media types this driver agrees to open.
var inputTypes = []string{
	"image/jpeg", "image/png", "image/bmp", "image/gif", "image/webp",
	"image/avif", "image/x-avif", "image/heic", "image/x-heic", "image/heif",
}

// sniffMimeType returns the media type read out of the bytes themselves,
// never out of a filename or a header somebody sent.
//
// It knows the types [Image.Extension] can name, and returns
// "application/octet-stream" for anything else, which is what makes
// Extension's last case, "bin", reachable.
func sniffMimeType(b []byte) string {
	switch {
	case len(b) >= 3 && b[0] == 0xFF && b[1] == 0xD8 && b[2] == 0xFF:
		return "image/jpeg"
	case len(b) >= 8 && string(b[:8]) == "\x89PNG\r\n\x1a\n":
		return "image/png"
	case len(b) >= 6 && (string(b[:6]) == "GIF87a" || string(b[:6]) == "GIF89a"):
		return "image/gif"
	case len(b) >= 12 && string(b[:4]) == "RIFF" && string(b[8:12]) == "WEBP":
		return "image/webp"
	case len(b) >= 2 && b[0] == 'B' && b[1] == 'M':
		return "image/bmp"
	case len(b) >= 4 && (string(b[:4]) == "II*\x00" || string(b[:4]) == "MM\x00*"):
		return "image/tiff"
	case len(b) >= 12 && string(b[4:8]) == "ftyp":
		switch string(b[8:12]) {
		case "avif", "avis":
			return "image/avif"
		case "heic", "heix", "heim", "heis", "hevc", "hevx", "hevm", "hevs", "mif1", "msf1":
			return "image/heic"
		}
	}
	if head := strings.TrimLeft(string(firstBytes(b, 256)), " \t\r\n"); strings.HasPrefix(head, "<svg") || strings.HasPrefix(head, "<?xml") {
		return "image/svg+xml"
	}
	return "application/octet-stream"
}

func firstBytes(b []byte, n int) []byte {
	if len(b) < n {
		return b
	}
	return b[:n]
}

// supportedInput reports whether this driver will open this media type at
// all.
func supportedInput(mediaType string) bool {
	for _, t := range inputTypes {
		if t == mediaType {
			return true
		}
	}
	return false
}

// DefaultMaxPixels is how many pixels an image may declare before this package
// refuses to decode it.
//
// The number is a memory bound and not a photographic one. The canvas every
// transformation runs on is four bytes a pixel, so this ceiling of 33,554,432
// pixels is a canvas of exactly 128 MiB -- and a file declaring more asks for
// that memory before a single pixel of it has been read, which is what makes an
// image somebody else chose the dimensions of worth bounding at all.
//
// It sits above every camera in ordinary use, since a 24-megapixel frame is
// 6000 by 4000 and an 8K frame is 7680 by 4320, and far below what a crafted
// file declares. Raise it with [ImageManager.MaxPixels] where the larger
// picture is the job.
const DefaultMaxPixels = 32 << 20

// refuseOversize reads the header, and only the header, and refuses an image
// whose declared dimensions come to more than maxPixels.
//
// This is the whole of the ceiling, and where it runs is the point of it:
// image.DecodeConfig parses the handful of bytes that carry the dimensions and
// allocates nothing for the pixels, so a file declaring 50000 by 50000 is
// refused for the ten gigabytes it was about to ask for rather than after
// asking for them.
//
// A header that cannot be read is left to the decoder rather than refused
// here. The decoder reads the same bytes through the same parser and fails on
// them before it allocates a canvas, and the error it produces names the media
// type that was detected, which this one could not.
func refuseOversize(b []byte, maxPixels int) error {
	cfg, _, err := stdimage.DecodeConfig(bytes.NewReader(b))
	if err != nil {
		return nil
	}
	if int64(cfg.Width)*int64(cfg.Height) > int64(maxPixels) {
		return tooLarge(cfg.Width, cfg.Height, maxPixels)
	}
	return nil
}

// decodeCanvas opens the bytes onto the working canvas and says which format
// they were in, so that an image nobody asked to convert is written back as
// what it was.
//
// The ceiling is checked first, and this is the only place in the package
// where pixels are allocated, so an image past it never reaches memory.
func decodeCanvas(b []byte, maxPixels int) (*stdimage.RGBA, string, error) {
	if err := refuseOversize(b, maxPixels); err != nil {
		return nil, "", err
	}
	src, format, err := stdimage.Decode(bytes.NewReader(b))
	if err != nil {
		return nil, "", failWith(err, "the image could not be decoded (this driver reads jpeg, png and gif; %s was detected)", sniffMimeType(b))
	}
	return toRGBA(src), format, nil
}

// encodeCanvas writes the canvas back out.
//
// Four of the nine accepted names are written -- jpg and jpeg are one format,
// so they are three formats -- and the other five fail here with a message that
// says so. That is deliberate: WebP, AVIF, HEIC and BMP have no encoder in the
// standard library, and the packages that do have one are dependencies this
// module does not carry. Failing at the encode with the format named is the
// honest version of that -- the alternative is
// [Image.ToWebp] quietly handing back a JPEG, which is a lie that reaches
// production as a broken Content-Type.
func encodeCanvas(canvas *stdimage.RGBA, format string, quality int) ([]byte, error) {
	var buf bytes.Buffer
	switch format {
	case "jpg", "jpeg":
		// JPEG has no alpha channel. Compositing over white first is what
		// every other encoder does with transparency, and skipping it paints
		// every transparent pixel black.
		opaque := flatten(canvas, color.RGBA{R: 255, G: 255, B: 255, A: 255})
		if err := jpeg.Encode(&buf, opaque, &jpeg.Options{Quality: quality}); err != nil {
			return nil, failWith(err, "encoding jpeg")
		}
	case "png":
		// No quality is passed to the PNG encoder: the format is lossless and
		// the number would mean compression effort, not fidelity.
		enc := png.Encoder{CompressionLevel: png.DefaultCompression}
		if err := enc.Encode(&buf, canvas); err != nil {
			return nil, failWith(err, "encoding png")
		}
	case "gif":
		if err := gif.Encode(&buf, canvas, nil); err != nil {
			return nil, failWith(err, "encoding gif")
		}
	case "webp", "avif", "heic", "heif", "bmp":
		return nil, fail("the [%s] format cannot be written: this driver writes jpeg, png and gif", format)
	default:
		return nil, fail("the [%s] format is not supported", format)
	}
	return buf.Bytes(), nil
}

// extensionFor maps a media type to a file extension, kept in one place
// because [Image.HashName] and [Image.Extension] both need it.
func extensionFor(mediaType string) string {
	switch mediaType {
	case "image/jpeg":
		return "jpg"
	case "image/png":
		return "png"
	case "image/gif":
		return "gif"
	case "image/webp":
		return "webp"
	case "image/avif", "image/x-avif":
		return "avif"
	case "image/heic", "image/x-heic", "image/heif":
		return "heic"
	case "image/bmp":
		return "bmp"
	case "image/svg+xml":
		return "svg"
	case "image/tiff":
		return "tiff"
	default:
		return "bin"
	}
}
