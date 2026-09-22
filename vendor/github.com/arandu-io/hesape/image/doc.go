// Package image describes an image by what should be done to it, done to it
// once, when somebody asks for the result.
//
// [ImageManager] is where images come from, [Image] is one image and every
// fluent call on it returns a new one, and the pixels are touched exactly
// once, by a [Driver], at [Image.ToBytes]:
//
//	images := image.NewImageManager()
//
//	img, err := images.FromUpload(upload)
//	thumb, err := img.Orient().Cover(400, 300).ToJpg().Quality(80).
//		Store(ctx, grant, disk, "avatars")
//
// # No third-party dependency
//
// The driver here is this package, built on image/jpeg, image/png and
// image/gif from the standard library, which add no third-party dependency.
// The [Driver] interface stays: registering a custom transformation handler
// and [ImageManager.Extend] both need something to be an implementation of,
// and optional codecs live in their own module rather than adding dependencies
// to every consumer of this package.
//
// The optional github.com/arandu-io/hesape/image/drivers/raster module adds
// static SVG rasterization and lossless WebP encoding through ImageManager.Extend.
// It also reads static WebP. PNG/WebP retain alpha; unsupported SVG features
// and animated inputs are refused. See that module's package documentation for
// registration, limits, format coverage and third-party license notices.
//
// # Which formats
//
// Standard driver: read JPEG, PNG, GIF; write JPEG, PNG, GIF.
//
// [Image.ToFormat] accepts nine names -- webp, jpg, jpeg, png, gif, avif,
// heic, heif, bmp. Four of them are written, and they are three formats, since
// jpg and jpeg are one. The other five -- webp, avif, heic, heif, bmp -- have
// no encoder in the standard library and fail at the encode, with the format
// named. That is the honest arrangement: the alternative is [Image.ToWebp]
// handing back a JPEG, which is a lie that leaves the building as a wrong
// Content-Type. The same goes for reading: a WebP arriving at [Image.ToBytes]
// is refused with its media type named rather than half-decoded.
//
// [Image.MimeType] and [Image.Extension] know the wider list, because they
// read what a file is rather than what this package can do with it -- an
// image that only needs storing never goes near the driver, and its type and
// extension are still right.
//
// # A ceiling before the decode
//
// An image is decoded only after its header has been read and the dimensions it
// declares have been found to be under [DefaultMaxPixels], or under whatever
// [ImageManager.MaxPixels] was told instead. Past it the decode does not
// happen and the failure is [ErrTooLarge], naming the dimensions and the limit.
//
// The order is the whole of it. The canvas is four bytes a pixel, so a file of
// a few dozen bytes declaring 50000 by 50000 asks for ten gigabytes, and asks
// for it inside the decoder, before anything this package wrote gets to look at
// the result. Reading the header first costs the header.
//
// What the ceiling does not cover is stated as plainly: it bounds what is
// decoded, not what a transformation is asked to produce. A resize to
// dimensions the application chose allocates what the application asked for,
// and an image nobody asked to transform is never decoded at all -- it can be
// stored, and its header can be read, at any size.
//
// # Stopping work nobody is waiting for
//
// Every method that reaches the pixels takes a context, and the pixel loops
// under it read that context as they go, once a line. A resize the caller
// stopped waiting for returns the context's error rather than finishing, and
// what it had built is dropped -- the Image is left exactly as it was, so the
// same call can be made again.
//
// [Image.ToResponse] is the one that takes no context, because the request it
// is handed already carries one: a browser that goes away mid-resize cancels
// it.
//
// # Resizing, and why it looks the way it does
//
// [Image.Resize], [Image.Cover], [Image.Contain] and [Image.Scale] resample
// with a triangle filter -- bilinear -- applied as two separable passes, and
// with one addition that matters: when the image is being made smaller, the
// filter's support is widened by the scale factor, which turns the same code
// into an area average.
//
// Plain bilinear reads at most two source pixels per output pixel. Reducing a
// 4000-pixel photograph to 400 with it ignores nine of every ten columns, and
// what comes out is not soft but wrong: brick walls turn into moire, small text
// turns into gravel. Nearest-neighbour, which is what an afternoon's work
// produces, is worse still. Widening the support costs a few lines and is the
// difference between a thumbnail somebody ships and one somebody complains
// about.
//
// [Image.Rotate] samples bilinearly too, except at the three right angles,
// where it copies pixels instead.
//
// # Orient, and the photograph that arrives sideways
//
// [Image.Orient] reads the EXIF orientation tag out of the bytes as they
// arrived and turns the image to match. A phone does not rotate its sensor when
// the person rotates the phone: it stores the frame as the sensor read it and
// writes down which way was up. Every upload form that skips this shows half
// its portrait photographs lying down, and it is the most reported bug any
// upload form has.
//
// The tag is read before any transformation runs, because re-encoding drops the
// EXIF block: a pipeline that resized first would have nothing left to read.
// The output carries no EXIF, which is the point -- the turn is baked into the
// pixels, so every viewer agrees.
//
// # Storing carries the Grant, and the read does too
//
// [Image.Store], its neighbours and [ImageManager.FromStorage] take an
// auth.Grant and a [Disk], and hand the Grant straight on. Nothing here builds a
// key, so the tenant the image lands under is the one the disk reads off the
// Grant, and no argument of these methods can name another.
//
// What the Grant proves is the tenant, not that a Policy ran: auth.Authorize is
// one exported way to hold one and auth.SystemGrant is another. A caller holding
// the second stores an image no Policy was asked about, and that is reported by
// `aru doctor` rather than refused here.
//
// [Disk] and [DiskReader] are interfaces rather than imports of the
// filesystem package, so that this package sits underneath it. A
// *filesystem.Disk is a [Disk] with no adapter.
package image
