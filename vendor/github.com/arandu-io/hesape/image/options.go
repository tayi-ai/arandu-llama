package image

// DefaultQuality is what the driver encodes with when nobody called
// [Image.Quality]: seventy.
const DefaultQuality = 70

// ImageOutputOptions is what the image is written as, once every
// transformation has run.
//
// The zero value means no output was requested: an empty Format means
// "encode it as whatever it arrived as", and a zero Quality means
// [DefaultQuality]. That is why [ImageOutputOptions.HasChanges] exists: a
// pipeline with no transformations and no output options is a pipeline the
// [Image] can skip entirely, handing back the original bytes untouched
// instead of decoding and re-encoding them for nothing.
type ImageOutputOptions struct {
	// Format is one of the names [Image.ToFormat] settles on, which is the
	// nine it accepts with heif already folded to heic: webp, jpg, jpeg, png,
	// gif, avif, heic, bmp. Empty keeps the source format.
	Format string
	// Quality is 1 to 100. Zero means [DefaultQuality]. It reaches only the
	// lossy encoders; PNG and GIF ignore it.
	Quality int
}

// HasChanges reports whether anybody asked for a format or a quality.
func (o ImageOutputOptions) HasChanges() bool {
	return o.Format != "" || o.Quality != 0
}

// quality is the number the encoder actually gets.
func (o ImageOutputOptions) quality() int {
	if o.Quality == 0 {
		return DefaultQuality
	}
	return o.Quality
}
