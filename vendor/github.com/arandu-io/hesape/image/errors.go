package image

import (
	"errors"
	"fmt"
)

// ErrImage is the sentinel every error this package returns wraps.
//
// This package can fail for many reasons -- an unsupported format, an
// undecodable stream, a resize with no dimension -- and every error it
// returns wraps this one sentinel, so a caller writes a single check:
//
//	if errors.Is(err, image.ErrImage) { ... }
//
// It is a sentinel and not a struct type for the reason
// model.ErrModelNotFound is one: the caller wants to know which failure it
// is from the message, and to classify it from the sentinel, and a struct would
// make it name a type to do either.
var ErrImage = errors.New("image")

// ErrTooLarge means the image declared more pixels than the ceiling allows and
// was refused before it was decoded.
//
// It wraps [ErrImage], so a caller classifying every failure of this package
// with one check still catches it. It is a second sentinel because this is the
// one failure a caller usually answers differently: an upload refused for the
// size it declared is a message to the person who sent it, not a fault in the
// application.
var ErrTooLarge = fmt.Errorf("%w: the image is too large", ErrImage)

// fail builds a wrapped [ErrImage] with the given message.
func fail(format string, args ...any) error {
	return fmt.Errorf("%w: %s", ErrImage, fmt.Sprintf(format, args...))
}

// failWith is fail for an error that already exists, keeping both in the chain:
// errors.Is finds ErrImage and the cause both.
func failWith(cause error, format string, args ...any) error {
	return fmt.Errorf("%w: %s: %w", ErrImage, fmt.Sprintf(format, args...), cause)
}

// tooLarge builds the refusal [ErrTooLarge] carries, with the dimensions the
// header declared and the ceiling they passed.
//
// The count is computed in int64 because each dimension can be as large as a
// signed 32-bit maximum, and the product of two of those overflows an int on a
// 32-bit build -- where it would come out negative, compare below any ceiling,
// and let through exactly the image this exists to refuse.
func tooLarge(width, height, limit int) error {
	return fmt.Errorf("%w: %d by %d is %d pixels and the limit is %d",
		ErrTooLarge, width, height, int64(width)*int64(height), limit)
}
