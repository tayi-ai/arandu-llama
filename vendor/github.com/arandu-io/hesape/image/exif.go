package image

import (
	"encoding/binary"
	stdimage "image"
)

// exifOrientation reads the EXIF orientation tag out of the bytes as they
// arrived, and returns 1 -- "already the right way up" -- for anything with no
// tag to read.
//
// A phone does not turn the sensor when the person turns the phone: it stores
// the frame the way the sensor read it and writes down which way was up.
// Software that ignores the note shows every portrait photograph on its side,
// which is the single most reported bug of any upload form ever built.
//
// The tag lives in the APP1 segment of a JPEG, in a TIFF header, and this
// walks there by hand rather than taking on a dependency for it. Every read
// is bounds-checked, because the bytes came from somebody's upload and a
// parser that trusts a length field in an upload is a crash waiting for a
// stranger to schedule it.
func exifOrientation(b []byte) int {
	const upright = 1

	if len(b) < 4 || b[0] != 0xFF || b[1] != 0xD8 {
		return upright // not a JPEG: PNG and GIF carry no orientation
	}

	for i := 2; i+3 < len(b); {
		if b[i] != 0xFF {
			return upright // out of step with the segment chain
		}
		marker := b[i+1]
		switch {
		case marker == 0xFF:
			i++ // fill byte
			continue
		case marker == 0xD8 || marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7):
			i += 2 // standalone marker, no payload
			continue
		case marker == 0xDA || marker == 0xD9:
			return upright // start of scan: the metadata is behind us
		}

		length := int(binary.BigEndian.Uint16(b[i+2 : i+4]))
		if length < 2 || i+2+length > len(b) {
			return upright
		}
		payload := b[i+4 : i+2+length]
		if marker == 0xE1 && len(payload) > 6 && string(payload[:6]) == "Exif\x00\x00" {
			if o, ok := tiffOrientation(payload[6:]); ok {
				return o
			}
			return upright
		}
		i += 2 + length
	}
	return upright
}

// tiffOrientation walks the TIFF header an EXIF segment opens with and looks
// for tag 0x0112 in the first directory.
func tiffOrientation(b []byte) (int, bool) {
	if len(b) < 8 {
		return 0, false
	}
	var order binary.ByteOrder
	switch string(b[:2]) {
	case "II":
		order = binary.LittleEndian
	case "MM":
		order = binary.BigEndian
	default:
		return 0, false
	}
	if order.Uint16(b[2:4]) != 42 {
		return 0, false
	}

	offset := int(order.Uint32(b[4:8]))
	if offset < 8 || offset+2 > len(b) {
		return 0, false
	}
	count := int(order.Uint16(b[offset : offset+2]))
	entries := b[offset+2:]
	for i := 0; i < count; i++ {
		if (i+1)*12 > len(entries) {
			return 0, false
		}
		e := entries[i*12 : (i+1)*12]
		if order.Uint16(e[0:2]) != 0x0112 {
			continue
		}
		// Type 3 is SHORT, and a single SHORT sits in the first two bytes of
		// the value field rather than at an offset.
		if order.Uint16(e[2:4]) != 3 || order.Uint32(e[4:8]) != 1 {
			return 0, false
		}
		v := int(order.Uint16(e[8:10]))
		if v < 1 || v > 8 {
			return 0, false
		}
		return v, true
	}
	return 0, false
}

// applyOrientation turns and mirrors the canvas so that what the camera saw is
// what a viewer sees, given the EXIF orientation the file declared.
//
// The eight cases are the eight of the EXIF specification, and the operations
// are the ones that undo them.
func applyOrientation(src *stdimage.RGBA, orientation int) *stdimage.RGBA {
	switch orientation {
	case 2: // mirrored left to right
		return flipHorizontal(src)
	case 3: // upside down
		return flipVertical(flipHorizontal(src))
	case 4: // mirrored top to bottom
		return flipVertical(src)
	case 5: // mirrored, then a three-quarter turn
		return rotate270(flipHorizontal(src))
	case 6: // a quarter turn clockwise
		return rotate90(src)
	case 7: // mirrored, then a quarter turn
		return rotate90(flipHorizontal(src))
	case 8: // three quarters of a turn clockwise
		return rotate270(src)
	default: // 1, and anything unreadable
		return src
	}
}

// rotate270 turns the canvas three quarters clockwise, pixel for pixel.
func rotate270(src *stdimage.RGBA) *stdimage.RGBA {
	return rotate90(rotate90(rotate90(src)))
}
