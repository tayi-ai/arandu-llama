// Package filetype classifies untrusted files from bounded reads of their
// content. It never consults a filename or a client-announced media type.
package filetype

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/xml"
	"image"
	"image/gif"
	"io"
	"mime"
	stdhttp "net/http"
	"strings"

	_ "image/jpeg"
	_ "image/png"
)

const (
	sniffLimit       = int64(512)
	imageHeaderLimit = int64(1 << 20)
	imageBodyLimit   = int64(64 << 20)
	svgBodyLimit     = int64(1 << 20)
	maxImagePixels   = int64(32 << 20)
	// gif.DecodeAll retains every frame, so both pixel storage and per-frame
	// overhead need budgets checked by the streaming preflight.
	maxGIFFrames = 256
	maxGIFPixels = maxImagePixels
	// trailingPaddingLimit bounds what may follow the marker that ends a PNG or
	// a JPEG. Cameras and encoders that align a file to a small boundary leave a
	// short run of null or whitespace bytes past that marker, so refusing every
	// trailing byte would refuse files that are otherwise well formed. One
	// 64-byte alignment block is the largest such run in practical use, and it
	// is far too small to carry a second document behind the image.
	trailingPaddingLimit = 64
)

// Inspection is the immutable result of examining one file. Image is true only
// for formats whose complete structure was decoded and whose content ends with
// that structure; SVG additionally records that accepting the document requires
// an explicit policy decision.
type Inspection struct {
	MediaType string
	Extension string
	Width     int
	Height    int
	Image     bool
	SVG       bool
	OK        bool
}

// Inspect reads all security metadata in one pass. Generic sniffing reads at
// most 512 bytes. Recognized raster images are additionally checked through a
// bounded decoder or container reader before their image type is returned.
func Inspect(open func() (io.ReadCloser, error)) Inspection {
	mediaType, extension, ok := sniff(open)
	if !ok {
		return Inspection{}
	}

	result := Inspection{MediaType: mediaType, Extension: extension, OK: true}
	switch mediaType {
	case "image/png", "image/jpeg", "image/gif":
		width, height, valid := decodeRaster(open, mediaType)
		if !valid {
			return unknownInspection()
		}
		result.Width, result.Height, result.Image = width, height, true
	case "image/bmp", "image/x-ms-bmp":
		if _, _, valid := inspectBMP(open); !valid {
			return unknownInspection()
		}
	case "image/webp":
		if _, _, valid := inspectWebP(open); !valid {
			return unknownInspection()
		}
	case "image/svg+xml":
		if !validSVG(open) {
			return unknownInspection()
		}
		result.Image, result.SVG = true, true
	}
	return result
}

func unknownInspection() Inspection {
	return Inspection{MediaType: "application/octet-stream", Extension: "bin", OK: true}
}

// Detect returns the normalized media type and canonical extension read from
// the content. The bool is false when the content cannot be opened or is empty.
func Detect(open func() (io.ReadCloser, error)) (mediaType, extension string, ok bool) {
	result := Inspect(open)
	return result.MediaType, result.Extension, result.OK
}

func sniff(open func() (io.ReadCloser, error)) (mediaType, extension string, ok bool) {
	head, ok := readPrefix(open, sniffLimit)
	if !ok || len(head) == 0 {
		return "", "", false
	}

	if hasSVGRoot(head) {
		return "image/svg+xml", "svg", true
	}

	detected := stdhttp.DetectContentType(head)
	mediaType, _, err := mime.ParseMediaType(detected)
	if err != nil || mediaType == "" {
		return "", "", false
	}
	mediaType = strings.ToLower(mediaType)
	return mediaType, canonicalExtension(mediaType), true
}

// Image validates a supported image from its bytes and returns its dimensions.
// SVG is accepted only when allowSVG is true; it has no pixel dimensions, so a
// valid SVG returns zeroes. Raster dimensions and encoded reads are bounded.
func Image(open func() (io.ReadCloser, error), allowSVG bool) (width, height int, ok bool) {
	result := Inspect(open)
	if !result.OK || !result.Image || result.SVG && !allowSVG {
		return 0, 0, false
	}
	return result.Width, result.Height, true
}

func readPrefix(open func() (io.ReadCloser, error), limit int64) ([]byte, bool) {
	if open == nil {
		return nil, false
	}
	reader, err := open()
	if err != nil {
		return nil, false
	}
	defer reader.Close()

	body, err := io.ReadAll(io.LimitReader(reader, limit))
	return body, err == nil
}

func canonicalExtension(mediaType string) string {
	switch mediaType {
	case "image/jpeg":
		return "jpeg"
	case "image/png":
		return "png"
	case "image/gif":
		return "gif"
	case "image/bmp", "image/x-ms-bmp":
		return "bmp"
	case "image/webp":
		return "webp"
	case "image/svg+xml":
		return "svg"
	case "image/x-icon", "image/vnd.microsoft.icon":
		return "ico"
	case "application/pdf":
		return "pdf"
	case "application/postscript":
		return "ps"
	case "application/zip":
		return "zip"
	case "application/gzip", "application/x-gzip":
		return "gz"
	case "audio/aiff":
		return "aiff"
	case "audio/mpeg":
		return "mp3"
	case "application/ogg", "audio/ogg":
		return "ogg"
	case "audio/midi":
		return "midi"
	case "video/avi":
		return "avi"
	case "audio/wave", "audio/wav", "audio/x-wav":
		return "wav"
	case "video/mp4":
		return "mp4"
	case "video/webm":
		return "webm"
	case "application/vnd.ms-fontobject":
		return "eot"
	case "font/ttf":
		return "ttf"
	case "font/otf":
		return "otf"
	case "font/collection":
		return "ttc"
	case "font/woff":
		return "woff"
	case "font/woff2":
		return "woff2"
	case "application/x-rar-compressed", "application/vnd.rar":
		return "rar"
	case "application/wasm":
		return "wasm"
	case "text/html":
		return "html"
	case "text/plain":
		return "txt"
	case "text/xml", "application/xml":
		return "xml"
	case "application/json":
		return "json"
	case "application/octet-stream":
		return "bin"
	default:
		return ""
	}
}

func hasSVGRoot(head []byte) bool {
	decoder := xml.NewDecoder(bytes.NewReader(head))
	for {
		token, err := decoder.Token()
		if err != nil {
			return false
		}
		switch value := token.(type) {
		case xml.StartElement:
			return strings.EqualFold(value.Name.Local, "svg") &&
				(value.Name.Space == "" || value.Name.Space == "http://www.w3.org/2000/svg")
		case xml.CharData:
			if strings.TrimSpace(string(value)) != "" {
				return false
			}
		case xml.Directive:
			return false
		case xml.ProcInst:
			if !strings.EqualFold(value.Target, "xml") {
				return false
			}
		}
	}
}

func decodeRaster(open func() (io.ReadCloser, error), mediaType string) (int, int, bool) {
	if mediaType == "image/gif" {
		return decodeGIF(open)
	}

	reader, err := open()
	if err != nil {
		return 0, 0, false
	}
	config, format, err := image.DecodeConfig(io.LimitReader(reader, imageHeaderLimit))
	_ = reader.Close()
	if err != nil || !formatMatches(mediaType, format) || !safeDimensions(config.Width, config.Height) {
		return 0, 0, false
	}

	width, height, ok := decodeRasterBody(open, mediaType)
	if !ok || width != config.Width || height != config.Height {
		return 0, 0, false
	}
	return config.Width, config.Height, true
}

// decodeRasterBody decodes the whole image and reports the bounds it decoded.
//
// The content has to end where the image ends. A PNG decoder stops at IEND and
// a JPEG decoder stops at EOI, so without this check anything at all could ride
// behind a valid image header and still be stored under an image media type and
// an image extension. Only a short run of alignment padding is tolerated.
func decodeRasterBody(open func() (io.ReadCloser, error), mediaType string) (int, int, bool) {
	reader, err := open()
	if err != nil {
		return 0, 0, false
	}
	defer reader.Close()

	limited := &io.LimitedReader{R: reader, N: imageBodyLimit}
	// The buffer has to be ours: whatever a decoder buffers for itself is
	// unreachable once it returns, and that is where the trailer would hide.
	buffered := bufio.NewReader(limited)
	var body io.Reader = buffered
	if mediaType == "image/jpeg" {
		body = &throttledReader{reader: buffered}
	}
	decoded, decodedFormat, err := image.Decode(body)
	if err != nil || !formatMatches(mediaType, decodedFormat) {
		return 0, 0, false
	}
	// An image that ended exactly on the encoded-body budget leaves the limiter
	// with no allowance, so the trailer is given one of its own.
	limited.N = trailingPaddingLimit + 1
	if !endsAfterPadding(buffered) {
		return 0, 0, false
	}
	bounds := decoded.Bounds()
	return bounds.Dx(), bounds.Dy(), true
}

// throttledReader hands out a single byte per read.
//
// The JPEG decoder reads ahead in blocks and keeps what it did not use, so the
// content it was reading from ends up past the end of the image by an amount
// only the decoder knows. Never handing it more than it asks for leaves the
// content positioned exactly where the image ends, at a cost the decode itself
// dwarfs. The PNG decoder reads exact chunk lengths and already stops there, so
// it is not throttled: doing so would cost more than its own decode.
type throttledReader struct {
	reader *bufio.Reader
}

func (t *throttledReader) Read(body []byte) (int, error) {
	if len(body) > 1 {
		body = body[:1]
	}
	return t.reader.Read(body)
}

// Peek keeps the decoder from wrapping this reader in a buffer of its own,
// which would put the trailer back out of reach.
func (t *throttledReader) Peek(count int) ([]byte, error) { return t.reader.Peek(count) }

// endsAfterPadding reports whether nothing but padding remains: at most
// trailingPaddingLimit bytes, each of them null or whitespace.
func endsAfterPadding(rest io.Reader) bool {
	trailer, err := io.ReadAll(io.LimitReader(rest, trailingPaddingLimit+1))
	if err != nil || int64(len(trailer)) > trailingPaddingLimit {
		return false
	}
	for _, character := range trailer {
		switch character {
		case 0x00, ' ', '\t', '\n', '\v', '\f', '\r':
		default:
			return false
		}
	}
	return true
}

type gifMetadata struct {
	width  int
	height int
	frames int
	pixels int64
}

func decodeGIF(open func() (io.ReadCloser, error)) (int, int, bool) {
	metadata, ok := preflightGIF(open)
	if !ok {
		return 0, 0, false
	}

	reader, err := open()
	if err != nil {
		return 0, 0, false
	}
	decoded, err := gif.DecodeAll(io.LimitReader(reader, imageBodyLimit))
	_ = reader.Close()
	if err != nil || decoded == nil ||
		decoded.Config.Width != metadata.width || decoded.Config.Height != metadata.height ||
		!safeDimensions(decoded.Config.Width, decoded.Config.Height) ||
		len(decoded.Image) != metadata.frames || len(decoded.Delay) != metadata.frames ||
		len(decoded.Disposal) != metadata.frames {
		return 0, 0, false
	}

	screen := image.Rect(0, 0, metadata.width, metadata.height)
	var pixels int64
	for _, frame := range decoded.Image {
		if frame == nil || frame.Bounds().Empty() || !frame.Bounds().In(screen) {
			return 0, 0, false
		}
		framePixels := int64(frame.Bounds().Dx()) * int64(frame.Bounds().Dy())
		if framePixels > maxGIFPixels-pixels {
			return 0, 0, false
		}
		pixels += framePixels
	}
	if pixels != metadata.pixels {
		return 0, 0, false
	}
	return metadata.width, metadata.height, true
}

func preflightGIF(open func() (io.ReadCloser, error)) (gifMetadata, bool) {
	if open == nil {
		return gifMetadata{}, false
	}
	reader, err := open()
	if err != nil {
		return gifMetadata{}, false
	}
	defer reader.Close()

	limited := &io.LimitedReader{R: reader, N: imageBodyLimit}
	stream := &gifStream{reader: bufio.NewReader(limited)}
	var header [13]byte
	if !stream.readFull(header[:]) ||
		string(header[:6]) != "GIF87a" && string(header[:6]) != "GIF89a" {
		return gifMetadata{}, false
	}

	metadata := gifMetadata{
		width:  int(binary.LittleEndian.Uint16(header[6:8])),
		height: int(binary.LittleEndian.Uint16(header[8:10])),
	}
	if !safeDimensions(metadata.width, metadata.height) {
		return gifMetadata{}, false
	}
	if header[10]&0x80 != 0 && !skipGIFColorTable(stream, header[10]) {
		return gifMetadata{}, false
	}

	for {
		blockType, err := stream.readByte()
		if err != nil {
			return gifMetadata{}, false
		}
		switch blockType {
		case 0x21:
			if !skipGIFExtension(stream) {
				return gifMetadata{}, false
			}
		case 0x2c:
			if !inspectGIFFrame(stream, &metadata) {
				return gifMetadata{}, false
			}
		case 0x3b:
			if metadata.frames == 0 || stream.bytes >= imageBodyLimit {
				return gifMetadata{}, false
			}
			if _, err := stream.readByte(); err != io.EOF {
				return gifMetadata{}, false
			}
			return metadata, true
		default:
			return gifMetadata{}, false
		}
	}
}

type gifStream struct {
	reader io.Reader
	bytes  int64
}

func (stream *gifStream) Read(body []byte) (int, error) {
	read, err := stream.reader.Read(body)
	stream.bytes += int64(read)
	return read, err
}

func (stream *gifStream) readByte() (byte, error) {
	var body [1]byte
	_, err := io.ReadFull(stream, body[:])
	return body[0], err
}

func (stream *gifStream) readFull(body []byte) bool {
	_, err := io.ReadFull(stream, body)
	return err == nil
}

func (stream *gifStream) skip(size int) bool {
	_, err := io.CopyN(io.Discard, stream, int64(size))
	return err == nil
}

func skipGIFColorTable(stream *gifStream, fields byte) bool {
	entries := 1 << (1 + uint(fields&0x07))
	return stream.skip(3 * entries)
}

func skipGIFExtension(stream *gifStream) bool {
	extension, err := stream.readByte()
	if err != nil {
		return false
	}
	switch extension {
	case 0x01:
		size, err := stream.readByte()
		if err != nil || size != 12 || !stream.skip(int(size)) {
			return false
		}
	case 0xf9:
		var control [6]byte
		return stream.readFull(control[:]) && control[0] == 4 && control[5] == 0
	case 0xfe:
		return skipGIFSubBlocks(stream, false)
	case 0xff:
		size, err := stream.readByte()
		if err != nil || !stream.skip(int(size)) {
			return false
		}
	default:
		return false
	}
	return skipGIFSubBlocks(stream, false)
}

func inspectGIFFrame(stream *gifStream, metadata *gifMetadata) bool {
	var descriptor [9]byte
	if !stream.readFull(descriptor[:]) {
		return false
	}
	left := int(binary.LittleEndian.Uint16(descriptor[0:2]))
	top := int(binary.LittleEndian.Uint16(descriptor[2:4]))
	width := int(binary.LittleEndian.Uint16(descriptor[4:6]))
	height := int(binary.LittleEndian.Uint16(descriptor[6:8]))
	if !safeDimensions(width, height) || width > metadata.width || height > metadata.height ||
		left > metadata.width-width || top > metadata.height-height {
		return false
	}

	framePixels := int64(width) * int64(height)
	if metadata.frames >= maxGIFFrames || framePixels > maxGIFPixels-metadata.pixels {
		return false
	}
	metadata.frames++
	metadata.pixels += framePixels

	if descriptor[8]&0x80 != 0 && !skipGIFColorTable(stream, descriptor[8]) {
		return false
	}
	codeSize, err := stream.readByte()
	if err != nil || codeSize < 2 || codeSize > 8 {
		return false
	}
	return skipGIFSubBlocks(stream, true)
}

func skipGIFSubBlocks(stream *gifStream, requireData bool) bool {
	hasData := false
	for {
		size, err := stream.readByte()
		if err != nil {
			return false
		}
		if size == 0 {
			return hasData || !requireData
		}
		if !stream.skip(int(size)) {
			return false
		}
		hasData = true
	}
}

func formatMatches(mediaType, format string) bool {
	return mediaType == "image/"+format || mediaType == "image/jpeg" && format == "jpeg"
}

func safeDimensions(width, height int) bool {
	return width > 0 && height > 0 && int64(width) <= maxImagePixels/int64(height)
}

func inspectBMP(open func() (io.ReadCloser, error)) (int, int, bool) {
	reader, err := open()
	if err != nil {
		return 0, 0, false
	}
	defer reader.Close()

	header := make([]byte, 54)
	if _, err := io.ReadFull(reader, header); err != nil || string(header[:2]) != "BM" {
		return 0, 0, false
	}
	declared := int64(binary.LittleEndian.Uint32(header[2:6]))
	offset := int64(binary.LittleEndian.Uint32(header[10:14]))
	dibSize := int64(binary.LittleEndian.Uint32(header[14:18]))
	width := int64(int32(binary.LittleEndian.Uint32(header[18:22])))
	height := int64(int32(binary.LittleEndian.Uint32(header[22:26])))
	planes := binary.LittleEndian.Uint16(header[26:28])
	bitsPerPixel := int64(binary.LittleEndian.Uint16(header[28:30]))
	compression := binary.LittleEndian.Uint32(header[30:34])
	if height < 0 {
		height = -height
	}
	if declared < int64(len(header)) || declared > imageBodyLimit || dibSize < 40 || dibSize > declared-14 ||
		offset < 14+dibSize || offset >= declared || planes != 1 || compression != 0 ||
		!oneOf(bitsPerPixel, 1, 4, 8, 16, 24, 32) || width <= 0 || height <= 0 ||
		width > maxImagePixels/height {
		return 0, 0, false
	}
	rowBytes := ((width*bitsPerPixel + 31) / 32) * 4
	if rowBytes <= 0 || rowBytes > imageBodyLimit || height > (declared-offset)/rowBytes {
		return 0, 0, false
	}
	if copied, err := io.CopyN(io.Discard, reader, declared-int64(len(header))); err != nil || copied != declared-int64(len(header)) {
		return 0, 0, false
	}
	var extra [1]byte
	if n, _ := reader.Read(extra[:]); n != 0 {
		return 0, 0, false
	}
	return int(width), int(height), true
}

func oneOf(value int64, allowed ...int64) bool {
	for _, candidate := range allowed {
		if value == candidate {
			return true
		}
	}
	return false
}

func inspectWebP(open func() (io.ReadCloser, error)) (int, int, bool) {
	reader, err := open()
	if err != nil {
		return 0, 0, false
	}
	defer reader.Close()

	header := make([]byte, 12)
	if _, err := io.ReadFull(reader, header); err != nil || string(header[:4]) != "RIFF" || string(header[8:]) != "WEBP" {
		return 0, 0, false
	}
	total := int64(binary.LittleEndian.Uint32(header[4:8])) + 8
	if total < 20 || total > imageBodyLimit {
		return 0, 0, false
	}

	remaining := total - int64(len(header))
	width, height := 0, 0
	canvasWidth, canvasHeight := 0, 0
	extended := false
	extendedFlags := byte(0)
	alphaFlag := false
	foundAlpha := false
	foundICC := false
	foundEXIF := false
	foundXMP := false
	foundImage := false
	chunkIndex := 0
	for remaining > 0 {
		if remaining < 8 {
			return 0, 0, false
		}
		chunkHeader := make([]byte, 8)
		if _, err := io.ReadFull(reader, chunkHeader); err != nil {
			return 0, 0, false
		}
		size := int64(binary.LittleEndian.Uint32(chunkHeader[4:8]))
		padded := size + size%2
		if padded > remaining-8 || size > imageBodyLimit {
			return 0, 0, false
		}
		prefixSize := min(size, 32)
		prefix := make([]byte, prefixSize)
		if _, err := io.ReadFull(reader, prefix); err != nil {
			return 0, 0, false
		}
		if copied, err := io.CopyN(io.Discard, reader, padded-prefixSize); err != nil || copied != padded-prefixSize {
			return 0, 0, false
		}

		switch string(chunkHeader[:4]) {
		case "VP8X":
			if chunkIndex != 0 || extended || size != 10 || len(prefix) < 10 ||
				prefix[0]&0xc3 != 0 || prefix[1] != 0 || prefix[2] != 0 || prefix[3] != 0 {
				return 0, 0, false
			}
			extended = true
			extendedFlags = prefix[0]
			alphaFlag = prefix[0]&0x10 != 0
			canvasWidth = (int(prefix[4]) | int(prefix[5])<<8 | int(prefix[6])<<16) + 1
			canvasHeight = (int(prefix[7]) | int(prefix[8])<<8 | int(prefix[9])<<16) + 1
			if !safeDimensions(canvasWidth, canvasHeight) {
				return 0, 0, false
			}
		case "VP8 ":
			if foundImage || len(prefix) < 10 {
				return 0, 0, false
			}
			frameTag := uint32(prefix[0]) | uint32(prefix[1])<<8 | uint32(prefix[2])<<16
			partitionSize := int64(frameTag >> 5)
			if frameTag&1 != 0 || (frameTag>>1)&7 > 3 || (frameTag>>4)&1 == 0 ||
				partitionSize == 0 || partitionSize > size-10 ||
				prefix[3] != 0x9d || prefix[4] != 0x01 || prefix[5] != 0x2a {
				return 0, 0, false
			}
			width = int(binary.LittleEndian.Uint16(prefix[6:8]) & 0x3fff)
			height = int(binary.LittleEndian.Uint16(prefix[8:10]) & 0x3fff)
			foundImage = true
		case "VP8L":
			if foundImage || foundAlpha || size < 6 || len(prefix) < 6 || prefix[0] != 0x2f {
				return 0, 0, false
			}
			bits := binary.LittleEndian.Uint32(prefix[1:5])
			if bits>>29 != 0 {
				return 0, 0, false
			}
			width = int(bits&0x3fff) + 1
			height = int((bits>>14)&0x3fff) + 1
			foundImage = true
		case "ALPH":
			if !extended || foundAlpha || foundImage || size < 1 {
				return 0, 0, false
			}
			foundAlpha = true
		case "ICCP":
			if !extended || foundICC || extendedFlags&0x20 == 0 || size < 1 {
				return 0, 0, false
			}
			foundICC = true
		case "EXIF":
			if !extended || foundEXIF || extendedFlags&0x08 == 0 || size < 1 {
				return 0, 0, false
			}
			foundEXIF = true
		case "XMP ":
			if !extended || foundXMP || extendedFlags&0x04 == 0 || size < 1 {
				return 0, 0, false
			}
			foundXMP = true
		case "ANIM", "ANMF":
			// Animated WebP needs a complete codec decoder, which the standard
			// library does not provide. Classification therefore fails closed.
			return 0, 0, false
		default:
			return 0, 0, false
		}
		remaining -= 8 + padded
		chunkIndex++
	}
	var extra [1]byte
	if n, _ := reader.Read(extra[:]); n != 0 || !foundImage || !safeDimensions(width, height) {
		return 0, 0, false
	}
	if extended {
		if width != canvasWidth || height != canvasHeight || alphaFlag != foundAlpha ||
			(extendedFlags&0x20 != 0) != foundICC || (extendedFlags&0x08 != 0) != foundEXIF ||
			(extendedFlags&0x04 != 0) != foundXMP {
			return 0, 0, false
		}
	} else if foundAlpha {
		return 0, 0, false
	}
	return width, height, true
}

func validSVG(open func() (io.ReadCloser, error)) bool {
	body, ok := readPrefix(open, svgBodyLimit+1)
	if !ok || len(body) == 0 || int64(len(body)) > svgBodyLimit {
		return false
	}

	decoder := xml.NewDecoder(bytes.NewReader(body))
	depth := 0
	seenRoot := false
	closedRoot := false
	for {
		token, err := decoder.Token()
		if err == io.EOF {
			return seenRoot && closedRoot && depth == 0
		}
		if err != nil {
			return false
		}
		switch value := token.(type) {
		case xml.StartElement:
			if closedRoot || !seenRoot && (!strings.EqualFold(value.Name.Local, "svg") ||
				(value.Name.Space != "" && value.Name.Space != "http://www.w3.org/2000/svg")) {
				return false
			}
			seenRoot = true
			depth++
		case xml.EndElement:
			depth--
			if depth < 0 {
				return false
			}
			if seenRoot && depth == 0 {
				closedRoot = true
			}
		case xml.CharData:
			if (!seenRoot || closedRoot) && strings.TrimSpace(string(value)) != "" {
				return false
			}
		case xml.Directive:
			return false
		case xml.ProcInst:
			if seenRoot || !strings.EqualFold(value.Target, "xml") {
				return false
			}
		}
	}
}
