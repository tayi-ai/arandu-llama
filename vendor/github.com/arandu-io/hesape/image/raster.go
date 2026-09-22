package image

import (
	"context"
	"fmt"
	stdimage "image"
	"image/color"
	"image/draw"
	"math"
	"strconv"
	"strings"
)

// interrupted returns the context's error once the context is done, and nil
// while it is still live.
//
// The routines that resample, turn and convolve call it once per line of the
// pass they are in -- a row, or a column where the pass runs down the canvas. A
// line is short enough that work abandoned by the caller stops within a
// fraction of the whole, and long enough that the check costs nothing
// measurable beside the arithmetic of the line itself.
//
// The routines that only move pixels -- the crop, the flips, the quarter turns
// -- do not call it. They run at the speed of memory, and a check between two
// copies would be most of the work rather than a fraction of it.
func interrupted(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return failWith(ctx.Err(), "processing the image")
	default:
		return nil
	}
}

// The canvas every transformation runs on is *image.RGBA from the standard
// library: eight bits a channel, alpha premultiplied.
//
// Premultiplied is not an implementation detail worth hiding. A weighted sum of
// premultiplied pixels is the correct blend and a weighted sum of
// straight-alpha pixels is not -- resize a logo with a transparent border in
// straight alpha and the edge picks up a halo of whatever colour the invisible
// pixels happened to carry. Every routine below therefore works in
// premultiplied space and keeps each colour channel clamped at or under the
// alpha channel, which is what makes the value a legal RGBA in the first place.

// toRGBA converts anything the standard decoders produce into the canvas, with
// its origin moved to (0,0) so every later routine can index from zero.
func toRGBA(src stdimage.Image) *stdimage.RGBA {
	b := src.Bounds()
	if r, ok := src.(*stdimage.RGBA); ok && b.Min.X == 0 && b.Min.Y == 0 {
		return r
	}
	dst := stdimage.NewRGBA(stdimage.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(dst, dst.Bounds(), src, b.Min, draw.Src)
	return dst
}

// contribution is one output pixel's source window and the weight of each
// source pixel in it.
type contribution struct {
	start   int
	weights []float64
}

// weights builds the resampling kernel for one axis.
//
// The filter is the triangle -- linear interpolation, the "bilinear" of the two
// separable passes below -- with one change that matters: when the image is
// being made smaller, the triangle's support is widened by the scale factor so
// that every source pixel inside the output pixel's footprint contributes.
//
// Without that widening a bilinear downscale reads at most two source pixels
// per output pixel, so a 4000-pixel photograph reduced to 400 simply ignores
// nine of every ten columns. What comes out is not soft, it is wrong: a picket
// fence turns into moire, and a line of text turns into gravel. Widening the
// support turns the same code into an area average, which is what a downscale
// wants, and leaves the upscale path exactly bilinear.
func weights(dstSize, srcSize int) []contribution {
	ratio := float64(srcSize) / float64(dstSize)
	scale := math.Max(ratio, 1)
	radius := scale // triangle support is 1, stretched by scale

	out := make([]contribution, dstSize)
	for i := range out {
		center := (float64(i)+0.5)*ratio - 0.5
		left := int(math.Ceil(center - radius))
		right := int(math.Floor(center + radius))

		w := make([]float64, 0, right-left+1)
		sum := 0.0
		for j := left; j <= right; j++ {
			t := math.Abs((float64(j) - center) / scale)
			v := 0.0
			if t < 1 {
				v = 1 - t
			}
			w = append(w, v)
			sum += v
		}
		if sum == 0 {
			// The window fell between samples, which only happens for a
			// degenerate scale; take the nearest source pixel rather than
			// dividing by zero.
			left = int(math.Round(center))
			w = []float64{1}
			sum = 1
		}
		for k := range w {
			w[k] /= sum
		}
		out[i] = contribution{start: left, weights: w}
	}
	return out
}

// resample scales the canvas to exactly w by h.
//
// Two separable passes, horizontal then vertical, because a separable filter
// over w*h*(kx+ky) samples produces the same result as the square kernel over
// w*h*kx*ky and is the reason a resize of a large photograph finishes.
//
// The context bounds both passes. The horizontal one runs first, so a resize
// abandoned at its first row never reaches the allocation the vertical pass
// makes, which for an enlargement is the whole of the output canvas.
func resample(ctx context.Context, src *stdimage.RGBA, w, h int) (*stdimage.RGBA, error) {
	sw, sh := src.Rect.Dx(), src.Rect.Dy()
	if w < 1 {
		w = 1
	}
	if h < 1 {
		h = 1
	}
	if w == sw && h == sh {
		return src, nil
	}
	wide, err := resampleHorizontal(ctx, src, w)
	if err != nil {
		return nil, err
	}
	return resampleVertical(ctx, wide, h)
}

func resampleHorizontal(ctx context.Context, src *stdimage.RGBA, w int) (*stdimage.RGBA, error) {
	sw, sh := src.Rect.Dx(), src.Rect.Dy()
	if w == sw {
		return src, nil
	}
	if err := interrupted(ctx); err != nil {
		return nil, err
	}
	dst := stdimage.NewRGBA(stdimage.Rect(0, 0, w, sh))
	cols := weights(w, sw)
	for y := 0; y < sh; y++ {
		if err := interrupted(ctx); err != nil {
			return nil, err
		}
		row := src.Pix[y*src.Stride : y*src.Stride+sw*4]
		out := dst.Pix[y*dst.Stride : y*dst.Stride+w*4]
		for x, c := range cols {
			var r, g, b, a float64
			for k, weight := range c.weights {
				sx := clampInt(c.start+k, 0, sw-1)
				p := row[sx*4:]
				r += float64(p[0]) * weight
				g += float64(p[1]) * weight
				b += float64(p[2]) * weight
				a += float64(p[3]) * weight
			}
			o := out[x*4:]
			o[3] = round8(a)
			o[0] = clamp8(r, o[3])
			o[1] = clamp8(g, o[3])
			o[2] = clamp8(b, o[3])
		}
	}
	return dst, nil
}

func resampleVertical(ctx context.Context, src *stdimage.RGBA, h int) (*stdimage.RGBA, error) {
	sw, sh := src.Rect.Dx(), src.Rect.Dy()
	if h == sh {
		return src, nil
	}
	if err := interrupted(ctx); err != nil {
		return nil, err
	}
	dst := stdimage.NewRGBA(stdimage.Rect(0, 0, sw, h))
	rows := weights(h, sh)
	for y, c := range rows {
		if err := interrupted(ctx); err != nil {
			return nil, err
		}
		out := dst.Pix[y*dst.Stride : y*dst.Stride+sw*4]
		for x := 0; x < sw; x++ {
			var r, g, b, a float64
			for k, weight := range c.weights {
				sy := clampInt(c.start+k, 0, sh-1)
				p := src.Pix[sy*src.Stride+x*4:]
				r += float64(p[0]) * weight
				g += float64(p[1]) * weight
				b += float64(p[2]) * weight
				a += float64(p[3]) * weight
			}
			o := out[x*4:]
			o[3] = round8(a)
			o[0] = clamp8(r, o[3])
			o[1] = clamp8(g, o[3])
			o[2] = clamp8(b, o[3])
		}
	}
	return dst, nil
}

// crop cuts a w by h rectangle whose top left corner is at (x, y).
//
// A rectangle that reaches past the edge is not an error: the part that
// exists is copied and the rest is left transparent, which is what a caller
// cropping a thumbnail out of a picture that turned out to be smaller than
// expected wants.
func crop(src *stdimage.RGBA, x, y, w, h int) *stdimage.RGBA {
	if w < 1 {
		w = 1
	}
	if h < 1 {
		h = 1
	}
	dst := stdimage.NewRGBA(stdimage.Rect(0, 0, w, h))
	draw.Draw(dst, dst.Bounds(), src, stdimage.Pt(x, y), draw.Src)
	return dst
}

// flipHorizontal mirrors left to right.
func flipHorizontal(src *stdimage.RGBA) *stdimage.RGBA {
	w, h := src.Rect.Dx(), src.Rect.Dy()
	dst := stdimage.NewRGBA(stdimage.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		row := src.Pix[y*src.Stride:]
		out := dst.Pix[y*dst.Stride:]
		for x := 0; x < w; x++ {
			copy(out[(w-1-x)*4:(w-1-x)*4+4], row[x*4:x*4+4])
		}
	}
	return dst
}

// flipVertical mirrors top to bottom.
func flipVertical(src *stdimage.RGBA) *stdimage.RGBA {
	w, h := src.Rect.Dx(), src.Rect.Dy()
	dst := stdimage.NewRGBA(stdimage.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		copy(dst.Pix[(h-1-y)*dst.Stride:(h-1-y)*dst.Stride+w*4], src.Pix[y*src.Stride:y*src.Stride+w*4])
	}
	return dst
}

// rotate90 turns the canvas a quarter turn clockwise. It is exact: no pixel is
// interpolated, which is why the three right angles do not go through [rotate].
func rotate90(src *stdimage.RGBA) *stdimage.RGBA {
	w, h := src.Rect.Dx(), src.Rect.Dy()
	dst := stdimage.NewRGBA(stdimage.Rect(0, 0, h, w))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			dx, dy := h-1-y, x
			copy(dst.Pix[dy*dst.Stride+dx*4:dy*dst.Stride+dx*4+4], src.Pix[y*src.Stride+x*4:y*src.Stride+x*4+4])
		}
	}
	return dst
}

// rotate turns the canvas clockwise by an arbitrary angle in degrees, filling
// what the turn leaves empty with bg.
//
// Right angles are handed to [rotate90], which copies pixels instead of
// sampling them; everything else is sampled bilinearly from the source, and the
// canvas grows to the bounding box of the turned rectangle so nothing is cut
// off.
func rotate(ctx context.Context, src *stdimage.RGBA, angle float64, bg color.RGBA) (*stdimage.RGBA, error) {
	// Degrees are taken modulo a full turn so that 450 and 90 agree, and a
	// negative angle turns the other way.
	a := math.Mod(angle, 360)
	if a < 0 {
		a += 360
	}
	switch a {
	case 0:
		return src, nil
	case 90:
		return rotate90(src), nil
	case 180:
		return flipVertical(flipHorizontal(src)), nil
	case 270:
		return rotate90(rotate90(rotate90(src))), nil
	}

	if err := interrupted(ctx); err != nil {
		return nil, err
	}

	rad := a * math.Pi / 180
	sin, cos := math.Sin(rad), math.Cos(rad)
	sw, sh := float64(src.Rect.Dx()), float64(src.Rect.Dy())

	w := int(math.Ceil(math.Abs(sw*cos) + math.Abs(sh*sin)))
	h := int(math.Ceil(math.Abs(sw*sin) + math.Abs(sh*cos)))
	dst := stdimage.NewRGBA(stdimage.Rect(0, 0, w, h))
	fill(dst, bg)

	scx, scy := sw/2, sh/2
	dcx, dcy := float64(w)/2, float64(h)/2
	for y := 0; y < h; y++ {
		if err := interrupted(ctx); err != nil {
			return nil, err
		}
		for x := 0; x < w; x++ {
			rx := float64(x) + 0.5 - dcx
			ry := float64(y) + 0.5 - dcy
			// The inverse of the clockwise turn, so the destination asks the
			// source where its colour came from rather than scattering source
			// pixels into a destination and leaving holes between them.
			sx := rx*cos + ry*sin + scx - 0.5
			sy := -rx*sin + ry*cos + scy - 0.5
			if sx < -0.5 || sy < -0.5 || sx > sw-0.5 || sy > sh-0.5 {
				continue
			}
			p := dst.Pix[y*dst.Stride+x*4:]
			r, g, b, al := sampleBilinear(src, sx, sy)
			p[0], p[1], p[2], p[3] = r, g, b, al
		}
	}
	return dst, nil
}

// sampleBilinear reads the source at a fractional position, mixing the four
// pixels around it.
func sampleBilinear(src *stdimage.RGBA, x, y float64) (uint8, uint8, uint8, uint8) {
	w, h := src.Rect.Dx(), src.Rect.Dy()
	x0, y0 := int(math.Floor(x)), int(math.Floor(y))
	fx, fy := x-float64(x0), y-float64(y0)

	var r, g, b, a float64
	for dy := 0; dy < 2; dy++ {
		wy := fy
		if dy == 0 {
			wy = 1 - fy
		}
		sy := clampInt(y0+dy, 0, h-1)
		for dx := 0; dx < 2; dx++ {
			wx := fx
			if dx == 0 {
				wx = 1 - fx
			}
			sx := clampInt(x0+dx, 0, w-1)
			weight := wx * wy
			p := src.Pix[sy*src.Stride+sx*4:]
			r += float64(p[0]) * weight
			g += float64(p[1]) * weight
			b += float64(p[2]) * weight
			a += float64(p[3]) * weight
		}
	}
	al := round8(a)
	return clamp8(r, al), clamp8(g, al), clamp8(b, al), al
}

// fill paints the whole canvas one colour.
func fill(dst *stdimage.RGBA, c color.RGBA) {
	draw.Draw(dst, dst.Bounds(), &stdimage.Uniform{C: c}, stdimage.Point{}, draw.Src)
}

// drawOver composites src onto dst inside the rectangle at, which is how
// [contain] places the scaled image on its padded canvas.
func drawOver(dst *stdimage.RGBA, at stdimage.Rectangle, src *stdimage.RGBA) {
	draw.Draw(dst, at, src, src.Rect.Min, draw.Over)
}

// flatten composites the canvas over an opaque colour, which is what an encoder
// with no alpha channel needs: JPEG has no transparency, and handing it
// premultiplied pixels with a zero alpha paints them black.
func flatten(src *stdimage.RGBA, bg color.RGBA) *stdimage.RGBA {
	dst := stdimage.NewRGBA(src.Rect)
	fill(dst, bg)
	draw.Draw(dst, dst.Bounds(), src, src.Rect.Min, draw.Over)
	return dst
}

// averageColor is the mean of every pixel, unpremultiplied -- equivalent to
// resizing the image down to a single pixel and reading it, without the
// intermediate canvas.
//
// It reads every pixel of the canvas, which is why it takes a context: this is
// the whole of the work behind a dominant colour, and a dominant colour nobody
// is waiting for any more should stop like a resize does.
func averageColor(ctx context.Context, src *stdimage.RGBA) (color.RGBA, error) {
	w, h := src.Rect.Dx(), src.Rect.Dy()
	if w == 0 || h == 0 {
		return color.RGBA{}, nil
	}
	var r, g, b, a float64
	for y := 0; y < h; y++ {
		if err := interrupted(ctx); err != nil {
			return color.RGBA{}, err
		}
		row := src.Pix[y*src.Stride : y*src.Stride+w*4]
		for x := 0; x < w; x++ {
			p := row[x*4:]
			r += float64(p[0])
			g += float64(p[1])
			b += float64(p[2])
			a += float64(p[3])
		}
	}
	n := float64(w * h)
	al := round8(a / n)
	return color.RGBA{R: clamp8(r/n, al), G: clamp8(g/n, al), B: clamp8(b/n, al), A: al}, nil
}

// hex renders a colour as seven characters, "#rrggbb", with the alpha
// dropped.
func hex(c color.RGBA) string {
	r, g, b := unpremultiply(c)
	return fmt.Sprintf("#%02x%02x%02x", r, g, b)
}

// unpremultiply divides the colour channels back out by alpha, which is what
// makes "#ffffff" come back out of a white pixel that is half transparent.
func unpremultiply(c color.RGBA) (uint8, uint8, uint8) {
	if c.A == 0 || c.A == 255 {
		return c.R, c.G, c.B
	}
	f := 255 / float64(c.A)
	return round8(float64(c.R) * f), round8(float64(c.G) * f), round8(float64(c.B) * f)
}

// parseHexColor reads a background colour spelled as "#rgb", "#rgba",
// "#rrggbb", or "#rrggbbaa", with or without the hash. The result is
// premultiplied, because that is what the canvas holds.
func parseHexColor(s string) (color.RGBA, error) {
	v := strings.TrimPrefix(strings.TrimSpace(s), "#")
	var r, g, b, a uint64 = 0, 0, 0, 255
	var err error
	parse := func(t string) uint64 {
		if err != nil {
			return 0
		}
		var n uint64
		n, err = strconv.ParseUint(t, 16, 16)
		return n
	}
	switch len(v) {
	case 3, 4:
		r, g, b = parse(v[0:1]+v[0:1]), parse(v[1:2]+v[1:2]), parse(v[2:3]+v[2:3])
		if len(v) == 4 {
			a = parse(v[3:4] + v[3:4])
		}
	case 6, 8:
		r, g, b = parse(v[0:2]), parse(v[2:4]), parse(v[4:6])
		if len(v) == 8 {
			a = parse(v[6:8])
		}
	default:
		return color.RGBA{}, fail("the background colour %q is not a hex colour", s)
	}
	if err != nil {
		return color.RGBA{}, fail("the background colour %q is not a hex colour", s)
	}
	f := float64(a) / 255
	return color.RGBA{
		R: round8(float64(r) * f),
		G: round8(float64(g) * f),
		B: round8(float64(b) * f),
		A: uint8(a),
	}, nil
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// round8 rounds a channel accumulator to a byte.
func round8(v float64) uint8 {
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v + 0.5)
}

// clamp8 rounds a colour channel and holds it at or under alpha, which is the
// invariant premultiplied RGBA has and the one that rounding and sharpening
// both like to break.
func clamp8(v float64, alpha uint8) uint8 {
	out := round8(v)
	if out > alpha {
		return alpha
	}
	return out
}
