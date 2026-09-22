package image

import (
	"context"
	stdimage "image"
)

// grayscale replaces every pixel with its luminance, keeping alpha.
//
// The coefficients are Rec. 601's -- the same ones GD and ImageMagick use for a
// plain desaturation -- so a red and a blue of equal brightness on screen do
// not come out as the same grey.
func grayscale(ctx context.Context, src *stdimage.RGBA) (*stdimage.RGBA, error) {
	w, h := src.Rect.Dx(), src.Rect.Dy()
	if err := interrupted(ctx); err != nil {
		return nil, err
	}
	dst := stdimage.NewRGBA(stdimage.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		if err := interrupted(ctx); err != nil {
			return nil, err
		}
		in := src.Pix[y*src.Stride : y*src.Stride+w*4]
		out := dst.Pix[y*dst.Stride : y*dst.Stride+w*4]
		for x := 0; x < w; x++ {
			p := in[x*4:]
			g := clamp8(0.299*float64(p[0])+0.587*float64(p[1])+0.114*float64(p[2]), p[3])
			o := out[x*4:]
			o[0], o[1], o[2], o[3] = g, g, g, p[3]
		}
	}
	return dst, nil
}

// blurRadius turns a 0-to-100 amount into a radius in pixels.
//
// Amount is a strength, not a distance, so it is scaled down: five points to
// the pixel. The default blur(5) softens by one pixel and blur(100) by
// twenty.
func blurRadius(amount int) int {
	if amount <= 0 {
		return 0
	}
	r := (amount + 4) / 5
	if r < 1 {
		return 1
	}
	return r
}

// blur softens the canvas.
//
// Three box blurs in a row, which is the standard cheap Gaussian: the box is a
// rectangular kernel and it shows as banding when used once, and by the third
// pass the kernel has converged close enough to a bell that nothing shows.
// Each pass is separable and runs on a sliding sum, so the cost is the number
// of pixels and not the number of pixels times the radius.
func blur(ctx context.Context, src *stdimage.RGBA, radius int) (*stdimage.RGBA, error) {
	if radius < 1 {
		return src, nil
	}
	out := src
	for i := 0; i < 3; i++ {
		wide, err := boxBlurHorizontal(ctx, out, radius)
		if err != nil {
			return nil, err
		}
		if out, err = boxBlurVertical(ctx, wide, radius); err != nil {
			return nil, err
		}
	}
	return out, nil
}

func boxBlurHorizontal(ctx context.Context, src *stdimage.RGBA, r int) (*stdimage.RGBA, error) {
	w, h := src.Rect.Dx(), src.Rect.Dy()
	if err := interrupted(ctx); err != nil {
		return nil, err
	}
	dst := stdimage.NewRGBA(stdimage.Rect(0, 0, w, h))
	window := 2*r + 1
	for y := 0; y < h; y++ {
		if err := interrupted(ctx); err != nil {
			return nil, err
		}
		row := src.Pix[y*src.Stride : y*src.Stride+w*4]
		out := dst.Pix[y*dst.Stride : y*dst.Stride+w*4]
		at := func(x int) []uint8 { return row[clampInt(x, 0, w-1)*4:] }

		var sum [4]int
		for i := -r; i <= r; i++ {
			p := at(i)
			for c := 0; c < 4; c++ {
				sum[c] += int(p[c])
			}
		}
		for x := 0; x < w; x++ {
			o := out[x*4:]
			a := uint8(sum[3] / window)
			o[3] = a
			o[0] = clamp8(float64(sum[0]/window), a)
			o[1] = clamp8(float64(sum[1]/window), a)
			o[2] = clamp8(float64(sum[2]/window), a)

			gone, came := at(x-r), at(x+r+1)
			for c := 0; c < 4; c++ {
				sum[c] += int(came[c]) - int(gone[c])
			}
		}
	}
	return dst, nil
}

func boxBlurVertical(ctx context.Context, src *stdimage.RGBA, r int) (*stdimage.RGBA, error) {
	w, h := src.Rect.Dx(), src.Rect.Dy()
	if err := interrupted(ctx); err != nil {
		return nil, err
	}
	dst := stdimage.NewRGBA(stdimage.Rect(0, 0, w, h))
	window := 2*r + 1
	for x := 0; x < w; x++ {
		if err := interrupted(ctx); err != nil {
			return nil, err
		}
		at := func(y int) []uint8 { return src.Pix[clampInt(y, 0, h-1)*src.Stride+x*4:] }

		var sum [4]int
		for i := -r; i <= r; i++ {
			p := at(i)
			for c := 0; c < 4; c++ {
				sum[c] += int(p[c])
			}
		}
		for y := 0; y < h; y++ {
			o := dst.Pix[y*dst.Stride+x*4:]
			a := uint8(sum[3] / window)
			o[3] = a
			o[0] = clamp8(float64(sum[0]/window), a)
			o[1] = clamp8(float64(sum[1]/window), a)
			o[2] = clamp8(float64(sum[2]/window), a)

			gone, came := at(y-r), at(y+r+1)
			for c := 0; c < 4; c++ {
				sum[c] += int(came[c]) - int(gone[c])
			}
		}
	}
	return dst, nil
}

// sharpen is an unsharp mask: the difference between the canvas and a blurred
// copy of it, added back on top.
//
// Amount is 0 to 100, the strength of the mask, with 100 adding the full
// difference twice over -- past that the halo around every edge is more
// visible than the detail it was meant to recover, which is why the number is
// clamped before it arrives.
func sharpen(ctx context.Context, src *stdimage.RGBA, amount int) (*stdimage.RGBA, error) {
	if amount <= 0 {
		return src, nil
	}
	k := float64(amount) / 100 * 2
	wide, err := boxBlurHorizontal(ctx, src, 1)
	if err != nil {
		return nil, err
	}
	soft, err := boxBlurVertical(ctx, wide, 1)
	if err != nil {
		return nil, err
	}

	w, h := src.Rect.Dx(), src.Rect.Dy()
	dst := stdimage.NewRGBA(stdimage.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		if err := interrupted(ctx); err != nil {
			return nil, err
		}
		in := src.Pix[y*src.Stride : y*src.Stride+w*4]
		lo := soft.Pix[y*soft.Stride : y*soft.Stride+w*4]
		out := dst.Pix[y*dst.Stride : y*dst.Stride+w*4]
		for x := 0; x < w; x++ {
			p, b, o := in[x*4:], lo[x*4:], out[x*4:]
			a := p[3]
			o[3] = a
			for c := 0; c < 3; c++ {
				v := float64(p[c]) + k*(float64(p[c])-float64(b[c]))
				o[c] = clamp8(v, a)
			}
		}
	}
	return dst, nil
}
