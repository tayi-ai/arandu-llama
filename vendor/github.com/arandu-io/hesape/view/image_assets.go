package view

import (
	"context"
	"encoding/binary"
	"fmt"
	"io/fs"
	"path"
	"strings"

	"github.com/arandu-io/hesape/image"
)

// RegisterImageAsset converts a trusted static SVG to WebP once, before
// serving requests, using an image driver with SVG and WebP support. Other
// source extensions are refused and must use RegisterAsset. The driver's
// configured viewport controls raster dimensions; this function does not
// infer display size from markup or modify the input bytes.
//
// Asset resolves the source name to the generated .webp URL. Assets lists
// only the generated file, whose hash and content type describe its WebP
// bytes. A source named logo.svg is not served as WebP under an SVG URL.
// Non-image assets continue to use RegisterAsset.
//
// Conversion and name collisions return errors without publishing a partial
// registration. There is no silent fallback to SVG and no conversion on an
// HTTP request. The caller owns the lifetime and configuration of the driver.
func RegisterImageAsset(ctx context.Context, name string, body []byte, driver image.Driver) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if !fs.ValidPath(name) || strings.ContainsAny(name, `\?#%`) {
		return fmt.Errorf("view: invalid image asset name %q", name)
	}
	ext := path.Ext(name)
	if !strings.EqualFold(ext, ".svg") {
		return fmt.Errorf("view: image conversion only accepts SVG, got %q; use RegisterAsset for other files", ext)
	}
	if len(body) == 0 || driver == nil {
		return fmt.Errorf("view: image asset %s needs bytes and an image driver", name)
	}
	target := strings.TrimSuffix(name, ext) + ".webp"
	pipeline := image.NewImagePipeline()
	pipeline.Output.Format = "webp"
	encoded, err := driver.Process(ctx, append([]byte(nil), body...), pipeline)
	if err != nil {
		return fmt.Errorf("view: convert image asset %s: %w", name, err)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(encoded) < 20 || string(encoded[:4]) != "RIFF" || string(encoded[8:12]) != "WEBP" ||
		uint64(binary.LittleEndian.Uint32(encoded[4:8]))+8 != uint64(len(encoded)) {
		return fmt.Errorf("view: image driver did not return WebP for %s", name)
	}
	encoded = append([]byte(nil), encoded...)
	assetsMu.Lock()
	defer assetsMu.Unlock()
	for _, key := range []string{name, target} {
		if _, exists := assets[key]; exists || assetAliases[key] != "" {
			return fmt.Errorf("view: image asset name %s is already registered", key)
		}
	}
	assets[target] = newAsset(target, "image/webp", encoded)
	if name != target {
		assetAliases[name] = target
	}
	return nil
}
