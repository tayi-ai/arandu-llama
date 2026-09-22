package filesystem

import (
	"context"
	"encoding/json"
	"fmt"
	"io"

	"github.com/arandu-io/hesape/auth"
)

// ReadStream opens a file for reading and returns the stream. The caller closes
// it.
//
// It is [Disk.Get] without the metadata: the same bytes, for a caller that is
// going to io.Copy them somewhere and has nothing to do with the size or the
// content type. Ask for Get when either of those matters -- serving the file,
// for one, needs both.
func (d *Disk) ReadStream(ctx context.Context, g auth.Grant, key string) (io.ReadCloser, error) {
	f, err := d.Get(ctx, g, key)
	if err != nil {
		return nil, err
	}
	return f.Body, nil
}

// WriteStream writes a stream to a key.
//
// It is [Disk.Put] under a second name. There is one implementation underneath,
// because Put already takes an io.Reader.
func (d *Disk) WriteStream(ctx context.Context, g auth.Grant, key string, body io.Reader, contentType string) error {
	return d.Put(ctx, g, key, body, contentType)
}

// Json reads a file and decodes it as JSON.
//
// It answers with map[string]any because the files this is for -- a manifest,
// an export, a stored payload -- are objects. A caller holding a struct should
// [Disk.Get] and
// unmarshal into it, which is one line more and type-checked.
func (d *Disk) Json(ctx context.Context, g auth.Grant, key string) (map[string]any, error) {
	f, err := d.Get(ctx, g, key)
	if err != nil {
		return nil, err
	}
	defer f.Body.Close()

	body, err := io.ReadAll(f.Body)
	if err != nil {
		return nil, d.wrap("read", key, err)
	}
	var out map[string]any
	if err := json.Unmarshal(body, &out); err != nil {
		return nil, fmt.Errorf("filesystem: %s on disk %q is not JSON: %w", key, d.name, err)
	}
	return out, nil
}
