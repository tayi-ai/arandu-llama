package filesystem

import (
	"bytes"
	"context"
	"io"

	"github.com/arandu-io/hesape/auth"
)

// TB is the part of *testing.T these assertions use.
//
// It is declared here rather than importing testing so that a package under
// test can hold a Disk without the test binary's flags leaking into a
// production build. It is satisfied by *testing.T and *testing.B unchanged.
type TB interface {
	Helper()
	Errorf(format string, args ...any)
}

// AssertExists fails the test when the key is not there for this tenant.
//
// content is optional: pass one value to also assert what the file holds.
func (d *Disk) AssertExists(ctx context.Context, t TB, g auth.Grant, key string, content ...[]byte) {
	t.Helper()

	f, err := d.Get(ctx, g, key)
	if err != nil {
		t.Errorf("expected %s to be on disk %s, and reading it said: %v", key, d.name, err)
		return
	}
	defer f.Body.Close()

	if len(content) == 0 {
		return
	}
	body, err := io.ReadAll(f.Body)
	if err != nil {
		t.Errorf("expected %s to be readable on disk %s, and it said: %v", key, d.name, err)
		return
	}
	if !bytes.Equal(body, content[0]) {
		t.Errorf("expected %s to hold %q, and it holds %q", key, content[0], body)
	}
}

// AssertMissing fails the test when the key is there.
func (d *Disk) AssertMissing(ctx context.Context, t TB, g auth.Grant, key string) {
	t.Helper()

	missing, err := d.Missing(ctx, g, key)
	if err != nil {
		t.Errorf("expected %s to be gone from disk %s, and looking said: %v", key, d.name, err)
		return
	}
	if !missing {
		t.Errorf("expected %s to be gone from disk %s, and it is there", key, d.name)
	}
}

// AssertCount fails the test when a directory does not hold exactly count
// files.
//
// recursive counts everything underneath rather than the directory's own files,
// which is the difference between [Disk.AllFiles] and [Disk.Files].
func (d *Disk) AssertCount(ctx context.Context, t TB, g auth.Grant, directory string, count int, recursive bool) {
	t.Helper()

	list := d.Files
	if recursive {
		list = d.AllFiles
	}
	keys, err := list(ctx, g, directory)
	if err != nil {
		t.Errorf("expected %d files under %q on disk %s, and listing said: %v", count, directory, d.name, err)
		return
	}
	if len(keys) != count {
		t.Errorf("expected %d files under %q on disk %s, and there are %d: %v", count, directory, d.name, len(keys), keys)
	}
}

// AssertDirectoryEmpty fails the test when a directory holds anything.
func (d *Disk) AssertDirectoryEmpty(ctx context.Context, t TB, g auth.Grant, directory string) {
	t.Helper()
	d.AssertCount(ctx, t, g, directory, 0, true)
}
