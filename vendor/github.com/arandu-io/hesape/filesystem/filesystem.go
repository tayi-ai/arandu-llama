package filesystem

import (
	"crypto/md5"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"hash"
	"io"
	"io/fs"
	"mime"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// Filesystem is the local file API the framework itself runs on.
//
// It is NOT a [Disk], and the difference is the whole reason both exist. A Disk
// holds customer data, so every one of its methods takes an [auth.Grant] and
// every path it builds starts with a tenant. A Filesystem
// holds the application's own files -- a stub, a compiled view, a session file,
// a cache entry -- which belong to the process and to nobody else, and it takes
// absolute or working-directory-relative paths exactly as os.ReadFile does.
//
// Storing a tenant's upload through this type is the bug it is worth naming
// here: there is no prefix, so there is no isolation. The way in for anything a
// customer sent is [Disk.Put].
//
// The zero value is usable, and so is the pointer [NewFilesystem] returns; this
// type holds no state.
type Filesystem struct{}

// NewFilesystem returns a Filesystem. It exists so wiring reads the same as the
// rest of the collection; the zero value works too.
func NewFilesystem() *Filesystem { return &Filesystem{} }

// Exists reports whether a file or directory exists at the path.
func (f *Filesystem) Exists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// Missing reports whether nothing exists at the path.
func (f *Filesystem) Missing(path string) bool { return !f.Exists(path) }

// Get returns the contents of a file.
//
// lock takes a shared lock for the read, which is what makes it safe to read a
// file another process is replacing with [Filesystem.Put] under an exclusive
// lock. There is no default: Go has none, and a bool at the call site says
// which of the two this read is.
//
// It returns [ErrNotFound] for a file that is not there.
func (f *Filesystem) Get(path string, lock bool) ([]byte, error) {
	if lock {
		return f.SharedGet(path)
	}
	body, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, fmt.Errorf("filesystem: %s: %w", path, ErrNotFound)
	}
	if err != nil {
		return nil, fmt.Errorf("filesystem: reading %s: %w", path, err)
	}
	return body, nil
}

// Json returns the decoded contents of a JSON file.
//
// It answers with map[string]any because the files this is for -- configuration
// and manifest files -- are objects. A caller that has a struct should read with
// [Filesystem.Get] and unmarshal into it, which is one call more and
// type-checked.
func (f *Filesystem) Json(path string, lock bool) (map[string]any, error) {
	body, err := f.Get(path, lock)
	if err != nil {
		return nil, err
	}
	var out map[string]any
	if err := json.Unmarshal(body, &out); err != nil {
		return nil, fmt.Errorf("filesystem: %s is not JSON: %w", path, err)
	}
	return out, nil
}

// SharedGet reads a file with a shared lock held for the whole read.
func (f *Filesystem) SharedGet(path string) ([]byte, error) {
	file, err := NewLockableFile(path, os.O_RDONLY)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	if err := file.GetSharedLock(true); err != nil {
		return nil, err
	}
	defer file.ReleaseLock()

	size, err := file.Size()
	if err != nil {
		return nil, err
	}
	return file.Read(size)
}

// Lines returns the file split on newlines, with the trailing newline of the
// last line dropped.
//
// It reads the whole file and splits it rather than streaming one line at a
// time: a caller that needs the lazy shape has bufio.Scanner, and a second,
// lazier Lines beside this one would be a second way to read a file.
func (f *Filesystem) Lines(path string) ([]string, error) {
	body, err := f.Get(path, false)
	if err != nil {
		return nil, err
	}
	text := strings.ReplaceAll(string(body), "\r\n", "\n")
	text = strings.TrimSuffix(text, "\n")
	if text == "" {
		return nil, nil
	}
	return strings.Split(text, "\n"), nil
}

// Hash returns the hash of a file's contents, in lowercase hex.
//
// The empty algorithm is "md5", which is what [Filesystem.HasSameHash] compares
// with. It is a change-detection hash and not
// a security one: two files that hash the same here were not proven to be the
// same file by an attacker who wanted them to collide. The security answer is
// [Disk.Checksum], which is SHA-256 and has no algorithm option.
//
// Recognised: "md5", "sha1", "sha256".
func (f *Filesystem) Hash(path, algorithm string) (string, error) {
	var sum hash.Hash
	switch algorithm {
	case "", "md5":
		sum = md5.New()
	case "sha1":
		sum = sha1.New()
	case "sha256":
		sum = sha256.New()
	default:
		return "", fmt.Errorf("filesystem: %q is not a hash this knows: md5, sha1 or sha256", algorithm)
	}

	file, err := os.Open(path)
	if errors.Is(err, fs.ErrNotExist) {
		return "", fmt.Errorf("filesystem: %s: %w", path, ErrNotFound)
	}
	if err != nil {
		return "", fmt.Errorf("filesystem: reading %s: %w", path, err)
	}
	defer file.Close()

	if _, err := io.Copy(sum, file); err != nil {
		return "", fmt.Errorf("filesystem: reading %s: %w", path, err)
	}
	return hex.EncodeToString(sum.Sum(nil)), nil
}

// Put writes the contents to a file, creating it if it is not there.
//
// lock takes an exclusive lock for the write, which is what a reader calling
// [Filesystem.SharedGet] waits on.
func (f *Filesystem) Put(path string, contents []byte, lock bool) error {
	if !lock {
		if err := os.WriteFile(path, contents, 0o644); err != nil {
			return fmt.Errorf("filesystem: writing %s: %w", path, err)
		}
		return nil
	}

	file, err := NewLockableFile(path, os.O_CREATE|os.O_RDWR)
	if err != nil {
		return err
	}
	defer file.Close()

	if err := file.GetExclusiveLock(true); err != nil {
		return err
	}
	defer file.ReleaseLock()

	if err := file.Truncate(); err != nil {
		return err
	}
	_, err = file.Write(contents)
	return err
}

// Replace writes the contents atomically, so a reader never sees half of them.
//
// It writes a temporary file beside the target and renames it into place, which
// is what makes the swap atomic on a POSIX filesystem. A mode of 0 keeps the
// mode of the file that was already there, or 0644 when there was none: a
// rename would otherwise hand the file the temporary's private permissions.
func (f *Filesystem) Replace(path string, content []byte, mode fs.FileMode) error {
	if mode == 0 {
		mode = 0o644
		if info, err := os.Stat(path); err == nil {
			mode = info.Mode().Perm()
		}
	}

	tmp, err := os.CreateTemp(filepath.Dir(path), "."+filepath.Base(path)+".*")
	if err != nil {
		return fmt.Errorf("filesystem: writing %s: %w", path, err)
	}
	name := tmp.Name()
	write := func() error {
		if _, err := tmp.Write(content); err != nil {
			return err
		}
		if err := tmp.Chmod(mode); err != nil {
			return err
		}
		return tmp.Close()
	}
	if err := write(); err != nil {
		_ = tmp.Close()
		_ = os.Remove(name)
		return fmt.Errorf("filesystem: writing %s: %w", path, err)
	}
	if err := os.Rename(name, path); err != nil {
		_ = os.Remove(name)
		return fmt.Errorf("filesystem: writing %s: %w", path, err)
	}
	return nil
}

// ReplaceInFile substitutes every occurrence of search with replace inside a
// file.
func (f *Filesystem) ReplaceInFile(search, replace, path string) error {
	body, err := f.Get(path, false)
	if err != nil {
		return err
	}
	return f.Put(path, []byte(strings.ReplaceAll(string(body), search, replace)), false)
}

// Prepend writes data to the front of a file, creating it when it is not there.
func (f *Filesystem) Prepend(path string, data []byte) error {
	if f.Exists(path) {
		existing, err := f.Get(path, false)
		if err != nil {
			return err
		}
		return f.Put(path, append(append([]byte{}, data...), existing...), false)
	}
	return f.Put(path, data, false)
}

// Append writes data to the end of a file, creating it when it is not there.
func (f *Filesystem) Append(path string, data []byte, lock bool) error {
	flags := os.O_CREATE | os.O_WRONLY | os.O_APPEND
	file, err := os.OpenFile(path, flags, 0o644)
	if err != nil {
		return fmt.Errorf("filesystem: appending to %s: %w", path, err)
	}
	defer file.Close()

	if lock {
		if err := lockFile(file, true, true); err != nil {
			return err
		}
		defer unlockFile(file)
	}
	if _, err := file.Write(data); err != nil {
		return fmt.Errorf("filesystem: appending to %s: %w", path, err)
	}
	return nil
}

// Chmod reads or sets the permissions of a path.
//
// A mode of 0 reads and does not write; anything else is set. The returned mode
// is the one in force after the call.
func (f *Filesystem) Chmod(path string, mode fs.FileMode) (fs.FileMode, error) {
	if mode != 0 {
		if err := os.Chmod(path, mode); err != nil {
			return 0, fmt.Errorf("filesystem: chmod %s: %w", path, err)
		}
		return mode, nil
	}
	info, err := os.Stat(path)
	if errors.Is(err, fs.ErrNotExist) {
		return 0, fmt.Errorf("filesystem: %s: %w", path, ErrNotFound)
	}
	if err != nil {
		return 0, fmt.Errorf("filesystem: reading %s: %w", path, err)
	}
	return info.Mode().Perm(), nil
}

// Delete removes the given files. Removing what is not there is not an error.
//
// It is variadic, so one path and a list of them are the same call.
func (f *Filesystem) Delete(paths ...string) error {
	var failed []string
	for _, path := range paths {
		if err := os.Remove(path); err != nil && !errors.Is(err, fs.ErrNotExist) {
			failed = append(failed, path)
		}
	}
	if len(failed) > 0 {
		return fmt.Errorf("filesystem: deleting %s", strings.Join(failed, ", "))
	}
	return nil
}

// Move renames a file.
func (f *Filesystem) Move(path, target string) error {
	if err := os.Rename(path, target); err != nil {
		return fmt.Errorf("filesystem: moving %s to %s: %w", path, target, err)
	}
	return nil
}

// Copy duplicates a file.
func (f *Filesystem) Copy(path, target string) error {
	src, err := os.Open(path)
	if errors.Is(err, fs.ErrNotExist) {
		return fmt.Errorf("filesystem: %s: %w", path, ErrNotFound)
	}
	if err != nil {
		return fmt.Errorf("filesystem: reading %s: %w", path, err)
	}
	defer src.Close()

	info, err := src.Stat()
	if err != nil {
		return fmt.Errorf("filesystem: reading %s: %w", path, err)
	}
	dst, err := os.OpenFile(target, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, info.Mode().Perm())
	if err != nil {
		return fmt.Errorf("filesystem: writing %s: %w", target, err)
	}
	if _, err := io.Copy(dst, src); err != nil {
		_ = dst.Close()
		return fmt.Errorf("filesystem: writing %s: %w", target, err)
	}
	if err := dst.Close(); err != nil {
		return fmt.Errorf("filesystem: writing %s: %w", target, err)
	}
	return nil
}

// Link creates a symbolic link to the target.
//
// It is a symbolic link on every platform, including the ones where that needs
// a privilege. A hard link is a different object with different semantics --
// deleting the target leaves the link working, which is the opposite of what a
// link into a build directory is for.
func (f *Filesystem) Link(target, link string) error {
	if err := os.Symlink(target, link); err != nil {
		return fmt.Errorf("filesystem: linking %s to %s: %w", link, target, err)
	}
	return nil
}

// RelativeLink creates a symbolic link whose target is written relative to the
// directory holding the link.
//
// That is what survives the tree being moved or mounted somewhere else, which is
// the whole reason it exists beside Link.
func (f *Filesystem) RelativeLink(target, link string) error {
	relative, err := filepath.Rel(filepath.Dir(link), target)
	if err != nil {
		return fmt.Errorf("filesystem: linking %s to %s: %w", link, target, err)
	}
	return f.Link(relative, link)
}

// Name returns the file name without its extension.
func (f *Filesystem) Name(path string) string {
	base := filepath.Base(path)
	return strings.TrimSuffix(base, filepath.Ext(base))
}

// Basename returns the trailing name component of a path.
func (f *Filesystem) Basename(path string) string { return filepath.Base(path) }

// Dirname returns the parent directory of a path.
func (f *Filesystem) Dirname(path string) string { return filepath.Dir(path) }

// Extension returns the extension of a path, without the dot.
//
// filepath.Ext returns ".pdf" and this returns "pdf", so a caller comparing
// against a configured list of extensions gets the comparison it wrote.
func (f *Filesystem) Extension(path string) string {
	return strings.TrimPrefix(filepath.Ext(path), ".")
}

// GuessExtension returns the extension a file's content type implies, without
// the dot, or "" when nothing is known about it.
func (f *Filesystem) GuessExtension(path string) (string, error) {
	kind, err := f.MimeType(path)
	if err != nil {
		return "", err
	}
	extensions, err := mime.ExtensionsByType(kind)
	if err != nil || len(extensions) == 0 {
		return "", nil
	}
	// Sorted, because the table answers in map order and a guess that differs
	// between two runs of the same build is one nobody can reproduce.
	sort.Strings(extensions)
	return strings.TrimPrefix(extensions[0], "."), nil
}

// Type returns "dir" for a directory and "file" for anything else.
func (f *Filesystem) Type(path string) (string, error) {
	info, err := os.Stat(path)
	if errors.Is(err, fs.ErrNotExist) {
		return "", fmt.Errorf("filesystem: %s: %w", path, ErrNotFound)
	}
	if err != nil {
		return "", fmt.Errorf("filesystem: reading %s: %w", path, err)
	}
	if info.IsDir() {
		return "dir", nil
	}
	return "file", nil
}

// MimeType returns the content type implied by the path's extension.
//
// From the extension and never from the bytes, which is the same rule [Put]
// follows: content sniffing is what turns an uploaded file into stored XSS the
// day something serves it back. Unknown extensions answer
// application/octet-stream, which browsers download rather than render.
func (f *Filesystem) MimeType(path string) (string, error) {
	if _, err := os.Stat(path); err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return "", fmt.Errorf("filesystem: %s: %w", path, ErrNotFound)
		}
		return "", fmt.Errorf("filesystem: reading %s: %w", path, err)
	}
	return TypeOf(path), nil
}

// Size returns how many bytes a file holds.
func (f *Filesystem) Size(path string) (int64, error) {
	info, err := os.Stat(path)
	if errors.Is(err, fs.ErrNotExist) {
		return 0, fmt.Errorf("filesystem: %s: %w", path, ErrNotFound)
	}
	if err != nil {
		return 0, fmt.Errorf("filesystem: reading %s: %w", path, err)
	}
	return info.Size(), nil
}

// LastModified returns when a file was last written.
func (f *Filesystem) LastModified(path string) (time.Time, error) {
	info, err := os.Stat(path)
	if errors.Is(err, fs.ErrNotExist) {
		return time.Time{}, fmt.Errorf("filesystem: %s: %w", path, ErrNotFound)
	}
	if err != nil {
		return time.Time{}, fmt.Errorf("filesystem: reading %s: %w", path, err)
	}
	return info.ModTime(), nil
}

// IsDirectory reports whether the path is a directory.
func (f *Filesystem) IsDirectory(directory string) bool {
	info, err := os.Stat(directory)
	return err == nil && info.IsDir()
}

// IsEmptyDirectory reports whether a directory holds nothing.
//
// ignoreDotFiles treats a directory holding only dot files as empty, which is
// what a check for "did anything get published here" wants: a .gitkeep is not
// content.
func (f *Filesystem) IsEmptyDirectory(directory string, ignoreDotFiles bool) bool {
	entries, err := os.ReadDir(directory)
	if err != nil {
		return false
	}
	for _, entry := range entries {
		if ignoreDotFiles && strings.HasPrefix(entry.Name(), ".") {
			continue
		}
		return false
	}
	return true
}

// IsReadable reports whether the process can read the path.
func (f *Filesystem) IsReadable(path string) bool {
	file, err := os.Open(path)
	if err != nil {
		return false
	}
	_ = file.Close()
	return true
}

// IsWritable reports whether the process can write the path.
//
// A directory is writable when a file can be created in it, which is the
// question a caller is really asking and the only one an access bit cannot
// answer on its own.
func (f *Filesystem) IsWritable(path string) bool {
	info, err := os.Stat(path)
	if err != nil {
		return false
	}
	if info.IsDir() {
		probe, err := os.CreateTemp(path, ".hesape-writable-*")
		if err != nil {
			return false
		}
		name := probe.Name()
		_ = probe.Close()
		_ = os.Remove(name)
		return true
	}
	file, err := os.OpenFile(path, os.O_WRONLY, 0)
	if err != nil {
		return false
	}
	_ = file.Close()
	return true
}

// HasSameHash reports whether two files hold the same contents.
func (f *Filesystem) HasSameHash(firstFile, secondFile string) bool {
	first, err := f.Hash(firstFile, "")
	if err != nil {
		return false
	}
	second, err := f.Hash(secondFile, "")
	if err != nil {
		return false
	}
	return first == second
}

// IsFile reports whether the path is a regular file.
func (f *Filesystem) IsFile(file string) bool {
	info, err := os.Stat(file)
	return err == nil && !info.IsDir()
}

// Glob returns the paths matching a shell pattern, sorted.
func (f *Filesystem) Glob(pattern string) ([]string, error) {
	matches, err := filepath.Glob(pattern)
	if err != nil {
		return nil, fmt.Errorf("filesystem: %q is not a pattern: %w", pattern, err)
	}
	sort.Strings(matches)
	return matches, nil
}

// Files returns the files in a directory, sorted.
//
// hidden includes names starting with a dot. depth is how many levels below the
// directory to descend: 0 is the directory itself, and [Filesystem.AllFiles] is
// this with no limit.
func (f *Filesystem) Files(directory string, hidden bool, depth int) ([]string, error) {
	return f.walk(directory, hidden, depth, false)
}

// AllFiles returns every file under a directory, at any depth, sorted.
func (f *Filesystem) AllFiles(directory string, hidden bool) ([]string, error) {
	return f.walk(directory, hidden, -1, false)
}

// Directories returns the directories in a directory, sorted.
func (f *Filesystem) Directories(directory string, depth int) ([]string, error) {
	return f.walk(directory, false, depth, true)
}

// AllDirectories returns every directory under a directory, at any depth,
// sorted.
func (f *Filesystem) AllDirectories(directory string) ([]string, error) {
	return f.walk(directory, false, -1, true)
}

// walk is the one traversal the four listing methods share, so a rule about dot
// files or about depth cannot hold in one of them and not the others.
//
// A negative depth means no limit. Depth 0 is the directory's own entries.
func (f *Filesystem) walk(directory string, hidden bool, depth int, wantDirs bool) ([]string, error) {
	root := filepath.Clean(directory)
	var out []string
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if path == root {
			return nil
		}
		if !hidden && strings.HasPrefix(entry.Name(), ".") {
			if entry.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		relative, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		level := strings.Count(relative, string(filepath.Separator))
		if depth >= 0 && level > depth {
			if entry.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if entry.IsDir() == wantDirs {
			out = append(out, path)
		}
		return nil
	})
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil, fmt.Errorf("filesystem: %s: %w", directory, ErrNotFound)
		}
		return nil, fmt.Errorf("filesystem: listing %s: %w", directory, err)
	}
	sort.Strings(out)
	return out, nil
}

// EnsureDirectoryExists creates a directory when it is not already there.
//
// A mode of 0 means 0755.
func (f *Filesystem) EnsureDirectoryExists(path string, mode fs.FileMode, recursive bool) error {
	if f.IsDirectory(path) {
		return nil
	}
	return f.MakeDirectory(path, mode, recursive, false)
}

// MakeDirectory creates a directory.
//
// force removes whatever is in the way first. recursive creates the parents.
// A mode of 0 means 0755.
func (f *Filesystem) MakeDirectory(path string, mode fs.FileMode, recursive, force bool) error {
	if mode == 0 {
		mode = 0o755
	}
	if force {
		if err := os.RemoveAll(path); err != nil {
			return fmt.Errorf("filesystem: creating %s: %w", path, err)
		}
	}
	make := os.Mkdir
	if recursive {
		make = os.MkdirAll
	}
	if err := make(path, mode); err != nil {
		return fmt.Errorf("filesystem: creating %s: %w", path, err)
	}
	// Mkdir subtracts the process umask from the mode, so a directory asked for
	// as 0755 under a 0027 umask arrives as 0750 and the next process cannot
	// read it, so the mode is set again once the directory exists.
	if err := os.Chmod(path, mode); err != nil {
		return fmt.Errorf("filesystem: creating %s: %w", path, err)
	}
	return nil
}

// MoveDirectory renames a directory.
//
// overwrite removes the destination first. Without it, a destination that is
// already there is an error rather than a merge: merging two trees silently is
// how a deploy ends up with half of the previous release in it.
func (f *Filesystem) MoveDirectory(from, to string, overwrite bool) error {
	if overwrite && f.IsDirectory(to) {
		if err := os.RemoveAll(to); err != nil {
			return fmt.Errorf("filesystem: moving %s to %s: %w", from, to, err)
		}
	}
	if err := os.Rename(from, to); err != nil {
		return fmt.Errorf("filesystem: moving %s to %s: %w", from, to, err)
	}
	return nil
}

// CopyDirectory copies a directory, recursively, to a destination.
func (f *Filesystem) CopyDirectory(directory, destination string) error {
	if !f.IsDirectory(directory) {
		return fmt.Errorf("filesystem: %s: %w", directory, ErrNotFound)
	}
	info, err := os.Stat(directory)
	if err != nil {
		return fmt.Errorf("filesystem: reading %s: %w", directory, err)
	}
	if err := f.EnsureDirectoryExists(destination, info.Mode().Perm(), true); err != nil {
		return err
	}

	entries, err := os.ReadDir(directory)
	if err != nil {
		return fmt.Errorf("filesystem: listing %s: %w", directory, err)
	}
	for _, entry := range entries {
		source := filepath.Join(directory, entry.Name())
		target := filepath.Join(destination, entry.Name())
		switch {
		case entry.IsDir():
			if err := f.CopyDirectory(source, target); err != nil {
				return err
			}
		default:
			if err := f.Copy(source, target); err != nil {
				return err
			}
		}
	}
	return nil
}

// DeleteDirectory removes a directory and everything under it.
//
// preserve keeps the directory itself and removes only what is inside, which is
// what [Filesystem.CleanDirectory] is named for.
func (f *Filesystem) DeleteDirectory(directory string, preserve bool) error {
	if !f.IsDirectory(directory) {
		return fmt.Errorf("filesystem: %s: %w", directory, ErrNotFound)
	}
	if !preserve {
		if err := os.RemoveAll(directory); err != nil {
			return fmt.Errorf("filesystem: deleting %s: %w", directory, err)
		}
		return nil
	}
	entries, err := os.ReadDir(directory)
	if err != nil {
		return fmt.Errorf("filesystem: listing %s: %w", directory, err)
	}
	for _, entry := range entries {
		if err := os.RemoveAll(filepath.Join(directory, entry.Name())); err != nil {
			return fmt.Errorf("filesystem: deleting %s: %w", entry.Name(), err)
		}
	}
	return nil
}

// DeleteDirectories removes every directory directly inside a directory,
// leaving the files alone. It reports whether it removed anything.
func (f *Filesystem) DeleteDirectories(directory string) (bool, error) {
	entries, err := os.ReadDir(directory)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return false, nil
		}
		return false, fmt.Errorf("filesystem: listing %s: %w", directory, err)
	}
	deleted := false
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		if err := os.RemoveAll(filepath.Join(directory, entry.Name())); err != nil {
			return deleted, fmt.Errorf("filesystem: deleting %s: %w", entry.Name(), err)
		}
		deleted = true
	}
	return deleted, nil
}

// CleanDirectory empties a directory and keeps the directory itself.
func (f *Filesystem) CleanDirectory(directory string) error {
	return f.DeleteDirectory(directory, true)
}

// JoinPaths joins path segments with a separator, dropping the empty ones.
//
// The empty segments are dropped, so JoinPaths(base, "", "views") is base/views
// and not base//views -- which is a different string that names the same file,
// and therefore two cache keys.
func JoinPaths(base string, paths ...string) string {
	out := make([]string, 0, len(paths)+1)
	if base != "" {
		out = append(out, base)
	}
	for _, p := range paths {
		if p != "" {
			out = append(out, p)
		}
	}
	return filepath.Join(out...)
}
