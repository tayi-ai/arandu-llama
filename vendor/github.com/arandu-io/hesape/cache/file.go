package cache

import (
	"context"
	"crypto/sha1"
	"encoding/hex"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Filesystem is the slice of a file system FileStore needs.
//
// It is the handful of calls FileStore makes, and it is declared here rather
// than imported so that this store does not depend on hesape/filesystem to
// compile. The Filesystem of that module satisfies this interface.
//
// PutIfAbsent is the one method that is not an ordinary file operation: it is
// "create it only if nobody else did", the file system offers it atomically,
// and FileStore.Add is nothing but that.
type Filesystem interface {
	// Get returns the contents of a file.
	Get(path string) ([]byte, error)

	// Put writes contents to a file, creating or replacing it. The write is
	// atomic as far as a reader is concerned: nobody sees half of it.
	Put(path string, contents []byte, mode fs.FileMode) error

	// PutIfAbsent writes contents only if the file does not exist, and reports
	// whether it did.
	PutIfAbsent(path string, contents []byte, mode fs.FileMode) (bool, error)

	// Exists reports whether a path is there.
	Exists(path string) bool

	// Delete removes a file. Removing what is not there is not an error.
	Delete(path string) error

	// MakeDirectory creates a directory and its parents.
	MakeDirectory(path string, mode fs.FileMode) error

	// DeleteDirectory removes a directory and everything under it.
	DeleteDirectory(path string) error

	// IsDirectory reports whether a path is a directory.
	IsDirectory(path string) bool

	// Files returns the paths of the files directly inside a directory, and
	// nothing from its subdirectories.
	Files(path string) ([]string, error)

	// Directories returns the paths of the directories directly inside a
	// directory.
	Directories(path string) ([]string, error)

	// Chmod sets a path's permissions.
	Chmod(path string, mode fs.FileMode) error
}

// LocalFilesystem is the Filesystem over the machine's own disk.
//
// It is the default FileStore is built with, and it is the whole of what this
// package knows about files: everything else here goes through the interface.
type LocalFilesystem struct{}

var _ Filesystem = LocalFilesystem{}

// Get returns the contents of a file.
func (LocalFilesystem) Get(path string) ([]byte, error) { return os.ReadFile(path) }

// Put writes contents to a file through a temporary one, then renames it.
//
// The rename is what makes it atomic: a reader arriving mid-write sees the old
// file or the new one, never half of the new one.
func (LocalFilesystem) Put(path string, contents []byte, mode fs.FileMode) error {
	tmp, err := os.CreateTemp(filepath.Dir(path), ".tmp-*")
	if err != nil {
		return err
	}
	name := tmp.Name()
	defer func() { _ = os.Remove(name) }()

	if _, err := tmp.Write(contents); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if mode != 0 {
		if err := os.Chmod(name, mode); err != nil {
			return err
		}
	}
	return os.Rename(name, path)
}

// PutIfAbsent writes contents only if the file does not exist.
//
// It writes a temporary file and hard-links it into place, because the link is
// the atomic step: exactly one of N processes racing for the same path gets it,
// and the losers are told the file exists rather than overwriting each other.
func (LocalFilesystem) PutIfAbsent(path string, contents []byte, mode fs.FileMode) (bool, error) {
	tmp, err := os.CreateTemp(filepath.Dir(path), ".tmp-*")
	if err != nil {
		return false, err
	}
	name := tmp.Name()
	defer func() { _ = os.Remove(name) }()

	if _, err := tmp.Write(contents); err != nil {
		_ = tmp.Close()
		return false, err
	}
	if err := tmp.Close(); err != nil {
		return false, err
	}
	if mode != 0 {
		if err := os.Chmod(name, mode); err != nil {
			return false, err
		}
	}

	switch err := os.Link(name, path); {
	case err == nil:
		return true, nil
	case errors.Is(err, fs.ErrExist):
		return false, nil
	default:
		return false, err
	}
}

// Exists reports whether a path is there.
func (LocalFilesystem) Exists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// Delete removes a file. Removing what is not there is not an error.
func (LocalFilesystem) Delete(path string) error {
	if err := os.Remove(path); err != nil && !errors.Is(err, fs.ErrNotExist) {
		return err
	}
	return nil
}

// MakeDirectory creates a directory and its parents.
func (LocalFilesystem) MakeDirectory(path string, mode fs.FileMode) error {
	if mode == 0 {
		mode = 0o755
	}
	return os.MkdirAll(path, mode)
}

// DeleteDirectory removes a directory and everything under it.
func (LocalFilesystem) DeleteDirectory(path string) error { return os.RemoveAll(path) }

// IsDirectory reports whether a path is a directory.
func (LocalFilesystem) IsDirectory(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.IsDir()
}

// Files returns the paths of the files directly inside a directory.
func (LocalFilesystem) Files(path string) ([]string, error) {
	entries, err := os.ReadDir(path)
	if err != nil {
		return nil, err
	}
	out := make([]string, 0, len(entries))
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		out = append(out, filepath.Join(path, e.Name()))
	}
	return out, nil
}

// Directories returns the paths of the directories directly inside a directory.
func (LocalFilesystem) Directories(path string) ([]string, error) {
	entries, err := os.ReadDir(path)
	if err != nil {
		return nil, err
	}
	out := make([]string, 0, len(entries))
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		out = append(out, filepath.Join(path, e.Name()))
	}
	return out, nil
}

// Chmod sets a path's permissions.
func (LocalFilesystem) Chmod(path string, mode fs.FileMode) error { return os.Chmod(path, mode) }

// FileStore is the cache on disk.
//
// It is the store for a single machine that wants its cache to survive a
// restart, and for several processes on that machine to share one -- which is
// exactly what ArrayStore cannot do. It is still not the store for N replicas:
// a Forget on one machine leaves the others serving the old value until the ttl
// runs out, and when that matters the store is the RESP one in hesape/redis, a
// separate module registered through CacheManager.Extend.
//
// # The layout
//
// One file per entry, at directory/xx/yy/<sha1 of the key>, where xx and yy are
// the first four characters of that hash. The two levels are there because a
// hundred thousand files in one directory is a directory nothing can list.
//
// The file begins with ten digits of unix expiry, then three digits of
// millisecond, eight hex digits of key length, then the key, then the value.
//
// The key is stored because Store.Flush takes a prefix -- one tenant's slice of
// one namespace -- and a hashed path cannot be matched against one. Without it,
// a cache:clear for one customer would empty the cache for all of them, which is
// the outage this package refuses to make possible. The millisecond is there
// because a store whose resolution is a second cannot honour a ttl shorter than
// one, and every test that pins the behaviour of a cache is written in fractions
// of a second.
type FileStore struct {
	files      Filesystem
	directory  string
	lockDir    string
	permission fs.FileMode

	// mu serializes the read-modify-write of Increment within this process.
	//
	// Across processes it is not serialized: two of them incrementing the same
	// counter in the same instant can lose one of the two. A counter that has
	// to be exact belongs in a store that counts atomically, which is what the
	// RESP store does.
	mu sync.Mutex
}

var (
	_ Store         = (*FileStore)(nil)
	_ Locking       = (*FileStore)(nil)
	_ CanFlushLocks = (*FileStore)(nil)
	_ CurrentOwner  = (*FileStore)(nil)
	_ Taggable      = (*FileStore)(nil)
)

// NewFileStore returns a store over a directory.
//
// Pass LocalFilesystem{} for files unless something is standing in for the
// disk. A permission of zero leaves whatever the umask produced.
func NewFileStore(files Filesystem, directory string, permission fs.FileMode) *FileStore {
	if files == nil {
		files = LocalFilesystem{}
	}
	return &FileStore{files: files, directory: directory, permission: permission}
}

// Path is the file a key is stored in.
//
// It is exported because something eventually has to go and look.
func (s *FileStore) Path(key string) string {
	sum := sha1.Sum([]byte(key))
	hash := hex.EncodeToString(sum[:])
	return filepath.Join(s.directory, hash[0:2], hash[2:4], hash)
}

// lockPath is Path against the lock directory.
func (s *FileStore) lockPath(key string) string {
	dir := s.lockDir
	if dir == "" {
		dir = s.directory
	}
	sum := sha1.Sum([]byte(key))
	hash := hex.EncodeToString(sum[:])
	return filepath.Join(dir, hash[0:2], hash[2:4], hash)
}

// GetFilesystem returns the file system this store writes through.
func (s *FileStore) GetFilesystem() Filesystem { return s.files }

// GetDirectory is the working directory of the cache.
func (s *FileStore) GetDirectory() string { return s.directory }

// SetDirectory sets the working directory of the cache and returns the store.
//
// It mutates rather than derives: a store is built at wiring time and read
// afterwards, and the deriving that Repository does is for the object an
// application passes around, which this is not.
func (s *FileStore) SetDirectory(directory string) *FileStore {
	s.directory = directory
	return s
}

// SetLockDirectory sets where locks are kept and returns the store.
//
// Point it somewhere other than the cache directory and FlushLocks becomes
// possible: see HasSeparateLockStore.
func (s *FileStore) SetLockDirectory(directory string) *FileStore {
	s.lockDir = directory
	return s
}

// GetPrefix is the empty string: the prefixing that matters -- tenant and
// namespace -- happens in Repository, where it can be got right once.
func (s *FileStore) GetPrefix() string { return "" }

// HasSeparateLockStore reports whether locks live apart from entries: a lock
// directory was set, and it is not the cache directory.
func (s *FileStore) HasSeparateLockStore() bool {
	return s.lockDir != "" && s.lockDir != s.directory
}

// Get returns the stored bytes, or ErrNotFound.
//
// An expired entry is deleted on the way out rather than left to accumulate,
// which is what FileStore::getPayload() does and is the only cleanup this store
// has.
func (s *FileStore) Get(_ context.Context, key string) ([]byte, error) {
	value, _, err := s.payload(key)
	return value, err
}

// payload reads an entry and returns its value and how long it has left.
//
// A file that cannot be read, cannot be parsed, or has expired is a miss --
// and the last two also delete it, because a file nothing can use is a file
// nothing will ever remove.
func (s *FileStore) payload(key string) ([]byte, time.Duration, error) {
	path := s.Path(key)

	raw, err := s.files.Get(path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil, 0, ErrNotFound
		}
		return nil, 0, fmt.Errorf("cache: reading %s: %w", path, err)
	}

	storedKey, value, expires, ok := decodeFileEntry(raw)
	if !ok || storedKey != key {
		_ = s.files.Delete(path)
		return nil, 0, ErrNotFound
	}
	if !time.Now().Before(expires) {
		_ = s.files.Delete(path)
		return nil, 0, ErrNotFound
	}
	return value, time.Until(expires), nil
}

// Many returns the stored bytes for several keys at once.
func (s *FileStore) Many(ctx context.Context, keys []string) (map[string][]byte, error) {
	out := make(map[string][]byte, len(keys))
	for _, key := range keys {
		value, err := s.Get(ctx, key)
		if err != nil {
			if errors.Is(err, ErrNotFound) {
				out[key] = nil
				continue
			}
			return nil, err
		}
		out[key] = value
	}
	return out, nil
}

// Put stores value for ttl, replacing whatever was there.
func (s *FileStore) Put(_ context.Context, key string, value []byte, ttl time.Duration) error {
	if ttl <= 0 {
		return ErrNoTTL
	}
	return s.write(s.Path(key), key, value, ttl)
}

// PutMany stores several values under one ttl.
func (s *FileStore) PutMany(ctx context.Context, values map[string][]byte, ttl time.Duration) error {
	for key, value := range values {
		if err := s.Put(ctx, key, value, ttl); err != nil {
			return err
		}
	}
	return nil
}

// write puts one entry at a path, creating the two directory levels first.
func (s *FileStore) write(path, key string, value []byte, ttl time.Duration) error {
	if err := s.ensureDirectory(path); err != nil {
		return err
	}
	return s.files.Put(path, encodeFileEntry(key, value, time.Now().Add(ttl)), s.permission)
}

// ensureDirectory creates the shard a path lives in.
//
// The permission is applied twice: two levels are created at once, so both of
// them are checked.
func (s *FileStore) ensureDirectory(path string) error {
	dir := filepath.Dir(path)
	if s.files.Exists(dir) {
		return nil
	}
	if err := s.files.MakeDirectory(dir, 0o777); err != nil {
		return fmt.Errorf("cache: creating %s: %w", dir, err)
	}
	if s.permission != 0 {
		_ = s.files.Chmod(dir, s.permission)
		_ = s.files.Chmod(filepath.Dir(dir), s.permission)
	}
	return nil
}

// Add stores value only if the key is absent, and reports whether it did.
//
// It is atomic across processes: the entry is written to a temporary file and
// linked into place, and exactly one of N racers gets the link.
//
// An entry that is there but expired is replaced.
func (s *FileStore) Add(_ context.Context, key string, value []byte, ttl time.Duration) (bool, error) {
	if ttl <= 0 {
		return false, ErrNoTTL
	}
	path := s.Path(key)
	if err := s.ensureDirectory(path); err != nil {
		return false, err
	}

	// Two passes, not a loop: the first fails when somebody holds the key, the
	// second when somebody took it in the instant between finding it expired and
	// removing it. A third pass would be a third racer, and at that point the
	// honest answer is that somebody else has it.
	for range 2 {
		added, err := s.files.PutIfAbsent(path, encodeFileEntry(key, value, time.Now().Add(ttl)), s.permission)
		if err != nil {
			return false, fmt.Errorf("cache: writing %s: %w", path, err)
		}
		if added {
			return true, nil
		}

		// Taken. It counts only while it is live, so read it, and if it has
		// expired take it away from nobody and go round again.
		if _, _, err := s.payload(key); err == nil {
			return false, nil
		} else if !errors.Is(err, ErrNotFound) {
			return false, err
		}
		if err := s.files.Delete(path); err != nil {
			return false, err
		}
	}
	return false, nil
}

// Forever stores a value with no expiry the caller has to think about.
//
// The expiry is written a century out rather than left off: see
// Repository.Forever.
func (s *FileStore) Forever(ctx context.Context, key string, value []byte) error {
	return s.Put(ctx, key, value, foreverTTL)
}

// Increment adds delta to the counter under key and returns the new value.
//
// The part that is easy to miss: the counter keeps the deadline it was created
// with. The remaining time is written back unchanged, because refreshing it
// instead would make a fixed window one that never closes.
func (s *FileStore) Increment(_ context.Context, key string, delta int64, ttl time.Duration) (int64, error) {
	if ttl <= 0 {
		return 0, ErrNoTTL
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	n := int64(0)
	remaining := ttl
	switch value, left, err := s.payload(key); {
	case err == nil:
		parsed, perr := strconv.ParseInt(string(value), 10, 64)
		if perr != nil {
			return 0, errNotACounter(key)
		}
		n = parsed
		remaining = left
	case errors.Is(err, ErrNotFound):
	default:
		return 0, err
	}

	n += delta
	if err := s.write(s.Path(key), key, []byte(strconv.FormatInt(n, 10)), remaining); err != nil {
		return 0, err
	}
	return n, nil
}

// Decrement subtracts delta from the counter under key. It is Increment with
// the sign turned round.
func (s *FileStore) Decrement(ctx context.Context, key string, delta int64, ttl time.Duration) (int64, error) {
	return s.Increment(ctx, key, -delta, ttl)
}

// Touch gives a live entry a new expiry and reports whether there was one. An
// absent key is false and is not created.
func (s *FileStore) Touch(_ context.Context, key string, ttl time.Duration) (bool, error) {
	if ttl <= 0 {
		return false, ErrNoTTL
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	value, _, err := s.payload(key)
	if errors.Is(err, ErrNotFound) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	if err := s.write(s.Path(key), key, value, ttl); err != nil {
		return false, err
	}
	return true, nil
}

// Forget removes a key, present or not.
//
// The companion entry goes with it: a flexible value keeps its timestamp under
// a second key, and leaving that behind would have the next reader age a value
// that is no longer there.
func (s *FileStore) Forget(_ context.Context, key string) error {
	if err := s.files.Delete(s.Path(key)); err != nil {
		return err
	}
	return s.files.Delete(s.Path(flexibleCreatedKey(key)))
}

// Flush removes every entry whose key begins with prefix.
//
// This store holds every tenant's cache, so it reads each entry's key and
// removes the ones in the prefix -- which is what the key in the file header is
// for. An empty prefix removes everything.
func (s *FileStore) Flush(_ context.Context, prefix string) error {
	if !s.files.IsDirectory(s.directory) {
		return nil
	}
	if prefix == "" {
		return s.flushEverything(s.directory)
	}
	return s.walk(s.directory, func(path string, key string) error {
		if !strings.HasPrefix(key, prefix) {
			return nil
		}
		return s.files.Delete(path)
	})
}

// flushEverything removes the shard directories under root.
func (s *FileStore) flushEverything(root string) error {
	dirs, err := s.files.Directories(root)
	if err != nil {
		return err
	}
	for _, dir := range dirs {
		if err := s.files.DeleteDirectory(dir); err != nil {
			return err
		}
	}
	return nil
}

// FlushLocks releases every lock this store holds.
//
// A store keeping its locks in the cache directory cannot empty the first
// without emptying the second, and says so instead of doing it. Call
// SetLockDirectory to make it possible.
func (s *FileStore) FlushLocks(_ context.Context) error {
	if !s.HasSeparateLockStore() {
		return fmt.Errorf("%w: flushing locks needs a lock directory separate from the cache directory, and this store has none", ErrUnsupported)
	}
	if !s.files.IsDirectory(s.lockDir) {
		return nil
	}
	return s.flushEverything(s.lockDir)
}

// walk visits every entry file under root and hands its path and stored key to
// fn.
func (s *FileStore) walk(root string, fn func(path, key string) error) error {
	firsts, err := s.files.Directories(root)
	if err != nil {
		return err
	}
	for _, first := range firsts {
		seconds, err := s.files.Directories(first)
		if err != nil {
			return err
		}
		for _, second := range seconds {
			files, err := s.files.Files(second)
			if err != nil {
				return err
			}
			for _, path := range files {
				raw, err := s.files.Get(path)
				if err != nil {
					// A file that vanished between the listing and the read was
					// removed by somebody else, which is the outcome wanted.
					continue
				}
				key, _, _, ok := decodeFileEntry(raw)
				if !ok {
					continue
				}
				if err := fn(path, key); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// AcquireLock takes the lock if it is free.
//
// The lock is one more entry, in the lock directory when there is one, and the
// exclusive create is what makes it a lock at all.
func (s *FileStore) AcquireLock(_ context.Context, key, token string, ttl time.Duration) (bool, error) {
	if ttl <= 0 {
		return false, ErrNoTTL
	}
	path := s.lockPath(key)
	if err := s.ensureDirectory(path); err != nil {
		return false, err
	}

	for range 2 {
		taken, err := s.files.PutIfAbsent(path, encodeFileEntry(key, []byte(token), time.Now().Add(ttl)), s.permission)
		if err != nil {
			return false, err
		}
		if taken {
			return true, nil
		}
		owner, err := s.CurrentOwner(context.Background(), key)
		if err != nil {
			return false, err
		}
		if owner != "" {
			return false, nil
		}
		if err := s.files.Delete(path); err != nil {
			return false, err
		}
	}
	return false, nil
}

// ReleaseLock releases the lock only if token still holds it.
func (s *FileStore) ReleaseLock(ctx context.Context, key, token string) error {
	owner, err := s.CurrentOwner(ctx, key)
	if err != nil {
		return err
	}
	if owner != token {
		// Expired, or held by somebody else. Deleting it here is the bug the
		// token exists to prevent.
		return nil
	}
	return s.files.Delete(s.lockPath(key))
}

// CurrentOwner returns the token holding the lock, or the empty string.
func (s *FileStore) CurrentOwner(_ context.Context, key string) (string, error) {
	path := s.lockPath(key)
	raw, err := s.files.Get(path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return "", nil
		}
		return "", err
	}
	storedKey, value, expires, ok := decodeFileEntry(raw)
	if !ok || storedKey != key {
		_ = s.files.Delete(path)
		return "", nil
	}
	if !time.Now().Before(expires) {
		_ = s.files.Delete(path)
		return "", nil
	}
	return string(value), nil
}

// Lock returns a handle on a named lock. It does not touch the store.
func (s *FileStore) Lock(name string, ttl time.Duration, owner string) *Lock {
	return &Lock{store: s, name: name, ttl: ttl, owner: owner, held: owner != ""}
}

// RestoreLock returns a handle on a lock owner already holds.
func (s *FileStore) RestoreLock(name, owner string) *Lock { return s.Lock(name, 0, owner) }

// fileHeader is the fixed part of an entry file: ten digits of unix expiry,
// three of the millisecond within that second, and eight hex digits of key
// length.
const fileHeader = 10 + 3 + 8

// encodeFileEntry lays out one entry.
//
// The expiry comes first and is ten digits wide, which stays ten digits until
// the year 2286. Anything past that is clamped, because an eleven-digit header
// would be a file the previous version of the binary reads as a different
// entry.
//
// The three digits after it are the millisecond. Without them a ttl of eighty
// milliseconds would be anything from expired-on-arrival to alive for a second
// -- and the tests that pin the behaviour of a cache are all written in
// fractions of a second, because nobody waits an hour for a suite. The
// precision is here so that a store on disk expires when it was told to.
func encodeFileEntry(key string, value []byte, expires time.Time) []byte {
	unix := expires.Unix()
	millis := expires.Nanosecond() / int(time.Millisecond)
	if unix > 9999999999 || unix < 0 {
		unix, millis = 9999999999, 0
	}

	out := make([]byte, 0, fileHeader+len(key)+len(value))
	out = append(out, fmt.Sprintf("%010d%03d%08x", unix, millis, len(key))...)
	out = append(out, key...)
	out = append(out, value...)
	return out
}

// decodeFileEntry reads one back. A file that is too short, or whose header does
// not parse, is not an entry -- which is what a half-written or truncated file
// looks like, and treating it as a miss is what lets the store clean it up.
func decodeFileEntry(raw []byte) (key string, value []byte, expires time.Time, ok bool) {
	if len(raw) < fileHeader {
		return "", nil, time.Time{}, false
	}
	unix, err := strconv.ParseInt(string(raw[:10]), 10, 64)
	if err != nil {
		return "", nil, time.Time{}, false
	}
	millis, err := strconv.ParseInt(string(raw[10:13]), 10, 32)
	if err != nil {
		return "", nil, time.Time{}, false
	}
	length, err := strconv.ParseUint(string(raw[13:fileHeader]), 16, 32)
	if err != nil {
		return "", nil, time.Time{}, false
	}
	if uint64(len(raw)) < uint64(fileHeader)+length {
		return "", nil, time.Time{}, false
	}
	return string(raw[fileHeader : uint64(fileHeader)+length]),
		raw[uint64(fileHeader)+length:],
		time.Unix(unix, millis*int64(time.Millisecond)),
		true
}
