// Package filesystem is the file contract: what a tenant uploaded, and where it
// went.
//
// The shape, in one paragraph. A [Disk] is what an application calls, and every
// one of its methods takes an [auth.Grant]; an [Adapter] is what a driver
// implements, and it never hears of a tenant. Between the two sits [Key], which
// turns a Grant and a key into the one stored path that Grant may reach. That
// split is the whole design: tenant isolation is a property of this package,
// not of each driver remembering to ask.
//
// The contract lives in the collection and so does the driver that needs
// nothing installed: [LocalFilesystemAdapter] is a directory, and it is the one
// to develop against and to run a single machine on.
//
// Object storage is github.com/arandu-io/hesape/filesystem/s3 -- the same
// contract over the S3 protocol, with Cloudflare R2 as the default, and a module
// of its own so that a project storing on disk does not carry it. In Go there is
// no optional dependency, which is the whole reason that line exists. Register
// it with [Disks.Extend]: there is no built-in S3 driver name, because the root
// module cannot import its own submodule.
//
// That module speaks the protocol over net/http and signs its own requests, so
// importing it brings no cloud SDK with it, and it hands back no client object
// for a caller to reach past the adapter with.
//
// A file is customer data, and a path without a tenant is a leak with a
// directory name -- which is why the Grant is on the collection's side of the
// split and not on the driver's, the same way it is for database.Repository and
// queue.Queue.
//
// There is no symlink into a document root. Publishing a storage directory that
// way makes every stored file world-readable by URL and turns authorization
// into "hope nobody guesses the name". Here a file is served by a route, and
// the route runs a Policy like any other -- see [Serve] for the serving half and
// [URLSigner] for the link that stands in for a session.
//
// The way in from a form is [Upload], checked against [UploadRules]. It is the
// "own contract" that validation refuses uploads in favour of: a file is not in
// url.Values, and validating one inside the form pipeline would be a second way
// to validate.
//
// # Two file APIs, and the line between them
//
// [Disk] is customer data: every method takes an [auth.Grant] and every path it
// builds starts with a tenant. [Filesystem] is the application's own files --
// stubs, compiled views, session files, cache entries -- and takes plain paths,
// exactly as os.ReadFile does. The failure worth naming is putting an upload
// through the second one, where there is no prefix and therefore no isolation.
//
// [LockableFile] is what makes the second one safe to share between processes,
// and [Filesystem.SharedGet] and [Filesystem.Put] with lock are the pattern.
//
// # Testing
//
// A test builds the disk it wants and passes it, the same way it passes every
// other collaborator:
//
//	adapter, err := filesystem.NewLocalFilesystemAdapter(t.TempDir())
//	if err != nil {
//	    t.Fatal(err)
//	}
//	disk := filesystem.NewDisk("local", adapter)
//
//	archiveInvoice(ctx, g, disk)
//
//	disk.AssertExists(ctx, t, g, "invoices/2026-114.pdf")
//
// t.TempDir() is what keeps it isolated: the directory is made for this test and
// removed when it ends, so two tests running in parallel cannot see each other's
// files. A directory of your own in place of it is the version that outlives the
// run. The assertions are already on [Disk]: AssertExists, AssertMissing,
// AssertCount and AssertDirectoryEmpty.
//
// # URL, GetVisibility and MakeDirectory
//
// All three are here, and each of them means less than its name suggests:
//
//   - [Disk.URL] is the permanent public address of a file. It carries no
//     authorization at all, so it answers [ErrNoURL] unless the disk was
//     configured with one. The answer for a tenant's file is
//     [Disk.TemporaryURL] or [URLSigner.TemporaryURL], which expire.
//   - [Disk.GetVisibility] and [Disk.SetVisibility] report and set what the
//     STORE will hand out to somebody who never came through the application.
//     They do not replace a Policy: every read here still takes a Grant.
//   - [Disk.MakeDirectory] makes a directory on a driver that has them, and
//     succeeds without doing anything on one that does not -- an object store
//     has no directories, and a marker object standing in for one would put a
//     key in every listing that [Disk.Get] then answers [ErrNotFound] for.
package filesystem
