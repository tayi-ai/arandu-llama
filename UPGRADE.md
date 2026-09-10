# Upgrade Guide

## Unreleased

## v0.2.0

Nothing to upgrade from. This is the first release of this repository.

### Do not resolve v0.1.x of this module path

`github.com/tayi-ai/arandu-llama` has `v0.1.0`, `v0.1.1` and `v0.1.2` in the Go
proxy from a previous repository, and their checksums do not match this
history. A `go.mod` still requiring one of those gets a checksum mismatch,
which the go command reports as a security error rather than as a version
problem:

```text
SECURITY ERROR
This download does NOT match the one reported by the checksum server.
```

Move to `v0.2.0`:

```bash
go get github.com/tayi-ai/arandu-llama@v0.2.0
```

Nothing else changes: the import path, the package name and the exported API
are the same.

### Installing for the first time

Three things beyond `go get`, in this order:

1. **Build the static archives.** The module ships no compiled code and will
   compile but not link without them — `ld: library 'binding' not found` is the
   expected first failure. See [docs/building.md](docs/building.md).
2. **Register the module** in `bootstrap/app.go`, and pass a `Tenant`. It is
   required and has no default.
3. **Run `aru migrate`** before the application serves. The package owns the
   `llamas` table.

The policy denies every action until you open one. A 404 from a route you
believe exists is usually the tenant check, not routing.

### Published views move out of `vendor/`

The view a package publishes lands in `resources/views/modules/<slug>/` and
compiles to `storage/framework/views/modules/<slug>`. It used to be `vendor/` in
both, and that address could not work: the go command reserves the name twice,
and a tree of published views hit both rules.

A file under a directory named `vendor` is left out of the module zip at any
depth. The file stays in the package's repository and is missing for everyone
who downloads it, so the `go:embed` that names its directory matches nothing and
the person building the project reads

```
pattern resources/views: no matching files found
```

— an error about the package, raised in their project. And a package whose
import path carries the element cannot be imported at all:

```
bootstrap/app.go:98:2: use of vendored package not allowed
```

which is exactly the import `(*Module).Boot` asks for. A published view is
compiled into a Go package the application has to import for its `init()` to
register anything, so the second rule refused the last step of the install.

Both were reproduced before this changed: `zip.CheckDir` reports the view as
`file is in vendor directory`, and a package under
`storage/framework/views/vendor/<slug>` is refused at import.

To move a package already released:

1. `git mv resources/views/vendor resources/views/modules`.
2. Rename the `vendorDir` constant in `views.go` to `moduleDir`, with the value
   `modules`.
3. Release the package, and tell the projects that installed it to publish
   again. The old files are theirs now, so `aru vendor:publish --apply` writes
   the new tree beside the old one and the old one is deleted by hand, along
   with its lines in `vendor-publish.lock` and its import in `bootstrap/app.go`.

Framework `v0.46.4` and Hesape `v0.37.0` refuse a publication that carries the
reserved name, so a package that has not moved fails its own tests with a
message naming both rules, rather than failing in the first project that
installs it.

