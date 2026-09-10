package llama

import (
	"embed"
	"io/fs"
	"strings"

	"github.com/arandu-io/framework/foundation"
)

// The view sources this package hands to the project that installs it.
//
// They are embedded rather than read off disk because what a module publishes
// is a tree it carries, not a directory it points at: the program that writes
// the files is the application's own binary, running somewhere this repository
// does not exist as files. Whatever is handed over has to already be inside the
// program doing the handing.
//
// The paths inside the archive are the paths the files take in the project, not
// paths of this repository that something has to translate on the way. A view
// sits here at the address it will sit at there, so the publication names
// neither a source directory nor a destination one, and there is no second
// spelling of a destination for the first one to disagree with.
//
//go:embed resources/views
var viewSources embed.FS

// Where a view is written and what it is called.
//
// viewRoot is the directory every path in the archive starts with, and it is
// what is cut off to get the name a view is registered under. viewSuffix is the
// extension: it ends in .go so the build tag on the first line keeps the
// compiler out of a file that is markup below the package clause.
const (
	viewRoot   = "resources/views"
	viewSuffix = ".kyse.go"
)

// compiledRoot is where the view compiler writes the Go it produces, mirroring
// the tree of the sources. It is build output: gitignored, rebuilt on demand,
// and never edited.
const compiledRoot = "storage/framework/views"

// moduleDir is the directory an application keeps installed packages' views in.
//
// It is part of the path in the archive and not something the publication adds,
// which is what makes the archive a literal picture of what lands in the
// project. Two packages with a view called index are two files under two names
// below it, and neither shadows the other or the application's own.
//
// It is not called vendor, and that is the one thing about this name worth
// writing down. The go command reserves that word twice over, and a tree of
// published views hits both rules:
//
//   - a file under a directory named vendor is left out of the module zip at
//     any depth, so it stays in this repository and is missing for everyone who
//     downloads the module. The embed above then matches nothing, and what the
//     person building the project reads is "pattern resources/views: no
//     matching files found" -- an error about this package, raised in theirs.
//   - a package whose import path carries the element cannot be imported at
//     all: "use of vendored package not allowed". A published view is compiled
//     into a Go package the application has to import for its init() to
//     register anything, so the second rule refuses exactly the import Boot
//     asks for.
//
// Both were measured, not reasoned about, and the second is why the name could
// not be fixed on this side alone: the destination is the address the
// application looks a package's views up at, and it is the destination that
// carried the word.
const moduleDir = "modules"

// Publishes declares the files this package offers, each at the path it takes
// relative to the root of the project.
//
// One tree, under one tag. A publication may carry a page, a component, a
// configuration file, a migration, a catalogue of sentences or an asset, and
// this package offers the first of those and nothing else. Each absence is a
// decision rather than an omission:
//
//   - configuration is the Config struct New is handed, checked by the compiler
//     and validated before the module exists. A file copied into the project
//     beside it would be a second place to say the same thing, and only one of
//     the two could be the one the code reads.
//   - a stylesheet, a script or any other asset is registered with the view
//     layer and served from an address derived from its own bytes. Copying one
//     into the project would put a second copy of those bytes under a second
//     address, and a page can only reference one of them.
//   - translations are overridden by writing the lines the application wants
//     into its own override tree, which the catalogue loader already reads. A
//     copy of every line this package ships is a copy that goes stale, and it
//     goes stale without saying so.
//   - migrations are declared and collected, never copied. A copy in the
//     project's own migration directory is found by the runner as well, so one
//     schema change applies twice under two names.
//
// What is left is the markup, and it is here for the reason the others are not:
// it is the one thing the project is expected to edit. A package cannot know
// what a screen should say in a product it has never seen.
func (m *Module) Publishes() []foundation.Publication {
	return []foundation.Publication{{Tag: foundation.PublishView, Files: viewSources}}
}

// PublishedPaths are the files in the archive, each relative to the root of the
// application, sorted.
func PublishedPaths() []string { return append([]string(nil), publishedPaths...) }

// ViewNames are the names the published views are rendered by, sorted.
//
// The name is the path under the view root with its separators turned into
// dots, which is what the view compiler writes into the registration call. It
// is derived from the same archive the publication carries, so a view that was
// renamed cannot keep an old name here.
func ViewNames() []string { return append([]string(nil), viewNames...) }

// ViewPackages are the directories the compiled views land in, each relative to
// the root of the application, sorted and without repeats.
//
// Importing one is what puts its views in the binary: a compiled view calls the
// registry from init(), and a package nothing imports is not linked at all. The
// import is named rather than written into bootstrap/app.go, because one line
// somebody reads beats a file that changed while they were not looking.
func ViewPackages() []string {
	var out []string
	seen := make(map[string]bool)
	for _, path := range publishedPaths {
		dir := compiledRoot + strings.TrimPrefix(path[:strings.LastIndexByte(path, '/')], viewRoot)
		if seen[dir] {
			continue
		}
		seen[dir] = true
		out = append(out, dir)
	}
	return out
}

// The archive, read once at load.
//
// A failure here is a broken binary rather than a condition to recover from:
// the files are compiled in, so either they are all there or the build that
// produced this program was wrong.
var publishedPaths, viewNames = readArchive()

func readArchive() (paths, names []string) {
	err := fs.WalkDir(viewSources, viewRoot, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() || !strings.HasSuffix(path, viewSuffix) {
			return nil
		}
		paths = append(paths, path)
		names = append(names, viewName(path))
		return nil
	})
	if err != nil {
		panic("llama: reading the embedded views: " + err.Error())
	}
	if len(paths) == 0 {
		panic("llama: the embedded view directory holds no view, so every name this package renders would be missing and nothing would say so")
	}
	return paths, names
}

// viewName turns an archive path into the name the view is registered under.
//
//	resources/views/modules/llama/index.kyse.go -> modules.llama.index
func viewName(path string) string {
	name := strings.TrimPrefix(strings.TrimPrefix(path, viewRoot), "/")
	name = strings.TrimSuffix(name, viewSuffix)
	return strings.ReplaceAll(name, "/", ".")
}
