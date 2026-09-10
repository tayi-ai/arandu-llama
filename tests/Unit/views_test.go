package unit_test

import (
	"context"
	"io"
	"io/fs"
	"path"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/arandu-io/framework/data"
	"github.com/arandu-io/framework/foundation"
	"github.com/arandu-io/framework/security"
	"github.com/arandu-io/hesape/view"

	llama "github.com/tayi-ai/arandu-llama"
)

// How a view of this package reaches a page, and why there is only one way.
//
// A view is not read at request time. It is compiled to Go before the binary
// exists, and the generated file registers itself from init(), so by the time
// anything serves a request the set of views is fixed and the process has no
// choice left to make. A renderer that looked in two directories would be
// looking after the decision was taken.
//
// So the application takes ownership instead of shadowing. Publishing copies
// the source into the project, `aru view:build` compiles what is in the
// project, and from then on there is one source for that page and it is the
// project's. Nothing of this package is compiled beside it, so no name is
// registered twice and no rule has to say which of two files won.
//
// The consequence is the property held below: a view that was never published
// is not registered, and the module refuses to boot rather than answering the
// first request that reaches it with a 500.

// sessionKey is any thirty-two bytes; a real application reads its own from the
// environment.
const sessionKey = "0123456789abcdef0123456789abcdef"

func module(t *testing.T) *llama.Module {
	t.Helper()

	sessions := security.NewSessionStore([]byte(sessionKey), time.Hour, false, security.NewMemoryBackend())
	m, err := llama.New(llama.Config{Tenant: "acme"}, data.Wrap(nil, data.DialectSQLite), sessions)
	if err != nil {
		t.Fatalf("building the module: %v", err)
	}
	return m
}

// TestTheModuleDeclaresWhatItPublishes holds the contract itself: the module is
// asked for its files the way it is asked for its migrations, through an
// optional interface it either answers or does not.
//
// The declaration is read the way whatever publishes reads it -- through
// foundation.Publications, which is where a tag outside the closed set is
// refused -- rather than by calling the method and trusting what comes back.
func TestTheModuleDeclaresWhatItPublishes(t *testing.T) {
	t.Parallel()

	var mod foundation.Module = module(t)
	if _, ok := mod.(foundation.Publishable); !ok {
		t.Fatal("the module does not answer the publishing contract, so nothing can find its files")
	}

	publications, err := foundation.Publications(mod)
	if err != nil {
		t.Fatalf("reading what the module publishes: %v", err)
	}
	if len(publications) != 1 {
		t.Fatalf("the module declares %d publication(s), want one: this package hands over markup and nothing else", len(publications))
	}
	if tag := publications[0].Tag; tag != foundation.PublishView {
		t.Errorf("the files are published as %q, want %q", tag, foundation.PublishView)
	}
	if publications[0].Files == nil {
		t.Error("the publication carries no files, so the declaration promises what nothing can write")
	}

	if _, ok := mod.(foundation.Bootable); !ok {
		t.Fatal("the module does not boot, so a view that was never published would be answered as a 500 per request")
	}
}

// TestThePublicationLandsWhereTheModuleSaysItDoes keeps the declaration and the
// refusal reading one set of paths.
//
// Boot checks PublishedPaths, and what publishes writes wherever the
// publication resolves to: a source or destination directory on the publication
// would move the files without moving the check, and the fail-closed
// destination below would be guarding paths nothing writes.
func TestThePublicationLandsWhereTheModuleSaysItDoes(t *testing.T) {
	t.Parallel()

	publications, err := foundation.Publications(module(t))
	if err != nil {
		t.Fatalf("reading what the module publishes: %v", err)
	}

	var targets []string
	for _, publication := range publications {
		from := publication.From
		if from == "" {
			from = "."
		}
		err := fs.WalkDir(publication.Files, from, func(name string, entry fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if entry.IsDir() {
				return nil
			}
			relative := name
			if from != "." {
				relative = strings.TrimPrefix(strings.TrimPrefix(name, from), "/")
			}
			targets = append(targets, path.Join(publication.To, relative))
			return nil
		})
		if err != nil {
			t.Fatalf("walking the %s publication: %v", publication.Tag, err)
		}
	}

	slices.Sort(targets)
	if want := llama.PublishedPaths(); !slices.Equal(targets, want) {
		t.Errorf("the publication writes %v, and the module checks %v", targets, want)
	}
}

// TestEveryPublishedFileLandsUnderTheModuleNamespace is the fail-closed half of
// the destination. What publishes writes what the archive says, so an archive
// naming resources/views/home.kyse.go would land on a page the application
// wrote -- and the person running the install command would find out from the
// page.
func TestEveryPublishedFileLandsUnderTheModuleNamespace(t *testing.T) {
	t.Parallel()

	m := module(t)
	prefix := "resources/views/modules/" + m.Name() + "/"

	paths := llama.PublishedPaths()
	if len(paths) == 0 {
		t.Fatal("the package publishes nothing, so every check below would pass by having nothing to read")
	}
	for _, path := range paths {
		if !strings.HasPrefix(path, prefix) {
			t.Errorf("%s is published outside %s, where it lands on a file the application wrote", path, prefix)
		}
		if !strings.HasSuffix(path, ".kyse.go") {
			t.Errorf("%s is published as something the view compiler does not read", path)
		}
	}
}

// TestTheNameAViewIsRenderedByComesFromItsPath keeps the three spellings of one
// view together: the path in the archive, the name a handler renders it by, and
// the package the compiled form lands in. They are derived from one another
// rather than written down three times, and this is where that is checked.
func TestTheNameAViewIsRenderedByComesFromItsPath(t *testing.T) {
	t.Parallel()

	paths := llama.PublishedPaths()
	names := llama.ViewNames()
	if len(names) != len(paths) {
		t.Fatalf("%d view(s) published and %d name(s) declared", len(paths), len(names))
	}

	for i, path := range paths {
		bare := strings.TrimSuffix(strings.TrimPrefix(path, "resources/views/"), ".kyse.go")
		if want := strings.ReplaceAll(bare, "/", "."); names[i] != want {
			t.Errorf("%s is rendered as %q, want %q", path, names[i], want)
		}
	}

	packages := llama.ViewPackages()
	if len(packages) == 0 {
		t.Fatal("the views compile into no package, so there is no import that would link them")
	}
	for _, pkg := range packages {
		if !strings.HasPrefix(pkg, "storage/framework/views/") {
			t.Errorf("%s is not where the view compiler writes, so importing it would not link anything", pkg)
		}
	}
}

// TestTheModuleRefusesToBootUntilItsViewsAreLinked is the resolution, run in the
// order it happens.
//
// Before: nothing published, nothing compiled, nothing imported, so the registry
// holds none of these names and the module refuses -- naming the view, the
// command and the package to import, because a refusal that does not say what
// to do next is a refusal somebody works around.
//
// After: the names are in the registry, which is the state publishing produces
// once the views are compiled and the generated package is imported. The module
// boots.
//
// It is one test and it does not run in parallel, because the registry is the
// process's and registering a name is not something that can be taken back --
// which is itself the reason there is no shadowing: a second file claiming a
// registered name is refused, not preferred.
func TestTheModuleRefusesToBootUntilItsViewsAreLinked(t *testing.T) {
	m := module(t)
	ctx := context.Background()

	names := llama.ViewNames()
	if len(names) == 0 {
		t.Fatal("the package declares no view, so neither half of this test would prove anything")
	}

	err := m.Boot(ctx)
	if err == nil {
		t.Fatal("the module booted with none of its views linked, so a request would have been the first thing to say so")
	}
	for _, name := range names {
		if !strings.Contains(err.Error(), name) {
			t.Errorf("the refusal does not name the missing view %s: %v", name, err)
		}
	}
	if !strings.Contains(err.Error(), llama.PublishCommand) {
		t.Errorf("the refusal does not say how to publish the views: %v", err)
	}
	for _, pkg := range llama.ViewPackages() {
		if !strings.Contains(err.Error(), pkg) {
			t.Errorf("the refusal does not name %s, the package whose import links the views: %v", pkg, err)
		}
	}

	// What the compiled view does from init(), done here because there is no
	// compiled view in a package that is not an application.
	for _, name := range names {
		view.Register(name, func(io.Writer, any) error { return nil })
	}

	if err := m.Boot(ctx); err != nil {
		t.Fatalf("the module refused to boot with every view linked: %v", err)
	}
}

// TestThePackageShipsNoCommandOfItsOwn holds the mechanism rather than the
// declaration: publishing is declared here and performed by the CLI, which
// reads the modules an application registered and writes what each of them
// offers.
//
// A program in this module that copied files into a project would be a second
// answer to "how do I get these views", and two answers disagree the first time
// one of them learns something the other does not. The file behind a build tag
// is not one, because the compiler never reads it.
func TestThePackageShipsNoCommandOfItsOwn(t *testing.T) {
	t.Parallel()

	for _, source := range productionGoFiles(t, packageRoot(t)) {
		if source.file.Name.Name == "main" && buildable(source.file) {
			t.Errorf("%s is a command this module carries; what a package publishes is declared, and %s writes it",
				source.path, llama.PublishCommand)
		}
	}
}
