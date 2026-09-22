package foundation

import (
	"fmt"
	"io/fs"
	"path"
	"strings"
)

// PublishTag names the kind of file a module publishes.
//
// The set is closed at six, and closing it is the point. A tag added on demand
// turns publishing into a plugin surface, where every module teaches the
// command a new word and nobody can say what `aru vendor:publish` does without
// reading seventeen modules. What does not fit one of the six is written in Go
// rather than published.
//
// A tag says what a file is, never where it goes: the destination is on the
// publication, because a module publishes into the directories a project
// already has and those are not one per kind.
type PublishTag string

const (
	// PublishView is a page or a layout the project is meant to edit.
	PublishView PublishTag = "view"
	// PublishComponent is a piece a view composes.
	PublishComponent PublishTag = "component"
	// PublishConfig is configuration the project owns once it is published.
	PublishConfig PublishTag = "config"
	// PublishMigration is a schema change the project applies and keeps.
	PublishMigration PublishTag = "migration"
	// PublishTranslation is a catalogue of sentences.
	PublishTranslation PublishTag = "translation"
	// PublishAsset is a file served as it is: an image, a font, a stylesheet.
	PublishAsset PublishTag = "asset"
)

// PublishTags lists the six, in the order a listing prints them.
func PublishTags() []PublishTag {
	return []PublishTag{
		PublishView,
		PublishComponent,
		PublishConfig,
		PublishMigration,
		PublishTranslation,
		PublishAsset,
	}
}

// Valid reports whether t is one of the six.
func (t PublishTag) Valid() bool {
	for _, known := range PublishTags() {
		if t == known {
			return true
		}
	}
	return false
}

// Publication is one tree of files a module publishes under one tag.
//
// It is data and not behaviour: the module says what it has and where it goes,
// and the engine that writes it lives elsewhere, so a module that publishes
// does not drag file IO into every binary that merely compiles it.
type Publication struct {
	// Tag is what kind of file this is, and what `--tag` selects on.
	Tag PublishTag
	// Files is the tree, as the module embedded it.
	Files fs.FS
	// From is the directory inside Files to publish. Empty means all of it.
	From string
	// To is the directory the files land in, relative to the project root.
	// Empty means the root itself.
	To string
}

// Publishable is optional: the module declares the files it offers a project.
//
// It is the same category as [Migratable] -- a capability a module claims about
// itself -- and it is here for the same reason: a contract only a template
// knows is a convention, and conventions drift between the modules that follow
// them.
type Publishable interface {
	Publishes() []Publication
}

// Publications returns what a module publishes, and nothing when it publishes
// nothing.
//
// It is where the closed set is enforced, because it is the one place every
// caller passes through. A tag from outside the six is an error here rather
// than a directory nobody expected: the module compiles either way, so the
// refusal has to happen when the publication is read.
func Publications(module any) ([]Publication, error) {
	publisher, ok := module.(Publishable)
	if !ok {
		return nil, nil
	}

	publications := publisher.Publishes()
	for _, publication := range publications {
		if !publication.Tag.Valid() {
			name := "a module"
			if named, ok := module.(interface{ Name() string }); ok {
				name = named.Name()
			}
			return nil, fmt.Errorf("foundation: %s publishes under %q, which is not one of the tags: %v",
				name, publication.Tag, PublishTags())
		}
		if publication.Files == nil {
			return nil, fmt.Errorf("foundation: a %s publication carries no files", publication.Tag)
		}
		if err := checkVendorSegment(module, publication); err != nil {
			return nil, err
		}
	}
	return publications, nil
}

// vendorSegment is the directory name the go command reserves.
//
// It is not a style preference. Two rules of the go command act on a path that
// carries it, and between them they make such a path unusable for a module that
// publishes files:
//
//   - a file inside a directory named vendor is left out of the module zip, at
//     any depth. The file stays in the repository and is absent for everyone
//     who downloads the module, so an embed that names its directory fails to
//     match anything and the refusal lands on whoever built the project, not on
//     whoever published the module.
//   - a package whose import path carries the element cannot be imported:
//     "use of vendored package not allowed". A published view is compiled into
//     a Go package the application has to import, so a view under this name is
//     written, compiled, and then refused by the compiler.
const vendorSegment = "vendor"

// checkVendorSegment refuses a publication the go command would take apart.
//
// The archive is checked for every tag, because the zip rule applies to any
// file. The destination is checked for the two tags that become Go: a view and
// a component are compiled into packages the application imports, and a
// configuration file, a migration, a catalogue or an asset is read at runtime
// and never named by an import path.
//
// It is checked here rather than where the files are written because here is
// the side that still has them: for a module already published, the archive has
// no such file left to find, and the only thing anybody sees is an embed that
// matched nothing. Reading the publication in the module's own tests is the
// last moment the truth is still on disk.
func checkVendorSegment(module any, publication Publication) error {
	name := "a module"
	if named, ok := module.(interface{ Name() string }); ok {
		name = named.Name()
	}

	if hasVendorSegment(publication.To) && (publication.Tag == PublishView || publication.Tag == PublishComponent) {
		return fmt.Errorf("foundation: %s publishes a %s into %q, and the go command refuses to import a package whose path carries a %q element: "+
			"the file is written and compiled, and then the build says \"use of vendored package not allowed\". Name the directory something else",
			name, publication.Tag, publication.To, vendorSegment)
	}

	from := publication.From
	if from == "" {
		from = "."
	}

	return fs.WalkDir(publication.Files, from, func(file string, entry fs.DirEntry, err error) error {
		if err != nil || entry.IsDir() {
			return err
		}
		if !hasVendorSegment(file) {
			return nil
		}
		return fmt.Errorf("foundation: %s carries %q in its %s publication, and the go command leaves a file under a %q directory out of the module zip: "+
			"it is in the repository and absent for everyone who downloads the module. Name the directory something else",
			name, file, publication.Tag, vendorSegment)
	})
}

// hasVendorSegment reports whether any element of a slash-separated path is the
// reserved name.
func hasVendorSegment(p string) bool {
	if p == "" {
		return false
	}
	for _, element := range strings.Split(path.Clean(p), "/") {
		if element == vendorSegment {
			return true
		}
	}
	return false
}
