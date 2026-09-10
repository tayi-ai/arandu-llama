package unit_test

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

func TestTheManifestFrameworkFloorMatchesGoMod(t *testing.T) {
	root := packageRoot(t)
	goMod := readReleaseFile(t, root, "go.mod")
	manifest := readReleaseFile(t, root, "arandu.mod.toml")

	required := captureReleaseValue(t, goMod,
		`(?m)^\s*github\.com/arandu-io/framework v([0-9]+\.[0-9]+)\.[0-9]+\s*$`,
		"Framework version in go.mod")
	declared := captureReleaseValue(t, manifest,
		`(?m)^framework = ">= ([0-9]+\.[0-9]+)"$`,
		"Framework floor in arandu.mod.toml")

	if declared != required {
		t.Fatalf("manifest Framework floor = %s, want %s from go.mod", declared, required)
	}
}

func TestTheReleaseSkillUsesTheManifestFrameworkFloor(t *testing.T) {
	root := packageRoot(t)
	manifest := readReleaseFile(t, root, "arandu.mod.toml")
	skill := readReleaseFile(t, root, ".agents/skills/llama-release/SKILL.md")
	declared := captureReleaseValue(t, manifest,
		`(?m)^framework = ">= ([0-9]+\.[0-9]+)"$`,
		"Framework floor in arandu.mod.toml")

	want := `framework = ">= ` + declared + `"`
	if !strings.Contains(skill, want) {
		t.Fatalf("release skill does not teach manifest floor %q", want)
	}
}

// releasedChangelog is CHANGELOG.md with the [Unreleased] section removed.
//
// Everything a tag shipped has to be under a version heading. The section above
// the first one is where work waits, and a release that forgets to move it is a
// published version whose own changelog calls its contents unreleased -- which
// is what a released version of a package cloned from here once did.
//
// A package that has released nothing has no version heading, and the tests
// reading this skip rather than fail: there is nothing yet to have forgotten.
func releasedChangelog(t *testing.T) string {
	t.Helper()
	body := readReleaseFile(t, packageRoot(t), "CHANGELOG.md")
	first := regexp.MustCompile(`(?m)^## \[[0-9]`).FindStringIndex(body)
	if first == nil {
		t.Skip("nothing released yet: this package has no versioned changelog entry")
	}
	return body[first[0]:]
}

// TestEveryActionIsNamedInAReleasedChangelogEntry is the gate that catches a
// tag pushed without filing what it shipped.
//
// An action is added in the same change that adds the capability behind it, so
// an action still sitting in [Unreleased] means the version that introduced it
// went out undocumented. It is the cheapest signal of that, and it needs no git
// history to read.
func TestEveryActionIsNamedInAReleasedChangelogEntry(t *testing.T) {
	policy := readReleaseFile(t, packageRoot(t), "policy.go")
	released := releasedChangelog(t)

	names := regexp.MustCompile(`(?m)^\t([A-Z][A-Za-z]*) security\.Action = `).FindAllStringSubmatch(policy, -1)
	if len(names) == 0 {
		t.Fatal("policy.go declares no actions")
	}
	for _, name := range names {
		if !strings.Contains(released, "`"+name[1]+"`") {
			t.Errorf("no released changelog entry names %s", name[1])
		}
	}
}

// TestEveryMigrationIsNamedInAReleasedChangelogEntry holds the same for schema.
//
// A migration is the one thing an operator has to run before a version serves,
// so a version that shipped one and did not say so is a version that fails at
// the first request against a column that is not there.
func TestEveryMigrationIsNamedInAReleasedChangelogEntry(t *testing.T) {
	module := readReleaseFile(t, packageRoot(t), "module.go")
	released := releasedChangelog(t)

	ids := regexp.MustCompile(`"([0-9]{8}_[0-9]{4}_[a-z_]+)"`).FindAllStringSubmatch(module, -1)
	if len(ids) == 0 {
		t.Fatal("module.go declares no migrations")
	}
	seen := map[string]bool{}
	for _, id := range ids {
		if seen[id[1]] {
			continue
		}
		seen[id[1]] = true
		if !strings.Contains(released, id[1]) {
			t.Errorf("no released changelog entry names migration %s", id[1])
		}
	}
}

// TestEveryChangelogVersionHasUpgradeNotes keeps the two files describing the
// same set of releases.
//
// They drifted once in the other direction: two packages cloned from here
// carried this repository's own release history, renamed into them by
// configure, numbered over the tags of the same name there. The history is
// inside a configure:template section now, so a clone starts without it.
func TestEveryChangelogVersionHasUpgradeNotes(t *testing.T) {
	root := packageRoot(t)
	changelog := readReleaseFile(t, root, "CHANGELOG.md")
	upgrade := readReleaseFile(t, root, "UPGRADE.md")

	inChangelog := regexp.MustCompile(`(?m)^## \[([0-9]+\.[0-9]+\.[0-9]+)\] - `).FindAllStringSubmatch(changelog, -1)
	inUpgrade := regexp.MustCompile(`(?m)^## v([0-9]+\.[0-9]+\.[0-9]+)$`).FindAllStringSubmatch(upgrade, -1)
	if len(inChangelog) == 0 && len(inUpgrade) == 0 {
		t.Skip("nothing released yet: neither file has a version heading")
	}

	versions := func(matches [][]string) map[string]bool {
		out := map[string]bool{}
		for _, m := range matches {
			out[m[1]] = true
		}
		return out
	}
	logged, upgraded := versions(inChangelog), versions(inUpgrade)
	for v := range logged {
		if !upgraded[v] {
			t.Errorf("CHANGELOG.md has %s and UPGRADE.md has no notes for it", v)
		}
	}
	for v := range upgraded {
		if !logged[v] {
			t.Errorf("UPGRADE.md has notes for %s and CHANGELOG.md has no entry for it", v)
		}
	}
}

// The section markers, spelled in halves.
//
// configure removes whole lines from the first marker to the second in every
// file it reads, this one included. A test naming them in full would be a test
// configure deleted the middle of, and that is not a thought experiment: it
// happened on the first run, and the clone failed to build with
// "undefined: start".
const (
	markerHalf = "configure:template"
	openMarker = markerHalf + "-start"
	shutMarker = markerHalf + "-end"
)

func TestCIGuardsIncompatibleAPIChanges(t *testing.T) {
	ci := readReleaseFile(t, packageRoot(t), ".github/workflows/ci.yml")
	required := []string{
		"fetch-depth: 0",
		"name: api diff against the last release",
		`modpath=$(GOWORK=off go list -m -f '{{.Path}}')`,
		`git show "${tag}:go.mod"`,
		"git worktree add",
		"golang.org/x/exp/cmd/apidiff@latest -m -w",
		"-m -incompatible",
		`git diff --quiet "$latest" -- UPGRADE.md`,
		`added=$(git diff "$latest" -- UPGRADE.md`,
	}
	for _, want := range required {
		if !strings.Contains(ci, want) {
			t.Errorf("CI does not contain the API compatibility gate %q", want)
		}
	}
}

func TestTheReleasePublishesThePreVersionedChangelogEntryOnce(t *testing.T) {
	root := packageRoot(t)
	release := readReleaseFile(t, root, ".github/workflows/release.yml")
	changelog := readReleaseFile(t, root, "CHANGELOG.md")

	// No version is named here. A clone of this repository has released
	// nothing, and a test that pinned a version of this one would be a test a
	// clone inherits and cannot satisfy -- which is how two packages came to
	// hold this repository's release notes in place.
	seen := map[string]bool{}
	for _, heading := range regexp.MustCompile(`(?m)^## \[([0-9]+\.[0-9]+\.[0-9]+)\] - `).FindAllStringSubmatch(changelog, -1) {
		if seen[heading[1]] {
			t.Errorf("CHANGELOG.md has more than one %s heading", heading[1])
		}
		seen[heading[1]] = true
	}
	required := []string{
		`tags: ["v*"]`,
		"contents: write",
		`version="${TAG#v}"`,
		`heading="## [$version] - "`,
		`gh release create "$TAG"`,
		"--verify-tag",
		`--notes-file "$notes"`,
	}
	for _, want := range required {
		if !strings.Contains(release, want) {
			t.Errorf("release workflow does not contain %q", want)
		}
	}
	if got := strings.Count(release, `gh release create "$TAG"`); got != 1 {
		t.Errorf("release creation commands = %d, want exactly one", got)
	}
	for _, forbidden := range []string{
		"Write it into the changelog",
		"git commit",
		"git push",
		"git tag",
		"gh release delete",
		"CHANGELOG.next",
	} {
		if strings.Contains(release, forbidden) {
			t.Errorf("release workflow mutates the pre-versioned changelog through %q", forbidden)
		}
	}
}

func readReleaseFile(t *testing.T, root, name string) string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(name)))
	if err != nil {
		t.Fatalf("reading %s: %v", name, err)
	}
	return string(raw)
}

func captureReleaseValue(t *testing.T, body, pattern, label string) string {
	t.Helper()
	match := regexp.MustCompile(pattern).FindStringSubmatch(body)
	if len(match) != 2 {
		t.Fatalf("%s is missing", label)
	}
	return match[1]
}
