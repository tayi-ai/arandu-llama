package unit_test

import (
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	llama "github.com/tayi-ai/arandu-llama"
)

// What a capture has to be able to say about itself before a card is involved.
//
// CaptureVersion embeds the commit llama.cpp is pinned at, because the pin
// moving is the one event that silently changes every number a capture holds:
// the tensor keeps its name, the shape keeps its width, and a cache keyed
// without the commit hands a trainer a target from a different model. The
// gitlink is the pin; the constant has to follow it.

func TestCaptureVersionNamesThePinnedEngine(t *testing.T) {
	t.Parallel()

	root := packageRoot(t)
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git is not on PATH: the gitlink cannot be read")
	}
	command := exec.Command("git", "ls-tree", "HEAD", "llama.cpp")
	command.Dir = root
	output, err := command.Output()
	if err != nil {
		t.Skipf("not a git checkout at %s: %v", filepath.Base(root), err)
	}
	// "160000 commit <sha>\tllama.cpp"
	fields := strings.Fields(string(output))
	if len(fields) < 3 || fields[1] != "commit" {
		t.Skipf("llama.cpp is not a gitlink at HEAD: %q", strings.TrimSpace(string(output)))
	}
	pinned := fields[2]
	if len(pinned) < 8 {
		t.Fatalf("gitlink %q is not a commit id", pinned)
	}
	want := "llama.cpp." + pinned[:8]
	if !strings.HasSuffix(llama.CaptureVersion, want) {
		t.Fatalf("CaptureVersion %q does not end with %q: the submodule moved and the constant did not, so every cached capture is labelled as if it came from this engine",
			llama.CaptureVersion, want)
	}
}
