//go:build libtorch && cgo

package decoder

import (
	"bytes"
	"os"
	"testing"
)

func TestNativeConstrainedFactoryPreservesVersionOneBytes(t *testing.T) {
	c, _, _, _ := sessionFixture(t)
	c.Recipe.Initial.Path = "/fixture/initial.safetensors"
	body, err := c.document()
	if err != nil {
		t.Fatal(err)
	}
	legacy, err := os.ReadFile("testdata/constrained-factory-v1.json")
	if err != nil {
		t.Fatal(err)
	}
	// Captured with the complete synthetic fixture before adding later fields.
	const digest = "e885b3ca6d788230947db0f80fe92c7d0135a6b5aa931fbb87bbaad53218cf45"
	if len(legacy) != 16210 || assemblyHash(legacy) != digest || !bytes.Equal(body, legacy) {
		t.Fatal("version-one canonical factory bytes changed")
	}
	actual, err := c.Digest()
	if err != nil || actual != digest {
		t.Fatalf("version-one factory digest changed: %v", err)
	}
}
