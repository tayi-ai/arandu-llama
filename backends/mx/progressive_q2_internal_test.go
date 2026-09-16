package mx

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const progressiveQ2Manifest = `b241668d66096cf67160bacd71d91d62182e5a1deb789e3077baeafb98a227a3  Tayi-Flash-Q2-Q2_K-00001-of-00012.gguf
d28b4f020631972f4adb92d27f1aefdbf31775575b683c39cd123b9370f2dc09  Tayi-Flash-Q2-Q2_K-00002-of-00012.gguf
97856824817a9c829d5a756340899e171c78d88d5c37233c2096d11154377ed9  Tayi-Flash-Q2-Q2_K-00003-of-00012.gguf
89bdb4e1858d47a362a816850178c0a9c31dc02643593944454991f132b9a943  Tayi-Flash-Q2-Q2_K-00004-of-00012.gguf
8f0753a018afa5c624b5a66eb91b418e9aa0c7972ee55cea39b841666e4955c8  Tayi-Flash-Q2-Q2_K-00005-of-00012.gguf
8e7be8baca55c5e0604f523f32d8c4161b11a6737ecab136503f1327afc984dd  Tayi-Flash-Q2-Q2_K-00006-of-00012.gguf
3ffe3821af5f5227d282d0c0f963d55ff81a4db04c6e39d99eee40c08630fddc  Tayi-Flash-Q2-Q2_K-00007-of-00012.gguf
238ab3cfccedb5ab618ab436c6acd67f26d0dcdb7087c9e12ec740db41c8d34c  Tayi-Flash-Q2-Q2_K-00008-of-00012.gguf
77c39d6c7eeb5ad9c1634a993854bfe053a08c04a439f3c638700423ac82e351  Tayi-Flash-Q2-Q2_K-00009-of-00012.gguf
4da5d34508297d9137fcef6ba45034d10d27c75f52307e05e1557b790956d404  Tayi-Flash-Q2-Q2_K-00010-of-00012.gguf
76a1f98364a585f299aa222a69f0bb1082021553a0e338936f5d972e9edb5a6a  Tayi-Flash-Q2-Q2_K-00011-of-00012.gguf
0991aa96e406357fdb712172bd05badc7164100c48bf06d2fa4b1def05ad69d4  Tayi-Flash-Q2-Q2_K-00012-of-00012.gguf
`

func TestProgressiveQ2AdmissionIsDistinctFromExternalControl(t *testing.T) {
	progressive, err := AdmittedMXManifestFor(MXModelTayiQ2Progressive)
	if err != nil {
		t.Fatal(err)
	}
	external, err := AdmittedMXManifestFor(MXModelQ2K)
	if err != nil {
		t.Fatal(err)
	}
	if progressive.ModelDigest != "b24af56efc29d742065e1a76c3b99e6af5e74df822c2bb8b7fac7b64b5057f50" || progressive.ModelShards != 12 {
		t.Fatalf("progressive identity = %+v", progressive)
	}
	if progressive.ModelDigest == external.ModelDigest || progressive.ModelFirstShard == external.ModelFirstShard {
		t.Fatal("progressive Q2 collided with external Q2 control identity")
	}
	if got, err := MXModelRecipeForDigest(progressive.ModelDigest); err != nil || got != MXModelTayiQ2Progressive {
		t.Fatalf("progressive recipe recovery = %q, %v", got, err)
	}
	if got, err := MXModelRecipeForDigest(external.ModelDigest); err != nil || got != MXModelQ2K {
		t.Fatalf("external control recipe changed = %q, %v", got, err)
	}
	if got := AdmittedMXManifest(); got.ModelRecipe != MXModelMXFP4 {
		t.Fatalf("default recipe changed to %q", got.ModelRecipe)
	}
}

func TestProgressiveQ2CacheAdmissionUsesMeasuredManifest(t *testing.T) {
	model, err := admittedMXModel(MXModelTayiQ2Progressive)
	if err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "SHA256SUMS"), []byte(progressiveQ2Manifest), 0o600); err != nil {
		t.Fatal(err)
	}
	first := filepath.Join(root, model.FirstShard)
	if err := os.WriteFile(first, []byte("GGUF"), 0o600); err != nil {
		t.Fatal(err)
	}
	backend := &MXBackend{modelRoot: root, model: model}
	got, err := backend.admittedModel()
	if err != nil || got != first {
		t.Fatalf("progressive cache admission = %q, %v", got, err)
	}
	if err := os.WriteFile(filepath.Join(root, "SHA256SUMS"), []byte(strings.Replace(progressiveQ2Manifest, "b241668d", "0241668d", 1)), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := backend.admittedModel(); err == nil || !strings.Contains(err.Error(), "manifest") {
		t.Fatalf("mutated progressive manifest refusal = %v", err)
	}
}
