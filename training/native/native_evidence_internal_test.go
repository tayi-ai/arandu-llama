package native

import (
	"os"
	"testing"
)

// nativeDiagnosticEvidence is supplied with its own immutable document hash.
// Historical scientific identities stay in the private evidence store.
type nativeDiagnosticEvidence struct {
	TrainRelease      string `json:"train_release"`
	FinalAdapter      string `json:"final_adapter"`
	FinalArtifact     string `json:"final_artifact"`
	SourceRevision    string `json:"source_revision"`
	HeldoutSHA256     string `json:"heldout_sha256"`
	DiagnosticRelease string `json:"diagnostic_release"`
	CapturedInput     string `json:"captured_input"`
	PostAttention     string `json:"post_attention"`
}

func privateNativeEvidence(t *testing.T) (*NativeExperiment, nativeDiagnosticEvidence) {
	t.Helper()
	if os.Getenv("TAYI_NATIVE_CONFIG") == "" || os.Getenv("TAYI_NATIVE_CONFIG_SHA256") == "" || os.Getenv("TAYI_NATIVE_EVIDENCE_CONFIG") == "" || os.Getenv("TAYI_NATIVE_EVIDENCE_CONFIG_SHA256") == "" {
		t.Skip("requires explicit hash-bound diagnostic configuration")
	}
	e, err := LoadNativeExperiment(os.Getenv("TAYI_NATIVE_CONFIG"), os.Getenv("TAYI_NATIVE_CONFIG_SHA256"))
	if err != nil {
		t.Fatal(err)
	}
	body, err := readTrainingDocument(os.Getenv("TAYI_NATIVE_EVIDENCE_CONFIG"), os.Getenv("TAYI_NATIVE_EVIDENCE_CONFIG_SHA256"))
	if err != nil {
		t.Fatal(err)
	}
	var evidence nativeDiagnosticEvidence
	if err = experimentDecode(body, &evidence, true); err != nil {
		t.Fatal(err)
	}
	for _, hash := range []string{evidence.TrainRelease, evidence.FinalAdapter, evidence.FinalArtifact, evidence.HeldoutSHA256, evidence.DiagnosticRelease, evidence.CapturedInput, evidence.PostAttention} {
		if !experimentSHA256(hash) {
			t.Fatal("private diagnostic identity missing")
		}
	}
	if evidence.SourceRevision == "" {
		t.Fatal("private diagnostic source revision missing")
	}
	return e, evidence
}
