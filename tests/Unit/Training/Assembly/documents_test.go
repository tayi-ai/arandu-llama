package assembly_test

import (
	"os"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/ornith"
)

func TestFrozenAssemblyDocumentsOptIn(t *testing.T) {
	paths := []string{os.Getenv("TAYI_ORNITH_INDEX"), os.Getenv("TAYI_ORNITH_CONFIG"), os.Getenv("TAYI_ORNITH_REFERENCE")}
	if paths[0] == "" && paths[1] == "" && paths[2] == "" {
		t.Skip("set TAYI_ORNITH_INDEX, TAYI_ORNITH_CONFIG and TAYI_ORNITH_REFERENCE for frozen-document integration")
	}
	data := make([][]byte, 3)
	for i, path := range paths {
		if path == "" {
			t.Fatal("all three frozen-document paths are required")
		}
		var err error
		data[i], err = os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
	}
	p, err := ornith.PlanTextAssembly(data[0], data[1], data[2], ornith.AssemblyIdentity{
		IndexSHA256:          "d5c7fee99574e9a05f901282aee04fc4fc3dccf094a659df48c6b8e9f39109c3",
		ConfigSHA256:         "1f1b3751c38f16a63340df90a55e870bef0f0b2968d833825a605b7cf930a313",
		ReferenceSHA256:      "69825f715b8f422e2e155be4bd2ed309e14d77ac9d45a65405a3b5886ca228dc",
		InitialAdapterSHA256: initialDigest,
	}, admittedLimits())
	if err != nil {
		t.Fatal(err)
	}
	if p.Summary().PersistentBytes != admittedLimits().PersistentBytes {
		t.Fatal("frozen document geometry differs")
	}
	t.Logf("frozen_document_plan=%+v", p.Summary())
}
