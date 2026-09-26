package assembly_test

import (
	"encoding/json"
	"github.com/tayi-ai/arandu-llama/training/decoder"
	"os"
	"testing"
)

func TestAdmittedAssemblyDocumentsOptIn(t *testing.T) {
	configuration := os.Getenv("TRAINING_ASSEMBLY_TEST_CONFIG")
	if configuration == "" {
		t.Skip("set TRAINING_ASSEMBLY_TEST_CONFIG for admitted-document integration")
	}
	var spec struct {
		Index, Config, Reference string
		Identity                 decoder.AssemblyIdentity
		Limits                   decoder.AssemblyLimits
	}
	data, err := os.ReadFile(configuration)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &spec); err != nil {
		t.Fatal(err)
	}
	documents := make([][]byte, 3)
	for i, p := range []string{spec.Index, spec.Config, spec.Reference} {
		if p == "" {
			t.Fatal("explicit document paths required")
		}
		documents[i], err = os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
	}
	plan, err := decoder.PlanTextAssembly(documents[0], documents[1], documents[2], spec.Identity, spec.Limits)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Summary().BaseTensors == 0 {
		t.Fatal("empty assembly")
	}
}
