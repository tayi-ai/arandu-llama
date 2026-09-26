package native

import (
	"testing"
)

func TestNativeDeadlineBudgetsStayFiniteAndModeBounded(t *testing.T) {
	ids := ExperimentNativeNodeIDs(testNativeExperiment())
	for _, tc := range []struct {
		name          string
		canary, train int
		accepted      bool
	}{
		{"historical bounded canary", 900, 5400, true},
		{"complete qualification budget", 1800, 5400, true},
		{"zero canary", 0, 5400, false},
		{"negative canary", -1, 5400, false},
		{"canary exceeds ceiling", 1801, 5400, false},
		{"training ceiling preserved", 1800, 5401, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a := experimentAdmission{SchemaVersion: 2, ContractVersion: ExperimentNativeContractVersion,
				ModelManifestSHA256: testNativeConfiguration().Recipe.Model.ManifestSHA256, TopologyIDs: ids,
				NNodes: len(ids), ProcessWorldSize: len(ids), GPUsPerNode: 2,
				GlobalBatchSize: 21, OptimizerSteps: 18, Examples: 378, Epochs: 1,
				VRAMFraction: "0.80", DeadlinesSeconds: map[string]int{"canary": tc.canary, "train": tc.train},
				Jobs: map[string]experimentJobIdentity{"canary": {ID: "native-budget-canary", Generation: 1}, "train": {ID: "native-budget-train", Generation: 2}}}
			err := experimentValidateNativeIdentity(testNativeExperiment(), a, Job{ID: "native-budget-canary", Generation: 1}, "canary", ids)
			if (err == nil) != tc.accepted {
				t.Fatalf("canary=%d train=%d accepted=%t, error=%v", tc.canary, tc.train, tc.accepted, err)
			}
		})
	}
}
