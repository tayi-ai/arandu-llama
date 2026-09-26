package native

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	decoder "github.com/tayi-ai/arandu-llama/training/decoder"
)

func TestNativeInstallationIsDetachedAndBoundToContract(t *testing.T) {
	config := testNativeConfiguration()
	first, err := NewNativeExperiment(config)
	if err != nil {
		t.Fatal(err)
	}
	digest := first.RecipeSHA256()
	config.Recipe.Model.Repository = "private/another-student"
	config.Recipe.Distribution.Nodes[0] = "another-node"
	rule := config.Recipe.Data.TeacherRules["anchor_preserve"]
	rule.Sources["anchor"] = .5
	rule.Sources["another-teacher"] = .5
	config.Assembly.DeviceByLayer[0] = 1
	exported := ExperimentNativeRecipe(first)
	exported.Data.TeacherRules["anchor_preserve"].Sources["anchor"] = .01
	snapshot := first.Config()
	snapshot.Assembly.DeviceByLayer[0] = 8
	snapshot.Initializer.Projections[0].Name = "changed"
	snapshot.Recipe.Data.TeacherRules["anchor_preserve"].Sources["anchor"] = .02
	if first.RecipeSHA256() != digest || ExperimentNativeRecipe(first).Model.Repository != "fixture/student" || first.config.Assembly.DeviceByLayer[0] != 0 {
		t.Fatal("caller mutated admitted recipe")
	}
	secondConfig := testNativeConfiguration()
	secondConfig.Recipe.Model.Repository = "private/another-student"
	second, err := NewNativeExperiment(secondConfig)
	if err != nil {
		t.Fatal(err)
	}
	if first.RecipeSHA256() == second.RecipeSHA256() {
		t.Fatal("source identity did not bind configuration")
	}
	secondConfig = testNativeConfiguration()
	secondConfig.ManifestSHA256 = strings.Repeat("b", 64)
	second, err = NewNativeExperiment(secondConfig)
	if err != nil {
		t.Fatal(err)
	}
	if digest != second.RecipeSHA256() {
		t.Fatal("manifest-to-contract binding introduced a circular hash")
	}
}

func TestNativeInstallationRefusesMissingOrUnsupportedInputs(t *testing.T) {
	for name, change := range map[string]func(*NativeExperimentConfig){
		"source":           func(c *NativeExperimentConfig) { c.Recipe.Model.Repository = "" },
		"pin":              func(c *NativeExperimentConfig) { c.TokenizerSHA256 = "" },
		"duplicate-node":   func(c *NativeExperimentConfig) { c.Recipe.Distribution.Nodes[1] = c.Recipe.Distribution.Nodes[0] },
		"numeric-mode":     func(c *NativeExperimentConfig) { c.Recipe.Optimization.LearningRate = .1 },
		"false-ready":      func(c *NativeExperimentConfig) { c.Recipe.Backend.ReadyGPU = true },
		"missing-teachers": func(c *NativeExperimentConfig) { c.Recipe.Data.TeacherRules = nil },
		"mixture-mass": func(c *NativeExperimentConfig) {
			r := c.Recipe.Data.TeacherRules["anchor_preserve"]
			r.Sources["anchor"] = 2
		},
		"placement":   func(c *NativeExperimentConfig) { c.Assembly.DeviceByLayer = nil },
		"initializer": func(c *NativeExperimentConfig) { c.Initializer.Projections = nil },
		"rotary":      func(c *NativeExperimentConfig) { c.Rotary.ExpectedSHA256 = "" },
	} {
		t.Run(name, func(t *testing.T) {
			c := testNativeConfiguration()
			change(&c)
			if _, err := NewNativeExperiment(c); err == nil {
				t.Fatal("invalid installation accepted")
			}
		})
	}
}

func TestNativeInstallationDocumentRefusesTamperingAndUnknownFields(t *testing.T) {
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, "private-installation.json")
	body, _ := json.Marshal(testNativeConfiguration())
	if err = os.WriteFile(path, body, 0600); err != nil {
		t.Fatal(err)
	}
	e, err := LoadNativeExperiment(path, experimentHash(body))
	if err != nil || e.sourcePath != path {
		t.Fatalf("load: %v", err)
	}
	if actualPath, actualDigest := e.SourceIdentity(); actualPath != path || actualDigest != experimentHash(body) {
		t.Fatal("explicit source document identity changed")
	}
	if err = os.WriteFile(path, append(body, ' '), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err = LoadNativeExperiment(path, experimentHash(body)); err == nil {
		t.Fatal("changed installation admitted")
	}
	body = append([]byte(`{"command":"sh",`), body[1:]...)
	if _, err = DecodeNativeExperiment(strings.NewReader(string(body))); err == nil {
		t.Fatal("unknown executable field accepted")
	}
}

func TestNativeRotaryMustMatchAdmittedDecoderGeometry(t *testing.T) {
	e := testNativeExperiment()
	geometry := decoder.TextGeometry{Dimension: 128, RoPE: decoder.RotaryGeometry{Theta: e.config.Rotary.Theta, Partial: .5}}
	if err := validateNativeRotaryGeometry(e, geometry); err != nil {
		t.Fatal(err)
	}
	changed := geometry
	changed.RoPE.Theta *= 2
	if err := validateNativeRotaryGeometry(e, changed); err == nil {
		t.Fatal("different frequency basis admitted")
	}
	changed = geometry
	changed.RoPE.Partial = 1
	if err := validateNativeRotaryGeometry(e, changed); err == nil {
		t.Fatal("different rotary dimension admitted")
	}
}
