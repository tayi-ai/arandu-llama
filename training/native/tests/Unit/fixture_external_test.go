package native_test

import (
	"bytes"
	"encoding/json"

	services "github.com/tayi-ai/arandu-llama/training/native"
	fixtures "github.com/tayi-ai/arandu-llama/training/native/testfixtures"
)

func testNativeConfiguration() services.NativeExperimentConfig {
	var config services.NativeExperimentConfig
	if err := json.Unmarshal(fixtures.NativeDocument(), &config); err != nil {
		panic(err)
	}
	return config
}
func testNativeExperiment() *services.NativeExperiment {
	e, err := services.DecodeNativeExperiment(bytes.NewReader(fixtures.NativeDocument()))
	if err != nil {
		panic(err)
	}
	return e
}
