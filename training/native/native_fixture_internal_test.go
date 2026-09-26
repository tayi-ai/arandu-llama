package native

import (
	"bytes"
	"encoding/json"

	fixtures "github.com/tayi-ai/arandu-llama/training/native/testfixtures"
)

func testNativeConfiguration() NativeExperimentConfig {
	var config NativeExperimentConfig
	if err := json.Unmarshal(fixtures.NativeDocument(), &config); err != nil {
		panic(err)
	}
	return config
}
func testNativeExperiment() *NativeExperiment {
	e, err := DecodeNativeExperiment(bytes.NewReader(fixtures.NativeDocument()))
	if err != nil {
		panic(err)
	}
	return e
}
