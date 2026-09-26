// Package fixtures supplies synthetic configuration without production identities.
package fixtures

import (
	"encoding/json"
	"fmt"
)

// NativeDocument returns a fresh document for the currently supported numerical
// kernel. Its fake artifact identities never authorize a real training job.
func NativeDocument() []byte {
	digest := "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	layers := make([]int, 32)
	projections := []map[string]any{}
	for i := range layers {
		layers[i] = i / 16
		if (i+1)%4 == 0 {
			for _, target := range []string{"q_proj", "v_proj"} {
				projections = append(projections, map[string]any{"Name": fmt.Sprintf("base_model.model.model.layers.%d.self_attn.%s", i, target), "Input": 4096, "Output": 4096, "Rank": 4})
			}
		}
	}
	document := map[string]any{
		"assembly": map[string]any{
			"PersistentBytes": []int64{11042374656, 11042382848}, "DeviceByLayer": layers, "EmbeddingDevice": 0, "OutputDevice": 1, "AdapterRank": 4, "AdapterAlpha": 8,
			"HeaderLimits":   map[string]any{"MaxHeaderBytes": 16 << 20, "MaxTensors": 65536, "MaxDimensions": 32, "MaxMetadataEntries": 65536, "MaxChunkBytes": 4 << 20},
			"HashChunkBytes": 4 << 20, "TensorCopyBytes": int64(6102712320), "MaxInputElements": 4624384, "MaxScoreElements": 20394256, "MaxWorkingElements": 891134976,
			"Sequence": map[string]any{"ChunkTokens": 8, "MaxTokens": 1129, "MaxOwnedElements": 99215936},
		},
		"initializer":    map[string]any{"Seed": 83, "PreludeBlocks": 24, "PreludeWidth": 4096, "PreludeLow": -.015625, "PreludeHigh": .015625, "ExpectedSHA256": digest, "Projections": projections},
		"rotary":         map[string]any{"Theta": 10000000, "Dimension": 64, "MaxTokens": 4096, "HalfPrecision": true, "ExpectedSHA256": digest},
		"schema_version": 1, "model_recipe": "fixture-decoder", "bundle_name": "native-fixture-v1", "manifest_sha256": digest, "base_path": "/cache/tayi/checkpoints/fixture",
		"tokenizer_sha256": digest, "rotary_sha256": digest, "index_sha256": digest, "config_sha256": digest, "reference_sha256": digest,
		"recipe": map[string]any{
			"version": "native-go-v2",
			"model":   map[string]any{"repository": "fixture/student", "revision": "fixture-revision", "manifest_sha256": digest, "frozen": true, "base_precision": "float16", "attention_precision": "float32"},
			"data": map[string]any{"teacher_rules": map[string]any{
				"anchor_preserve":          map[string]any{"sources": map[string]float64{"anchor": 1}, "weight": .70, "gate": "decoder_anchor_1.0"},
				"teacher_a_teacher_b_gain": map[string]any{"sources": map[string]float64{"teacher-a": .75, "teacher-b": .25}, "weight": .45, "gate": "teacher_a_0.75_teacher_b_0.25"},
				"teacher_a_only_gain":      map[string]any{"sources": map[string]float64{"teacher-a": 1}, "weight": .35, "gate": "teacher_a_1.0"},
				"teacher_b_only_gain":      map[string]any{"sources": map[string]float64{"teacher-b": 1}, "weight": .25, "gate": "teacher_b_1.0"},
				"persistent_error_all":     map[string]any{"sources": map[string]float64{}, "weight": 0, "gate": "hard_label_only"},
			}, "id": "fixture-development", "sha256": digest, "examples": 378, "demonstrations": 4, "epochs": 1, "protocol": "raw_four_shot_answer_space", "candidate_vocabulary_ids_abcd": []int{357, 417, 351, 414}},
			"lora":         map[string]any{"rank": 4, "alpha": 8, "targets": []string{"q_proj", "v_proj"}, "tensors": 32, "parameters": 557056, "bias": "none", "precision": "float32", "fresh": true, "expected_initial_digest": digest},
			"optimization": map[string]any{"optimizer": "AdamW", "seed": 83, "learning_rate": 2e-6, "betas": []float64{0.9, 0.999}, "epsilon": 1e-8, "weight_decay": 0, "gradient_clip": 1, "clip_epsilon": 1e-12, "backward_loss_scale": 1.0 / 1024, "updates": 18, "save_steps": []int{9, 18}, "candidate_step": 18, "gradient_reduction": "sum_20_rank_local_sums_divide_21_examples", "update_order": "scale_each_loss_backward_sum_local_fp32_sum_rank_order_divide_21_unscale_global_norm_clip_adamw"},
			"distribution": map[string]any{"nodes": []string{"one", "two", "three", "four", "five", "six", "seven", "eight", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "twentyone"}, "processes": 20, "gpus": 40, "gpus_per_process": 2, "local_batch": 2, "local_batch_mode": "sequential_extra_example_on_rank_update_minus_one", "global_batch": 21, "layers_per_gpu": []int{16, 16}},
			"guards":       map[string]any{"vram_fraction": 0.80, "host_reserve_bytes": 6 << 30, "preload_ram_qualification_required": true, "canary_seconds": 1800, "train_seconds": 5400, "calibration_rows": 8, "score_tolerance": 0.03, "base_file_count": 18},
			"backend":      map[string]any{"status": "unqualified", "loss_precision": "go_float64", "unqualified_capabilities": []string{"forward", "backward", "collective"}},
		},
	}
	body, err := json.Marshal(document)
	if err != nil {
		panic(err)
	}
	return body
}
