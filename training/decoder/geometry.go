package decoder

import (
	"encoding/json"
	"fmt"
	"math"
)

// TextGeometry declares the supported gated-attention/recurrent text structure.
// Source names identify a storage format; model lineage is always caller-owned.
type TextGeometry struct {
	Hidden        int64          `json:"hidden_size"`
	Vocab         int64          `json:"vocab_size"`
	MLP           int64          `json:"intermediate_size"`
	Layers        int            `json:"num_hidden_layers"`
	Heads         int64          `json:"num_attention_heads"`
	KVHeads       int64          `json:"num_key_value_heads"`
	Dimension     int64          `json:"head_dim"`
	LayerTypes    []string       `json:"layer_types"`
	KHeads        int64          `json:"linear_num_key_heads"`
	VHeads        int64          `json:"linear_num_value_heads"`
	KDimension    int64          `json:"linear_key_head_dim"`
	VDimension    int64          `json:"linear_value_head_dim"`
	Conv          int64          `json:"linear_conv_kernel_dim"`
	Epsilon       float64        `json:"rms_norm_eps"`
	Activation    string         `json:"hidden_act"`
	AttentionBias bool           `json:"attention_bias"`
	Dropout       float64        `json:"attention_dropout"`
	Gate          bool           `json:"attn_output_gate"`
	Tie           bool           `json:"tie_word_embeddings"`
	RoPE          RotaryGeometry `json:"rope_parameters"`
}

// RotaryGeometry describes supported text-only rotary coordinates.
type RotaryGeometry struct {
	Theta       float64 `json:"rope_theta"`
	Type        string  `json:"rope_type"`
	Partial     float64 `json:"partial_rotary_factor"`
	Interleaved bool    `json:"mrope_interleaved"`
	Sections    []int   `json:"mrope_section"`
}

// Validate rejects unsupported operations instead of accepting a model by name.
func (c TextGeometry) Validate() error {
	for _, dimension := range []int64{c.Hidden, c.Vocab, c.MLP, c.Heads, c.KVHeads, c.Dimension, c.KHeads, c.VHeads, c.KDimension, c.VDimension, c.Conv} {
		if dimension < 1 || dimension > 1<<20 {
			return fmt.Errorf("%w: geometry dimension outside bounds", ErrAssembly)
		}
	}
	if c.Layers < 1 || c.Layers > 4096 || len(c.LayerTypes) != c.Layers || c.Heads%c.KVHeads != 0 || c.VHeads%c.KHeads != 0 || c.Epsilon <= 0 || math.IsNaN(c.Epsilon) || math.IsInf(c.Epsilon, 0) || c.Activation != "silu" || c.AttentionBias || c.Dropout != 0 || !c.Gate || c.Tie || c.RoPE.Type != "default" || c.RoPE.Theta <= 0 || math.IsNaN(c.RoPE.Theta) || math.IsInf(c.RoPE.Theta, 0) || c.RoPE.Theta > math.MaxFloat32 || c.RoPE.Partial <= 0 || c.RoPE.Partial > 1 || math.IsNaN(c.RoPE.Partial) {
		return fmt.Errorf("%w: unsupported text geometry/config", ErrAssembly)
	}
	if c.Heads*c.Dimension > 1<<24 || c.KVHeads*c.Dimension > 1<<24 || 2*c.KHeads*c.KDimension+c.VHeads*c.VDimension > 1<<24 {
		return fmt.Errorf("%w: projected channels exceed bounds", ErrAssembly)
	}
	rotary := float64(c.Dimension) * c.RoPE.Partial
	if rotary != math.Trunc(rotary) || int64(rotary)%2 != 0 || rotary < 2 {
		return fmt.Errorf("%w: rotary dimension must be positive and even", ErrAssembly)
	}
	attention := 0
	for _, kind := range c.LayerTypes {
		switch kind {
		case "full_attention":
			attention++
		case "linear_attention":
		default:
			return fmt.Errorf("%w: unsupported layer operation", ErrAssembly)
		}
	}
	if attention == 0 {
		return fmt.Errorf("%w: at least one trainable attention layer required", ErrAssembly)
	}
	return nil
}

func validateAssemblyConfig(data []byte) (TextGeometry, error) {
	var document struct {
		Tie  bool         `json:"tie_word_embeddings"`
		Text TextGeometry `json:"text_config"`
	}
	if err := json.Unmarshal(data, &document); err != nil {
		return TextGeometry{}, fmt.Errorf("%w: config: %v", ErrAssembly, err)
	}
	if document.Tie {
		return TextGeometry{}, fmt.Errorf("%w: tied embedding unsupported", ErrAssembly)
	}
	return document.Text, document.Text.Validate()
}
