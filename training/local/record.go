package local

type example struct {
	ID           string  `json:"id"`
	InputIDs     []int64 `json:"input_ids"`
	Labels       []int64 `json:"labels"`
	PromptTokens int     `json:"prompt_tokens"`
}

type stepManifest struct {
	RecipeSHA256      string  `json:"recipe_sha256"`
	Step              uint64  `json:"step"`
	ExampleID         string  `json:"example_id"`
	SupervisedTokens  int     `json:"supervised_tokens"`
	LossBefore        float64 `json:"loss_before"`
	UpdatedAdapterSHA string  `json:"updated_adapter_sha256"`
	AdapterFileSHA    string  `json:"adapter_file_sha256"`
	OptimizerFileSHA  string  `json:"optimizer_file_sha256"`
	LogitsAfterSHA    string  `json:"logits_after_sha256"`
	BaseRevision      string  `json:"base_revision"`
}
