package mx

import (
	"encoding/hex"
	"errors"
	"net/url"
	"path/filepath"
	"strings"
)

// MXCatalog holds an immutable, explicitly admitted installation catalog.
// It does not discover models, read environment variables or choose defaults.
type MXCatalog struct {
	models  map[MXModelRecipe]MXModelIdentity
	digests map[string]MXModelRecipe
}

// NewMXCatalog validates and copies every identity. Duplicate recipes and
// ambiguous manifest digests are rejected before any backend can be selected.
func NewMXCatalog(entries []MXModelIdentity) (*MXCatalog, error) {
	if len(entries) == 0 {
		return nil, errors.New("llama: MX catalog requires explicit model identities")
	}
	c := &MXCatalog{models: make(map[MXModelRecipe]MXModelIdentity, len(entries)), digests: make(map[string]MXModelRecipe, len(entries))}
	for _, model := range entries {
		if err := validateModel(model); err != nil {
			return nil, err
		}
		if _, exists := c.models[model.Recipe]; exists {
			return nil, errors.New("llama: MX catalog has duplicate model recipes")
		}
		if _, exists := c.digests[model.Manifest]; exists {
			return nil, errors.New("llama: MX catalog has ambiguous model manifests")
		}
		c.models[model.Recipe], c.digests[model.Manifest] = model, model.Recipe
	}
	return c, nil
}

// ModelFor returns a copy of one explicit identity. Zero or unknown selectors
// are rejected, including when the catalog itself is nil or uninitialized.
func (c *MXCatalog) ModelFor(recipe MXModelRecipe) (MXModelIdentity, error) {
	if c == nil || recipe == "" {
		return MXModelIdentity{}, errors.New("llama: MX model recipe is not admitted")
	}
	model, ok := c.models[recipe]
	if !ok {
		return MXModelIdentity{}, errors.New("llama: MX model recipe is not admitted")
	}
	return model, nil
}

// ManifestFor combines the selected installation model identity with the
// qualified runtime. Returned maps are owned by the caller.
func (c *MXCatalog) ManifestFor(recipe MXModelRecipe) (MXManifest, error) {
	model, err := c.ModelFor(recipe)
	if err != nil {
		return MXManifest{}, err
	}
	return runtimeManifest(model), nil
}

// RecipeForDigest resolves a persisted model manifest only inside this catalog.
// An absent or malformed digest cannot select a different model on retry.
func (c *MXCatalog) RecipeForDigest(digest string) (MXModelRecipe, error) {
	if c == nil || !hexIdentity(digest, 64) {
		return "", errors.New("llama: MX model digest is not admitted")
	}
	recipe, ok := c.digests[digest]
	if !ok {
		return "", errors.New("llama: MX model digest is not admitted")
	}
	return recipe, nil
}

func validateModel(model MXModelIdentity) error {
	if !identifier(string(model.Recipe), 128) {
		return errors.New("llama: MX model recipe is malformed")
	}
	if !repositoryIdentity(model.Repository) {
		return errors.New("llama: MX model repository is malformed")
	}
	if !hexIdentity(model.Revision, 40) && !hexIdentity(model.Revision, 64) {
		return errors.New("llama: MX model revision is malformed")
	}
	if !hexIdentity(model.Manifest, 64) {
		return errors.New("llama: MX model manifest digest is malformed")
	}
	if !identifier(model.Quantisation, 64) {
		return errors.New("llama: MX model quantisation is malformed")
	}
	if !identifier(model.FirstShard, 255) || filepath.Base(model.FirstShard) != model.FirstShard || !strings.HasSuffix(model.FirstShard, ".gguf") {
		return errors.New("llama: MX model first shard is malformed")
	}
	if model.Shards < 1 || model.Shards > 99999 {
		return errors.New("llama: MX model shard count is invalid")
	}
	return nil
}

func identifier(value string, limit int) bool {
	if len(value) == 0 || len(value) > limit || value[0] == '.' || value[0] == '-' {
		return false
	}
	for _, ch := range value {
		if ch >= 'a' && ch <= 'z' || ch >= 'A' && ch <= 'Z' || ch >= '0' && ch <= '9' || ch == '_' || ch == '-' || ch == '.' {
			continue
		}
		return false
	}
	return true
}

func hexIdentity(value string, size int) bool {
	if len(value) != size || value != strings.ToLower(value) || strings.Trim(value, "0") == "" {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func repositoryIdentity(value string) bool {
	if strings.Contains(value, "://") {
		u, err := url.Parse(value)
		if err != nil || u.Scheme != "https" || u.Hostname() == "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" || u.RawPath != "" {
			return false
		}
		value = strings.TrimPrefix(u.Path, "/")
	}
	parts := strings.Split(value, "/")
	if len(parts) < 2 || len(parts) > 8 {
		return false
	}
	for _, part := range parts {
		if !identifier(part, 255) {
			return false
		}
	}
	return true
}
