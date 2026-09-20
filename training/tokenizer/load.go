package tokenizer

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
	"unicode/utf8"
)

type byteLevel struct {
	Type           string `json:"type"`
	AddPrefixSpace *bool  `json:"add_prefix_space"`
	TrimOffsets    *bool  `json:"trim_offsets"`
	UseRegex       *bool  `json:"use_regex"`
}
type addedToken struct {
	ID         *int64 `json:"id"`
	Content    string `json:"content"`
	SingleWord *bool  `json:"single_word"`
	LStrip     *bool  `json:"lstrip"`
	RStrip     *bool  `json:"rstrip"`
	Normalized *bool  `json:"normalized"`
	Special    *bool  `json:"special"`
}
type configuration struct {
	Version    string          `json:"version"`
	Truncation json.RawMessage `json:"truncation"`
	Padding    json.RawMessage `json:"padding"`
	Added      []addedToken    `json:"added_tokens"`
	Normalizer struct {
		Type string `json:"type"`
	} `json:"normalizer"`
	PreTokenizer struct {
		Type          string            `json:"type"`
		PreTokenizers []json.RawMessage `json:"pretokenizers"`
	} `json:"pre_tokenizer"`
	PostProcessor byteLevel `json:"post_processor"`
	Decoder       byteLevel `json:"decoder"`
	Model         struct {
		Type        string             `json:"type"`
		Dropout     json.RawMessage    `json:"dropout"`
		Unknown     json.RawMessage    `json:"unk_token"`
		Prefix      json.RawMessage    `json:"continuing_subword_prefix"`
		Suffix      json.RawMessage    `json:"end_of_word_suffix"`
		FuseUnknown *bool              `json:"fuse_unk"`
		Fallback    *bool              `json:"byte_fallback"`
		Ignore      *bool              `json:"ignore_merges"`
		Vocabulary  map[string]tokenID `json:"vocab"`
		Merges      []string           `json:"merges"`
	} `json:"model"`
}

// An explicit integer decoder rejects null IDs, which encoding/json otherwise
// silently converts to zero when the destination is an integer map value.
type tokenID int64

func (id *tokenID) UnmarshalJSON(body []byte) error {
	value, err := strconv.ParseInt(string(body), 10, 64)
	if err != nil {
		return err
	}
	*id = tokenID(value)
	return nil
}

// Load reads at most MaxJSONBytes+1 bytes and verifies the required lowercase
// SHA256 before parsing. It accepts only NFC, the exact seven-branch split,
// ByteLevel without an added prefix/regex, and deterministic BPE string pairs.
// All 256 byte symbols must exist. Added tokens must be literal and unnormalized.
// It neither opens files nor downloads data; a blocked Reader cannot be cancelled.
func Load(ctx context.Context, source io.Reader, expectedSHA256 string, limits Limits) (*Tokenizer, error) {
	if ctx == nil || source == nil {
		return nil, errors.New("tokenizer: context and source required")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if limits.MaxJSONBytes <= 0 || limits.MaxJSONBytes == math.MaxInt64 || limits.MaxInputBytes <= 0 ||
		limits.MaxInputBytes == math.MaxInt64 || limits.MaxPieceBytes <= 0 || limits.MaxPieces <= 0 ||
		limits.MaxTokens <= 0 || limits.MaxVocabulary < 256 || limits.MaxMerges <= 0 ||
		limits.MaxAddedTokens <= 0 || limits.MaxAddedTokenBytes <= 0 {
		return nil, ErrBudget
	}
	want, err := hex.DecodeString(expectedSHA256)
	if err != nil || len(want) != sha256.Size || expectedSHA256 != strings.ToLower(expectedSHA256) {
		return nil, ErrIdentity
	}
	body, err := readBounded(ctx, source, limits.MaxJSONBytes)
	if err != nil {
		return nil, err
	}
	digest := sha256.Sum256(body)
	if !bytes.Equal(digest[:], want) {
		return nil, ErrIdentity
	}
	if err := uniqueKeys(ctx, body); err != nil {
		return nil, err
	}
	var config configuration
	if err := decode(body, &config); err != nil {
		return nil, fmt.Errorf("%w: invalid configuration: %v", ErrSchema, err)
	}
	if err := validateSchema(config); err != nil {
		return nil, err
	}
	if len(config.Model.Vocabulary) > limits.MaxVocabulary || len(config.Model.Merges) > limits.MaxMerges || len(config.Added) > limits.MaxAddedTokens {
		return nil, ErrBudget
	}
	t := &Tokenizer{digest: expectedSHA256, limits: limits, merges: make(map[pair]merge, len(config.Model.Merges))}
	ids := make(map[int64]string, len(config.Model.Vocabulary)+len(config.Added))
	for value, sourceID := range config.Model.Vocabulary {
		id := int64(sourceID)
		if value == "" || !utf8.ValidString(value) || id < 0 || id > math.MaxInt32 {
			return nil, fmt.Errorf("%w: invalid vocabulary entry", ErrSchema)
		}
		if _, exists := ids[id]; exists {
			return nil, fmt.Errorf("%w: repeated vocabulary ID", ErrSchema)
		}
		ids[id] = value
	}
	for index, value := range byteAlphabet() {
		id, found := config.Model.Vocabulary[value]
		if !found {
			return nil, fmt.Errorf("%w: byte symbol %d absent", ErrSchema, index)
		}
		t.bytes[index] = int64(id)
	}
	for rank, value := range config.Model.Merges {
		if rank&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		left, right, found := strings.Cut(value, " ")
		leftID, foundLeft := config.Model.Vocabulary[left]
		rightID, foundRight := config.Model.Vocabulary[right]
		id, foundResult := config.Model.Vocabulary[left+right]
		if !found || left == "" || right == "" || strings.Contains(right, " ") || !foundLeft || !foundRight || !foundResult {
			return nil, fmt.Errorf("%w: invalid merge at rank %d", ErrSchema, rank)
		}
		key := pair{int64(leftID), int64(rightID)}
		if _, exists := t.merges[key]; exists {
			return nil, fmt.Errorf("%w: repeated merge pair", ErrSchema)
		}
		t.merges[key] = merge{rank: rank, id: int64(id)}
	}
	seenAdded := make(map[string]bool, len(config.Added))
	for _, added := range config.Added {
		if added.ID == nil || *added.ID < 0 || *added.ID > math.MaxInt32 || added.Content == "" ||
			!utf8.ValidString(added.Content) || !disabled(added.SingleWord) || !disabled(added.LStrip) ||
			!disabled(added.RStrip) || !disabled(added.Normalized) || added.Special == nil || seenAdded[added.Content] {
			return nil, fmt.Errorf("%w: unsupported added token", ErrSchema)
		}
		if len(added.Content) > limits.MaxAddedTokenBytes {
			return nil, ErrBudget
		}
		if prior, exists := ids[*added.ID]; exists && prior != added.Content {
			return nil, fmt.Errorf("%w: added token ID collision", ErrSchema)
		}
		if id, exists := config.Model.Vocabulary[added.Content]; exists && int64(id) != *added.ID {
			return nil, fmt.Errorf("%w: added token vocabulary collision", ErrSchema)
		}
		ids[*added.ID], seenAdded[added.Content] = added.Content, true
		node := &t.added
		for i := 0; i < len(added.Content); i++ {
			if node.next == nil {
				node.next = make(map[byte]*addedNode)
			}
			if node.next[added.Content[i]] == nil {
				node.next[added.Content[i]] = &addedNode{}
			}
			node = node.next[added.Content[i]]
		}
		node.id, node.match = *added.ID, true
	}
	return t, ctx.Err()
}

func validateSchema(c configuration) error {
	if c.Version != "1.0" || !literal(c.Truncation, "null") || !literal(c.Padding, "null") ||
		c.Added == nil || c.Normalizer.Type != "NFC" || c.PreTokenizer.Type != "Sequence" ||
		len(c.PreTokenizer.PreTokenizers) != 2 || !validByteLevel(c.PostProcessor) || !validByteLevel(c.Decoder) ||
		c.Model.Type != "BPE" || !literal(c.Model.Dropout, "null") || !literal(c.Model.Unknown, "null") ||
		!literal(c.Model.Prefix, `""`) || !literal(c.Model.Suffix, `""`) || !disabled(c.Model.FuseUnknown) ||
		!disabled(c.Model.Fallback) || !disabled(c.Model.Ignore) || c.Model.Vocabulary == nil || c.Model.Merges == nil {
		return ErrSchema
	}
	var split struct {
		Type    string `json:"type"`
		Pattern struct {
			Regex string `json:"Regex"`
		} `json:"pattern"`
		Behavior string `json:"behavior"`
		Invert   *bool  `json:"invert"`
	}
	var level byteLevel
	if decode(c.PreTokenizer.PreTokenizers[0], &split) != nil || decode(c.PreTokenizer.PreTokenizers[1], &level) != nil ||
		split.Type != "Split" || split.Pattern.Regex != splitPattern || split.Behavior != "Isolated" || !disabled(split.Invert) || !validByteLevel(level) {
		return ErrSchema
	}
	return nil
}

func disabled(value *bool) bool                       { return value != nil && !*value }
func literal(value json.RawMessage, want string) bool { return string(bytes.TrimSpace(value)) == want }
func validByteLevel(value byteLevel) bool {
	return value.Type == "ByteLevel" && disabled(value.AddPrefixSpace) && disabled(value.TrimOffsets) && disabled(value.UseRegex)
}
func decode(body []byte, value any) error {
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(value); err != nil {
		return err
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		return errors.New("trailing JSON value")
	}
	return nil
}

func byteAlphabet() [256]string {
	var result [256]string
	next := rune(256)
	for b := range 256 {
		r := rune(b)
		if !(b >= 33 && b <= 126 || b >= 161 && b <= 172 || b >= 174) {
			r = next
			next++
		}
		result[b] = string(r)
	}
	return result
}

type contextReader struct {
	ctx    context.Context
	source io.Reader
}

func (r contextReader) Read(p []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	return r.source.Read(p)
}
func readBounded(ctx context.Context, source io.Reader, limit int64) ([]byte, error) {
	if limit < 0 {
		return nil, ErrBudget
	}
	body, err := io.ReadAll(io.LimitReader(contextReader{ctx, source}, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(body)) > limit {
		return nil, ErrBudget
	}
	return body, ctx.Err()
}

// uniqueKeys prevents a hash-admitted document from silently redefining schema
// fields or vocabulary entries through JSON's duplicate-key ambiguity.
func uniqueKeys(ctx context.Context, body []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.UseNumber()
	count := 0
	var walk func(int) error
	walk = func(depth int) error {
		if depth > 32 {
			return ErrSchema
		}
		count++
		if count&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		value, err := decoder.Token()
		if err != nil {
			return errors.Join(ErrSchema, err)
		}
		switch value {
		case json.Delim('{'):
			seen := make(map[string]bool)
			for decoder.More() {
				key, err := decoder.Token()
				if err != nil {
					return errors.Join(ErrSchema, err)
				}
				name, ok := key.(string)
				if !ok || seen[name] {
					return ErrSchema
				}
				seen[name] = true
				if err := walk(depth + 1); err != nil {
					return err
				}
			}
			_, err = decoder.Token()
			return err
		case json.Delim('['):
			for decoder.More() {
				if err := walk(depth + 1); err != nil {
					return err
				}
			}
			_, err = decoder.Token()
			return err
		default:
			return nil
		}
	}
	if err := walk(0); err != nil {
		return err
	}
	if _, err := decoder.Token(); err != io.EOF {
		return ErrSchema
	}
	return ctx.Err()
}
