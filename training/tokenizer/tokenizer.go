// Package tokenizer encodes raw text using the explicitly admitted Ornith BPE
// schema. It does not apply a chat template or insert BOS/EOS tokens.
// The pinned Go 1.27 and x/text tables use Unicode 17 for classification and NFC.
// Compatibility with Hugging Face Tokenizers requires corpus qualification;
// this package does not promise equivalent behavior for all Unicode inputs.
package tokenizer

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

// ErrBudget reports a configured input, piece, token or model-size limit.
var ErrBudget = errors.New("tokenizer: explicit budget exceeded or invalid")

// ErrSchema reports a tokenizer configuration outside the supported schema.
var ErrSchema = errors.New("tokenizer: unsupported or invalid schema")

// ErrIdentity reports missing or mismatched source SHA256 identity.
var ErrIdentity = errors.New("tokenizer: source SHA256 differs or is missing")

// Limits bounds serialized input, normalized text and algorithm-owned indices.
// These are payload limits, not a measured process RSS or allocator peak.
// MaxInputBytes applies to both raw input and the sum of normalized segments.
type Limits struct {
	MaxJSONBytes, MaxInputBytes, MaxPieceBytes int64
	MaxPieces, MaxTokens                       int
	MaxVocabulary, MaxMerges                   int
	MaxAddedTokens, MaxAddedTokenBytes         int
}

// DefaultLimits admits the pinned 12.8 MB tokenizer and bounded raw prompts.
func DefaultLimits() Limits {
	return Limits{MaxJSONBytes: 16 << 20, MaxInputBytes: 1 << 20, MaxPieceBytes: 64 << 10,
		MaxPieces: 65536, MaxTokens: 1 << 20, MaxVocabulary: 300000, MaxMerges: 300000,
		MaxAddedTokens: 256, MaxAddedTokenBytes: 1024}
}

// Tokenizer is immutable after Load and supports independent concurrent Encode
// calls. Its expected hash is supplied by the owner of the model admission.
type Tokenizer struct {
	digest string
	limits Limits
	bytes  [256]int64
	merges map[pair]merge
	added  addedNode
}

type addedNode struct {
	next  map[byte]*addedNode
	id    int64
	match bool
}

// SHA256 returns the digest verified by Load, or an empty string for a nil value.
func (t *Tokenizer) SHA256() string {
	if t == nil {
		return ""
	}
	return t.digest
}

// Encode returns the complete raw sequence. Literal added tokens are recognized
// before NFC normalization, including tokens marked special in the source;
// special-token insertion is always disabled. There is no truncation or padding.
// Cancellation is checked between bounded phases and during segmentation/BPE.
// Unicode classification and NFC follow this package's pinned Unicode 17 tables;
// exact token parity must be established for the caller's corpus.
func (t *Tokenizer) Encode(ctx context.Context, input string) ([]int64, error) {
	if ctx == nil || t == nil || t.digest == "" {
		return nil, errors.New("tokenizer: initialized tokenizer and context required")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if int64(len(input)) > t.limits.MaxInputBytes {
		return nil, ErrBudget
	}
	if !utf8.ValidString(input) {
		return nil, errors.New("tokenizer: input is not valid UTF-8")
	}
	result := make([]int64, 0)
	pieces, normalizedBytes := 0, int64(0)
	encode := func(raw string) error {
		if raw == "" {
			return nil
		}
		// Streaming NFC avoids allocating an unbounded normalized expansion.
		reader := norm.NFC.Reader(strings.NewReader(raw))
		normalized, err := readBounded(ctx, reader, t.limits.MaxInputBytes-normalizedBytes)
		if err != nil {
			return err
		}
		normalizedBytes += int64(len(normalized))
		text := string(normalized)
		// x/text enforces stream-safe normalization by inserting CGJ after long
		// non-starter runs. The admitted NFC contract does not insert it: refuse
		// such inputs instead of silently changing their token sequence.
		if strings.Count(text, "\u034f") != strings.Count(raw, "\u034f") {
			return fmt.Errorf("%w: NFC requires a non-stream-safe combining sequence", ErrSchema)
		}
		for len(text) != 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
			end, err := splitEnd(ctx, text, t.limits.MaxPieceBytes)
			if err != nil {
				return err
			}
			pieces++
			if pieces > t.limits.MaxPieces {
				return ErrBudget
			}
			ids, err := t.bpe(ctx, text[:end], t.limits.MaxTokens-len(result))
			if err != nil {
				return err
			}
			result = append(result, ids...)
			text = text[end:]
		}
		return nil
	}
	start := 0
	for at := 0; at < len(input); {
		if at&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		length, id := t.addedAt(input[at:])
		if length == 0 {
			at++
			continue
		}
		if err := encode(input[start:at]); err != nil {
			return nil, err
		}
		pieces++
		if pieces > t.limits.MaxPieces || len(result) >= t.limits.MaxTokens {
			return nil, ErrBudget
		}
		result = append(result, id)
		at += length
		start = at
	}
	if err := encode(input[start:]); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return result, nil
}

func (t *Tokenizer) addedAt(input string) (length int, id int64) {
	node := &t.added
	for i := 0; i < len(input); i++ {
		node = node.next[input[i]]
		if node == nil {
			break
		}
		if node.match {
			length, id = i+1, node.id
		}
	}
	return
}

const splitPattern = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`

func letterMark(r rune) bool { return unicode.IsLetter(r) || unicode.IsMark(r) }
func number(r rune) bool     { return unicode.IsNumber(r) }
func newline(r rune) bool    { return r == '\r' || r == '\n' }

// splitEnd implements the seven alternatives of splitPattern in their declared
// order, including greedy backtracking in the final whitespace alternatives.
// Returning an over-budget piece is forbidden; pieces are never truncated.
func splitEnd(ctx context.Context, text string, limit int64) (int, error) {
	finish := func(end int) (int, error) {
		if end <= 0 || int64(end) > limit {
			return 0, ErrBudget
		}
		return end, ctx.Err()
	}
	scan := func(start int, accepts func(rune) bool) (int, error) {
		at := start
		for at < len(text) {
			r, size := utf8.DecodeRuneInString(text[at:])
			if !accepts(r) {
				break
			}
			at += size
			if int64(at) > limit {
				return 0, ErrBudget
			}
			if at&255 < utf8.UTFMax {
				if err := ctx.Err(); err != nil {
					return 0, err
				}
			}
		}
		return at, nil
	}
	// (?i:'s|'t|'re|'ve|'m|'ll|'d)
	if text[0] == '\'' {
		for _, ending := range []string{"s", "t", "re", "ve", "m", "ll", "d"} {
			at := 1
			matched := true
			for _, want := range ending {
				if at >= len(text) {
					matched = false
					break
				}
				got, size := utf8.DecodeRuneInString(text[at:])
				if !strings.EqualFold(string(got), string(want)) {
					matched = false
					break
				}
				at += size
			}
			if matched {
				return finish(at)
			}
		}
	}
	first, size := utf8.DecodeRuneInString(text)
	// [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
	start := 0
	if !newline(first) && !unicode.IsLetter(first) && !number(first) && size < len(text) {
		next, _ := utf8.DecodeRuneInString(text[size:])
		if letterMark(next) {
			start = size
		}
	}
	if firstOfRun, _ := utf8.DecodeRuneInString(text[start:]); letterMark(firstOfRun) {
		end, err := scan(start, letterMark)
		if err != nil {
			return 0, err
		}
		return finish(end)
	}
	// \p{N}: one Unicode number, never a run of decimal digits.
	if number(first) {
		return finish(size)
	}
	//  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
	start = 0
	if first == ' ' && size < len(text) {
		start = size
	}
	punctuation := func(r rune) bool { return !unicode.IsSpace(r) && !letterMark(r) && !number(r) }
	if next, _ := utf8.DecodeRuneInString(text[start:]); punctuation(next) {
		end, err := scan(start, punctuation)
		if err != nil {
			return 0, err
		}
		end, err = scan(end, newline)
		if err != nil {
			return 0, err
		}
		return finish(end)
	}
	if unicode.IsSpace(first) {
		// \s*[\r\n]+ backtracks to the last CR/LF in the whitespace run.
		at, lastNewline, lastStart := 0, 0, 0
		for at < len(text) {
			r, width := utf8.DecodeRuneInString(text[at:])
			if !unicode.IsSpace(r) {
				break
			}
			lastStart = at
			at += width
			if newline(r) {
				lastNewline = at
			}
			// A newline permits backtracking past the following whitespace.
			// Without one, only the last rune can be left behind by (?!\S).
			if int64(lastNewline) > limit || lastNewline == 0 && int64(lastStart) > limit {
				return 0, ErrBudget
			}
			if at&255 < utf8.UTFMax {
				if err := ctx.Err(); err != nil {
					return 0, err
				}
			}
		}
		if lastNewline != 0 {
			return finish(lastNewline)
		}
		// \s+(?!\S) leaves the last whitespace before non-whitespace.
		if at < len(text) && lastStart != 0 {
			return finish(lastStart)
		}
		// End-of-input for that branch, or the final \s+ alternative.
		return finish(at)
	}
	return 0, fmt.Errorf("%w: input not covered by admitted split pattern", ErrSchema)
}
