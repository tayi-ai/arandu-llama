package tokenizer_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"math/rand/v2"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/tokenizer"
)

const pattern = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`

type fixture struct {
	vocab  map[string]int64
	merges []string
	pairs  map[string]bool
	added  []any
	next   int64
}

func encoded(raw string) string {
	var alphabet [256]rune
	extra := rune(256)
	for b := range 256 {
		alphabet[b] = rune(b)
		if b < 33 || b > 126 && b < 161 || b == 173 {
			alphabet[b] = extra
			extra++
		}
	}
	var out strings.Builder
	for i := 0; i < len(raw); i++ {
		out.WriteRune(alphabet[raw[i]])
	}
	return out.String()
}

func newFixture() *fixture {
	f := &fixture{vocab: map[string]int64{}, merges: []string{}, pairs: map[string]bool{}, added: []any{}, next: 1000}
	for b := range 256 {
		f.vocab[encoded(string([]byte{byte(b)}))] = int64(b)
	}
	return f
}
func (f *fixture) merge(left, right string) int64 {
	a, b := encoded(left), encoded(right)
	key := a + " " + b
	if !f.pairs[key] {
		f.merges = append(f.merges, key)
		f.pairs[key] = true
	}
	if _, ok := f.vocab[a+b]; !ok {
		f.vocab[a+b] = f.next
		f.next++
	}
	return f.vocab[a+b]
}
func (f *fixture) word(raw string) int64 {
	for i := 1; i < len(raw); i++ {
		f.merge(raw[:i], raw[i:i+1])
	}
	return f.vocab[encoded(raw)]
}
func (f *fixture) add(raw string, id int64, special bool) {
	f.added = append(f.added, map[string]any{"id": id, "content": raw, "single_word": false,
		"lstrip": false, "rstrip": false, "normalized": false, "special": special})
}
func (f *fixture) config() map[string]any {
	level := map[string]any{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}
	return map[string]any{"version": "1.0", "truncation": nil, "padding": nil, "added_tokens": f.added,
		"normalizer": map[string]any{"type": "NFC"},
		"pre_tokenizer": map[string]any{"type": "Sequence", "pretokenizers": []any{
			map[string]any{"type": "Split", "pattern": map[string]any{"Regex": pattern}, "behavior": "Isolated", "invert": false}, level}},
		"post_processor": level, "decoder": level, "model": map[string]any{"type": "BPE", "dropout": nil,
			"unk_token": nil, "continuing_subword_prefix": "", "end_of_word_suffix": "", "fuse_unk": false,
			"byte_fallback": false, "ignore_merges": false, "vocab": f.vocab, "merges": f.merges}}
}
func marshal(t *testing.T, value any) []byte {
	t.Helper()
	body, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return body
}
func digest(body []byte) string { sum := sha256.Sum256(body); return hex.EncodeToString(sum[:]) }
func load(t *testing.T, f *fixture, limits tokenizer.Limits) *tokenizer.Tokenizer {
	t.Helper()
	body := marshal(t, f.config())
	result, err := tokenizer.Load(context.Background(), bytes.NewReader(body), digest(body), limits)
	if err != nil {
		t.Fatal(err)
	}
	if result.SHA256() != digest(body) {
		t.Fatal("lost admitted identity")
	}
	return result
}
func assertIDs(t *testing.T, tk *tokenizer.Tokenizer, input string, want []int64) {
	t.Helper()
	got, err := tk.Encode(context.Background(), input)
	if err != nil || !slices.Equal(got, want) {
		t.Fatalf("Encode(%q): got %v, err %v; want %v", input, got, err, want)
	}
}

func TestRawByteAlphabetAndNoAutomaticSpecialTokens(t *testing.T) {
	tk := load(t, newFixture(), tokenizer.DefaultLimits())
	for _, input := range []string{"", "A B\n", "日本🙂\t", "\x00\u0085"} {
		want := make([]int64, len(input))
		for i := range want {
			want[i] = int64(input[i])
		}
		assertIDs(t, tk, input, want)
	}
	if _, err := tk.Encode(context.Background(), string([]byte{0xff})); err == nil {
		t.Fatal("accepted invalid UTF-8")
	}
}

func TestSplitPrioritiesContractionsNumbersAndWhitespace(t *testing.T) {
	cases := []struct {
		input  string
		pieces []string
	}{
		{"we're I'LL he'd", []string{"we", "'re", " I", "'LL", " he", "'d"}},
		{"12²٣", []string{"1", "2", "²", "٣"}},
		{"  cat", []string{" ", " cat"}},
		{"a   ", []string{"a", "   "}},
		{"x\t\tY", []string{"x", "\t", "\tY"}},
		{"a!\r\nb", []string{"a", "!\r\n", "b"}},
		{" \n \tZ", []string{" \n", " ", "\tZ"}},
		{"🙂x 中文", []string{"🙂x", " 中文"}},
		{"\u0301a", []string{"\u0301a"}},
		{"!\u0301", []string{"!\u0301"}},
		{"'ſ", []string{"'ſ"}},
	}
	for _, test := range cases {
		t.Run(test.input, func(t *testing.T) {
			f := newFixture()
			want := make([]int64, len(test.pieces))
			for i, piece := range test.pieces {
				want[i] = f.word(piece)
			}
			// Admit cross-boundary merges too: a tokenizer that silently joins
			// regex pieces will produce a different sequence.
			for i := 1; i < len(test.pieces); i++ {
				f.merge(test.pieces[i-1], test.pieces[i])
			}
			assertIDs(t, load(t, f, tokenizer.DefaultLimits()), test.input, want)
		})
	}
}

func TestNFCAndLiteralAddedTokensBeforeNormalization(t *testing.T) {
	f := newFixture()
	composed := f.word("é")
	f.add("<x>", 2000, true)
	f.add("<x>long", 2001, false)
	f.add("e\u0301!", 2002, false)
	tk := load(t, f, tokenizer.DefaultLimits())
	assertIDs(t, tk, "e\u0301", []int64{composed})
	assertIDs(t, tk, "<x>long<x>", []int64{2001, 2000})
	assertIDs(t, tk, "e\u0301!é", []int64{2002, composed})
	assertIDs(t, tk, "<x> <x>", []int64{2000, 32, 2000})
	if _, err := tk.Encode(context.Background(), "a"+strings.Repeat("\u0301", 31)); !errors.Is(err, tokenizer.ErrSchema) {
		t.Fatalf("stream-safe normalization must refuse implicit CGJ insertion: %v", err)
	}
}

func TestBPERankOrderingAndLeftmostTie(t *testing.T) {
	f := newFixture()
	bc := f.merge("b", "c")
	f.merge("a", "b")
	aa := f.merge("a", "a")
	assertIDs(t, load(t, f, tokenizer.DefaultLimits()), "abc", []int64{97, bc})
	assertIDs(t, load(t, f, tokenizer.DefaultLimits()), "aaa", []int64{aa, 97})
}

func TestHeapBPEMatchesIndependentQuadraticReduction(t *testing.T) {
	f := newFixture()
	// All pairs of small strings admit competing merges, including lower-ranked
	// merges that become available only after a higher-ranked merge is taken.
	words := []string{"a", "b", "c"}
	for length := 2; length <= 4; length++ {
		prior := slices.Clone(words)
		for _, word := range prior {
			if len(word) != length-1 {
				continue
			}
			for _, suffix := range []string{"a", "b", "c"} {
				words = append(words, word+suffix)
			}
		}
	}
	for _, word := range words {
		f.word(word)
	}
	random := rand.New(rand.NewPCG(19, 67))
	random.Shuffle(len(f.merges), func(i, j int) { f.merges[i], f.merges[j] = f.merges[j], f.merges[i] })
	ranks := map[string]int{}
	for i, pair := range f.merges {
		ranks[pair] = i
	}
	tk := load(t, f, tokenizer.DefaultLimits())
	for range 200 {
		var input strings.Builder
		for range 1 + random.IntN(40) {
			input.WriteByte("abc"[random.IntN(3)])
		}
		parts := strings.Split(input.String(), "")
		for {
			best, bestRank := -1, len(f.merges)
			for i := 0; i+1 < len(parts); i++ {
				if rank, ok := ranks[parts[i]+" "+parts[i+1]]; ok && rank < bestRank {
					best, bestRank = i, rank
				}
			}
			if best < 0 {
				break
			}
			parts[best] += parts[best+1]
			parts = append(parts[:best+1], parts[best+2:]...)
		}
		want := make([]int64, len(parts))
		for i, part := range parts {
			want[i] = f.vocab[part]
		}
		assertIDs(t, tk, input.String(), want)
	}
}

func TestStrictSchemaAndIdentity(t *testing.T) {
	f := newFixture()
	mutations := map[string]func(map[string]any){
		"normalizer":    func(c map[string]any) { c["normalizer"] = map[string]any{"type": "NFKC"} },
		"unknown field": func(c map[string]any) { c["other"] = true },
		"padding":       func(c map[string]any) { c["padding"] = map[string]any{"length": 10} },
		"regex": func(c map[string]any) {
			c["pre_tokenizer"].(map[string]any)["pretokenizers"].([]any)[0].(map[string]any)["pattern"] = map[string]any{"Regex": ".+"}
		},
		"byte regex":    func(c map[string]any) { c["decoder"].(map[string]any)["use_regex"] = true },
		"missing flag":  func(c map[string]any) { delete(c["model"].(map[string]any), "ignore_merges") },
		"dropout":       func(c map[string]any) { c["model"].(map[string]any)["dropout"] = 0.1 },
		"missing byte":  func(c map[string]any) { delete(c["model"].(map[string]any)["vocab"].(map[string]int64), "a") },
		"unknown merge": func(c map[string]any) { c["model"].(map[string]any)["merges"] = []string{"a z"} },
		"duplicate id":  func(c map[string]any) { c["model"].(map[string]any)["vocab"].(map[string]int64)["a"] = 98 },
		"normalized added": func(c map[string]any) {
			c["added_tokens"] = []any{map[string]any{"id": 2000, "content": "<x>", "normalized": true, "single_word": false, "lstrip": false, "rstrip": false, "special": true}}
		},
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			c := newFixture().config()
			mutate(c)
			body := marshal(t, c)
			if _, err := tokenizer.Load(context.Background(), bytes.NewReader(body), digest(body), tokenizer.DefaultLimits()); !errors.Is(err, tokenizer.ErrSchema) {
				t.Fatalf("expected schema refusal: %v", err)
			}
		})
	}
	body := marshal(t, f.config())
	for _, hash := range []string{"", strings.Repeat("0", 64), strings.ToUpper(digest(body))} {
		if _, err := tokenizer.Load(context.Background(), bytes.NewReader(body), hash, tokenizer.DefaultLimits()); !errors.Is(err, tokenizer.ErrIdentity) {
			t.Fatalf("identity accepted: %v", err)
		}
	}
	duplicate := append([]byte(`{"version":"wrong",`), body[1:]...)
	if _, err := tokenizer.Load(context.Background(), bytes.NewReader(duplicate), digest(duplicate), tokenizer.DefaultLimits()); !errors.Is(err, tokenizer.ErrSchema) {
		t.Fatalf("duplicate JSON key accepted: %v", err)
	}
}

func TestExplicitBudgetsAndCancellation(t *testing.T) {
	f := newFixture()
	f.word("abcd")
	f.add("<x>", 2000, true)
	for name, change := range map[string]func(*tokenizer.Limits){
		"input":  func(l *tokenizer.Limits) { l.MaxInputBytes = 3 },
		"piece":  func(l *tokenizer.Limits) { l.MaxPieceBytes = 3 },
		"tokens": func(l *tokenizer.Limits) { l.MaxTokens = 1 },
		"pieces": func(l *tokenizer.Limits) { l.MaxPieces = 1 },
	} {
		t.Run(name, func(t *testing.T) {
			l := tokenizer.DefaultLimits()
			change(&l)
			tk := load(t, f, l)
			input := "abcd"
			if name == "tokens" || name == "pieces" {
				input = "ab1"
			}
			if _, err := tk.Encode(context.Background(), input); !errors.Is(err, tokenizer.ErrBudget) {
				t.Fatalf("budget accepted: %v", err)
			}
		})
	}
	l := tokenizer.DefaultLimits()
	l.MaxInputBytes = 2
	if _, err := load(t, f, l).Encode(context.Background(), "\u0344"); !errors.Is(err, tokenizer.ErrBudget) {
		t.Fatalf("normalized expansion accepted: %v", err)
	}
	body := marshal(t, f.config())
	l = tokenizer.DefaultLimits()
	l.MaxJSONBytes = int64(len(body) - 1)
	if _, err := tokenizer.Load(context.Background(), bytes.NewReader(body), digest(body), l); !errors.Is(err, tokenizer.ErrBudget) {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := tokenizer.Load(ctx, bytes.NewReader(body), digest(body), tokenizer.DefaultLimits()); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	tk := load(t, f, tokenizer.DefaultLimits())
	if _, err := tk.Encode(ctx, "abc"); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if _, err := tk.Encode(nil, "abc"); err == nil {
		t.Fatal("nil context accepted")
	}
	if _, err := tokenizer.Load(context.Background(), io.MultiReader(bytes.NewReader(body), strings.NewReader("null")), digest(append(slices.Clone(body), []byte("null")...)), tokenizer.DefaultLimits()); !errors.Is(err, tokenizer.ErrSchema) {
		t.Fatalf("trailing JSON accepted: %v", err)
	}
}

func TestWhitespaceLookaheadRespectsReturnedPieceBudget(t *testing.T) {
	f := newFixture()
	want := []int64{f.word("\n"), f.word("   "), f.word(" a")}
	limits := tokenizer.DefaultLimits()
	limits.MaxPieceBytes = 3
	tk := load(t, f, limits)
	// The first branch backtracks to the newline; the four following spaces
	// belong to two later pieces, neither longer than three bytes.
	assertIDs(t, tk, "\n    a", want)
	// A later newline makes the first piece exceed the limit. It must not be
	// silently split at the earlier newline to make the input fit.
	if _, err := tk.Encode(context.Background(), "\n    \na"); !errors.Is(err, tokenizer.ErrBudget) {
		t.Fatalf("over-budget whitespace piece accepted: %v", err)
	}
}

func TestImmutableTokenizerSupportsConcurrentCalls(t *testing.T) {
	f := newFixture()
	want := f.word("hello")
	tk := load(t, f, tokenizer.DefaultLimits())
	var group sync.WaitGroup
	for range 8 {
		group.Add(1)
		go func() {
			defer group.Done()
			for range 10 {
				assertIDs(t, tk, "hello", []int64{want})
			}
		}()
	}
	group.Wait()
}

// The external reference is explicitly provided and never downloaded.
func TestAdmittedArtifactOptIn(t *testing.T) {
	path := os.Getenv("TRAINING_TOKENIZER_TEST_JSON")
	if path == "" {
		t.Skip("set TRAINING_TOKENIZER_TEST_JSON and TRAINING_TOKENIZER_TEST_SHA256")
	}
	expected := os.Getenv("TRAINING_TOKENIZER_TEST_SHA256")
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	tk, err := tokenizer.Load(context.Background(), file, expected, tokenizer.DefaultLimits())
	if err != nil {
		t.Fatal(err)
	}
	for _, input := range []string{"synthetic input", " A", " B"} {
		ids, err := tk.Encode(context.Background(), input)
		if err != nil || len(ids) == 0 {
			t.Fatalf("tokenization failed: %v", err)
		}
	}
}
