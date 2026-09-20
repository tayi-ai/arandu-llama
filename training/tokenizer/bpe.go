package tokenizer

import (
	"container/heap"
	"context"
)

type pair struct{ left, right int64 }
type merge struct {
	rank int
	id   int64
}
type symbol struct {
	id             int64
	previous, next int
	version        int
	live           bool
}
type candidate struct {
	rank, left, right, leftVersion, rightVersion int
	id                                           int64
}
type candidates []candidate

func (h candidates) Len() int { return len(h) }
func (h candidates) Less(i, j int) bool {
	return h[i].rank < h[j].rank || h[i].rank == h[j].rank && h[i].left < h[j].left
}
func (h candidates) Swap(i, j int)   { h[i], h[j] = h[j], h[i] }
func (h *candidates) Push(value any) { *h = append(*h, value.(candidate)) }
func (h *candidates) Pop() any {
	last := len(*h) - 1
	value := (*h)[last]
	*h = (*h)[:last]
	return value
}

func (t *Tokenizer) bpe(ctx context.Context, piece string, remaining int) ([]int64, error) {
	if remaining <= 0 {
		return nil, ErrBudget
	}
	symbols := make([]symbol, len(piece))
	// ByteLevel maps individual UTF-8 bytes, not Unicode code points.
	for i := 0; i < len(piece); i++ {
		symbols[i] = symbol{id: t.bytes[piece[i]], previous: i - 1, next: i + 1, live: true}
	}
	symbols[len(symbols)-1].next = -1
	queue := make(candidates, 0, len(piece))
	push := func(left, right int) {
		if left < 0 || right < 0 {
			return
		}
		a, b := symbols[left], symbols[right]
		if rule, ok := t.merges[pair{a.id, b.id}]; ok {
			heap.Push(&queue, candidate{rank: rule.rank, id: rule.id, left: left, right: right,
				leftVersion: a.version, rightVersion: b.version})
		}
	}
	for i := 0; i+1 < len(symbols); i++ {
		push(i, i+1)
	}
	operations := 0
	for len(queue) != 0 {
		operations++
		if operations&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		entry := heap.Pop(&queue).(candidate)
		left, right := &symbols[entry.left], &symbols[entry.right]
		if !left.live || !right.live || left.next != entry.right ||
			left.version != entry.leftVersion || right.version != entry.rightVersion {
			continue
		}
		left.id, left.next = entry.id, right.next
		left.version++
		right.live = false
		if right.next >= 0 {
			symbols[right.next].previous = entry.left
		}
		push(left.previous, entry.left)
		push(entry.left, left.next)
	}
	result := make([]int64, 0, min(len(piece), remaining))
	for at := 0; at >= 0; at = symbols[at].next {
		if len(result) >= remaining {
			return nil, ErrBudget
		}
		result = append(result, symbols[at].id)
	}
	return result, ctx.Err()
}
