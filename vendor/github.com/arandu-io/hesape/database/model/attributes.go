package model

import (
	"encoding/json"
	"fmt"
	"maps"
	"reflect"
	"slices"
	"sort"
)

// GetAttributes returns every column of the row, as the database sees it.
//
// The row lives in the entity struct, so the map is built from it -- plus
// the raw attributes a column with no field behind it left behind (a
// withCount alias, a column a migration added and the struct has not caught
// up with).
func (m *Model[T]) GetAttributes() map[string]any {
	entity, ok := m.entityValue()
	if !ok {
		// A literal has no entity to read columns off. The raw attributes are
		// still whatever was put there, and an empty map is the honest answer
		// rather than a panic.
		return maps.Clone(m.attributes)
	}
	out := make(map[string]any, len(fieldsOf(entity.Type()))+len(m.attributes))
	for _, f := range fieldsOf(entity.Type()) {
		out[f.column] = valueAt(entity, f)
	}
	for key, value := range m.attributes {
		out[key] = value
	}
	return out
}

// SetRawAttributes replaces the row without checking anything, and
// optionally syncs the original.
//
// It is what Hydrate uses, and it is the only path that puts a value the caller
// did not declare on the model: a key with no field behind it is kept as a raw
// attribute rather than dropped, because a select the caller wrote has a reason
// for every column in it.
func (m *Model[T]) SetRawAttributes(attributes map[string]any, sync bool) error {
	m.resetEntity()
	m.attributes = nil
	if err := m.setAttributes(attributes, true); err != nil {
		return err
	}
	if sync {
		m.SyncOriginal()
	}
	return nil
}

// setAttributes writes a map onto the entity. keepUnknown decides what happens
// to a key with no field behind it: Fill drops it, ForceFill and
// SetRawAttributes keep it.
func (m *Model[T]) setAttributes(attributes map[string]any, keepUnknown bool) error {
	if len(attributes) == 0 {
		return nil
	}

	// The entity's reflect.Value is taken once rather than once per column, and
	// the walk is over the schema rather than over a sorted copy of the map's
	// keys. Sorting existed to make the first conversion error deterministic; so
	// does declaration order, and it costs no allocation and no sort per row.
	entity, ok := m.entityValue()
	if !ok {
		return ErrUnwired
	}
	schema := schemaOf(entity.Type())

	var discarded []string
	written := 0
	for i := range schema.fields {
		f := schema.fields[i]
		value, present := attributes[f.column]
		if !present {
			continue
		}
		written++
		dst, settable := settableAt(entity, f)
		if !settable {
			continue
		}
		if err := assign(dst, value); err != nil {
			return fmt.Errorf("model: %s.%s: %w", m.GetTable(), f.column, err)
		}
	}

	// Whatever the schema did not claim. A key with no field behind it is kept
	// as a raw attribute or reported, and the order it is reported in is the
	// sorted one, because a violation message that changes between runs is a
	// message nobody can assert on.
	if written == len(attributes) {
		return nil
	}
	for _, key := range sortedKeys(attributes) {
		if _, ok := schema.byName[key]; ok {
			continue
		}
		if keepUnknown {
			if m.attributes == nil {
				m.attributes = map[string]any{}
			}
			m.attributes[key] = attributes[key]
			continue
		}
		discarded = append(discarded, key)
	}
	return m.handleDiscardedAttributeViolation(discarded)
}

// SetAttribute converts value to the field's type and assigns it, or reports
// the conversion error. A column the entity does not declare is kept as a raw
// attribute instead.
func (m *Model[T]) SetAttribute(key string, value any) error {
	known, err := m.setAttribute(key, value)
	if err != nil {
		return err
	}
	if !known {
		if m.attributes == nil {
			m.attributes = map[string]any{}
		}
		m.attributes[key] = value
	}
	return nil
}

func (m *Model[T]) setAttribute(key string, value any) (bool, error) {
	entity, ok := m.entityValue()
	if !ok {
		return false, ErrUnwired
	}
	f, ok := fieldByColumn(entity.Type(), key)
	if !ok {
		return false, nil
	}
	dst, settable := settableAt(entity, f)
	if !settable {
		return false, nil
	}
	if err := assign(dst, value); err != nil {
		return true, fmt.Errorf("model: %s.%s: %w", m.GetTable(), key, err)
	}
	return true, nil
}

// GetAttribute returns the value for key: a column value if key names a
// field, else a raw attribute, else a loaded relation.
//
// A key that matches none of those reads as nil. PreventAccessingMissingAttributes
// turns that into a reported violation, though it catches much less here than
// it would need to elsewhere: a typo like found.Entity.Naem fails to
// compile, so it never reaches this check at all.
func (m *Model[T]) GetAttribute(key string) any {
	entity, live := m.entityValue()
	if !live {
		return m.attributes[key]
	}
	if f, ok := fieldByColumn(entity.Type(), key); ok {
		return valueAt(entity, f)
	}
	if value, ok := m.attributes[key]; ok {
		return value
	}
	if related, ok := m.relations[key]; ok {
		return related
	}
	m.handleMissingAttributeViolation(key)
	return nil
}

// AttributesToArray returns the row as it is serialised, with the hidden
// columns removed and the appended ones added.
func (m *Model[T]) AttributesToArray() map[string]any {
	attributes := m.GetAttributes()
	out := make(map[string]any, len(attributes))
	for key, value := range attributes {
		if !m.isVisible(key) {
			continue
		}
		out[key] = value
	}
	for _, key := range m.appends {
		if !m.isVisible(key) {
			continue
		}
		out[key] = m.GetAttribute(key)
	}
	return out
}

// isVisible reports whether key should be serialised: the visible list wins
// when it is set, otherwise the hidden list removes.
func (m *Model[T]) isVisible(key string) bool {
	if len(m.visible) > 0 {
		return slices.Contains(m.visible, key)
	}
	return !slices.Contains(m.hidden, key)
}

// ToArray returns the serialised row together with the loaded relations.
func (m *Model[T]) ToArray() map[string]any {
	out := m.AttributesToArray()
	for name, related := range m.relations {
		if !m.isVisible(name) {
			continue
		}
		out[name] = related
	}
	return out
}

// ToJSON encodes the serialised row as JSON. Go initialisms are upper case,
// hence ToJSON rather than ToJson, and it returns bytes rather than a string.
func (m *Model[T]) ToJSON() ([]byte, error) {
	out, err := json.Marshal(m.ToArray())
	if err != nil {
		return nil, fmt.Errorf("model: encoding %s: %w", m.GetTable(), err)
	}
	return out, nil
}

// ToPrettyJSON encodes the serialised row as indented JSON.
func (m *Model[T]) ToPrettyJSON() ([]byte, error) {
	out, err := json.MarshalIndent(m.ToArray(), "", "    ")
	if err != nil {
		return nil, fmt.Errorf("model: encoding %s: %w", m.GetTable(), err)
	}
	return out, nil
}

// GetOriginal returns the row as it was when it was last synced.
func (m *Model[T]) GetOriginal() map[string]any {
	return copyMap(m.original)
}

// GetRawOriginal returns the original value for one key, uncast.
//
// It is the same value GetOriginal would give for that key, because the cast
// is the field's type and the original was already cast when it was read.
// The two methods are kept apart anyway, because a caller that asks for the
// raw one is saying something about intent.
func (m *Model[T]) GetRawOriginal(key string) any { return m.original[key] }

// SyncOriginal replaces the original snapshot with the row's current values.
func (m *Model[T]) SyncOriginal() *Model[T] {
	m.original = m.GetAttributes()
	return m
}

// SyncOriginalAttribute replaces the original snapshot for one column with
// its current value.
func (m *Model[T]) SyncOriginalAttribute(attribute string) *Model[T] {
	return m.SyncOriginalAttributes(attribute)
}

// SyncOriginalAttributes replaces the original snapshot for the named
// columns with their current values.
func (m *Model[T]) SyncOriginalAttributes(attributes ...string) *Model[T] {
	current := m.GetAttributes()
	if m.original == nil {
		m.original = map[string]any{}
	}
	for _, key := range attributes {
		m.original[key] = current[key]
	}
	return m
}

// SyncChanges records the current dirty columns as the last save's changes,
// and captures what each one held before it.
func (m *Model[T]) SyncChanges() *Model[T] {
	m.changes = m.GetDirty()
	m.previous = map[string]any{}
	for key := range m.changes {
		if original, ok := m.original[key]; ok {
			m.previous[key] = original
		}
	}
	return m
}

// GetDirty returns the columns that differ from the original.
func (m *Model[T]) GetDirty() map[string]any {
	dirty := map[string]any{}
	for key, value := range m.GetAttributes() {
		if !m.OriginalIsEquivalent(key) {
			dirty[key] = value
		}
	}
	return dirty
}

// GetChanges returns what changed on the last save.
func (m *Model[T]) GetChanges() map[string]any { return copyMap(m.changes) }

// GetPrevious returns what the changed columns held before the last save.
func (m *Model[T]) GetPrevious() map[string]any { return copyMap(m.previous) }

// IsDirty reports whether the given columns differ from the original. With
// no argument it asks about the whole row.
func (m *Model[T]) IsDirty(attributes ...string) bool {
	return hasChanges(m.GetDirty(), attributes)
}

// IsClean reports the opposite of IsDirty.
func (m *Model[T]) IsClean(attributes ...string) bool { return !m.IsDirty(attributes...) }

// WasChanged reports whether the last save touched these columns.
func (m *Model[T]) WasChanged(attributes ...string) bool {
	return hasChanges(m.changes, attributes)
}

// DiscardChanges resets the row to its original values and clears the
// recorded changes.
func (m *Model[T]) DiscardChanges() error {
	original := copyMap(m.original)
	if err := m.SetRawAttributes(original, false); err != nil {
		return err
	}
	m.original = original
	m.changes = nil
	m.previous = nil
	return nil
}

// hasChanges reports whether changes contains any of attributes, or is
// non-empty when attributes is empty.
func hasChanges(changes map[string]any, attributes []string) bool {
	if len(attributes) == 0 {
		return len(changes) > 0
	}
	for _, attribute := range attributes {
		if _, ok := changes[attribute]; ok {
			return true
		}
	}
	return false
}

// OriginalIsEquivalent reports whether key's current value equals its
// original value.
//
// The field has one static type, and both the current and the original value
// went through assign to reach it, so this is a plain comparison rather than
// a ladder of type coercions. The one case a plain comparison cannot handle
// is an uncomparable field (a slice, a map), which reflect.DeepEqual handles
// instead.
func (m *Model[T]) OriginalIsEquivalent(key string) bool {
	original, ok := m.original[key]
	if !ok {
		return false
	}
	current := m.GetAttribute(key)
	if current == nil || original == nil {
		return current == nil && original == nil
	}
	return reflect.DeepEqual(current, original)
}

// Only returns a subset of the row, by column.
func (m *Model[T]) Only(attributes ...string) map[string]any {
	out := make(map[string]any, len(attributes))
	for _, attribute := range attributes {
		out[attribute] = m.GetAttribute(attribute)
	}
	return out
}

// Except returns the row without the named columns.
func (m *Model[T]) Except(attributes ...string) map[string]any {
	out := map[string]any{}
	for key := range m.GetAttributes() {
		if slices.Contains(attributes, key) {
			continue
		}
		out[key] = m.GetAttribute(key)
	}
	return out
}

// GetHidden returns the columns hidden from serialisation.
func (m *Model[T]) GetHidden() []string { return slices.Clone(m.hidden) }

// SetHidden replaces the columns hidden from serialisation.
func (m *Model[T]) SetHidden(hidden ...string) *Model[T] {
	m.hidden = slices.Clone(hidden)
	return m
}

// GetVisible returns the columns allowed in serialisation, when the visible
// list is in use.
func (m *Model[T]) GetVisible() []string { return slices.Clone(m.visible) }

// SetVisible replaces the columns allowed in serialisation.
func (m *Model[T]) SetVisible(visible ...string) *Model[T] {
	m.visible = slices.Clone(visible)
	return m
}

// MakeVisible takes the named columns out of the hidden list, and adds them
// to the visible list when that list is in use.
func (m *Model[T]) MakeVisible(attributes ...string) *Model[T] {
	m.hidden = slices.DeleteFunc(slices.Clone(m.hidden), func(key string) bool {
		return slices.Contains(attributes, key)
	})
	if len(m.visible) > 0 {
		m.visible = appendUnique(m.visible, attributes...)
	}
	return m
}

// MakeHidden adds the named columns to the hidden list.
func (m *Model[T]) MakeHidden(attributes ...string) *Model[T] {
	m.hidden = appendUnique(m.hidden, attributes...)
	return m
}

// Append adds a name that is serialised with the row without being a column.
//
// The value comes from a raw attribute or a loaded relation -- the two
// things a model can hold that the entity struct does not declare.
func (m *Model[T]) Append(attributes ...string) *Model[T] {
	m.appends = appendUnique(m.appends, attributes...)
	return m
}

// GetAppends returns the names appended to serialisation.
func (m *Model[T]) GetAppends() []string { return slices.Clone(m.appends) }

// SetAppends replaces the names appended to serialisation.
func (m *Model[T]) SetAppends(appends ...string) *Model[T] {
	m.appends = slices.Clone(appends)
	return m
}

// HasAppended reports whether attribute is in the appended list.
func (m *Model[T]) HasAppended(attribute string) bool {
	return slices.Contains(m.appends, attribute)
}

func appendUnique(list []string, values ...string) []string {
	out := slices.Clone(list)
	for _, value := range values {
		if !slices.Contains(out, value) {
			out = append(out, value)
		}
	}
	return out
}

func copyMap(in map[string]any) map[string]any {
	out := make(map[string]any, len(in))
	for key, value := range in {
		out[key] = value
	}
	return out
}

// sortedKeys is the answer to a Go map having no order.
//
// The same order is used everywhere a map becomes a column list, so what is
// compiled and what is bound are derived from the same sequence.
func sortedKeys(in map[string]any) []string {
	out := make([]string, 0, len(in))
	for key := range in {
		out = append(out, key)
	}
	sort.Strings(out)
	return out
}

// resetEntity clears the entity's columns, and puts the model back afterwards
// when the entity is the one it lives inside.
//
// This used to be `*m.Entity = *new(T)` written where SetRawAttributes calls it,
// and that line zeroed the model along with the columns: an entity that embeds
// Model[T] contains this very model, so the assignment set the receiver's own
// Entity to nil in the middle of the call, and the next line reflected over a
// nil pointer. The failure was a panic inside hydration -- the path every row of
// every query takes.
//
// A T that does not embed Model[T] is the simple case, and takes the simple
// path: nothing of the model lives in the entity, so zeroing it is just zeroing
// it.
func (m *Model[T]) resetEntity() {
	if m.Entity == nil {
		return
	}
	index := m.entityIndex()
	if index < 0 {
		*m.Entity = *new(T)
		return
	}

	saved := *m
	*m.Entity = *new(T)
	// m is interior to the entity, so it has just been zeroed with it. Writing
	// the copy back through it restores the same allocation the caller holds.
	*m = saved
}
