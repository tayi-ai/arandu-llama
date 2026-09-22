package concerns

import (
	"context"
	"math"
	"sort"
	"strconv"
	"time"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/database/query"
)

// PivotHost is what InteractsWithPivotTable asks of the relation it is mixed
// into.
//
// In PHP the trait reads $this->parent, $this->foreignPivotKey and the rest
// straight off BelongsToMany, because a trait is compiled into the class. Go
// composes rather than inlines, so the reads are a contract. Everything on it
// is a method BelongsToMany already has for its own reasons.
type PivotHost interface {
	// GetTable answers BelongsToMany::getTable: the intermediate table.
	GetTable() string

	// GetParent answers Relation::getParent.
	GetParent() Model

	// GetRelated answers Relation::getRelated.
	GetRelated() Model

	// ParentKeyValue answers $this->parent->{$this->parentKey}.
	ParentKeyValue() any

	// GetForeignPivotKeyName answers the method of the same name.
	GetForeignPivotKeyName() string

	// GetRelatedPivotKeyName answers the method of the same name.
	GetRelatedPivotKeyName() string

	// GetQualifiedForeignPivotKeyName answers the method of the same name.
	GetQualifiedForeignPivotKeyName() string

	// GetQualifiedRelatedPivotKeyName answers the method of the same name.
	GetQualifiedRelatedPivotKeyName() string

	// GetRelatedKeyName answers the method of the same name.
	GetRelatedKeyName() string

	// CreatedAt answers BelongsToMany::createdAt.
	CreatedAt() string

	// UpdatedAt answers BelongsToMany::updatedAt.
	UpdatedAt() string

	// NewPivotStatement answers the method of the same name: a bare query over
	// the intermediate table, already scoped to the Grant's tenant.
	NewPivotStatement(g auth.Grant) (*query.Builder, error)

	// NewPivotQuery answers the method of the same name: NewPivotStatement plus
	// the relation's own pivot constraints.
	NewPivotQuery(g auth.Grant) (*query.Builder, error)

	// NewPivot answers the method of the same name.
	NewPivot(attributes map[string]any, exists bool) Model

	// TouchIfTouching answers the method of the same name.
	TouchIfTouching(ctx context.Context, g auth.Grant) error
}

// PivotValue is one entry of BelongsToMany::$pivotValues.
type PivotValue struct {
	Column string
	Value  any
}

// AttachRecord is one entry of the id => attributes list that attach, sync and
// toggle pass around.
//
// PHP keys an array by the id and lets the value be either the attributes or,
// for a plain list, the id itself. A Go map would lose the order the records
// were given in, and order is the difference between a deterministic INSERT and
// one that reshuffles between runs, so it is a slice of pairs.
type AttachRecord struct {
	// ID is the related model's key.
	ID any

	// Attributes are the extra pivot columns for this row.
	Attributes map[string]any
}

// SyncChanges answers the array{attached, detached, updated} that sync and
// toggle return.
type SyncChanges struct {
	Attached []any
	Detached []any
	Updated  []any
}

// InteractsWithPivotTable answers the trait of the same name: everything a
// many-to-many relation does to the intermediate table.
//
// The three fields the PHP declares on BelongsToMany and this trait writes --
// $pivotColumns, $pivotValues and $using -- live here rather than there,
// because Go's embedding puts them on BelongsToMany all the same and a trait
// that reaches into its host through an interface for its own state would be
// plumbing with nothing on the other end.
//
// Every method that touches the database takes a context and an auth.Grant, and
// every statement it builds is filtered by the tenant that Grant carries. A
// pivot row is the sentence "this customer's user has that customer's role",
// and an unfiltered attach is how one customer's user ends up holding another's
// role with no policy ever having been violated.
type InteractsWithPivotTable struct {
	// Host is the relation this trait was mixed into.
	Host PivotHost

	// PivotColumns is BelongsToMany::$pivotColumns.
	PivotColumns []string

	// PivotValues is BelongsToMany::$pivotValues.
	PivotValues []PivotValue

	// Using reports whether a custom pivot model was named with using(). The
	// PHP holds the class name; a Go relation holds a factory, on
	// BelongsToMany, and this is the flag the trait branches on.
	Using bool
}

// WithPivot answers InteractsWithPivotTable::withPivot.
func (p *InteractsWithPivotTable) WithPivot(columns ...string) *InteractsWithPivotTable {
	p.PivotColumns = append(p.PivotColumns, columns...)
	return p
}

// HasPivotColumn answers InteractsWithPivotTable::hasPivotColumn.
func (p *InteractsWithPivotTable) HasPivotColumn(column string) bool {
	for _, have := range p.PivotColumns {
		if have == column {
			return true
		}
	}
	return false
}

// Attach answers InteractsWithPivotTable::attach.
//
// touch is the PHP's third argument, and it is variadic here only so the common
// call reads like the PHP's: Attach(ctx, g, ids, attributes) touches, and
// Attach(ctx, g, ids, attributes, false) does not.
func (p *InteractsWithPivotTable) Attach(ctx context.Context, g auth.Grant, ids any, attributes map[string]any, touch ...bool) error {
	// The PHP writes formatAttachRecords($this->parseIds($ids), $attributes),
	// and gets away with it because parseIds maps an array's values and keeps
	// its keys -- so an id => attributes list survives the call. ParseIDs here
	// answers keys only, so the list goes through whole and FormatRecordsList
	// inside does the same normalization.
	records := p.formatAttachRecords(ids, attributes)
	if len(records) > 0 {
		if p.Using {
			// attachUsingCustomClass: one save per record, so the pivot model's
			// own casts and events run.
			for _, record := range records {
				if err := p.Host.NewPivot(record, false).Save(ctx, g); err != nil {
					return err
				}
			}
		} else {
			statement, err := p.Host.NewPivotStatement(g)
			if err != nil {
				return err
			}
			if err := insert(ctx, statement, p.stampTenant(records, g)); err != nil {
				return err
			}
		}
	}

	if shouldTouch(touch) {
		return p.Host.TouchIfTouching(ctx, g)
	}
	return nil
}

// Detach answers InteractsWithPivotTable::detach.
//
// ids may be nil, which detaches everything the relation reaches -- and reaches
// is doing the work: the query is the relation's own, so it is already narrowed
// to this parent and this tenant. A detach that dropped the tenant filter would
// delete another customer's rows on a shared pivot table.
func (p *InteractsWithPivotTable) Detach(ctx context.Context, g auth.Grant, ids any, touch ...bool) (int64, error) {
	statement, err := p.Host.NewPivotQuery(g)
	if err != nil {
		return 0, err
	}

	if ids != nil {
		parsed := p.ParseIDs(ids)
		if len(parsed) == 0 {
			return 0, nil
		}
		statement = statement.WhereIn(p.Host.GetQualifiedRelatedPivotKeyName(), parsed)
	}

	affected, err := deleteFrom(ctx, statement)
	if err != nil {
		return 0, err
	}

	if shouldTouch(touch) {
		if err := p.Host.TouchIfTouching(ctx, g); err != nil {
			return affected, err
		}
	}
	return affected, nil
}

// Sync answers InteractsWithPivotTable::sync.
func (p *InteractsWithPivotTable) Sync(ctx context.Context, g auth.Grant, ids any, detaching ...bool) (SyncChanges, error) {
	changes := SyncChanges{Attached: []any{}, Detached: []any{}, Updated: []any{}}

	detach := len(detaching) == 0 || detaching[0]
	records := p.FormatRecordsList(ids)

	if len(records) == 0 && !detach {
		return changes, nil
	}

	current, err := p.currentlyAttachedKeys(ctx, g)
	if err != nil {
		return changes, err
	}

	if detach {
		missing := difference(current, keysOf(records))
		if len(missing) > 0 {
			if _, err := p.Detach(ctx, g, missing, false); err != nil {
				return changes, err
			}
			changes.Detached = p.CastKeys(missing)
		}
	}

	attached, updated, err := p.attachNew(ctx, g, records, current, false)
	if err != nil {
		return changes, err
	}
	changes.Attached = attached
	changes.Updated = updated

	if len(changes.Attached) > 0 || len(changes.Updated) > 0 || len(changes.Detached) > 0 {
		if err := p.Host.TouchIfTouching(ctx, g); err != nil {
			return changes, err
		}
	}
	return changes, nil
}

// SyncWithoutDetaching answers InteractsWithPivotTable::syncWithoutDetaching.
func (p *InteractsWithPivotTable) SyncWithoutDetaching(ctx context.Context, g auth.Grant, ids any) (SyncChanges, error) {
	return p.Sync(ctx, g, ids, false)
}

// SyncWithPivotValues answers InteractsWithPivotTable::syncWithPivotValues:
// every id is synced carrying the same extra pivot columns.
func (p *InteractsWithPivotTable) SyncWithPivotValues(ctx context.Context, g auth.Grant, ids any, values map[string]any, detaching ...bool) (SyncChanges, error) {
	parsed := p.ParseIDs(ids)
	records := make([]AttachRecord, 0, len(parsed))
	for _, id := range parsed {
		records = append(records, AttachRecord{ID: id, Attributes: values})
	}
	return p.Sync(ctx, g, records, detaching...)
}

// Toggle answers InteractsWithPivotTable::toggle: what is attached comes off,
// what is not goes on.
func (p *InteractsWithPivotTable) Toggle(ctx context.Context, g auth.Grant, ids any, touch ...bool) (SyncChanges, error) {
	changes := SyncChanges{Attached: []any{}, Detached: []any{}}

	records := p.FormatRecordsList(ids)

	current, err := p.currentlyAttachedKeys(ctx, g)
	if err != nil {
		return changes, err
	}

	detach := intersection(current, keysOf(records))
	if len(detach) > 0 {
		if _, err := p.Detach(ctx, g, detach, false); err != nil {
			return changes, err
		}
		changes.Detached = p.CastKeys(detach)
	}

	attach := make([]AttachRecord, 0, len(records))
	detached := make(map[string]struct{}, len(detach))
	for _, key := range detach {
		detached[dictionaryKey(key)] = struct{}{}
	}
	for _, record := range records {
		if _, ok := detached[dictionaryKey(record.ID)]; !ok {
			attach = append(attach, record)
		}
	}

	if len(attach) > 0 {
		if err := p.Attach(ctx, g, attach, nil, false); err != nil {
			return changes, err
		}
		for _, record := range attach {
			changes.Attached = append(changes.Attached, record.ID)
		}
	}

	if shouldTouch(touch) && (len(changes.Attached) > 0 || len(changes.Detached) > 0) {
		if err := p.Host.TouchIfTouching(ctx, g); err != nil {
			return changes, err
		}
	}
	return changes, nil
}

// UpdateExistingPivot answers InteractsWithPivotTable::updateExistingPivot.
func (p *InteractsWithPivotTable) UpdateExistingPivot(ctx context.Context, g auth.Grant, id any, attributes map[string]any, touch ...bool) (int64, error) {
	values := make(map[string]any, len(attributes)+1)
	for key, value := range attributes {
		values[key] = value
	}
	if p.HasPivotColumn(p.Host.UpdatedAt()) {
		values[p.Host.UpdatedAt()] = p.freshTimestamp()
	}

	statement, err := p.NewPivotStatementForID(g, id)
	if err != nil {
		return 0, err
	}

	updated, err := update(ctx, statement, values)
	if err != nil {
		return 0, err
	}

	if shouldTouch(touch) {
		if err := p.Host.TouchIfTouching(ctx, g); err != nil {
			return updated, err
		}
	}
	return updated, nil
}

// NewPivotStatementForID answers InteractsWithPivotTable::newPivotStatementForId.
func (p *InteractsWithPivotTable) NewPivotStatementForID(g auth.Grant, id any) (*query.Builder, error) {
	statement, err := p.Host.NewPivotQuery(g)
	if err != nil {
		return nil, err
	}
	return statement.WhereIn(p.Host.GetQualifiedRelatedPivotKeyName(), p.ParseIDs(id)), nil
}

// NewExistingPivot answers InteractsWithPivotTable::newExistingPivot.
func (p *InteractsWithPivotTable) NewExistingPivot(attributes map[string]any) Model {
	return p.Host.NewPivot(attributes, true)
}

// AllRelatedIDs answers BelongsToMany::allRelatedIds. The PHP spells the last
// word Ids.
func (p *InteractsWithPivotTable) AllRelatedIDs(ctx context.Context, g auth.Grant) ([]any, error) {
	return p.currentlyAttachedKeys(ctx, g)
}

// FormatRecordsList answers InteractsWithPivotTable::formatRecordsList.
//
// It accepts what the PHP accepts -- a scalar, a model, a list of either, or a
// list of id => attributes, which is []AttachRecord here.
func (p *InteractsWithPivotTable) FormatRecordsList(ids any) []AttachRecord {
	switch value := ids.(type) {
	case []AttachRecord:
		return value
	case AttachRecord:
		return []AttachRecord{value}
	case map[string]map[string]any:
		// A map loses the caller's order, so the rows are emitted in key order
		// rather than in an order that changes between runs.
		keys := make([]string, 0, len(value))
		for key := range value {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		records := make([]AttachRecord, 0, len(keys))
		for _, key := range keys {
			records = append(records, AttachRecord{ID: key, Attributes: value[key]})
		}
		return records
	}

	parsed := p.ParseIDs(ids)
	records := make([]AttachRecord, 0, len(parsed))
	for _, id := range parsed {
		records = append(records, AttachRecord{ID: id})
	}
	return records
}

// ParseIDs answers InteractsWithPivotTable::parseIds. The PHP spells it
// parseIds.
func (p *InteractsWithPivotTable) ParseIDs(value any) []any {
	switch ids := value.(type) {
	case nil:
		return nil
	case Model:
		return []any{ids.GetAttribute(p.Host.GetRelatedKeyName())}
	case []Model:
		out := make([]any, 0, len(ids))
		for _, model := range ids {
			out = append(out, model.GetAttribute(p.Host.GetRelatedKeyName()))
		}
		return out
	case []AttachRecord:
		out := make([]any, 0, len(ids))
		for _, record := range ids {
			out = append(out, record.ID)
		}
		return out
	case []any:
		out := make([]any, 0, len(ids))
		for _, id := range ids {
			if model, ok := id.(Model); ok {
				out = append(out, model.GetAttribute(p.Host.GetRelatedKeyName()))
				continue
			}
			out = append(out, id)
		}
		return out
	case []string:
		out := make([]any, 0, len(ids))
		for _, id := range ids {
			out = append(out, id)
		}
		return out
	case []int:
		out := make([]any, 0, len(ids))
		for _, id := range ids {
			out = append(out, id)
		}
		return out
	case []int64:
		out := make([]any, 0, len(ids))
		for _, id := range ids {
			out = append(out, id)
		}
		return out
	default:
		return []any{value}
	}
}

// ParseID answers InteractsWithPivotTable::parseId.
func (p *InteractsWithPivotTable) ParseID(value any) any {
	if model, ok := value.(Model); ok {
		return model.GetAttribute(p.Host.GetRelatedKeyName())
	}
	return value
}

// CastKeys answers InteractsWithPivotTable::castKeys.
func (p *InteractsWithPivotTable) CastKeys(keys []any) []any {
	out := make([]any, 0, len(keys))
	for _, key := range keys {
		out = append(out, p.CastKey(key))
	}
	return out
}

// CastKey answers InteractsWithPivotTable::castKey.
func (p *InteractsWithPivotTable) CastKey(key any) any {
	return GetTypeSwapValue(p.Host.GetRelated().GetKeyType(), key)
}

// GetTypeSwapValue answers InteractsWithPivotTable::getTypeSwapValue.
func GetTypeSwapValue(keyType string, value any) any {
	switch keyType {
	case "int", "integer":
		if number, ok := toInt64(value); ok {
			return number
		}
		return value
	case "string":
		key, err := GetDictionaryKey(value)
		if err != nil {
			return value
		}
		return key
	default:
		return value
	}
}

// attachNew answers InteractsWithPivotTable::attachNew.
func (p *InteractsWithPivotTable) attachNew(ctx context.Context, g auth.Grant, records []AttachRecord, current []any, touch bool) (attached, updated []any, err error) {
	attached, updated = []any{}, []any{}

	have := make(map[string]struct{}, len(current))
	for _, key := range current {
		have[dictionaryKey(key)] = struct{}{}
	}

	for _, record := range records {
		if _, ok := have[dictionaryKey(record.ID)]; !ok {
			if err := p.Attach(ctx, g, []AttachRecord{record}, nil, touch); err != nil {
				return attached, updated, err
			}
			attached = append(attached, p.CastKey(record.ID))
			continue
		}

		if len(record.Attributes) == 0 {
			continue
		}
		count, err := p.UpdateExistingPivot(ctx, g, record.ID, record.Attributes, touch)
		if err != nil {
			return attached, updated, err
		}
		if count > 0 {
			updated = append(updated, p.CastKey(record.ID))
		}
	}

	return attached, updated, nil
}

// formatAttachRecords answers InteractsWithPivotTable::formatAttachRecords.
func (p *InteractsWithPivotTable) formatAttachRecords(ids any, attributes map[string]any) []map[string]any {
	timed := p.HasPivotColumn(p.Host.CreatedAt()) || p.HasPivotColumn(p.Host.UpdatedAt())

	records := p.FormatRecordsList(ids)
	out := make([]map[string]any, 0, len(records))
	for _, record := range records {
		out = append(out, p.formatAttachRecord(record, attributes, timed))
	}
	return out
}

// formatAttachRecord answers InteractsWithPivotTable::formatAttachRecord.
func (p *InteractsWithPivotTable) formatAttachRecord(record AttachRecord, attributes map[string]any, timed bool) map[string]any {
	row := p.BaseAttachRecord(record.ID, timed)
	for key, value := range record.Attributes {
		row[key] = value
	}
	for key, value := range attributes {
		row[key] = value
	}
	return row
}

// BaseAttachRecord answers InteractsWithPivotTable::baseAttachRecord.
//
// Exported because MorphToMany overrides it, and an override in Go is a call
// the subtype makes rather than a method the parent dispatches.
func (p *InteractsWithPivotTable) BaseAttachRecord(id any, timed bool) map[string]any {
	record := map[string]any{
		p.Host.GetRelatedPivotKeyName(): id,
		p.Host.GetForeignPivotKeyName(): p.Host.ParentKeyValue(),
	}

	if timed {
		record = p.AddTimestampsToAttachment(record, false)
	}

	for _, value := range p.PivotValues {
		record[value.Column] = value.Value
	}

	return record
}

// AddTimestampsToAttachment answers
// InteractsWithPivotTable::addTimestampsToAttachment.
func (p *InteractsWithPivotTable) AddTimestampsToAttachment(record map[string]any, exists bool) map[string]any {
	fresh := p.freshTimestamp()

	if !exists && p.HasPivotColumn(p.Host.CreatedAt()) {
		record[p.Host.CreatedAt()] = fresh
	}
	if p.HasPivotColumn(p.Host.UpdatedAt()) {
		record[p.Host.UpdatedAt()] = fresh
	}
	return record
}

func (p *InteractsWithPivotTable) freshTimestamp() time.Time {
	if parent := p.Host.GetParent(); parent != nil {
		return parent.FreshTimestamp()
	}
	return time.Now()
}

// stampTenant puts the tenant on every row on its way into the pivot table.
//
// The where clause on a select is only half of it. A row inserted without the
// column is a row no scoped read will ever return -- an attach that appears to
// work and a relation that comes back empty -- and on a table with a nullable
// tenant it is a row every tenant can see.
func (p *InteractsWithPivotTable) stampTenant(records []map[string]any, g auth.Grant) []map[string]any {
	tenant := auth.Tenant(g)
	if tenant == "" {
		return records
	}
	if column := TenantColumnFor(p.Host.GetParent()); column != "" {
		for _, record := range records {
			if _, ok := record[column]; !ok {
				record[column] = tenant
			}
		}
	}
	return records
}

// currentlyAttachedKeys answers the pluck of
// getCurrentlyAttachedPivots()->pluck($this->relatedPivotKey).
func (p *InteractsWithPivotTable) currentlyAttachedKeys(ctx context.Context, g auth.Grant) ([]any, error) {
	statement, err := p.Host.NewPivotQuery(g)
	if err != nil {
		return nil, err
	}

	rows, err := selectRows(ctx, statement.Select(p.Host.GetRelatedPivotKeyName()))
	if err != nil {
		return nil, err
	}

	keys := make([]any, 0, len(rows))
	for _, row := range rows {
		keys = append(keys, row[p.Host.GetRelatedPivotKeyName()])
	}
	return keys, nil
}

// GetCurrentlyAttachedPivots answers
// InteractsWithPivotTable::getCurrentlyAttachedPivots.
func (p *InteractsWithPivotTable) GetCurrentlyAttachedPivots(ctx context.Context, g auth.Grant) ([]Model, error) {
	return p.GetCurrentlyAttachedPivotsForIDs(ctx, g, nil)
}

// GetCurrentlyAttachedPivotsForIDs answers
// InteractsWithPivotTable::getCurrentlyAttachedPivotsForIds.
func (p *InteractsWithPivotTable) GetCurrentlyAttachedPivotsForIDs(ctx context.Context, g auth.Grant, ids any) ([]Model, error) {
	statement, err := p.Host.NewPivotQuery(g)
	if err != nil {
		return nil, err
	}
	if ids != nil {
		statement = statement.WhereIn(p.Host.GetQualifiedRelatedPivotKeyName(), p.ParseIDs(ids))
	}

	rows, err := selectRows(ctx, statement)
	if err != nil {
		return nil, err
	}

	pivots := make([]Model, 0, len(rows))
	for _, row := range rows {
		pivots = append(pivots, p.Host.NewPivot(row, true))
	}
	return pivots, nil
}

func shouldTouch(touch []bool) bool { return len(touch) == 0 || touch[0] }

func keysOf(records []AttachRecord) []any {
	out := make([]any, 0, len(records))
	for _, record := range records {
		out = append(out, record.ID)
	}
	return out
}

func dictionaryKey(value any) string {
	key, err := GetDictionaryKey(value)
	if err != nil {
		return ""
	}
	return key
}

// difference answers array_diff: what is in left and not in right.
func difference(left, right []any) []any {
	have := make(map[string]struct{}, len(right))
	for _, value := range right {
		have[dictionaryKey(value)] = struct{}{}
	}

	out := make([]any, 0, len(left))
	for _, value := range left {
		if _, ok := have[dictionaryKey(value)]; !ok {
			out = append(out, value)
		}
	}
	return out
}

// intersection answers array_intersect.
func intersection(left, right []any) []any {
	have := make(map[string]struct{}, len(right))
	for _, value := range right {
		have[dictionaryKey(value)] = struct{}{}
	}

	out := make([]any, 0, len(left))
	for _, value := range left {
		if _, ok := have[dictionaryKey(value)]; ok {
			out = append(out, value)
		}
	}
	return out
}

// toInt64 answers value as an int64, and whether it is one at all.
//
// The type set is the one GetDictionaryKey renders as a number, because these
// two read the same keys and a value one of them can count and the other cannot
// is the shape a key comparison goes wrong in.
//
// Text is part of that set, and it is where this went wrong. A driver may return
// an integer column as bytes -- MySQL's text protocol returns every column that
// way -- and the connection renders those to a string before a relation ever
// sees them. So an int primary key reaches here as "7" and left as "7", while
// the same key from the caller was int64: one change set answering two types for
// one column.
//
// Text that is not a base-ten integer is refused rather than read as far as it
// parses. "seven" has no number in it, and "7.9" has one that is not this key:
// truncating it would make "7.9" and "7" the same key, which is the failure this
// function exists to prevent rather than a lesser version of it. Refusing leaves
// the value as it came, where it compares unequal and stays visible.
//
// A number is held to that same rule, because the rule is about the value and
// not about the type a driver boxed it in. A fraction and a magnitude the int64
// range does not hold are refused here for the reason they are refused in text:
// both conversions are total in Go, so each of them answers a key that another
// value already holds.
func toInt64(value any) (int64, bool) {
	switch number := value.(type) {
	case int:
		return int64(number), true
	case int8:
		return int64(number), true
	case int16:
		return int64(number), true
	case int32:
		return int64(number), true
	case int64:
		return number, true
	case uint:
		return unsignedIntegerKey(uint64(number))
	case uint8:
		return int64(number), true
	case uint16:
		return int64(number), true
	case uint32:
		return int64(number), true
	case uint64:
		return unsignedIntegerKey(number)
	case float32:
		return floatIntegerKey(float64(number))
	case float64:
		return floatIntegerKey(number)
	case string:
		return parseIntegerKey(number)
	case []byte:
		return parseIntegerKey(string(number))
	}
	return 0, false
}

// unsignedIntegerKey reads a key that arrived unsigned, and refuses one past the
// int64 range.
//
// The conversion wraps rather than fails, so 2^63 would answer the most negative
// int64 -- the key a genuinely negative id answers, byte for byte, while the
// dictionary renders the two apart. Refusing leaves the value as it came, where
// it still names its own row.
func unsignedIntegerKey(number uint64) (int64, bool) {
	if number > math.MaxInt64 {
		return 0, false
	}
	return int64(number), true
}

// floatIntegerKey reads a key that arrived as a float, and refuses one that is
// not the whole of an int64.
//
// A fraction is refused rather than truncated: 7.9 read as 7 is two keys
// arriving at one. Out of range is refused because the conversion is undefined
// there, so every value past the range would land on whatever one key it
// happens to produce. A value that is no number at all needs no case of its own:
// it equals nothing, itself included, so the whole-number test below refuses it.
func floatIntegerKey(number float64) (int64, bool) {
	// int64 spans [-2^63, 2^63). Both bounds are exact as a float64 and the
	// largest int64 is not, so the upper one is compared as the exclusive 2^63.
	const (
		lowest = -9223372036854775808.0
		limit  = 9223372036854775808.0
	)

	if number != math.Trunc(number) {
		return 0, false
	}
	if number < lowest || number >= limit {
		return 0, false
	}
	return int64(number), true
}

// parseIntegerKey reads a key that arrived as text, and refuses anything that is
// not the whole of a base-ten integer.
//
// Whitespace, a fraction, a suffix and a value too large for an int64 are all
// refused, because each of them would map two different keys onto one.
func parseIntegerKey(text string) (int64, bool) {
	number, err := strconv.ParseInt(text, 10, 64)
	if err != nil {
		return 0, false
	}
	return number, true
}
