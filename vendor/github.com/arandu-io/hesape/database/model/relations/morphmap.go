package relations

import (
	"fmt"
	"sort"
	"sync"
)

// ErrMorphNotMapped is what a polymorphic read returns when the value in the
// type column names nothing.
//
// It is what an unregistered type produces in both directions: writing a model
// with no alias, and reading an alias nothing was registered under. There is no
// type name in the column to fall back to.
var ErrMorphNotMapped = fmt.Errorf("relations: morph type is not in the morph map")

var morphMap = struct {
	sync.RWMutex
	entries  map[string]func() Model
	required bool
}{entries: map[string]func() Model{}, required: true}

// FlushMorphMap empties the alias registry.
//
// It exists for the tests, and it exists because the registry is package state
// with no other way out of it: a test that registers "post" leaves it there for
// every test that runs after it in the same binary, and the failure that
// eventually shows up names a type the failing test never mentioned.
//
// It is not a runtime facility. An application registers its aliases once, at
// wiring, and calling this in one would unregister them for every goroutine at
// the same time.
func FlushMorphMap() {
	morphMap.Lock()
	defer morphMap.Unlock()
	morphMap.entries = map[string]func() Model{}
	morphMap.required = true
}

// MorphMap registers the aliases a polymorphic type column holds, and returns
// the map.
//
// # The map is mandatory here, and that is the better half of the trade
//
// Go has no type name at run time to write into the column, so the map is the
// mechanism rather than a recommendation. What the column holds is an alias the
// application chose -- "post", "video" -- and the type it points at can be
// renamed, moved between packages, or split in two, without a single row of data
// becoming unreadable.
//
// The values are factories rather than instances, because every read needs a
// fresh model and a shared one would be a data race the first time two requests
// loaded the same morph type.
//
//	relations.MorphMap(map[string]func() relations.Model{
//	    "post":  func() relations.Model { return &Post{} },
//	    "video": func() relations.Model { return &Video{} },
//	})
//
// merge defaults to true.
func MorphMap(entries map[string]func() Model, merge ...bool) map[string]func() Model {
	morphMap.Lock()
	defer morphMap.Unlock()

	shouldMerge := len(merge) == 0 || merge[0]
	if entries != nil {
		if !shouldMerge {
			morphMap.entries = map[string]func() Model{}
		}
		for alias, factory := range entries {
			morphMap.entries[alias] = factory
		}
	}

	return copyMorphMap()
}

// GetMorphMap returns the registered aliases. The PHP reads the public static
// property; a package variable cannot be exported without letting a caller
// replace the map mid-flight, so it is read through a function.
func GetMorphMap() map[string]func() Model {
	morphMap.RLock()
	defer morphMap.RUnlock()
	return copyMorphMap()
}

// EnforceMorphMap answers Relation::enforceMorphMap.
func EnforceMorphMap(entries map[string]func() Model, merge ...bool) map[string]func() Model {
	RequireMorphMap(true)
	return MorphMap(entries, merge...)
}

// RequireMorphMap answers Relation::requireMorphMap.
//
// It is on by default here, which the PHP's is not. Turning it off cannot
// restore the PHP's behaviour -- there is no class name in the column to
// resolve -- so what it changes is only whether an unmapped alias is refused
// early or fails later with a less helpful message.
func RequireMorphMap(required ...bool) {
	morphMap.Lock()
	defer morphMap.Unlock()
	morphMap.required = len(required) == 0 || required[0]
}

// RequiresMorphMap answers Relation::requiresMorphMap.
func RequiresMorphMap() bool {
	morphMap.RLock()
	defer morphMap.RUnlock()
	return morphMap.required
}

// GetMorphedModel answers Relation::getMorphedModel: the factory registered
// under an alias, or nil.
func GetMorphedModel(alias string) func() Model {
	morphMap.RLock()
	defer morphMap.RUnlock()
	return morphMap.entries[alias]
}

// CreateModelByType answers MorphTo::createModelByType together with
// Model::getActualClassNameForMorph.
//
// It carries an error where the PHP would either instantiate a class by name or
// throw: an alias nobody registered is a row this process cannot read, and the
// message says which alias and what is registered, because the answer is
// always "add it to the morph map".
func CreateModelByType(alias string) (Model, error) {
	factory := GetMorphedModel(alias)
	if factory == nil {
		return nil, fmt.Errorf("%w: %q is not registered. Registered: %v", ErrMorphNotMapped, alias, registeredAliases())
	}
	return factory(), nil
}

// GetMorphAlias answers Relation::getMorphAlias.
//
// The PHP searches the map for a class name. A model here already knows the
// alias it was registered under -- that is what GetMorphClass answers -- so the
// search is a lookup, and its only remaining job is refusing a model that was
// never registered while requireMorphMap is on.
func GetMorphAlias(model Model) (string, error) {
	alias := model.GetMorphClass()
	if !RequiresMorphMap() {
		return alias, nil
	}
	if GetMorphedModel(alias) == nil {
		return "", fmt.Errorf("%w: the model on table %q answers the morph alias %q, which is not registered", ErrMorphNotMapped, model.GetTable(), alias)
	}
	return alias, nil
}

func copyMorphMap() map[string]func() Model {
	out := make(map[string]func() Model, len(morphMap.entries))
	for alias, factory := range morphMap.entries {
		out[alias] = factory
	}
	return out
}

func registeredAliases() []string {
	morphMap.RLock()
	defer morphMap.RUnlock()

	aliases := make([]string, 0, len(morphMap.entries))
	for alias := range morphMap.entries {
		aliases = append(aliases, alias)
	}
	sort.Strings(aliases)
	return aliases
}
