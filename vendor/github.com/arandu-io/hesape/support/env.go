package support

import (
	"fmt"
	"os"
	"regexp"
	"strings"
	"sync"
)

// EnvRepository is a source of environment variables.
type EnvRepository interface {
	// Get returns the raw value of a variable and whether it is set at all.
	Get(key string) (string, bool)
}

// EnvRepositoryFunc lets a plain function be an [EnvRepository].
type EnvRepositoryFunc func(key string) (string, bool)

// Get calls f.
func (f EnvRepositoryFunc) Get(key string) (string, bool) { return f(key) }

// processEnvRepository reads the process environment.
type processEnvRepository struct{}

func (processEnvRepository) Get(key string) (string, bool) { return os.LookupEnv(key) }

// envAdapters is the ordered list of repositories a read walks.
type envAdapters struct {
	repositories []EnvRepository
}

// Get asks each repository in turn and returns the first value found.
func (a envAdapters) Get(key string) (string, bool) {
	for _, repository := range a.repositories {
		if v, ok := repository.Get(key); ok {
			return v, true
		}
	}
	return "", false
}

type envFacade struct{}

// Env reads environment variables, parsing the well-known literals into the
// values they name. It is a value rather than a type because a process has one
// environment: support.Env.Get(key) is the whole call.
var Env envFacade

var (
	envMu         sync.RWMutex
	envPutenv     = true
	envRepository EnvRepository
	envCustom     []func() EnvRepository
	envCustomName = map[string]int{}
)

// EnablePutenv puts the process environment back in the read path, and drops
// the repository built without it so the next read rebuilds.
func (envFacade) EnablePutenv() {
	envMu.Lock()
	defer envMu.Unlock()
	envPutenv = true
	envRepository = nil
}

// DisablePutenv takes the process environment out of the read path, so only
// the repositories registered through Extend answer. The repository built with
// it is dropped, so the next read rebuilds.
func (envFacade) DisablePutenv() {
	envMu.Lock()
	defer envMu.Unlock()
	envPutenv = false
	envRepository = nil
}

// Extend registers another repository, built on first use, and drops the
// repository already built so the next read rebuilds. The variadic argument
// names it: a name given twice replaces the repository registered under it,
// and an unnamed repository is always appended.
func (envFacade) Extend(callback func() EnvRepository, name ...string) {
	envMu.Lock()
	defer envMu.Unlock()
	envRepository = nil
	if len(name) > 0 && name[0] != "" {
		if index, ok := envCustomName[name[0]]; ok {
			envCustom[index] = callback
			return
		}
		envCustomName[name[0]] = len(envCustom)
	}
	envCustom = append(envCustom, callback)
}

// GetRepository returns the repository every read goes through, building it
// once and keeping it. It asks the process environment first, unless
// DisablePutenv took it out, then each registered repository in turn.
func (envFacade) GetRepository() EnvRepository {
	envMu.Lock()
	defer envMu.Unlock()
	if envRepository != nil {
		return envRepository
	}
	adapters := envAdapters{}
	if envPutenv {
		adapters.repositories = append(adapters.repositories, processEnvRepository{})
	}
	for _, custom := range envCustom {
		if built := custom(); built != nil {
			adapters.repositories = append(adapters.repositories, built)
		}
	}
	envRepository = adapters
	return envRepository
}

// quotedEnvValue matches a value wrapped in matching single or double quotes.
var quotedEnvValue = regexp.MustCompile(`^(['"])([\s\S]*)(['"])$`)

// Get returns the value of a variable, falling back to the optional default,
// which is nil when not given. A default that is a func() any is invoked and
// its result returned.
//
// The strings "true", "(true)", "false", "(false)", "empty", "(empty)", "null"
// and "(null)" become the value they name, in any case, and a quoted value
// loses its quotes. A variable set to "null" reads as nil and does not fall
// back to the default: the variable is set, and nil is what it is set to.
func (envFacade) Get(key string, def ...any) any {
	raw, ok := Env.GetRepository().Get(key)
	if !ok {
		return value(firstOr(def, nil))
	}
	return parseEnvValue(raw)
}

// GetOrFail returns the value of a variable, parsed as Get parses it, or an
// error naming the variable when it is not set.
func (envFacade) GetOrFail(key string) (any, error) {
	raw, ok := Env.GetRepository().Get(key)
	if !ok {
		return nil, fmt.Errorf("Environment variable [%s] has no value.", key)
	}
	return parseEnvValue(raw), nil
}

func parseEnvValue(raw string) any {
	switch strings.ToLower(raw) {
	case "true", "(true)":
		return true
	case "false", "(false)":
		return false
	case "empty", "(empty)":
		return ""
	case "null", "(null)":
		return nil
	}
	if matches := quotedEnvValue.FindStringSubmatch(raw); matches != nil && matches[1] == matches[3] {
		return matches[2]
	}
	return raw
}
