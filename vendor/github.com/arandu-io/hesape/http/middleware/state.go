package middleware

import (
	"net/http"
	"sync"
)

// globalState holds the package-level configuration TrustHosts and
// HandleCors read at request time: with no dependency-injection container,
// there is nowhere else for cross-cutting settings like these to live.
// TrustProxies takes its trusted prefixes as a plain argument instead,
// since it has exactly one caller per deployment and no need to be told
// twice. The two that do use it are gathered in one struct so that
// [FlushState] clears both in one call, which is what a test resetting
// state between cases wants.
//
// The mutex guards concurrent access, since Go runs tests -- and requests
// -- with a race detector watching.
var globalState = struct {
	sync.RWMutex

	// alwaysTrustHosts is the hosts [At] configured as always trusted.
	alwaysTrustHosts []string
	// alwaysTrustHostsSet distinguishes "no hosts configured" from "an
	// empty list was configured".
	alwaysTrustHostsSet bool
	// trustSubdomains is whether [At] asked for subdomains of the trusted
	// hosts to be trusted too.
	trustSubdomains bool

	// skipCallbacks are the callbacks [SkipWhen] registered.
	skipCallbacks []func(*http.Request) bool
}{}

// At names the hosts that should always be trusted, whatever a particular
// [TrustHosts] instance was built with.
//
// It takes the list directly rather than a callback that would defer
// reading configuration until later: there is no container here to boot
// before such a callback would run.
func At(hosts []string, subdomains bool) {
	globalState.Lock()
	globalState.alwaysTrustHosts = hosts
	globalState.alwaysTrustHostsSet = true
	globalState.trustSubdomains = subdomains
	globalState.Unlock()
}

// Hosts is the host patterns [At] configured as always trusted.
//
// It is empty until [At] is called. There is no configuration repository
// behind this package, so a caller with hosts of its own passes them to
// [TrustHosts] directly.
func Hosts() []string {
	globalState.RLock()
	defer globalState.RUnlock()
	if !globalState.alwaysTrustHostsSet {
		return nil
	}
	out := make([]string, len(globalState.alwaysTrustHosts))
	copy(out, globalState.alwaysTrustHosts)
	return out
}

// TrustsSubdomains reports whether [At] asked for subdomains of the trusted
// hosts to be trusted too.
func TrustsSubdomains() bool {
	globalState.RLock()
	defer globalState.RUnlock()
	return globalState.trustSubdomains
}

// SkipWhen skips the CORS handling for the requests the callback picks out.
//
// Each call appends; nothing removes one. [FlushState] is how a test undoes
// it.
func SkipWhen(callback func(*http.Request) bool) {
	globalState.Lock()
	globalState.skipCallbacks = append(globalState.skipCallbacks, callback)
	globalState.Unlock()
}

// shouldSkipCors reports whether any registered [SkipWhen] callback matches
// the request.
func shouldSkipCors(r *http.Request) bool {
	globalState.RLock()
	callbacks := globalState.skipCallbacks
	globalState.RUnlock()
	for _, callback := range callbacks {
		if callback(r) {
			return true
		}
	}
	return false
}

// FlushState forgets everything [At] and [SkipWhen] were told.
//
// It is one function rather than three because the state they configure
// lives in one Go package; a test resetting between cases calls this once
// instead of clearing each independently.
func FlushState() {
	globalState.Lock()
	globalState.alwaysTrustHosts = nil
	globalState.alwaysTrustHostsSet = false
	globalState.trustSubdomains = false
	globalState.skipCallbacks = nil
	globalState.Unlock()
}
