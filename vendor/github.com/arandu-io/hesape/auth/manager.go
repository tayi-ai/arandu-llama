package auth

import (
	"fmt"
	"sync"
)

// ManagerConfig is everything [NewAuthManager] needs to build a guard: the two
// configuration maps, and the six services the built-in drivers are wired from.
//
// The manager is handed what it needs rather than resolving it, which is why
// there is nothing here to set an application on afterwards.
//
// Guards and Providers are maps rather than structs because a custom driver
// registered with [AuthManager.Extend] reads keys this package has never heard
// of.
type ManagerConfig struct {
	// DefaultGuard is the guard an empty name means.
	DefaultGuard string

	// DefaultProvider is the provider an empty name means.
	DefaultProvider string

	// Guards maps a guard's name to its settings. The settings a built-in
	// driver reads are "driver" ("session" or "token"), "provider", "remember",
	// "input_key", "storage_key" and "hash".
	Guards map[string]map[string]any

	// Providers maps a provider's name to its settings, of which only "driver"
	// is read here.
	Providers map[string]map[string]any

	// Session is the session store the session driver needs.
	Session Session

	// Cookies is the jar the "remember me" cookie is queued on.
	Cookies CookieJar

	// Events is the dispatcher the guards fire on.
	Events Dispatcher

	// Request is what the guards read cookies and tokens from.
	//
	// There is nothing to refresh between requests: a guard is built per
	// request, or its request is replaced with SetRequest.
	Request Request

	// Hasher is what SessionGuard needs for LogoutOtherDevices.
	Hasher Hasher

	// RehashOnLogin asks the provider to upgrade a weakly hashed password on a
	// sign-in that proved the plain one.
	RehashOnLogin bool

	// TimeboxDuration is how long an attempt is held open for, in microseconds.
	// Zero means 200000.
	TimeboxDuration int

	// HashKey is the application key SessionGuard signs the recaller's password
	// segment with.
	HashKey string
}

// AuthManager is the factory that turns a guard's name into a guard, and the
// registry of the user providers those guards read from.
//
// It caches what it resolves. Read that twice before sharing one across
// requests: a guard carries the user it resolved and the request it read
// cookies from, so a cached SessionGuard is per-request state. Build the manager
// per request, or call [AuthManager.ForgetGuards] between them.
//
// There is no forwarding of unknown calls to the default guard: call
// [AuthManager.Guard] and then the method.
type AuthManager struct {
	// mu guards everything below it, because a Go server answers requests
	// concurrently.
	mu sync.Mutex

	// defaultGuard is the guard an empty name means, which SetDefaultDriver
	// rewrites.
	defaultGuard string

	// defaultProvider is the provider an empty name means.
	defaultProvider string

	// guardConfig is the guard settings, by guard name.
	guardConfig map[string]map[string]any

	// providerConfig is the provider settings, by provider name.
	providerConfig map[string]map[string]any

	// guards is the drivers already resolved, by name.
	guards map[string]Guard

	// customCreators is the guard driver registry Extend writes.
	customCreators map[string]func(manager *AuthManager, name string, config map[string]any) (Guard, error)

	// customProviderCreators is the provider driver registry Provider writes.
	customProviderCreators map[string]func(config map[string]any) (UserProvider, error)

	// userResolver answers who is acting on a given guard.
	userResolver func(guard string) Authenticatable

	// The services the drivers are built from. See [ManagerConfig].
	session         Session
	cookies         CookieJar
	events          Dispatcher
	request         Request
	hasher          Hasher
	rehashOnLogin   bool
	timeboxDuration int
	hashKey         string
}

// NewAuthManager returns a manager over the given configuration.
//
// The user resolver it starts with answers with the user of the named guard, or
// of the default one; [AuthManager.ResolveUsersUsing] replaces it.
func NewAuthManager(config ManagerConfig) *AuthManager {
	manager := &AuthManager{
		defaultGuard:           config.DefaultGuard,
		defaultProvider:        config.DefaultProvider,
		guardConfig:            config.Guards,
		providerConfig:         config.Providers,
		guards:                 map[string]Guard{},
		customCreators:         map[string]func(manager *AuthManager, name string, config map[string]any) (Guard, error){},
		customProviderCreators: map[string]func(config map[string]any) (UserProvider, error){},
		session:                config.Session,
		cookies:                config.Cookies,
		events:                 config.Events,
		request:                config.Request,
		hasher:                 config.Hasher,
		rehashOnLogin:          config.RehashOnLogin,
		timeboxDuration:        config.TimeboxDuration,
		hashKey:                config.HashKey,
	}
	manager.userResolver = manager.resolveUserThroughGuard

	return manager
}

// Guard is the guard of that name, from the cache or newly resolved.
//
// An empty name means the default guard. A guard that is not configured is an
// error.
func (m *AuthManager) Guard(name string) (Guard, error) {
	if name == "" {
		name = m.GetDefaultDriver()
	}

	m.mu.Lock()
	cached, ok := m.guards[name]
	m.mu.Unlock()

	if ok {
		return cached, nil
	}

	guard, err := m.resolve(name)
	if err != nil {
		return nil, err
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	// Somebody else may have resolved the same name while this one was being
	// built. Theirs wins, so that everyone in the request holds one guard.
	if existing, ok := m.guards[name]; ok {
		return existing, nil
	}
	m.guards[name] = guard

	return guard, nil
}

// resolve builds the guard of that name from its configured driver.
//
// The two built-in drivers are a switch, not a method found by name: a driver
// that is neither is registered with [AuthManager.Extend].
func (m *AuthManager) resolve(name string) (Guard, error) {
	config, ok := m.guardConfigFor(name)
	if !ok {
		return nil, fmt.Errorf("auth: guard [%s] is not defined", name)
	}

	driver := configString(config, "driver")

	if creator, ok := m.customCreator(driver); ok {
		return m.callCustomCreator(creator, name, config)
	}

	switch driver {
	case "session":
		return m.CreateSessionDriver(name, config)
	case "token":
		return m.CreateTokenDriver(name, config)
	}

	return nil, fmt.Errorf("auth: driver [%s] for guard [%s] is not defined", driver, name)
}

// callCustomCreator runs a registered driver creator, handing it the manager so
// that it can build what it needs from the same configuration.
func (m *AuthManager) callCustomCreator(
	creator func(manager *AuthManager, name string, config map[string]any) (Guard, error),
	name string,
	config map[string]any,
) (Guard, error) {
	return creator(m, name, config)
}

// CreateSessionDriver builds a [SessionGuard] from the guard's settings.
func (m *AuthManager) CreateSessionDriver(name string, config map[string]any) (*SessionGuard, error) {
	provider, err := m.CreateUserProvider(configString(config, "provider"))
	if err != nil {
		return nil, err
	}

	guard := NewSessionGuard(
		name,
		provider,
		m.session,
		m.request,
		nil,
		m.rehashOnLogin,
		m.timeboxDuration,
		m.hashKey,
	)

	guard.Hasher = m.hasher

	// The cookie jar is what lets the guard write an encrypted "remember me"
	// cookie; the dispatcher is what lets anything listen to the sign-in.
	guard.SetCookieJar(m.cookies)

	guard.SetDispatcher(m.events)

	if remember, ok := configInt(config, "remember"); ok {
		guard.SetRememberDuration(remember)
	}

	return guard, nil
}

// CreateTokenDriver builds a [TokenGuard] from the guard's settings.
func (m *AuthManager) CreateTokenDriver(name string, config map[string]any) (*TokenGuard, error) {
	provider, err := m.CreateUserProvider(configString(config, "provider"))
	if err != nil {
		return nil, err
	}

	// The token guard is the basic API token guard: it takes a token field off
	// the request and matches it against the users wherever they are kept. It
	// keeps no name of its own, so the argument goes unread.
	_ = name

	return NewTokenGuard(
		provider,
		m.request,
		configString(config, "input_key"),
		configString(config, "storage_key"),
		configBool(config, "hash"),
	), nil
}

// GetDefaultDriver is the guard an empty name means.
func (m *AuthManager) GetDefaultDriver() string {
	m.mu.Lock()
	defer m.mu.Unlock()

	return m.defaultGuard
}

// ShouldUse makes this the default guard, and points the user resolver back at
// whatever the default is.
func (m *AuthManager) ShouldUse(name string) {
	if name == "" {
		name = m.GetDefaultDriver()
	}

	m.SetDefaultDriver(name)

	m.mu.Lock()
	defer m.mu.Unlock()

	m.userResolver = m.resolveUserThroughGuard
}

// SetDefaultDriver sets the guard an empty name means.
func (m *AuthManager) SetDefaultDriver(name string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.defaultGuard = name
}

// ViaRequest registers a driver that is one callback, built as a
// [RequestGuard].
func (m *AuthManager) ViaRequest(driver string, callback func(request Request, provider UserProvider) Authenticatable) *AuthManager {
	return m.Extend(driver, func(manager *AuthManager, name string, config map[string]any) (Guard, error) {
		_ = name
		_ = config

		provider, err := manager.CreateUserProvider("")
		if err != nil {
			return nil, err
		}

		return NewRequestGuard(callback, manager.request, provider), nil
	})
}

// UserResolver is the callback that answers who is acting, which the Gate and
// the request both use.
func (m *AuthManager) UserResolver() func(guard string) Authenticatable {
	m.mu.Lock()
	defer m.mu.Unlock()

	return m.userResolver
}

// ResolveUsersUsing replaces that callback, and returns the manager.
func (m *AuthManager) ResolveUsersUsing(userResolver func(guard string) Authenticatable) *AuthManager {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.userResolver = userResolver

	return m
}

// Extend registers a custom guard driver.
//
// The manager is the callback's first argument, so that the driver can build
// what it needs from the same configuration.
func (m *AuthManager) Extend(driver string, callback func(manager *AuthManager, name string, config map[string]any) (Guard, error)) *AuthManager {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.customCreators[driver] = callback

	return m
}

// Provider registers a custom user provider creator.
func (m *AuthManager) Provider(name string, callback func(config map[string]any) (UserProvider, error)) *AuthManager {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.customProviderCreators[name] = callback

	return m
}

// HasResolvedGuards reports that at least one guard has been resolved.
func (m *AuthManager) HasResolvedGuards() bool {
	m.mu.Lock()
	defer m.mu.Unlock()

	return len(m.guards) > 0
}

// ForgetGuards drops every resolved guard.
//
// It is what a long-running process calls between requests, because a guard
// remembers the user it resolved.
func (m *AuthManager) ForgetGuards() *AuthManager {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.guards = map[string]Guard{}

	return m
}

// CreateUserProvider is the provider of that name, or the default one when the
// name is empty.
//
// There are no built-in drivers: DatabaseUserProvider and ModelUserProvider
// are hesape/auth/users, and the root of auth imports nothing outside the
// standard library (see doc.go). Register them with [AuthManager.Provider].
//
// A provider that is not configured at all is nil and no error: a guard may
// have no provider.
func (m *AuthManager) CreateUserProvider(provider string) (UserProvider, error) {
	config, ok := m.providerConfiguration(provider)
	if !ok {
		return nil, nil
	}

	driver := configString(config, "driver")

	if creator, ok := m.customProviderCreator(driver); ok {
		return creator(config)
	}

	return nil, fmt.Errorf("auth: user provider [%s] is not defined", driver)
}

// GetDefaultUserProvider is the provider an empty name means.
func (m *AuthManager) GetDefaultUserProvider() string {
	m.mu.Lock()
	defer m.mu.Unlock()

	return m.defaultProvider
}

// providerConfiguration is the settings of the named provider, or of the
// default one, and whether there are any.
func (m *AuthManager) providerConfiguration(provider string) (map[string]any, bool) {
	if provider == "" {
		provider = m.GetDefaultUserProvider()
	}
	if provider == "" {
		return nil, false
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	config, ok := m.providerConfig[provider]

	return config, ok && config != nil
}

// guardConfigFor is the settings of the named guard, and whether there are any.
func (m *AuthManager) guardConfigFor(name string) (map[string]any, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()

	config, ok := m.guardConfig[name]

	return config, ok && config != nil
}

// customCreator reads the driver registry [AuthManager.Extend] writes.
func (m *AuthManager) customCreator(driver string) (func(manager *AuthManager, name string, config map[string]any) (Guard, error), bool) {
	m.mu.Lock()
	defer m.mu.Unlock()

	creator, ok := m.customCreators[driver]

	return creator, ok
}

// customProviderCreator reads the registry [AuthManager.Provider] writes.
func (m *AuthManager) customProviderCreator(driver string) (func(config map[string]any) (UserProvider, error), bool) {
	m.mu.Lock()
	defer m.mu.Unlock()

	creator, ok := m.customProviderCreators[driver]

	return creator, ok
}

// resolveUserThroughGuard is the user resolver [NewAuthManager] and ShouldUse
// both install: the user of the named guard, or of the default one.
func (m *AuthManager) resolveUserThroughGuard(guard string) Authenticatable {
	resolved, err := m.Guard(guard)
	if err != nil {
		return nil
	}
	return resolved.User()
}

// configString reads a string out of a settings map, and answers "" when the
// key is missing or holds something else.
func configString(config map[string]any, key string) string {
	value, _ := config[key].(string)
	return value
}

// configBool reads a bool out of a settings map.
func configBool(config map[string]any, key string) bool {
	value, _ := config[key].(bool)
	return value
}

// configInt reads an int out of a settings map, and says whether it was there.
// Settings read from JSON or YAML hold numbers as float64, so both are taken.
func configInt(config map[string]any, key string) (int, bool) {
	switch value := config[key].(type) {
	case int:
		return value, true
	case int64:
		return int(value), true
	case float64:
		return int(value), true
	}
	return 0, false
}
