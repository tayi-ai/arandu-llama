package session

// Encrypter is the part of an encrypter [EncryptedStore] uses.
//
// It is declared here, minimally, rather than imported, so that a project that
// does not encrypt its sessions does not carry the cipher package behind them.
// *encryption.Encrypter satisfies it unchanged.
type Encrypter interface {
	// EncryptString returns the sealed payload.
	EncryptString(value string) (string, error)
	// DecryptString opens it, and reports an error for a payload that was not
	// sealed with this key.
	DecryptString(payload string) (string, error)
}

// EncryptedStore is a [Store] whose payload is sealed before it reaches the
// handler.
//
// It is what makes [CookieSessionHandler] usable at all: with that handler the
// whole session travels in the browser, and unencrypted that means the person
// holding the machine can read it and -- absent a signature -- write it. It is
// also worth having over a shared cache or a database somebody else operates.
//
// It is not a substitute for a session id nobody can guess. Encrypting the
// payload does not stop somebody who stole the cookie from using it: that is
// what the expiry and [Store.Regenerate] are for.
type EncryptedStore struct {
	*Store
	encrypter Encrypter
}

// NewEncryptedStore returns a session whose payload is encrypted with the
// application key.
//
// The arguments are [NewStore]'s with the encrypter added.
func NewEncryptedStore(name string, handler SessionHandler, encrypter Encrypter, id string) *EncryptedStore {
	store := NewStore(name, handler, id)
	s := &EncryptedStore{Store: store, encrypter: encrypter}

	// The two hooks, replaced here rather than shadowed. See the fields on
	// [Store] for why Go leaves no other way to do this.
	store.prepareForStorage = func(data string) (string, error) { return encrypter.EncryptString(data) }
	store.prepareForUnserialize = func(data string) (string, error) { return encrypter.DecryptString(data) }
	return s
}

// GetEncrypter returns the encrypter underneath.
func (s *EncryptedStore) GetEncrypter() Encrypter { return s.encrypter }
