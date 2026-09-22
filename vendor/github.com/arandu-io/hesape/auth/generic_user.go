package auth

// GenericUser is an [Authenticatable] that is nothing but the row it was built
// from.
//
// It is what a user provider hands back when there is no model to hydrate --
// the database provider's every result is one of these. An application with its
// own user type does not use it: that type satisfies Authenticatable by having
// the seven methods.
type GenericUser struct {
	// Attributes is the row, column for column.
	//
	// It is exported so that the columns beyond the seven the contract names are
	// reachable at all: read one with [GenericUser.Get] or by indexing, write one
	// by assigning, ask with the two-value index, drop one with delete.
	Attributes map[string]any
}

var _ Authenticatable = (*GenericUser)(nil)

// NewGenericUser returns a user over the row, and over an empty map when the
// row is nil.
func NewGenericUser(attributes map[string]any) *GenericUser {
	if attributes == nil {
		attributes = map[string]any{}
	}
	return &GenericUser{Attributes: attributes}
}

// GetAuthIdentifierName is the column holding the id: "id".
func (u *GenericUser) GetAuthIdentifierName() string {
	return "id"
}

// GetAuthIdentifier is the id itself.
func (u *GenericUser) GetAuthIdentifier() any {
	return u.Attributes[u.GetAuthIdentifierName()]
}

// GetAuthIdentifierForBroadcasting is the identifier itself. A user type that
// publishes a different id on a channel answers otherwise; see
// [BroadcastsIdentifier] for why that is worth doing.
func (u *GenericUser) GetAuthIdentifierForBroadcasting() any {
	return u.GetAuthIdentifier()
}

// GetAuthPasswordName is the column holding the hash: "password".
func (u *GenericUser) GetAuthPasswordName() string {
	return "password"
}

// GetAuthPassword is the hash, never the plain password.
//
// The contract types it as a string and the map holds any, so a column that is
// not a string reads as "".
func (u *GenericUser) GetAuthPassword() string {
	return u.stringAttribute(u.GetAuthPasswordName())
}

// GetRememberToken is the token the "remember me" cookie is checked against.
func (u *GenericUser) GetRememberToken() string {
	return u.stringAttribute(u.GetRememberTokenName())
}

// SetRememberToken writes that token onto the row.
func (u *GenericUser) SetRememberToken(value string) {
	u.Attributes[u.GetRememberTokenName()] = value
}

// GetRememberTokenName is the column holding it: "remember_token".
func (u *GenericUser) GetRememberTokenName() string {
	return "remember_token"
}

// Get is one column, whatever type it came back as.
//
// A column that is not there is nil.
func (u *GenericUser) Get(key string) any {
	return u.Attributes[key]
}

// stringAttribute reads a column the contract types as a string.
func (u *GenericUser) stringAttribute(key string) string {
	value, _ := u.Attributes[key].(string)
	return value
}
