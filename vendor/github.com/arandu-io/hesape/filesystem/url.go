package filesystem

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"strings"
	"time"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/encryption"
)

// Presigner is the optional half of an [Adapter] that can hand out a URL
// pointing at the store itself.
//
// Object stores can; a directory on disk cannot. It is an optional interface
// and not a method on Adapter for that reason: a driver that cannot presign
// would otherwise have to carry a method that returns an error nobody can act
// on. [URLSigner.TemporaryURL] asks, and answers the same question either way,
// so no caller has to know which kind of disk it is holding.
//
// The path it receives is already resolved by [Key]: a presigned URL is a
// bearer token for one object, and the object is the tenant's.
type Presigner interface {
	PresignGet(ctx context.Context, path string, ttl time.Duration) (string, error)
}

// ErrBadURL is returned by [URLSigner.Redeem] for a link that was not issued
// here, was tampered with, or has run out. It unwraps encryption.ErrSignature,
// so a caller that wants to tell "expired" apart can still ask for
// encryption.ErrExpired.
var ErrBadURL = errors.New("filesystem: the link is not valid")

// urlPurpose is what these tokens are signed for. A token issued for a file
// does not work on a password reset, and the reverse, because the purpose is
// part of the signature.
const urlPurpose = "filesystem.url"

// URLSigner issues links that stand in for a session.
//
// It exists for the case a route cannot cover: an <img> tag, a download link in
// an e-mail, a file handed to a program that has no cookies. The link is proof
// that a Policy said yes at the moment it was made, it names one file, and it
// expires.
//
// It is not a way around authorization. TemporaryURL takes a Grant, which means
// a Policy has already run; the signature carries the decision instead of
// repeating it, exactly as a signed verification link carries the decision to
// let somebody confirm an address.
type URLSigner struct {
	signer *encryption.Signer
	base   string
}

// NewURLSigner returns a signer whose fallback links point under base.
//
// base is the path the redeeming route is mounted at -- "/files", say, for a
// route that reads the last segment and calls [Redeem] then [Serve]. It is
// unused when the disk can presign, because then the URL comes from the store.
func NewURLSigner(s *encryption.Signer, base string) *URLSigner {
	return &URLSigner{signer: s, base: strings.TrimSuffix(base, "/")}
}

// TemporaryURL returns a URL that serves one file for ttl and then stops.
//
// When the disk's adapter is a [Presigner], the store issues it and the bytes
// never pass through the application. When it is not, the link points back at
// base and carries a signed token that [Redeem] reads. Both are one call,
// because a caller choosing between them is a caller who gets it wrong on the
// disk that changed driver.
func (u *URLSigner) TemporaryURL(ctx context.Context, g auth.Grant, d *Disk, key string, ttl time.Duration) (string, error) {
	if ttl <= 0 {
		return "", fmt.Errorf("filesystem: a temporary URL needs a lifetime, and %s is not one", ttl)
	}
	full, err := Key(g, key)
	if err != nil {
		return "", err
	}
	if p, ok := d.GetAdapter().(Presigner); ok {
		link, err := p.PresignGet(ctx, full, ttl)
		if err != nil {
			return "", d.wrap("presign", key, err)
		}
		return link, nil
	}
	// url.Values and not a separator: a key may contain anything CleanKey
	// allows, including whatever character would otherwise have been the
	// delimiter, and a payload that can be re-read two ways is a payload an
	// attacker picks the reading of.
	payload := url.Values{
		"a": {string(g.Action())},
		"t": {auth.Tenant(g)},
		"k": {key},
		"d": {d.Name()},
	}.Encode()
	return u.base + "/" + u.signer.Sign(urlPurpose, payload, ttl), nil
}

// Redeem reads a token issued by [TemporaryURL] and returns what it authorizes.
//
// The Grant it returns is for this action, in this tenant, and it is meant for
// the key returned beside it -- pass the two together to [Serve] and to nothing
// else. It is a system grant, so it carries no person: a link is not a session,
// and treating it as one would mean a download link could be used to write.
//
// The disk name travels in the token so the redeeming route does not have to
// guess which disk a link was made against; ask [Disks.Disk] for it.
func (u *URLSigner) Redeem(token string) (grant auth.Grant, disk, key string, err error) {
	payload, err := u.signer.Verify(urlPurpose, token)
	if err != nil {
		return auth.Grant{}, "", "", fmt.Errorf("%w: %w", ErrBadURL, err)
	}
	values, err := url.ParseQuery(payload)
	if err != nil {
		return auth.Grant{}, "", "", fmt.Errorf("%w: the payload is unreadable", ErrBadURL)
	}
	action, tenant, key := values.Get("a"), values.Get("t"), values.Get("k")
	if action == "" || tenant == "" || key == "" {
		return auth.Grant{}, "", "", fmt.Errorf("%w: the payload names no file", ErrBadURL)
	}
	g := auth.SystemGrant(auth.Action(action), tenant)
	// A tenant that was valid when the link was issued and is not now yields
	// the zero Grant, which reaches no file. Saying so here is better than
	// handing back a Grant that fails later with no mention of the link.
	if err := g.Check(auth.Action(action)); err != nil {
		return auth.Grant{}, "", "", fmt.Errorf("%w: %w", ErrBadURL, err)
	}
	return g, values.Get("d"), key, nil
}
