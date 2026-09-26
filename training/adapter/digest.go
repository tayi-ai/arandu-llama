package adapter

import (
	"crypto/sha256"
	"encoding/hex"
)

// Digest returns the SHA-256 of an adapter's serialized bytes.
func Digest(body []byte) string {
	sum := sha256.Sum256(body)
	return hex.EncodeToString(sum[:])
}
