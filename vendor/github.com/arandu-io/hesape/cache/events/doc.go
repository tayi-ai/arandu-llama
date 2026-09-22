// Package events holds every event a cache Repository fires.
//
// A listener written against events.CacheHit reads e.Key, e.Value and
// e.StoreName; the other events carry the fields their names imply, and
// CacheEvent carries the ones they all share.
//
// They are values, not an interface. Nothing here dispatches: cache.Dispatcher
// is the one method the Repository needs, and what is on the other side of it
// is the application's business.
package events
