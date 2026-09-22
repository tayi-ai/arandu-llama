// Package relations holds the sixteen relation types, and the eager loading that
// keeps them from being N+1 queries.
//
// The thirteen walking methods BelongsToMany and HasOneOrManyThrough each expose
// -- Chunk, ChunkByID, ChunkByIDDesc, OrderedChunkByID, Each, EachByID, Lazy,
// LazyByID, LazyByIDDesc, Cursor, Paginate, SimplePaginate and CursorPaginate --
// have one body between them, in chunking.go. The only line that differs is
// BelongsToMany hydrating the pivot on each page, which is a field of the shared
// body rather than a second copy of it.
//
// # The four methods
//
// AddEagerConstraints, InitRelation, Match and GetEager are what turn a hundred
// queries into two. One query fetches the parents; AddEagerConstraints widens
// the relation's where into a single `in (...)` over every parent key;
// InitRelation seeds each parent so that one with no children answers empty
// instead of going back to the database; Match buckets the flat result set by
// key and hands each parent its own. EagerLoadRelation is the loop, and the
// test for it counts the statements the connection was asked for -- because the
// values come out right either way, which is why an N+1 passes review.
//
// # Every read takes the Grant, and every read filters by tenant
//
// GetResults, Get, First, GetEager, Attach, Detach and Sync all take a context
// and an auth.Grant, and every statement is narrowed to auth.Tenant(g) before it
// leaves -- on the eager path as much as the lazy one. The eager path is where
// it matters most: a parent query that is correctly scoped and a child query
// that is not returns the right parents carrying another customer's children,
// and every row on the screen looks like it belongs there. A Grant carrying no
// tenant is refused rather than compiled into a comparison with the empty
// string.
//
// The query builder decides no such thing -- it builds SQL. What is enforced
// here is that a relation cannot be executed without the Grant that authorized
// it, because the Grant is in the signature.
//
// # The morph map is mandatory here, and that is the trade being bought
//
// There is no type resolved from a name at run time in Go, so a *_type column
// holds an alias registered with MorphMap. The type it names can then be
// renamed, moved or split without a single stored row becoming unreadable. An
// unregistered alias is an error that says which alias and what is registered.
//
// # Construction and overrides
//
// An initialism is upper case: ParseIDs, ThroughKey. A shared behaviour is a
// struct to embed, and the parts a subtype must supply arrive as function
// fields.
//
// The one shape that had to move is virtual dispatch. Go embedding does not
// dispatch to the outer type, so each concrete constructor calls its own
// AddConstraints as its last statement, and an override the shared half needs to
// reach -- the aliased pivot columns, the one-of-many relation query -- is a
// field the subtype sets rather than a method it redeclares.
package relations
