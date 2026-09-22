// Package model holds the Model, its query Builder, its Collection and soft
// deletes.
//
// # There is no dynamic attribute, and that is the whole design
//
// A model here is the application's own struct, and the machinery works over it
// with a type parameter:
//
//	type User struct {
//		model.Model[User]
//
//		ID        int64      `db:"id"`
//		Name      string     `db:"name"`
//		Email     string     `db:"email"`
//		CreatedAt time.Time  `db:"created_at"`
//		UpdatedAt time.Time  `db:"updated_at"`
//		DeletedAt *time.Time `db:"deleted_at"`
//	}
//
//	users := model.Query[User](db)
//
//	user, err := users.Where("email", "=", email).First(ctx, g)
//	if err != nil {
//		return err
//	}
//	user.Name = "Ada"        // a struct field, checked by the compiler
//	_, err = user.Save(ctx, g)
//
// # The row is the model
//
// A terminal hands back *T -- the application's own struct -- and not a wrapper
// over it. Reading a column is reading a field, and the methods a row is saved
// and deleted through are the ones Go promotes out of the embedded Model[T].
// Collection is the same thing for a set of rows: []*T.
//
// The embedding is what makes the second half work, and it is worth saying what
// a T that does not embed Model[T] gets instead. Everything still runs: the
// query, the hydration, the eager load, the write. What that row cannot do is
// answer for itself -- there is no field in it pointing at the model that
// hydrated it, so Save, GetAttribute and the loaded relations are not reachable
// from the value a terminal returned. The columns are all there, and nothing
// else is. See ModelOf.
//
// A column is a field. The tag `db:"..."` names it; without a tag the name is
// the field name in snake case. A field tagged `db:"-"` is not a column.
//
// # Reading a relation back, for each of the two shapes
//
// With marks a relation to eager load, and the terminal attaches what it matched
// to the model behind each row. Reading it back takes the row:
//
//	users, err := model.Query[User](db).With("posts").Get(ctx, g)
//	posts, ok := model.Related[User, Post](users[0], "posts")
//
// and loading one afterwards is Load, promoted out of the embedded model:
//
//	err := users[0].Load(ctx, g, "posts")
//
// Both of those reach the model through the row, so both need a T that embeds
// Model[T]. A T that does not still eager loads -- the query runs, the rows are
// matched, the relation is attached -- but it is attached to a model beside the
// row rather than inside it, and no terminal hands that model back. Related
// answers false there, and Collection.Load, Collection.LoadMissing,
// Collection.LoadAggregate and Builder.EagerLoadRelations report
// ErrRowHasNoModel rather than reporting success having loaded onto nothing.
// The way to read a relation off a row is to give the entity the embedded model.
//
// Query is the entry point when the defaults are right: it works the table out
// of the type and takes the grammar and the processor off the connection.
// NewModel is the one to reach for when they are not -- another table, another
// key, soft deletes, a table with no tenant column -- and the model it returns
// answers NewQuery.
//
// # There is no mass-assignment allowlist, and nothing replaces one
//
// The allowlist already exists and the compiler enforces it: reflection cannot
// write an unexported field, from any package, ever.
//
//	type User struct {
//		Name  string `db:"name"`  // Fill sets this
//		admin bool   `db:"admin"` // nothing outside the package can, at all
//	}
//
// So Fill writes the exported fields it finds and drops keys it does not know.
// ForceFill keeps the unknown keys as raw attributes instead. Neither can reach
// an unexported field.
//
// The cost is stated plainly: an unexported field is not a column at all, so it
// is not persisted either. A value that must be stored but must never come from
// a request is an exported field the caller does not put in the map -- in this
// framework a request never becomes a model, it becomes a validated struct
// first.
//
// # Every read carries the Grant
//
// All, Find, First, Get, Value, Pluck, Paginate, Chunk and Cursor take an
// auth.Grant and filter by auth.Tenant(g), exactly as Insert, Update and Delete
// do. A query builder reached without a Grant compiles SQL and cannot run it, on
// a read as much as on a write.
//
// What the Grant settles here is the tenant, not the Policy. auth.Authorize is
// the path a Policy answered on, and it is not the only exported way to obtain a
// Grant: auth.SystemGrant issues one for work that has no subject. So these
// methods can be reached holding a Grant nothing was ever asked about, and what
// reports that is `aru doctor` -- a lint, not the type system.
//
// The tenant filter is on by default and comes off only by naming it: a model
// whose table has no tenant column sets TenantColumn to the empty string, in
// its constructor, where a reader sees it. A Grant carrying no tenant is refused
// with ErrNoTenant before any SQL is built, and ErrNoTenant names which Grants
// those are.
//
// The tenant written on insert and matched on select is always auth.Tenant(g).
// A value the caller put in the struct for that column is overwritten: the
// tenant comes from the Grant and from nowhere else.
//
// # What the signatures do NOT carry, and why
//
// There is no context.Context. The connection contract in this component is
// query.Connection, which takes none; declaring a second, context-carrying
// connection interface here would be a second way to reach the database. A
// signature that accepted a context it could not pass on would be worse than one
// that does not accept it.
//
// # Column order on insert
//
// Values reach the grammar as a map, and a Go map has no order, so columns and
// their bindings go in sorted order on every insert. The grammar must sort
// identically -- both sides sort by column name, which is the only ordering
// either side can derive from the values alone.
//
// # What is not here
//
// Relations live in model/relations. The Relation interface in relation.go is
// what this package asks of one, and it is the contract declared there plus the
// one method the builder needs that the tree does not put in its own interface.
// It is not a second contract: a relation satisfies it by being a relation, and
// the constructors in relationsof.go are how an application builds one.
//
// A relation is registered by name in Model.RelationResolvers, in its
// Unconstrained form -- the builder resolves it from the model it queries
// through, which carries no key, and narrows it to the batch. relationsof.go
// says why the split exists.
//
// There is no automatic eager loading. Reading an unloaded relation would run a
// query behind the caller, and that query carries no auth.Grant.
// PreventLazyLoading is what is left of the pair, and its doc comment says what
// it means here.
package model
