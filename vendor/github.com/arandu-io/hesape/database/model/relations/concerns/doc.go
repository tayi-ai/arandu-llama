// Package concerns holds the shared halves the sixteen relation types are
// assembled from: AsPivot, CanBeOneOfMany, ComparesRelatedModels,
// InteractsWithDictionary, InteractsWithPivotTable, SupportsDefaultModels and
// SupportsInverseRelations.
//
// # Each one is a struct you embed, and what it needs back arrives as a field
//
// A relation embeds the struct, which gives it the fields and the promoted
// methods, and supplies what the shared half has to call back into as a function
// field it sets when it builds itself: ComparesRelatedModels.ParentKey,
// SupportsDefaultModels.NewRelatedInstanceFor, CanBeOneOfMany.AddConstraints.
//
// # Why Model and Builder are declared here
//
// Model and Builder are the surfaces a relation asks of the model package.
// They are declared in this leaf package rather than in relations because Go
// forbids a subpackage from importing its parent, and both packages have to
// agree on one type. relations aliases them straight back, so relations.Model
// and concerns.Model are the same type.
//
// # Everything that runs takes the Grant
//
// Attach, Detach, Sync, Toggle, UpdateExistingPivot and every read behind them
// take a context and an auth.Grant, and every statement they build is filtered
// by the tenant that Grant carries -- including the INSERT, which stamps it. The
// pivot table is where it is easiest to forget: the query that says which roles
// a user has is two joins away from the policy that authorized the user.
package concerns
