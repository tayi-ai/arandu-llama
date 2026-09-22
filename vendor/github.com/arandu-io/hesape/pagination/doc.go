// Package pagination turns a page of rows into the values a pager renders from.
//
// It holds three paginators, one per question a list page can answer:
//
//   - [LengthAwarePaginator], from [Paginate]: knows the total, so it can say
//     "page 3 of 47" and print numbered links. It costs one COUNT query.
//   - [Paginator], from [SimplePaginate]: knows only whether one more row
//     exists, so it can say "previous" and "next" and nothing else. It costs
//     one extra row.
//   - [CursorPaginator], from [CursorPaginate]: keyset paging, forward and
//     backward, over a signed [Cursor]. It has no total and no page number,
//     and it is the only one of the three that does not skip or repeat rows
//     when the data changes underneath.
//
// # What a paginator is here
//
// A paginator is a value: the items of one page, plus the arithmetic needed to
// write the URL of the pages around it. It runs no query, holds no database
// handle and knows no Grant. A repository reads the rows -- through a Policy,
// like every other read -- and hands them here.
//
// That division is why this package is a leaf. It imports the standard library
// and the application key, and every dependency runs the other way: the
// repository knows about pagination, pagination knows nothing about
// repositories.
//
// # The cursor is signed
//
// A page number is a number, and a reader who edits one reaches a page they
// could have reached by typing it. A cursor is the boundary row of a page, sent
// to the client and handed back, so a reader who edits one names the row the
// next query starts at. [CursorSigner] is what closes that: it signs the token
// on the way out and checks it on the way in, and it is the only way a [Cursor]
// becomes a string.
//
// # No global resolvers
//
// The request is read explicitly. [OptionsFrom] takes the *url.URL and returns
// the [Options] every constructor takes, and [ResolveCurrentPage],
// [ResolveCurrentPath], [ResolveQueryString] and [ResolveCurrentCursor] read
// the four pieces out of it -- the last of them against the [CursorSigner], the
// cursor being checked before it is read. There is nothing to install and
// nothing to install it from.
//
// # The views are names here, and components in the view layer
//
// [DefaultView], [DefaultSimpleView], [UseTailwind], [UseBootstrapFive] and
// their siblings name a pager, and a name is all they are. Nothing in this
// package writes HTML: rendering belongs to the view layer.
//
// What a component renders from is [LengthAwarePaginator.Links]: a flat []Link
// holding the numbered pages and a [Separator] wherever the window left pages
// out, in the order a pager draws them. [LengthAwarePaginator.LinkCollection]
// is the same list with previous and next at its ends, and is what the JSON
// payload carries.
//
// # No interfaces
//
// There is no Paginator interface, on purpose: an interface belongs in the
// package that consumes it, and until something consumes one, declaring it here
// would only be a name to keep in step with the structs.
//
// # Naming
//
// An initialism is spelled in one case, so [LengthAwarePaginator.URL] and
// [LengthAwarePaginator.ToJSON]. Three names are functions rather than methods,
// because a Go method cannot introduce a type parameter and the point of these
// three is to change the element type: [Through], [ThroughSimple] and
// [ThroughCursor]. One name each, because Go cannot overload.
package pagination
