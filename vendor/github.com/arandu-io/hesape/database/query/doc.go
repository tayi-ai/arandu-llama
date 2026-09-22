// Package query builds SQL: the Builder, the raw Expression, the index hint and
// the join clause.
//
// Builder is split across files by what each half does: builder.go holds the
// state and the clauses everything else builds on, wheres.go and havings.go the
// two filter vocabularies, joins.go and subqueries.go the join and subquery
// families, aggregates.go the counting and the pagination, chunk.go the walks,
// write.go the statements that change rows, execute.go the ones that read them,
// and misc.go the rest.
//
// # Nothing here decides who may run what
//
// A Builder compiles SQL. The tenant of the security.Grant is put on the
// statement at execution, in scoped, and on every subquery the statement carries
// -- a union, a `where exists`, a from, a join or a select subquery. That is not
// optional: a subquery is compiled whole, so a filter on the outer query says
// which of its rows survive and nothing about which rows went in.
//
// A lateral join is JoinClause.Lateral rather than a type of its own, so the
// grammar has one type to hold. Dumping a query is Builder.Dump and
// Builder.DumpRawSQL, and neither ends the process.
package query
