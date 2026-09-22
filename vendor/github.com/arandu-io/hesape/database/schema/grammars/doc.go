// Package grammars turns a schema.Blueprint into statements, one grammar per
// engine: MySQLGrammar, PostgresGrammar and SQLiteGrammar over BaseGrammar.
//
// A grammar executes nothing: schema.Builder is the half that runs what is
// compiled here. Neither asks for an authorization credential, because DDL names
// a table rather than rows -- there is no tenant column to scope by and no
// subject to attribute a statement to.
//
// # What the three do not agree on
//
// The column type is where they diverge most, and the divergence is not
// cosmetic:
//
//	string    varchar(255)   varchar(255)              varchar
//	text      text           text                      text
//	json      json           json                      text (or json)
//	jsonb     json           jsonb                     text (or jsonb)
//	boolean   tinyint(1)     boolean                   tinyint(1)
//	binary    blob           bytea                     blob
//	uuid      char(36)       uuid                      varchar
//	increment auto_increment serial / bigserial        primary key autoincrement
//
// SQLite has five storage classes and no varchar length, which is why a
// migration that works there and nowhere else is the failure the conformance
// suite exists to catch. Its json and jsonb depend on the use_native_json and
// use_native_jsonb connection options.
//
// MySQLGrammar serves MariaDB as well: the two types it spells differently are
// reachable from Connection.IsMaria, so there is no second grammar. There is no
// SQL Server grammar -- it is a driver this ecosystem does not carry.
//
// # How a driver grammar extends the base
//
// Go embedding does not dispatch back to the outer type, so each driver grammar
// hands itself to BaseGrammar at construction and the base calls back through an
// unexported dialect interface -- typeOf, modify and wrapValue. Everything the
// base cannot spell on its own returns an error wrapping schema.ErrUnsupported,
// so a driver that does not override it reports the same refusal.
package grammars
