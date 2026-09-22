package migrations

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/arandu-io/hesape/str"
)

// MigrationCreator is what `aru make:migration` writes the file with.
//
// It fills a stub: the name takes a date prefix, the stub is chosen by whether a
// table was named and whether it is being created, a custom stub directory wins
// over the built-in one, and the post-create hooks run afterwards.
type MigrationCreator struct {
	// customStubPath is a project directory whose stubs win over the
	// built-in ones.
	customStubPath string

	// postCreate is the hooks AfterCreate registered.
	postCreate []func(table, path string)

	// now is where the date prefix comes from. A variable is what makes the
	// prefix testable without freezing the process clock.
	now func() time.Time
}

// NewMigrationCreator builds a MigrationCreator.
//
// It holds no filesystem: the file operations involved are two calls to the
// standard library.
func NewMigrationCreator(customStubPath string) *MigrationCreator {
	return &MigrationCreator{customStubPath: customStubPath, now: time.Now}
}

// Create writes the migration and returns the path it was written to.
//
// table empty means a migration with no table in mind. create says whether
// the named table is being created or altered, which is the difference
// between the two table-shaped stubs.
func (c *MigrationCreator) Create(name, path, table string, create bool) (string, error) {
	if err := c.ensureMigrationDoesntAlreadyExist(name, path); err != nil {
		return "", err
	}

	stub, err := c.getStub(table, create)
	if err != nil {
		return "", err
	}

	fullPath := c.GetPath(name, path)

	if err := os.MkdirAll(filepath.Dir(fullPath), 0o755); err != nil {
		return "", fmt.Errorf("creating the migration directory: %w", err)
	}

	if err := os.WriteFile(fullPath, []byte(c.populateStub(stub, name, table)), 0o644); err != nil {
		return "", fmt.Errorf("writing %s: %w", fullPath, err)
	}

	c.firePostCreateHooks(table, fullPath)

	return fullPath, nil
}

// ensureMigrationDoesntAlreadyExist fails when a file for name already
// exists in migrationPath.
//
// Two migrations declaring the same type is a build failure Go catches on
// its own, so the check that is left is the one the compiler cannot make: a
// second file whose name differs only by its date prefix, which compiles
// and then applies twice under two names.
func (c *MigrationCreator) ensureMigrationDoesntAlreadyExist(name, migrationPath string) error {
	if migrationPath == "" {
		return nil
	}

	entries, err := os.ReadDir(migrationPath)
	if err != nil {
		// A directory that is not there yet is not a conflict; Create makes it.
		return nil
	}

	suffix := "_" + name + ".go"
	for _, entry := range entries {
		if strings.HasSuffix(entry.Name(), suffix) {
			return fmt.Errorf("a %s migration already exists, in %s", name, filepath.Join(migrationPath, entry.Name()))
		}
	}
	return nil
}

// getStub returns the custom stub when the project has one, the built-in
// otherwise.
func (c *MigrationCreator) getStub(table string, create bool) (string, error) {
	var name, builtin string
	switch {
	case table == "":
		name, builtin = "migration.stub", blankStub
	case create:
		name, builtin = "migration.create.stub", createStub
	default:
		name, builtin = "migration.update.stub", updateStub
	}

	if c.customStubPath != "" {
		custom := filepath.Join(c.customStubPath, name)
		if contents, err := os.ReadFile(custom); err == nil {
			return string(contents), nil
		}
	}
	return builtin, nil
}

// populateStub fills in the placeholders a stub carries.
//
// It replaces DummyTable, {{ table }} and {{table}} when a table is given,
// plus two a Go stub needs: the type name and the migration's own name,
// which is its identity here because there is no file path to read it back
// off.
func (c *MigrationCreator) populateStub(stub, name, table string) string {
	replacements := []string{
		"{{ class }}", c.GetClassName(name),
		"{{class}}", c.GetClassName(name),
		"{{ name }}", c.migrationName(name),
		"{{name}}", c.migrationName(name),
	}
	if table != "" {
		replacements = append(replacements,
			"DummyTable", table,
			"{{ table }}", table,
			"{{table}}", table,
		)
	}
	return strings.NewReplacer(replacements...).Replace(stub)
}

// migrationName is the identity the generated migration answers from GetName:
// the date prefix and the name, which is exactly the file's base name.
func (c *MigrationCreator) migrationName(name string) string {
	return c.GetDatePrefix() + "_" + name
}

// GetClassName returns the Go type name a migration file for name would
// declare.
//
// It is exported here because a caller with no filesystem access -- a test,
// a generator that prints rather than writes -- has no other way to ask.
func (c *MigrationCreator) GetClassName(name string) string { return str.Studly(name) }

// GetPath returns the file path a migration for name would be written to,
// under path.
func (c *MigrationCreator) GetPath(name, path string) string {
	return filepath.Join(path, c.migrationName(name)+".go")
}

// firePostCreateHooks runs every hook registered with AfterCreate.
func (c *MigrationCreator) firePostCreateHooks(table, path string) {
	for _, callback := range c.postCreate {
		callback(table, path)
	}
}

// AfterCreate registers a hook to run after a migration file is written.
func (c *MigrationCreator) AfterCreate(callback func(table, path string)) {
	c.postCreate = append(c.postCreate, callback)
}

// GetDatePrefix returns the current time formatted as the migration file's
// date prefix: "2006_01_02_150405" in Go's reference-time layout.
//
// It is UTC rather than local: a team spread over two time zones otherwise
// generates prefixes that sort against the order the migrations were
// actually written in.
func (c *MigrationCreator) GetDatePrefix() string {
	now := time.Now
	if c.now != nil {
		now = c.now
	}
	return now().UTC().Format("2006_01_02_150405")
}

// StubPath returns the custom stub directory, the only one there is a path
// to: the built-in stubs are string constants compiled into the binary,
// with no directory of their own to name.
func (c *MigrationCreator) StubPath() string { return c.customStubPath }

// The three built-in stubs, in the Go a migration is written in.
//
// They are constants rather than files because a binary that had to find
// its own source directory to generate a file is a binary that works from
// one working directory.
const (
	blankStub = `package migrations

import (
	"context"

	"github.com/arandu-io/hesape/database/migrations"
)

func init() { migrations.Register({{ class }}{}) }

// {{ class }} is a schema change.
//
// It is compatible with the binary that is still running while a rollout
// finishes: a new column is nullable or has a default, and removing one takes
// two releases.
type {{ class }} struct{ migrations.BaseMigration }

// GetName is this migration's identity. The date prefix carries the order.
func ({{ class }}) GetName() string { return "{{ name }}" }

// Up applies the change.
//
// This is the stub for a migration that is not a table change -- a backfill, an
// extension, a grant -- so it sends a statement. A migration that does change a
// table goes through conn.Schema() instead: the Blueprint says what the table
// should look like and each grammar writes the SQL its engine wants, which is
// what makes one migration run on all three.
//
//	return conn.Schema().Table(ctx, "invoices", func(table *schema.Blueprint) {
//	    table.String("reference").Nullable()
//	})
func ({{ class }}) Up(ctx context.Context, conn migrations.Connection) error {
	_, err := conn.Statement(ctx, ` + "`" + `` + "`" + `, nil)
	return err
}

// Down reverses it.
func ({{ class }}) Down(ctx context.Context, conn migrations.Connection) error {
	_, err := conn.Statement(ctx, ` + "`" + `` + "`" + `, nil)
	return err
}
`

	createStub = `package migrations

import (
	"context"

	"github.com/arandu-io/hesape/database/migrations"
	"github.com/arandu-io/hesape/database/schema"
)

func init() { migrations.Register({{ class }}{}) }

// {{ class }} creates the {{ table }} table.
type {{ class }} struct{ migrations.BaseMigration }

// GetName is this migration's identity. The date prefix carries the order.
func ({{ class }}) GetName() string { return "{{ name }}" }

// Up creates the table.
//
// The columns are String and not Text, and that is the one portability rule
// worth knowing: a column in a key cannot be TEXT on MySQL without a prefix
// length. String is what each grammar renders as the keyable kind.
//
// tenant_id is here because every query in this collection is scoped by it, and
// the index leads with it for the same reason -- see data.Tenant.
func ({{ class }}) Up(ctx context.Context, conn migrations.Connection) error {
	return conn.Schema().Create(ctx, "{{ table }}", func(table *schema.Blueprint) {
		table.String("id").Primary()
		table.String("tenant_id")

		// Timestamps writes created_at and updated_at, which the model stamps.
		table.Timestamps()

		table.Index([]string{"tenant_id", "created_at", "id"})
	})
}

// Down drops it.
func ({{ class }}) Down(ctx context.Context, conn migrations.Connection) error {
	return conn.Schema().DropIfExists(ctx, "{{ table }}")
}
`

	updateStub = `package migrations

import (
	"context"

	"github.com/arandu-io/hesape/database/migrations"
	"github.com/arandu-io/hesape/database/schema"
)

func init() { migrations.Register({{ class }}{}) }

// {{ class }} alters the {{ table }} table.
//
// A column added here is nullable or has a default, so the binary of the
// previous release keeps working while the rollout finishes.
type {{ class }} struct{ migrations.BaseMigration }

// GetName is this migration's identity. The date prefix carries the order.
func ({{ class }}) GetName() string { return "{{ name }}" }

// Up applies the change.
//
// A column added to a table that already has rows is Nullable or has a Default.
// The rule is not the Blueprint's: the previous release's binary is still
// inserting without the column for as long as the rollout takes, and NOT NULL
// with no default fails every one of those inserts.
func ({{ class }}) Up(ctx context.Context, conn migrations.Connection) error {
	return conn.Schema().Table(ctx, "{{ table }}", func(table *schema.Blueprint) {
	})
}

// Down reverses it.
//
// DropColumn takes the names Up added. SQLite refuses to drop a column an index
// still names, so an index added above is dropped here first: the Blueprint
// runs its commands in the order they were added.
func ({{ class }}) Down(ctx context.Context, conn migrations.Connection) error {
	return conn.Schema().Table(ctx, "{{ table }}", func(table *schema.Blueprint) {
	})
}
`
)
