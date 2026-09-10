package unit_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/arandu-io/framework/data"
	"github.com/arandu-io/framework/security"
	"github.com/arandu-io/hesape/database/model"

	llama "github.com/tayi-ai/arandu-llama"
)

// The four properties this package exists to keep are checked here, and they
// are checked against the code rather than described in a document:
//
//  1. the policy denies every action, and has no branch that allows one;
//  2. the service authorizes before constructing or executing a Model query;
//  3. the tenant comes from the Grant;
//  4. nothing reaches the database without passing through the first two.
//
// The fourth is checked by the handle these tests pass in. It wraps a nil
// *sql.DB, so any statement that were issued would panic and fail the test
// loudly -- which makes "the refusal happened before the Model" a fact the
// suite proves rather than a comment. The structural twin in audit_test.go
// keeps that order visible on every service method, including an allowed path.

// everyAction is the whole set the policy answers about. A test that listed
// four of five would pass while the fifth was open.
var everyAction = []security.Action{
	llama.LlamaView,
	llama.LlamaList,
	llama.LlamaCreate,
	llama.LlamaUpdate,
	llama.LlamaDelete,
}

// administrator is the most privileged subject an application can produce. It
// is the one to test the default with: a policy that refuses an administrator
// refuses everyone.
func administrator() security.Subject {
	return security.Subject{ID: "user-1", Tenant: "acme", Roles: []string{"admin"}, Verified: true}
}

func TestThePolicyDeniesEveryActionByDefault(t *testing.T) {
	t.Parallel()

	for _, action := range everyAction {
		t.Run(string(action), func(t *testing.T) {
			t.Parallel()

			_, err := security.Authorize(context.Background(), llama.LlamaPolicy{},
				administrator(), action, llama.Llama{})
			if !errors.Is(err, security.ErrForbidden) {
				t.Fatalf("an unopened policy allowed %s: got %v, want ErrForbidden", action, err)
			}
		})
	}
}

func TestThePolicyDeniesARecordOfAnotherTenant(t *testing.T) {
	t.Parallel()

	other := llama.Llama{ID: "record-1", TenantID: "globex", Name: "theirs"}

	err := llama.LlamaPolicy{}.Can(context.Background(),
		administrator(), llama.LlamaView, other)
	if err == nil {
		t.Fatal("the policy allowed a record belonging to another tenant")
	}
	// The message is asserted because the tenant check is the one refusal that
	// has to survive somebody opening the actions below it.
	if !strings.Contains(err.Error(), "another tenant") {
		t.Fatalf("the refusal did not name the tenant: %v", err)
	}
}

func TestThePolicyDeniesAGuest(t *testing.T) {
	t.Parallel()

	for _, action := range everyAction {
		_, err := security.Authorize(context.Background(), llama.LlamaPolicy{},
			security.Guest("acme"), action, llama.Llama{})
		if !errors.Is(err, security.ErrForbidden) {
			t.Fatalf("a guest was allowed %s: got %v, want ErrForbidden", action, err)
		}
	}
}

func TestAuthorizeRefusesASubjectThatIsNobody(t *testing.T) {
	t.Parallel()

	// The zero Subject is a session that failed to load, not an anonymous
	// reader, and it is refused before the policy is consulted. A package that
	// answered it as a guest would answer a broken session as a visitor.
	_, err := security.Authorize(context.Background(), llama.LlamaPolicy{},
		security.Subject{}, llama.LlamaView, llama.Llama{})
	if !errors.Is(err, security.ErrForbidden) {
		t.Fatalf("an empty subject was authorized: got %v, want ErrForbidden", err)
	}
}

// nilHandle is a handle over no database.
//
// Any statement issued through it panics, which is what makes these tests
// prove that the refusal came first: a service that reached the Model before
// authorizing would crash here rather than pass.
func nilHandle() *data.DB { return data.Wrap(nil, data.DialectSQLite) }

func TestTheServiceRefusesBeforeReachingTheModel(t *testing.T) {
	t.Parallel()

	// A nil handle makes even construction of Llamas panic at
	// GetQueryGrammar. This catches moving the configured Model entry point --
	// not only its terminal -- ahead of authorization.
	service := llama.NewLlamaService(nil)
	ctx := context.Background()

	if _, err := service.Find(ctx, administrator(), "record-1"); !errors.Is(err, security.ErrForbidden) {
		t.Fatalf("Find reached the Model before the policy refusal: %v", err)
	}
	if _, err := service.List(ctx, administrator(), data.Query{}); !errors.Is(err, security.ErrForbidden) {
		t.Fatalf("List reached the Model before the policy refusal: %v", err)
	}
	if _, err := service.Create(ctx, administrator(), llama.CreateRequest{Name: "one", Subject: "repair-skeleton-validation", Policy: "ea5fa653", Quantisation: "Q4_K_M", Loss: 45.5, Tokens: 132}); !errors.Is(err, security.ErrForbidden) {
		t.Fatalf("Create reached the Model before the policy refusal: %v", err)
	}
}

func TestLlamasReturnsAWiredTenantScopedModel(t *testing.T) {
	t.Parallel()

	rows := llama.Llamas(nilHandle())
	if rows.GetTable() != "llamas" {
		t.Fatalf("Llamas table = %q, want llamas", rows.GetTable())
	}
	if rows.KeyType != "string" || rows.Incrementing {
		t.Fatalf("Llamas key is type %q, incrementing %t; want application-generated text", rows.KeyType, rows.Incrementing)
	}
	if rows.TenantColumn != "tenant_id" {
		t.Fatalf("Llamas tenant column = %q, want tenant_id", rows.TenantColumn)
	}
	if model.ModelOf(rows.Entity) != rows {
		t.Fatal("Llamas returned an entity whose embedded Model is not wired to it")
	}
}

func TestASystemGrantWithoutATenantReachesNothing(t *testing.T) {
	t.Parallel()

	// A system grant with no tenant names no customer. The Model refuses it
	// while preparing the query, before the nil handle can issue a statement.
	_, err := llama.Llamas(nilHandle()).NewQuery().WhereKey("record-1").First(
		context.Background(), security.SystemGrant(llama.LlamaView, ""))
	if !errors.Is(err, model.ErrNoTenant) {
		t.Fatalf("a system grant with no tenant returned %v, want ErrNoTenant", err)
	}
}

func TestTheTenantComesFromTheGrant(t *testing.T) {
	t.Parallel()

	g := security.SystemGrant(llama.LlamaView, "acme")
	if got := data.Tenant(g); got != "acme" {
		t.Fatalf("data.Tenant(g) = %q, want %q", got, "acme")
	}

	// And a Grant nobody issued carries no tenant at all, so a statement that
	// took its tenant from anywhere else would be reading rows this Grant does
	// not name.
	if got := data.Tenant(security.Grant{}); got != "" {
		t.Fatalf("the zero Grant carries the tenant %q, want none", got)
	}
}

func TestTheRequestValidatesItsInput(t *testing.T) {
	t.Parallel()

	if errs := (llama.CreateRequest{}).Validate(); !errs.Any() {
		t.Fatal("an empty request validated")
	}
	if errs := (llama.CreateRequest{Name: strings.Repeat("a", 121), Subject: "s", Policy: "p", Quantisation: "q", Loss: 1, Tokens: 1}).Validate(); !errs.Any() {
		t.Fatal("a name past the maximum validated")
	}
	if errs := (llama.CreateRequest{Name: "one", Subject: "repair-skeleton-validation", Policy: "ea5fa653", Quantisation: "Q4_K_M", Loss: 45.5, Tokens: 132}).Validate(); errs.Any() {
		t.Fatalf("a valid request was rejected: %v", errs)
	}
}

func TestTheConfigurationRefusesWhatCannotWork(t *testing.T) {
	t.Parallel()

	for name, cfg := range map[string]llama.Config{
		"no tenant":        {},
		"tenant with a /":  {Tenant: "acme/reports"},
		"tenant uppercase": {Tenant: "Acme"},
		"relative prefix":  {Tenant: "acme", Prefix: "llama"},
		"page size too big": {Tenant: "acme",
			PageSize: llama.MaxPageSize + 1},
		"negative page size": {Tenant: "acme", PageSize: -1},
	} {
		if err := cfg.Validate(); err == nil {
			t.Errorf("the configuration with %s was accepted", name)
		}
	}

	if err := (llama.Config{Tenant: "acme"}).Validate(); err != nil {
		t.Fatalf("a valid configuration was refused: %v", err)
	}
}
