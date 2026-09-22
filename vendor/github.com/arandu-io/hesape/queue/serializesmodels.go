package queue

import (
	"context"
	"fmt"

	"github.com/arandu-io/hesape/auth"
)

// ModelIdentifier is a record reduced to what finds it again.
//
// It is what [SerializesModels] puts on the wire in place of a record. Class is
// what the application registered a finder under, ID is the primary key, and
// Relations are the relations to load back with it.
type ModelIdentifier struct {
	// Class names the kind of record. It is a string a finder is registered
	// under -- "invoice", "user" -- because Go cannot reach a type from a name
	// in a string.
	Class string
	// ID is the primary key.
	ID string
	// Relations are the relations to load with it. Empty means none, which is
	// the right default: a job that needs a relation says so, and one that does
	// not should not pay for it.
	Relations []string
	// Connection names the database connection it came from, and empty means
	// the default.
	Connection string
	// TenantID is who it belongs to, and it is the thing that makes restoring
	// one safe: the finder is given a Grant built for this tenant, so a payload
	// that names another customer's row finds nothing.
	TenantID string
}

// ModelFinder loads a record back from its identifier.
//
// It is what the application registers so a job can carry an id instead of a
// document. It takes the Grant the job runs under, which is what makes the
// reload obey the same policy the original read did.
type ModelFinder func(ctx context.Context, g auth.Grant, id ModelIdentifier) (any, error)

// ErrMissingModel is returned when a record a job's payload names is gone.
//
// A job whose record was deleted cannot succeed on any retry. A handler that
// wants that treated as success rather than as failure says so with
// attributes.Attributes.DeleteWhenMissingModels.
var ErrMissingModel = fmt.Errorf("queue: the record this job is about no longer exists")

// SerializesModels keeps records out of job payloads.
//
// The rule it enforces is that a payload carries ids and facts, never a
// document: a record goes on the wire as a [ModelIdentifier] and is loaded back
// on the other side. This type is that rule with a name and a finder registry,
// and [SerializesModels.RestoreModel] is the reload.
//
//	var models queue.SerializesModels
//	models.FindModelsUsing("invoice", func(ctx context.Context, g auth.Grant, id queue.ModelIdentifier) (any, error) {
//		return invoices.Find(ctx, g, id.ID)
//	})
//
// Why it matters is not size. A job that carries a serialized record acts on
// the record as it was when the job was queued, and the queue exists precisely
// because time passes between those two moments: the invoice was voided, the
// address was corrected, the user was deleted. Reloading is what makes the job
// act on what is true now.
type SerializesModels struct {
	finders map[string]ModelFinder
}

// FindModelsUsing registers how to load a kind of record back.
//
// The class on an identifier is a string, and this is what turns it into code
// -- the same trade the [Worker] makes for job names, because Go cannot reach a
// type from a name in a string.
func (s *SerializesModels) FindModelsUsing(class string, find ModelFinder) *SerializesModels {
	if s.finders == nil {
		s.finders = map[string]ModelFinder{}
	}
	s.finders[class] = find
	return s
}

// RestoreModel loads the record an identifier names.
//
// The Grant is rebuilt from the identifier's tenant, so the reload is scoped to
// the customer the job belongs to and a payload naming another customer's row
// finds nothing.
//
// A record that is gone comes back as an error wrapping [ErrMissingModel], not
// as a nil the handler has to remember to check.
func (s *SerializesModels) RestoreModel(ctx context.Context, action auth.Action, id ModelIdentifier) (any, error) {
	find, registered := s.finders[id.Class]
	if !registered {
		return nil, fmt.Errorf("queue: nothing knows how to load a %q. Register it with FindModelsUsing in bootstrap/app.go", id.Class)
	}
	if id.TenantID == "" {
		return nil, fmt.Errorf("queue: the identifier for %s carries no tenant, and a record cannot be loaded without one", id.Class)
	}

	model, err := find(ctx, auth.SystemGrant(action, id.TenantID), id)
	if err != nil {
		return nil, err
	}
	if model == nil {
		return nil, fmt.Errorf("%w: %s %s", ErrMissingModel, id.Class, id.ID)
	}
	return model, nil
}

// GetSerializedPropertyValue is what goes on the wire in place of a record.
//
// A value that can identify itself is reduced to its identifier; everything
// else is passed through, because a payload of plain facts is already what it
// should be.
func (s *SerializesModels) GetSerializedPropertyValue(value any) any {
	if model, can := value.(Identifiable); can {
		return model.ModelIdentifier()
	}
	return value
}

// GetRestoredPropertyValue is the record an identifier names, or the value
// itself when it is not one.
func (s *SerializesModels) GetRestoredPropertyValue(ctx context.Context, action auth.Action, value any) (any, error) {
	if id, is := value.(ModelIdentifier); is {
		return s.RestoreModel(ctx, action, id)
	}
	return value, nil
}

// Identifiable is a record that can say what finds it again.
//
// It is what a domain type implements so [SerializesModels] can reduce it: a
// record says what finds it again, rather than a base class being recognized.
type Identifiable interface {
	// ModelIdentifier is what finds this record again.
	ModelIdentifier() ModelIdentifier
}
