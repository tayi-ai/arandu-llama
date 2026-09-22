// Package events is a dispatcher, its listeners, and an outbox underneath.
//
// # The dispatcher
//
// Listen registers, Dispatch fires, and the two are all most code touches.
// Until returns the first non-nil response and leaves the listeners behind it
// uncalled; a listener that returns false stops the ones behind it without
// discarding what has already been collected; a wildcard listener is handed the
// event name and the payload, and every other listener is handed the payload
// spread across its parameters. Push and Flush hold a dispatch back and release
// it, Subscribe registers a value that registers its own listeners, and Forget
// removes them.
//
// A listener registered against an interface the event happens to implement is
// not found: Go cannot go from a name to a type at run time, and an interface
// is structural rather than declared, so there is nothing to walk. Register
// against the event.
//
// The dispatcher is wired in bootstrap/app.go, which is where an Arandu project
// wires everything.
//
// # The outbox is the mechanism
//
// A dispatcher on its own loses data in both directions: if the process dies
// between the write and the dispatch, the event never leaves; if a listener
// runs and the transaction rolls back, the rest of the system reacted to
// something that did not happen.
//
// So an event that has to survive the request is stored in the same transaction
// as the write that produced it -- Outbox.Store -- and Relay publishes it
// afterwards, into a Publisher. A *Dispatcher is a Publisher: Publish hands the
// stored row to the listeners registered for its name. That is the one place
// the two halves meet, and it is why an application registers everything with
// Listen and never has to know which half delivered it.
//
// Delivery is at-least-once, which is the price of never losing an event: the
// consumer deduplicates on Stored.ID, and that is why the id is stable and why
// it travels with the event.
//
// An event that has to leave the process goes through the outbox. The
// dispatcher itself broadcasts nothing.
//
// # Testing
//
// There is nothing global to swap, and a package-level dispatcher a test could
// swap would be shared mutable state that two tests calling t.Parallel would
// fight over. What a test writes instead is the dispatcher it builds for
// itself, which is already isolated by having none of the application's
// listeners on it:
//
//	d := events.NewDispatcher()
//
//	var shipped []OrderShipped
//	d.Listen(func(e OrderShipped) { shipped = append(shipped, e) })
//
//	placeOrder(ctx, g, d)
//
//	if len(shipped) != 1 {
//		t.Fatalf("the order shipped %d times, want 1", len(shipped))
//	}
//
// Keeping some listeners is registering the ones you want rather than removing
// the ones you do not, and scoping the substitution is the scope of the
// variable: d stops existing when the function that built it returns.
package events
