---
name: llama-policy
description: Decide who may reach a reading — open an action, add an authorised use case on the service, or work on anything that answers 403 or 404. Use when the request is to "let admins read", "open the list endpoint", "add a permission", "allow the owner", "add a method to the service", "it returns 403", "make readings public", or when a change touches policy.go or service.go.
license: MIT
---

# The policy denies everything, and that is the finished state

`LlamaPolicy.Can` ends in a refusal and has no branch that allows anything.
That is not a placeholder to delete. A policy shipped with a permissive branch
is a hole in every application that installs the package, and the hole looks
like working code until somebody reads it.

Open one action at a time, inside `arandu:begin custom` / `arandu:end custom`.
What is not written there stays closed — **including every action added
later**, which is the property that makes the default worth keeping.

```go
// arandu:begin custom
if a == LlamaList && s.HasRole("operator") {
	return nil
}
// arandu:end custom
```

## Tenant isolation comes first, and stays first

```go
if record.ID != "" && record.TenantID != s.Tenant {
	return fmt.Errorf("llama belongs to another tenant")
}
```

Before every rule, applying to every action. Without it a rule allowing an
owner to read their own record allows it across customers as soon as two of
them have a record with the same identifier.

The empty id is the candidate not yet stored. It belongs to nobody until it is
written with the tenant off the Grant. Never take the tenant from the URL, the
body or a header.

## 404, not 403

A record in another tenant answers **not found**. The two are deliberately
indistinguishable: 403 confirms the row exists, which is the fact being
protected.

`ErrNotFound` covers both cases and its doc comment says so. Do not add a
distinct "forbidden" error to make debugging easier — that is exactly the
disclosure.

## Every service method authorises before it reaches the Model

The order is fixed and the structural test enforces it:

1. `security.Authorize` for the action, on the record or the zero value
2. only then `Llamas(s.db)` and a Model terminal with it, using the Grant
3. the tenant written from the Grant, never from the request

A method that reads the Model to find out whether it may read the Model has
already lost. The denial tests run with a **nil database**, so any path that
touches storage before authorisation panics rather than passing quietly.

If a new method cannot be written that way, the method is wrong, not the rule.

## What does not go in here

- **No CRUD Repository beside the Model.** The Model is the data path; a
  repository wrapping it becomes a second, unauthorised one.
- **No container lookup, no provider.** An Arandu application registers a
  module explicitly. Anything discovered at runtime is a dependency nobody can
  see in `bootstrap/app.go`.
- **No `HasRole("admin")` shortcut at the top of `Can`.** It reads as
  convenience and behaves as a bypass for every action added afterwards. Name
  the actions.

## Before you finish

```bash
go test -race ./...
```

The suite has two halves that both matter: the denial tests prove the closed
path, and the structural twin reads every exported service method and checks
the same order on the allowed path. A new method with no test in either is a
new method with no proof.

Add the denial test **first**, watch it pass with the policy still closed, then
open the rule and add the allow test. A test written after the rule tends to
test the rule rather than the boundary.
