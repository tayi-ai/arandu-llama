---
name: llama-vault-notes
description: Write the Obsidian vault notes that record this module, when the checkout sits inside the Arandu vault. Use when the request is to "write the note", "record this in the vault", "update the journal", "open a gap", "document the module", "add it to the index", "the audit is failing", "layer=... is not in the schema", "which folder does this note go in", or when a change to this package closed or opened a defect somebody else has to know about. Covers how to tell you are in the vault, where a module note lives, which layer a defect belongs to, what the day note has to say, and the one instrument that decides whether any of it was written correctly.
license: MIT
---

# Recording this module in the vault

## First: are you even in the vault?

This skill only applies when this checkout sits beside the Arandu vault. Three
things are true at once when it does, and none of them alone is enough:

```sh
ls ../MOC-arandu.md ../plans/go.mod ../plans/cmd/audit-vault/ ../45-modules/ 2>/dev/null
```

`MOC-arandu.md` at the vault root is the single canonical index.
`plans/cmd/audit-vault` is the Go instrument that decides whether a note is
well formed. `45-modules/` is
where a note about *this* package goes.

If those are absent, stop here. Do not create a vault, do not create the folders,
and do not write notes into this repository imitating them. The vault is one
directory that happens to contain the product repositories; a second one is a
second source of truth, which is the problem it exists to prevent.

## Second: are you the one who writes?

**Agents do not write vault notes. The orchestrator does.** Two agents editing
the same note overwrite each other with no warning, which is exactly why each
session writes its own `##` section in the day note rather than editing another
session's.

So the normal shape of this skill is: you produce the note **content**, name the
file it belongs in, and hand it over. You write directly only when you are the
session driving the work — and then you write your own section, never inside
somebody else's.

## Where a note about this module goes

`45-modules/MOD-<repository-name>.md`, and nowhere else. **The prefix decides the
folder** — a `MOD-` in `40-packages/` is misfiled even if it reads well there.

`40-packages/` is not the neighbour it looks like. That folder answers a closed
question — *do the fourteen main Laravel packages have an equivalent here?* — and
the verdict for most of them is *do not build*. A `PKG-` note can describe
something that will never exist as code. A `MOD-` note cannot: it is written from
a published repository, so the repository comes first and the note second.

A module here means a first-party Go repository that implements
`foundation.Module`, is registered explicitly in an application's
`bootstrap/app.go`, and imports the framework rather than the other way round. It
lives outside `arandu-io` and is not in the root `go.work`.

### What the note has to answer

Four questions, in this order, because it is the order somebody reads them in:

1. **What the module declares about itself.** Paste `arandu.mod.toml` whole — the
   permissions especially. It is what `tests/Unit/audit_test.go` compares against
   what the code actually calls, in this repository — nothing downstream does,
   because `aru doctor` reads the application's own tree and never opens an
   installed package.
2. **How an application wires it.** The typed construction and the explicit
   registration. Never a service provider, a container or discovery — writing
   those into a note is how the next reader builds one.
3. **What it requires of the core.** The framework and hesape floors, from
   `go.mod`, matching the `framework = ">= x.y"` line in the manifest.
4. **What has already broken in it.** The `GAP-` notes whose `layer:` is this
   module.

## Which layer a defect belongs to

This is the rule worth getting right, and it is not about where the defect was
found. **A defect's layer is where its fix has to happen.**

Integrating a module against the core surfaces defects in both directions, and
they are filed in opposite places:

| the fix lands in | `layer:` | example |
| --- | --- | --- |
| this module | the module's slug | a wrong module path, a public API narrower than the format it emits |
| the framework or hesape | `framework`, `hesape`, `aru`, … | a router that stores a domain and never dispatches on it |

Both kinds are `GAP-` notes in `70-gaps/`. Filing a core defect under the module
hides it from the front that owns the fix, and filing a module defect under the
core sends somebody to change a repository that is not at fault. The Arandu
WhatsApp integration opened seven gaps and **not one** of them is `layer:
whatsapp`, because none of the seven is fixed in that repository.

## Registering a new module slug

Read the `layers` set in `plans/cmd/audit-vault/main.go` at the vault root;
it is the only authority for accepted layer names. Do not copy that list into
this skill or infer a layer from a retired repository name. The Redis adapter
uses `redis`; a defect fixed in the parent library still belongs to `hesape`.

If a new module slug is absent, add it to that set together with its `MOD-`
note and the row in `45-modules/MOC-modules.md`. Run the vault audit to verify
the actual metadata, rather than maintaining another registry here.

## The day note

`_journal/YYYY-MM-DD.md`, one note per day, never two for the same date. Several
sessions in one day are several `##` sections inside it.

Four labels, in English, always in this order:

**Changed** one item per logical unit · **Broke** the decisive line of the error,
not the whole log · **Left open** what the next session needs to know ·
**Verification** what ran and what came out.

Facts, not narration. A decision does not go here — a decision closes in an ADR.

Two rules about *when*, and both were paid for:

- **The note is written when state changes, not batched at the end.** Notes
  written afterwards tell people to do work that is already done, and the reader
  redoes it.
- **The vault is a dated snapshot, not a live mirror.** Divergence between a note
  and the disk *after* its date is the project having moved. That becomes a line
  in `_journal/`; it does not become an edit to the old note.

An append-only log inside an existing note is the exception to that second rule,
and it is marked as one: the `**Sessions**` block in a front note and the day
list in `MOC-journal` are written to be added to.

## Frontmatter

Twelve keys in every note, and the set is closed — the audit rejects a key
outside it, including one that would be useful:

```yaml
id: MOD-arandu-swagger
type: module         # module, for a note in 45-modules/
layer: swagger       # this module's slug
front: F2
status: implemented  # absent | declared | implemented | tested | frozen
closure: open        # open | complete | deferred | non-goal
owner: F2
evidence: "github.com/<owner>/<repo>@v0.1.0 · arandu.mod.toml:8-21"
source: "what produced this note"
aliases: ["how somebody would search for it"]
updated: 2026-08-26
tags: [type/module, layer/swagger, front/f2]
```

`evidence:` is the field that decides whether the note can be checked. A path
with line numbers, a tag, a command — something a reader can open. "Tested and
working" is not evidence.

`closure:` is the outcome of the *question*, not of the work. A gap whose answer
is "we are not doing this" is `non-goal` and closed, not `open` forever.

File names are English kebab-case, ASCII only, and **unique across the whole
vault** — Obsidian resolves `[[link]]` by name and not by path, so two notes with
the same name in different folders break the graph silently.

Structure in English, prose in Portuguese. Field names, section titles and labels
are English; what is written under them is Portuguese.

## The gate

One command decides whether any of this was done right:

```sh
GOWORK=off go -C plans run ./cmd/audit-vault
```

Thirteen checks: file names, duplicate names, missing frontmatter, frontmatter outside
the schema, ADRs without aliases, supersession symmetry, unresolved `[[links]]`,
orphan notes, the front dependency graph, and the queue. It exits non-zero if any
of them finds something, and green output names the note count.

**The instrument outranks the documentation.** If this file and the audit
disagree, the audit is right and this file is what needs correcting — the same
way a note that contradicts the code is wrong rather than the code. An unresolved
`[[link]]` and an orphan note are both real findings: a note nothing links to is
a note nobody will read again.
