# Skills

Procedures an assistant follows when working on this package.

They live in `.agents/skills/<name>/SKILL.md`, which is the path the coding
assistants read from. It is one directory rather than a file per vendor, so a
skill written once is read by whatever this package is being written with.

Each file opens with frontmatter carrying a `name` and a `description`. The
`name` has to equal the directory name exactly, or the skill is not loaded. The
description names the situation you are in rather than the topic it covers.

| skill | when it fires |
| --- | --- |
| `llama-engine` | touching cgo, the wrapper, the submodule, the archives or a link failure |
| `llama-measure` | changing what a reading is, how a loss is computed, or what an adapter is applied at |
| `llama-module` | adding a route, a handler, a config field, a response field or a migration |
| `llama-policy` | opening an action, adding an authorised use case, or anything answering 403 |
| `llama-release` | the gates, the manifest, a dependency, a version, a tag |
| `llama-package` | installing and wiring this package **into an application** |

The last one has a different audience from the others, and that is on purpose:
it travels with the package so that an assistant working in somebody else's
project — the one running `go get` — has the archives, the build tag and the
linker flag in front of it instead of guessing.

## Why these exist

This package is two things at once, and the failure modes of the two halves do
not resemble each other.

As an **Arandu package** the usual mistakes apply: a provider, a container
lookup, a CRUD Repository beside the configured Model, a tenant read from the
URL, a Policy branch returning nil for administrators "for now". None belongs
here, and the last three are security failures rather than style
disagreements.

The configured Model is the whole data path. `Llamas(db)` returns it, every
read and write goes through it, and a Repository wrapping it is a second path
the policy does not guard.

As an **engine** the mistakes are quieter and cost more. A number that is
plausible and wrong is the characteristic failure of everything in
`llama-measure`: an adapter applied at the wrong magnitude, a mean of means, a
loss read off a KV cache filled under a different adapter. None of those fails
anything. They produce a figure, the figure enters a comparison, and the
comparison moves a decision.

That is why `llama-measure` exists as a skill of its own rather than as a
section of `llama-engine`. The engine half is where a mistake **crashes**; the
measurement half is where a mistake **is believed**.

## Adding your own

A skill in this directory is yours and travels with the repository. Keep it a
procedure rather than a description: a file that says "read the documentation"
never changes what anybody does.
