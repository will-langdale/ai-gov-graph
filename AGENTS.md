# Working in ai gov graph

## Copy

Write in British English. Use sentence case headings. Write plain English for a technical reader.

Keep sentences direct and simple. Use semicolons, colons and hyphens only when they reduce complexity.

## Scope

Follow YAGNI principles. Implement only behaviour the active request requires.

## Tests

When adding or changing tests, read [the testing guide](docs/testing.md). Use the terms in [the glossary](docs/glossary.md) in test names and test case IDs.

## Validation

Run the relevant `just` recipe before handing over a change. Use `just format` for formatting and linting, `just check` for type checking and `just test` for tests.

## Agent skills

### Experiment CLI

When extending source acquisition or graph construction workflows, preserve their
separate Typer applications and explicit evidence boundaries. Read the
[Commands section](README.md#commands) for the command shape and lineage contract.

### Issue tracker

Issues and specs are tracked in this repository's GitHub Issues. See `docs/agents/issue-tracker.md`.

### Triage labels

The default canonical triage labels are used. See `docs/agents/triage-labels.md`.

### Domain docs

This repository uses a single-context domain-doc layout. See `docs/agents/domain.md`.
