# Testing ai gov graph

These tests measure whether the knowledge graph pipeline produces the right observable result. Favour a few well chosen tests that tell a clear story over broad, shallow coverage.

## The rule for keeping a test

A test earns its place only when it can fail for a reason that no other test would catch.

Before adding a test, name the one regression or invariant it guards. If its reason is that the code obviously works, or another test already covers the path, leave it out.

A test also earns its place when it guards a fixture or pins a past regression. State that reason in the docstring.

## Test against real objects

Use real collaborators. Mock only a boundary that is external, expensive or non deterministic. State the reason in the test.

Assert on behaviour that a caller can observe. A test should survive a refactor that preserves that behaviour.

## Cover methodologies with an oracle

When a methodology has a known answer, generate data with that answer and score the result against it. Add each methodology as a `pytest.param`, rather than creating another test body.

Use hand built fixtures for plans and end to end behaviour that needs known rows. Keep methodology correctness separate from the flow that carries results through a plan.

## Name tests as tags

A test name categorises a test. It does not explain it. A file's `def test_*` lines should read as an index. A shared prefix names the feature. A short suffix names the case.

```python
def test_claim_resolution_one_to_many(...) -> None:
    """A mention resolves to an entity that has several identifiers."""


def test_claim_resolution_one_to_none(...) -> None:
    """A mention does not resolve to an existing entity."""
```

The suffix names a case that could have been listed in advance, such as `one_to_none`, `with_deduplication` or `rejects_invalid_evidence`. Put the behavioural explanation and the reason the test exists in the docstring.

For a parametrised family, the case is the `pytest.param(..., id=...)`. The function name is the family alone. A lone test still uses a feature prefix. Use the vocabulary in [the glossary](glossary.md) for both families and cases.

## Where a test lives

The test tree shadows `src`. Put unit tests at `test/<module>/test_<thing>.py` for `src/aigg/<module>/<thing>.py`. Put cross module and end to end behaviour in its own test area.

Use a small scenario fixture for plans and end to end tests that assert on known rows. Use a generated oracle for methodology correctness at scale.

## Conventions

- Parametrise every set of cases with `pytest.param(..., id=...)`. Pass parameter names as a tuple, including a trailing comma for one name. Keep the test signature typed.
- Give every test a docstring that describes the behaviour and its reason to exist.
- Prefer `unittest.mock.patch` to the `monkeypatch` fixture.
- Give each test one reason to fail. Split a test that guards more than one behaviour.
