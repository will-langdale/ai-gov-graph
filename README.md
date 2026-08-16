# 🏛️ ai-gov-graph

`ai-gov-graph` is an experiment in using LLMs to build and evolve a knowledge graph from GOV.UK content.

The aim is to represent knowledge that can improve the correctness and discoverability of government content for citizens, visitors, businesses, civil society and civil servants.

The system starts with a small fixed semantic foundation rather than a GOV.UK domain ontology. It extracts source supported claims from GOV.UK content, resolves entities and time, and lets the ontology evolve as new knowledge is encountered. Existing ontologies may be reused where the agent judges them to be a better fit.

Claims remain linked to the exact source text that supports them. Claims that cannot yet fit the ontology are retained rather than discarded.

Documents are first class graph entities. This allows the graph to support both knowledge queries and content analysis.

Initial questions include

- What do we know about the Permanent Secretary?
- What supported claims do not currently fit the graph?
- Which important concepts have no guide or explanatory coverage?
- Which central concepts are supported only by expired policy claims?
- How did the ontology evolve as the corpus was processed?
- Does ontology change naturally reduce as the graph matures?

The first experiment will process about 100 heterogeneous pages from one GOV.UK department.

## 🚀 Getting started

You need Python 3.14 or later and [uv](https://docs.astral.sh/uv/) installed.

Clone the repository, then install the project and its development tools:

```shell
uv sync
```

Confirm the available commands:

```shell
uv run aigg acquire --help
uv run aigg graph --help
```

To create a small local evidence corpus, fetch up to ten documents from a
GOV.UK organisation into a new directory:

```shell
uv run aigg acquire documents fetch \
  --corpus-directory evidence/department-for-business-and-trade \
  --organisation department-for-business-and-trade \
  --maximum 10
```

The corpus directory must not already contain files. The command needs network
access to the GOV.UK APIs but does not require an LLM credential.

## Reasoning configuration

Live reasoning loads its OpenRouter credential only from
`OPENROUTER_API_KEY`. The default model is `deepseek/deepseek-v4-pro`. Each
provider request has a 60-second network timeout and no SDK retries, so a
provider failure cannot enter the SDK's long retry back-off. Set
`OPENROUTER_MODEL`, `OPENROUTER_TIMEOUT_MS` or `OPENROUTER_MAX_RETRIES` in the
environment to override these defaults.

## 🧭 CLI at a glance

The project keeps its workflows in separate command line applications so that
source evidence and graph construction have clear boundaries:

- `acquire`: discover GOV.UK documents and record an immutable, manifest-backed
  evidence corpus.
- `graph`: initialise and evolve a graph experiment lineage from local,
  verified evidence.
- Planned workflows will extend these applications rather than blur the evidence
  and graph boundaries. See the [design document](docs/plans/ai-gov-graph.md)
  for the experiment programme.

## 📝 MVP todo

- [ ] Build the GOV.UK Content API downloader
- [x] Create deterministic source text and evidence references
- [ ] Define the claim representation
- [ ] Build open claim extraction
- [ ] Build entity and temporal resolution
- [ ] Create the RDF dataset and validation pipeline
- [ ] Build autonomous ontology research and evolution
- [ ] Add LangGraph orchestration
- [ ] Record graph and ontology observations after every document
- [ ] Add the initial knowledge and coverage queries
- [ ] Run the first 100 page corpus
- [ ] Verify the graph can be rebuilt entirely from version controlled artefacts

The detailed architecture, experimental questions and spike programme live in the [design document](docs/plans/ai-gov-graph.md).

## 💻 Commands

Use `aigg` for the project command line interface. It keeps separate
subcommands for acquiring source documents and constructing a graph from local
evidence:

```shell
uv run aigg acquire --help
uv run aigg graph --help
```

Initialise a graph experiment lineage from a JSON configuration object:

```shell
uv run aigg graph experiment initialise \
  --lineage-directory lineages/first-run \
  --configuration configuration.json
```

Initialisation records the configuration as a schema-versioned, content-addressed
artefact under the lineage. `lineage.json` records its `sha256:` identity and
schema versions. Consumers must use `ArtefactStore.read_json()` so the content
hash is checked before an artefact is used.

Acquire an immutable, manifest-backed evidence corpus for a GOV.UK organisation:

```shell
uv run aigg acquire documents fetch \
  --corpus-directory evidence/department-for-business-and-trade \
  --organisation department-for-business-and-trade \
  --maximum 100
```

The command enumerates the organisation through the GOV.UK Search API, stores
each Content API response byte-for-byte under its SHA-256 hash, and writes a
versioned canonical text rendering alongside it. `manifest.json` records an
explicit base-path order and the content ID, locale, update time, source and
canonical hashes, and canonicaliser version that identify each source version.
A non-empty corpus directory is refused. Failed acquisitions produce
`acquisition-failure.json` and no manifest, so partial evidence cannot be
mistaken for a complete corpus.

Project the retained Source documents in a complete corpus into a named-graph
TriG RDF dataset:

```shell
uv run aigg graph documents run \
  --corpus-directory evidence/department-for-business-and-trade \
  --dataset-path source-documents.trig
```

The dataset retains each Source document version as a first-class resource. Its
original JSON and each top-level JSON value are stored as source assertions in
the source-documents graph. This preserves GOV.UK schemas, document types,
taxons, organisations, links and publication metadata without asserting them
as semantic classes.
