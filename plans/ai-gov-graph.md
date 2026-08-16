# Design for an autonomous GOV.UK knowledge graph agent

## Purpose

This project will test whether an autonomous LLM based system can construct and evolve a useful knowledge graph from GOV.UK content.

The graph should represent source supported knowledge that helps improve the correctness and discoverability of GOV.UK content.

The intended audiences include citizens, visitors, businesses, civil society and civil servants.

The graph should support tasks such as these.

1. Discover what GOV.UK currently says about an entity, role, policy, service, requirement or topic.

2. Find claims that conflict with other claims or appear to have become false or outdated.

3. Find important concepts for which GOV.UK has weak, missing or stale explanatory coverage.

4. Find relationships between content that improve discovery.

5. Expose knowledge that appears useful but which the current ontology cannot represent.

6. Show how the ontology emerged and changed as more evidence was processed.

The graph is not intended to represent every true statement found on GOV.UK.

A claim should be considered in scope when representing it could materially support content correctness, content discovery, coverage analysis or the needs of GOV.UK audiences.

This makes scope dependent on purpose rather than subject matter.

For example, the statement that oranges are fruit may be relevant where GOV.UK content concerns plant health, food classification, trade or import rules. The statement that oranges are sweet may be true but normally irrelevant to the purpose of the graph.

Scope decisions made by the agent must be retained with their rationale.

## Experiment scope

The first experiment will use approximately 100 heterogeneous pages associated with the Department for Business, Innovation, Science and Trade.

The content acquisition process will be separate from the agent.

A small Typer command line application will retrieve Content API JSON from GOV.UK and write the evidence corpus to disk.

The agent will then be run against either one JSON file or a directory.

For example

```text
govuk-kg run -f evidence/page.json
```

or

```text
govuk-kg run -f evidence/
```

The evidence given through this interface is the only GOV.UK evidence available to the agent.

The agent may use the web to research external ontologies. It must not fetch additional GOV.UK evidence during the run.

Each source document will go through open claim extraction once.

Claims produced by that extraction may later be reconsidered many times as entities are resolved, time is resolved and the ontology changes.

The first experiment will make one pass through the corpus.

## Core design principle

The system separates evidence, claims, ontology and accepted knowledge.

A source document does not directly modify the knowledge graph.

The system first extracts candidate claims from the source.

Those claims are resolved and mapped against the current ontology.

The proposed graph change is then checked against the current graph, ontology, SHACL shapes and conflict policy.

Accepted claims contribute to the canonical semantic graph.

Claims that cannot currently be accepted remain in the RDF dataset with their evidence and disposition.

The event history and claim history therefore explain how the accepted graph came to exist.

## The RDF dataset

The semantic output of the system is one RDF dataset.

Documents, claims, entities, ontology releases and system observations are all first class RDF resources.

Different named graphs may distinguish different kinds of information while remaining within one RDF dataset.

The dataset should contain at least the following logical areas.

### Accepted knowledge

This is the current semantic projection of accepted claims.

It contains entities, concepts and relationships that the system currently accepts.

### Claims

Claims are first class resources.

A claim may be accepted, unresolved, in conflict, out of scope, rejected or unable to fit the current ontology.

Claims are never discarded merely because the current ontology cannot represent them.

### Evidence

Evidence connects claims to the GOV.UK documents from which they were extracted.

Evidence must identify the exact text that supports the claim.

### Documents

GOV.UK content items are first class nodes.

Their authoritative source metadata is retained exactly as supplied by GOV.UK.

This includes information such as content identifier, path, document type, schema, publication times, organisations, taxons and links.

### Ontology history

Every ontology release is represented.

Changes between ontology releases are represented.

The evidence and reasoning that caused a change are represented.

### System observations

The graph state, claim state and ontology state before and after processing each document are represented so that the experiment can later be analysed.

## Documents and source vocabularies

GOV.UK already contains implicit and explicit vocabularies.

Document types, schemas, taxons, organisations and similar source classifications must not automatically become classes in the emergent ontology.

They are source assertions.

For example, a GOV.UK source value such as `detailed_guide` should be retained unchanged.

The agent may separately conclude that several source classifications have a common semantic purpose.

For example, it might decide that `Guide`, `UserGuide` and `DetailedGuidance` all refer to closely related content functions.

The source terms must remain intact.

The semantic alignment between them is a separate and revisable assertion.

SKOS may be used where concepts are judged to be exact matches, close matches, broader concepts or narrower concepts.

The agent must not collapse terms into identity merely because they appear similar.

This separation lets the graph preserve GOV.UK vocabulary while discovering a more coherent cross departmental vocabulary of its own.

## Claims and evidence

The initial claim envelope will follow the nanopublication pattern.

A claim therefore has an assertion, provenance and publication information.

The provenance should use standard vocabularies where practical.

The exact source evidence should use the Web Annotation model.

Each GOV.UK JSON source will also have a deterministic canonical text representation.

The canonical representation exists to provide stable evidence positions.

Each evidence reference should retain at least the following information.

1. GOV.UK content identifier.

2. GOV.UK URL.

3. Hash of the original Content API JSON.

4. Hash of the canonical text.

5. Canonicaliser version.

6. Starting line.

7. Ending line.

8. Starting character position.

9. Ending character position.

10. Exact selected text.

11. Suitable surrounding prefix and suffix text.

The original JSON and canonical text should both be version controlled.

The canonicaliser must be deterministic.

A claim must always be traceable back to the text that caused it to exist.

## Claim extraction

Claim extraction is intentionally open.

The extractor should not be limited to relationships already defined by the current ontology.

Doing so would cause the existing ontology to constrain what the system is capable of noticing.

The extraction step therefore answers a question closer to this.

What source supported propositions in this document could be useful for GOV.UK correctness, discoverability or coverage analysis?

The extractor returns candidate claims and their evidence spans.

It does not decide whether those claims fit the ontology.

It does not decide whether two mentions refer to the same entity.

It does not need to normalise temporal language.

These are later steps.

## Claim granularity

The system should prefer claims that can be independently supported, challenged and temporally qualified.

This usually means decomposing compound prose into smaller propositions.

The system must not assume that every useful claim can be represented as one simple subject predicate object triple.

Government knowledge contains appointments, eligibility conditions, exceptions, thresholds, obligations and other structures that often require an intermediate entity or event.

The exact rules for claim granularity remain a spike.

## Entity resolution

Entity resolution is an explicit replaceable component.

The initial implementation will use an LLM.

Its input and output contract must not depend on the LLM implementation.

A future resolver could therefore use Senzing, deterministic identifiers, another entity resolution system or a hybrid approach.

The resolver receives mentions, evidence context and a bounded set of relevant candidate entities.

For each mention it returns one of three broad results.

1. Resolve to an existing entity.

2. Create a new provisional entity.

3. Leave the mention unresolved.

Merges and splits must be reversible.

The fact that two mentions were judged to refer to one entity must itself have provenance.

## Time

Time is first class.

Temporal information participates in claim interpretation, conflict detection, querying and coverage analysis.

The system must distinguish at least source publication time, system processing time, event time and the period during which a claim applies.

These times must not be conflated.

Natural language temporal expressions are handled through an explicit temporal resolution component.

For example, expressions such as `next April`, `the next financial year` or `three months after commencement` are temporal mentions.

The temporal resolver should convert each expression to the strongest machine readable and comparable temporal constraint justified by the evidence.

Where a precise interval can be established, the resulting representation should be precise.

Where only bounds can be established, the representation should retain those bounds rather than inventing an exact date.

Where the expression cannot yet be resolved, it remains temporally unresolved.

The original temporal expression and the evidence supporting it must always be retained.

OWL Time provides the canonical vocabulary for resolved temporal entities and relationships.

Temporal resolution is conceptually similar to entity resolution.

It may depend on context already present in the knowledge graph.

A later fact may allow a previously unresolved temporal expression to become resolvable without running open claim extraction again.

## Fixed ontology foundation

The system begins with a small fixed semantic foundation.

This exists to prevent the emergent ontology from unnecessarily reinventing basic semantic machinery.

The initial foundation should include suitable parts of the following.

1. RDF.

2. RDFS.

3. OWL.

4. SHACL.

5. PROV O.

6. SKOS.

7. W3C ORG.

8. Selected EU Core Vocabularies.

9. Web Annotation.

10. Nanopublication vocabularies.

11. OWL Time.

This foundation does not define GOV.UK domain concepts such as Permanent Secretary, consultation, funding scheme or ministerial portfolio.

Those concepts must emerge from the evidence or be adopted from external ontologies.

## External ontologies

The agent may research and reuse external ontologies.

There is no fixed whitelist beyond the foundation.

The ontology research step should ask whether an established model represents a required concept better than inventing a local one.

The ontology critic should also periodically consider whether the ontology has evolved into an unnecessarily parochial or awkward structure.

An external ontology may therefore be introduced during later refactoring even if the system originally created local concepts.

External ontologies used by the graph must be vendored into the repository.

The exact source, version where available, retrieval time and content hash must be recorded.

The agent must never rely on the current content of a remote ontology URL as part of the reproducibility boundary.

## Ontology evolution

Ontology evolution is autonomous.

There is no human review step during the first experiment.

The ontology agent may create new concepts, reuse external concepts, specialise external concepts, change mappings, deprecate local concepts and refactor earlier modelling choices.

It must record why it made each change.

Ontology releases and SHACL shape releases are conceptually linked but independently identified.

Each release must pass deterministic mechanical checks before becoming current.

These checks include valid RDF, valid SHACL, valid references and any required migration consistency checks.

The deterministic checks do not decide whether the ontology is intellectually good.

That responsibility belongs to the ontology reasoning stages.

## Ontology evolution workflow

Ontology evolution contains several distinct reasoning roles.

The ontology researcher looks for relevant external models.

The ontology proposer suggests the smallest useful change that addresses current evidence.

The ontology critic searches for conceptual errors, unnecessary invention, poor reuse, overfitting, accidental synonyms and larger structural problems.

The ontology synthesiser produces the proposed release after considering the critique.

The resulting release is then subjected to deterministic checks.

If it passes, it becomes current.

Claims affected by the change are reconsidered.

## Ontology pressure and annealing

The first experiment will run the ontology review step after every document.

This deliberately avoids inventing an annealing policy before evidence exists.

The system will still calculate and store possible ontology pressure signals after each document.

These observations will later be used to determine whether an adaptive trigger can replace the simple always review policy.

Candidate signals include gap count, gap rate, number of gap clusters, cluster entropy, size of the largest cluster, share represented by the largest cluster, number of new clusters and the rate at which new concepts appear.

No single metric is assumed to be correct.

In particular, high entropy does not always imply strong ontology pressure.

One large recurring unresolved concept may have low entropy while still presenting a compelling reason to change the ontology.

## Acceptance pipeline

Candidate claims pass through an explicit pipeline.

The initial pipeline is as follows.

1. Extract source supported candidate claims.

2. Resolve entity mentions.

3. Resolve temporal mentions.

4. Map the claim to the current ontology.

5. Construct a candidate RDF change.

6. Validate the current graph plus the candidate change using SHACL and other deterministic checks.

7. Detect epistemic conflicts against current accepted knowledge.

8. Consider temporal overlap when determining whether claims conflict.

9. Apply scope and acceptance policy.

10. Record the decision.

11. Project accepted knowledge into the canonical graph.

A claim may remain unresolved at several stages.

Failure to fit the ontology is not equivalent to falsehood.

Failure to satisfy a SHACL shape is not equivalent to contradiction.

Two sources disagreeing is not automatically a logical inconsistency.

The system must preserve these distinctions.

## Claim dispositions

The initial dispositions should distinguish at least the following cases.

1. Accepted.

2. Ontology gap.

3. Constraint violation.

4. Conflict.

5. Entity unresolved.

6. Time unresolved.

7. Out of scope.

8. Low confidence.

9. Rejected.

10. Superseded.

These dispositions may evolve if the experiment reveals a better classification.

They must not be reduced to accepted and rejected.

## Conflict and correctness

Conflict detection should compare candidate claims against accepted knowledge within relevant temporal scope.

Claims that apply to different time periods may represent legitimate succession rather than contradiction.

The system should expose conflicts with enough evidence to allow later analysis.

The graph should support discovering potentially false, contradictory or outdated GOV.UK claims.

It should not claim that a statement is false solely because another source disagrees.

Any stronger judgement of falsehood needs an explicit evidential or policy basis.

## Content coverage

Documents are part of the graph because content coverage is a core use case.

The system should support analysis such as the following.

Which important concepts have no guide or explanatory page?

Which important concepts are covered only by stale guidance?

Which central concepts are supported only by pages whose underlying policy claims have expired?

Which structurally important concepts depend on one fragile bridge between otherwise separated parts of the knowledge graph?

The graph must contain enough relationships between documents, claims, concepts, entities and source classifications to construct these analyses.

## Analytical graph projections

Measures such as PageRank, bridges and articulation points depend on the graph projection being analysed.

They are not intrinsic properties of an entity.

The system must therefore define and version analytical projections.

The canonical semantic graph remains directed.

PageRank may operate on a directed semantic projection.

Classical bridge and articulation analysis requires a suitable undirected projection.

Each analytical result must identify the graph state, projection definition and algorithm version from which it was derived.

Analytical results are derived data.

They do not become source claims.

## Experimental telemetry

The first run should produce enough observations to study how ontology formation changes as the graph grows.

A deterministic state snapshot will be taken before and after every document.

The observation should include measures from the incoming document, claim queue, ontology and graph state.

### Document observations

Useful measures include document type, schema, semantic content function, taxons, organisations, text size and temporal complexity.

### Claim observations

Useful measures include claims extracted, accepted claims, ontology gaps, conflicts, unresolved entities, unresolved times, out of scope claims, novel predicates and novel entity types.

### Ontology observations

Useful measures include class count, property count, axiom count, external terms used, ontology changes introduced and claims reclassified by a release.

### Gap queue observations

Useful measures include gap count, gap rate, cluster count, cluster entropy, largest cluster size, largest cluster share, new cluster count and source diversity within each cluster.

### Graph observations

Useful measures include semantic node count, semantic edge count, connected component count, largest component share, degree statistics, PageRank distribution, bridge count and articulation point count.

### Evolution observations

Every ontology change must record the release before and after the change, the triggering document, all claim clusters considered, supporting claims, supporting documents, additions, removals, migrations, reused external terms and the reasoning produced by the proposer, critic and synthesiser.

The triggering document and the causal evidence set must remain distinct.

A document may cause the ontology review to run while the actual ontology change is supported by evidence accumulated across many earlier documents.

## Research questions

The first experiment should leave behind a dataset that supports at least the following research questions.

### Ontology annealing

How quickly does ontology change decrease as more content is processed?

Which measurements change as the ontology stabilises?

Does gap entropy predict ontology change?

Does cluster size predict ontology change better?

Does the arrival of a new domain produce a recognisable increase in ontology pressure?

### Ontology history

How did the ontology evolve?

Which claims caused each concept to appear?

Which locally created concepts were later replaced by established external concepts?

Which ontology changes caused previously unresolved claims to become accepted?

### Document effects

What kinds of documents cause large ontology changes?

Do policy documents cause more changes than guides?

Does temporal complexity explain ontology change better than document category?

Do certain GOV.UK schemas consistently introduce new semantic structures?

### Graph state effects

Which properties of the existing graph make a large ontology change more likely?

Does a fragmented graph behave differently from a mature connected graph?

Does ontology change correlate with the arrival of claims that connect previously separate areas?

### Path dependence

Would processing the same corpus in a different order lead to a materially different ontology?

This does not need to be tested in the first run.

The first run must retain enough information to make a later ordering experiment straightforward.

## Agent orchestration

LangGraph is the initial orchestration tool.

The system is not one unconstrained conversational agent.

It is a deterministic workflow containing bounded LLM reasoning steps.

The initial workflow contains roughly the following stages.

```text
load document
↓
snapshot state before processing
↓
extract open claims
↓
resolve entities
↓
resolve time
↓
map claims to ontology
↓
construct candidate graph changes
↓
validate
↓
detect conflicts
↓
record claim decisions
↓
project accepted knowledge
↓
measure ontology pressure
↓
research external ontologies
↓
propose ontology revision
↓
critique ontology revision
↓
synthesise ontology release
↓
run mechanical checks
↓
release ontology if changed
↓
reconsider affected claims
↓
snapshot state after processing
↓
load next document
```

For the first experiment, ontology consideration occurs after every source document.

Later versions may replace this with adaptive routing based on measured ontology pressure.

## Command line boundaries

Content acquisition and graph construction are separate commands.

The fetch command should enumerate and download GOV.UK Content API material.

The run command should operate only on already downloaded evidence.

A directory run must use a recorded manifest order rather than arbitrary filesystem ordering.

This matters because ontology construction may be path dependent.

The manifest should preserve the exact corpus sequence.

## Version control

Git is the durable experiment record.

The following should be committed.

1. Source code.

2. Prompts.

3. Configuration.

4. GOV.UK Content API JSON.

5. Canonical text derived from each GOV.UK source.

6. Evidence manifests.

7. Ontology releases.

8. SHACL releases.

9. Vendored external ontology artifacts.

10. External ontology metadata and hashes.

11. Structured LLM outputs where required for reproducibility.

12. Claim decisions.

13. Ontology reasoning outputs.

14. Experimental observations.

15. Exported RDF dataset releases where useful.

Runtime databases are not authoritative and should not be committed.

The Oxigraph working store and LangGraph checkpoint database should be disposable.

Deleting them and rebuilding from version controlled material must reproduce the experiment state to the greatest degree permitted by model reproducibility.

## Reproducibility and model behaviour

Exact reproducibility is difficult when hosted LLM implementations can change.

Every LLM invocation must therefore record enough information to understand and, where possible, replay it.

This includes model identifier, provider, model parameters, prompt or prompt hash, structured input and structured output.

The first experiment should retain model outputs rather than assuming that calling the same model again will reproduce them.

The repository should distinguish replaying a previously recorded run from rerunning the reasoning process.

## Acceptance questions

The first experiment should answer at least the following questions.

### Knowledge

What do we know about the Permanent Secretary?

The answer must be derived from the ontology that actually emerged rather than from a predetermined Permanent Secretary model.

### Knowledge that does not fit

What source supported claims are not represented in the accepted graph, and why?

The answer should distinguish ontology gaps, conflicts, unresolved entities, unresolved time, scope decisions and other relevant dispositions.

### Missing explanatory coverage

Which articulation points, bridge connected concepts or high PageRank concepts have no evidence from content that the graph considers to be a guide or explanatory page?

The analysis must use semantic content function rather than rely only on one raw GOV.UK document type.

### Stale coverage

Which central concepts are supported only by pages whose underlying policy claims have expired?

Which central concepts have guide coverage whose relevant claims are no longer current?

### Ontology history

How did the ontology evolve and why?

Which evidence drove each change?

What knowledge became representable after each change?

### Annealing

How did ontology change magnitude, ontology gap rate, gap entropy, novelty and graph structure change over the course of the corpus?

### Rebuild

Can all runtime stores be deleted and reconstructed from the repository while preserving the recorded ontology history, claims, evidence and RDF dataset?

## Spike programme

The following spikes are deliberately separated from implementation tasks.

Each spike exists to answer a question that the current design does not yet resolve.

## Claim representation spikes

### Spike 1 claim granularity

Question

What is the smallest useful claim unit for GOV.UK content?

Questions within the spike

1. When should one sentence produce several claims?

2. When should an event or intermediate entity represent an n ary relationship?

3. How should conditions and exceptions attach to a claim?

4. How should a claim about another claim be represented?

Experiment

Process a small set of policy, guidance, appointment and eligibility pages with several candidate decomposition strategies.

Compare whether the resulting claims can be individually supported, temporally qualified and queried without losing meaning.

Evidence to collect

Number of extracted claims, number of compound claims later requiring decomposition, query complexity, ontology complexity and the amount of duplicated evidence.

Exit condition

Choose a claim contract that covers the common structures in the sample without forcing every statement into a simple triple.

### Spike 2 evidence anchoring

Question

Can evidence positions remain stable and useful across extraction, replay and source updates?

Questions within the spike

1. Is deterministic canonical text enough?

2. Are line positions and character positions both useful?

3. How should HTML tables, lists and headings be represented?

4. How should a changed GOV.UK source be related to evidence extracted from an earlier version?

Experiment

Canonicalise several difficult Content API documents and verify exact evidence references before and after rebuilding the repository.

Exit condition

A claim can always be traced to an exact source passage in the exact source version used during extraction.

## Resolution spikes

### Spike 3 entity resolution context

Question

How much graph context should the entity resolver receive?

Questions within the spike

1. How should candidate entities be generated?

2. When should a new provisional entity be preferred over a weak merge?

3. What confidence should be required for merging?

4. How should later evidence split an incorrect merge?

Experiment

Use pages with repeated mentions of people, departments, programmes and ambiguous organisations.

Run the resolver with several sizes of candidate context.

Exit condition

A stable resolver interface and a default context strategy that avoids obvious duplicate creation without aggressively merging unrelated entities.

### Spike 4 temporal resolution

Question

How should natural language time be converted into machine comparable temporal values?

Questions within the spike

1. How should expressions such as next April be anchored?

2. How should financial years be resolved?

3. How should dates relative to legislation, appointments or other events be resolved?

4. How should incomplete but bounded expressions be represented?

5. When must temporal resolution remain unresolved?

Experiment

Collect temporal expressions from the first corpus and resolve them using the current graph and document context.

Compare LLM resolution with deterministic calculations for cases where the correct answer is known.

Exit condition

A stable temporal resolver contract and a normalized representation suitable for temporal comparison and OWL Time projection.

## Ontology formation spikes

### Spike 5 ontology pressure

Question

What signals predict when another ontology evolution pass is useful?

Questions within the spike

1. Does gap entropy predict ontology change?

2. Does largest cluster size matter more?

3. Does gap rate matter?

4. Does the arrival of a new domain have a detectable signature?

5. Can ontology review frequency safely decrease as the corpus matures?

Experiment

Run ontology consideration after every document.

Record all proposed pressure metrics without using them to control execution.

After the run, compare the measurements with actual ontology changes.

Exit condition

Either identify a useful adaptive trigger or conclude that another experiment is needed.

### Spike 6 ontology refactoring

Question

How much freedom should the ontology agent have to replace earlier modelling decisions?

Questions within the spike

1. When may classes be split or merged?

2. When may a local concept be replaced by an external ontology term?

3. How are accepted claims migrated?

4. Can the agent recognise that a locally coherent ontology is globally awkward?

5. How much historical knowledge should be revalidated after a substantial refactor?

Experiment

Allow the ontology critic to propose structural refactors during the first run and inspect the proposals and migrations.

Exit condition

Define the minimum safeguards needed for autonomous ontology refactoring without introducing human approval.

### Spike 7 external ontology judgement

Question

Can the agent reliably decide whether an external ontology is better than local invention?

Questions within the spike

1. How does it judge semantic fit?

2. How does it detect an ontology that is technically usable but conceptually poor?

3. When should one external term be reused rather than importing a larger model?

4. How should licence, provenance and version stability affect the decision?

Experiment

Provide several ontology gaps for which good external models exist and several where available models are poor fits.

Exit condition

A simple external ontology review policy and a useful ontology critic prompt.

### Spike 8 path dependence

Question

How sensitive is the emergent ontology to corpus order?

This spike is not required for the first run.

Experiment

Replay the same recorded evidence in several deterministic orders while keeping all other inputs fixed.

Compare ontology structures, external dependencies, claim acceptance and final query results.

Exit condition

Measure whether path dependence is small enough to ignore or important enough to affect future orchestration.

## Validation and conflict spikes

### Spike 9 SHACL boundary

Question

Which rules belong in SHACL and which belong in other forms of reasoning?

Questions within the spike

1. Which observed regularities should become constraints?

2. Which constraints should remain soft?

3. How should shapes change with the ontology?

4. How do we stop the ontology agent turning accidental corpus patterns into universal rules?

Experiment

Review shapes created during the first twenty to fifty documents and classify their failures.

Exit condition

A practical distinction between graph shape validation, logical semantics and epistemic conflict.

### Spike 10 temporal conflict

Question

Can the system reliably distinguish contradiction from legitimate change over time?

Experiment

Construct or locate claims involving office holders, policy thresholds, eligibility rules and programme dates that change over time.

Compare conflict detection before and after temporal normalization.

Exit condition

Conflict logic only reports contradiction where applicable temporal scopes overlap or where another explicit relation justifies the conflict.

## Content semantics spikes

### Spike 11 source vocabulary alignment

Question

Can the agent discover stable semantic content functions across inconsistent source classifications?

Questions within the spike

1. When are two source page categories functionally equivalent?

2. When is one narrower than another?

3. How confidently can the system make these mappings?

4. How often do mappings change as additional departments are introduced?

Experiment

Start with GOV.UK source classifications from the first department.

Later include a small sample from another department with different source terminology.

Exit condition

A mapping approach that preserves source vocabulary while supporting useful cross departmental semantic categories.

### Spike 12 evidence coverage

Question

What does it mean for an important concept to have guide coverage?

Questions within the spike

1. Is a mention enough?

2. Must the page explain the concept?

3. Must accepted claims about the concept form a minimum level of coverage?

4. Must the page itself have a guide content function?

5. How should several weak pages compare with one strong guide?

Experiment

Take a small set of concepts and manually inspect the pages that the graph considers supporting guidance.

Exit condition

A versioned analytical definition of explanatory coverage suitable for the centrality queries.

### Spike 13 stale coverage

Question

When should guide coverage be considered stale?

Questions within the spike

1. Is a page stale when any major claim is expired?

2. Does a replacement policy claim make an older guide stale even if the guide contains no explicit end date?

3. What portion of a page must be stale before the page itself is considered stale?

Experiment

Use several examples where policy changed while explanatory content persisted.

Exit condition

A derived stale coverage rule that uses claim level time rather than publication age alone.

## Graph analytics spikes

### Spike 14 analytical projection

Question

Which nodes and relationships should participate in PageRank, bridge and articulation analysis?

Questions within the spike

1. Should documents participate directly in semantic centrality?

2. Should claim resources participate or only their semantic projections?

3. Should common generic predicates carry less weight?

4. How should direction be handled for each metric?

5. Should ontology nodes participate?

Experiment

Build several projections from the same graph and compare whether the resulting important concepts appear meaningful.

Exit condition

One or more explicitly versioned graph projections with clear intended interpretations.

### Spike 15 graph state and ontology change

Question

Which properties of the current graph predict how strongly a new document changes the ontology?

Experiment

Use the pre processing and post processing observations from every document to construct a dataset containing document features, claim features, ontology features and graph features.

Model or inspect the relationship between those inputs and subsequent ontology changes.

Exit condition

Identify useful explanatory variables or confirm which further measurements should be collected in a larger run.

## Reproducibility spikes

### Spike 16 LLM replay

Question

What does reproducibility mean when the underlying hosted model may change?

Experiment

Support two execution modes.

One mode replays recorded structured LLM outputs.

The other reruns the model from recorded inputs.

Compare the resulting RDF dataset and ontology history.

Exit condition

A clear distinction between exact experiment replay and fresh reasoning replay.

### Spike 17 one pass semantics

Question

What may be reconsidered after extraction without violating the one pass corpus rule?

The proposed rule is that open extraction runs once.

Entity resolution, time resolution, ontology mapping, validation and acceptance may run repeatedly.

No new claim may be silently invented from the original document during reconsideration.

Experiment

Observe whether ontology evolution reveals useful propositions that the original open extraction did not capture.

Exit condition

Either confirm the rule or identify a justified additional extraction stage with explicit provenance.

## Operational spikes

### Spike 18 failure behaviour

Question

How should the first autonomous run behave when one reasoning step fails?

Questions within the spike

1. Should invalid structured output be retried?

2. What happens when an ontology proposal fails mechanical checks?

3. What happens when an external ontology disappears during retrieval?

4. Can one problematic document stop the whole run?

For the first experiment, the bias should be towards failing loudly rather than hiding errors.

Exit condition

A minimal failure policy that preserves evidence and makes the run diagnosable without building a production workflow engine.

## Initial implementation

The minimal local stack is expected to be Python, Typer, LangGraph, an LLM client, PyOxigraph, pySHACL and Git.

The content downloader is a deterministic application rather than an agent.

The graph workflow is autonomous once invoked.

The first 100 document run intentionally favours observability over efficiency.

Expensive graph metrics may be calculated after every document.

Ontology consideration may run after every document.

Intermediate reasoning may be retained extensively.

These choices would be inappropriate at full GOV.UK scale but are useful because the purpose of the first run is to understand the behaviour of the system.

## Definition of done for the first experiment

The experiment is complete when approximately 100 heterogeneous GOV.UK documents have been processed once through open claim extraction and the resulting repository can demonstrate all of the following.

1. An ontology emerged without a hand authored GOV.UK domain model.

2. Claims are individually traceable to exact source evidence.

3. Entities and temporal expressions have explicit resolution histories.

4. Claims that do not fit remain queryable.

5. Ontology releases explain what changed and why.

6. The graph can answer what is known about the Permanent Secretary.

7. The graph can identify important concepts without explanatory guide coverage.

8. The graph can identify central concepts supported only by expired policy claims or stale guidance.

9. The run contains enough observations to analyse ontology annealing.

10. The run contains enough observations to study which incoming documents and graph states caused large ontology changes.

11. Runtime state can be destroyed and reconstructed from version controlled artifacts.

12. The unresolved spikes are supported by concrete evidence from the run rather than speculation.
