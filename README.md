# ai-gov-graph

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

## MVP todo

- [ ] Build the GOV.UK Content API downloader
- [ ] Create deterministic source text and evidence references
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
