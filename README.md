# Forward Deployed AI Engineer — Master Roadmap & Curriculum

A complete, self-paced, hands-on curriculum for becoming a Forward Deployed AI Engineer (FDE) — someone who takes Generative AI from theory to real, deployed, client-facing systems. This repository is both a learning path and a growing personal knowledge base: every topic is written as a plain-language, example-driven guide, backed by current research and industry practice, with an interview-question bank attached to each one.

If you are new here, start with Module 1 and move through the modules in order. Each module builds directly on the concepts introduced in the ones before it.

---

## What This Repository Is

This repo covers the full journey from understanding how a Large Language Model works internally, to integrating it via API, to building retrieval systems and agents on top of it, to fine-tuning and deploying models, to the specific discipline of Forward Deployed Engineering — scoping, deploying, and supporting AI systems inside a real client's environment. A self-paced System Design and Data Structures & Algorithms track is included as an elective for technical interview preparation.

Every topic in this repository follows the same format:

- A plain-language explanation of the concept, written for a beginner but held to a technically accurate standard.
- A concrete, real-world example or analogy for every major idea introduced.
- Clear structure, so the material is easy to scan, reference, and revisit.
- A short section on how the topic has evolved recently, so the material stays current rather than frozen at whatever the concept looked like when it was first written.
- An interview-question bank with full answers, so the same material doubles as interview preparation.

---

## Curriculum Overview and Progress Tracker

| Module | Module Name | Duration | Capstone Project | Status |
| --- | --- | --- | --- | --- |
| 01 | [LLM Engineering & Model Integration](./01-llm-engineering-and-model-integration/README.md) | 3 Weeks | AI Interview Coach | In Progress |
| 02 | [Backend & API Engineering](./02-backend-and-api-engineering/README.md) | 2 Weeks | Personal AI Morning Briefing | Not Started |
| 03 | [Retrieval Systems & Vector Intelligence](./03-retrieval-systems-and-vector-intelligence/README.md) | 3 Weeks | Chat with Your Own Documents | Not Started |
| 04 | [Advanced Retrieval & Model Specialization](./04-advanced-retrieval-and-model-specialization/README.md) | 3 Weeks | Train Your Own Mini AI on Custom Data | Not Started |
| 05 | [Agentic & Multi-Agent Systems](./05-agentic-and-multi-agent-systems/README.md) | 4 Weeks | Personal AI Assistant That Works On Its Own | Not Started |
| 06 | [Cloud & DevOps](./06-cloud-and-devops/README.md) | 2 Weeks | Put Your AI on the Internet | Not Started |
| 07 | [Forward Deployed Engineering](./07-forward-deployed-engineering/README.md) | 5 Weeks | Run a Full Client Engagement End to End | Not Started |
| 08 | [Electives (Self-Paced)](./08-electives/README.md) | Self-Paced | System Design & DSA Tech Interviews | Not Started |

Update the Status column as you complete each module: Not Started, In Progress, or Complete.

---

## Repository Structure

```text
Forward-Deployed-Engineer-Roadmap/
│
├── README.md                                          <- you are here
│
├── 01-llm-engineering-and-model-integration/
│   ├── README.md
│   ├── notes/
│   │   ├── 01-the-evolution-of-ai.md
│   │   ├── 02-how-llms-actually-work.md
│   │   ├── 03-from-raw-llms-to-real-applications.md
│   │   ├── 04-llm-fundamentals-kv-cache-moe-reasoning-tools.md
│   │   └── ...
│   └── project-ai-interview-coach/
│
├── 02-backend-and-api-engineering/
│   ├── README.md
│   ├── notes/
│   └── project-personal-ai-morning-briefing/
│
├── 03-retrieval-systems-and-vector-intelligence/
│   ├── README.md
│   ├── notes/
│   └── project-chat-with-your-own-documents/
│
├── 04-advanced-retrieval-and-model-specialization/
│   ├── README.md
│   ├── notes/
│   └── project-train-your-own-mini-ai/
│
├── 05-agentic-and-multi-agent-systems/
│   ├── README.md
│   ├── notes/
│   └── project-personal-ai-assistant/
│
├── 06-cloud-and-devops/
│   ├── README.md
│   ├── notes/
│   └── project-put-your-ai-on-the-internet/
│
├── 07-forward-deployed-engineering/
│   ├── README.md
│   ├── notes/
│   └── project-full-client-engagement/
│
└── 08-electives/
    ├── README.md
    ├── system-design/
    └── dsa/
```

Each module folder holds its own `README.md` (an index of that module's topics), a `notes/` folder containing one file per topic in the style described above, and a `project-*` folder for that module's hands-on capstone.

---

## Module Breakdown and Syllabus

### Module 1: LLM Engineering & Model Integration (3 Weeks)

**Capstone Project:** AI Interview Coach

- **LLM Fundamentals** — Transformer architecture, the attention mechanism, KV Cache, Mixture of Experts (MoE), reasoning models (o1, Claude Extended Thinking), the LLM ecosystem (ChatGPT, Claude, Copilot, Cursor), and no-code tools (Bolt, Lovable, v0, n8n).
- **Prompt Design** — Zero-shot and few-shot prompting, role/task/context/format/persona design, Chain-of-Thought (CoT), Tree of Thought (ToT), self-consistency, reflection and plan-and-execute patterns, prompt chaining, system prompt architecture, and prompt injection defense.
- **Advanced Techniques** — Function calling and structured outputs (Pydantic, Instructor, JSON Schema), tokenization and cost optimization (tiktoken, BPE, prompt compression), automatic prompt optimization with DSPy, the ReAct pattern, safety prompting, multimodal inputs, and working across the OpenAI, Anthropic, and Gemini APIs, including prompt versioning.

### Module 2: Backend & API Engineering (2 Weeks)

**Capstone Project:** Personal AI Morning Briefing

- **Python Foundations** — Core Python, file handling, JSON, object-oriented design and custom error handling, async Python (`asyncio`, `async`/`await`), environment management with `venv` and `python-dotenv`, secrets hygiene, Git workflows, and clean project structure (`src/`, `prompts/`, `tests/`, `notebooks/`).
- **APIs & SDKs** — HTTP fundamentals (`requests`, status codes, pagination, JSON parsing), the OpenAI SDK (chat completions and streaming), the Anthropic SDK (prompt caching via `cache_control`), and hands-on prompt compilation with DSPy.

### Module 3: Retrieval Systems & Vector Intelligence (3 Weeks)

**Capstone Project:** Chat with Your Own Documents

- **Embeddings & Retrieval** — Embedding models (`text-embedding-3`, `sentence-transformers`, BGE), cosine similarity, vector databases (Pinecone, ChromaDB, FAISS), and document chunking strategies.
- **RAG Pipeline** — Building a retrieval chain with LangChain (LCEL, chains, retrievers, prompt templates), designing the full ingest-to-generate pipeline, evaluating a production RAG system with RAGAS (faithfulness, answer relevancy, context recall), and context window strategies (stuffing, map-reduce, refine, and the "lost in the middle" problem).
- **Hybrid Search** — BM25 keyword search, Hypothetical Document Embeddings (HyDE), Reciprocal Rank Fusion (RRF), and cross-encoder reranking.

### Module 4: Advanced Retrieval & Model Specialization (3 Weeks)

**Capstone Project:** Train Your Own Mini AI on Custom Data

- **Advanced RAG** — Query rewriting, multi-hop retrieval, cross-encoder reranking in a full pipeline, and self-correcting RAG variants (CRAG, Self-RAG, Agentic RAG).
- **Fine-Tuning** — LoRA and QLoRA (with `bitsandbytes` 4-bit quantization), adapters and prefix tuning (HuggingFace `transformers`, PEFT, `datasets`), configuring training runs with Axolotl, renting GPUs (RunPod, vast.ai), knowledge distillation, RLHF and DPO (`trl`), and tracking training with Weights & Biases.
- **Local Deployment** — Running open models locally with Ollama, GGUF quantization levels (`Q4_K_M`, `Q8_0`), working with Llama 3, Phi-3, and Mistral, and reasoning about model selection and cost-performance trade-offs.

### Module 5: Agentic & Multi-Agent Systems (4 Weeks)

**Capstone Project:** Personal AI Assistant That Works On Its Own

- **Agent Architecture** — The ReAct loop, the planner-executor pattern, the tool arbiter pattern, function calling and the Model Context Protocol (MCP), and agent tool use including live search and parallel tool calls.
- **Frameworks** — LangGraph (state machines, nodes, edges, checkpointing, human-in-the-loop steps), CrewAI (role-based multi-agent teams, sequential versus hierarchical workflows), Agentic RAG, and workflow automation with n8n.
- **Memory & Safety** — Short-term, long-term, and episodic memory, multi-agent orchestration patterns (supervisor, peer-to-peer, fault isolation), evaluation frameworks (RAGAS, DeepEval, PromptFoo, trajectory evaluation), and safety layers including Guardrails AI, prompt injection defense, PII scrubbing, and output filtering.

### Module 6: Cloud & DevOps (2 Weeks)

**Capstone Project:** Put Your AI on the Internet

- **Infrastructure & DevOps** — Building production APIs with FastAPI (async endpoints, SSE streaming, auth middleware, rate limiting), containerizing with Docker, setting up CI/CD with GitHub Actions (including prompt regression testing), and deploying on AWS (EC2, ECS Fargate, ECR, S3, RDS PostgreSQL, Spot instances, auto-scaling).
- **Observability & Safety** — Tracing and monitoring with LangSmith, moderation layers with Guardrails AI and the OpenAI moderation API, running evaluation suites (DeepEval, PromptFoo) in CI, and the fundamentals of LLMOps.

### Module 7: Forward Deployed Engineering (5 Weeks)

**Capstone Project:** Run a Full Client Engagement End to End

- **Discovery & Scoping** — Separating the stated problem from the actual problem, qualifying whether an LLM or deterministic software is the right tool, scoping under political and budget constraints, defining non-goals and acceptance criteria, and writing engagement artefacts (PRD-lite, SOW, ADRs) alongside baseline capture and success metric design.
- **Enterprise Data Access** — Working with legacy databases (read replicas, change data capture), reverse-engineering undocumented schemas, entity resolution across fragmented sources, PII identification and masking at extraction, and building connectors (REST, gRPC, OAuth service accounts) that handle pagination, rate limits, and incremental sync.
- **Deploying Into Someone Else's Environment** — VPC-only and air-gapped inference, enterprise authentication (SSO, SAML, OIDC, RBAC), multi-tenant isolation, model placement trade-offs (hosted, in-tenant, self-hosted), scaling inference and vector search past ten million documents, and cost modeling.
- **Compliance, Security & the Human Loop** — The basics of the DPDP Act, GDPR, and data residency, audit logging and model cards, SOC 2 and ISO 27001 evidence requests, prompt injection risk in multi-tenant contexts, human-in-the-loop approval gates, and incident response inside a client's live environment.
- **Stakeholder Defense & Handover** — Defending architecture decisions under review, articulating trade-offs to client architects, demoing failure modes deliberately to build trust, handling questions you cannot answer, writing runbooks for client team enablement, and tracking a pilot's path to rollout and impact.

### Module 8: Electives (Self-Paced)

- **System Design for Tech Interviews** — Load balancing, caching, and CDN strategy; SQL versus NoSQL, sharding, and replication; consistency models and the CAP theorem; message queues and event-driven architecture; rate limiting, retries, and circuit breakers; capacity estimation; and classic design problems (URL shortener, key-value store, news feed, chat/notification systems, search autocomplete).
- **DSA for Tech Interviews** — Arrays, strings, two pointers, and sliding window; hashing, prefix sums, and frequency maps; stacks, queues, and monotonic structures; linked lists, trees, BSTs, and traversals; heaps, tries, and union-find; recursion, backtracking, and pruning; binary search on the answer; and graphs (BFS, DFS, shortest paths) and dynamic programming (1D, 2D, and state design).

---

## How to Use This Repository

1. Work through the modules in order — later modules assume the concepts covered in earlier ones.
2. Inside a module, read the notes in the `notes/` folder before attempting that module's capstone project — the notes are written to be read first, then applied.
3. Build the capstone project inside that module's `project-*` folder as you go, rather than waiting until the end of the module.
4. Update the progress tracker at the top of this file as you complete each module.
5. Each individual note file ends with an interview-question bank — use these both to check your own understanding and as direct interview preparation.

---

## Notes on How This Curriculum Stays Current

Generative AI moves quickly, and a static curriculum written once and never revisited goes stale fast. Every note file in this repository includes a short, clearly labeled section covering what has changed most recently in that specific topic, based on current research and industry sources at the time the note was written. Treat that section as a snapshot rather than a permanent statement — it is worth periodically checking whether anything material has shifted since a given note was last updated, especially in fast-moving areas like model architecture, tooling, and pricing.

---

## Attribution

This roadmap is based on the Forward Deployed AI Engineer curriculum from Bosscoder Academy, adapted here into a self-contained, example-driven note format with an added interview-question bank for each topic.