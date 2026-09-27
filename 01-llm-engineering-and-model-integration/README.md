# Module 1 – LLM Engineering & Model Integration (3 Weeks)

> **Capstone Project**: [The Receipt & Expense Tracker](./project-receipt-expense-tracker/README.md)  
> Built using Google Gemini Multimodal Vision API (Free Tier), Pydantic structured output validation, prompt injection defense, and deterministic arithmetic guardrails.

---

## 🎯 Module Objective
Master the foundations of Large Language Models, prompt engineering architectures, structured outputs, tool integration, token economics, and API integration across major providers (OpenAI, Anthropic, Gemini).

---

## 📋 Syllabus & Topic Breakdown

### 1. LLM Fundamentals
- [x] **Transformer Architecture**: Self-attention, multi-head attention, encoder-decoder vs decoder-only models.
- [x] **Attention & KV Cache**: Key-Value caching mechanics for inference efficiency.
- [x] **Mixture of Experts (MoE)**: Routing mechanisms, sparse activation models.
- [x] **Reasoning Models**: OpenAI o1, Claude Extended Thinking architectures.
- [x] **Ecosystem Tooling**: Strengths, weaknesses, and selection criteria for ChatGPT, Claude, Copilot, Cursor.
- [x] **No-Code / Low-Code AI Tools**: Rapid prototyping with Bolt, Lovable, V0, and n8n.

### 2. Prompt Design
- [x] **Prompt Engineering Basics**: Zero-shot, few-shot, role, task, context, format, persona & tone conditioning.
- [x] **Advanced Prompting Reasoning**: Chain-of-Thought (CoT), Tree of Thought (ToT), self-consistency sampling.
- [x] **Execution & Control**: Reflection, self-critique, Plan-and-Execute agentic patterns, prompt chaining patterns.
- [x] **Architecture**: System prompt architecture, state management across multi-turn design.
- [x] **Security**: Prompt injection defense, jailbreak mitigation, system prompt protection.

### 3. Advanced Techniques
- [x] **Structured Outputs & Tooling**: Function calling, tool-integrated LLMs, validation with Pydantic, `instructor`, `json_schema`.
- [x] **Tokenization & Cost Economics**: BPE tokenization (`tiktoken`), context window management, cost calculation, prompt compression.
- [x] **Automated Prompting**: Few-shot optimization & automatic prompt compilation with DSPy.
- [x] **Safety & Responsibility**: ReAct framework, safety prompting, guardrails, responsible AI principles.
- [x] **Multimodality & Application**: Multimodal inputs, document understanding, code review prompting.
- [x] **API Integration & Governance**: OpenAI, Anthropic, Gemini SDKs, prompt versioning and evaluation pipelines.

---

## 📚 Notes & Study Guides

Available study guides in [`notes/`](./notes/):
1. [01-the-evolution-of-ai.md](./notes/01-the-evolution-of-ai.md) — History, milestones, and the role of a Forward Deployed AI Engineer.
2. [02-how-llms-actually-work.md](./notes/02-how-llms-actually-work.md) — Deep dive into tokens, embeddings, attention mechanisms, and Transformers.
3. [03-from-raw-llms-to-real-applications.md](./notes/03-from-raw-llms-to-real-applications.md) — Transitioning from raw completion models to robust enterprise applications.
4. [04-llm-fundamentals-kv-cache-moe-reasoning-tools.md](./notes/04-llm-fundamentals-kv-cache-moe-reasoning-tools.md) — KV Cache, MoE, Reasoning models, and modern AI tooling.
5. [05-prompt-design.md](./notes/05-prompt-design.md) — Prompt levers, CoT, ToT, execution patterns, and prompt injection defense.
6. [06-advanced-techniques.md](./notes/06-advanced-techniques.md) — Structured outputs, cost optimization, DSPy, ReAct, and multimodal prompting.

---

## 🛠️ Capstone Project: The Receipt & Expense Tracker
- **Directory**: [`project-receipt-expense-tracker/`](./project-receipt-expense-tracker/)
- **Documentation**: [Project Guide & Architecture](./project-receipt-expense-tracker/README.md)
- **Technology Stack**: Python (FastAPI, Google Gemini Vision API `google-genai`, Pydantic, Pillow, SQLite) + Responsive Modern Web Dashboard.
- **Key Concepts Implemented**:
  - Multimodal document understanding (receipts photographed with phone camera)
  - Native Pydantic `response_schema` enforcement (no brittle regex or manual JSON parsing)
  - System prompt boundary defense (protecting against printed prompt injections)
  - Image downsampling with Pillow (reducing input tokens and API latency)
  - Deterministic arithmetic sanity guardrails (reconciling line items + tax with total)
