# Topic 6: Advanced Techniques — Structured Outputs, Cost Optimization, DSPy, ReAct & Multimodal Prompting

> Part of the **Forward Deployed Engineering (FDE) Mastery Repo** — a one-stop learning path for becoming a Forward Deployed Engineer. This is Topic 6, the third and final subsection of **Module 1: LLM Engineering & Model Integration**. It builds on Topic 3 (raw API integration and tool-use), Topic 4 (KV Cache and reasoning models), and Topic 5 (prompt design and system prompt architecture) — the techniques here are the practical engineering layer that turns everything in those topics into something reliable enough for production.

![Advanced LLM Engineering Techniques](./assets/topic-06-advanced-techniques.jpg)

---

## Table of Contents

1. [Function Calling & Structured Outputs](#1-function-calling--structured-outputs)
2. [Tokenization & Cost Optimization](#2-tokenization--cost-optimization)
3. [DSPy: Treating Prompts as Optimizable Programs](#3-dspy-treating-prompts-as-optimizable-programs)
4. [ReAct, Safety Prompting, and Responsible AI](#4-react-safety-prompting-and-responsible-ai)
5. [Multimodal Inputs, Document Understanding, and Code Review Prompting](#5-multimodal-inputs-document-understanding-and-code-review-prompting)
6. [LLM APIs and Prompt Versioning](#6-llm-apis-and-prompt-versioning)
7. [2026 Industry Update](#7-2026-industry-update)
8. [Interview Questions & Answers](#8-interview-questions--answers)

---

## 1. Function Calling & Structured Outputs

### The Problem: LLMs Naturally Produce Free-Form Text

An LLM's raw output, by default, is unstructured prose — great for a chat response, terrible for feeding directly into another piece of software. If your code expects a clean JSON object with a `name` field and an `age` field, and the model instead replies with "Sure! The user's name is John and he's 30 years old," your program has no reliable way to extract that data without fragile string-parsing.

### Function Calling: Letting the Model Request an Action

**Function Calling** (also called Tool Calling) lets a developer describe available functions/tools to the model — their names, purposes, and expected parameters — so that instead of writing a free-form answer, the model can respond by specifying *which* function should be called and *with what arguments*, in a structured format the application can act on directly. This is the exact mechanism underlying the tool-use flow described in Topic 3's calculator/weather-API examples.

**Real-World Example:** This is like giving a new employee a laminated card listing every action they're allowed to formally request — "Request Refund (requires: order ID, reason)," "Escalate to Manager (requires: ticket ID, urgency)" — instead of letting them describe what they want in an email that someone else then has to interpret and act on manually. The laminated card format is fast, unambiguous, and directly actionable.

### Structured Outputs: Guaranteeing the Shape of the Answer

**Structured Outputs** go a step further than function calling: rather than the model choosing to call a tool, you directly force its entire response to conform to a specific schema — for example, guaranteeing every response is valid JSON matching an exact set of fields and types, every single time.

- **`json_schema` / JSON Mode:** The lowest-level mechanism — you provide a formal schema definition, and the provider constrains the model's output generation so it's structurally impossible (or heavily discouraged, depending on the provider's implementation) to produce anything that doesn't match.
- **Pydantic:** A widely used Python library for defining data models with type validation — instead of hand-writing a raw JSON Schema, developers describe the desired output as a Pydantic class, which is easier to read, write, and integrate with the rest of a Python codebase.
- **Instructor:** A popular open-source library that sits between your application code and any LLM provider, automatically converting a Pydantic model into whatever structured-output mechanism that specific provider supports (JSON mode, native structured outputs, or tool calling), and automatically validating and retrying if the model's first attempt doesn't match the schema.

**Real-World Example:** Structured Outputs are like a fill-in-the-blank government form instead of an open essay question — a form has fixed fields ("Full Legal Name: \_\_\_", "Date of Birth: \_\_\_") that are trivially machine-readable, whereas an open essay might mention the same information but in a way that requires a human (or fragile code) to hunt for it. Pydantic is the tool for designing what that form looks like in your own codebase, and Instructor is the clerk who takes your form design, figures out exactly how to phrase the request to whichever government office (LLM provider) you're dealing with, and automatically re-submits the request if the first response comes back incomplete.

### Why This Matters for Reliability

Without structured outputs, a downstream parsing failure (a missing field, a malformed number, an unexpected string where a boolean was expected) can silently break an entire pipeline — this is especially dangerous in **agentic** systems (covered in Module 5) where one step's malformed output becomes the next step's broken input. Structured outputs push this failure mode as far upstream as possible, catching and retrying it at the model-response layer instead of crashing deep inside application logic.

---

## 2. Tokenization & Cost Optimization

### Recap and What's New Here

Topic 2 explained *how* tokenization works (sub-word tokens via something like Byte Pair Encoding, or BPE) and Topic 3 explained *why* it matters for billing (input and output tokens are both charged). This section is about the practical engineering techniques an FDE actually uses to keep token costs under control in a real, high-volume production system.

### `tiktoken` and Counting Tokens Before You Send Them

`tiktoken` is a widely used tokenizer library that lets a developer count exactly how many tokens a given piece of text will consume for a specific model **before** sending it to the API — critical for staying under context-window limits and for estimating cost ahead of time, rather than discovering a request was too expensive or too long only after the fact.

### Prompt Compression: Saying the Same Thing With Fewer Tokens

**Prompt compression** techniques shrink a prompt's token count while trying to preserve its meaning, using either simple heuristics (removing redundant filler words, tightening instructions) or dedicated compression models (like LLMLingua) that are specifically trained to identify and strip low-information tokens from a prompt without materially degrading the model's ability to understand it.

**Real-World Example:** This is similar to how an experienced telegram writer in the era of pay-per-word telegrams would rewrite "I would like to inform you that I will be arriving at the airport tomorrow at three o'clock in the afternoon" as "ARRIVING AIRPORT TOMORROW 3PM" — same essential meaning, a fraction of the cost. A prompt compression tool does this automatically and at scale, across every request a production system sends.

### Prompt Caching: Not Paying Full Price for Repeated Content

Both Anthropic and OpenAI offer explicit **prompt/context caching**: if a large chunk of a prompt (a long system prompt, a big reference document, a set of few-shot examples) stays exactly the same across many requests, the provider can cache the processed version of that chunk and charge a significantly reduced rate for it on every subsequent call that reuses it, rather than charging full price to reprocess identical content over and over.

**Real-World Example:** This is like a courtroom that keeps a standing copy of the relevant law on file, rather than having a lawyer read the entire legal code out loud from scratch before every single case. Reading it once (caching it) and simply referencing "as established in the code we already have on file" for every subsequent case is dramatically faster and cheaper than repeating the full reading every time — this is exactly why an FDE building something like the Support Ticket Summarizer from Topic 3 would cache the (unchanging) system prompt and policy documents, while only sending the (constantly changing) individual customer ticket fresh on each call.

### Output Length Constraints

Because output tokens are typically billed at a higher rate than input tokens, one of the simplest and most overlooked cost levers is simply **constraining how much the model is allowed to generate** — setting explicit length limits, or using structured outputs (Section 1) to force a compact, schema-bound response instead of a free-form, potentially verbose one.

**Real-World Example:** Asking a busy consultant, "Summarize this in exactly two sentences" versus "Tell me about this" produces a dramatically shorter (and cheaper, in billed-by-the-hour terms) answer, purely because of how the question was framed — the same principle applies directly to output token costs in an LLM API call.

---

## 3. DSPy: Treating Prompts as Optimizable Programs

### The Problem With Hand-Written Prompts

A carefully hand-tuned prompt is **brittle**: it might work beautifully with one model, then noticeably degrade when the underlying model is swapped for a different one, or when the data it's applied to shifts even slightly. Every time a team wants to add a new reasoning step, tweak a rule, or evaluate whether one of twenty candidate prompt variations actually performs better, that work has traditionally been done by hand — a slow, unscientific process of trial and error.

### DSPy's Core Idea: Separate "What" From "How"

**DSPy (Declarative Self-improving Python)**, developed at Stanford, reframes this problem entirely. Instead of writing the literal words of a prompt, a developer defines:

- A **Signature:** a declarative description of the input/output contract for a given step (e.g., "given a `question` and `context`, produce an `answer`") — describing *what* the step should accomplish, not the exact phrasing to use.
- A **Metric:** a scoring function that measures whether a given output is actually good, based on real data.
- A set of **training examples** the system can learn from.

DSPy's **optimizer** (compiler) then automatically searches over possible instruction phrasings and few-shot example selections, testing different candidates against the metric, and converges on a prompt that scores well — all without a human hand-tuning the wording directly. This is conceptually very similar to how a traditional machine learning framework trains model weights against a loss function, except here, what's being "trained" is the prompt (and, optionally, which few-shot examples to include) rather than the model's internal weights.

**Real-World Example:** This is the difference between a sales manager who writes one pitch script based on gut instinct and hopes it works, versus a data-driven team that runs many different pitch variations against real customer responses, measures which ones actually close more deals, and automatically converges on the best-performing script — updating it again if the market (or in DSPy's case, the underlying model) changes. DSPy is doing exactly this second process for prompts, automatically and continuously, instead of leaving it to one person's intuition.

### Why This Matters for an FDE

In client engagements, the underlying model a client is using can change (a provider ships a new version, or the client wants to switch providers for cost reasons), and a hand-tuned prompt that worked perfectly on the old model may quietly degrade on the new one with no obvious warning. A DSPy-style approach — defining the task and a measurable success metric rather than hardcoding a single prompt's exact wording — makes a system meaningfully more resilient to exactly that kind of change, since the prompt can simply be re-optimized against the new model rather than manually rewritten from scratch.

---

## 4. ReAct, Safety Prompting, and Responsible AI

### The ReAct Pattern: Reason, Then Act, Then Observe

**ReAct (Reason + Act)** interleaves a model's internal reasoning with concrete actions and their results, in a repeating loop: the model **thinks** about what it needs to do next, **acts** by calling a tool (this connects directly to Section 1's Function Calling and Topic 3's tool-use flow), **observes** the real result that tool returns, and then uses that fresh observation to decide its next thought — repeating this cycle until the task is complete.

**Real-World Example:** This is exactly how a detective works a case, rather than trying to solve it in one single leap of pure deduction. The detective reasons ("the suspect claimed to be out of town — I should verify that"), acts (checks the suspect's phone records), observes the actual result (the records place the suspect at the scene), and then reasons again based on that new, real information ("the alibi is false — who else might corroborate this?") — each step is informed by the *actual* outcome of the previous action, not just an assumption about what that outcome would probably be.

### Safety Prompting and Responsible AI

**Safety prompting** refers to the baseline instructions and practices built into a system prompt (and reinforced through testing) to keep model behavior within acceptable bounds — refusing genuinely harmful requests, avoiding generating dangerous or illegal content, and being upfront about uncertainty rather than confidently fabricating an answer. **Responsible AI** as a broader discipline covers the wider set of practices around this: testing a system for bias across different user groups, being transparent with end users about the fact that they're interacting with an AI, and building in appropriate human oversight for consequential decisions — this connects directly forward to Module 7's later coverage of audit logging, model cards, and human-in-the-loop approval gates for enterprise deployments.

**Real-World Example:** This is the difference between a new customer service hire who was simply told "be helpful" versus one who was given clear, explicit guidance on what topics are off-limits, what to do when a request seems inappropriate or dangerous, when to escalate to a human supervisor rather than guessing, and how to be transparent about the limits of their own knowledge — the second version is dramatically more predictable and trustworthy in front of real customers, exactly as a well-designed safety-conscious system prompt is more predictable and trustworthy in front of real users.

---

## 5. Multimodal Inputs, Document Understanding, and Code Review Prompting

### Beyond Plain Text: Multimodal Prompting

**Multimodal prompting** means constructing prompts that combine text with other input types the model can process — most commonly images and PDF documents, but increasingly audio and video as well. Rather than transcribing an image into a text description yourself and then prompting on that description, you send the image (or document) directly alongside your text instructions, letting the model interpret the visual content natively.

**Real-World Example:** Instead of manually typing out "the chart shows revenue rising from $2M in January to $5M in June, with a dip in March," you simply hand the model the actual chart image and ask it to summarize the trend directly — this is both faster and more accurate, since you're not introducing a layer of your own potential misreading or omission between the source material and the model.

### Document Understanding

**Document understanding** applies this multimodal capability specifically to structured or semi-structured real-world documents — invoices, contracts, scanned forms, financial filings — where the goal is usually to extract specific structured data (an invoice number, a total amount, a signature date) directly out of a visually complex layout, often combined with Section 1's structured-outputs techniques so the extracted data comes back in a clean, directly usable format rather than a paragraph description of what the document contains.

**Real-World Example:** This is directly relevant to the FDE's Enterprise Data Access work discussed in Topic 3 and expanded on in Module 7 — a client with years of scanned paper invoices in a filing system doesn't need someone to describe what's in each invoice; they need the invoice number, vendor name, and total amount extracted into a clean spreadsheet row, which is exactly what a document-understanding pipeline (multimodal model plus a structured-output schema) is built to do at scale.

### Code Review Prompting

**Code review prompting** applies the same broad prompt-design toolkit from Topic 5 (role, context, format, Chain-of-Thought) specifically to the task of having a model critique or improve source code — asking it to check for bugs, security issues, style violations, or missed edge cases, typically with an explicit persona ("You are a senior security-focused code reviewer") and a structured output format (a list of findings, each with a severity level and a suggested fix) so the results plug cleanly into an existing review workflow rather than arriving as unstructured prose.

**Real-World Example:** This is the difference between asking a junior developer "does this code look okay?" (a vague prompt likely to get a vague answer) and asking a senior engineer to "review this pull request specifically for SQL injection vulnerabilities and missing null checks, and list each finding with its line number and severity" — the second, far more specific and structured request reliably produces a far more useful review, whether the reviewer is a human or an LLM.

---

## 6. LLM APIs and Prompt Versioning

### Working Across Multiple Providers

As Topic 3 covered, every major LLM provider — OpenAI, Anthropic, and Google (Gemini) — exposes broadly similar API concepts (an endpoint, headers with an API key, a body specifying the model and prompt), but each has its own SDK, its own specific parameter names, and its own particular strengths. An FDE is expected to be comfortable working across all three, since different clients standardize on different providers, and even within one client engagement, different tasks may genuinely be better served by different providers (for example, one provider's strength in extremely long-document reasoning, another's strength in strict, schema-constrained output formatting, and another's strength in fast multimodal extraction).

**Real-World Example:** This is similar to a contractor who's comfortable working with electrical systems from more than one manufacturer — the underlying principles of wiring and safety are the same across brands, but each manufacturer's specific components, connectors, and documentation differ in the details, and a contractor who's only ever touched one brand will struggle the first time a client's building uses a different one.

### Prompt Versioning: Treating Prompts Like Code

**Prompt Versioning** means tracking changes to a production prompt with the same discipline normally reserved for source code — recording exactly what changed, when, and why, and being able to test a new prompt version against a known evaluation set *before* it replaces the version currently running in production. Without this discipline, a well-meaning small tweak to a system prompt (perhaps made to fix one specific complaint) can silently break behavior on an entirely different set of cases that nobody happened to test by hand.

**Real-World Example:** This is exactly the discipline of software version control (Git branches, commits, and pull requests, as introduced conceptually in Topic 3) applied specifically to prompt text. Just as a team wouldn't want a developer silently pushing an untested code change straight to production, a mature LLM-powered system doesn't want someone silently swapping in a "slightly improved" prompt without first running it against a held-out test set to confirm it hasn't quietly regressed on cases the team already knows matter.

---

## 7. 2026 Industry Update

### Structured Outputs Have Become the Default Expectation, Not a Nice-to-Have

By 2026, the ecosystem around getting reliable structured data out of LLMs has matured substantially. Tools like Instructor now support well over a dozen different LLM providers through one consistent Pydantic-based interface, automatically mapping the same model definition to whichever mechanism a given provider actually supports underneath (native structured-output constraints, JSON mode, or tool/function calling mapped onto the schema) — meaning an FDE can generally write one schema and expect the library to handle the provider-specific plumbing, rather than hand-rolling parsing and retry logic per provider.

### DSPy's Optimizer Toolkit Has Expanded Well Beyond Simple Few-Shot Bootstrapping

DSPy in 2026 offers a genuine menu of optimizer strategies rather than one single approach: **MIPROv2** performs a joint search over both instruction phrasing and few-shot example selection; **COPRO** optimizes instruction wording alone via a hill-climbing search, useful specifically when you can describe a task's rules but can't show real examples for privacy reasons; and **GEPA**, a newer reflective optimizer, works by having the system read its own execution trace, reflect in natural language on what specifically went wrong, and propose targeted prompt revisions to address those exact gaps — a meaningfully more sophisticated approach than blind trial-and-error search. A genuine, honest caveat from current practitioners: an automatically optimized prompt can become substantially longer and less human-readable than a hand-written one, which raises a real question of whether a given optimization run found a true underlying improvement or simply learned to exploit quirks in the evaluation metric — meaning optimized prompts still need to be validated on a genuinely held-out test set, not just the set they were optimized against.

### Cost Optimization Has Become a Full Engineering Stack, Not a Single Trick

The token-cost playbook in 2026 typically stacks several independent techniques rather than relying on any single one: semantic caching (returning a cached response when a new request is embedded and found to be near-duplicate of a previous one, at effectively zero token cost), prompt/context caching for large static content (which can cut input-token costs by roughly 80 to 90 percent for applications with persistent system prompts or reference material), model routing (automatically sending simpler requests to a cheaper, faster model and reserving an expensive frontier model only for genuinely complex ones), and dedicated prompt-compression tooling (which can shrink token counts by roughly 40 to 60 percent on long or verbose prompts without materially degrading output quality). The clear industry framing is that these techniques stack, and a serious production system in 2026 typically layers several of them together rather than picking just one.

**What This Means Practically for an FDE:** When scoping a client's cost model (a skill covered more fully in Module 7), the realistic 2026 baseline isn't "which single model is cheapest" — it's designing the right combination of caching, routing, and compression around whichever model best fits the task, since that combination is usually what actually determines the client's real-world bill far more than the sticker price of any one model.

**Sources for this update:** dev.to, "Top 5 Structured Output Libraries for LLMs in 2026"; AITechConnect, "Guaranteed JSON: Instructor + Pydantic for Structured LLM Output" (May 2026); FutureAGI, "What is DSPy? Stanford's Compiled Prompt Framework in 2026" and "DSPy Optimizers Explained"; FutureAGI, "Automated Prompt Improvement in 2026: DSPy, AdalFlow, Skill Patterns"; Kong, "LLM Cost Optimization" cookbook; tkmxai.it.com, "LLM Cost Optimization in 2026" series; PointFive, "Top 10 Prompt Compression Solutions (2026)."

---

## 8. Interview Questions & Answers

### Conceptual / Foundational

**1. What problem does Function Calling solve that a plain text response doesn't?** A plain text response is free-form and hard for application code to reliably parse, whereas Function Calling lets the model respond by specifying exactly which predefined function to call and with what structured arguments, giving the application a directly actionable, unambiguous instruction instead of prose it has to interpret.

**2. What's the difference between Function Calling and Structured Outputs?** Function Calling is about the model choosing to invoke one of several available tools with structured arguments. Structured Outputs force the model's entire response — not tied to any tool choice — to conform to a specific schema, such as guaranteeing every response is valid JSON matching an exact set of fields.

**3. Why do developers use Pydantic and Instructor instead of hand-writing raw JSON Schema for every request?** Pydantic lets developers define the desired output shape as a readable Python class rather than a verbose raw JSON Schema document, and Instructor automatically maps that Pydantic model to whichever structured-output mechanism a given provider actually supports, plus handles validation and automatic retries if the model's first attempt doesn't match — reducing both the amount of code needed and the number of providers-specific quirks a developer has to handle manually.

**4. What is `tiktoken` used for, and why does it matter before a request is ever sent?** It counts exactly how many tokens a given piece of text will consume for a specific model, which matters for staying under context-window limits and for estimating cost ahead of time, rather than discovering a request was too long or too expensive only after it's already been sent.

**5. Explain prompt caching and why it saves money on repeated content.** Prompt caching lets a provider store the processed version of a prompt segment that stays identical across many requests (like a long system prompt or reference document), charging a significantly reduced rate on subsequent calls that reuse it instead of full price for reprocessing identical content every single time.

**6. In your own words, what does DSPy change about how prompts get written?** Instead of a developer hand-writing the exact wording of a prompt, DSPy has the developer declare what a step should accomplish (a Signature) and a metric for success, and then an optimizer automatically searches for the instruction phrasing and few-shot examples that actually score well against that metric on real data.

**7. Describe the ReAct loop and why it's more reliable than a single, non-interleaved reasoning pass for tasks requiring tool use.** ReAct alternates between the model reasoning about what to do next, acting by calling a tool, and observing the tool's real result before reasoning again — each step is grounded in an actual, current observation rather than an assumption about what a prior action probably returned, which prevents errors from compounding silently across multiple steps.

**8. What is the practical difference between safety prompting and the broader discipline of Responsible AI?** Safety prompting refers specifically to the instructions built into a system prompt (and tested for) to keep a model's individual responses within acceptable bounds. Responsible AI is the broader set of organizational practices around this, including bias testing across user groups, transparency about AI involvement, and appropriate human oversight for consequential decisions.

### Applied / Scenario-Based

**9. A pipeline occasionally crashes because the model's JSON response is missing a required field. What's the most direct fix, and why?** I'd switch from a plain-text or loosely-instructed JSON request to a proper structured-outputs approach using a schema (via Pydantic and a library like Instructor), which constrains or validates the model's output against the required fields and automatically retries on a mismatch, catching the problem at the model-response layer instead of letting it crash deeper in the application.

**10. A client's monthly LLM bill has grown sharply as usage scaled. What levers would you check first?** I'd check whether large, unchanging content (system prompts, reference documents) is being resent on every call rather than cached, whether every request is routed to the most expensive model regardless of complexity, whether output length is unconstrained, and whether prompt compression or semantic caching could eliminate redundant work — these typically account for most avoidable cost in a scaled system.

**11. A hand-tuned prompt that worked well on one model starts underperforming after the client switches providers. How would a DSPy-style approach have made this less painful?** Because DSPy separates the task definition and success metric from the literal prompt wording, the system can simply be re-optimized against the new model and re-validated against the same metric, rather than requiring a person to manually rediscover and rewrite the right wording for the new model from scratch.

**12. A client wants an AI reviewer to check pull requests for security issues. How would you design the prompt differently than just asking "does this code look okay?"** I'd give it an explicit persona (a security-focused senior reviewer), the specific categories to check for (e.g., injection vulnerabilities, missing validation), and a structured output format (a list of findings with line numbers and severity) so the review is both more targeted and directly usable in an existing review workflow, rather than a vague, unstructured impression.

**13. A team wants to extract structured data (invoice number, vendor, total) from thousands of scanned invoices. What's the right approach, and why not just OCR plus manual text parsing?** I'd use a multimodal model directly on the document images combined with a structured-output schema for the exact fields needed, since this avoids the structural information loss that a plain OCR-to-text pipeline often introduces on visually complex layouts, and produces directly usable, schema-validated data rather than raw extracted text that still needs further parsing.

**14. Why would an FDE avoid silently swapping in an "improved" system prompt directly to production without any testing process?** A prompt change that fixes one specific complaint can unintentionally change behavior on other cases that weren't part of the original problem, so without prompt versioning and testing against a held-out evaluation set, a well-intentioned tweak can introduce a regression nobody notices until a client reports it.

### FDE Role-Specific

**15. A client asks why their team should learn all three major LLM APIs instead of just standardizing on one. How would you justify this?** Different providers currently have different relative strengths — one might excel at long-document reasoning, another at strict, schema-constrained formatting, another at fast multimodal extraction — and different tasks within the same engagement, or different future client engagements, may genuinely be better served by different providers, so provider flexibility avoids being locked into a suboptimal tool for a given job.

**16. How would you explain to a skeptical client stakeholder why an "optimized" DSPy prompt that's much longer and harder to read might still be a good outcome?** I'd explain that the goal of the optimization is measurable performance against a real metric on real data, not human readability of the underlying prompt, though I'd also flag the legitimate risk that an optimizer can sometimes exploit quirks in the evaluation metric rather than find a genuine improvement, which is exactly why the optimized prompt still needs to be validated against a separate, held-out test set before being trusted in production.

**17. A client's cost model assumes a single flat per-token price for their entire AI feature. What would you point out is likely missing?** I'd point out that a realistic cost model should account for caching (a large fraction of input tokens on repeated requests can often be served at a fraction of the standard rate), model routing (not every request needs the most expensive model), and prompt compression — treating cost as a single flat per-token number usually significantly overestimates what a well-engineered system will actually spend.

**18. Why does structured output validation matter more, not less, once a system moves from a single LLM call to a multi-step agentic pipeline?** In a single call, a malformed response is an isolated problem a human might catch by reading the output. In a multi-step pipeline, one step's malformed output silently becomes the next step's broken input, so validating and retrying structured outputs at every step is what prevents a single small formatting error from cascading into a much larger, harder-to-diagnose failure further down the pipeline.