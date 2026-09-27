# Topic 5: Prompt Design — From Zero-Shot Prompting to Prompt Injection Defense

> Part of the **Forward Deployed Engineering (FDE) Mastery Repo** — a one-stop learning path for becoming a Forward Deployed Engineer. This is Topic 5, the second subsection of **Module 1: LLM Engineering & Model Integration**. It builds on Topic 2 (how LLMs actually generate text) and Topic 4 (reasoning models and thinking tokens) — the techniques in this guide are, in large part, different ways of shaping what goes into the model *before* generation, to get more reliable behavior out.

---

## Table of Contents

1. [Why Prompt Design Is a Real Engineering Discipline](#1-why-prompt-design-is-a-real-engineering-discipline)
2. [The Foundations: Zero-Shot, Few-Shot, and the Prompt Levers](#2-the-foundations-zero-shot-few-shot-and-the-prompt-levers)
3. [Chain-of-Thought, Tree of Thought, and Self-Consistency](#3-chain-of-thought-tree-of-thought-and-self-consistency)
4. [Reflection, Self-Critique, and Plan-and-Execute](#4-reflection-self-critique-and-plan-and-execute)
5. [Prompt Chaining Patterns](#5-prompt-chaining-patterns)
6. [System Prompt Architecture and Multi-Turn Design](#6-system-prompt-architecture-and-multi-turn-design)
7. [Prompt Injection Defense](#7-prompt-injection-defense)
8. [2026 Industry Update: Prompt Injection Defense Has Moved From "Better Wording" to Architecture](#8-2026-industry-update-prompt-injection-defense-has-moved-from-better-wording-to-architecture)
9. [Interview Questions & Answers](#9-interview-questions--answers)

---

## 1. Why Prompt Design Is a Real Engineering Discipline

It's tempting to think of a prompt as just "the question you type in." In production systems, a prompt is closer to a **small program written in natural language** — it sets the model's role, defines the task, supplies context, constrains the output format, and often chains together multiple steps. Getting this right is the difference between an AI feature that behaves consistently for thousands of users and one that gives a different-quality answer every time depending on how the model happens to interpret a vague instruction.

**Real-World Example:** Think of the difference between telling a new employee "handle the customer emails" versus giving them a clear playbook: who you are speaking as, what tone to use, what information you're allowed to reference, what format the reply should take, and what to do if a request falls outside your scope. The vague instruction *might* work with a brilliant employee on a good day — the playbook works reliably, every day, regardless of who's on shift. A well-designed prompt is that playbook, handed to the model on every single request.

---

## 2. The Foundations: Zero-Shot, Few-Shot, and the Prompt Levers

### Zero-Shot vs. Few-Shot Prompting

- **Zero-Shot Prompting** means asking the model to perform a task with **no examples** of what a correct answer looks like — you're relying entirely on the model's general training to interpret your instructions correctly.
- **Few-Shot Prompting** means showing the model **a small number of example input/output pairs** directly in the prompt before asking it to handle a new case, letting it infer the pattern you want from those examples rather than from your description of the pattern alone.

**Real-World Example:** Zero-shot is like telling a new translator, "Translate this into formal French," and trusting their general skill to get the tone right. Few-shot is like first showing them two or three example sentences you've already translated in exactly the tone you want, then asking them to translate a new sentence "in this same style" — the examples do a lot of the instructional work that a written description alone often can't capture, especially for tone or format consistency.

### The Core Prompt Levers: Role, Task, Context, Format, Persona, Tone

A well-constructed prompt typically combines several distinct "levers," each doing a different job:

- **Role:** Who is the model acting as? ("You are a senior customer support agent for a food delivery app.")
- **Task:** What specifically should it do? ("Summarize this support ticket in two lines.")
- **Context:** What background information does it need? ("The company's refund policy is 30 days from purchase; here is the customer's order history.")
- **Format:** What shape should the output take? ("Respond only in valid JSON with fields `summary` and `urgency`.")
- **Persona / Tone:** What personality or voice should come through? ("Be warm and empathetic, never robotic or dismissive.")

**Real-World Example:** Imagine briefing an actor before a scene. You tell them their **character** (Role: "You're playing a weary but kind small-town doctor"), what happens in the scene (**Task**: "deliver this diagnosis to the patient"), relevant backstory (**Context**: "the patient just lost a family member, be gentle"), how the scene should be shot (**Format**: "in a single continuous take, under two minutes"), and the emotional register (**Tone**: "calm, unhurried, compassionate"). Skip any one of these and the performance drifts — skip all of them and you get something generic and inconsistent. A prompt works the same way.

---

## 3. Chain-of-Thought, Tree of Thought, and Self-Consistency

These three techniques all address the same underlying problem: a model that jumps straight to a final answer on a hard, multi-step problem is more likely to make an error than one that reasons through it first (this connects directly to Topic 4's discussion of reasoning/"thinking" tokens — CoT, ToT, and self-consistency are ways of eliciting that same step-by-step behavior through prompting, even on models that don't have built-in extended thinking).

### Chain-of-Thought (CoT)

**Chain-of-Thought prompting** asks the model to work through a problem in explicit intermediate steps before giving its final answer, rather than jumping directly to a conclusion. The simplest version of this, called **Zero-Shot CoT**, is just adding a phrase like "Let's think step by step" to the prompt. A more reliable version, **Few-Shot CoT**, shows the model two or three worked examples that include the full reasoning chain, not just the final answer.

**Real-World Example:** This is the difference between asking a student to just write down the final numeric answer to a word problem versus asking them to "show your work." Research introducing this technique found that simply prompting models to reason step by step substantially improved accuracy on grade-school math problems, with no changes to the model itself — purely from changing how the question was asked.

### Tree of Thought (ToT)

**Tree of Thought** extends Chain-of-Thought by letting the model explore **multiple different reasoning paths as branches**, evaluate how promising each branch looks partway through, and prune the weaker ones — rather than committing to a single linear chain of reasoning from the start. This mirrors classic search algorithms (breadth-first or depth-first search) applied to a "tree" of possible reasoning paths.

**Real-World Example:** Chain-of-Thought is like solving a maze by picking one path and walking it start to finish. Tree of Thought is like standing at each fork in the maze, briefly considering two or three plausible directions, backtracking out of the ones that look like dead ends, and only committing fully to the path that still seems promising. This costs more time and effort than picking one path blindly, but it's far better suited to problems with several plausible approaches, where a single early misstep can doom an entire straight-line attempt.

### Self-Consistency

**Self-Consistency** takes a different approach: instead of trying to build one better reasoning chain, it runs the **same** Chain-of-Thought prompt multiple times (usually at a higher temperature, so the reasoning paths genuinely differ from each other — recall Topic 2's discussion of temperature), and then takes the **majority-vote answer** across all those independent attempts.

**Real-World Example:** This is like asking five different colleagues to independently solve the same tricky problem without conferring with each other, and then going with whatever answer the majority of them arrived at. Any one person might make an isolated mistake, but it's much less likely that a majority of independent attempts converge on the *same* wrong answer — this is exactly the statistical intuition behind self-consistency, and it is the most expensive of the three techniques here, since it means generating and comparing several full responses instead of just one.

### When to Reach for Each

A practical progression: start with a simple prompt for easy tasks; add few-shot examples if formatting or tone consistency is the problem; add Chain-of-Thought when the task needs genuine multi-step reasoning; add Self-Consistency or Tree of Thought only when accuracy on a hard problem is worth the extra cost of running the model multiple times or exploring multiple branches. None of these are free — every added step costs real tokens and real latency, so the right amount of "reasoning scaffolding" is the least amount that reliably gets the accuracy the task actually needs.

---

## 4. Reflection, Self-Critique, and Plan-and-Execute

### Reflection and Self-Critique

**Reflection** (also called Self-Critique or Self-Refine) has the model play multiple roles on the same task in sequence: first as a **generator** producing an initial draft answer, then as a **critic** evaluating that draft's own weaknesses, and finally as a **reviser** producing an improved version based on its own critique. This loop can repeat multiple times until the output stops meaningfully improving or a set limit is reached.

**Real-World Example:** This is exactly how a careful writer edits their own work: write a first draft, then re-read it wearing an "editor's hat" to spot weak arguments or unclear sentences, then rewrite based on that critique. The interesting result from research on this technique is that a single model, given the right prompting structure, can meaningfully improve its own output through this generate-critique-revise loop — without any human feedback or additional training in between.

### Plan-and-Execute

The **Plan-and-Execute** pattern separates a task into two distinct phases: first, the model produces an explicit **plan** — a numbered list of steps needed to complete the task — and only afterward does it (or a separate call) actually **execute** each step. This is different from letting the model figure out its approach implicitly while generating a single continuous response.

**Real-World Example:** This is the difference between an employee who starts writing a report immediately as thoughts occur to them, versus one who first drafts an outline, gets that outline approved, and only then writes the full report section by section. The second approach catches structural problems (a missing section, a wrong ordering, an unnecessary tangent) while they're cheap to fix — before a single word of the final output has been generated — rather than discovering them only after the full response is already written.

---

## 5. Prompt Chaining Patterns

**Prompt Chaining** means breaking one large, complex task into a **sequence of smaller prompts**, where the output of one prompt becomes part of the input to the next, rather than trying to accomplish everything in a single giant prompt.

### Why Chain Instead of Using One Big Prompt

- **Reliability:** A single prompt trying to do five things at once (extract data, summarize it, classify it, translate it, and format it) gives the model far more opportunities to drop or garble one of those five things. Five focused prompts, each doing one job well, are individually easier to get right and easier to debug when something goes wrong.
- **Debuggability:** If a chained pipeline produces a bad final result, you can inspect the output of each individual step to find exactly where it went wrong — with one giant prompt, you're stuck guessing which part of a single large instruction the model misunderstood.
- **Cost control:** Later steps in a chain can use a smaller, cheaper model for simple sub-tasks (like reformatting), reserving a more expensive model only for the step that genuinely needs its capability.

**Real-World Example:** This is exactly how an assembly line works, compared to asking one person to build an entire car alone from raw materials. Each station on the line (weld the frame, install the engine, paint the body, attach the interior) does one job well and passes its output to the next station. If the final car has a paint defect, you know exactly which station to go inspect — you don't have to re-examine the entire build from scratch. A prompt chain for, say, the Support Ticket Summarizer from Topic 3 might look like: Step 1 extracts key facts from the raw angry ticket, Step 2 classifies urgency, and Step 3 generates the final two-line summary using the outputs of Steps 1 and 2 — three focused prompts instead of one prompt trying to juggle extraction, classification, and summarization simultaneously.

---

## 6. System Prompt Architecture and Multi-Turn Design

### What a System Prompt Actually Does

A **System Prompt** is a special instruction, typically set once by the developer (not the end user), that establishes the model's persistent role, behavior boundaries, and constraints for the entire conversation — it sits "above" whatever the user types and is treated with higher priority than the user's own messages.

**Real-World Example:** This is like an employee handbook that a new hire reads once on their first day and is expected to follow for their entire employment, as opposed to instructions a customer gives them on any individual call. A well-designed system prompt for the Support Ticket Summarizer might say: "You are a summarization assistant for a food delivery company's support team. Only summarize the ticket provided. Never answer unrelated questions, write code, or perform calculations unrelated to this task. If asked to do something outside this scope, politely decline and explain your purpose."

### Multi-Turn Design and Managing State

As established in Topic 3, a raw API call has no memory between requests (Context Amnesia). **Multi-turn design** is the discipline of deliberately managing what gets re-sent on every new request so the model behaves as though it remembers the conversation:

- **Conversation history:** Re-sending some or all of the prior back-and-forth so the model has continuity.
- **Summarized memory:** For very long conversations, periodically compressing older turns into a shorter summary instead of resending the full transcript verbatim, to control token cost (this connects to the Token Economy discussed in Topic 3).
- **State tracking:** Explicitly tracking structured facts established earlier in the conversation (e.g., "customer's order ID is 48213") separately from the raw chat transcript, so critical details don't get lost or diluted the way Topic 2's discussion of long-context "lost in the middle" effects would predict.

**Real-World Example:** This is like a doctor's patient file. Rather than re-reading a patient's entire life story verbatim before every single appointment, the doctor keeps a structured summary of key facts (allergies, chronic conditions, current medications) that gets carried forward and only updated when something genuinely changes — while still being able to refer back to the full detailed notes if something specific needs it. Good multi-turn design applies that same instinct to what a system re-sends to the model on every turn.

---

## 7. Prompt Injection Defense

### What Prompt Injection Actually Is

As introduced in Topic 3, **Prompt Injection** happens when a user (or content the model reads, like a document or a web page) includes text specifically crafted to make the model ignore its original instructions and instead follow the attacker's embedded instructions. The reason this is so hard to fully prevent is structural: an LLM reads its developer-provided system instructions and the untrusted user/document content through the **same channel** — as far as the underlying next-token-prediction mechanism from Topic 2 is concerned, it's all just text, and the model has no built-in, foolproof way to distinguish "an instruction I should obey" from "text that merely looks like an instruction."

### Core Mitigation Techniques

- **Input Sanitization:** Screening incoming text for known attack patterns or suspicious phrasing before it ever reaches the model.
- **Structural Delimiters:** Wrapping user-provided content in clear boundary markers (e.g., specific tags) so the model has a stronger structural signal about what's an instruction versus what's data to be processed.
- **Privilege Separation:** Ensuring the part of the system that reads untrusted content is never the same part of the system that has the power to take a sensitive action (like sending an email, deleting a record, or accessing another customer's data).
- **Output Validation:** Checking the model's final response for signs it was hijacked (e.g., it starts producing content wildly outside its intended scope) before that response ever reaches the user or triggers a downstream action.

**Real-World Example:** Think of a mailroom clerk who is supposed to only sort incoming mail into the correct department bins. Input Sanitization is like scanning envelopes for obviously suspicious markings before they even reach the clerk's desk. Structural Delimiters are like requiring every legitimate internal instruction to arrive on official letterhead in a sealed envelope, so a instruction scribbled on a piece of junk mail is visibly not the same kind of thing. Privilege Separation is the actual safety net: even if a piece of junk mail says "please also unlock the supply closet and hand over the master keys," the clerk role should have never been given supply-closet key access in the first place — so the instruction, even if "obeyed," accomplishes nothing dangerous.

---

## 8. 2026 Industry Update: Prompt Injection Defense Has Moved From "Better Wording" to Architecture

Everything in Sections 2 through 6 (zero/few-shot, CoT/ToT/self-consistency, reflection, chaining, system prompts) is still exactly how prompt design works in 2026 — these are durable techniques, not trends. Prompt Injection Defense specifically, however, has matured significantly, and it's worth understanding where the field has landed, because the honest answer is less comforting than most beginners expect.

### There Is No Fully Reliable Detection-Based Defense

Formal research published in 2026 has argued that reliably preventing prompt injection through better model training or smarter detection alone is not fully solvable without also breaking the model's legitimate ability to follow instructions found in the content it processes — since a helpful "read this document and act on it" agent and a "read this document and get hijacked by it" agent are, structurally, doing the same basic operation. A separate line of research has formalized this as a **"Defense Trilemma":** any injection-defense layer can be *sound* (blocks all unsafe prompts), *complete* (never blocks a legitimate one), and *utility-preserving* — but not reliably all three simultaneously. Practically, this means claims of a defense that "completely solves" prompt injection should be treated with real skepticism; the honest industry framing in 2026 is **risk reduction through layered defense**, not a single fix.

### The "Lethal Trifecta" — A Useful Way to Reason About Real Risk

A widely adopted mental model in 2026 for judging how exposed a given system actually is: an agent becomes a genuine exfiltration risk when it simultaneously has (1) access to private/sensitive data, (2) exposure to untrusted external content, and (3) the ability to communicate results externally (e.g., sending an email, posting to the internet, calling an arbitrary API). Removing just **one** of these three legs breaks the exploit path even if the injection attempt itself technically succeeds — which is precisely why Privilege Separation (Section 7) is considered the single highest-leverage architectural control: it's usually the most practical of the three legs to cut.

### Dual-LLM and Privileged Execution Boundaries

A concrete architectural pattern that has gained real traction: a **Dual-LLM design**, where one model's only job is to read and interpret untrusted content (with no access to sensitive tools or data), and a separate, privileged model or code path — which never directly reads raw untrusted content — is the only component actually allowed to take consequential actions. Complementary lightweight techniques with measured impact include **spotlighting** (interleaving a special marker token throughout retrieved external content so the model has a continuous signal that "this is data, not instructions," which one major vendor's internal evaluation found cut indirect-injection success rates from over 50 percent down to under 2 percent on summarization and Q&A tasks) and **encoding** (transforming untrusted content, e.g., into base64, and having the model explicitly decode-then-process it, which creates a further structural gap between "parsing instructions" mode and "processing data" mode).

**What This Means Practically for an FDE:** When a client asks "have you made sure the AI can't be tricked?", the honest, credible answer in 2026 is not "yes, we've made it impossible" — it's "we've reduced the realistic attack surface through layered, architectural defenses, and we can show you exactly which of those layers are in place and what residual risk remains." This is a direct extension of the FDE mindset from Topic 1: reliability and trust come from disciplined engineering around the model, not from the model somehow becoming un-hijackable on its own.

**Sources for this update:** "AI Agents May Always Fall for Prompt Injections" (Abdelnabi & Bagdasarian, arXiv, May 2026); "Defense Trilemma" formalization, ICLR 2026; TrueFoundry, "Prompt Injection Defense at the Gateway" (June 2026); Klu.ai, "Prompt Injection Defense" (August 2026); Tian Pan, "Prompt Injection in Production" (spotlighting/encoding evaluation findings); Cheng & Tsao, "Agent Privilege Separation in OpenClaw," arXiv 2603.13424 (March 2026); Mind-Core, "How to Prevent Prompt Injection Attacks in AI Systems."

---

## 9. Interview Questions & Answers

### Conceptual / Foundational

**1. What is the core difference between zero-shot and few-shot prompting?** Zero-shot prompting asks the model to perform a task with no examples, relying purely on its general training to interpret the instructions. Few-shot prompting includes a small number of example input/output pairs directly in the prompt, letting the model infer the desired pattern from those examples, which is especially useful for controlling tone or output format.

**2. Name the core prompt-design levers and what each one controls.** Role defines who the model is acting as, Task defines what it should specifically do, Context supplies the background information it needs, Format specifies the shape of the output, and Persona/Tone controls the voice and emotional register of the response.

**3. What problem does Chain-of-Thought prompting solve, and how does it work?** It addresses the fact that jumping straight to a final answer on a hard, multi-step problem increases the chance of error. It works by asking the model to generate explicit intermediate reasoning steps before its final answer, either simply ("let's think step by step") or by showing worked examples that include the reasoning chain.

**4. How does Tree of Thought differ from standard Chain-of-Thought?** Chain-of-Thought commits to a single linear sequence of reasoning steps. Tree of Thought explores multiple different reasoning branches in parallel, evaluates how promising each looks partway through, and prunes weaker branches, similar to a search algorithm exploring a tree of possibilities.

**5. How does Self-Consistency improve on plain Chain-of-Thought, and what's the cost trade-off?** It runs the same Chain-of-Thought prompt multiple times, typically at a higher temperature, and takes the majority-vote answer across all the independent attempts, on the assumption that a majority of independent reasoning paths converging on the same wrong answer is less likely than any single path making an isolated mistake. The trade-off is cost: it requires generating and comparing several full responses instead of one.

**6. What is the Reflection (Self-Refine) pattern, and why is it notable?** It has the model act as generator, critic, and reviser on the same task in sequence — producing a draft, critiquing its own weaknesses, and then revising based on that critique. It's notable because a single model can meaningfully improve its own output through this loop without any additional human feedback or retraining.

**7. What does the Plan-and-Execute pattern change compared to letting a model just generate a response directly?** It separates the task into an explicit planning phase (producing a numbered list of steps) before any execution happens, rather than letting the model figure out its approach implicitly while writing one continuous response. This surfaces structural problems in the approach while they're still cheap to fix.

**8. Why would you break a task into a prompt chain instead of writing one large prompt?** A single prompt trying to do several things at once gives the model more chances to drop or garble one part of the task, and makes it hard to tell which part failed if the result is wrong. A chain of focused, single-purpose prompts is more reliable, easier to debug, and lets cheaper models handle simpler sub-steps.

### Applied / System Design

**9. What is a system prompt, and how does its priority differ from a user's message?** A system prompt is a developer-set instruction that establishes the model's persistent role, behavior, and constraints for the whole conversation, and it's treated with higher priority than what an end user types — the user's messages operate within the boundaries the system prompt sets, not above them.

**10. How would you design multi-turn memory for a long-running conversation without just resending the entire transcript every time?** I'd track structured key facts separately from the raw chat log (e.g., an order ID or a stated preference), periodically summarize older turns into a shorter form rather than resending them verbatim, and only keep the most recent exchanges in full detail — balancing conversational continuity against the token cost and "lost in the middle" risk of an ever-growing transcript.

**11. Why is prompt injection fundamentally difficult to solve through model training or detection alone?** The model reads developer instructions and untrusted user/document content through the same channel, and at the level of next-token prediction there's no reliable, built-in way to distinguish "an instruction I should obey" from "text that merely looks like an instruction" — 2026 research has formalized this as a genuine structural limitation, not just a current engineering gap.

**12. What is the "lethal trifecta," and how would you use it to assess risk in a client's proposed AI agent?** It's the combination of an agent having access to private data, exposure to untrusted external content, and the ability to communicate externally — when all three are present at once, the system becomes a real exfiltration risk. I'd assess a client's proposed agent by checking which of the three legs are present, and look for ways to remove at least one, since that breaks the exploit path even if an injection attempt technically succeeds.

**13. What is Privilege Separation, and why is it considered the highest-leverage prompt injection defense?** It means ensuring the component that reads untrusted content is never the same component with the power to take a sensitive action, like sending data externally or modifying a record. It's high-leverage because it limits the consequences of a successful injection rather than trying to prevent the injection from happening at all, which research suggests can never be fully guaranteed.

**14. What is "spotlighting," and why does it help against indirect prompt injection?** Spotlighting interleaves a special marker token throughout retrieved external content, giving the model a continuous structural signal that the content is data to be processed, not instructions to follow. Evaluated in production settings, this measurably reduced indirect injection success rates on summarization and Q&A tasks compared to having no such marking at all.

### FDE Role-Specific

**15. A client asks, "Have you made it impossible for someone to trick our AI?" How would you answer honestly, based on current research?** I'd explain that no defense fully eliminates prompt injection risk according to current research, so instead of claiming it's impossible, I'd walk them through the specific layered defenses in place (privilege separation, input/output validation, architectural boundaries) and what residual risk remains — framing it as risk reduction through defense-in-depth rather than a solved problem.

**16. A client's proposed AI feature reads incoming customer emails, has access to the full customer database, and can send automated replies. How would you evaluate this using the concepts in this guide?** This matches the "lethal trifecta" pattern exactly — untrusted content (emails), sensitive data access (the database), and external communication ability (sending replies) — so I'd push to remove or constrain at least one leg, for example by having a privilege-separated design where the email-reading component never directly has database write access or send authority itself.

**17. Why might an FDE choose to build a support ticket summarizer (from Topic 3) as a prompt chain instead of one large prompt?** Breaking it into focused steps, like extracting key facts, classifying urgency, and then generating the final summary, makes each step easier to get right, easier to debug if the final summary is wrong, and allows cheaper models to be used for the simpler sub-steps rather than running the most expensive model on the entire task at once.

**18. A client wants their support bot to reliably refuse off-topic requests, referencing the Prompt Injection scenario from Topic 3. What would you actually build, beyond just adding a line to the system prompt?** I'd combine a clear system prompt defining scope with structural defenses beyond wording alone — input sanitization to catch obviously off-topic or manipulative requests, output validation to catch responses that drift outside the intended scope, and, where the bot has any access to sensitive actions or data, privilege separation so that even a successful redirection attempt has limited real consequence.

**19. How would you decide, for a given client task, whether to use Chain-of-Thought alone versus adding Self-Consistency or Tree of Thought on top of it?** I'd start with the simplest approach that meets the accuracy bar and only add cost: plain prompting for easy tasks, Chain-of-Thought once genuine multi-step reasoning is needed, and Self-Consistency or Tree of Thought only when the task is hard enough, and the cost of an error high enough, to justify the extra tokens and latency of running multiple reasoning attempts.

**20. Why does good system prompt design matter as much as, or more than, model choice for many production features?** A model has broad general capability, but without a clear system prompt defining role, scope, and constraints, that capability isn't reliably channeled toward the specific behavior a client needs — a well-designed system prompt is often what actually determines whether a feature behaves consistently in production, which is why prompt and guardrail design is treated as a first-class engineering skill rather than an afterthought.