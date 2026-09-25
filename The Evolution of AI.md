# Topic 1: The Evolution of AI → GenAI, and the Forward Deployed Engineer (FDE)

> Part of the **Forward Deployed Engineering (FDE) Mastery Repo** — a one-stop learning path for becoming a Forward Deployed Engineer. This is Topic 1. More topics will be added as separate READMEs/folders.

---

## Table of Contents

1. [The Role of a Forward Deployed Engineer (FDE)](#1-the-role-of-a-forward-deployed-engineer-fde)
2. [The Origins of AI & The Turing Test](#2-the-origins-of-ai--the-turing-test)
3. [Phase 1: Rule-Based AI](#3-phase-1-rule-based-ai)
4. [Phase 2: Machine Learning (ML)](#4-phase-2-machine-learning-ml)
5. [Phase 3: Neural Networks & Deep Learning](#5-phase-3-neural-networks--deep-learning)
6. [The NLP Bottleneck](#6-the-nlp-bottleneck-why-text-is-harder-than-images)
7. [The Breakthrough: Transformers & the GenAI Era](#7-the-breakthrough-transformers--the-genai-era)
8. [Big Picture Timeline](#8-big-picture-timeline)
9. [Interview Questions](#9-interview-questions)

---

## 1. The Role of a Forward Deployed Engineer (FDE)

### What is an FDE?

A **Forward Deployed Engineer** is an engineer who sits at the intersection of **software engineering, product thinking, and client-facing consulting**. Instead of building generic software for an unknown mass market (like a traditional SWE at a product company), an FDE goes "forward" — physically or virtually into a **specific client's environment** — to build, customize, and deploy solutions that solve that client's exact problem.

| Traditional Software Engineer | Forward Deployed Engineer |
| --- | --- |
| Builds generic, reusable products for many customers | Builds tailored solutions embedded in one client's workflow |
| Works mostly with internal teams (PM, Design, Eng) | Works directly with the client — ops teams, execs, end users |
| Success = shipping features on a roadmap | Success = a measurable business outcome for the client |
| Requirements are usually well-defined | Requirements start vague and must be *discovered* |
| Rarely sees how the product is actually used | Lives inside the client's real workflow and data |

**Real-World Example:** Imagine a hospital says, "We want AI to help our nurses." A traditional SWE might build a generic chatbot widget. An FDE instead sits with the nurses, watches how they currently handle patient intake, realizes the real pain point is "nurses spend 20 minutes per patient manually transcribing notes into the EHR system," and builds a narrow, high-trust tool that auto-fills the EHR from voice notes — because *that* is the actual business outcome the hospital needs.

### Core Responsibility: Translating Vague Requests into Business Outcomes

Clients rarely say what they actually need in engineering terms. The FDE's job is translation.

- **Client says:** "Build us an AI chatbot for customer support."
- **What they actually mean:** "Our support team is overwhelmed, resolution time is too slow, and customers are getting frustrated with wait times."
- **FDE's real target metric:** *Reduce average resolution time* — **without** sacrificing customer trust (i.e., don't let the bot make things up, escalate badly, or feel robotic).

**Real-World Example:** A retail company asks for "an AI shopping assistant." The FDE digs deeper and discovers the real KPI leadership cares about is *cart abandonment rate*. So instead of building a broad Q&A bot, the FDE builds a narrow assistant focused specifically on answering shipping and sizing questions at the exact moment a user hesitates on the product page — because that's where abandonment actually happens.

### Why "Just Plugging in an LLM" Doesn't Work

A raw LLM (like a general-purpose GPT model) has **no idea about your company**. It only knows general patterns from its training data. If you just hook it up to a chat widget with no guardrails:

- It will **hallucinate** — confidently make up facts (e.g., inventing a fake refund policy).
- It has **no access to real, live business data** (order history, account status, inventory).
- It has **no concept of your company's specific rules, tone, or edge cases**.

**Real-World Example:** A customer asks a bare LLM-powered bot, "Can I return this jacket after 90 days?" With no context, the model might confidently say "Yes, of course!" — because it sounds statistically plausible — even though the company's real policy is a strict 30-day window. This single hallucination can cost the company real money and erode customer trust.

### How FDEs Make AI Reliable: Providing "Context"

The FDE's core technical job is **grounding** the model in reality by feeding it the right context at the right time. This typically means:

- **Retrieval-Augmented Generation (RAG):** Pulling the actual, current return policy document and injecting it into the prompt before the model answers.
- **Tool/API access:** Letting the model *call* the order-management system to check a real customer's real order status instead of guessing.
- **Guardrails & prompt engineering:** Constraining the model to only answer from provided context, and to say "I don't know, let me connect you to a human" rather than inventing an answer.
- **Evaluation loops:** Continuously testing the system against real client edge cases before and after deployment.

**Real-World Example:** Instead of asking the LLM "What's our return policy?" cold, the FDE builds a pipeline that fetches the live return-policy document *and* the specific customer's order data, then asks the model: "Given this policy and this customer's order (purchased 45 days ago), answer their question — and only use the info provided." Now the answer is grounded, accurate, and trustworthy.

---

## 2. The Origins of AI & The Turing Test

### Alan Turing's Foundational Question (1950)

In 1950, mathematician **Alan Turing** published *"Computing Machinery and Intelligence,"* posing a deceptively simple question: **"Can machines think?"**

Rather than getting stuck debating the philosophy of what "thinking" truly means, Turing reframed the problem into something testable.

### The Turing Test (The Imitation Game)

The setup:

- A human **judge** has text-based conversations with two hidden participants: one **human**, one **machine**.
- If the judge **cannot reliably tell which is which**, the machine is said to have passed the test.

**The core philosophy:** Turing argued that if a machine's *behavior* is indistinguishable from a human's, then debating whether it "truly" thinks internally becomes almost irrelevant for practical purposes — **mimicking intelligence convincingly enough is functionally the same as having it.**

**Real-World Example:** This is exactly the experience many people have today chatting with ChatGPT — if you didn't know it was a machine, many everyday conversations would feel indistinguishable from talking to a knowledgeable human. That "indistinguishability" is precisely Turing's test, playing out 70+ years later.

### John McCarthy Coins "Artificial Intelligence"

In **1956**, computer scientist **John McCarthy** coined the term **"Artificial Intelligence"** at the Dartmouth Conference, officially naming this new field of study — the formal starting gun for AI as an academic discipline.

---

## 3. Phase 1: Rule-Based AI

### The "If-Else" Approach

The earliest AI systems didn't "learn" anything. Humans manually wrote out **massive decision trees of hardcoded rules** — essentially giant `IF... THEN... ELSE` ladders.

```
IF age < 18 THEN output "Minor"
ELSE IF age < 65 THEN output "Adult"
ELSE output "Senior"
```

These systems are often called **"Expert Systems"** — the "intelligence" is really just a human expert's knowledge translated into rigid logic by a programmer.

**Real-World Example: ELIZA (1966)** Built by Joseph Weizenbaum at MIT, **ELIZA** simulated a Rogerian psychotherapist. It didn't understand language at all — it simply matched keywords and reflected sentences back:

- **User:** "I am feeling sad today."
- **ELIZA:** "Why do you say you are feeling sad today?"

ELIZA just detected the pattern "I am \[X\]" and mechanically flipped it into a question. It felt eerily human to many users at the time — despite having zero actual understanding — because rigid pattern-matching can go a surprisingly long way for narrow conversations.

### The Fatal Flaw

Rule-based systems **cannot scale to real-world language and situations** because:

- You cannot write a rule for every possible sentence, typo, slang term, or phrasing.
- They have **zero flexibility** — a slightly rephrased question that wasn't explicitly coded for completely breaks the system.
- Human language is filled with ambiguity, exceptions, and context — impossible to hardcode exhaustively.

**Real-World Example:** If ELIZA was only coded to recognize "I am sad," a user typing "I'm kinda down today ngl" would completely stump it — a trivial rephrasing for a human, but a total blind spot for a rigid rule system.

---

## 4. Phase 2: Machine Learning (ML)

### From Rules → Data

Instead of a programmer hardcoding every rule, **Machine Learning** flips the approach: give the machine **lots of data and examples**, and let it statistically figure out the patterns itself.

### Supervised Learning vs. Unsupervised Learning

**Supervised Learning** — the data comes with **labels** (the "correct answers"), and the model learns to map inputs to those known outputs.

**Real-World Example:** Feeding a model thousands of emails each labeled "Spam" or "Not Spam." The model learns which word patterns correlate with each label, so it can classify *new*, unseen emails.

**Unsupervised Learning** — the data has **no labels**. The model's job is to find hidden structure or natural groupings on its own.

**Real-World Example:** An e-commerce company feeds a model raw customer purchase data with no labels at all. The model discovers, on its own, that customers naturally cluster into groups like "bargain hunters," "brand loyalists," and "gift buyers" — groupings no human explicitly defined.

### The Limitation: Feature Engineering

Early ML still required a human to manually decide **which characteristics ("features") of the data the model should even look at.**

**Real-World Example:** To detect spam, a human engineer had to manually decide: "Let's create a feature for *number of exclamation marks*, a feature for *presence of the word 'free'*, a feature for *sender's domain reputation*," etc. If the engineer failed to think of the right feature, the model simply could not learn it — no matter how much data you gave it. This **doesn't scale** to genuinely complex problems like understanding an image or a full paragraph of text, where the number of potentially relevant "features" is practically infinite and impossible for a human to enumerate by hand.

---

## 5. Phase 3: Neural Networks & Deep Learning

### Mimicking the Human Brain

**Neural Networks** were designed loosely inspired by how neurons in the human brain connect and fire. A network is structured in layers:

- **Input Layer:** Receives the raw data (e.g., pixel values of an image).
- **Hidden Layer(s):** Intermediate layers that progressively detect more abstract patterns.
- **Output Layer:** Produces the final prediction (e.g., "This is a cat").

The key innovation over classic ML: the network **discovers the important features on its own** — no human has to manually hand-engineer them. This is why it's called **"Deep" Learning** — "deep" refers to having many stacked hidden layers.

### Weights: The Machine's "Learning"

Every connection between neurons has a numerical **"weight"** — essentially a measure of how much influence one neuron has on the next. During training, the network makes a prediction, checks how wrong it was (the "error" or "loss"), and then mathematically adjusts every weight slightly to reduce that error next time (a process called **backpropagation**). Repeated millions of times, this is how the network gradually "learns."

**Real-World Example:** Think of weights like the dials on a giant mixing board. Initially all the dials (weights) are set randomly, so the output is noise/garbage. Every time the network sees a labeled example and gets it wrong, it nudges thousands of tiny dials a tiny bit in the right direction. After millions of nudges, the dials are tuned just right to reliably produce the correct output.

### CNNs & The Computer Vision Breakthrough

**Convolutional Neural Networks (CNNs)** are a specialized neural network architecture that excel at scanning images in small chunks (like a sliding window), letting them progressively build up from simple patterns (edges, curves) in early layers to complex concepts (eyes, faces, objects) in deeper layers — all learned automatically from data, without a human ever hand-defining "what an eye looks like."

**Real-World Examples:**

- **FaceID** on your phone: a CNN learns the unique geometric pattern of your face to unlock your device.
- **Self-driving cars:** CNNs process camera feeds in real time to detect pedestrians, lane lines, and other vehicles.
- **Auto-tagging photos:** Google Photos or iPhone Photos automatically recognizing and grouping pictures of "your dog" or "the beach" without you labeling a single image.

---

## 6. The NLP Bottleneck (Why Text is Harder than Images)

While Computer Vision was booming through the 2010s thanks to CNNs, **Natural Language Processing (NLP) lagged behind significantly**. Text turned out to be a fundamentally harder problem than images.

### Why Language Is So Hard

- **Sarcasm:** "Oh great, another Monday" — the literal words say "great," but the actual meaning is the opposite.
- **Context:** The word "bank" means something totally different in "river bank" vs. "savings bank."
- **Ambiguity:** "I saw the man with the telescope" — did *I* have the telescope, or did *the man*?
- **Word order:** "Dog bites man" vs. "Man bites dog" — same words, opposite meaning.
- **References (coreference):** In "The trophy didn't fit in the suitcase because **it** was too big," what does "it" refer to — the trophy or the suitcase? Humans intuit this instantly; machines historically struggled badly.

**Real-World Example:** An image of a cat is a cat in any context — pixels don't change meaning based on what came before them. But the sentence "That's just great" needs the surrounding conversational context (tone, prior messages) to know if it's genuine praise or sarcastic frustration — a purely statistical model has a very hard time capturing that.

### Early Attempts & Why They Failed

**Statistical Models (N-grams):** These models predicted the next word purely based on the probability of word sequences seen in training data (e.g., "the probability that 'York' follows 'New' is very high").

**Real-World Example:** Early smartphone keyboard autocomplete (pre-2015) used this approach — it could reasonably guess the next word in short phrases, but completely fell apart on longer, more meaningful sentences because it had no real understanding, just word-frequency statistics.

**RNNs (Recurrent Neural Networks):** RNNs process text **sequentially, one word at a time**, carrying forward a "memory" (hidden state) of what came before to the next step.

**The fatal flaw — Context Loss:** As the sequence gets longer, information from early words gets diluted and effectively "forgotten" by the time the network reaches later words — like a game of telephone where the message degrades the further it travels.

**Real-World Example:** If an RNN reads a 500-word paragraph and the very first sentence establishes "The patient is allergic to penicillin," by the time it reaches the end of the paragraph to generate a recommendation, that crucial detail may have effectively faded from its memory — leading to a dangerously wrong (or just incoherent) output.

---

## 7. The Breakthrough: Transformers & The GenAI Era

### "Attention Is All You Need" (2017)

In 2017, a team of researchers at Google published the paper **"Attention Is All You Need,"** introducing the **Transformer architecture** — arguably the single most important breakthrough behind the entire modern Generative AI boom (ChatGPT, Claude, Gemini, and beyond).

### The Transformer & The Attention Mechanism

Unlike RNNs, which process text one word at a time in sequence, Transformers process an **entire block of text all at once, in parallel**. The core innovation is the **Attention mechanism**: for every single word, the model mathematically calculates how relevant *every other word in the text* is to understanding it — no matter how far apart they are.

This completely solves the context-loss problem RNNs suffered from, because there's no "sequential memory" to degrade — every word has direct, weighted access to every other word simultaneously.

**Real-World Example:** Revisit the tricky sentence: "The trophy didn't fit in the suitcase because **it** was too big." A Transformer's attention mechanism directly computes a strong mathematical connection between "it" and "trophy" (or "suitcase," depending on context) by weighing relevance across the *entire* sentence at once — rather than relying on a fading memory carried word-by-word, the way an RNN would.

### What is a Large Language Model (LLM)?

A **Large Language Model** is a Transformer-based neural network trained on an enormous amount of text data (much of the internet, books, code, etc.) with **billions (or trillions) of parameters (weights)**, enabling it to generate remarkably coherent, context-aware, human-like text.

### Breaking Down GPT

**GPT = Generative Pre-trained Transformer**

- **Generative:** It *generates* new content (text, code, etc.) rather than just classifying or labeling existing content.
- **Pre-trained:** It's first trained broadly on massive general internet-scale data (learning grammar, facts, reasoning patterns) before being fine-tuned for specific tasks or behaviors.
- **Transformer:** It's built on the Transformer architecture described above — the attention mechanism is the engine under the hood.

**Real-World Example — Tying the Whole Journey Together:** GPT is the literal culmination of every phase in this guide: it doesn't use hardcoded ELIZA-style rules (Phase 1), it doesn't require hand-engineered features (Phase 2), it uses deep neural network weights learned through training (Phase 3), and it solves the exact NLP context-loss problem that stumped RNNs (Phase/Section 6) — all thanks to the Transformer's attention mechanism (Section 7). This is precisely *why* GenAI tools like ChatGPT feel like such a massive leap: they're the product of 70+ years of compounding breakthroughs, from Turing's original 1950 question all the way to today.

---

## 8. Big Picture Timeline

| Year | Milestone |
| --- | --- |
| 1950 | Alan Turing asks "Can machines think?" and proposes the Turing Test |
| 1956 | John McCarthy coins the term "Artificial Intelligence" |
| 1960s | Rule-based systems like ELIZA simulate conversation via keyword matching |
| 1980s–2000s | Machine Learning rises — supervised/unsupervised learning, feature engineering |
| 2012 | CNNs cause a breakthrough in Computer Vision (e.g., ImageNet) |
| 2010s | RNNs attempt NLP but suffer from context loss on long text |
| 2017 | "Attention Is All You Need" introduces the Transformer architecture |
| 2018–Present | GPT and other LLMs launch the modern Generative AI era |
| Today | FDEs deploy grounded, context-aware LLM applications into real businesses |

---

## 9. Interview Questions & Answers

### Conceptual / Foundational

**1. What is the Turing Test, and what philosophical point was Alan Turing actually making with it?** The Turing Test has a human judge hold text conversations with a hidden human and a hidden machine; if the judge can't reliably tell which is which, the machine passes. Turing's real point was philosophical: instead of endlessly debating what "true thinking" means, he reframed intelligence as a matter of observable behavior — if a machine's output is indistinguishable from a human's, arguing about its "inner" thoughts becomes practically irrelevant.

**2. What is the core difference between Rule-Based AI and Machine Learning?** Rule-Based AI relies on humans hardcoding every decision as explicit `IF/ELSE` logic — the system has zero ability to handle anything not explicitly coded. Machine Learning flips this: instead of hand-written rules, you give the system data and let it statistically infer the patterns itself, so it can generalize to new, unseen inputs.

**3. Explain the difference between Supervised and Unsupervised Learning with an example of each.** Supervised Learning trains on labeled data (input plus the correct answer), like emails labeled "Spam"/"Not Spam," so the model learns to predict labels for new emails. Unsupervised Learning works on unlabeled data and finds hidden structure on its own, like clustering customers into groups (e.g., "bargain hunters," "brand loyalists") without anyone defining those groups in advance.

**4. What is "feature engineering," and why did it become a bottleneck for classical Machine Learning?** Feature engineering is the manual process of a human deciding which characteristics of the raw data the model should pay attention to (e.g., "number of exclamation marks" for spam detection). It became a bottleneck because it doesn't scale — for complex problems like images or language, the number of potentially relevant features is effectively infinite, and a model can only ever learn what a human thought to hand it.

**5. How do Neural Networks conceptually mimic the human brain? What are input, hidden, and output layers?** Neural Networks are loosely inspired by neurons connecting and firing in the brain. The input layer receives raw data (e.g., pixel values), hidden layers progressively extract more abstract patterns, and the output layer produces the final prediction. Unlike classical ML, the network discovers which features matter on its own rather than relying on a human to define them.

**6. What is a "weight" in a neural network, and how does the network learn to adjust it?** A weight is a numerical value on the connection between two neurons that determines how much influence one has on the next. During training, the network makes a prediction, measures how wrong it was (the loss), and adjusts weights slightly to reduce that error via backpropagation — repeated millions of times until the weights converge on values that produce accurate outputs.

**7. Why did CNNs succeed in Computer Vision before NLP saw similar breakthroughs?** CNNs are well suited to images because visual patterns (edges, shapes, objects) are spatially local and consistent regardless of surrounding context — a cat is a cat regardless of what's next to it. Language, by contrast, is sequential and deeply context-dependent (meaning shifts based on word order, sarcasm, and references), which the tools available at the time (N-grams, RNNs) weren't equipped to fully capture.

**8. Name three specific reasons why human language is harder for machines to process than images.** Sarcasm (literal words can mean the opposite of intent), ambiguity (a word like "bank" means different things in different contexts), and coreference (figuring out what a pronoun like "it" refers to) are three — word order and long-range context dependency are two more.

**9. What is an N-gram model, and what is its core limitation?** An N-gram model predicts the next word purely based on the statistical probability of word sequences seen in training data (e.g., "York" is likely after "New"). Its core limitation is that it has no real understanding — it breaks down quickly on longer, more meaningful sentences because it's only tracking shallow word-frequency patterns, not actual meaning.

**10. Explain the "context loss" problem in RNNs. Why does it get worse with longer text?** RNNs process text sequentially, one word at a time, carrying forward a "memory" (hidden state) to the next step. As the sequence gets longer, information from earlier words gets progressively diluted in that memory — like a game of telephone — so by the time the model reaches the end of a long passage, important early details may have effectively been forgotten.

**11. What was the key idea introduced in "Attention Is All You Need" (2017)?** The paper introduced the Transformer architecture, which processes an entire block of text in parallel rather than word-by-word, and introduced the "Attention" mechanism, which directly calculates the relevance between every word and every other word in the text regardless of distance.

**12. How does the Attention mechanism solve the context-loss problem that RNNs suffered from?** Because Attention gives every word direct, weighted access to every other word in the sequence simultaneously, there's no sequential memory that has to carry information forward and gradually degrade. A word at the end of a paragraph can attend directly back to a critical word at the very start, with no information loss in between.

**13. What does each letter in "GPT" stand for, and what does each term actually mean?** Generative means it produces new content rather than just classifying existing content. Pre-trained means it's first trained broadly on massive general data before being adapted for specific tasks. Transformer refers to the underlying architecture built on the Attention mechanism.

**14. What is a Large Language Model (LLM), in your own words?** An LLM is a Transformer-based neural network trained on enormous amounts of text with billions (or trillions) of parameters, enabling it to generate coherent, context-aware, human-like language by predicting text based on patterns learned during training.

### Applied / Scenario-Based

**15. A junior engineer says, "We can just plug ChatGPT into our support widget and we're done." What risks would you flag, and why?** A raw LLM has no knowledge of the company's actual policies, live order data, or edge cases, so it will confidently hallucinate answers (e.g., inventing a return policy) that sound plausible but are wrong. I'd flag the need for grounding the model in real, current business context (via retrieval and tool access) and guardrails before ever exposing it to customers.

**16. Walk through how you'd design a system so that an LLM never invents a fake company policy.** I'd retrieve the actual, current policy document at query time and inject it into the model's context (RAG), explicitly instruct the model to answer only from the provided context, add a fallback instruction to say "I don't know, let me connect you to a human" rather than guess, and continuously evaluate the system against real edge cases before and after launch.

**17. If a client says "we want an AI assistant," what questions would you ask before writing a single line of code?** I'd ask what specific business metric they're trying to move (resolution time, cart abandonment, conversion), what the current process looks like today, where the actual friction or bottleneck is, what data and systems the AI would need access to, and what "success" looks like in concrete, measurable terms.

**18. Explain how Retrieval-Augmented Generation (RAG) helps ground an LLM's answers in real, current data.** RAG fetches relevant, up-to-date information (like a policy document or a customer's order record) from an external source at query time and injects it into the model's prompt, so the model answers based on real retrieved facts instead of relying purely on its static training data — which reduces hallucination and keeps answers current.

**19. Why can't you simply hardcode all possible business rules the way ELIZA did, for a modern customer support use case?** Real customer questions have effectively infinite phrasings, edge cases, and combinations of context — you cannot enumerate every possible `IF/ELSE` branch. Rule-based systems break the moment a user phrases something even slightly differently than what was explicitly coded for, which doesn't scale to real-world language.

**20. A stakeholder is impressed that a chatbot "sounds human." How would you (gently) explain why that alone doesn't mean it's reliable or safe to deploy?** I'd explain that sounding human (passing something like a casual Turing Test) only measures fluency and style, not factual accuracy — a model can generate a confident, human-sounding sentence that is completely wrong. Reliability requires separately verifying the model is grounded in real data and tested against edge cases, not just that its tone feels natural.

### FDE Role-Specific

**21. How does the day-to-day job of a Forward Deployed Engineer differ from a traditional backend/product engineer?** A traditional engineer typically builds generic, reusable features against a fairly well-defined roadmap for an unknown mass of users. An FDE works directly inside a specific client's environment, discovers ambiguous and evolving requirements firsthand, and is judged by whether a measurable business outcome improved — not just whether a feature shipped.

**22. Describe the process of translating a vague client ask ("build us an AI chatbot") into a measurable business outcome.** I'd start by understanding the underlying business pain (e.g., slow support resolution), identify the metric leadership actually cares about, observe the current workflow to find where the real friction is, scope a narrow solution targeted at that specific friction point, and define a clear success metric to validate against after deployment.

**23. What does "providing context" to an LLM practically mean, and why is it the crux of an FDE's technical work?** Practically, it means retrieving relevant company data (policies, order history, documentation) and feeding it into the model's prompt at the right moment, and/or giving the model tool access to query live systems directly. It's the crux of FDE work because a generic LLM has no inherent knowledge of a specific business — grounding it in real context is what turns a generic model into a reliable, business-specific tool.

**24. Why might reducing hallucinations be more important to a business outcome than making the AI maximally "capable" or clever?** A highly capable but unreliable AI that occasionally invents wrong policies or facts can actively damage customer trust and cost the business money — one bad hallucinated answer can outweigh many good ones. For most business deployments, consistent trustworthiness matters more than raw cleverness, since the AI is representing the company directly to customers or employees.

**25. What trade-offs would you weigh between building a narrow, highly reliable AI tool vs. a broad, general-purpose one for a client?** A narrow tool is easier to ground with the right context, easier to test exhaustively, and more reliable in production, but it only solves one specific problem. A broad tool is more flexible and impressive in demos but harder to guardrail against hallucination and edge cases. In most early client deployments, I'd favor starting narrow and reliable, then expanding scope once trust and evaluation infrastructure are established.
