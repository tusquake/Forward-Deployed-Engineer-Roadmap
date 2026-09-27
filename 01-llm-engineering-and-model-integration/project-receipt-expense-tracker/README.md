# Module 1 Capstone Project: The Receipt & Expense Tracker

> Part of the **Forward Deployed Engineering (FDE) Mastery Repo** — this replaces the original "AI Interview Coach" capstone for Module 1, using Google's Gemini API free tier so the entire project can be built and run at zero cost. It is designed to exercise every major concept from Topics 4, 5, and 6 in a single, genuinely useful application: point a phone camera at a paper receipt, and get back clean, structured spending data you can actually use.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Why Gemini for This Project](#2-why-gemini-for-this-project)
3. [Module 1 Concepts This Project Exercises](#3-module-1-concepts-this-project-exercises)
4. [Architecture](#4-architecture)
5. [Setup & Quickstart](#5-setup--quickstart)
6. [Step-by-Step Build](#6-step-by-step-build)
7. [Making It Robust: Guardrails and Cost Discipline](#7-making-it-robust-guardrails-and-cost-discipline)
8. [Extending the Project](#8-extending-the-project)
9. [Interview Questions & Answers](#9-interview-questions--answers)

---

## 1. Project Overview

The **Receipt & Expense Tracker** takes a photo of a paper (or digital) receipt and turns it into clean, structured expense data: merchant name, date, line items, category, and total — ready to drop into a spreadsheet, a budgeting app, or a database, with no manual data entry. A user should be able to snap a photo after any purchase and have it show up, correctly categorized, in a running expense log seconds later.

This is a deliberately practical choice for a capstone: it's something you (or a client) would genuinely use, it touches multimodal input, structured outputs, reasoning, and cost-conscious API design all in one small project, and it runs entirely on Gemini's free tier, so there's no billing risk while you're learning.

---

## 2. Why Gemini for This Project

Gemini's free tier (via Google AI Studio) is the practical choice here for one specific reason: it's a genuinely **multimodal**, permanently free tier — you can send an image directly alongside a text prompt, with no credit card and a generous enough daily request/token allowance to comfortably build and test a real project. This is a direct application of Topic 6's multimodal and document-understanding concepts — rather than running a separate OCR step and then feeding extracted text to a model, Gemini can read the receipt image natively in the same call that also produces your structured output.

A quick, honest caveat worth internalizing as an FDE habit: free-tier limits, model names, and pricing structures for every provider (Gemini included) shift frequently — check `ai.google.dev` for Gemini's current free-tier limits and the current recommended Flash model name before you build, rather than trusting any specific number as permanent.

---

## 3. Module 1 Concepts This Project Exercises

| Concept (from Topics 4–6) | How This Project Uses It |
| --- | --- |
| Multimodal inputs (Topic 6) | Sending the receipt photo directly to the model alongside a text instruction |
| Document understanding (Topic 6) | Extracting structured fields (merchant, date, items, total) from a visually messy, real-world document |
| Structured outputs / Pydantic (Topic 6) | Forcing every response into a strict schema so it can be inserted directly into a spreadsheet or database with no fragile text-parsing |
| Function calling (Topic 6) | Optionally letting the model call a "categorize spending" or "flag as duplicate" tool rather than just describing what it sees |
| Chain-of-Thought (Topic 5) | Prompting the model to reason step by step when line-item totals don't cleanly match the printed total, before deciding how to resolve the discrepancy |
| System prompt architecture (Topic 5) | A persistent system prompt that defines the extraction task, the exact schema, and what to do with illegible or ambiguous receipts |
| Prompt injection defense (Topic 5) | Recognizing that a receipt is untrusted, uncontrolled input (anyone could hand you a receipt with text printed on it) and treating it as data, never as instructions |
| Tokenization & cost (Topic 6) | Being deliberate about image size and prompt length, since image tokens are a real, measurable cost even on a free tier with daily caps |
| Reasoning models (Topic 4) | Optionally reaching for a reasoning-capable model only for the harder cases (crumpled, faded, or non-English receipts), not for every request |

---

## 4. Architecture

```text
[Phone Camera / Image Upload]
             │
             ▼
[Web UI: Drag & Drop + Live Preview + Expense Table]
             │
             ▼
[FastAPI Backend Service (`src/app.py`)]
   1. Validates upload & downsamples image (`Pillow`) for token reduction
   2. Injects system prompt with prompt-injection defense boundaries
   3. Calls Gemini API (`gemini-2.5-flash` / `gemini-1.5-flash`) with Pydantic `response_schema`
   4. Runs deterministic arithmetic sanity check (sum of line items + tax vs total)
   5. Persists validated record to SQLite database & CSV backup
             │
             ▼
[Real-Time Dashboard & CSV Export]
```

---

## 5. Setup & Quickstart

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Configure API Key**:
   Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com).
   Create a `.env` file in this directory:
   ```bash
   GEMINI_API_KEY="your_actual_key_here"
   ```
3. **Run the Application**:
   ```bash
   python -m src.app
   ```
4. Open your browser at `http://127.0.0.1:8000` to interact with the UI.

---

## 6. Project Structure

```text
project-receipt-expense-tracker/
├── README.md
├── requirements.txt
├── .env.example
├── sample_receipts/
├── src/
│   ├── __init__.py
│   ├── schemas.py       # Pydantic schemas (Receipt, LineItem, Category)
│   ├── extractor.py     # Gemini multimodal API client + guardrails
│   ├── storage.py       # SQLite database & CSV export engine
│   └── app.py           # FastAPI server & REST API
└── static/
    ├── index.html       # Sleek responsive web dashboard
    ├── style.css        # Premium modern design system
    └── app.js           # Drag & drop, camera capture & live updates
```
