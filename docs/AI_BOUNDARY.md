# AI Change Boundary

This document defines the scope and limits for any AI-assisted changes in this repository.

## Purpose

The project is a finance-oriented RAG + Agent demo for a bachelor's-level portfolio. AI should improve clarity, reliability, and demo value, not inflate the codebase with unrelated or low-value features.

## Core Principle

Only make changes that strengthen one of these:

- finance-domain usefulness
- retrieval quality
- tool-use usefulness
- demo clarity
- maintainability

If a change does not clearly improve at least one of the above, do not add it.

## Allowed Scope

### RAG

- improve retrieval quality
- improve chunking, reranking, or source display
- improve answer grounding
- improve multi-turn understanding

### Agent

- improve tool selection
- improve calculator or search use
- improve tool-call transparency
- improve answer reliability

### Knowledge Base

- add official or clearly credible finance documents
- organize documents by topic
- keep the corpus small enough to stay understandable

### UI

- improve readability
- improve source display
- improve tool trace display
- improve the user flow of the two main apps

## Disallowed Scope

- generic chatbot features unrelated to finance
- random “smart” features without a clear user story
- heavy abstractions that do not remove real duplication or confusion
- large-scale framework replacement
- premature optimization
- speculative modules with no demo value
- adding many new dependencies without a strong reason

## Knowledge Base Policy

- Prefer official sources first.
- Prefer short, topic-focused documents.
- Do not mix unrelated domains into one file.
- Do not invent content to fill gaps.
- If an official source is missing, record the gap instead of guessing.

## Editing Policy

Before changing code, ask:

1. What user problem does this solve?
2. Does this fit the finance RAG / Agent story?
3. Does it make the demo better to explain?
4. Is the change small enough to verify?

If any answer is weak, stop and reconsider.

## Default Priority Order

1. Keep the current system working.
2. Improve finance knowledge grounding.
3. Improve tool behavior.
4. Improve UI clarity.
5. Add new capabilities only when they have a clear portfolio value.

## Good Examples

- unify duplicate session/database logic
- add official Basel / IFRS material
- improve citation/source display
- add a focused risk scoring helper
- make the startup path clearer

## Bad Examples

- adding a generic chat mode
- adding unrelated productivity tools
- adding a flashy feature with no finance use
- replacing the whole stack for no user-visible gain

## Final Rule

When in doubt, choose the smallest change that improves the finance RAG / Agent story.
