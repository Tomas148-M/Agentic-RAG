---
name: agent-implementation
description: Implement assigned project changes using repository conventions,
  and address findings assigned after independent review.
---

# Implementation workflow

Implement only the assigned scope and preserve unrelated user changes. Add
meaningful tests and documentation when appropriate. Coordinate file ownership
with the parent agent, and do not edit files while an independent review is in
progress.

Follow [PEP 8](https://peps.python.org/pep-0008/) when writing or changing
Python code. Apply repository conventions and explicit user instructions
where they differ. Use judgment for readability and avoid unrelated,
style-only edits.

Keep application constants and configuration parameters, except prompt text,
in `app/core/config.py`. Reference these values from implementation modules.
Use environment-backed settings for values users or deployments may need to
change.

Keep each application system prompt in a separate, purpose-specific file
under `app/agents/prompts/`. Import or load prompts from those files. Do not
embed system prompt text in services, API handlers, graph nodes, adapters, or
configuration values. Keep variable user questions and retrieved context
separate from system prompt text.

After implementation, report changes, checks, and unresolved issues to the
parent. When assigned post-review fixes, assess the findings, make relevant
changes, and report the result. Do not claim user acceptance.
