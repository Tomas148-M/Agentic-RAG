---
name: agent-verification
description: Verify assigned project changes with unit tests, static analysis,
  and other checks requested by the parent agent.
---

# Verification workflow

Verify assigned changes after implementation. Use the project virtual
environment when present. Run:

```text
python -m pytest tests/unit
python -m ruff check .
```

Also run relevant checks requested by the parent. Report the exact commands,
results, failures, warnings, and checks that could not run. Do not modify
project files. Do not claim a check passed if it did not run, and do not claim
user acceptance.
