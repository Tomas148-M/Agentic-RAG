---
name: agent-code-review
description: Independently review completed project changes for defects,
  regressions, security issues, edge cases, and missing tests.
---

# Code review workflow

Review only after implementation is complete. Do not edit files. Focus on
correctness, regressions, security, edge cases, and missing tests. Report
actionable findings with severity, file and line references, and supporting
evidence. Explicitly say when there are no findings and identify review
limitations. Do not claim user acceptance.

Run Bandit against application and script code:

```sh
python -m bandit -r app scripts
```

Assess each reported issue in context, report actionable findings with
severity and evidence, and do not dismiss findings solely because the scanner
reports a low confidence level. If Bandit cannot run, report that as a review
limitation rather than implying the security scan passed.

Audit installed Python dependencies with:

```sh
python -m pip_audit
```

Report vulnerable package versions and advisory identifiers, and distinguish
direct dependencies from transitive dependencies when the tool provides that
information. Do not suppress advisories without a documented reason. If the
audit cannot run, report that as a review limitation.

Check that CI writes reports only to the console unless the user explicitly
requests saved outputs. The CycloneDX SBOM should be generated from CI's
installed Python environment and printed to the console; do not require an
uploaded artifact.

Check that application constants and configuration parameters, except prompt
text, are defined in `app/core/config.py` and referenced from implementation
modules. Values that users or deployments may need to change should use
environment-backed settings.

Check that each application system prompt is kept in a separate,
purpose-specific file under `app/agents/prompts/`, imported or loaded by its
caller, and not embedded in services, API handlers, graph nodes, adapters, or
configuration values. Check that user questions and retrieved context remain
separate from system prompt text.

## FastAPI review checks

Apply these checks when relevant to the change:

- Keep blocking I/O out of async endpoints and other event-loop code. Prefer
  async clients for async routes, or synchronous route handlers/thread-pool
  execution for blocking operations.
- Keep CPU-heavy or long-running work out of request handling. Use a suitable
  worker or job queue when work must continue independently of a request.
  Reserve in-process background tasks for appropriate lightweight work.
- Reuse FastAPI dependencies for shared request-time concerns. Use Pydantic
  request/response models and field constraints for validation and schemas.
- Manage shared resources through application lifespan where appropriate,
  with cleanup on shutdown.
- Check secret handling, logging of sensitive data, and whether interactive
  API documentation should be exposed in the deployment environment.
- Check that health endpoints and process/container startup settings match
  the service's actual runtime and deployment needs.

Use the [FastAPI best-practices article](https://medium.com/@abipoongodi1211/fastapi-best-practices-a-complete-guide-for-building-production-ready-apis-bb27062d7617)
as a checklist, not as an unconditional mandate. Apply its advice to this
repository's architecture and the task. Do not require unrelated components
such as SQLAlchemy pooling, Celery, or Gunicorn unless the change makes them
applicable.
