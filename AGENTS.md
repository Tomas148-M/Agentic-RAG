# Workflow for Project Changes

These rules apply to implementation tasks in this repository. Explicit user requests and higher-priority instructions take precedence.

The main agent coordinates the workflow and communicates with the user. When custom subagents and delegation are available and permitted, delegate implementation and post-review fixes to `implementer`, verification to `verifier`, and the independent review to `reviewer`. Complete implementation before review; do not run edits concurrently with review. If a role is unavailable or delegation is not permitted, the main agent performs the work and discloses that limitation. The main agent assesses all results and remains responsible for the final outcome.

1. **Task intake:** Review the relevant code and existing changes. Clarify ambiguities that prevent progress; make other reasonable decisions independently. Preserve unrelated user changes.
2. **Implementation:** Make the requested change and add meaningful unit tests and documentation where appropriate. Keep the user informed of significant findings throughout the work.
3. **Verification:** Run the unit tests with `python -m pytest tests/unit` and static analysis with `python -m ruff check .`. Use the project's virtual environment if one exists. Run additional relevant checks as appropriate for the change. Fix failures and rerun the checks. If a check cannot be run, explain why and do not claim it passed.
4. **Code review:** Review the resulting diff for bugs, regressions, security issues, edge cases, and missing tests. If delegation is available and permitted, ask an independent subagent to review the implementation after it is complete; otherwise, conduct the review yourself and disclose that fact. The reviewer must not edit the files under review concurrently.
5. **Post-review changes:** Evaluate the feedback, fix relevant findings, and rerun the affected checks after making changes. Tell the user about any unresolved findings and why they remain unresolved.
6. **User acceptance:** Summarize the changes, check results, and any limitations. Explicitly ask the user to review and accept the work; mark its status as “awaiting user acceptance.” Completing the implementation or passing checks alone does not constitute acceptance. Wait for the user's response in the conversation without actively polling. If the user requests changes, make them and repeat the relevant checks and review. Mark the task complete only after explicit acceptance.

The acceptance rules do not apply to questions, analyses, or tasks for which the user explicitly requests a different process. Do not add approval steps of your own for routine intermediate operations.
