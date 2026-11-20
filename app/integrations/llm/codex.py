# This adapter must invoke the configured Codex CLI and does not use a shell.
import subprocess  # nosec B404

from app.integrations.llm.base import LLMProvider, LLMProviderError


class CodexProvider(LLMProvider):
    """Generate responses using an authenticated, non-interactive Codex CLI."""

    def __init__(self, executable: str, model: str | None, timeout: float) -> None:
        self.executable = executable
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        """Generate text using Codex CLI.

        Args:
            prompt (str): Input text passed to the CLI through stdin.

        Returns:
            str: The final CLI message.

        Raises:
            LLMProviderError: If the process fails or returns no text.
        """
        command = [self.executable, "exec", "--sandbox", "read-only", "--skip-git-repo-check"]
        if self.model:
            command.extend(["--model", self.model])
        command.append("-")
        try:
            # The command uses fixed arguments; the untrusted prompt is sent via
            # stdin.
            result = subprocess.run(
                command, input=prompt, capture_output=True, text=True,
                encoding="utf-8", timeout=self.timeout, check=False,
            )  # nosec B603
        except (OSError, subprocess.TimeoutExpired, UnicodeError) as exc:
            raise LLMProviderError("Codex CLI could not complete the request") from exc
        if result.returncode != 0:
            raise LLMProviderError(f"Codex CLI failed with exit code {result.returncode}")
        if not result.stdout.strip():
            raise LLMProviderError("Codex CLI returned an empty response")
        return result.stdout.strip()
