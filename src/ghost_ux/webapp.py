from __future__ import annotations

import asyncio
import mimetypes
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from ghost_ux.agent import GhostUXAgent
from ghost_ux.config import AgentConfig, BrowserConfig, ModelConfig, ReportConfig, SessionConfig
from ghost_ux.models import RunArtifacts


@dataclass
class WebFormState:
    start_url: str = ""
    persona: str = ""
    goal: str = ""
    provider: str = "openai"
    model: str = "gpt-4o"
    language: str = "en"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str = ""
    replay_path: str = ""
    max_steps: str = "15"
    headed: bool = False
    output_dir: str = "artifacts"


def _bool_from_form(value: str | None) -> bool:
    return str(value or "").lower() in {"1", "true", "on", "yes"}


def build_config_from_form(form: dict[str, str]) -> SessionConfig:
    return SessionConfig(
        start_url=form["start_url"].strip(),
        browser=BrowserConfig(headless=not _bool_from_form(form.get("headed"))),
        model=ModelConfig(
            provider=form.get("provider", "openai").strip() or "openai",
            model=form.get("model", "gpt-4o").strip() or "gpt-4o",
            language=form.get("language", "en").strip() or "en",
            api_key_env=form.get("api_key_env", "OPENAI_API_KEY").strip() or "OPENAI_API_KEY",
            base_url=form.get("base_url", "").strip() or None,
            replay_path=Path(form["replay_path"].strip()) if form.get("replay_path", "").strip() else None,
        ),
        agent=AgentConfig(
            max_steps=int(form.get("max_steps", "15") or "15"),
            persona=form["persona"].strip(),
            goal=form["goal"].strip(),
        ),
        report=ReportConfig(output_dir=Path(form.get("output_dir", "artifacts").strip() or "artifacts")),
    )


def _default_form_state() -> WebFormState:
    return WebFormState(
        persona="A designer who is not deeply technical and wants to test whether the page feels clear, understandable, and easy to act on.",
        goal="Understand what this page is trying to do and attempt the most obvious next step.",
    )


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _recent_runs(output_root: Path, limit: int = 8) -> list[Path]:
    if not output_root.exists():
        return []
    return sorted(
        [path for path in output_root.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:limit]


def build_app_html(
    *,
    state: WebFormState | None = None,
    result: RunArtifacts | None = None,
    error: str | None = None,
    output_root: Path = Path("artifacts"),
) -> str:
    state = state or _default_form_state()
    runs = _recent_runs(output_root)
    recent_runs_html = "".join(
        f"""
        <a class="recent-run" href="/artifacts/{_escape(run.name)}/playback.html">
          <strong>{_escape(run.name)}</strong>
          <span>Open playback</span>
        </a>
        """
        for run in runs
    ) or "<p class='empty-hint'>No runs yet. Start your first test to see playback and reports here.</p>"

    result_html = ""
    if result:
        result_html = f"""
        <section class="result-card">
          <p class="eyebrow">Latest Run</p>
          <h2>Run complete</h2>
          <p class="result-copy">This run has been saved to your artifacts folder. Open the playback or Markdown report to continue reviewing the journey.</p>
          <div class="result-grid">
            <div><span>Final Status</span><strong>{_escape(result.final_status)}</strong></div>
            <div><span>Session</span><strong>{_escape(result.session_id)}</strong></div>
          </div>
          <div class="result-actions">
            <a class="primary-link" href="/artifacts/{_escape(result.session_id)}/playback.html">Open playback</a>
            <a class="secondary-link" href="/artifacts/{_escape(result.session_id)}/report.md">Open report</a>
          </div>
        </section>
        """

    error_html = (
        f"<section class='error-card'><strong>Run failed</strong><p>{_escape(error)}</p></section>"
        if error
        else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Ghost-UX Studio</title>
    <style>
      :root {{
        --bg: #f6efe3;
        --panel: rgba(255, 250, 242, 0.92);
        --ink: #1c1a16;
        --muted: #63584b;
        --line: rgba(72, 52, 25, 0.15);
        --accent: #db5d14;
        --accent-deep: #8f3610;
        --accent-soft: #fff0df;
        --danger: #9f1239;
        --shadow: 0 20px 60px rgba(98, 61, 25, 0.12);
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        min-height: 100vh;
        font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(255, 204, 153, 0.4), transparent 32%),
          linear-gradient(160deg, #f6efe3 0%, #fbf7ef 45%, #fffdf9 100%);
      }}
      .shell {{
        width: min(1200px, calc(100% - 32px));
        margin: 24px auto 40px;
        display: grid;
        grid-template-columns: minmax(0, 1.2fr) minmax(300px, 0.8fr);
        gap: 22px;
      }}
      .hero, .panel, .result-card, .error-card {{
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 24px;
        box-shadow: var(--shadow);
        backdrop-filter: blur(16px);
      }}
      .hero {{
        grid-column: 1 / -1;
        padding: 28px 30px;
        display: grid;
        grid-template-columns: minmax(0, 1fr) 280px;
        gap: 24px;
        align-items: end;
      }}
      .hero h1 {{
        margin: 0;
        font-size: clamp(2rem, 4vw, 3.4rem);
        line-height: 0.95;
        letter-spacing: -0.04em;
      }}
      .hero p {{
        margin: 14px 0 0;
        max-width: 58ch;
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.6;
      }}
      .hero-card {{
        padding: 18px;
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(255, 240, 223, 0.95), rgba(255, 248, 239, 0.8));
      }}
      .hero-card strong {{
        display: block;
        margin-top: 8px;
        font-size: 1.1rem;
      }}
      .panel {{
        padding: 24px;
      }}
      .panel h2, .result-card h2 {{
        margin: 0 0 8px;
        font-size: 1.4rem;
      }}
      .panel p, .result-card p {{
        color: var(--muted);
        line-height: 1.6;
      }}
      .eyebrow {{
        margin: 0 0 10px;
        color: var(--accent-deep);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.76rem;
        font-weight: 700;
      }}
      .stack {{
        display: grid;
        gap: 16px;
      }}
      label {{
        display: grid;
        gap: 8px;
        font-weight: 600;
      }}
      .hint {{
        margin: 0;
        color: var(--muted);
        font-size: 0.9rem;
        font-weight: 400;
      }}
      input, textarea, select {{
        width: 100%;
        border: 1px solid rgba(112, 76, 32, 0.18);
        border-radius: 16px;
        padding: 14px 15px;
        background: #fffdfa;
        color: var(--ink);
        font: inherit;
      }}
      textarea {{
        min-height: 132px;
        resize: vertical;
      }}
      .row {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 14px;
      }}
      .checkbox {{
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 600;
      }}
      .checkbox input {{
        width: 18px;
        height: 18px;
        margin: 0;
      }}
      details {{
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 14px 16px;
        background: rgba(255, 255, 255, 0.5);
      }}
      summary {{
        cursor: pointer;
        font-weight: 700;
      }}
      .actions {{
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        align-items: center;
      }}
      button {{
        border: none;
        border-radius: 999px;
        padding: 14px 22px;
        background: linear-gradient(135deg, var(--accent), #f08d24);
        color: white;
        font: inherit;
        font-weight: 700;
        cursor: pointer;
        box-shadow: 0 10px 30px rgba(219, 93, 20, 0.22);
      }}
      .muted-note {{
        color: var(--muted);
        font-size: 0.92rem;
      }}
      .recent-list {{
        display: grid;
        gap: 10px;
      }}
      .recent-run, .primary-link, .secondary-link {{
        text-decoration: none;
      }}
      .recent-run {{
        display: grid;
        gap: 4px;
        padding: 14px 16px;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.65);
        border: 1px solid rgba(112, 76, 32, 0.12);
        color: inherit;
      }}
      .recent-run span {{
        color: var(--muted);
        font-size: 0.92rem;
      }}
      .result-card, .error-card {{
        padding: 22px;
        margin-bottom: 18px;
      }}
      .error-card {{
        border-color: rgba(159, 18, 57, 0.2);
      }}
      .error-card strong {{
        display: block;
        margin-bottom: 8px;
        color: var(--danger);
      }}
      .result-grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
        margin: 18px 0;
      }}
      .result-grid div {{
        padding: 14px;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.62);
      }}
      .result-grid span {{
        display: block;
        color: var(--muted);
        font-size: 0.82rem;
        margin-bottom: 6px;
      }}
      .result-actions {{
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
      }}
      .primary-link, .secondary-link {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-height: 44px;
        padding: 0 18px;
        border-radius: 999px;
        font-weight: 700;
      }}
      .primary-link {{
        background: var(--accent);
        color: #fff;
      }}
      .secondary-link {{
        background: rgba(255, 240, 223, 0.9);
        color: var(--accent-deep);
      }}
      .empty-hint {{
        margin: 0;
      }}
      @media (max-width: 980px) {{
        .shell, .hero {{
          grid-template-columns: 1fr;
        }}
      }}
      @media (max-width: 640px) {{
        .row {{
          grid-template-columns: 1fr;
        }}
      }}
    </style>
  </head>
  <body>
    <main class="shell">
      <section class="hero">
        <div>
          <p class="eyebrow">Ghost-UX Studio</p>
          <h1>Paste a URL, describe the user, and launch a UX ghost test.</h1>
          <p>This page is built for designers who do not want to touch the command line. Enter a URL, a persona, and a goal. Ghost-UX will open the page, observe it, act on it, and generate replayable results.</p>
        </div>
        <aside class="hero-card">
          <p class="eyebrow">Writing tip</p>
          <strong>Who is this person, what state are they in, and what are they trying to do?</strong>
          <p>Example: "A 62-year-old retired first-time visitor with average vision who wants to find the pricing page and start a trial." You can write personas in either English or Chinese.</p>
        </aside>
      </section>

      <section>
        {error_html}
        {result_html}
        <section class="panel">
          <p class="eyebrow">New Session</p>
          <h2>Start a run</h2>
          <p>Begin with a small, specific goal. That usually produces more stable behavior and a much easier playback to review.</p>
          <form method="post" action="/run" class="stack">
            <label>
              URL
              <input type="url" name="start_url" required placeholder="https://example.com" value="{_escape(state.start_url)}" />
              <span class="hint">Supports public websites and absolute local fixture pages via <code>file:///</code>.</span>
            </label>
            <label>
              Persona
              <textarea name="persona" required placeholder="Describe who this user is, what state they are in, and what limitations they may have.">{_escape(state.persona)}</textarea>
            </label>
            <label>
              Goal
              <textarea name="goal" required placeholder="Describe what this user is trying to complete, for example: find pricing and start a trial.">{_escape(state.goal)}</textarea>
            </label>

            <details>
              <summary>Advanced settings</summary>
              <div class="stack" style="margin-top: 14px;">
                <div class="row">
                  <label>Provider
                    <input type="text" name="provider" value="{_escape(state.provider)}" />
                  </label>
                  <label>Model
                    <input type="text" name="model" value="{_escape(state.model)}" />
                  </label>
                </div>
                <div class="row">
                  <label>Language
                    <select name="language">
                      <option value="en" {"selected" if state.language == "en" else ""}>English (default)</option>
                      <option value="zh" {"selected" if state.language == "zh" else ""}>Chinese</option>
                    </select>
                  </label>
                  <label>API Key Env
                    <input type="text" name="api_key_env" value="{_escape(state.api_key_env)}" />
                  </label>
                </div>
                <div class="row">
                  <label>Max Steps
                    <input type="number" min="1" max="50" name="max_steps" value="{_escape(state.max_steps)}" />
                  </label>
                </div>
                <div class="row">
                  <label>Base URL
                    <input type="text" name="base_url" value="{_escape(state.base_url)}" placeholder="Optional: base_url for an OpenAI-compatible endpoint" />
                  </label>
                  <label>Replay Path
                    <input type="text" name="replay_path" value="{_escape(state.replay_path)}" placeholder="Optional: path to a mock/replay JSON file" />
                  </label>
                </div>
                <div class="row">
                  <label>Output Dir
                    <input type="text" name="output_dir" value="{_escape(state.output_dir)}" />
                  </label>
                  <label class="checkbox" style="margin-top: 30px;">
                    <input type="checkbox" name="headed" {"checked" if state.headed else ""} />
                    Run with a visible browser window
                  </label>
                </div>
              </div>
            </details>

            <div class="actions">
              <button type="submit">Run test</button>
              <span class="muted-note">This page will wait for the run to finish, then show direct links to playback and report.</span>
            </div>
          </form>
        </section>
      </section>

      <aside class="panel">
        <p class="eyebrow">Recent Runs</p>
        <h2>Recent runs</h2>
        <p>Open any run to jump straight into <code>playback.html</code>. For comparison testing, keep the page the same and rerun with different personas.</p>
        <div class="recent-list">
          {recent_runs_html}
        </div>
      </aside>
    </main>
  </body>
</html>
"""


def _artifact_root(output_root: Path) -> Path:
    return output_root.resolve()


def resolve_artifact_path(output_root: Path, request_path: str) -> Path | None:
    relative = request_path.removeprefix("/artifacts/").strip("/")
    if not relative:
        return None
    candidate = (_artifact_root(output_root) / unquote(relative)).resolve()
    try:
        candidate.relative_to(_artifact_root(output_root))
    except ValueError:
        return None
    if not candidate.exists() or not candidate.is_file():
        return None
    return candidate


async def run_web_session(config: SessionConfig) -> RunArtifacts:
    agent = GhostUXAgent(config)
    return await agent.run()


def _read_form_data(handler: BaseHTTPRequestHandler) -> dict[str, str]:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    raw = handler.rfile.read(length).decode("utf-8")
    parsed = parse_qs(raw, keep_blank_values=True)
    return {key: values[-1] for key, values in parsed.items()}


class GhostUXWebHandler(BaseHTTPRequestHandler):
    workspace_root: Path = Path.cwd()
    output_root: Path = Path("artifacts")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._write_html(build_app_html(output_root=self.output_root))
            return
        if parsed.path.startswith("/artifacts/"):
            artifact = resolve_artifact_path(self.output_root, parsed.path)
            if not artifact:
                self.send_error(HTTPStatus.NOT_FOUND, "Artifact not found.")
                return
            mime_type, _ = mimetypes.guess_type(str(artifact))
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime_type or "application/octet-stream")
            self.end_headers()
            self.wfile.write(artifact.read_bytes())
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Route not found.")

    def do_POST(self) -> None:
        if self.path != "/run":
            self.send_error(HTTPStatus.NOT_FOUND, "Route not found.")
            return
        form = _read_form_data(self)
        state = WebFormState(**{**_default_form_state().__dict__, **form, "headed": _bool_from_form(form.get("headed"))})
        try:
            config = build_config_from_form(form)
            result = asyncio.run(run_web_session(config))
        except Exception as exc:  # noqa: BLE001
            self._write_html(build_app_html(state=state, error=str(exc), output_root=Path(form.get("output_dir", "artifacts") or "artifacts")))
            return
        self._write_html(build_app_html(state=state, result=result, output_root=config.report.output_dir))

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _write_html(self, html: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def serve_web_app(host: str = "127.0.0.1", port: int = 8765, *, output_root: Path = Path("artifacts")) -> ThreadingHTTPServer:
    resolved_output_root = output_root

    class Handler(GhostUXWebHandler):
        workspace_root = Path.cwd()
        output_root = resolved_output_root

    return ThreadingHTTPServer((host, port), Handler)
