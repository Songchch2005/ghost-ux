from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from ghost_ux.agent import GhostUXAgent
from ghost_ux.config import AgentConfig, BrowserConfig, ModelConfig, ReportConfig, SessionConfig
from ghost_ux.diagnostics import run_diagnostics, run_leak_diagnosis
from ghost_ux.webapp import serve_web_app


console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ghost-UX: AI ghost tester for UX journeys.")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run a Ghost-UX session.")
    run_parser.add_argument("--config", type=Path, help="Path to a JSON config file.")
    run_parser.add_argument("--url", help="Target URL.")
    run_parser.add_argument("--persona", help="Persona prompt for the AI tester.")
    run_parser.add_argument("--goal", help="Task goal for the AI tester.")
    run_parser.add_argument("--model", default="gpt-4o", help="Vision-capable model name.")
    run_parser.add_argument("--provider", default="openai", help="Model provider key.")
    run_parser.add_argument("--replay-path", type=Path, help="Path to a mock replay JSON fixture.")
    run_parser.add_argument("--base-url", help="Optional OpenAI-compatible API base URL.")
    run_parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="API key environment variable.")
    run_parser.add_argument("--headed", action="store_true", help="Run browser in headed mode.")
    run_parser.add_argument("--max-steps", type=int, default=15, help="Maximum agent loop steps.")
    run_parser.add_argument("--output-dir", type=Path, default=Path("artifacts"), help="Artifacts directory.")

    doctor_parser = subparsers.add_parser("doctor", help="Inspect runtime health for Ghost-UX.")
    doctor_parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="API key environment variable.")
    doctor_parser.add_argument(
        "--browser-check",
        action="store_true",
        help="Attempt to launch Chromium and load a test page.",
    )

    diagnose_parser = subparsers.add_parser("diagnose-leak", help="Diagnose DOM/image/context leakage sources.")
    diagnose_parser.add_argument("--config", type=Path, required=True, help="Path to a JSON config file.")

    app_parser = subparsers.add_parser("app", help="Launch the local Ghost-UX web app.")
    app_parser.add_argument("--host", default="127.0.0.1", help="Host to bind the local web app.")
    app_parser.add_argument("--port", type=int, default=8765, help="Port to bind the local web app.")
    app_parser.add_argument("--output-dir", type=Path, default=Path("artifacts"), help="Artifacts directory.")

    parser.set_defaults(command="run")
    return parser


def build_config_from_args(args: argparse.Namespace) -> SessionConfig:
    if args.config:
        return SessionConfig.from_json_file(args.config)
    if not args.url or not args.persona or not args.goal:
        raise SystemExit("When --config is absent, --url, --persona, and --goal are required.")
    return SessionConfig(
        start_url=args.url,
        browser=BrowserConfig(headless=not args.headed),
        model=ModelConfig(
            provider=args.provider,
            model=args.model,
            replay_path=args.replay_path,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
        ),
        agent=AgentConfig(
            max_steps=args.max_steps,
            persona=args.persona,
            goal=args.goal,
        ),
        report=ReportConfig(output_dir=args.output_dir),
    )


async def _run(args: argparse.Namespace) -> int:
    if args.command == "doctor":
        return await run_diagnostics(
            api_key_env=args.api_key_env,
            include_browser=args.browser_check,
        )
    if args.command == "diagnose-leak":
        config = SessionConfig.from_json_file(args.config)
        exit_code, diagnosis_path = await run_leak_diagnosis(config)
        console.print(
            Panel.fit(
                f"Leak diagnosis written to: {diagnosis_path}",
                title="Ghost-UX Diagnosis",
                border_style="cyan",
            )
        )
        return exit_code
    if args.command == "app":
        server = serve_web_app(args.host, args.port, output_root=args.output_dir)
        console.print(
            Panel.fit(
                (
                    f"Ghost-UX Studio is running.\n"
                    f"Open: http://{args.host}:{args.port}\n"
                    f"Artifacts: {args.output_dir}"
                ),
                title="Ghost-UX App",
                border_style="cyan",
            )
        )
        try:
            await asyncio.to_thread(server.serve_forever)
        finally:
            server.server_close()
        return 0

    config = build_config_from_args(args)
    agent = GhostUXAgent(config)
    result = await agent.run()
    console.print(
        Panel.fit(
            (
                f"Final status: {result.final_status}\n"
                f"Artifacts: {result.session_dir}\n"
                f"Report: {result.report_path}\n"
                f"Playback: {result.playback_path}"
            ),
            title="Ghost-UX Complete",
            border_style="green",
        )
    )
    return 0


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run(args)))
