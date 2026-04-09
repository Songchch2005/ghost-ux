from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ghost_ux.models import StepRecord


console = Console()


def setup_logging(log_path: Path) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format=(
            "<green>{time:HH:mm:ss}</green> "
            "<level>{level: <8}</level> "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
            "<level>{message}</level>"
        ),
    )
    logger.add(log_path, enqueue=False, backtrace=True, diagnose=False, level="DEBUG")


def print_banner(session_id: str, url: str, persona: str, goal: str) -> None:
    body = Text()
    body.append("Ghost-UX Session\n", style="bold cyan")
    body.append(f"Session: {session_id}\n", style="white")
    body.append(f"URL: {url}\n", style="white")
    body.append(f"Persona: {persona}\n", style="magenta")
    body.append(f"Goal: {goal}", style="yellow")
    console.print(Panel(body, border_style="cyan"))


def print_thought(step_index: int, thought: str, confidence: float) -> None:
    title = f"Step {step_index} | Confidence {confidence:.2f}"
    console.print(Panel(thought, title=title, border_style="magenta"))


def print_step_summary(records: list[StepRecord]) -> None:
    table = Table(title="Ghost-UX Timeline", show_lines=False)
    table.add_column("Step", justify="right")
    table.add_column("Action")
    table.add_column("Success")
    table.add_column("Details")
    for record in records:
        table.add_row(
            str(record.step_index),
            record.action.action_type.value,
            "yes" if record.execution_success else "no",
            record.execution_detail[:80],
        )
    console.print(table)
