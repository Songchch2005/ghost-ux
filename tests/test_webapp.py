from pathlib import Path
from uuid import uuid4

from ghost_ux.webapp import WebFormState, build_app_html, build_config_from_form, resolve_artifact_path


def _workspace_temp_dir() -> Path:
    path = Path("tests/.tmp") / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_build_config_from_form_creates_session_config() -> None:
    config = build_config_from_form(
        {
            "start_url": "https://example.com",
            "persona": "A first-time visitor.",
            "goal": "Find the main CTA.",
            "provider": "mock",
            "model": "mock-replay",
            "language": "zh",
            "api_key_env": "GHOST_UX_API_KEY",
            "replay_path": "examples/mock_replay.json",
            "max_steps": "8",
            "headed": "on",
            "output_dir": "artifacts",
        }
    )

    assert config.start_url == "https://example.com"
    assert config.browser.headless is False
    assert config.agent.max_steps == 8
    assert config.model.provider == "mock"
    assert config.model.language == "zh"
    assert str(config.model.replay_path).endswith("mock_replay.json")


def test_build_app_html_contains_designer_friendly_fields() -> None:
    tmp_path = _workspace_temp_dir()
    html = build_app_html(
        state=WebFormState(start_url="https://example.com"),
        output_root=tmp_path,
    )

    assert "Ghost-UX Studio" in html
    assert "Persona" in html
    assert "Goal" in html
    assert "English (default)" in html
    assert 'form method="post" action="/run"' in html
    assert "Run test" in html


def test_resolve_artifact_path_stays_inside_output_root() -> None:
    tmp_path = _workspace_temp_dir()
    session_dir = tmp_path / "demo"
    session_dir.mkdir()
    playback = session_dir / "playback.html"
    playback.write_text("ok", encoding="utf-8")

    assert resolve_artifact_path(tmp_path, "/artifacts/demo/playback.html") == playback.resolve()
    assert resolve_artifact_path(tmp_path, "/artifacts/../../secret.txt") is None
