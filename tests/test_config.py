from ghost_ux.config import AgentConfig, SessionConfig


def _agent_config() -> AgentConfig:
    return AgentConfig(
        persona="A cautious tester.",
        goal="Open the page and inspect the main controls.",
    )


def test_session_config_accepts_http_url() -> None:
    config = SessionConfig(
        start_url="https://example.com",
        agent=_agent_config(),
    )
    assert config.start_url == "https://example.com"


def test_session_config_accepts_file_url() -> None:
    config = SessionConfig(
        start_url="file:///C:/Users/SCC/Desktop/gi/examples/symbol_cognition_fixture.html",
        agent=_agent_config(),
    )
    assert config.start_url.startswith("file:///")
