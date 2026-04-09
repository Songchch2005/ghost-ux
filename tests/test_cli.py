from ghost_ux.cli import build_parser


def test_parser_defaults_to_run_command() -> None:
    parser = build_parser()
    args = parser.parse_args([])
    assert args.command == "run"


def test_parser_supports_doctor_browser_check() -> None:
    parser = build_parser()
    args = parser.parse_args(["doctor", "--browser-check"])
    assert args.command == "doctor"
    assert args.browser_check is True


def test_parser_supports_mock_replay_path() -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "--url", "https://example.com", "--persona", "P", "--goal", "G", "--provider", "mock", "--replay-path", "examples/mock_replay.json"])
    assert args.command == "run"
    assert str(args.replay_path).endswith("mock_replay.json")


def test_parser_supports_app_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["app", "--host", "127.0.0.1", "--port", "9000"])
    assert args.command == "app"
    assert args.host == "127.0.0.1"
    assert args.port == 9000
