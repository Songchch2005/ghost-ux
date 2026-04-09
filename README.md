# Ghost-UX

Ghost-UX is an AI UX testing agent built on top of Vision LLMs and Playwright.
You give it a website, a persona, and a task goal. It opens the page, observes what is visible, decides what to do next, and produces a replayable UX test report.

English is the default project language, but Chinese personas, keywords, and test inputs are still supported.

## Current capabilities

- Async Playwright-driven agent loop
- Screenshot + compact DOM + structured `UIAction`
- OpenAI-compatible model adapter layer, plus mock / replay mode
- Markdown report and `playback.html`
- Input-side sensory filters:
  - `colorblindness`
  - `low_patience`
  - `symbol_cognition`
  - `cognitive`
- Output-side motor noise:
  - `motor_noise` for tremor, drunk, subway one-hand use, and related action drift
- Leak diagnosis tooling and replayable browser fixtures
- Local designer-friendly web app

## Installation

```powershell
pip install -e .[dev]
playwright install chromium
```

You can also create a local `.env` file in the project root:

```env
OPENAI_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```

## Quick start

Run a session from config:

```powershell
ghost-ux run --config examples/config.json
```

Or launch the local web app for designers:

```powershell
ghost-ux app
```

Then open [http://127.0.0.1:8765](http://127.0.0.1:8765) in your browser.

The web app lets you fill in:

- URL
- Persona
- Goal
- Optional provider / model / language / API key env / replay settings

Run an environment check:

```powershell
ghost-ux doctor
ghost-ux doctor --browser-check
```

## Outputs

Each run writes to `artifacts/<session_id>/`:

- `step_01.png`, `step_02.png`: per-step screenshots
- `session.log`: detailed runtime log
- `report.md`: Markdown report
- `playback.html`: replayable HTML report

## Model configuration

Ghost-UX uses an OpenAI-compatible API style by default.

- `provider=openai`: OpenAI-compatible chat completions endpoint
- `provider=gemini`: Gemini's OpenAI-compatible endpoint
- `provider=mock`: local replay fixture for CI and regression tests
- Other compatible services: set `base_url`

Minimal example:

```json
{
  "start_url": "https://example.com",
  "model": {
    "provider": "openai",
    "model": "gpt-4o",
    "api_key_env": "OPENAI_API_KEY",
    "language": "en"
  },
  "agent": {
    "persona": "A first-time visitor who is cautious, impatient with unclear UI, and wants obvious next steps.",
    "goal": "Find pricing and start a free trial."
  }
}
```

## Sensory filters

Ghost-UX inserts a filter pipeline between `Observe -> Think` so the model does not receive perfect input.

### `colorblindness`

Supports:

- `achromatopsia`
- `protanopia`

### `low_patience`

Simulates a user who stops reading long text and loses interest in content that sits too low on the page.

### `symbol_cognition`

Blinds icon-only controls when they have no visible text support.

### `cognitive`

Scrubs domain jargon for non-expert users. It supports built-in dictionaries for `general`, `b2b_saas`, `ai`, and `web3`, as well as custom terms.

By default, jargon gets replaced with:

```text
[unfamiliar jargon]
```

## Motor noise

Ghost-UX can also intervene on the action side.

`motor_noise` perturbs click and scroll execution so the agent does not behave like a perfectly precise robot.

Profiles include:

- `tipsy`
- `drunk`
- `subway_one_hand`
- `tremor`
- `parkinson_light`
- `parkinson_strong`

Playback and Markdown reports show:

- intended click point
- actual click point
- offset
- hit target summary
- tactile feedback shown to the model

## Leak diagnosis

Use the leak diagnosis command to understand whether a model is reading hidden information from DOM, screenshot OCR, or prompt context:

```powershell
ghost-ux diagnose-leak --config examples/test_cognitive_jargon.json
```

Artifacts include:

- raw observation
- filtered observation
- prompt payload probes
- diagnosis summary

## Notes on language support

- Default UI and runtime feedback are English
- Chinese persona descriptions and keyword triggers are still supported
- You can set `"language": "zh"` inside the `model` config if you want tactile feedback and related runtime prompt text in Chinese

## Example configs

See [examples/config.json](C:\Users\SCC\Desktop\gi\examples\config.json) for the baseline starter config.

Other useful examples:

- [examples/test_cognitive_b2b_saas.json](C:\Users\SCC\Desktop\gi\examples\test_cognitive_b2b_saas.json)
- [examples/test_cognitive_jargon.json](C:\Users\SCC\Desktop\gi\examples\test_cognitive_jargon.json)
- [examples/test_colorblind.json](C:\Users\SCC\Desktop\gi\examples\test_colorblind.json)
- [examples/test_icon_normal.json](C:\Users\SCC\Desktop\gi\examples\test_icon_normal.json)
- [examples/test_low_patience.json](C:\Users\SCC\Desktop\gi\examples\test_low_patience.json)
