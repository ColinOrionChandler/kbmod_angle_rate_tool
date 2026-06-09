# KBMOD Angle Rate Tool

This repository contains interactive and CLI tooling for KBMOD rate/angle
sampling analysis.

## Installation

```bash
python -m pip install -e .
```

To enable the Streamlit app, install the interactive extra:

```bash
python -m pip install -e ".[interactive]"
```

## Command-line evaluator

```bash
kbmod-rate-angle-evaluator 10
```

`kbmod-rate-angle-evaluator` estimates the configured angle and velocity sampling
required for a KBMOD-like search and writes:

- a scaled search-grid PNG (`.png`)
- a matching PDF (`.pdf`)
- a text report (`*_report.txt`)

By default, outputs are emitted next to the working directory using a descriptive
filename and include both KBMOD defaults and any requested YAML overrides.

## Streamlit app

```bash
kbmod-rate-angle-evaluator-app
```

`kbmod-rate-angle-evaluator-app` launches a Streamlit interface for the same
rate/angle model:

- interactive controls for patch size, timespan, angles, and velocity range
- live metrics and diagnostics
- on-demand zip export of:
  - the current plot as PNG/PDF
  - report text

### Streamlit deployment

For Streamlit Cloud or local script-based deployment, use this file as the app entry:

```bash
streamlit run streamlit_app.py
```

`streamlit_app.py` is a direct entrypoint that calls the same interactive app, so it
is safe to use in hosted deployments.

## Source config

Use this file when you want to reuse the KBMOD search-grid defaults:

- `examples/search_config.yaml`

## Files

- `src/kbmod_angle_rate_tool/kbmod_rate_angle_evaluator.py`
- `src/kbmod_angle_rate_tool/kbmod_rate_angle_interactive.py`
- `examples/search_config.yaml`
- `tests/test_kbmod_rate_angle_evaluator.py`
