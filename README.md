# KBMOD Angle Rate Tool

Planning tools for KBMOD moving-object searches in reflex-corrected frames.
Hosted app: https://kbmodangleratetool.streamlit.app/

Two modes:

- **Recommend a configuration** (new in 0.2). Give it a reflex-correction
  distance and the shell of true distances, inclinations and solar elongations the
  search must cover, plus the window span and PSF. It maps that shell into
  reflex-frame velocity space, sizes single-fan, two-tier and velocity-box grids,
  picks one, and writes a complete pool-stage kbmod search YAML in which every
  setting carries the measured result it rests on.
- **Evaluate a grid.** The original rate/angle evaluator: for an existing
  `EclipticCenteredSearch` grid, how far adjacent angle rays diverge over the
  window, how many angles the PSF would need, and reference TNO/Trojan rates.

## What the recommender encodes

The physics is the analytic reflex-frame model of the 2026-09-18 KBMOD
search-strategy audit (section 5), validated there against kbmod's own
reflex-corrected tracks of the Sorcha injections and six real TNOs on Rubin patch
471085. The tests in `tests/test_reflex_search_space.py` pin the module to the
audit's tables (orbital rates, cancellation distances, coherent-window lengths, the
velocity-floor blind zone, the distance ladder, trajectory counts).

| quantity | rule |
|---|---|
| orbital rate | `omega(d) = k d^-1.5` (px/d): 65 at 42 au, 30 at 70 au |
| along-ecliptic rate | `omega(d) cos i + C1 |cos e| (1/d_guess - 1/d)`; objects inside the guess slow down and eventually look retrograde (cancellation at ~61 au for a 70 au node) |
| cross-ecliptic rate | `+/- omega(d) sin i`, independent of the guess; the wide-angle tail of a population is entirely slow |
| curvature | departure from a straight line grows as `|1/d - 1/d_guess| x (T^2 off opposition, T^3 at opposition)`; this sets the longest window a shell of distances tolerates at one node |
| grid steps | `dv <= 2 tol / T`, `dtheta <= 2 tol / (v T)` (kbmod-ops `velocity_grid_coverage`) |

Measured policy carried into every configuration (E1-E3 and the 2026-10-01 arm
cells on the 42 au golden patches; see the 42 au search recipe in
kbmod-mission-control):

- `results_per_pixel` 64 (128 for windows under 3.5 d): the largest lever at short
  spans (4/8/16/64/128 recovered 39/53/77/131/168 at 3 d), marginal at 7 d.
- A narrow fan recovers more than +/-90 deg at fixed `results_per_pixel`; the
  recommender accepts a single fan up to +/-45 deg and otherwise proposes two speed
  tiers (wide and slow, narrow and fast) or a Cartesian velocity box.
- Grid step at the `dv x span <= FWHM` law (finer steps bought nothing at 3-8 d).
- Search into a pool: clustering off, `max_results: -1`, `lh_level` 4.0, GPU
  sigma-G on, stamps off (`coadds: []`, `stamp_type: null`, `save_all_stamps: false`).
- Windows of 3-7 d; the recommender warns when the span exceeds the coherent window
  of the requested shell and gives the distance ladder that would cover it.

The model assumes circular barycentric orbits and a field on the ecliptic; it has
no population prior, so the measured 471085 fan fractions are shown as context.

## Installation

```bash
python -m pip install -e .
```

To enable the Streamlit app, install the interactive extra:

```bash
python -m pip install -e ".[interactive]"
```

## Command line

Recommend a configuration for a 70 au node covering 65-75 au over 7-day windows and
write the YAMLs, report and plots to a directory:

```bash
kbmod-reflex-recommend --node 70 --half-width 5 --span 7 --inclination 30 --elongation 135 180 --out rec70
```

Evaluate an existing grid (the original tool):

```bash
kbmod-rate-angle-evaluator 10
```

`kbmod-rate-angle-evaluator` estimates the configured angle and velocity sampling
required for a KBMOD-like search and writes a scaled search-grid PNG and PDF and a
text report (`*_report.txt`). Angle bounds are offsets from KBMOD's per-field,
WCS-derived ecliptic direction; pass `--ecliptic-angle-deg` to resolve absolute
image angles.

## Streamlit app

```bash
kbmod-rate-angle-evaluator-app
```

or, for Streamlit Cloud or any script-based deployment:

```bash
streamlit run streamlit_app.py
```

The sidebar switches between the two modes. In recommend mode the page shows the
occupied region of velocity space with the three grid designs drawn over it, the
coherent window and distance ladder for the requested coverage, a design table, the
selected kbmod YAML, and downloads for one YAML or a ZIP of everything.

## Files

- `src/kbmod_angle_rate_tool/reflex_search_space.py`: the reflex-frame model, grid
  designs, recommendation, YAML/report writers, plots and CLI.
- `src/kbmod_angle_rate_tool/kbmod_rate_angle_evaluator.py`: the rate/angle grid
  evaluator (CLI and library).
- `src/kbmod_angle_rate_tool/kbmod_rate_angle_interactive.py`: the Streamlit app.
- `examples/search_config.yaml`: a KBMOD search config with the production grid.
- `tests/`: regression tests, including the audit anchors.
