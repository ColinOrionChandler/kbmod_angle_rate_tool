"""Streamlit app: recommend a KBMOD search configuration, or evaluate an existing grid."""

from __future__ import annotations

import inspect
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from kbmod_angle_rate_tool import kbmod_rate_angle_evaluator as evaluator
from kbmod_angle_rate_tool import reflex_search_space as reflex

MODE_RECOMMEND = "Recommend a configuration"
MODE_EVALUATE = "Evaluate a grid"
CUSTOM_NODE = "Custom"
DEFAULT_NODE_PRESET = "New Horizons field (70 au)"


@dataclass(frozen=True)
class AppInputs:
    """Validated interactive evaluator inputs."""

    timespan_days: float
    patch_width_arcmin: float
    patch_height_arcmin: float
    angle_min_deg: float
    angle_max_deg: float
    angle_samples: int
    velocity_unit: str
    velocity_min: float
    velocity_max: float
    velocity_samples: int
    seeing_arcsec: float
    show_tno_reference: bool
    show_trojan_reference: bool


@dataclass(frozen=True)
class RecommendInputs:
    """Validated inputs for the reflex-frame recommender."""

    node_au: float
    distance_min_au: float
    distance_max_au: float
    inclination_max_deg: float
    elongation_min_deg: float
    elongation_max_deg: float
    span_days: float
    pixel_scale_arcsec_per_pixel: float
    psf_fwhm_pixels: float
    tolerance_preset: str
    tilt_deg: float
    velocity_margin_fraction: float
    min_angle_half_width_deg: float
    single_fan_max_half_width_deg: float
    tier_overlap_fraction: float
    show_production_reference: bool


def run_app() -> None:
    """Launch the Streamlit app."""

    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - exercised without optional extra.
        raise RuntimeError(
            "Install the interactive extra with: pip install -e '.[interactive]'"
        ) from exc

    st.set_page_config(page_title="KBMOD Search Planner", layout="wide")
    with st.sidebar:
        mode = st.radio("Mode", [MODE_RECOMMEND, MODE_EVALUATE], index=0)
    if mode == MODE_RECOMMEND:
        _run_recommend_mode(st)
    else:
        _run_evaluate_mode(st)


# --------------------------------------------------------------------------- recommend mode
def _run_recommend_mode(st: object) -> None:
    st.title("KBMOD search planner: from reflex node and coverage to a configuration")
    st.caption(
        "Choose a reflex-correction distance and the shell of true distances, inclinations and "
        "elongations the search must cover. The planner maps that shell into reflex-frame velocity "
        "space, sizes the grids from the window span and the PSF, and writes a pool-stage kbmod "
        "configuration that carries the measured reason for every setting."
    )
    inputs = _render_recommend_controls(st)
    try:
        spec = reflex.CoverageSpec(
            guess_distance_au=inputs.node_au,
            distance_min_au=inputs.distance_min_au,
            distance_max_au=inputs.distance_max_au,
            inclination_max_deg=inputs.inclination_max_deg,
            elongation_min_deg=inputs.elongation_min_deg,
            elongation_max_deg=inputs.elongation_max_deg,
            pixel_scale_arcsec_per_pixel=inputs.pixel_scale_arcsec_per_pixel,
        )
        rules = reflex.DesignRules(
            velocity_margin_fraction=inputs.velocity_margin_fraction,
            min_angle_half_width_deg=inputs.min_angle_half_width_deg,
            tier_overlap_fraction=inputs.tier_overlap_fraction,
            pixel_frame_tilt_deg=inputs.tilt_deg,
            single_fan_max_half_width_deg=inputs.single_fan_max_half_width_deg,
        )
        recommendation = reflex.recommend_search_configuration(
            spec,
            span_days=inputs.span_days,
            psf_fwhm_pixels=inputs.psf_fwhm_pixels,
            tolerance_preset=inputs.tolerance_preset,
            rules=rules,
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    _render_recommend_warnings(st, recommendation)
    _render_recommendation_headline(st, recommendation)

    left, right = st.columns([3, 2], vertical_alignment="top")
    with left:
        fig = reflex.build_velocity_space_plot(
            recommendation, show_production_reference=inputs.show_production_reference
        )
        _show_plot(st, fig)
        import matplotlib.pyplot as plt

        plt.close(fig)
        st.caption(
            "Points: covered objects (colour = true distance; one arc per distance spans "
            "inclination 0 to the ceiling, both signs). Outlines: the three grid designs; "
            "dotted grey: the 64x64 production grid for scale. The dotted box is the pixel-frame "
            "velocity box after tilt padding, rotated back into this frame."
        )
    with right:
        _render_region_metrics(st, recommendation)
        _render_design_table(st, recommendation)

    _render_configuration_section(st, recommendation)
    _render_coherence_section(st, recommendation)
    _render_context_section(st, recommendation)


def _render_recommend_controls(st: object) -> RecommendInputs:
    with st.sidebar:
        st.header("Reflex node and coverage")
        preset_names = [*reflex.NODE_PRESETS, CUSTOM_NODE]
        preset = st.selectbox("Node preset", preset_names, index=preset_names.index(DEFAULT_NODE_PRESET))
        default_node = reflex.NODE_PRESETS.get(preset, 42.0)
        node_au = st.number_input(
            "Reflex-correction distance (au)",
            min_value=1.5,
            value=float(default_node),
            step=1.0,
            format="%.2f",
            key=f"node_{preset}",
            help="The helio_guess_dists value the WorkUnit was reprojected to.",
        )
        shape = st.radio("Coverage", ["node +/- half-width", "explicit distance range"], horizontal=True)
        if shape == "node +/- half-width":
            half_width = st.number_input(
                "Half-width (au)",
                min_value=0.1,
                value=5.0,
                step=0.5,
                format="%.2f",
                help="Cover true distances from node - half-width to node + half-width.",
            )
            distance_min_au = float(node_au) - float(half_width)
            distance_max_au = float(node_au) + float(half_width)
            st.caption(f"Covers {distance_min_au:.1f} to {distance_max_au:.1f} au.")
        else:
            distance_min_au = st.number_input(
                "Nearest true distance (au)",
                min_value=1.1,
                value=max(1.1, float(node_au) - 5.0),
                step=0.5,
                format="%.2f",
                key=f"dmin_{preset}",
            )
            distance_max_au = st.number_input(
                "Farthest true distance (au)",
                min_value=1.2,
                value=float(node_au) + 5.0,
                step=0.5,
                format="%.2f",
                key=f"dmax_{preset}",
            )
        inclination_max_deg = st.number_input(
            "Inclination ceiling (deg)",
            min_value=0.0,
            max_value=90.0,
            value=reflex.DEFAULT_INCLINATION_MAX_DEG,
            step=5.0,
            format="%.1f",
            help=(
                "Largest orbital inclination to cover. Cross-ecliptic rate is omega(d) sin(i). On the "
                "471085 model TNO population the effective inclination was about 17 deg at the 90th "
                "percentile and 46 deg at the 99th."
            ),
        )
        elongation = st.slider(
            "Solar elongation range (deg)",
            min_value=90,
            max_value=180,
            value=(int(reflex.DEFAULT_ELONGATION_RANGE_DEG[0]), int(reflex.DEFAULT_ELONGATION_RANGE_DEG[1])),
            step=5,
            help=(
                "Elongation of the field over the observations; roughly 180 minus the days from "
                "opposition. The 2025 New Horizons field ran 140-179 deg. Curvature is worst at the "
                "low end, so that end sets the coherent window."
            ),
        )

        st.header("Window and image")
        span_days = st.number_input(
            "Window span (days of data)",
            min_value=0.001,
            value=reflex.DEFAULT_SPAN_DAYS,
            step=1.0,
            format="%.3f",
            help=(
                "The span the search really covers (read the WINDOW log line, not the label). "
                "Grid steps scale as 1/span and the trajectory count as span^2."
            ),
        )
        pixel_scale = st.number_input(
            "Pixel scale (arcsec/px)",
            min_value=0.01,
            value=reflex.DEFAULT_PIXEL_SCALE_ARCSEC_PER_PIXEL,
            step=0.01,
            format="%.3f",
        )
        psf_fwhm_px = st.number_input(
            "PSF FWHM (px)",
            min_value=0.5,
            value=reflex.DEFAULT_PSF_FWHM_PIXELS,
            step=0.25,
            format="%.2f",
            help="5.35 px (1.07 arcsec) measured on the 471085 WorkUnit layers.",
        )
        st.caption(f"= {psf_fwhm_px * pixel_scale:.2f} arcsec")
        preset_keys = list(reflex.TOLERANCE_PRESETS)
        tolerance_preset = st.selectbox(
            "Step and curvature tolerance",
            preset_keys,
            index=preset_keys.index(reflex.DEFAULT_TOLERANCE_PRESET),
            format_func=lambda key: reflex.TOLERANCE_PRESETS[key][0],
            help=(
                "The nearest grid node drifts step x span / 2 over the window, and an off-node "
                "track departs from a straight line; both are held within this many pixels."
            ),
        )

        with st.expander("Design knobs", expanded=False):
            tilt_deg = st.number_input(
                "Ecliptic-to-pixel tilt to pad the velocity box for (deg)",
                min_value=0.0,
                max_value=90.0,
                value=reflex.DEFAULT_PIXEL_FRAME_TILT_DEG,
                step=1.0,
                format="%.1f",
                help="8 deg on patch 471085; up to 23.4 deg near the equinox points.",
            )
            velocity_margin = st.number_input(
                "Margin on velocity bounds (fraction)",
                min_value=0.0,
                max_value=1.0,
                value=reflex.DEFAULT_VELOCITY_MARGIN_FRACTION,
                step=0.05,
                format="%.2f",
            )
            min_half_width = st.number_input(
                "Smallest fan half-width (deg)",
                min_value=0.0,
                max_value=180.0,
                value=reflex.DEFAULT_MIN_ANGLE_HALF_WIDTH_DEG,
                step=1.0,
                format="%.1f",
                help="Guards against ecliptic-angle error and off-ecliptic fields.",
            )
            single_fan_max = st.number_input(
                "Widest fan to accept as a single fan (deg)",
                min_value=0.0,
                max_value=180.0,
                value=reflex.SINGLE_FAN_MAX_HALF_WIDTH_DEG,
                step=5.0,
                format="%.1f",
                help="The 2026-10-01 arm cells measured +/-30 and +/-45 deg fans; wider regions go to two tiers.",
            )
            tier_overlap = st.number_input(
                "Tier overlap (fraction of the split speed)",
                min_value=0.0,
                max_value=0.49,
                value=reflex.DEFAULT_TIER_OVERLAP_FRACTION,
                step=0.02,
                format="%.2f",
            )
            show_reference = st.toggle("Show the production 64x64 grid for scale", value=True)

    return RecommendInputs(
        node_au=float(node_au),
        distance_min_au=float(distance_min_au),
        distance_max_au=float(distance_max_au),
        inclination_max_deg=float(inclination_max_deg),
        elongation_min_deg=float(elongation[0]),
        elongation_max_deg=float(elongation[1]),
        span_days=float(span_days),
        pixel_scale_arcsec_per_pixel=float(pixel_scale),
        psf_fwhm_pixels=float(psf_fwhm_px),
        tolerance_preset=str(tolerance_preset),
        tilt_deg=float(tilt_deg),
        velocity_margin_fraction=float(velocity_margin),
        min_angle_half_width_deg=float(min_half_width),
        single_fan_max_half_width_deg=float(single_fan_max),
        tier_overlap_fraction=float(tier_overlap),
        show_production_reference=bool(show_reference),
    )


def _render_recommend_warnings(st: object, recommendation: reflex.Recommendation) -> None:
    for warning in recommendation.warnings:
        if warning.startswith("Model limits"):
            continue
        st.warning(warning)


def _render_recommendation_headline(st: object, recommendation: reflex.Recommendation) -> None:
    design = recommendation.recommended
    st.success(f"**Recommended: {design.title}.** {recommendation.recommendation_reason}")
    columns = st.columns(5)
    columns[0].metric(
        "Trajectories / pixel",
        f"{design.trajectories:,}",
        help="Angle samples x velocity samples, summed over tiers.",
    )
    columns[1].metric(
        "vs production 64x64",
        f"{design.cost_vs_production:.2f}x",
        help="GPU search time scales about linearly with this ratio; post-processing does not.",
    )
    if design.tiers:
        fans = " / ".join(reflex._fmt(t.angle_half_width_deg, 0) for t in design.tiers)
        speeds = " / ".join(
            f"{reflex._fmt(t.velocity_min, 0)}-{reflex._fmt(t.velocity_max, 0)}" for t in design.tiers
        )
        columns[2].metric(
            "Fan half-width (deg)",
            fans,
            help="Symmetric offsets from the ecliptic direction, one value per tier (slow / fast).",
        )
        columns[3].metric("Velocity range (px/d)", speeds, help="One range per tier (slow / fast).")
    elif design.box is not None:
        box = design.box
        columns[2].metric("vx range (px/d)", f"{reflex._fmt(box.vx_min, 0)} to {reflex._fmt(box.vx_max, 0)}")
        columns[3].metric("vy range (px/d)", f"{reflex._fmt(box.vy_min, 0)} to {reflex._fmt(box.vy_max, 0)}")
    settings = {line.key: line.value for line in recommendation.settings}
    columns[4].metric(
        "results_per_pixel",
        settings["results_per_pixel"],
        help="64 for windows longer than 3.5 d, 128 for shorter ones.",
    )


def _render_region_metrics(st: object, recommendation: reflex.Recommendation) -> None:
    region = recommendation.region
    coh = recommendation.coherence
    spec = recommendation.spec
    st.subheader("Occupied region")
    retro = " (negative: retrograde-looking)" if region.retrograde_looking else ""
    coherent_note = (
        ""
        if coh.coverage_is_coherent
        else f" **The {reflex._fmt(coh.span_days, 1)} d span exceeds this.**"
    )
    st.markdown(
        "\n".join(
            [
                f"- **v_along** (along the ecliptic): {reflex._fmt(region.v_along_min, 1)} to "
                f"{reflex._fmt(region.v_along_max, 1)} px/d{retro}",
                f"- **|v_perp|** (across the ecliptic): up to {reflex._fmt(region.v_perp_max, 1)} px/d, "
                "set by omega(nearest distance) x sin(inclination ceiling); independent of the node",
                f"- **speed**: {reflex._fmt(region.speed_min, 1)} to {reflex._fmt(region.speed_max, 1)} px/d",
                f"- **widest angle from the ecliptic**: {reflex._fmt(region.angle_max_deg, 1)} deg, "
                "the half-width a single fan needs (set by the slowest covered objects)",
                f"- **longest coherent window at the coverage edges**: {reflex._fmt(coh.max_span_at_edges_days, 1)} d "
                f"({reflex._fmt(coh.max_span_near_edge_days, 1)} d at {reflex._fmt(spec.distance_min_au, 1)} au, "
                f"{reflex._fmt(coh.max_span_far_edge_days, 1)} d at {reflex._fmt(spec.distance_max_au, 1)} au; "
                f"elongation {reflex._fmt(spec.worst_elongation_deg, 0)} deg, tolerance "
                f"{reflex._fmt(coh.tolerance_px, 2)} px).{coherent_note}",
                f"- **distances coherent at {reflex._fmt(coh.span_days, 1)} d from this node**: "
                f"{reflex._fmt(coh.coherent_distance_min_au, 1)} to "
                f"{reflex._fmt_or_inf(coh.coherent_distance_max_au, 1)} au",
            ]
        )
    )


def _render_design_table(st: object, recommendation: reflex.Recommendation) -> None:
    st.subheader("Grid designs")
    rows = []
    for design in recommendation.designs:
        if design.box is not None:
            box = design.box
            grid = (
                f"vx [{reflex._fmt(box.vx_min, 1)}, {reflex._fmt(box.vx_max, 1)}] x {box.vx_samples}; "
                f"vy [{reflex._fmt(box.vy_min, 1)}, {reflex._fmt(box.vy_max, 1)}] x {box.vy_samples}"
            )
            steps = f"{reflex._fmt(box.step, 3)} px/d both axes"
        else:
            grid = "; ".join(
                f"[{reflex._fmt(t.velocity_min, 1)}, {reflex._fmt(t.velocity_max, 1)}] x {t.velocity_samples}, "
                f"+/-{reflex._fmt(t.angle_half_width_deg, 1)} deg x {t.angle_samples}"
                for t in design.tiers
            )
            steps = "; ".join(
                f"{reflex._fmt(t.velocity_step, 3)} px/d, {reflex._fmt(t.angle_step_deg, 3)} deg" for t in design.tiers
            )
        rows.append(
            {
                "design": ("* " if design.kind == recommendation.recommended_kind else "") + design.title,
                "trajectories/pixel": design.trajectories,
                "x production": round(design.cost_vs_production, 2),
                "grid": grid,
                "steps": steps,
            }
        )
    _dataframe(st, rows)
    for design in recommendation.designs:
        with st.expander(f"Notes: {design.title}", expanded=False):
            for note in design.notes:
                st.write(f"- {note}")


def _render_configuration_section(st: object, recommendation: reflex.Recommendation) -> None:
    st.subheader("kbmod configuration")
    choices: list[tuple[str, reflex.GridDesign, int]] = []
    for design in recommendation.designs:
        count = len(design.generator_configs())
        for index in range(count):
            label = design.title if count == 1 else f"{design.title}: {design.tiers[index].label}"
            if design.kind == recommendation.recommended_kind:
                label += " (recommended)"
            choices.append((label, design, index))
    default_index = next(i for i, (_, d, _) in enumerate(choices) if d.kind == recommendation.recommended_kind)
    selected = st.selectbox(
        "Grid to write",
        range(len(choices)),
        index=default_index,
        format_func=lambda i: choices[i][0],
    )
    _, design, index = choices[selected]
    yaml_text = reflex.search_config_yaml(recommendation, design, index)
    stem = reflex.design_file_stem(recommendation, design, index)
    columns = st.columns(2)
    columns[0].download_button(
        "Download this YAML",
        data=yaml_text,
        file_name=f"{stem}.yaml",
        mime="application/x-yaml",
    )
    figures = [
        ("velocity_space.png", reflex.build_velocity_space_plot(recommendation)),
        ("coherence.png", reflex.build_coherence_plot(recommendation)),
    ]
    try:
        payload = reflex.build_recommendation_zip(recommendation, figures)
    finally:
        import matplotlib.pyplot as plt

        for _, fig in figures:
            plt.close(fig)
    columns[1].download_button(
        "Download everything (all YAMLs, report, plots)",
        data=payload,
        file_name=(
            f"kbmod_recommendation_{reflex._token(recommendation.spec.guess_distance_au)}au_"
            f"{reflex._token(recommendation.span_days)}d.zip"
        ),
        mime="application/zip",
    )
    st.code(yaml_text, language="yaml")
    st.caption(
        "H200 cost anchors from the 42 au recipe: a 3 d, rpp 64 box cell on the Option A grid made 4.5 M rows "
        "at 161-216 GB in 1-1.7 h; rpp 128 reached 255-280 GB; a full-patch 14 d Option A rpp 64 cell with 94 "
        "layers took 395 GB and about 10 h. Memory follows rows x layers."
    )


def _render_coherence_section(st: object, recommendation: reflex.Recommendation) -> None:
    coh = recommendation.coherence
    spec = recommendation.spec
    with st.expander("Coherence: how long a window this coverage allows", expanded=False):
        left, right = st.columns([3, 2], vertical_alignment="top")
        with left:
            fig = reflex.build_coherence_plot(recommendation)
            _show_plot(st, fig)
            import matplotlib.pyplot as plt

            plt.close(fig)
        with right:
            st.write(
                f"Longest coherent window (departure <= {reflex._fmt(coh.tolerance_px, 2)} px) at the coverage edges:"
            )
            table = [
                {
                    "elongation (deg)": reflex._fmt(e, 0),
                    f"{reflex._fmt(spec.distance_min_au, 1)} au": f"{reflex._fmt(t_near, 1)} d",
                    f"{reflex._fmt(spec.distance_max_au, 1)} au": f"{reflex._fmt(t_far, 1)} d",
                }
                for e, t_near, t_far in coh.span_table
            ]
            _dataframe(st, table)
            nodes = ", ".join(reflex._fmt(n, 1) for n in coh.ladder_nodes_au)
            st.write(
                f"At {reflex._fmt(coh.span_days, 1)} d and elongation {reflex._fmt(spec.worst_elongation_deg, 0)} deg one node "
                f"holds {reflex._fmt(coh.coherent_distance_min_au, 1)}-{reflex._fmt_or_inf(coh.coherent_distance_max_au, 1)} au; "
                f"covering {reflex._fmt(spec.distance_min_au, 1)}-{reflex._fmt(spec.distance_max_au, 1)} au needs "
                f"{len(coh.ladder_nodes_au)} node(s): {nodes} au."
            )
            if recommendation.blind_zones_au:
                zones = "; ".join(
                    f"{reflex._fmt(lo, 1)}-{reflex._fmt(hi, 1)} au" for lo, hi in recommendation.blind_zones_au
                )
                st.write(
                    f"The recommended velocity floor ({reflex._fmt(recommendation.recommended.velocity_floor, 1)} px/d) "
                    f"hides ecliptic objects at {zones} at opposition."
                )


def _render_context_section(st: object, recommendation: reflex.Recommendation) -> None:
    with st.expander("Measured context and model limits", expanded=False):
        st.write(
            "The model has no population prior. For scale, the 153 golden Sorcha injections on patch 471085 "
            "at the 42 au node fell inside a fixed fan as follows (nothing looked retrograde):"
        )
        _dataframe(
            st,
            [
                {"fan half-width (deg)": reflex._fmt(a, 0), "fraction of population": f"{frac:.1%}"}
                for a, frac in reflex.MEASURED_FAN_FRACTIONS_42AU
            ],
        )
        _dataframe(
            st,
            [
                {"|v_perp| ceiling (px/d)": reflex._fmt(v, 0), "fraction of population": f"{frac:.1%}"}
                for v, frac in reflex.MEASURED_VPERP_FRACTIONS_42AU
            ],
        )
        st.write(
            "Pool-stage settings and the fan evidence come from the E2 golden-patch cells (467847, 468387) and "
            "the 2026-10-01 arm cells at 3.01 d: fan +/-30 deg 56->84 and 75->100, +/-45 deg 56->83 and 75->92, "
            "rpp 4/8/16/64/128 = 39/53/77/131/168, CPU sigma-G 56->54 and 75->62. Window-span results from 471085: "
            "windows past ~8 d recovered nothing at the as-run step."
        )
        for warning in recommendation.warnings:
            if warning.startswith("Model limits"):
                st.info(warning)
        st.text_area("Full text report", value=reflex.format_recommendation_report(recommendation), height=500)


def _dataframe(st: object, rows: list[dict]) -> None:
    if _supports_width(st.dataframe):
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.dataframe(rows, hide_index=True, use_container_width=True)


def _supports_width(function: object) -> bool:
    try:
        return "width" in inspect.signature(function).parameters
    except (TypeError, ValueError):  # pragma: no cover - defensive against proxies.
        return False


# --------------------------------------------------------------------------- evaluate mode
def _run_evaluate_mode(st: object) -> None:
    st.title("KBMOD Rate/Angle Evaluator")
    st.caption(
        "Check an existing EclipticCenteredSearch grid: how far adjacent angles diverge over the "
        "window and how many angles the PSF would need. Angles are offsets from the per-field, "
        "WCS-derived ecliptic direction."
    )
    inputs = _render_controls(st)
    try:
        evaluation = _evaluate_inputs(inputs)
    except ValueError as exc:
        st.error(str(exc))
        return

    left, right = st.columns([2, 1], vertical_alignment="top")
    with left:
        fig = evaluator.build_search_plot(
            evaluation,
            show_tno_reference=inputs.show_tno_reference,
            show_trojan_reference=inputs.show_trojan_reference,
        )
        _show_plot(st, fig)
        import matplotlib.pyplot as plt

        plt.close(fig)

    with right:
        _render_metrics(st, evaluation)
        report_text = evaluator.format_report(
            evaluation,
            show_tno_reference=inputs.show_tno_reference,
            show_trojan_reference=inputs.show_trojan_reference,
        )
        st.download_button(
            "Export report",
            data=evaluator.build_export_zip(
                evaluation,
                show_tno_reference=inputs.show_tno_reference,
                show_trojan_reference=inputs.show_trojan_reference,
            ),
            file_name=f"{evaluator.default_plot_output_path(evaluation).stem}.zip",
            mime="application/zip",
        )
        st.text_area("Report", value=report_text, height=520)


def _render_controls(st: object) -> AppInputs:
    with st.sidebar:
        st.header("Inputs")
        timespan_days = st.number_input(
            "Days",
            min_value=0.000001,
            value=10.0,
            step=1.0,
            format="%.6f",
        )
        patch_width_arcmin = st.number_input(
            "Patch width (arcmin)",
            min_value=0.000001,
            value=evaluator.DEFAULT_PATCH_SIZE_ARCMIN[0],
            step=1.0,
            format="%.6f",
        )
        patch_height_arcmin = st.number_input(
            "Patch height (arcmin)",
            min_value=0.000001,
            value=evaluator.DEFAULT_PATCH_SIZE_ARCMIN[1],
            step=1.0,
            format="%.6f",
        )
        seeing_arcsec = st.number_input(
            "Seeing FWHM (arcsec)",
            min_value=0.01,
            value=evaluator.DEFAULT_SEEING_ARCSEC,
            step=0.01,
            format="%.2f",
        )

        st.header("Angles")
        st.caption(
            "KBMOD EclipticCenteredSearch angles are offsets from the "
            "per-field, WCS-derived ecliptic direction."
        )
        angle_min_deg = st.number_input(
            "Minimum offset from ecliptic (deg)",
            value=evaluator.DEFAULT_ANGLE_RANGE_DEG[0],
            step=5.0,
            format="%.6f",
        )
        angle_max_deg = st.number_input(
            "Maximum offset from ecliptic (deg)",
            value=evaluator.DEFAULT_ANGLE_RANGE_DEG[1],
            step=5.0,
            format="%.6f",
        )
        angle_samples = st.number_input(
            "Inclusive angle samples",
            min_value=1,
            value=evaluator.DEFAULT_ANGLE_SAMPLES,
            step=1,
        )

        st.header("Velocity")
        velocity_unit = st.selectbox(
            "Velocity unit",
            options=["pixels/day", "arcsec/day"],
            index=0,
        )
        if velocity_unit == "pixels/day":
            default_min = evaluator.DEFAULT_VELOCITY_RANGE_PIXELS_PER_DAY[0]
            default_max = evaluator.DEFAULT_VELOCITY_RANGE_PIXELS_PER_DAY[1]
        else:
            default_min = evaluator.pixels_to_arcsec(
                evaluator.DEFAULT_VELOCITY_RANGE_PIXELS_PER_DAY[0],
                evaluator.DEFAULT_PIXEL_SCALE_ARCSEC_PER_PIXEL,
            )
            default_max = evaluator.pixels_to_arcsec(
                evaluator.DEFAULT_VELOCITY_RANGE_PIXELS_PER_DAY[1],
                evaluator.DEFAULT_PIXEL_SCALE_ARCSEC_PER_PIXEL,
            )
        velocity_min = st.number_input(
            f"Minimum velocity ({velocity_unit})",
            min_value=0.0,
            value=float(default_min),
            step=1.0,
            format="%.6f",
        )
        velocity_max = st.number_input(
            f"Maximum velocity ({velocity_unit})",
            min_value=0.000001,
            value=float(default_max),
            step=1.0,
            format="%.6f",
        )
        velocity_samples = st.number_input(
            "Inclusive velocity samples",
            min_value=1,
            value=evaluator.DEFAULT_VELOCITY_SAMPLES,
            step=1,
        )

        st.header("References")
        show_tno_reference = st.toggle("TNO reference", value=True)
        show_trojan_reference = st.toggle("Trojan reference", value=True)

    return AppInputs(
        timespan_days=float(timespan_days),
        patch_width_arcmin=float(patch_width_arcmin),
        patch_height_arcmin=float(patch_height_arcmin),
        angle_min_deg=float(angle_min_deg),
        angle_max_deg=float(angle_max_deg),
        angle_samples=int(angle_samples),
        velocity_unit=str(velocity_unit),
        velocity_min=float(velocity_min),
        velocity_max=float(velocity_max),
        velocity_samples=int(velocity_samples),
        seeing_arcsec=float(seeing_arcsec),
        show_tno_reference=bool(show_tno_reference),
        show_trojan_reference=bool(show_trojan_reference),
    )


def _evaluate_inputs(inputs: AppInputs) -> evaluator.RateAngleEvaluation:
    velocity_kwargs: dict[str, tuple[float, float]]
    if inputs.velocity_unit == "arcsec/day":
        velocity_kwargs = {
            "velocity_range_arcsec_per_day": (inputs.velocity_min, inputs.velocity_max)
        }
    else:
        velocity_kwargs = {
            "velocity_range_pixels_per_day": (inputs.velocity_min, inputs.velocity_max)
        }

    return evaluator.evaluate_rate_angle_sampling(
        inputs.timespan_days,
        patch_size_arcmin=(inputs.patch_width_arcmin, inputs.patch_height_arcmin),
        angle_range_deg=(inputs.angle_min_deg, inputs.angle_max_deg),
        seeing_arcsec=inputs.seeing_arcsec,
        angle_samples=inputs.angle_samples,
        velocity_samples=inputs.velocity_samples,
        **velocity_kwargs,
    )


def _render_metrics(st: object, evaluation: evaluator.RateAngleEvaluation) -> None:
    st.subheader("Key Metrics")
    first_row = st.columns(3)
    first_row[0].metric(
        "Configured angles",
        evaluation.search_grid.angle_samples,
        help="Number of ecliptic-relative angle offsets in the configured range.",
    )
    first_row[1].metric(
        "PSF-needed angles",
        evaluation.primary.angle_count,
        help="Number of angles required to keep endpoint spacing within one PSF half-step.",
    )
    first_row[2].metric(
        "Velocity samples",
        evaluation.search_grid.velocity_samples,
        help="Number of velocity samples used between the configured minimum and maximum.",
    )

    second_row = st.columns(3)
    second_row[0].metric(
        "Angle-offset spacing",
        f"{evaluator._fmt(evaluation.search_grid.angle_spacing_deg, 4)} deg",
        help="Increment between consecutive offsets from the per-field ecliptic direction.",
    )
    second_row[1].metric(
        "PSF spacing",
        f"{evaluator._fmt(evaluation.primary.spacing_deg, 4)} deg",
        help="Maximum spacing that still keeps travel endpoints within one seeing diameter.",
    )
    second_row[2].metric(
        "Max travel",
        f"{evaluator._fmt(evaluation.max_travel_pixels, 3)} px",
        help="Greatest endpoint travel distance over the selected timespan at max configured velocity.",
    )

    third_row = st.columns(3)
    third_row[0].metric(
        "Velocity spacing",
        f"{evaluator._fmt(evaluation.velocity_divided_by_samples_pixels_per_day, 4)} px/day",
        help=(
            "Estimated velocity sample spacing using max velocity divided by the"
            " configured number of velocity samples."
        ),
    )
    third_row[1].metric(
        "Velocity spacing",
        f"{evaluator._fmt(evaluation.velocity_divided_by_samples_arcsec_per_day, 4)} arcsec/day",
        help="Same spacing as above converted to arcseconds per day.",
    )
    third_row[2].metric(
        "Configured max velocity",
        f"{evaluator._fmt(evaluation.velocity_max_pixels_per_day)} px/day",
        help="Largest velocity in this search grid.",
    )

    st.subheader("Diagnostics")
    st.write(f"Strict full-step angles: {evaluation.strict_full_step.angle_count}")
    st.write(f"Patch corner-to-corner angles: {evaluation.patch_diagonal.angle_count}")
    st.write(
        "Configured maximum adjacent endpoint separation: "
        f"{evaluator._fmt_sigfig(evaluation.adjacent_angle_endpoint_separation_pixels)} px "
        f"({evaluator._fmt_sigfig(evaluation.adjacent_angle_endpoint_separation_arcsec)} arcsec)"
    )
    st.write(
        "Velocity-based spacing: "
        f"{evaluator._fmt(evaluation.velocity_divided_by_samples_pixels_per_day)} px/day "
        f"({evaluator._fmt(evaluation.velocity_divided_by_samples_arcsec_per_day)} arcsec/day)"
    )
    st.write(
        "Right-edge spanned angle: "
        f"{evaluator._fmt_sigfig(evaluation.right_edge_adjacent_angle_separation_pixels)} px "
        f"({evaluator._fmt_sigfig(evaluation.right_edge_adjacent_angle_separation_arcsec)} arcsec)"
    )
    if not evaluation.max_motion_reaches_patch_diagonal:
        st.warning(
            "Max motion is shorter than the patch diagonal; use the patch "
            "corner-to-corner count for a literal boundary guarantee."
        )


def _show_plot(st: object, fig: object) -> None:
    pyplot_parameters = inspect.signature(st.pyplot).parameters
    if "width" in pyplot_parameters:
        st.pyplot(fig, clear_figure=True, width="stretch")
    else:
        st.pyplot(fig, clear_figure=True, use_container_width=True)


def main() -> int:
    """Console-script entry point."""

    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError as exc:  # pragma: no cover - exercised without optional extra.
        raise RuntimeError(
            "Install the interactive extra with: pip install -e '.[interactive]'"
        ) from exc

    sys.argv = [
        "streamlit",
        "run",
        str(Path(__file__).resolve()),
        "--browser.gatherUsageStats=false",
        *sys.argv[1:],
    ]
    result = streamlit_cli.main()
    return int(result or 0)


if __name__ == "__main__":
    run_app()
