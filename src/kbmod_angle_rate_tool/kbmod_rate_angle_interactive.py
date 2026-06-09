"""Streamlit app for the KBMOD rate/angle evaluator."""

from __future__ import annotations

import inspect
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from kbmod_angle_rate_tool import kbmod_rate_angle_evaluator as evaluator


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


def run_app() -> None:
    """Launch the Streamlit app."""

    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - exercised without optional extra.
        raise RuntimeError(
            "Install the interactive extra with: pip install -e '.[interactive]'"
        ) from exc

    st.set_page_config(page_title="KBMOD Rate/Angle Evaluator", layout="wide")
    st.title("KBMOD Rate/Angle Evaluator")

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
        angle_min_deg = st.number_input(
            "Minimum angle (deg)",
            value=evaluator.DEFAULT_ANGLE_RANGE_DEG[0],
            step=5.0,
            format="%.6f",
        )
        angle_max_deg = st.number_input(
            "Maximum angle (deg)",
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
            default_min = evaluator.DEFAULT_VELOCITY_RANGE_PIXELS_PER_DAY[0]
            default_max = evaluator.DEFAULT_VELOCITY_RANGE_PIXELS_PER_DAY[1]
            default_min = evaluator.pixels_to_arcsec(
                default_min,
                evaluator.DEFAULT_PIXEL_SCALE_ARCSEC_PER_PIXEL,
            )
            default_max = evaluator.pixels_to_arcsec(
                default_max,
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
        help="Number of angles searched in the configured angular range.",
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
        "Angle spacing",
        f"{evaluator._fmt(evaluation.search_grid.angle_spacing_deg, 4)} deg",
        help="Angular increment between consecutive tested angles in the configured grid.",
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
