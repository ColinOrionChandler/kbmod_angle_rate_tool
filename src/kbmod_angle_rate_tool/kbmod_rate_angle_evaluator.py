"""Estimate KBMOD angle sampling needed for moving-object searches."""

from __future__ import annotations

import argparse
import io
import math
import os
import sys
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_PIXEL_SCALE_ARCSEC_PER_PIXEL = 0.2
DEFAULT_SEEING_ARCSEC = 1.0
DEFAULT_PATCH_SIZE_ARCMIN = (20.0, 20.0)
DEFAULT_ANGLE_RANGE_DEG = (-90.0, 90.0)
DEFAULT_ANGLE_SAMPLES = 64
DEFAULT_VELOCITY_RANGE_PIXELS_PER_DAY = (25.0, 225.0)
DEFAULT_VELOCITY_SAMPLES = 64
DEFAULT_PLOT_OUTPUT: Path | None = None
DEFAULT_OUTPUT_STEM_PREFIX = "kbmod_rate_angle_evaluator"
TYPICAL_TNO_SSB_RATE_ARCSEC_PER_HOUR = 0.6
TYPICAL_TROJAN_RATE_ARCSEC_PER_HOUR = 13.0
MAX_TRAVEL_ADJACENT_TARGET_FRACTION = 0.5
TNO_ADJACENT_TARGET_FRACTION = 0.75
TROJAN_ADJACENT_TARGET_FRACTION = 0.65
ANGLE_REFERENCE_ECLIPTIC_OFFSET = "offset_from_ecliptic"
ECLIPTIC_ANGLE_SOURCE_GIVEN = "generator_config.given_ecliptic"
ECLIPTIC_ANGLE_SOURCE_RUNTIME_WCS = "runtime_work_unit_wcs"
ECLIPTIC_ANGLE_SOURCE_EXPLICIT = "explicit_ecliptic_angle"


@dataclass(frozen=True)
class SearchGridConfig:
    """Configured KBMOD ``EclipticCenteredSearch`` grid.

    ``angle_min_deg`` and ``angle_max_deg`` are offsets from the per-run
    ecliptic angle.  They are not absolute image-frame angles.
    """

    angle_min_deg: float
    angle_max_deg: float
    angle_samples: int
    velocity_min_pixels_per_day: float
    velocity_max_pixels_per_day: float
    velocity_samples: int
    angle_reference: str = ANGLE_REFERENCE_ECLIPTIC_OFFSET
    ecliptic_angle_deg: float | None = None
    ecliptic_angle_source: str = ECLIPTIC_ANGLE_SOURCE_RUNTIME_WCS

    @property
    def angle_offset_min_deg(self) -> float:
        """Minimum configured offset from the ecliptic angle."""

        return self.angle_min_deg

    @property
    def angle_offset_max_deg(self) -> float:
        """Maximum configured offset from the ecliptic angle."""

        return self.angle_max_deg

    @property
    def angle_image_min_deg(self) -> float | None:
        """Minimum absolute image angle, when the ecliptic angle is known."""

        if self.ecliptic_angle_deg is None:
            return None
        return self.ecliptic_angle_deg + self.angle_min_deg

    @property
    def angle_image_max_deg(self) -> float | None:
        """Maximum absolute image angle, when the ecliptic angle is known."""

        if self.ecliptic_angle_deg is None:
            return None
        return self.ecliptic_angle_deg + self.angle_max_deg

    @property
    def angle_width_deg(self) -> float:
        return self.angle_max_deg - self.angle_min_deg

    @property
    def angle_spacing_deg(self) -> float:
        return _sample_spacing(self.angle_min_deg, self.angle_max_deg, self.angle_samples)

    @property
    def velocity_spacing_pixels_per_day(self) -> float:
        return _sample_spacing(
            self.velocity_min_pixels_per_day,
            self.velocity_max_pixels_per_day,
            self.velocity_samples,
        )


DEFAULT_SEARCH_GRID = SearchGridConfig(
    angle_min_deg=DEFAULT_ANGLE_RANGE_DEG[0],
    angle_max_deg=DEFAULT_ANGLE_RANGE_DEG[1],
    angle_samples=DEFAULT_ANGLE_SAMPLES,
    velocity_min_pixels_per_day=DEFAULT_VELOCITY_RANGE_PIXELS_PER_DAY[0],
    velocity_max_pixels_per_day=DEFAULT_VELOCITY_RANGE_PIXELS_PER_DAY[1],
    velocity_samples=DEFAULT_VELOCITY_SAMPLES,
)


@dataclass(frozen=True)
class AngleCoverage:
    """Angle-grid summary for one distance tolerance calculation."""

    distance_arcsec: float
    distance_pixels: float
    spacing_deg: float
    interval_count: int
    angle_count: int


@dataclass(frozen=True)
class RateAngleEvaluation:
    """Complete rate/angle sampling evaluation."""

    timespan_days: float
    pixel_scale_arcsec_per_pixel: float
    seeing_arcsec: float
    seeing_pixels: float
    patch_width_arcmin: float
    patch_height_arcmin: float
    patch_width_pixels: float
    patch_height_pixels: float
    patch_diagonal_arcsec: float
    patch_diagonal_pixels: float
    search_grid: SearchGridConfig
    velocity_min_arcsec_per_day: float
    velocity_max_arcsec_per_day: float
    velocity_min_pixels_per_day: float
    velocity_max_pixels_per_day: float
    angle_min_deg: float
    angle_max_deg: float
    angle_width_deg: float
    primary: AngleCoverage
    patch_diagonal: AngleCoverage
    strict_full_step: AngleCoverage

    @property
    def max_motion_reaches_patch_diagonal(self) -> bool:
        return self.primary.distance_arcsec >= self.patch_diagonal_arcsec

    @property
    def min_travel_pixels(self) -> float:
        return self.velocity_min_pixels_per_day * self.timespan_days

    @property
    def max_travel_pixels(self) -> float:
        return self.velocity_max_pixels_per_day * self.timespan_days

    @property
    def velocity_divided_by_samples_pixels_per_day(self) -> float:
        return self.search_grid.velocity_max_pixels_per_day / float(
            self.search_grid.velocity_samples
        )

    @property
    def velocity_divided_by_samples_arcsec_per_day(self) -> float:
        return pixels_to_arcsec(
            self.velocity_divided_by_samples_pixels_per_day,
            self.pixel_scale_arcsec_per_pixel,
        )

    @property
    def typical_tno_ssb_velocity_arcsec_per_day(self) -> float:
        return TYPICAL_TNO_SSB_RATE_ARCSEC_PER_HOUR * 24.0

    @property
    def typical_tno_ssb_velocity_pixels_per_day(self) -> float:
        return arcsec_to_pixels(
            self.typical_tno_ssb_velocity_arcsec_per_day,
            self.pixel_scale_arcsec_per_pixel,
        )

    @property
    def typical_tno_ssb_travel_arcsec(self) -> float:
        return self.typical_tno_ssb_velocity_arcsec_per_day * self.timespan_days

    @property
    def typical_tno_ssb_travel_pixels(self) -> float:
        return arcsec_to_pixels(
            self.typical_tno_ssb_travel_arcsec,
            self.pixel_scale_arcsec_per_pixel,
        )

    @property
    def typical_trojan_velocity_arcsec_per_day(self) -> float:
        return TYPICAL_TROJAN_RATE_ARCSEC_PER_HOUR * 24.0

    @property
    def typical_trojan_velocity_pixels_per_day(self) -> float:
        return arcsec_to_pixels(
            self.typical_trojan_velocity_arcsec_per_day,
            self.pixel_scale_arcsec_per_pixel,
        )

    @property
    def typical_trojan_travel_arcsec(self) -> float:
        return self.typical_trojan_velocity_arcsec_per_day * self.timespan_days

    @property
    def typical_trojan_travel_pixels(self) -> float:
        return arcsec_to_pixels(
            self.typical_trojan_travel_arcsec,
            self.pixel_scale_arcsec_per_pixel,
        )

    @property
    def adjacent_angle_endpoint_separation_arcsec(self) -> float:
        return pixels_to_arcsec(
            self.adjacent_angle_endpoint_separation_pixels,
            self.pixel_scale_arcsec_per_pixel,
        )

    @property
    def adjacent_angle_endpoint_separation_pixels(self) -> float:
        return adjacent_endpoint_separation_for_distance_pixels(
            self,
            self.max_travel_pixels,
            target_fraction=MAX_TRAVEL_ADJACENT_TARGET_FRACTION,
            clip_to_patch=True,
        )

    @property
    def typical_tno_adjacent_endpoint_separation_arcsec(self) -> float:
        return pixels_to_arcsec(
            self.typical_tno_adjacent_endpoint_separation_pixels,
            self.pixel_scale_arcsec_per_pixel,
        )

    @property
    def typical_tno_adjacent_endpoint_separation_pixels(self) -> float:
        return adjacent_endpoint_separation_for_distance_pixels(
            self,
            self.typical_tno_ssb_travel_pixels,
            target_fraction=TNO_ADJACENT_TARGET_FRACTION,
            clip_to_patch=True,
        )

    @property
    def typical_trojan_adjacent_endpoint_separation_arcsec(self) -> float:
        return pixels_to_arcsec(
            self.typical_trojan_adjacent_endpoint_separation_pixels,
            self.pixel_scale_arcsec_per_pixel,
        )

    @property
    def typical_trojan_adjacent_endpoint_separation_pixels(self) -> float:
        return adjacent_endpoint_separation_for_distance_pixels(
            self,
            self.typical_trojan_travel_pixels,
            target_fraction=TROJAN_ADJACENT_TARGET_FRACTION,
            clip_to_patch=True,
        )

    @property
    def right_edge_adjacent_angle_separation_arcsec(self) -> float:
        return pixels_to_arcsec(
            self.right_edge_adjacent_angle_separation_pixels,
            self.pixel_scale_arcsec_per_pixel,
        )

    @property
    def right_edge_adjacent_angle_separation_pixels(self) -> float:
        angles = _inclusive_samples(
            self.angle_min_deg,
            self.angle_max_deg,
            self.search_grid.angle_samples,
        )
        endpoints = _right_edge_adjacent_endpoints(
            angles,
            (0.0, self.patch_height_pixels / 2.0),
            patch_width_pixels=self.patch_width_pixels,
            patch_height_pixels=self.patch_height_pixels,
        )
        if endpoints is None:
            return 0.0
        (_, x0, y0), (_, x1, y1) = endpoints
        return math.hypot(x1 - x0, y1 - y0)


def arcsec_to_pixels(value_arcsec: float, pixel_scale_arcsec_per_pixel: float) -> float:
    """Convert arcseconds to pixels."""

    return value_arcsec / pixel_scale_arcsec_per_pixel


def pixels_to_arcsec(value_pixels: float, pixel_scale_arcsec_per_pixel: float) -> float:
    """Convert pixels to arcseconds."""

    return value_pixels * pixel_scale_arcsec_per_pixel


def chord_separation(distance_arcsec: float, angle_spacing_deg: float) -> float:
    """Return the endpoint separation for two same-length rays."""

    _require_positive("distance_arcsec", distance_arcsec)
    if angle_spacing_deg < 0.0:
        raise ValueError("angle_spacing_deg must be non-negative")
    return 2.0 * distance_arcsec * math.sin(math.radians(angle_spacing_deg) / 2.0)


def adjacent_endpoint_separation_for_distance_pixels(
    evaluation: RateAngleEvaluation,
    distance_pixels: float,
    *,
    target_fraction: float,
    clip_to_patch: bool,
) -> float:
    """Return separation between adjacent ray endpoints, optionally clipped to the patch."""

    angles = _inclusive_samples(
        evaluation.angle_min_deg,
        evaluation.angle_max_deg,
        evaluation.search_grid.angle_samples,
    )
    endpoints = _adjacent_endpoints_at_distance(
        angles,
        (0.0, evaluation.patch_height_pixels / 2.0),
        distance_pixels,
        target_fraction=target_fraction,
        patch_width_pixels=evaluation.patch_width_pixels,
        patch_height_pixels=evaluation.patch_height_pixels,
        clip_to_patch=clip_to_patch,
    )
    if endpoints is None:
        return 0.0
    (_, x0, y0), (_, x1, y1) = endpoints
    return math.hypot(x1 - x0, y1 - y0)


def calculate_angle_coverage(
    distance_arcsec: float,
    seeing_arcsec: float,
    pixel_scale_arcsec_per_pixel: float,
    angle_range_deg: tuple[float, float],
    *,
    nearest_grid_half_step: bool = True,
) -> AngleCoverage:
    """Calculate angle spacing and inclusive sample count for a travel distance.

    With ``nearest_grid_half_step=True``, every true angle is assumed to be
    matched to the nearest searched grid angle, so the allowed full spacing is
    twice the angular offset that produces one PSF FWHM of transverse miss.
    """

    _require_positive("distance_arcsec", distance_arcsec)
    _require_positive("seeing_arcsec", seeing_arcsec)
    _require_positive("pixel_scale_arcsec_per_pixel", pixel_scale_arcsec_per_pixel)
    angle_min_deg, angle_max_deg = _validate_ordered_pair(
        "angle_range_deg",
        angle_range_deg,
        strict=True,
    )

    ratio = min(1.0, seeing_arcsec / distance_arcsec)
    spacing_rad = math.asin(ratio)
    if nearest_grid_half_step:
        spacing_rad *= 2.0

    angle_width_rad = math.radians(angle_max_deg - angle_min_deg)
    interval_count = math.ceil(angle_width_rad / spacing_rad)

    return AngleCoverage(
        distance_arcsec=distance_arcsec,
        distance_pixels=arcsec_to_pixels(distance_arcsec, pixel_scale_arcsec_per_pixel),
        spacing_deg=math.degrees(spacing_rad),
        interval_count=interval_count,
        angle_count=interval_count + 1,
    )


def evaluate_rate_angle_sampling(
    timespan_days: float,
    *,
    pixel_scale_arcsec_per_pixel: float = DEFAULT_PIXEL_SCALE_ARCSEC_PER_PIXEL,
    seeing_arcsec: float = DEFAULT_SEEING_ARCSEC,
    patch_size_arcmin: tuple[float, float] = DEFAULT_PATCH_SIZE_ARCMIN,
    search_grid: SearchGridConfig | None = None,
    velocity_range_arcsec_per_day: tuple[float, float] | None = None,
    velocity_range_pixels_per_day: tuple[float, float] | None = None,
    angle_range_deg: tuple[float, float] | None = None,
    angle_samples: int | None = None,
    velocity_samples: int | None = None,
) -> RateAngleEvaluation:
    """Evaluate KBMOD angle sampling for the requested search geometry."""

    _require_positive("timespan_days", timespan_days)
    _require_positive("pixel_scale_arcsec_per_pixel", pixel_scale_arcsec_per_pixel)
    _require_positive("seeing_arcsec", seeing_arcsec)

    if velocity_range_arcsec_per_day is not None and velocity_range_pixels_per_day is not None:
        raise ValueError("velocity ranges may be supplied in arcsec/day or pixels/day, not both")

    grid = search_grid or DEFAULT_SEARCH_GRID
    grid = apply_search_grid_overrides(
        grid,
        pixel_scale_arcsec_per_pixel=pixel_scale_arcsec_per_pixel,
        velocity_range_arcsec_per_day=velocity_range_arcsec_per_day,
        velocity_range_pixels_per_day=velocity_range_pixels_per_day,
        angle_range_deg=angle_range_deg,
        angle_samples=angle_samples,
        velocity_samples=velocity_samples,
    )

    patch_width_arcmin, patch_height_arcmin = _validate_ordered_values(
        "patch_size_arcmin",
        patch_size_arcmin,
        positive=True,
    )
    angle_range = (grid.angle_min_deg, grid.angle_max_deg)

    velocity_min_arcsec_per_day = pixels_to_arcsec(
        grid.velocity_min_pixels_per_day,
        pixel_scale_arcsec_per_pixel,
    )
    velocity_max_arcsec_per_day = pixels_to_arcsec(
        grid.velocity_max_pixels_per_day,
        pixel_scale_arcsec_per_pixel,
    )
    max_motion_arcsec = velocity_max_arcsec_per_day * timespan_days
    patch_width_pixels = arcsec_to_pixels(
        patch_width_arcmin * 60.0,
        pixel_scale_arcsec_per_pixel,
    )
    patch_height_pixels = arcsec_to_pixels(
        patch_height_arcmin * 60.0,
        pixel_scale_arcsec_per_pixel,
    )
    patch_diagonal_arcsec = math.hypot(
        patch_width_arcmin * 60.0,
        patch_height_arcmin * 60.0,
    )

    primary = calculate_angle_coverage(
        max_motion_arcsec,
        seeing_arcsec,
        pixel_scale_arcsec_per_pixel,
        angle_range,
        nearest_grid_half_step=True,
    )
    patch_diagonal = calculate_angle_coverage(
        patch_diagonal_arcsec,
        seeing_arcsec,
        pixel_scale_arcsec_per_pixel,
        angle_range,
        nearest_grid_half_step=True,
    )
    strict_full_step = calculate_angle_coverage(
        max_motion_arcsec,
        seeing_arcsec,
        pixel_scale_arcsec_per_pixel,
        angle_range,
        nearest_grid_half_step=False,
    )

    return RateAngleEvaluation(
        timespan_days=timespan_days,
        pixel_scale_arcsec_per_pixel=pixel_scale_arcsec_per_pixel,
        seeing_arcsec=seeing_arcsec,
        seeing_pixels=arcsec_to_pixels(seeing_arcsec, pixel_scale_arcsec_per_pixel),
        patch_width_arcmin=patch_width_arcmin,
        patch_height_arcmin=patch_height_arcmin,
        patch_width_pixels=patch_width_pixels,
        patch_height_pixels=patch_height_pixels,
        patch_diagonal_arcsec=patch_diagonal_arcsec,
        patch_diagonal_pixels=arcsec_to_pixels(
            patch_diagonal_arcsec,
            pixel_scale_arcsec_per_pixel,
        ),
        search_grid=grid,
        velocity_min_arcsec_per_day=velocity_min_arcsec_per_day,
        velocity_max_arcsec_per_day=velocity_max_arcsec_per_day,
        velocity_min_pixels_per_day=grid.velocity_min_pixels_per_day,
        velocity_max_pixels_per_day=grid.velocity_max_pixels_per_day,
        angle_min_deg=grid.angle_min_deg,
        angle_max_deg=grid.angle_max_deg,
        angle_width_deg=grid.angle_width_deg,
        primary=primary,
        patch_diagonal=patch_diagonal,
        strict_full_step=strict_full_step,
    )


def apply_search_grid_overrides(
    search_grid: SearchGridConfig,
    *,
    pixel_scale_arcsec_per_pixel: float,
    velocity_range_arcsec_per_day: tuple[float, float] | None = None,
    velocity_range_pixels_per_day: tuple[float, float] | None = None,
    angle_range_deg: tuple[float, float] | None = None,
    angle_samples: int | None = None,
    velocity_samples: int | None = None,
) -> SearchGridConfig:
    """Apply explicit overrides to a search grid."""

    _require_positive("pixel_scale_arcsec_per_pixel", pixel_scale_arcsec_per_pixel)
    if velocity_range_arcsec_per_day is not None and velocity_range_pixels_per_day is not None:
        raise ValueError("velocity ranges may be supplied in arcsec/day or pixels/day, not both")

    grid = search_grid
    if angle_range_deg is not None:
        angle_min, angle_max = _validate_ordered_pair(
            "angle_range_deg",
            angle_range_deg,
            strict=True,
        )
        grid = replace(grid, angle_min_deg=angle_min, angle_max_deg=angle_max)
    if angle_samples is not None:
        grid = replace(grid, angle_samples=_coerce_positive_int("angle_samples", angle_samples))

    if velocity_range_pixels_per_day is not None:
        velocity_min, velocity_max = _validate_velocity_range_pixels_per_day(
            velocity_range_pixels_per_day
        )
        grid = replace(
            grid,
            velocity_min_pixels_per_day=velocity_min,
            velocity_max_pixels_per_day=velocity_max,
        )
    if velocity_range_arcsec_per_day is not None:
        velocity_min_arcsec, velocity_max_arcsec = _validate_velocity_range_arcsec_per_day(
            velocity_range_arcsec_per_day
        )
        grid = replace(
            grid,
            velocity_min_pixels_per_day=arcsec_to_pixels(
                velocity_min_arcsec,
                pixel_scale_arcsec_per_pixel,
            ),
            velocity_max_pixels_per_day=arcsec_to_pixels(
                velocity_max_arcsec,
                pixel_scale_arcsec_per_pixel,
            ),
        )
    if velocity_samples is not None:
        grid = replace(
            grid,
            velocity_samples=_coerce_positive_int("velocity_samples", velocity_samples),
        )
    return grid


def load_search_grid_from_kbmod_yaml(
    path: Path,
    *,
    ecliptic_angle_deg: float | None = None,
) -> SearchGridConfig:
    """Load a KBMOD ``EclipticCenteredSearch`` grid from a YAML config.

    The configured ``angles`` bounds are retained as offsets from the run's
    ecliptic angle.  If ``given_ecliptic`` is present, its value is recorded in
    degrees so absolute image-angle bounds are also available.  Otherwise KBMOD
    derives the ecliptic angle from each WorkUnit's WCS at runtime, so a YAML
    file alone cannot determine the absolute bounds.  ``ecliptic_angle_deg`` may
    be supplied when that runtime-derived value is known.
    """

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, Mapping):
        raise ValueError(f"{path} must contain a YAML mapping")
    generator_config = config.get("generator_config")
    if not isinstance(generator_config, Mapping):
        raise ValueError(f"{path} does not contain a generator_config mapping")

    generator_name = generator_config.get("name")
    if generator_name != "EclipticCenteredSearch":
        raise ValueError(
            "Only EclipticCenteredSearch generator_config values are supported; "
            f"got {generator_name!r}"
        )

    angles = _validate_three_value_list("generator_config.angles", generator_config.get("angles"))
    velocities = _validate_three_value_list(
        "generator_config.velocities",
        generator_config.get("velocities"),
    )

    angle_units = generator_config.get("angle_units", "degree")
    angle_min, angle_max = _convert_angle_range_to_degrees(
        angles[0],
        angles[1],
        angle_units,
        normalize=True,
    )
    given_ecliptic = generator_config.get("given_ecliptic")
    if given_ecliptic is not None:
        resolved_ecliptic_angle_deg = _convert_angle_to_degrees(
            given_ecliptic,
            angle_units,
        )
        if ecliptic_angle_deg is not None:
            explicit_ecliptic_angle_deg = _require_finite_float(
                "ecliptic_angle_deg",
                ecliptic_angle_deg,
            )
            if not math.isclose(
                explicit_ecliptic_angle_deg,
                resolved_ecliptic_angle_deg,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "ecliptic_angle_deg conflicts with "
                    "generator_config.given_ecliptic"
                )
        ecliptic_angle_source = ECLIPTIC_ANGLE_SOURCE_GIVEN
    elif ecliptic_angle_deg is not None:
        resolved_ecliptic_angle_deg = _require_finite_float(
            "ecliptic_angle_deg",
            ecliptic_angle_deg,
        )
        ecliptic_angle_source = ECLIPTIC_ANGLE_SOURCE_EXPLICIT
    else:
        resolved_ecliptic_angle_deg = None
        ecliptic_angle_source = ECLIPTIC_ANGLE_SOURCE_RUNTIME_WCS
    velocity_min, velocity_max = _convert_velocity_range_to_pixels_per_day(
        velocities[0],
        velocities[1],
        generator_config.get("velocity_units", "pix / d"),
    )

    return SearchGridConfig(
        angle_min_deg=angle_min,
        angle_max_deg=angle_max,
        angle_samples=_coerce_positive_int("generator_config.angles[2]", angles[2]),
        velocity_min_pixels_per_day=velocity_min,
        velocity_max_pixels_per_day=velocity_max,
        velocity_samples=_coerce_positive_int("generator_config.velocities[2]", velocities[2]),
        ecliptic_angle_deg=resolved_ecliptic_angle_deg,
        ecliptic_angle_source=ecliptic_angle_source,
    )


def resolve_search_grid_from_args(args: argparse.Namespace) -> SearchGridConfig:
    """Resolve built-in, YAML, and explicit CLI search-grid settings."""

    grid = DEFAULT_SEARCH_GRID
    if args.kbmod_config_yaml is not None:
        grid = load_search_grid_from_kbmod_yaml(
            args.kbmod_config_yaml,
            ecliptic_angle_deg=args.ecliptic_angle_deg,
        )
    elif args.ecliptic_angle_deg is not None:
        grid = replace(
            grid,
            ecliptic_angle_deg=_require_finite_float(
                "ecliptic_angle_deg",
                args.ecliptic_angle_deg,
            ),
            ecliptic_angle_source=ECLIPTIC_ANGLE_SOURCE_EXPLICIT,
        )
    return apply_search_grid_overrides(
        grid,
        pixel_scale_arcsec_per_pixel=args.pixel_scale,
        velocity_range_arcsec_per_day=(
            tuple(args.velocity_range_arcsec_per_day)
            if args.velocity_range_arcsec_per_day is not None
            else None
        ),
        velocity_range_pixels_per_day=(
            tuple(args.velocity_range_pixels_per_day)
            if args.velocity_range_pixels_per_day is not None
            else None
        ),
        angle_range_deg=tuple(args.angle_range_deg) if args.angle_range_deg is not None else None,
        angle_samples=args.angle_samples,
        velocity_samples=args.velocity_samples,
    )


def format_report(
    evaluation: RateAngleEvaluation,
    *,
    plot_output_paths: tuple[Path, ...] | None = None,
    report_output_path: Path | None = None,
    show_tno_reference: bool = True,
    show_trojan_reference: bool = True,
) -> str:
    """Format a human-readable report."""

    grid = evaluation.search_grid
    lines = [
        "KBMOD rate/angle evaluation",
        "",
        "Inputs",
        f"  timespan: {_fmt(evaluation.timespan_days)} days",
        (
            "  pixel scale: "
            f"{_fmt(evaluation.pixel_scale_arcsec_per_pixel)} arcsec/pixel"
        ),
        (
            "  seeing FWHM: "
            f"{_fmt(evaluation.seeing_arcsec)} arcsec "
            f"({_fmt(evaluation.seeing_pixels)} px)"
        ),
        (
            "  patch size: "
            f"{_fmt(evaluation.patch_width_arcmin)} x "
            f"{_fmt(evaluation.patch_height_arcmin)} arcmin "
            f"({_fmt(evaluation.patch_width_pixels)} x "
            f"{_fmt(evaluation.patch_height_pixels)} px)"
        ),
        (
            "  configured velocity range: "
            f"{_fmt(evaluation.velocity_min_pixels_per_day)} to "
            f"{_fmt(evaluation.velocity_max_pixels_per_day)} px/day "
            f"({_fmt(evaluation.velocity_min_arcsec_per_day)} to "
            f"{_fmt(evaluation.velocity_max_arcsec_per_day)} arcsec/day)"
        ),
        f"  configured velocity samples: {grid.velocity_samples}",
        (
            "  configured angle-offset range (from ecliptic): "
            f"{_fmt(evaluation.angle_min_deg)} to "
            f"{_fmt(evaluation.angle_max_deg)} deg "
            f"(width {_fmt(evaluation.angle_width_deg)} deg)"
        ),
        f"  configured angle samples: {grid.angle_samples}",
        *_format_ecliptic_angle_report_lines(grid),
        "",
        "Configured grid",
        f"  chosen angles: {grid.angle_samples}",
        f"  chosen angle spacing: {_fmt(grid.angle_spacing_deg, 5)} deg",
        (
            "  chosen velocity spacing: "
            f"{_fmt(grid.velocity_spacing_pixels_per_day, 5)} px/day"
        ),
        (
            "  velocity-based spacing: "
            f"{_fmt(evaluation.velocity_divided_by_samples_pixels_per_day, 5)} px/day "
            f"({_fmt(evaluation.velocity_divided_by_samples_arcsec_per_day)} arcsec/day)"
        ),
        (
            "  min travel over timespan: "
            f"{_fmt(evaluation.min_travel_pixels)} px "
            f"({_fmt(pixels_to_arcsec(evaluation.min_travel_pixels, evaluation.pixel_scale_arcsec_per_pixel))} arcsec)"
        ),
        (
            "  max travel over timespan: "
            f"{_fmt(evaluation.max_travel_pixels)} px "
            f"({_fmt(evaluation.primary.distance_arcsec)} arcsec)"
        ),
    ]
    if show_tno_reference:
        lines.extend(
            _format_reference_report_section(
                "Typical TNO SSB",
                rate_arcsec_per_hour=TYPICAL_TNO_SSB_RATE_ARCSEC_PER_HOUR,
                velocity_pixels_per_day=evaluation.typical_tno_ssb_velocity_pixels_per_day,
                travel_pixels=evaluation.typical_tno_ssb_travel_pixels,
                travel_arcsec=evaluation.typical_tno_ssb_travel_arcsec,
                adjacent_separation_pixels=(
                    evaluation.typical_tno_adjacent_endpoint_separation_pixels
                ),
                adjacent_separation_arcsec=(
                    evaluation.typical_tno_adjacent_endpoint_separation_arcsec
                ),
                evaluation=evaluation,
            )
        )
    if show_trojan_reference:
        lines.extend(
            _format_reference_report_section(
                "Typical Trojan",
                rate_arcsec_per_hour=TYPICAL_TROJAN_RATE_ARCSEC_PER_HOUR,
                velocity_pixels_per_day=evaluation.typical_trojan_velocity_pixels_per_day,
                travel_pixels=evaluation.typical_trojan_travel_pixels,
                travel_arcsec=evaluation.typical_trojan_travel_arcsec,
                adjacent_separation_pixels=(
                    evaluation.typical_trojan_adjacent_endpoint_separation_pixels
                ),
                adjacent_separation_arcsec=(
                    evaluation.typical_trojan_adjacent_endpoint_separation_arcsec
                ),
                evaluation=evaluation,
            )
        )
    lines.extend(
        [
            "",
            "Configured maximum travel adjacent endpoints",
            (
                "  separation: "
                f"{_fmt_sigfig(evaluation.adjacent_angle_endpoint_separation_pixels)} px "
                f"({_fmt_sigfig(evaluation.adjacent_angle_endpoint_separation_arcsec)} arcsec)"
            ),
            (
                "  right-edge spanned angle: "
                f"{_fmt_sigfig(evaluation.right_edge_adjacent_angle_separation_pixels)} px "
                f"({_fmt_sigfig(evaluation.right_edge_adjacent_angle_separation_arcsec)} arcsec)"
            ),
            "",
            "PSF-needed angle coverage",
            "  convention: nearest searched angle within half a grid step",
            "  transverse miss limit: <= 1 seeing FWHM",
            f"  allowed full spacing: {_fmt(evaluation.primary.spacing_deg, 5)} deg",
            f"  intervals: {evaluation.primary.interval_count}",
            f"  inclusive angles: {evaluation.primary.angle_count}",
            "",
            "Diagnostics",
            (
                "  strict full-step/no-half-step spacing: "
                f"{_fmt(evaluation.strict_full_step.spacing_deg, 5)} deg"
            ),
            (
                "  strict full-step/no-half-step angles: "
                f"{evaluation.strict_full_step.angle_count}"
            ),
            (
                "  patch diagonal: "
                f"{_fmt(evaluation.patch_diagonal_arcsec, 2)} arcsec "
                f"({_fmt(evaluation.patch_diagonal_pixels, 2)} px)"
            ),
            (
                "  patch corner-to-corner spacing: "
                f"{_fmt(evaluation.patch_diagonal.spacing_deg, 5)} deg"
            ),
            (
                "  patch corner-to-corner angles: "
                f"{evaluation.patch_diagonal.angle_count}"
            ),
        ]
    )

    if plot_output_paths:
        lines.extend(["", "Plots: wrote"])
        lines.extend(f"  {plot_output_path}" for plot_output_path in plot_output_paths)
    if report_output_path is not None:
        lines.extend(["", f"Report: wrote {report_output_path}"])

    if not evaluation.max_motion_reaches_patch_diagonal:
        lines.extend(
            [
                "",
                (
                    "Warning: max motion is shorter than the patch diagonal; "
                    "use the patch corner-to-corner count for a literal "
                    "boundary guarantee."
                ),
            ]
        )

    return "\n".join(lines)


def _format_ecliptic_angle_report_lines(grid: SearchGridConfig) -> list[str]:
    """Describe how configured angle offsets map to image-frame angles."""

    lines = [
        "  angle reference: offsets from KBMOD's per-field ecliptic direction",
    ]
    if grid.ecliptic_angle_deg is None:
        lines.extend(
            [
                (
                    "  ecliptic image angle: derived by KBMOD at runtime from "
                    "the WorkUnit WCS; unavailable from this YAML/configuration"
                ),
                "  absolute image-angle range: unavailable without the runtime ecliptic angle",
            ]
        )
    else:
        lines.extend(
            [
                (
                    "  ecliptic image angle: "
                    f"{_fmt(grid.ecliptic_angle_deg)} deg "
                    f"(source: {grid.ecliptic_angle_source})"
                ),
                (
                    "  absolute image-angle range: "
                    f"{_fmt(grid.angle_image_min_deg)} to "
                    f"{_fmt(grid.angle_image_max_deg)} deg"
                ),
            ]
        )
    return lines


def _format_reference_report_section(
    label: str,
    *,
    rate_arcsec_per_hour: float,
    velocity_pixels_per_day: float,
    travel_pixels: float,
    travel_arcsec: float,
    adjacent_separation_pixels: float,
    adjacent_separation_arcsec: float,
    evaluation: RateAngleEvaluation,
) -> list[str]:
    min_rate_arcsec_per_hour = evaluation.velocity_min_arcsec_per_day / 24.0
    max_rate_arcsec_per_hour = evaluation.velocity_max_arcsec_per_day / 24.0
    min_travel_arcsec = pixels_to_arcsec(
        evaluation.min_travel_pixels,
        evaluation.pixel_scale_arcsec_per_pixel,
    )
    max_travel_arcsec = pixels_to_arcsec(
        evaluation.max_travel_pixels,
        evaluation.pixel_scale_arcsec_per_pixel,
    )
    reference_to_max = rate_arcsec_per_hour / max_rate_arcsec_per_hour
    return [
        "",
        f"{label} reference",
        (
            "  rate: "
            f"{_fmt(rate_arcsec_per_hour)} arcsec/hour "
            f"({_fmt(velocity_pixels_per_day)} px/day)"
        ),
        (
            "  travel over timespan: "
            f"{_fmt(travel_pixels)} px ({_fmt(travel_arcsec)} arcsec)"
        ),
        (
            "  adjacent chosen-angle endpoint separation at reference travel: "
            f"{_fmt_sigfig(adjacent_separation_pixels)} px "
            f"({_fmt_sigfig(adjacent_separation_arcsec)} arcsec)"
        ),
        (
            "  configured search minimum: "
            f"{_fmt(evaluation.velocity_min_pixels_per_day)} px/day "
            f"({_fmt(min_rate_arcsec_per_hour)} arcsec/hour), "
            f"{_fmt(evaluation.min_travel_pixels)} px "
            f"({_fmt(min_travel_arcsec)} arcsec) over timespan"
        ),
        (
            "  configured search maximum: "
            f"{_fmt(evaluation.velocity_max_pixels_per_day)} px/day "
            f"({_fmt(max_rate_arcsec_per_hour)} arcsec/hour), "
            f"{_fmt(evaluation.max_travel_pixels)} px "
            f"({_fmt(max_travel_arcsec)} arcsec) over timespan"
        ),
        (
            "  reference compared to configured maximum: "
            f"{_fmt(reference_to_max, 5)}x"
        ),
    ]


def build_search_plot(
    evaluation: RateAngleEvaluation,
    *,
    show_tno_reference: bool = True,
    show_trojan_reference: bool = True,
) -> Any:
    """Build a scaled plot of the patch and configured search-angle fan."""

    _ensure_matplotlib_config_dir()
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(9.5, 7.6))
    origin = (0.0, evaluation.patch_height_pixels / 2.0)
    max_length = evaluation.max_travel_pixels
    min_length = evaluation.min_travel_pixels
    tno_length = evaluation.typical_tno_ssb_travel_pixels
    trojan_length = evaluation.typical_trojan_travel_pixels
    visible_reference_length = max(min_length, max_length, tno_length, 1.0)
    trojan_is_off_scale = trojan_length > visible_reference_length
    trojan_plot_length = visible_reference_length if trojan_is_off_scale else trojan_length

    patch = Rectangle(
        (0.0, 0.0),
        evaluation.patch_width_pixels,
        evaluation.patch_height_pixels,
        fill=False,
        edgecolor="black",
        linewidth=1.5,
        label="_nolegend_",
    )
    ax.add_patch(patch)
    ax.scatter([origin[0]], [origin[1]], color="black", s=25, zorder=5)

    xs = [0.0, evaluation.patch_width_pixels, origin[0]]
    ys = [0.0, evaluation.patch_height_pixels, origin[1]]
    angles = _inclusive_samples(
        evaluation.angle_min_deg,
        evaluation.angle_max_deg,
        evaluation.search_grid.angle_samples,
    )
    for angle_deg in angles:
        angle_rad = math.radians(angle_deg)
        end_x = origin[0] + max_length * math.cos(angle_rad)
        end_y = origin[1] + max_length * math.sin(angle_rad)
        xs.append(end_x)
        ys.append(end_y)
        ax.plot(
            [origin[0], end_x],
            [origin[1], end_y],
            color="#3b6fb6",
            alpha=0.18,
            linewidth=1.0,
        )

    span = max(
        evaluation.patch_width_pixels,
        evaluation.patch_height_pixels,
        max_length,
        1.0,
    )
    pad = span * 0.08

    adjacent_endpoints = _adjacent_endpoints_at_distance(
        angles,
        origin,
        max_length,
        target_fraction=MAX_TRAVEL_ADJACENT_TARGET_FRACTION,
        patch_width_pixels=evaluation.patch_width_pixels,
        patch_height_pixels=evaluation.patch_height_pixels,
        clip_to_patch=True,
    )
    if adjacent_endpoints is not None:
        (_, x0, y0), (_, x1, y1) = adjacent_endpoints
        ax.plot([x0, x1], [y0, y1], color="#4c4c4c", linestyle="--", linewidth=1.4)
        ax.text(
            (x0 + x1) / 2.0,
            (y0 + y1) / 2.0,
            (
                " adjacent ends "
                f'{_fmt_sigfig(evaluation.adjacent_angle_endpoint_separation_arcsec)}"'
            ),
            va="bottom",
            ha="left",
            color="#4c4c4c",
        )

    if show_tno_reference:
        tno_adjacent_endpoints = _adjacent_endpoints_at_distance(
            angles,
            origin,
            tno_length,
            target_fraction=TNO_ADJACENT_TARGET_FRACTION,
            patch_width_pixels=evaluation.patch_width_pixels,
            patch_height_pixels=evaluation.patch_height_pixels,
            clip_to_patch=True,
        )
        if tno_adjacent_endpoints is not None:
            (_, x0, y0), (_, x1, y1) = tno_adjacent_endpoints
            _plot_adjacent_endpoint_callout(
                ax,
                (x0, y0),
                (x1, y1),
                label=(
                    "TNO adjacent ends\n"
                    f'{_fmt_sigfig(evaluation.typical_tno_adjacent_endpoint_separation_arcsec)}"'
                ),
                color="#4c4c4c",
                text_offset=(pad * 0.8, pad * 0.8),
            )
            xs.extend([x0, x1])
            ys.extend([y0, y1])

    if show_trojan_reference:
        trojan_adjacent_endpoints = _adjacent_endpoints_at_distance(
            angles,
            origin,
            trojan_length,
            target_fraction=TROJAN_ADJACENT_TARGET_FRACTION,
            patch_width_pixels=evaluation.patch_width_pixels,
            patch_height_pixels=evaluation.patch_height_pixels,
            clip_to_patch=True,
        )
        if trojan_adjacent_endpoints is not None:
            (_, x0, y0), (_, x1, y1) = trojan_adjacent_endpoints
            _plot_adjacent_endpoint_callout(
                ax,
                (x0, y0),
                (x1, y1),
                label=(
                    "Trojan adjacent ends\n"
                    f'{_fmt_sigfig(evaluation.typical_trojan_adjacent_endpoint_separation_arcsec)}"'
                ),
                color="#984ea3",
                text_offset=(pad * 0.35, -pad * 2.1),
            )
            xs.extend([x0, x1])
            ys.extend([y0, y1])

    right_edge_endpoints = _right_edge_adjacent_endpoints(
        angles,
        origin,
        patch_width_pixels=evaluation.patch_width_pixels,
        patch_height_pixels=evaluation.patch_height_pixels,
    )
    if right_edge_endpoints is not None:
        (_, x0, y0), (_, x1, y1) = right_edge_endpoints
        ax.plot(
            [origin[0], x0],
            [origin[1], y0],
            color="#2c7fb8",
            alpha=0.55,
            linestyle=":",
            linewidth=1.5,
        )
        ax.plot(
            [origin[0], x1],
            [origin[1], y1],
            color="#2c7fb8",
            alpha=0.55,
            linestyle=":",
            linewidth=1.5,
        )
        _plot_adjacent_endpoint_callout(
            ax,
            (x0, y0),
            (x1, y1),
            label=(
                "edge-spanned\n"
                f'{_fmt_sigfig(evaluation.right_edge_adjacent_angle_separation_arcsec)}"'
            ),
            color="#2c7fb8",
            text_offset=(-pad * 1.8, -pad * 1.0),
        )
        xs.extend([x0, x1])
        ys.extend([y0, y1])

    bar_y_max = min(ys) - pad
    bar_y_min = bar_y_max - pad * 0.45
    bar_y_tno = bar_y_min - pad * 0.45
    bar_y_trojan = bar_y_tno - pad * 0.45
    ax.plot(
        [0.0, min_length],
        [bar_y_min, bar_y_min],
        color="#d95f02",
        linewidth=4,
        label=f"configured min: {_fmt(min_length)} px",
    )
    ax.plot(
        [0.0, max_length],
        [bar_y_max, bar_y_max],
        color="#1b9e77",
        linewidth=4,
        label=f"configured max: {_fmt(max_length)} px",
    )
    if show_tno_reference:
        ax.plot(
            [0.0, tno_length],
            [bar_y_tno, bar_y_tno],
            color="#4c4c4c",
            linewidth=4,
            label=(
                f'TNO SSB {_fmt(TYPICAL_TNO_SSB_RATE_ARCSEC_PER_HOUR)}"/hour: '
                f"{_fmt(tno_length)} px"
            ),
        )
        xs.append(tno_length)
        ys.append(bar_y_tno)
    if show_trojan_reference:
        trojan_label = (
            f'Trojan {_fmt(TYPICAL_TROJAN_RATE_ARCSEC_PER_HOUR)}"/hour: '
            f"{_fmt(trojan_length)} px"
        )
        if trojan_is_off_scale:
            trojan_label += " (off scale)"
        ax.plot(
            [0.0, trojan_plot_length],
            [bar_y_trojan, bar_y_trojan],
            color="#984ea3",
            linewidth=4,
            label=trojan_label,
        )
        if trojan_is_off_scale:
            ax.plot(
                [trojan_plot_length],
                [bar_y_trojan],
                marker=">",
                color="#984ea3",
                markersize=8,
                clip_on=False,
            )
        xs.append(trojan_plot_length)
        ys.append(bar_y_trojan)
    xs.extend([0.0, min_length, max_length])
    ys.extend([bar_y_min, bar_y_max])

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x pixels (diagnostic frame; ecliptic direction = +x)")
    ax.set_ylabel("y pixels (diagnostic frame)")
    ax.set_title("KBMOD configured angle-offset fan (relative to ecliptic)")
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.grid(True, alpha=0.25)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        title="Travel references",
        framealpha=0.92,
        fontsize=9,
        title_fontsize=9,
    )
    ax.text(
        1.06,
        1.0,
        _format_plot_parameter_box(evaluation),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#808080", "alpha": 0.92},
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.74, 1.0))
    return fig


def write_search_plot(
    evaluation: RateAngleEvaluation,
    output_path: Path,
    *,
    show_tno_reference: bool = True,
    show_trojan_reference: bool = True,
) -> tuple[Path, ...]:
    """Write a scaled plot of the patch and configured search-angle fan."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_paths = _plot_output_paths(output_path)
    fig = build_search_plot(
        evaluation,
        show_tno_reference=show_tno_reference,
        show_trojan_reference=show_trojan_reference,
    )
    import matplotlib.pyplot as plt

    for plot_output_path in output_paths:
        fig.savefig(plot_output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_paths


def build_export_zip(
    evaluation: RateAngleEvaluation,
    *,
    show_tno_reference: bool = True,
    show_trojan_reference: bool = True,
    stem: str | None = None,
) -> bytes:
    """Build an in-memory ZIP containing PNG, PDF, and text report artifacts."""

    artifact_stem = stem or default_plot_output_path(evaluation).stem
    report_text = format_report(
        evaluation,
        show_tno_reference=show_tno_reference,
        show_trojan_reference=show_trojan_reference,
    )
    fig = build_search_plot(
        evaluation,
        show_tno_reference=show_tno_reference,
        show_trojan_reference=show_trojan_reference,
    )
    import matplotlib.pyplot as plt

    try:
        png_buffer = io.BytesIO()
        pdf_buffer = io.BytesIO()
        fig.savefig(png_buffer, format="png", dpi=160, bbox_inches="tight")
        fig.savefig(pdf_buffer, format="pdf", bbox_inches="tight")
    finally:
        plt.close(fig)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{artifact_stem}.png", png_buffer.getvalue())
        archive.writestr(f"{artifact_stem}.pdf", pdf_buffer.getvalue())
        archive.writestr(f"{artifact_stem}_report.txt", report_text)
    return zip_buffer.getvalue()


def _plot_output_paths(output_path: Path) -> tuple[Path, ...]:
    suffix = output_path.suffix.lower()
    if suffix == ".png":
        return output_path, output_path.with_suffix(".pdf")
    if suffix == ".pdf":
        return output_path, output_path.with_suffix(".png")
    return output_path.with_suffix(".png"), output_path.with_suffix(".pdf")


def report_output_path(plot_output_path: Path) -> Path:
    """Return the path where the text report should be written."""

    return plot_output_path.with_name(f"{plot_output_path.stem}_report.txt")


def default_plot_output_path(evaluation: RateAngleEvaluation) -> Path:
    """Return a configuration-descriptive default plot path."""

    grid = evaluation.search_grid
    stem = (
        f"{DEFAULT_OUTPUT_STEM_PREFIX}_"
        f"{_filename_number_token(evaluation.timespan_days)}_days_"
        f"{_filename_number_token(evaluation.patch_width_arcmin)}X"
        f"{_filename_number_token(evaluation.patch_height_arcmin)}arcmin_"
        f"{_filename_number_token(grid.angle_min_deg)}_to_"
        f"{_filename_number_token(grid.angle_max_deg)}_eclipticOffsetDeg_"
        f"{grid.angle_samples}_angs_"
        f"{_filename_number_token(grid.velocity_min_pixels_per_day)}_to_"
        f"{_filename_number_token(grid.velocity_max_pixels_per_day)}_pixPerDay_"
        f"{grid.velocity_samples}rateIntervals"
    )
    return Path(f"{stem}.png")


def write_report_file(report_text: str, output_path: Path) -> Path:
    """Write a report text file and return its path."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Estimate KBMOD angle samples needed for rate/angle searches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "timespan_days",
        type=_positive_float,
        help="Search timespan in days.",
    )
    parser.add_argument(
        "--pixel-scale",
        type=_positive_float,
        default=DEFAULT_PIXEL_SCALE_ARCSEC_PER_PIXEL,
        help="Pixel scale in arcsec/pixel.",
    )
    parser.add_argument(
        "--seeing",
        type=_positive_float,
        default=DEFAULT_SEEING_ARCSEC,
        help="Seeing PSF FWHM in arcsec.",
    )
    parser.add_argument(
        "--patch-size-arcmin",
        nargs=2,
        type=_positive_float,
        default=DEFAULT_PATCH_SIZE_ARCMIN,
        metavar=("WIDTH", "HEIGHT"),
        help="Patch size in arcminutes.",
    )
    parser.add_argument(
        "--kbmod-config-yaml",
        type=Path,
        default=None,
        help="Optional KBMOD YAML config with generator_config search-grid values.",
    )
    parser.add_argument(
        "--ecliptic-angle-deg",
        type=float,
        default=None,
        help=(
            "Actual per-field ecliptic angle in image-frame degrees. "
            "Use this to resolve absolute image-angle bounds when KBMOD derived "
            "the angle from a WorkUnit WCS at runtime."
        ),
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=DEFAULT_PLOT_OUTPUT,
        help=(
            "PNG or PDF path for the scaled search-grid plot. "
            "A matching companion PDF or PNG is also written."
        ),
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not write the scaled search-grid plot.",
    )
    parser.add_argument(
        "--no-report-file",
        action="store_true",
        help="Do not write the text report to a _report.txt file.",
    )
    parser.add_argument(
        "--velocity-range-arcsec-per-day",
        nargs=2,
        type=_nonnegative_float,
        default=None,
        metavar=("MIN", "MAX"),
        help="Explicit object velocity range in arcsec/day.",
    )
    parser.add_argument(
        "--velocity-range-pixels-per-day",
        nargs=2,
        type=_nonnegative_float,
        default=None,
        metavar=("MIN", "MAX"),
        help="Explicit object velocity range in pixels/day.",
    )
    parser.add_argument(
        "--velocity-samples",
        type=_positive_int,
        default=None,
        help="Number of inclusive velocity samples in the configured grid.",
    )
    parser.add_argument(
        "--angle-offset-range-deg",
        "--angle-range-deg",
        dest="angle_range_deg",
        nargs=2,
        type=float,
        default=None,
        metavar=("MIN", "MAX"),
        help="Explicit searched angle-offset range from the ecliptic, in degrees.",
    )
    parser.add_argument(
        "--angle-samples",
        type=_positive_int,
        default=None,
        help="Number of inclusive angle samples in the configured grid.",
    )

    args = parser.parse_args(argv)
    try:
        _validate_cli_args(args)
        args.search_grid = resolve_search_grid_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def run(argv: list[str] | None = None) -> RateAngleEvaluation:
    """Run the evaluator, print its report, and optionally write a plot."""

    args = parse_args(argv)
    evaluation = evaluate_rate_angle_sampling(
        args.timespan_days,
        pixel_scale_arcsec_per_pixel=args.pixel_scale,
        seeing_arcsec=args.seeing,
        patch_size_arcmin=tuple(args.patch_size_arcmin),
        search_grid=args.search_grid,
    )
    plot_output_paths = None
    resolved_plot_output = args.plot_output or default_plot_output_path(evaluation)
    report_output = report_output_path(resolved_plot_output)
    if not args.no_plot:
        plot_output_paths = write_search_plot(evaluation, resolved_plot_output)
    report_text = format_report(
        evaluation,
        plot_output_paths=plot_output_paths,
        report_output_path=None if args.no_report_file else report_output,
    )
    if not args.no_report_file:
        write_report_file(report_text, report_output)
    print(report_text)
    return evaluation


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point."""

    try:
        run(argv)
    except Exception as exc:
        print(f"kbmod-rate-angle-evaluator: error: {exc}", file=sys.stderr)
        return 1
    return 0


def _validate_cli_args(args: argparse.Namespace) -> None:
    if args.velocity_range_arcsec_per_day is not None and args.velocity_range_pixels_per_day is not None:
        raise ValueError(
            "Use either --velocity-range-arcsec-per-day or "
            "--velocity-range-pixels-per-day, not both"
        )


def _convert_angle_range_to_degrees(
    lower: Any,
    upper: Any,
    units: Any,
    *,
    normalize: bool,
) -> tuple[float, float]:
    lower_float = float(lower)
    upper_float = float(upper)
    units_normalized = str(units).strip().lower()
    if units_normalized in {"degree", "degrees", "deg"}:
        values = (lower_float, upper_float)
    elif units_normalized in {"radian", "radians", "rad"}:
        values = (math.degrees(lower_float), math.degrees(upper_float))
    else:
        raise ValueError(f"Unsupported angle_units value {units!r}")

    if normalize:
        return (min(values), max(values))
    return _validate_ordered_pair("angle_range_deg", values, strict=True)


def _convert_angle_to_degrees(value: Any, units: Any) -> float:
    """Convert one finite angular value to degrees."""

    converted, _ = _convert_angle_range_to_degrees(
        value,
        value,
        units,
        normalize=True,
    )
    return _require_finite_float("angle", converted)


def _convert_velocity_range_to_pixels_per_day(
    lower: Any,
    upper: Any,
    units: Any,
) -> tuple[float, float]:
    units_normalized = str(units).strip().lower().replace(" ", "")
    if units_normalized not in {
        "pix/d",
        "pix/day",
        "pixel/d",
        "pixel/day",
        "pixels/d",
        "pixels/day",
    }:
        raise ValueError(f"Unsupported velocity_units value {units!r}")
    return _validate_velocity_range_pixels_per_day((float(lower), float(upper)))


def _validate_velocity_range_arcsec_per_day(values: tuple[float, float]) -> tuple[float, float]:
    lower, upper = _validate_ordered_pair(
        "velocity_range_arcsec_per_day",
        values,
        strict=False,
    )
    if lower < 0.0:
        raise ValueError("velocity_range_arcsec_per_day lower bound must be non-negative")
    _require_positive("velocity_range_arcsec_per_day upper bound", upper)
    return lower, upper


def _validate_velocity_range_pixels_per_day(values: tuple[float, float]) -> tuple[float, float]:
    lower, upper = _validate_ordered_pair(
        "velocity_range_pixels_per_day",
        values,
        strict=False,
    )
    if lower < 0.0:
        raise ValueError("velocity_range_pixels_per_day lower bound must be non-negative")
    _require_positive("velocity_range_pixels_per_day upper bound", upper)
    return lower, upper


def _validate_three_value_list(name: str, value: Any) -> tuple[Any, Any, Any]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ValueError(f"{name} must be a length-3 list")
    return value[0], value[1], value[2]


def _inclusive_samples(start: float, stop: float, count: int) -> list[float]:
    count = _coerce_positive_int("count", count)
    if count == 1:
        return [start]
    step = (stop - start) / float(count - 1)
    return [start + i * step for i in range(count)]


def _adjacent_endpoints_at_distance(
    angles: list[float],
    origin: tuple[float, float],
    distance_pixels: float,
    *,
    target_fraction: float,
    patch_width_pixels: float | None = None,
    patch_height_pixels: float | None = None,
    clip_to_patch: bool = False,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    if len(angles) < 2:
        return None
    if clip_to_patch and (patch_width_pixels is None or patch_height_pixels is None):
        raise ValueError("patch dimensions are required when clip_to_patch=True")
    target_angle = angles[0] + (angles[-1] - angles[0]) * target_fraction
    first_index = min(
        range(len(angles) - 1),
        key=lambda index: abs((angles[index] + angles[index + 1]) / 2.0 - target_angle),
    )
    endpoints = []
    for angle_deg in (angles[first_index], angles[first_index + 1]):
        end_x, end_y = _endpoint_at_distance(
            angle_deg,
            origin,
            distance_pixels,
            patch_width_pixels=patch_width_pixels,
            patch_height_pixels=patch_height_pixels,
            clip_to_patch=clip_to_patch,
        )
        endpoints.append((angle_deg, end_x, end_y))
    return endpoints[0], endpoints[1]


def _right_edge_adjacent_endpoints(
    angles: list[float],
    origin: tuple[float, float],
    *,
    patch_width_pixels: float,
    patch_height_pixels: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    if len(angles) < 2:
        return None
    valid_pairs: list[tuple[float, int]] = []
    for index in range(len(angles) - 1):
        first = _right_edge_endpoint(angles[index], origin, patch_width_pixels)
        second = _right_edge_endpoint(angles[index + 1], origin, patch_width_pixels)
        if first is None or second is None:
            continue
        _, y0 = first
        _, y1 = second
        if 0.0 <= y0 <= patch_height_pixels and 0.0 <= y1 <= patch_height_pixels:
            midpoint_angle = (angles[index] + angles[index + 1]) / 2.0
            valid_pairs.append((abs(midpoint_angle), index))
    if not valid_pairs:
        return None
    _, first_index = min(valid_pairs)
    endpoints = []
    for angle_deg in (angles[first_index], angles[first_index + 1]):
        endpoint = _right_edge_endpoint(angle_deg, origin, patch_width_pixels)
        if endpoint is None:
            return None
        x, y = endpoint
        endpoints.append((angle_deg, x, y))
    return endpoints[0], endpoints[1]


def _right_edge_endpoint(
    angle_deg: float,
    origin: tuple[float, float],
    patch_width_pixels: float,
) -> tuple[float, float] | None:
    angle_rad = math.radians(angle_deg)
    dx = math.cos(angle_rad)
    if dx <= 0.0:
        return None
    distance_pixels = (patch_width_pixels - origin[0]) / dx
    return (
        patch_width_pixels,
        origin[1] + distance_pixels * math.sin(angle_rad),
    )


def _endpoint_at_distance(
    angle_deg: float,
    origin: tuple[float, float],
    distance_pixels: float,
    *,
    patch_width_pixels: float | None,
    patch_height_pixels: float | None,
    clip_to_patch: bool,
) -> tuple[float, float]:
    angle_rad = math.radians(angle_deg)
    endpoint_distance = distance_pixels
    if clip_to_patch:
        if patch_width_pixels is None or patch_height_pixels is None:
            raise ValueError("patch dimensions are required when clip_to_patch=True")
        endpoint_distance = min(
            distance_pixels,
            _ray_patch_limit_pixels(
                angle_rad,
                origin,
                patch_width_pixels=patch_width_pixels,
                patch_height_pixels=patch_height_pixels,
            ),
        )
    return (
        origin[0] + endpoint_distance * math.cos(angle_rad),
        origin[1] + endpoint_distance * math.sin(angle_rad),
    )


def _ray_patch_limit_pixels(
    angle_rad: float,
    origin: tuple[float, float],
    *,
    patch_width_pixels: float,
    patch_height_pixels: float,
) -> float:
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)
    candidates: list[float] = []
    if dx > 0.0:
        candidates.append((patch_width_pixels - origin[0]) / dx)
    elif dx < 0.0:
        candidates.append((0.0 - origin[0]) / dx)
    if dy > 0.0:
        candidates.append((patch_height_pixels - origin[1]) / dy)
    elif dy < 0.0:
        candidates.append((0.0 - origin[1]) / dy)
    positive_candidates = [value for value in candidates if value >= 0.0 and math.isfinite(value)]
    if not positive_candidates:
        return 0.0
    return min(positive_candidates)


def _plot_adjacent_endpoint_callout(
    ax: Any,
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    *,
    label: str,
    color: str,
    text_offset: tuple[float, float],
) -> None:
    ax.plot(
        [point_a[0], point_b[0]],
        [point_a[1], point_b[1]],
        color=color,
        linestyle="--",
        linewidth=1.4,
    )
    mid_x = (point_a[0] + point_b[0]) / 2.0
    mid_y = (point_a[1] + point_b[1]) / 2.0
    ax.annotate(
        label,
        xy=(mid_x, mid_y),
        xytext=(mid_x + text_offset[0], mid_y + text_offset[1]),
        arrowprops={"arrowstyle": "->", "color": color, "linewidth": 1.0},
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": color, "alpha": 0.85},
        color=color,
        fontsize=9,
    )


def _format_plot_parameter_box(evaluation: RateAngleEvaluation) -> str:
    grid = evaluation.search_grid
    lines = [
        "Chosen Parameters",
        f"days: {evaluation.timespan_days:.2f}",
        (
            "angle offsets from ecliptic: "
            f"{grid.angle_min_deg:.2f} to {grid.angle_max_deg:.2f} deg"
        ),
    ]
    if grid.ecliptic_angle_deg is None:
        lines.append("absolute image angles: unavailable (runtime WCS needed)")
    else:
        lines.extend(
            [
                f"ecliptic image angle: {grid.ecliptic_angle_deg:.2f} deg",
                (
                    "absolute image angles: "
                    f"{grid.angle_image_min_deg:.2f} to "
                    f"{grid.angle_image_max_deg:.2f} deg"
                ),
            ]
        )
    lines.extend(
        [
            f"angles: {grid.angle_samples}",
            (
                "rate range: "
                f"{grid.velocity_min_pixels_per_day:.2f} to "
                f"{grid.velocity_max_pixels_per_day:.2f} px/day"
            ),
            f"rate intervals: {grid.velocity_samples}",
            f"seeing: {evaluation.seeing_arcsec:.2f} arcsec",
        ]
    )
    return "\n".join(lines)


def _sample_spacing(start: float, stop: float, count: int) -> float:
    count = _coerce_positive_int("count", count)
    if count == 1:
        return 0.0
    return (stop - start) / float(count - 1)


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected a number, got {value!r}") from exc
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError(f"Expected a positive value, got {value!r}")
    return parsed


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected a number, got {value!r}") from exc
    if parsed < 0.0:
        raise argparse.ArgumentTypeError(f"Expected a non-negative value, got {value!r}")
    return parsed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected an integer, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value!r}")
    return parsed


def _coerce_positive_int(name: str, value: Any) -> int:
    try:
        parsed_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not math.isfinite(parsed_float) or parsed_float <= 0.0 or not parsed_float.is_integer():
        raise ValueError(f"{name} must be a positive integer")
    return int(parsed_float)


def _require_positive(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


def _require_finite_float(name: str, value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number")
    return parsed


def _validate_ordered_pair(
    name: str,
    values: tuple[float, float] | list[float],
    *,
    strict: bool,
) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    lower = float(values[0])
    upper = float(values[1])
    if strict and lower >= upper:
        raise ValueError(f"{name} lower bound must be less than upper bound")
    if not strict and lower > upper:
        raise ValueError(f"{name} lower bound must not exceed upper bound")
    return lower, upper


def _validate_ordered_values(
    name: str,
    values: tuple[float, float] | list[float],
    *,
    positive: bool,
) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    first = float(values[0])
    second = float(values[1])
    if positive:
        _require_positive(f"{name} first value", first)
        _require_positive(f"{name} second value", second)
    return first, second


def _fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def _fmt_sigfig(value: float, sigfigs: int = 3) -> str:
    return f"{value:.{sigfigs}g}"


def _filename_number_token(value: float) -> str:
    formatted = _fmt(value)
    return formatted.replace(".", "p")


def _ensure_matplotlib_config_dir() -> None:
    if "MPLCONFIGDIR" in os.environ:
        return
    config_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "kbmod_angle_rate_tool_matplotlib"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(config_dir)


if __name__ == "__main__":
    raise SystemExit(main())
