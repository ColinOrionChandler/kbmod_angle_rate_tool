"""Reflex-frame search space and KBMOD search-configuration recommender.

The model here is the analytic reflex-frame model of the 2026-09-18 KBMOD
search-strategy audit (section 5), which was validated against kbmod's own
reflex-corrected ``ra_42.0``/``dec_42.0`` tracks of the 153 Sorcha injections on
Rubin patch 471085 and against the six real TNOs there.  It is deliberately free
of astropy so it can run anywhere the Streamlit app runs.

Conventions
-----------
* ``d`` is a barycentric distance in au; ``d_guess`` is the distance the
  WorkUnit was reflex-corrected to (``helio_guess_dists``).
* ``elongation`` is the solar elongation of the field in degrees, 90 (quadrature)
  to 180 (opposition).  A field observed ``n`` days from opposition sits at
  roughly ``180 - n`` degrees.
* Velocities are pixels per day in the reflex-corrected frame.  ``v_along`` is
  along the ecliptic, positive eastward (prograde-looking, what a 0 degree
  ``EclipticCenteredSearch`` offset means); ``v_perp`` is across the ecliptic.
* Angles are offsets from the ecliptic direction, as ``EclipticCenteredSearch``
  defines them.  kbmod's pixel frame may be parity-flipped relative to the sky,
  which flips the sign of an angle but never its magnitude, so every fan here is
  symmetric about the ecliptic.

Model
-----
::

    omega(d)        = k d^-1.5 * RAD2PX                      orbital angular rate, px/d
    v_along(d)      = omega(d) cos(i) + C1 |cos e| (1/d_guess - 1/d)
    v_perp(d)       = +/- omega(d) sin(i)                    independent of d_guess
    departure(T)    = C1 nE |sin e| |1/d - 1/d_guess| T^2 / 12
                    + C1 nE^2 |cos e| |1/d - 1/d_guess| T^3 / 120
    grid steps      : dv <= 2 tol / T ;  dtheta <= 2 tol / (v T)

with ``k`` the Gaussian gravitational constant, ``nE`` Earth's mean motion and
``C1 = nE * RAD2PX`` (17,741 px/d per au^-1 at 0.2 arcsec/px).  ``departure`` is
the maximum distance of a reflex-frame track from its best-fit straight line over a
centred window of ``T`` days: it vanishes for an object exactly at the guess
distance and is the reason a window at one reflex node can only cover a shell of
distances around that node.  The step rule is kbmod-ops' ``velocity_grid_coverage``
budget: the nearest grid node is at most half a step away and drifts ``step * T / 2``
over the window.

Measured results this module encodes as policy (E1-E3 and the 2026-10-01 arm cells on
the 42 au golden patches 467847 and 468387, documented in the 42 au search recipe):

* ``results_per_pixel`` is the largest lever at short spans (4/8/16/64/128 recovered
  39/53/77/131/168 at 3 d) and marginal at 7 d.
* A +/-90 degree fan is harmful at the pool stage, not merely wasteful: +/-30 and
  +/-45 degree fans recovered more at fixed ``results_per_pixel``.
* The grid step was null at 3-8 d once the fan and per-pixel crowding were
  controlled; a step at the ``dv * span <= FWHM`` law is enough.
* Clustering and any likelihood cap belong offline; the search writes a pool.
* The CPU sigma-G path loses objects relative to the GPU one.
"""

from __future__ import annotations

import argparse
import io
import math
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import yaml

ARCSEC_PER_RADIAN = 206264.80624709636
GAUSSIAN_GRAVITATIONAL_CONSTANT = 0.01720209895  # au^1.5 / day
EARTH_MEAN_MOTION_RAD_PER_DAY = 2.0 * math.pi / 365.25
GAUSSIAN_FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))

DEFAULT_PIXEL_SCALE_ARCSEC_PER_PIXEL = 0.2
# Measured PSF FWHM of the 471085 WorkUnit layers (5.35 px at 0.2 arcsec/px).
DEFAULT_PSF_FWHM_PIXELS = 5.35
DEFAULT_ELONGATION_RANGE_DEG = (135.0, 180.0)
DEFAULT_INCLINATION_MAX_DEG = 30.0
DEFAULT_SPAN_DAYS = 7.0
DEFAULT_VELOCITY_MARGIN_FRACTION = 0.10
DEFAULT_MIN_ANGLE_HALF_WIDTH_DEG = 5.0
DEFAULT_TIER_OVERLAP_FRACTION = 0.10
DEFAULT_PIXEL_FRAME_TILT_DEG = 10.0
PRODUCTION_TRAJECTORIES_PER_PIXEL = 64 * 64
OPTION_A_TRAJECTORIES_PER_PIXEL = 256 * 192
SINGLE_FAN_MAX_HALF_WIDTH_DEG = 45.0
SPAN_WARNING_DAYS = 8.0
SHORT_SPAN_RPP_DAYS = 3.5

TOLERANCE_PRESETS: dict[str, tuple[str, float]] = {
    # key: (label, fraction of the PSF FWHM)
    "sigma": ("1 PSF sigma (strict; audit section 5 sub-PSF budget)", 1.0 / GAUSSIAN_FWHM_PER_SIGMA),
    "half_fwhm": ("half FWHM (the dv x span <= FWHM law; E2 found finer steps bought nothing at 3-8 d)", 0.5),
    "fwhm": ("1 PSF FWHM (lenient; recovery on 471085 died by ~1.3 FWHM of smear)", 1.0),
}
DEFAULT_TOLERANCE_PRESET = "half_fwhm"

# Fixed-fan fractions of the 153 golden 42 au Sorcha injections on 471085 (measured
# from their own reflex columns, 2026-09-17).  No model prior exists for the real
# population, so these are reported as context, not used in the geometry.
MEASURED_FAN_FRACTIONS_42AU: tuple[tuple[float, float], ...] = (
    (15.0, 0.804),
    (30.0, 0.915),
    (45.0, 0.935),
    (60.0, 0.987),
    (90.0, 1.000),
)
MEASURED_VPERP_FRACTIONS_42AU: tuple[tuple[float, float], ...] = (
    (10.0, 0.745),
    (20.0, 0.902),
    (30.0, 0.915),
    (50.0, 0.987),
    (75.0, 1.000),
)

NODE_PRESETS: dict[str, float] = {
    "Jupiter Trojans (5.2 au)": 5.2,
    "17 au": 17.0,
    "28 au": 28.0,
    "Classical belt (42 au)": 42.0,
    "New Horizons field (70 au)": 70.0,
}


# --------------------------------------------------------------------------- physics
def radians_to_pixels(pixel_scale_arcsec_per_pixel: float = DEFAULT_PIXEL_SCALE_ARCSEC_PER_PIXEL) -> float:
    """Pixels per radian for a pixel scale."""

    _require_positive("pixel_scale_arcsec_per_pixel", pixel_scale_arcsec_per_pixel)
    return ARCSEC_PER_RADIAN / pixel_scale_arcsec_per_pixel


def parallax_rate_constant(pixel_scale_arcsec_per_pixel: float = DEFAULT_PIXEL_SCALE_ARCSEC_PER_PIXEL) -> float:
    """``C1``: Earth's mean motion in px/d per au^-1 (17,741 at 0.2 arcsec/px)."""

    return EARTH_MEAN_MOTION_RAD_PER_DAY * radians_to_pixels(pixel_scale_arcsec_per_pixel)


def orbital_rate_pixels_per_day(
    distance_au: float,
    pixel_scale_arcsec_per_pixel: float = DEFAULT_PIXEL_SCALE_ARCSEC_PER_PIXEL,
) -> float:
    """Circular-orbit angular rate about the barycentre, in px/d (65 at 42 au, 30 at 70 au)."""

    _require_positive("distance_au", distance_au)
    return GAUSSIAN_GRAVITATIONAL_CONSTANT * distance_au**-1.5 * radians_to_pixels(pixel_scale_arcsec_per_pixel)


def parallax_mismatch_rate(
    distance_au: float,
    guess_distance_au: float,
    elongation_deg: float,
    pixel_scale_arcsec_per_pixel: float = DEFAULT_PIXEL_SCALE_ARCSEC_PER_PIXEL,
) -> float:
    """Residual parallax rate of a point at ``distance_au`` in a frame corrected to ``guess_distance_au``.

    Positive (eastward) for objects beyond the guess, negative for objects inside it, zero at the guess
    and at quadrature.
    """

    _require_positive("distance_au", distance_au)
    _require_positive("guess_distance_au", guess_distance_au)
    cos_e = abs(math.cos(math.radians(elongation_deg)))
    return parallax_rate_constant(pixel_scale_arcsec_per_pixel) * cos_e * (1.0 / guess_distance_au - 1.0 / distance_au)


def along_ecliptic_rate(
    distance_au: float,
    guess_distance_au: float,
    elongation_deg: float,
    inclination_deg: float = 0.0,
    pixel_scale_arcsec_per_pixel: float = DEFAULT_PIXEL_SCALE_ARCSEC_PER_PIXEL,
) -> float:
    """Along-ecliptic reflex-frame rate, px/d, for an object at its node in a field on the ecliptic."""

    orbital = orbital_rate_pixels_per_day(distance_au, pixel_scale_arcsec_per_pixel)
    return orbital * math.cos(math.radians(inclination_deg)) + parallax_mismatch_rate(
        distance_au, guess_distance_au, elongation_deg, pixel_scale_arcsec_per_pixel
    )


def cross_ecliptic_rate_max(
    distance_au: float,
    inclination_deg: float,
    pixel_scale_arcsec_per_pixel: float = DEFAULT_PIXEL_SCALE_ARCSEC_PER_PIXEL,
) -> float:
    """Largest cross-ecliptic rate, px/d, of an orbit with this inclination (independent of the guess)."""

    return orbital_rate_pixels_per_day(distance_au, pixel_scale_arcsec_per_pixel) * abs(
        math.sin(math.radians(inclination_deg))
    )


def track_departure_pixels(
    span_days: float,
    distance_au: float,
    guess_distance_au: float,
    elongation_deg: float,
    pixel_scale_arcsec_per_pixel: float = DEFAULT_PIXEL_SCALE_ARCSEC_PER_PIXEL,
) -> float:
    """Maximum departure of the reflex-frame track from a straight line over a centred window."""

    if span_days < 0.0:
        raise ValueError("span_days must be non-negative")
    _require_positive("distance_au", distance_au)
    _require_positive("guess_distance_au", guess_distance_au)
    inverse_offset = abs(1.0 / distance_au - 1.0 / guess_distance_au)
    return _departure_for_inverse_offset(span_days, inverse_offset, elongation_deg, pixel_scale_arcsec_per_pixel)


def _departure_for_inverse_offset(
    span_days: float,
    inverse_offset_per_au: float,
    elongation_deg: float,
    pixel_scale_arcsec_per_pixel: float,
) -> float:
    e = math.radians(elongation_deg)
    c1 = parallax_rate_constant(pixel_scale_arcsec_per_pixel)
    n_e = EARTH_MEAN_MOTION_RAD_PER_DAY
    quadratic = c1 * n_e * abs(math.sin(e)) * inverse_offset_per_au * span_days**2 / 12.0
    cubic = c1 * n_e**2 * abs(math.cos(e)) * inverse_offset_per_au * span_days**3 / 120.0
    return quadratic + cubic


def max_coherent_span_days(
    distance_au: float,
    guess_distance_au: float,
    elongation_deg: float,
    tolerance_px: float,
    pixel_scale_arcsec_per_pixel: float = DEFAULT_PIXEL_SCALE_ARCSEC_PER_PIXEL,
    span_cap_days: float = 400.0,
) -> float:
    """Longest centred window over which the track stays within ``tolerance_px`` of a straight line."""

    _require_positive("tolerance_px", tolerance_px)
    if math.isclose(distance_au, guess_distance_au):
        return span_cap_days
    lo, hi = 0.0, span_cap_days
    if track_departure_pixels(hi, distance_au, guess_distance_au, elongation_deg, pixel_scale_arcsec_per_pixel) <= tolerance_px:
        return span_cap_days
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if track_departure_pixels(mid, distance_au, guess_distance_au, elongation_deg, pixel_scale_arcsec_per_pixel) <= tolerance_px:
            lo = mid
        else:
            hi = mid
    return lo


def max_inverse_distance_offset(
    span_days: float,
    elongation_deg: float,
    tolerance_px: float,
    pixel_scale_arcsec_per_pixel: float = DEFAULT_PIXEL_SCALE_ARCSEC_PER_PIXEL,
) -> float:
    """Largest ``|1/d - 1/d_guess|`` (au^-1) whose track stays coherent over ``span_days``."""

    _require_positive("span_days", span_days)
    _require_positive("tolerance_px", tolerance_px)
    lo, hi = 0.0, 1.0
    if _departure_for_inverse_offset(span_days, hi, elongation_deg, pixel_scale_arcsec_per_pixel) <= tolerance_px:
        return hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _departure_for_inverse_offset(span_days, mid, elongation_deg, pixel_scale_arcsec_per_pixel) <= tolerance_px:
            lo = mid
        else:
            hi = mid
    return lo


def coherent_distance_range(
    guess_distance_au: float,
    span_days: float,
    elongation_deg: float,
    tolerance_px: float,
    pixel_scale_arcsec_per_pixel: float = DEFAULT_PIXEL_SCALE_ARCSEC_PER_PIXEL,
) -> tuple[float, float]:
    """Distances whose tracks stay coherent at one node over ``span_days`` (upper bound may be ``inf``)."""

    _require_positive("guess_distance_au", guess_distance_au)
    inverse_offset = max_inverse_distance_offset(span_days, elongation_deg, tolerance_px, pixel_scale_arcsec_per_pixel)
    inverse_guess = 1.0 / guess_distance_au
    lower = 1.0 / (inverse_guess + inverse_offset)
    upper = math.inf if inverse_offset >= inverse_guess else 1.0 / (inverse_guess - inverse_offset)
    return lower, upper


def cancellation_distance_au(
    guess_distance_au: float,
    elongation_deg: float,
    pixel_scale_arcsec_per_pixel: float = DEFAULT_PIXEL_SCALE_ARCSEC_PER_PIXEL,
) -> float | None:
    """Distance inside the guess where the along-ecliptic rate reverses (about 61 au for a 70 au node).

    Returns ``None`` when there is no reversal (at quadrature the parallax mismatch vanishes).
    """

    _require_positive("guess_distance_au", guess_distance_au)
    if abs(math.cos(math.radians(elongation_deg))) < 1e-9:
        return None
    lo = max(0.05 * guess_distance_au, 1.05)
    hi = guess_distance_au
    if lo >= hi:
        return None
    rate = lambda d: along_ecliptic_rate(d, guess_distance_au, elongation_deg, 0.0, pixel_scale_arcsec_per_pixel)
    if rate(lo) >= 0.0:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if rate(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def blind_zones_au(
    guess_distance_au: float,
    elongation_deg: float,
    velocity_floor_pixels_per_day: float,
    pixel_scale_arcsec_per_pixel: float = DEFAULT_PIXEL_SCALE_ARCSEC_PER_PIXEL,
    distance_min_au: float | None = None,
    distance_max_au: float | None = None,
    samples: int = 4001,
) -> tuple[tuple[float, float], ...]:
    """Distance intervals where an ecliptic (i = 0) object moves slower than the velocity floor.

    A floor of 25 px/d at a 70 au node hides roughly 52-68 au.  Intervals are clipped to the
    requested distance range (default 0.3 to 3 times the guess).
    """

    _require_positive("guess_distance_au", guess_distance_au)
    if velocity_floor_pixels_per_day <= 0.0:
        return ()
    lo = distance_min_au if distance_min_au is not None else 0.3 * guess_distance_au
    hi = distance_max_au if distance_max_au is not None else 3.0 * guess_distance_au
    lo = max(lo, 1.05)
    if hi <= lo:
        return ()
    inverse = [1.0 / lo + (1.0 / hi - 1.0 / lo) * k / (samples - 1) for k in range(samples)]
    distances = [1.0 / value for value in inverse]
    slow = [
        abs(along_ecliptic_rate(d, guess_distance_au, elongation_deg, 0.0, pixel_scale_arcsec_per_pixel))
        < velocity_floor_pixels_per_day
        for d in distances
    ]
    zones: list[tuple[float, float]] = []
    start: float | None = None
    for d, is_slow in zip(distances, slow):
        if is_slow and start is None:
            start = d
        elif not is_slow and start is not None:
            zones.append((start, d))
            start = None
    if start is not None:
        zones.append((start, distances[-1]))
    return tuple(zones)


def required_velocity_step(span_days: float, tolerance_px: float) -> float:
    """Largest velocity step (px/d) whose half-step drift over the window stays within the tolerance."""

    _require_positive("span_days", span_days)
    _require_positive("tolerance_px", tolerance_px)
    return 2.0 * tolerance_px / span_days


def required_angle_step_deg(span_days: float, tolerance_px: float, speed_pixels_per_day: float) -> float:
    """Largest angle step (deg) at a given speed whose half-step cross-track drift stays within the tolerance."""

    _require_positive("speed_pixels_per_day", speed_pixels_per_day)
    return math.degrees(2.0 * tolerance_px / (speed_pixels_per_day * span_days))


def inclusive_sample_count(width: float, step: float) -> int:
    """Inclusive sample count for an interval of ``width`` at a step no larger than ``step``."""

    if width <= 0.0:
        return 1
    _require_positive("step", step)
    return int(math.ceil(width / step - 1e-9)) + 1


def tolerance_pixels(preset: str, psf_fwhm_pixels: float) -> float:
    """Resolve a tolerance preset against the PSF FWHM."""

    _require_positive("psf_fwhm_pixels", psf_fwhm_pixels)
    try:
        _, fraction = TOLERANCE_PRESETS[preset]
    except KeyError as exc:
        raise ValueError(f"unknown tolerance preset {preset!r}; choose from {sorted(TOLERANCE_PRESETS)}") from exc
    return fraction * psf_fwhm_pixels


# --------------------------------------------------------------------------- coverage specification
@dataclass(frozen=True)
class CoverageSpec:
    """What the search is meant to cover at one reflex node."""

    guess_distance_au: float
    distance_min_au: float
    distance_max_au: float
    inclination_max_deg: float = DEFAULT_INCLINATION_MAX_DEG
    elongation_min_deg: float = DEFAULT_ELONGATION_RANGE_DEG[0]
    elongation_max_deg: float = DEFAULT_ELONGATION_RANGE_DEG[1]
    pixel_scale_arcsec_per_pixel: float = DEFAULT_PIXEL_SCALE_ARCSEC_PER_PIXEL

    def __post_init__(self) -> None:
        _require_positive("guess_distance_au", self.guess_distance_au)
        if self.distance_min_au <= 1.0:
            raise ValueError("distance_min_au must exceed 1 au (the model needs the object beyond the Earth)")
        if self.distance_max_au <= self.distance_min_au:
            raise ValueError("distance_max_au must exceed distance_min_au")
        if not 0.0 <= self.inclination_max_deg <= 90.0:
            raise ValueError("inclination_max_deg must be between 0 and 90 degrees")
        if not 90.0 <= self.elongation_min_deg <= self.elongation_max_deg <= 180.0:
            raise ValueError("elongations must satisfy 90 <= min <= max <= 180 degrees")
        _require_positive("pixel_scale_arcsec_per_pixel", self.pixel_scale_arcsec_per_pixel)

    @classmethod
    def symmetric(cls, guess_distance_au: float, half_width_au: float, **kwargs: Any) -> "CoverageSpec":
        """Coverage of ``guess +/- half_width`` au."""

        _require_positive("half_width_au", half_width_au)
        return cls(
            guess_distance_au=guess_distance_au,
            distance_min_au=guess_distance_au - half_width_au,
            distance_max_au=guess_distance_au + half_width_au,
            **kwargs,
        )

    @property
    def half_width_au(self) -> float:
        return 0.5 * (self.distance_max_au - self.distance_min_au)

    @property
    def worst_elongation_deg(self) -> float:
        """The elongation with the largest curvature (farthest from opposition)."""

        return self.elongation_min_deg


@dataclass(frozen=True)
class RegionSample:
    distance_au: float
    elongation_deg: float
    inclination_deg: float
    v_along: float
    v_perp: float

    @property
    def speed(self) -> float:
        return math.hypot(self.v_along, self.v_perp)

    @property
    def angle_deg(self) -> float:
        return math.degrees(math.atan2(self.v_perp, self.v_along))


@dataclass(frozen=True)
class OccupiedRegion:
    """Bounding description of where the covered objects sit in reflex-frame velocity space."""

    spec: CoverageSpec
    v_along_min: float
    v_along_max: float
    v_perp_max: float
    speed_min: float
    speed_max: float
    angle_max_deg: float
    cancellation_distances_au: tuple[float, ...]
    samples: tuple[RegionSample, ...] = field(repr=False)
    # (speed, |angle|) per sample, sorted by speed, so tier scans do not recompute them.
    speed_angle_table: tuple[tuple[float, float], ...] = field(default=(), repr=False)

    @property
    def retrograde_looking(self) -> bool:
        """True when some covered object moves westward (inside the cancellation distance)."""

        return self.v_along_min < 0.0

    @property
    def fan_can_hold_region(self) -> bool:
        return not self.retrograde_looking

    def angle_max_above_speed(self, speed_floor: float) -> float:
        """Widest angle among covered objects at least as fast as ``speed_floor``."""

        table = self.speed_angle_table or tuple(sorted((s.speed, abs(s.angle_deg)) for s in self.samples))
        widest = 0.0
        for speed, angle in reversed(table):
            if speed < speed_floor:
                break
            if angle > widest:
                widest = angle
        return widest


def occupied_region(
    spec: CoverageSpec,
    *,
    distance_samples: int = 97,
    elongation_samples: int = 7,
    inclination_samples: int = 13,
) -> OccupiedRegion:
    """Sample the covered (distance, elongation, inclination) box in reflex-frame velocity space."""

    inverse_lo, inverse_hi = 1.0 / spec.distance_min_au, 1.0 / spec.distance_max_au
    distances = [
        1.0 / (inverse_lo + (inverse_hi - inverse_lo) * k / (distance_samples - 1)) for k in range(distance_samples)
    ]
    if spec.distance_min_au < spec.guess_distance_au < spec.distance_max_au:
        distances.append(spec.guess_distance_au)
    if elongation_samples == 1 or math.isclose(spec.elongation_min_deg, spec.elongation_max_deg):
        elongations = [spec.elongation_max_deg]
    else:
        elongations = [
            spec.elongation_min_deg + (spec.elongation_max_deg - spec.elongation_min_deg) * k / (elongation_samples - 1)
            for k in range(elongation_samples)
        ]
    if inclination_samples == 1 or spec.inclination_max_deg == 0.0:
        inclinations = [0.0]
    else:
        inclinations = [spec.inclination_max_deg * k / (inclination_samples - 1) for k in range(inclination_samples)]

    samples: list[RegionSample] = []
    for d in distances:
        for e in elongations:
            parallax = parallax_mismatch_rate(d, spec.guess_distance_au, e, spec.pixel_scale_arcsec_per_pixel)
            orbital = orbital_rate_pixels_per_day(d, spec.pixel_scale_arcsec_per_pixel)
            for i in inclinations:
                v_along = orbital * math.cos(math.radians(i)) + parallax
                v_perp = orbital * math.sin(math.radians(i))
                samples.append(RegionSample(d, e, i, v_along, v_perp))
                if v_perp > 0.0:
                    samples.append(RegionSample(d, e, -i, v_along, -v_perp))

    v_along_values = [s.v_along for s in samples]
    speeds = [s.speed for s in samples]
    angles = [abs(s.angle_deg) for s in samples]
    cancellations = tuple(
        sorted(
            {
                round(value, 6)
                for value in (
                    cancellation_distance_au(spec.guess_distance_au, e, spec.pixel_scale_arcsec_per_pixel)
                    for e in elongations
                )
                if value is not None and spec.distance_min_au <= value <= spec.distance_max_au
            }
        )
    )
    return OccupiedRegion(
        spec=spec,
        v_along_min=min(v_along_values),
        v_along_max=max(v_along_values),
        v_perp_max=max(abs(s.v_perp) for s in samples),
        speed_min=0.0 if (min(v_along_values) < 0.0 < max(v_along_values)) else min(speeds),
        speed_max=max(speeds),
        angle_max_deg=max(angles),
        cancellation_distances_au=cancellations,
        samples=tuple(samples),
        speed_angle_table=tuple(sorted(zip(speeds, angles))),
    )


# --------------------------------------------------------------------------- grid designs
@dataclass(frozen=True)
class FanTier:
    """One ``EclipticCenteredSearch`` grid."""

    label: str
    velocity_min: float
    velocity_max: float
    velocity_samples: int
    angle_half_width_deg: float
    angle_samples: int

    @property
    def velocity_step(self) -> float:
        if self.velocity_samples <= 1:
            return 0.0
        return (self.velocity_max - self.velocity_min) / (self.velocity_samples - 1)

    @property
    def angle_step_deg(self) -> float:
        if self.angle_samples <= 1:
            return 0.0
        return 2.0 * self.angle_half_width_deg / (self.angle_samples - 1)

    @property
    def trajectories(self) -> int:
        return self.velocity_samples * self.angle_samples

    def generator_config(self) -> dict[str, Any]:
        return {
            "name": "EclipticCenteredSearch",
            "angle_units": "degree",
            "angles": [-_round(self.angle_half_width_deg), _round(self.angle_half_width_deg), int(self.angle_samples)],
            "velocities": [_round(self.velocity_min), _round(self.velocity_max), int(self.velocity_samples)],
            "velocity_units": "pix / d",
            "given_ecliptic": None,
        }


@dataclass(frozen=True)
class VelocityBox:
    """One ``VelocityGridSearch`` grid in pixel axes, padded for the ecliptic-to-pixel tilt."""

    label: str
    vx_min: float
    vx_max: float
    vx_samples: int
    vy_min: float
    vy_max: float
    vy_samples: int
    tilt_deg: float
    ecliptic_box: tuple[float, float, float, float]  # (v_along_min, v_along_max, -v_perp, +v_perp) before padding
    unpadded_trajectories: int

    @property
    def step(self) -> float:
        if self.vx_samples <= 1:
            return 0.0
        return (self.vx_max - self.vx_min) / (self.vx_samples - 1)

    @property
    def trajectories(self) -> int:
        return self.vx_samples * self.vy_samples

    def generator_config(self) -> dict[str, Any]:
        return {
            "name": "VelocityGridSearch",
            "vx_steps": int(self.vx_samples),
            "min_vx": _round(self.vx_min),
            "max_vx": _round(self.vx_max),
            "vy_steps": int(self.vy_samples),
            "min_vy": _round(self.vy_min),
            "max_vy": _round(self.vy_max),
        }


@dataclass(frozen=True)
class GridDesign:
    kind: str  # "single_fan", "two_tier", "velocity_box"
    title: str
    tiers: tuple[FanTier, ...] = ()
    box: VelocityBox | None = None
    notes: tuple[str, ...] = ()

    @property
    def trajectories(self) -> int:
        if self.box is not None:
            return self.box.trajectories
        return sum(tier.trajectories for tier in self.tiers)

    @property
    def cost_vs_production(self) -> float:
        return self.trajectories / PRODUCTION_TRAJECTORIES_PER_PIXEL

    @property
    def cost_vs_option_a(self) -> float:
        return self.trajectories / OPTION_A_TRAJECTORIES_PER_PIXEL

    def generator_configs(self) -> tuple[dict[str, Any], ...]:
        if self.box is not None:
            return (self.box.generator_config(),)
        return tuple(tier.generator_config() for tier in self.tiers)

    @property
    def velocity_floor(self) -> float:
        if self.box is not None:
            return 0.0 if self.box.vx_min <= 0.0 <= self.box.vx_max else min(abs(self.box.vx_min), abs(self.box.vx_max))
        return min(tier.velocity_min for tier in self.tiers)


@dataclass(frozen=True)
class DesignRules:
    """Knobs that turn an occupied region into grids."""

    velocity_margin_fraction: float = DEFAULT_VELOCITY_MARGIN_FRACTION
    min_angle_half_width_deg: float = DEFAULT_MIN_ANGLE_HALF_WIDTH_DEG
    tier_overlap_fraction: float = DEFAULT_TIER_OVERLAP_FRACTION
    pixel_frame_tilt_deg: float = DEFAULT_PIXEL_FRAME_TILT_DEG
    single_fan_max_half_width_deg: float = SINGLE_FAN_MAX_HALF_WIDTH_DEG

    def __post_init__(self) -> None:
        if self.velocity_margin_fraction < 0.0:
            raise ValueError("velocity_margin_fraction must be non-negative")
        if self.min_angle_half_width_deg < 0.0 or self.min_angle_half_width_deg > 180.0:
            raise ValueError("min_angle_half_width_deg must be within 0-180 degrees")
        if not 0.0 <= self.tier_overlap_fraction < 0.5:
            raise ValueError("tier_overlap_fraction must be within [0, 0.5)")
        if not 0.0 <= self.pixel_frame_tilt_deg <= 90.0:
            raise ValueError("pixel_frame_tilt_deg must be within 0-90 degrees")


def _speed_bounds_with_margin(region: OccupiedRegion, rules: DesignRules) -> tuple[float, float]:
    margin = rules.velocity_margin_fraction
    v_min = max(0.0, region.speed_min * (1.0 - margin))
    v_max = region.speed_max * (1.0 + margin)
    return v_min, v_max


def _fan_tier(
    label: str,
    v_min: float,
    v_max: float,
    half_width_deg: float,
    span_days: float,
    tolerance_px: float,
    rules: DesignRules,
) -> FanTier:
    half_width = min(180.0, max(half_width_deg, rules.min_angle_half_width_deg))
    dv = required_velocity_step(span_days, tolerance_px)
    dtheta = required_angle_step_deg(span_days, tolerance_px, max(v_max, 1e-6))
    return FanTier(
        label=label,
        velocity_min=v_min,
        velocity_max=v_max,
        velocity_samples=inclusive_sample_count(v_max - v_min, dv),
        angle_half_width_deg=half_width,
        angle_samples=inclusive_sample_count(2.0 * half_width, dtheta),
    )


def design_single_fan(region: OccupiedRegion, span_days: float, tolerance_px: float, rules: DesignRules) -> GridDesign:
    """One ``EclipticCenteredSearch`` fan wide enough for the slowest covered object."""

    v_min, v_max = _speed_bounds_with_margin(region, rules)
    half_width = 180.0 if region.retrograde_looking else region.angle_max_deg
    tier = _fan_tier("single fan", v_min, v_max, half_width, span_days, tolerance_px, rules)
    notes = [
        "angle step sized at the top speed, so slower rings are over-sampled in angle",
    ]
    if region.retrograde_looking:
        notes.append(
            "coverage includes the cancellation distance: some objects look retrograde, so the fan must span 360 degrees"
        )
    return GridDesign(kind="single_fan", title="Single fan (EclipticCenteredSearch)", tiers=(tier,), notes=tuple(notes))


def design_two_tier(region: OccupiedRegion, span_days: float, tolerance_px: float, rules: DesignRules) -> GridDesign:
    """Slow wide fan plus fast narrow fan, split where the total trajectory count is smallest."""

    v_min, v_max = _speed_bounds_with_margin(region, rules)
    if region.retrograde_looking or v_max <= v_min * 1.2:
        single = design_single_fan(region, span_days, tolerance_px, rules)
        reason = (
            "westward-moving objects are inside the coverage, so no speed splits off a narrow tier"
            if region.retrograde_looking
            else "the speed range is too narrow to split"
        )
        return GridDesign(
            kind="two_tier",
            title="Two speed tiers (collapses to one fan here)",
            tiers=single.tiers,
            notes=(f"{reason}; this is the single fan again", *single.notes),
        )
    overlap = rules.tier_overlap_fraction
    best: tuple[int, FanTier, FanTier] | None = None
    candidates = 48
    for k in range(1, candidates):
        v_split = v_min + (v_max - v_min) * k / candidates
        slow_hi = min(v_max, v_split * (1.0 + overlap))
        fast_lo = max(v_min, v_split * (1.0 - overlap))
        fast_half = region.angle_max_above_speed(fast_lo)
        slow = _fan_tier("slow tier", v_min, slow_hi, region.angle_max_deg, span_days, tolerance_px, rules)
        fast = _fan_tier("fast tier", fast_lo, v_max, fast_half, span_days, tolerance_px, rules)
        total = slow.trajectories + fast.trajectories
        if best is None or total < best[0]:
            best = (total, slow, fast)
    assert best is not None
    _, slow, fast = best
    notes = (
        f"tiers overlap over {_fmt(fast.velocity_min, 1)}-{_fmt(slow.velocity_max, 1)} px/d so no object sits on the seam",
        "the wide-angle tail of a TNO population is entirely slow (471085: everything faster than ~80 px/d within +/-10 deg)",
        "each tier is a separate kbmod search; combine the pools before any offline stage",
    )
    return GridDesign(kind="two_tier", title="Two speed tiers (EclipticCenteredSearch x2)", tiers=(slow, fast), notes=notes)


def design_velocity_box(region: OccupiedRegion, span_days: float, tolerance_px: float, rules: DesignRules) -> GridDesign:
    """Cartesian ``VelocityGridSearch`` box bounding v_along and v_perp, padded for the pixel-frame tilt."""

    margin = rules.velocity_margin_fraction
    along_span = region.v_along_max - region.v_along_min
    pad_along = max(along_span * margin, region.speed_max * margin * 0.5)
    pad_perp = region.v_perp_max * margin
    min_perp = region.speed_max * math.sin(math.radians(rules.min_angle_half_width_deg))
    ax0 = region.v_along_min - pad_along
    ax1 = region.v_along_max + pad_along
    ay = max(region.v_perp_max + pad_perp, min_perp)
    step = required_velocity_step(span_days, tolerance_px)
    unpadded = inclusive_sample_count(ax1 - ax0, step) * inclusive_sample_count(2.0 * ay, step)

    # Rotate the ecliptic-frame box by the tilt and take its pixel-axis bounding box.
    phi = math.radians(rules.pixel_frame_tilt_deg)
    cx = 0.5 * (ax0 + ax1)
    half_x = 0.5 * (ax1 - ax0)
    half_y = ay
    centre_x = cx * math.cos(phi)
    centre_y = cx * math.sin(phi)
    bound_x = half_x * abs(math.cos(phi)) + half_y * abs(math.sin(phi))
    bound_y = half_x * abs(math.sin(phi)) + half_y * abs(math.cos(phi))
    vx_min, vx_max = centre_x - bound_x, centre_x + bound_x
    vy_min, vy_max = centre_y - bound_y, centre_y + bound_y
    box = VelocityBox(
        label="velocity box",
        vx_min=vx_min,
        vx_max=vx_max,
        vx_samples=inclusive_sample_count(vx_max - vx_min, step),
        vy_min=vy_min,
        vy_max=vy_max,
        vy_samples=inclusive_sample_count(vy_max - vy_min, step),
        tilt_deg=rules.pixel_frame_tilt_deg,
        ecliptic_box=(ax0, ax1, -ay, ay),
        unpadded_trajectories=unpadded,
    )
    notes = (
        "VelocityGridSearch is axis-aligned in pixel space and does not read the ecliptic angle: "
        "generate the box per WorkUnit from its ecliptic angle, or pad for the worst tilt as done here",
        f"tilt padding of {_fmt(rules.pixel_frame_tilt_deg, 1)} deg costs {box.trajectories / max(unpadded, 1):.2f}x "
        "over an ecliptic-aligned box (8 deg on 471085, up to 23.4 deg near the equinoxes)",
        "both axes use the along-track step, so the cross-track budget holds at every speed",
    )
    return GridDesign(kind="velocity_box", title="Velocity box (VelocityGridSearch)", box=box, notes=notes)


# --------------------------------------------------------------------------- recommendation
@dataclass(frozen=True)
class CoherenceSummary:
    tolerance_px: float
    span_days: float
    max_span_at_edges_days: float  # worst (shortest) over both coverage edges at the worst elongation
    max_span_near_edge_days: float
    max_span_far_edge_days: float
    coherent_distance_min_au: float  # at the chosen span and worst elongation
    coherent_distance_max_au: float
    span_table: tuple[tuple[float, float, float], ...]  # (elongation, Tmax near edge, Tmax far edge)
    ladder_nodes_au: tuple[float, ...]  # nodes needed to cover the spec coherently at this span

    @property
    def coverage_is_coherent(self) -> bool:
        return self.span_days <= self.max_span_at_edges_days + 1e-9


@dataclass(frozen=True)
class SettingLine:
    key: str
    value: Any
    why: str


@dataclass(frozen=True)
class Recommendation:
    spec: CoverageSpec
    region: OccupiedRegion
    span_days: float
    psf_fwhm_pixels: float
    tolerance_preset: str
    tolerance_px: float
    rules: DesignRules
    designs: tuple[GridDesign, ...]
    recommended_kind: str
    recommendation_reason: str
    coherence: CoherenceSummary
    settings: tuple[SettingLine, ...]
    warnings: tuple[str, ...]
    blind_zones_au: tuple[tuple[float, float], ...]

    @property
    def recommended(self) -> GridDesign:
        for design in self.designs:
            if design.kind == self.recommended_kind:
                return design
        return self.designs[0]

    def design(self, kind: str) -> GridDesign:
        for design in self.designs:
            if design.kind == kind:
                return design
        raise KeyError(kind)


def distance_ladder_nodes(
    distance_min_au: float,
    distance_max_au: float,
    span_days: float,
    elongation_deg: float,
    tolerance_px: float,
    pixel_scale_arcsec_per_pixel: float = DEFAULT_PIXEL_SCALE_ARCSEC_PER_PIXEL,
) -> tuple[float, ...]:
    """Guess distances such that every distance in the range is coherent at some node over the span."""

    if distance_max_au <= distance_min_au:
        raise ValueError("distance_max_au must exceed distance_min_au")
    inverse_offset = max_inverse_distance_offset(span_days, elongation_deg, tolerance_px, pixel_scale_arcsec_per_pixel)
    inverse_span = 1.0 / distance_min_au - 1.0 / distance_max_au
    n_nodes = max(1, int(math.ceil(inverse_span / (2.0 * inverse_offset) - 1e-9)))
    return tuple(1.0 / (1.0 / distance_min_au - (k + 0.5) * inverse_span / n_nodes) for k in range(n_nodes))


def pool_stage_settings(span_days: float, guess_distance_au: float) -> tuple[SettingLine, ...]:
    """Non-grid kbmod settings for a pool-stage search, with the measured reason for each."""

    rpp = 128 if span_days <= SHORT_SPAN_RPP_DAYS else 64
    rpp_why = (
        "largest lever at short spans: rpp 4/8/16/64/128 recovered 39/53/77/131/168 at 3 d on the 42 au golden "
        "patches; 128 still gains at 3 d (chance rate 1-2%), only +2-6 objects at 7 d"
    )
    return (
        SettingLine("results_per_pixel", rpp, rpp_why),
        SettingLine("lh_level", 4.0, "pool floor; clears 2020 KV11 (4.37) and 2013 TA228 (4.75); never a ranker"),
        SettingLine("num_obs", 3, "convention kept by PI decision 2026-10-01; no cell has varied it"),
        SettingLine("sigmaG_filter", True, "keep on; a sigma-G-off pool writes no obs_valid and must be repaired before scoring"),
        SettingLine("sigmaG_lims", [25, 75], "production limits, unchanged"),
        SettingLine("gpu_filter", True, "the CPU sigma-G path lost 15 objects pooled (56->54, 75->62) at 0.80x rows"),
        SettingLine("clip_negative", False, "production default, unchanged"),
        SettingLine("near_dup_thresh", 10, "production value; near_dup 0 cells were still running on 2026-10-01"),
        SettingLine("do_clustering", False, "search into a pool; nn_start_end eps 120 kept only a quarter to a half of objects"),
        SettingLine("cluster_type", "nn_start_end", "for the offline stage: eps <= 20 keeps 82-95%"),
        SettingLine("cluster_eps", 20.0, "offline value; eps 5-10 when completeness matters more than rows"),
        SettingLine("max_results", -1, "a likelihood cap on an unclustered pool recovers zero objects; cap after clustering, >= 250k"),
        SettingLine("generate_psi_phi", True, "curves and obs_valid feed every offline stage"),
        SettingLine("track_filtered", False, "true is for pencil/diagnostic searches; in a blind search it exhausts host memory"),
        SettingLine("coadds", [], "stamps were 96% of the result bytes on the 471085 canary"),
        SettingLine("stamp_type", None, "must be null explicitly: coadds: [] alone does not disable stamps"),
        SettingLine("save_all_stamps", False, "stampless pool"),
        SettingLine("compute_ra_dec", True, "set false above ~2e9 row-epochs (one astropy call over >2^31 coordinates crashes)"),
        SettingLine("psf_val", 1.4, "fallback only; the psf-from-psfsigma branch takes per-image PSFs from the WorkUnit"),
        SettingLine("max_masked_pixels", 0.95, "as run in E2"),
        SettingLine("encode_num_bytes", -1, "no GPU encoding"),
        SettingLine("x_pixel_bounds", None, "full patch; tile start pixels into boxes only for memory"),
        SettingLine("y_pixel_bounds", None, "full patch"),
        SettingLine("debug", False, "quiet"),
    )


def recommend_search_configuration(
    spec: CoverageSpec,
    *,
    span_days: float = DEFAULT_SPAN_DAYS,
    psf_fwhm_pixels: float = DEFAULT_PSF_FWHM_PIXELS,
    tolerance_preset: str = DEFAULT_TOLERANCE_PRESET,
    rules: DesignRules | None = None,
) -> Recommendation:
    """Turn a coverage specification into grid designs, a recommended one, settings and warnings."""

    _require_positive("span_days", span_days)
    rules = rules or DesignRules()
    tol = tolerance_pixels(tolerance_preset, psf_fwhm_pixels)
    region = occupied_region(spec)
    scale = spec.pixel_scale_arcsec_per_pixel

    designs = (
        design_single_fan(region, span_days, tol, rules),
        design_two_tier(region, span_days, tol, rules),
        design_velocity_box(region, span_days, tol, rules),
    )
    single, two_tier, box = designs

    # Coherence at the coverage edges, at the worst elongation and across the range.
    worst_e = spec.worst_elongation_deg
    t_near = max_coherent_span_days(spec.distance_min_au, spec.guess_distance_au, worst_e, tol, scale)
    t_far = max_coherent_span_days(spec.distance_max_au, spec.guess_distance_au, worst_e, tol, scale)
    table_elongations = sorted({180.0, 165.0, 150.0, 135.0, 120.0, spec.elongation_min_deg, spec.elongation_max_deg}, reverse=True)
    span_table = tuple(
        (
            e,
            max_coherent_span_days(spec.distance_min_au, spec.guess_distance_au, e, tol, scale),
            max_coherent_span_days(spec.distance_max_au, spec.guess_distance_au, e, tol, scale),
        )
        for e in table_elongations
    )
    coh_lo, coh_hi = coherent_distance_range(spec.guess_distance_au, span_days, worst_e, tol, scale)
    ladder = distance_ladder_nodes(spec.distance_min_au, spec.distance_max_au, span_days, worst_e, tol, scale)
    coherence = CoherenceSummary(
        tolerance_px=tol,
        span_days=span_days,
        max_span_at_edges_days=min(t_near, t_far),
        max_span_near_edge_days=t_near,
        max_span_far_edge_days=t_far,
        coherent_distance_min_au=coh_lo,
        coherent_distance_max_au=coh_hi,
        span_table=span_table,
        ladder_nodes_au=ladder,
    )

    # Which design to carry forward.
    if region.retrograde_looking:
        kind = "velocity_box"
        if region.cancellation_distances_au:
            cause = (
                "the coverage straddles the cancellation distance, where the along-ecliptic rate passes through zero "
                "and flips sign"
            )
        else:
            cause = (
                "the most inclined objects at the near edge of the coverage move westward in this frame (their "
                "along-ecliptic orbital component omega cos i is smaller than the parallax mismatch)"
            )
        reason = (
            f"{cause}: no angle fan can hold that without spanning 360 degrees and a zero velocity floor, while a "
            "Cartesian box bounds it directly. Splitting the coverage into two nodes, or lowering the inclination "
            "ceiling, is the better fix."
        )
    elif single.tiers[0].angle_half_width_deg <= rules.single_fan_max_half_width_deg:
        kind = "single_fan"
        reason = (
            f"the whole region fits inside +/-{_fmt(single.tiers[0].angle_half_width_deg, 1)} deg, within the "
            f"+/-{_fmt(rules.single_fan_max_half_width_deg, 0)} deg fan the 2026-10-01 arm cells showed to recover "
            "more than +/-90 deg at the same results_per_pixel"
        )
    else:
        kind = "two_tier" if two_tier.trajectories < single.trajectories else "single_fan"
        reason = (
            f"the slow end needs +/-{_fmt(region.angle_max_deg, 1)} deg while objects above "
            f"{_fmt(two_tier.tiers[-1].velocity_min, 0)} px/d stay within +/-{_fmt(two_tier.tiers[-1].angle_half_width_deg, 1)} deg; "
            f"two tiers hold the same region at {two_tier.trajectories / max(single.trajectories, 1):.2f}x the single-fan cost "
            "and keep each start pixel's results_per_pixel slots away from off-angle trajectories"
            if kind == "two_tier"
            else "a single fan is no larger than two tiers for this region"
        )

    settings = pool_stage_settings(span_days, spec.guess_distance_au)
    chosen = single if kind == "single_fan" else (two_tier if kind == "two_tier" else box)
    floor = chosen.velocity_floor
    blind = blind_zones_au(spec.guess_distance_au, spec.elongation_max_deg, floor, scale, 0.3 * spec.guess_distance_au, 3.0 * spec.guess_distance_au)

    warnings: list[str] = []
    if not coherence.coverage_is_coherent:
        if len(ladder) == 1:
            remedy = f"a single node re-centred at {_fmt(ladder[0], 1)} au would hold it"
        else:
            remedy = f"covering it needs {len(ladder)} nodes at {', '.join(_fmt(n, 1) for n in ladder)} au"
        warnings.append(
            f"A {_fmt(span_days, 1)} d window is longer than the coherent window at the coverage edges "
            f"({_fmt(t_near, 1)} d at {_fmt(spec.distance_min_au, 1)} au, {_fmt(t_far, 1)} d at {_fmt(spec.distance_max_au, 1)} au, "
            f"elongation {_fmt(worst_e, 0)} deg). At this span the {_fmt(spec.guess_distance_au, 1)} au node holds only "
            f"{_fmt(coh_lo, 1)}-{_fmt_or_inf(coh_hi, 1)} au; {remedy}, or shorten the window to "
            f"<= {_fmt(coherence.max_span_at_edges_days, 1)} d."
        )
    if span_days > SPAN_WARNING_DAYS:
        warnings.append(
            f"Windows past ~{_fmt(SPAN_WARNING_DAYS, 0)} d recovered nothing on 471085 at the as-run step, no E2 window "
            "exceeded 9 d, and the trajectory count grows as span^2; the regime above 8 d is unmeasured."
        )
    if region.retrograde_looking and region.cancellation_distances_au:
        cancels = region.cancellation_distances_au
        where = _fmt(cancels[0], 1) if len(cancels) == 1 else f"{_fmt(min(cancels), 1)}-{_fmt(max(cancels), 1)}"
        warnings.append(
            f"The cancellation distance ({where} au over the elongation range) lies inside the coverage: objects just "
            "inside it move slower than any velocity floor and objects farther inside look retrograde. A +/-90 deg fan "
            "excludes them by construction; prefer two nodes."
        )
    elif region.retrograde_looking:
        warnings.append(
            f"Objects near {_fmt(spec.distance_min_au, 1)} au with inclinations near {_fmt(spec.inclination_max_deg, 0)} deg "
            "look retrograde in this frame, so the speed floor of the region is zero and a fan would need 360 degrees. "
            "Lower the inclination ceiling or move the near edge of the coverage outward if that corner is not wanted."
        )
    for lo, hi in blind:
        if hi > spec.distance_min_au and lo < spec.distance_max_au:
            warnings.append(
                f"With a velocity floor of {_fmt(floor, 1)} px/d, ecliptic objects at {_fmt(lo, 1)}-{_fmt(hi, 1)} au move too "
                "slowly to be searched (the 25 px/d production floor at 70 au hid ~52-68 au and 2020 KV11)."
            )
    if spec.guess_distance_au > 60.0:
        warnings.append(
            "The pool-stage settings were measured at the 42 au node; the 70 au node has no controlled cell yet, and its "
            "magnitudes carry little brightness information near the detection limit."
        )
    if spec.guess_distance_au < 10.0:
        warnings.append(
            "Trojan-regime searches are single-night (26-minute DP1 baselines); the window and rpp policy here was measured on "
            "multi-night 42 au windows, and the believability gate removed 72-81% of DP1 Trojan knowns."
        )
    if spec.inclination_max_deg >= 45.0:
        warnings.append(
            "An inclination ceiling this high makes the slow end of the region almost isotropic in angle; on 471085 the "
            "objects beyond +/-45 deg were 6.5% of the population and all slower than ~80 px/d."
        )
    warnings.append(
        "Model limits: circular barycentric orbits, a field on the ecliptic (off-ecliptic fields rotate the residual "
        "direction), Earth's orbit as a circle, and no light-time or aberration; validated to 1-2% in speed on classical "
        "objects and within 30% on high-inclination ones."
    )

    return Recommendation(
        spec=spec,
        region=region,
        span_days=span_days,
        psf_fwhm_pixels=psf_fwhm_pixels,
        tolerance_preset=tolerance_preset,
        tolerance_px=tol,
        rules=rules,
        designs=designs,
        recommended_kind=kind,
        recommendation_reason=reason,
        coherence=coherence,
        settings=settings,
        warnings=tuple(warnings),
        blind_zones_au=blind,
    )


# --------------------------------------------------------------------------- output
def search_config_mapping(recommendation: Recommendation, design: GridDesign, tier_index: int = 0) -> dict[str, Any]:
    """A complete kbmod search configuration mapping for one grid of a design."""

    generator = design.generator_configs()[tier_index]
    mapping: dict[str, Any] = {"generator_config": generator}
    for line in recommendation.settings:
        mapping[line.key] = line.value
    return mapping


def search_config_yaml(recommendation: Recommendation, design: GridDesign, tier_index: int = 0) -> str:
    """kbmod search YAML with a provenance header and a reason comment on every setting."""

    spec = recommendation.spec
    generator = design.generator_configs()[tier_index]
    tier_label = ""
    if design.tiers and len(design.tiers) > 1:
        tier_label = f" ({design.tiers[tier_index].label})"
    header = [
        f"# KBMOD search configuration: {design.title}{tier_label}",
        f"# Reflex node {_fmt(spec.guess_distance_au, 2)} au; coverage {_fmt(spec.distance_min_au, 2)}-{_fmt(spec.distance_max_au, 2)} au, "
        f"inclination <= {_fmt(spec.inclination_max_deg, 1)} deg, elongation {_fmt(spec.elongation_min_deg, 0)}-{_fmt(spec.elongation_max_deg, 0)} deg.",
        f"# Window {_fmt(recommendation.span_days, 2)} d; PSF FWHM {_fmt(recommendation.psf_fwhm_pixels, 2)} px; step tolerance "
        f"{_fmt(recommendation.tolerance_px, 2)} px ({recommendation.tolerance_preset}); pixel scale {_fmt(spec.pixel_scale_arcsec_per_pixel, 3)} arcsec/px.",
        f"# Trajectories per start pixel: {design.trajectories:,} ({design.cost_vs_production:.2f}x the 64x64 production grid).",
        "# The reflex node is set where the WorkUnit is reprojected (helio_guess_dists), not here.",
        "# Generated by kbmod_angle_rate_tool.reflex_search_space; every setting carries its measured reason.",
        "",
        "generator_config:",
    ]
    lines = header
    for key, value in generator.items():
        lines.append(f"  {key}: {_yaml_scalar(value)}")
    lines.append("")
    for line in recommendation.settings:
        lines.append(f"# {line.why}")
        lines.append(f"{line.key}: {_yaml_scalar(line.value)}")
    return "\n".join(lines) + "\n"


def format_recommendation_report(recommendation: Recommendation) -> str:
    """Plain-text report of the region, coherence, designs, recommendation and warnings."""

    spec, region, coh = recommendation.spec, recommendation.region, recommendation.coherence
    lines = [
        "KBMOD reflex-frame search recommendation",
        "",
        "Coverage requested",
        f"  reflex node: {_fmt(spec.guess_distance_au, 2)} au",
        f"  true distances: {_fmt(spec.distance_min_au, 2)} to {_fmt(spec.distance_max_au, 2)} au (+/-{_fmt(spec.half_width_au, 2)} au)",
        f"  inclination: <= {_fmt(spec.inclination_max_deg, 1)} deg",
        f"  solar elongation: {_fmt(spec.elongation_min_deg, 0)} to {_fmt(spec.elongation_max_deg, 0)} deg (~{_fmt(180 - spec.elongation_min_deg, 0)} d from opposition at worst)",
        f"  window span: {_fmt(recommendation.span_days, 2)} d",
        f"  pixel scale: {_fmt(spec.pixel_scale_arcsec_per_pixel, 3)} arcsec/px; PSF FWHM {_fmt(recommendation.psf_fwhm_pixels, 2)} px",
        f"  step tolerance: {_fmt(recommendation.tolerance_px, 2)} px ({TOLERANCE_PRESETS[recommendation.tolerance_preset][0]})",
        "",
        "Occupied region in the reflex frame",
        f"  orbital rate omega: {_fmt(orbital_rate_pixels_per_day(spec.distance_min_au, spec.pixel_scale_arcsec_per_pixel), 1)} px/d at "
        f"{_fmt(spec.distance_min_au, 1)} au, {_fmt(orbital_rate_pixels_per_day(spec.guess_distance_au, spec.pixel_scale_arcsec_per_pixel), 1)} at the node, "
        f"{_fmt(orbital_rate_pixels_per_day(spec.distance_max_au, spec.pixel_scale_arcsec_per_pixel), 1)} at {_fmt(spec.distance_max_au, 1)} au",
        f"  along-ecliptic rate v_along: {_fmt(region.v_along_min, 1)} to {_fmt(region.v_along_max, 1)} px/d",
        f"  cross-ecliptic rate |v_perp|: up to {_fmt(region.v_perp_max, 1)} px/d",
        f"  speed: {_fmt(region.speed_min, 1)} to {_fmt(region.speed_max, 1)} px/d",
        f"  widest angle from the ecliptic: {_fmt(region.angle_max_deg, 1)} deg"
        + (" (objects look retrograde inside the cancellation distance)" if region.retrograde_looking else ""),
    ]
    cancels = [
        cancellation_distance_au(spec.guess_distance_au, e, spec.pixel_scale_arcsec_per_pixel)
        for e in (spec.elongation_max_deg, spec.elongation_min_deg)
    ]
    cancel_text = ", ".join(_fmt(c, 1) + " au" for c in cancels if c is not None) or "none (quadrature)"
    lines.append(f"  cancellation distance (v_along = 0): {cancel_text} at elongation {_fmt(spec.elongation_max_deg, 0)}/{_fmt(spec.elongation_min_deg, 0)} deg")
    floor = recommendation.recommended.velocity_floor
    if recommendation.blind_zones_au:
        zones = "; ".join(f"{_fmt(lo, 1)}-{_fmt(hi, 1)} au" for lo, hi in recommendation.blind_zones_au)
        lines.append(f"  velocity floor {_fmt(floor, 1)} px/d hides ecliptic objects at {zones} (opposition)")
    else:
        lines.append(f"  velocity floor {_fmt(floor, 1)} px/d hides no ecliptic object between {_fmt(0.3 * spec.guess_distance_au, 0)} and {_fmt(3 * spec.guess_distance_au, 0)} au")
    lines.extend(
        [
            "",
            "Coherence (track within tolerance of a straight line)",
            f"  longest coherent window at the coverage edges, elongation {_fmt(spec.worst_elongation_deg, 0)} deg: "
            f"{_fmt(coh.max_span_near_edge_days, 1)} d at {_fmt(spec.distance_min_au, 1)} au, {_fmt(coh.max_span_far_edge_days, 1)} d at {_fmt(spec.distance_max_au, 1)} au",
            f"  distances coherent at {_fmt(recommendation.span_days, 1)} d from this node: {_fmt(coh.coherent_distance_min_au, 1)} to {_fmt_or_inf(coh.coherent_distance_max_au, 1)} au",
            f"  nodes needed for the full coverage at this span: {len(coh.ladder_nodes_au)} ({', '.join(_fmt(n, 1) for n in coh.ladder_nodes_au)} au)",
            "  longest coherent window by elongation (near edge / far edge):",
        ]
    )
    for e, t_near, t_far in coh.span_table:
        lines.append(f"    {_fmt(e, 0):>4} deg: {_fmt(t_near, 1):>6} d / {_fmt(t_far, 1):>6} d")
    lines.extend(["", "Grid designs (steps from dv <= 2 tol / T, dtheta <= 2 tol / (v T))"])
    for design in recommendation.designs:
        marker = "  * " if design.kind == recommendation.recommended_kind else "    "
        lines.append(f"{marker}{design.title}: {design.trajectories:,} trajectories/pixel "
                     f"({design.cost_vs_production:.2f}x production 64x64, {design.cost_vs_option_a:.2f}x Option A 256x192)")
        for tier in design.tiers:
            lines.append(
                f"        {tier.label}: velocities [{_fmt(tier.velocity_min, 1)}, {_fmt(tier.velocity_max, 1)}] px/d x {tier.velocity_samples} "
                f"(step {_fmt(tier.velocity_step, 3)}), angles +/-{_fmt(tier.angle_half_width_deg, 1)} deg x {tier.angle_samples} "
                f"(step {_fmt(tier.angle_step_deg, 3)} deg)"
            )
        if design.box is not None:
            b = design.box
            lines.append(
                f"        ecliptic-frame box: v_along [{_fmt(b.ecliptic_box[0], 1)}, {_fmt(b.ecliptic_box[1], 1)}], "
                f"v_perp [{_fmt(b.ecliptic_box[2], 1)}, {_fmt(b.ecliptic_box[3], 1)}] px/d ({b.unpadded_trajectories:,} trajectories unpadded)"
            )
            lines.append(
                f"        pixel-frame box after {_fmt(b.tilt_deg, 1)} deg tilt padding: vx [{_fmt(b.vx_min, 1)}, {_fmt(b.vx_max, 1)}] x {b.vx_samples}, "
                f"vy [{_fmt(b.vy_min, 1)}, {_fmt(b.vy_max, 1)}] x {b.vy_samples} (step {_fmt(b.step, 3)})"
            )
        for note in design.notes:
            lines.append(f"        - {note}")
    lines.extend(
        [
            "",
            f"Recommended: {recommendation.recommended.title}",
            f"  {recommendation.recommendation_reason}",
            "",
            "Pool-stage settings (search once, decide everything else offline)",
        ]
    )
    for line in recommendation.settings:
        lines.append(f"  {line.key}: {_yaml_scalar(line.value)}  # {line.why}")
    lines.extend(["", "Measured context (42 au injections on 471085; the model has no population prior)"])
    lines.append("  fixed fan holds: " + ", ".join(f"+/-{_fmt(a, 0)} deg {frac:.1%}" for a, frac in MEASURED_FAN_FRACTIONS_42AU))
    lines.append("  |v_perp| box holds: " + ", ".join(f"<= {_fmt(v, 0)} px/d {frac:.1%}" for v, frac in MEASURED_VPERP_FRACTIONS_42AU))
    lines.extend(["", "Warnings and limits"])
    for warning in recommendation.warnings:
        lines.append(f"  - {warning}")
    return "\n".join(lines)


def design_file_stem(recommendation: Recommendation, design: GridDesign, tier_index: int = 0) -> str:
    spec = recommendation.spec
    tier = f"_{design.tiers[tier_index].label.replace(' ', '_')}" if len(design.tiers) > 1 else ""
    return (
        f"kbmod_search_{_token(spec.guess_distance_au)}au_"
        f"{_token(spec.distance_min_au)}to{_token(spec.distance_max_au)}au_"
        f"i{_token(spec.inclination_max_deg)}_{_token(recommendation.span_days)}d_{design.kind}{tier}"
    )


def build_recommendation_zip(recommendation: Recommendation, figures: Sequence[tuple[str, Any]] = ()) -> bytes:
    """ZIP of every design's YAML, the report and any (name, matplotlib figure) pairs given."""

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("recommendation_report.txt", format_recommendation_report(recommendation))
        for design in recommendation.designs:
            for index in range(len(design.generator_configs())):
                archive.writestr(f"{design_file_stem(recommendation, design, index)}.yaml", search_config_yaml(recommendation, design, index))
        for name, fig in figures:
            png = io.BytesIO()
            fig.savefig(png, format="png", dpi=160, bbox_inches="tight")
            archive.writestr(name, png.getvalue())
    return buffer.getvalue()


# --------------------------------------------------------------------------- plots
def build_velocity_space_plot(recommendation: Recommendation, *, show_production_reference: bool = True) -> Any:
    """Occupied region in (v_along, v_perp) with the three grid designs drawn over it."""

    _ensure_matplotlib_config_dir()
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle, Wedge

    region = recommendation.region
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    xs = [s.v_along for s in region.samples]
    ys = [s.v_perp for s in region.samples]
    ds = [s.distance_au for s in region.samples]
    scatter = ax.scatter(xs, ys, c=ds, cmap="viridis", s=5, alpha=0.6, linewidths=0, zorder=3)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.01, fraction=0.04)
    cbar.set_label("true barycentric distance (au)")

    # The ecliptic (i = 0) locus at opposition with distance ticks.
    spec = recommendation.spec
    zero = sorted((s for s in region.samples if s.inclination_deg == 0.0 and math.isclose(s.elongation_deg, spec.elongation_max_deg)), key=lambda s: s.distance_au)
    if zero:
        ax.plot([s.v_along for s in zero], [0.0] * len(zero), color="black", linewidth=1.2, zorder=4)
        for s in (zero[0], zero[-1]):
            ax.annotate(f"{_fmt(s.distance_au, 1)} au", (s.v_along, 0.0), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=8)
        node = min(zero, key=lambda s: abs(s.distance_au - spec.guess_distance_au))
        ax.plot([node.v_along], [0.0], marker="o", color="black", markersize=5, zorder=5)
        ax.annotate("node", (node.v_along, 0.0), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)

    colors = {"single_fan": "#d95f02", "two_tier": "#1b9e77", "velocity_box": "#7570b3"}
    extents_x = list(xs)
    extents_y = list(ys)
    for design in recommendation.designs:
        color = colors[design.kind]
        width = 2.4 if design.kind == recommendation.recommended_kind else 1.3
        style = "-" if design.kind == recommendation.recommended_kind else "--"
        label = design.title + (" (recommended)" if design.kind == recommendation.recommended_kind else "")
        first = True
        for tier in design.tiers:
            half = min(tier.angle_half_width_deg, 180.0)
            wedge = Wedge(
                (0.0, 0.0),
                tier.velocity_max,
                -half,
                half,
                width=tier.velocity_max - tier.velocity_min,
                fill=False,
                edgecolor=color,
                linewidth=width,
                linestyle=style,
                label=label if first else "_nolegend_",
                zorder=6,
            )
            ax.add_patch(wedge)
            extents_x.extend([-tier.velocity_max if half > 90 else 0.0, tier.velocity_max])
            extents_y.extend([-tier.velocity_max * math.sin(math.radians(min(half, 90.0))), tier.velocity_max * math.sin(math.radians(min(half, 90.0)))])
            first = False
        if design.box is not None:
            b = design.box
            ax.add_patch(
                Rectangle(
                    (b.ecliptic_box[0], b.ecliptic_box[2]),
                    b.ecliptic_box[1] - b.ecliptic_box[0],
                    b.ecliptic_box[3] - b.ecliptic_box[2],
                    fill=False,
                    edgecolor=color,
                    linewidth=width,
                    linestyle=style,
                    label=label,
                    zorder=6,
                )
            )
            # The padded pixel-frame box, rotated back into the ecliptic frame.
            phi = math.radians(-b.tilt_deg)
            corners = [(b.vx_min, b.vy_min), (b.vx_max, b.vy_min), (b.vx_max, b.vy_max), (b.vx_min, b.vy_max)]
            rotated = [(x * math.cos(phi) - y * math.sin(phi), x * math.sin(phi) + y * math.cos(phi)) for x, y in corners]
            ax.add_patch(Polygon(rotated, closed=True, fill=False, edgecolor=color, linewidth=0.9, linestyle=":", label="_nolegend_", zorder=6))
            extents_x.extend(x for x, _ in rotated)
            extents_y.extend(y for _, y in rotated)
    if show_production_reference:
        ax.add_patch(
            Wedge((0.0, 0.0), 225.0, -90, 90, width=200.0, fill=False, edgecolor="#9e9e9e", linewidth=1.0, linestyle=":", label="production 64x64: [25,225] px/d, +/-90 deg", zorder=2)
        )

    span_x = max(extents_x) - min(extents_x)
    span_y = max(extents_y) - min(extents_y)
    pad = 0.08 * max(span_x, span_y, 1.0)
    ax.set_xlim(min(extents_x) - pad, max(extents_x) + pad)
    ax.set_ylim(min(extents_y) - pad, max(extents_y) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0.0, color="#bbbbbb", linewidth=0.8, zorder=1)
    ax.axvline(0.0, color="#bbbbbb", linewidth=0.8, zorder=1)
    ax.set_xlabel("v_along: along the ecliptic, eastward (px/day)")
    ax.set_ylabel("v_perp: across the ecliptic (px/day)")
    ax.set_title(
        f"Reflex-frame velocity space at the {_fmt(spec.guess_distance_au, 1)} au node: "
        f"{_fmt(spec.distance_min_au, 1)}-{_fmt(spec.distance_max_au, 1)} au, i <= {_fmt(spec.inclination_max_deg, 0)} deg, "
        f"elongation {_fmt(spec.elongation_min_deg, 0)}-{_fmt(spec.elongation_max_deg, 0)} deg"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.12), fontsize=8, ncol=2, framealpha=0.9)
    fig.tight_layout()
    return fig


def build_coherence_plot(recommendation: Recommendation) -> Any:
    """Track departure from a straight line versus window span at the coverage edges."""

    _ensure_matplotlib_config_dir()
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    spec = recommendation.spec
    scale = spec.pixel_scale_arcsec_per_pixel
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    spans = [0.25 * k for k in range(1, 161)]
    for d, color in ((spec.distance_min_au, "#1f77b4"), (spec.distance_max_au, "#d62728")):
        for e, style in ((spec.elongation_max_deg, "-"), (spec.elongation_min_deg, "--")):
            deps = [track_departure_pixels(t, d, spec.guess_distance_au, e, scale) for t in spans]
            ax.plot(spans, deps, color=color, linestyle=style, label=f"{_fmt(d, 1)} au, elongation {_fmt(e, 0)} deg")
    fwhm = recommendation.psf_fwhm_pixels
    ax.axhline(fwhm / GAUSSIAN_FWHM_PER_SIGMA, color="#555555", linewidth=0.8, linestyle=":", label="1 PSF sigma")
    ax.axhline(0.5 * fwhm, color="#555555", linewidth=0.8, linestyle="--", label="half FWHM")
    ax.axhline(fwhm, color="#555555", linewidth=0.8, linestyle="-", label="1 PSF FWHM")
    ax.axhline(recommendation.tolerance_px, color="#2ca02c", linewidth=1.6, label=f"chosen tolerance {_fmt(recommendation.tolerance_px, 2)} px")
    ax.axvline(recommendation.span_days, color="#2ca02c", linewidth=1.2, linestyle="-.", label=f"chosen span {_fmt(recommendation.span_days, 1)} d")
    ax.set_yscale("log")
    ax.set_ylim(0.05, max(50.0, 4.0 * fwhm))
    ax.set_xlim(0.0, spans[-1])
    ax.set_xlabel("window span (days)")
    ax.set_ylabel("max departure from a straight line (px)")
    ax.set_title(f"Curvature in the {_fmt(spec.guess_distance_au, 1)} au reflex frame at the coverage edges")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recommend a KBMOD search configuration for a reflex node and a distance coverage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--node", type=float, required=True, help="Reflex-correction guess distance in au.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--half-width", type=float, default=None, help="Cover node +/- this many au.")
    group.add_argument("--distance-range", type=float, nargs=2, metavar=("MIN", "MAX"), default=None, help="Cover this true-distance range in au.")
    parser.add_argument("--inclination", type=float, default=DEFAULT_INCLINATION_MAX_DEG, help="Largest inclination to cover, degrees.")
    parser.add_argument("--elongation", type=float, nargs=2, metavar=("MIN", "MAX"), default=DEFAULT_ELONGATION_RANGE_DEG, help="Solar elongation range of the observations, degrees.")
    parser.add_argument("--span", type=float, default=DEFAULT_SPAN_DAYS, help="Window span in days.")
    parser.add_argument("--pixel-scale", type=float, default=DEFAULT_PIXEL_SCALE_ARCSEC_PER_PIXEL, help="arcsec per pixel.")
    parser.add_argument("--psf-fwhm-px", type=float, default=DEFAULT_PSF_FWHM_PIXELS, help="PSF FWHM in pixels.")
    parser.add_argument("--tolerance", choices=sorted(TOLERANCE_PRESETS), default=DEFAULT_TOLERANCE_PRESET, help="Step and curvature tolerance preset.")
    parser.add_argument("--tilt", type=float, default=DEFAULT_PIXEL_FRAME_TILT_DEG, help="Ecliptic-to-pixel tilt to pad the velocity box for, degrees.")
    parser.add_argument("--velocity-margin", type=float, default=DEFAULT_VELOCITY_MARGIN_FRACTION, help="Fractional margin on the velocity bounds.")
    parser.add_argument("--min-half-width", type=float, default=DEFAULT_MIN_ANGLE_HALF_WIDTH_DEG, help="Smallest fan half-width, degrees.")
    parser.add_argument("--out", type=Path, default=None, help="Directory for the YAML files, report and plots.")
    return parser


def run(argv: Sequence[str] | None = None) -> Recommendation:
    args = build_parser().parse_args(argv)
    if args.distance_range is not None:
        lo, hi = args.distance_range
        spec = CoverageSpec(args.node, lo, hi, args.inclination, args.elongation[0], args.elongation[1], args.pixel_scale)
    else:
        half = args.half_width if args.half_width is not None else 5.0
        spec = CoverageSpec.symmetric(
            args.node,
            half,
            inclination_max_deg=args.inclination,
            elongation_min_deg=args.elongation[0],
            elongation_max_deg=args.elongation[1],
            pixel_scale_arcsec_per_pixel=args.pixel_scale,
        )
    rules = DesignRules(
        velocity_margin_fraction=args.velocity_margin,
        min_angle_half_width_deg=args.min_half_width,
        pixel_frame_tilt_deg=args.tilt,
    )
    recommendation = recommend_search_configuration(
        spec, span_days=args.span, psf_fwhm_pixels=args.psf_fwhm_px, tolerance_preset=args.tolerance, rules=rules
    )
    report = format_recommendation_report(recommendation)
    print(report)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "recommendation_report.txt").write_text(report, encoding="utf-8")
        for design in recommendation.designs:
            for index in range(len(design.generator_configs())):
                (args.out / f"{design_file_stem(recommendation, design, index)}.yaml").write_text(
                    search_config_yaml(recommendation, design, index), encoding="utf-8"
                )
        import matplotlib.pyplot as plt

        for name, builder in (("velocity_space.png", build_velocity_space_plot), ("coherence.png", build_coherence_plot)):
            fig = builder(recommendation)
            fig.savefig(args.out / name, dpi=160, bbox_inches="tight")
            plt.close(fig)
        print(f"\nWrote {args.out}")
    return recommendation


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(argv)
    except Exception as exc:  # pragma: no cover - CLI surface
        print(f"kbmod-reflex-recommend: error: {exc}", file=sys.stderr)
        return 1
    return 0


# --------------------------------------------------------------------------- helpers
def _require_positive(name: str, value: float) -> None:
    if not (value > 0.0):
        raise ValueError(f"{name} must be positive")


def _round(value: float, digits: int = 3) -> float:
    return float(round(value, digits))


def _fmt(value: float, digits: int = 2) -> str:
    text = f"{value:.{digits}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def _fmt_or_inf(value: float, digits: int = 2) -> str:
    return "inf" if math.isinf(value) else _fmt(value, digits)


def _token(value: float) -> str:
    return _fmt(value, 2).replace(".", "p").replace("-", "m")


def _yaml_scalar(value: Any) -> str:
    return yaml.safe_dump(value, default_flow_style=True).strip().removesuffix("\n...").strip()


def _ensure_matplotlib_config_dir() -> None:
    import os

    if "MPLCONFIGDIR" in os.environ:
        return
    config_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "kbmod_angle_rate_tool_matplotlib"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(config_dir)


if __name__ == "__main__":
    raise SystemExit(main())
