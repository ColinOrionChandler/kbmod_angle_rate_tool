"""Regression tests for the reflex-frame model against the 2026-09-18 audit anchors.

Every expected number below is taken from the audit's validated tables
(``docs/audit_2026-09-18/physical-search-space/out/`` in kbmod-mission-control) or from
the 2026-08-18 70 au reflex analysis (``analysis/reflex70/RESULTS_SUMMARY.md`` in
kbmod_coc_tools), both of which were checked against kbmod's own reflex-corrected tracks.
"""

from __future__ import annotations

import io
import math
import zipfile

import pytest
import yaml

from kbmod_angle_rate_tool import reflex_search_space as reflex


def test_constants_match_audit():
    assert reflex.parallax_rate_constant() == pytest.approx(17741.0, abs=1.0)
    assert reflex.radians_to_pixels(0.2) == pytest.approx(1_031_324.0, rel=1e-6)


@pytest.mark.parametrize(
    ("distance", "expected"),
    [(5.2, 1496.0), (17.0, 253.0), (28.0, 120.0), (42.0, 65.0), (50.0, 50.0), (70.0, 30.0), (100.0, 18.0)],
)
def test_orbital_rate_matches_audit_table(distance, expected):
    assert reflex.orbital_rate_pixels_per_day(distance) == pytest.approx(expected, abs=0.6)


@pytest.mark.parametrize(
    ("distance", "expected"),
    [(50.0, -51.3), (65.0, 14.4), (70.0, 30.3), (80.0, 56.3), (100.0, 93.2), (150.0, 143.4)],
)
def test_along_ecliptic_rate_reproduces_70au_opposition_table(distance, expected):
    # analysis/reflex70 (real ephemerides): objects inside ~60 au look retrograde (sign negative here).
    # The circular-Earth analytic model matches to about 1%.
    assert reflex.along_ecliptic_rate(distance, 70.0, 180.0) == pytest.approx(expected, rel=0.012, abs=0.3)


def test_quadrature_removes_the_parallax_mismatch():
    assert reflex.parallax_mismatch_rate(50.0, 70.0, 90.0) == pytest.approx(0.0, abs=1e-9)
    assert reflex.along_ecliptic_rate(50.0, 70.0, 90.0) == pytest.approx(reflex.orbital_rate_pixels_per_day(50.0))


def test_inclination_splits_the_orbital_rate_into_components():
    omega = reflex.orbital_rate_pixels_per_day(42.0)
    along = reflex.along_ecliptic_rate(42.0, 42.0, 180.0, inclination_deg=30.0)
    perp = reflex.cross_ecliptic_rate_max(42.0, 30.0)
    assert along == pytest.approx(omega * math.cos(math.radians(30.0)))
    assert perp == pytest.approx(omega * 0.5)
    assert math.hypot(along, perp) == pytest.approx(omega)


def test_cancellation_distances():
    assert reflex.cancellation_distance_au(70.0, 180.0) == pytest.approx(61.0, abs=0.2)
    assert reflex.cancellation_distance_au(42.0, 180.0) == pytest.approx(34.9, abs=0.3)
    assert reflex.cancellation_distance_au(70.0, 90.0) is None


def test_blind_zone_at_the_production_floor():
    zones = reflex.blind_zones_au(70.0, 180.0, 25.0)
    assert len(zones) == 1
    lo, hi = zones[0]
    assert lo == pytest.approx(55.1, abs=0.3)
    assert hi == pytest.approx(68.3, abs=0.3)
    assert reflex.blind_zones_au(70.0, 180.0, 0.0) == ()


@pytest.mark.parametrize(
    ("span", "elongation", "expected"),
    [(24.0, 180.0, 3.41), (24.0, 150.0, 43.46), (7.0, 180.0, 0.10), (40.0, 150.0, 125.72)],
)
def test_departure_reproduces_70au_curvature_tables(span, elongation, expected):
    # analysis/reflex70: pile centred at opposition and 30 d before it, object at 50 au.
    assert reflex.track_departure_pixels(span, 50.0, 70.0, elongation) == pytest.approx(expected, rel=0.06, abs=0.05)


def test_object_at_the_node_is_linear():
    assert reflex.track_departure_pixels(60.0, 70.0, 70.0, 150.0) == 0.0
    assert reflex.max_coherent_span_days(70.0, 70.0, 150.0, 2.27) == 400.0


@pytest.mark.parametrize(
    ("guess", "true", "elongations", "expected"),
    [
        (42.0, 44.0, (180.0, 160.0, 150.0, 135.0), (36.3, 15.0, 12.6, 10.7)),
        (42.0, 50.0, (180.0, 160.0, 150.0, 135.0), (23.9, 8.1, 6.8, 5.7)),
        (70.0, 78.0, (180.0, 160.0, 150.0, 135.0), (32.8, 13.0, 10.9, 9.2)),
        (70.0, 72.0, (180.0, 150.0, 135.0), (50.8, 20.6, 17.6)),
    ],
)
def test_max_coherent_span_matches_audit_coherent_window_table(guess, true, elongations, expected):
    for elongation, value in zip(elongations, expected):
        assert reflex.max_coherent_span_days(true, guess, elongation, 2.27) == pytest.approx(value, abs=0.1)


def test_coherent_distance_range_inverts_the_span_rule():
    lo, hi = reflex.coherent_distance_range(42.0, 7.0, 150.0, 2.27)
    assert lo < 42.0 < hi
    # Both edges sit exactly at the coherence limit for this span.
    assert reflex.max_coherent_span_days(lo, 42.0, 150.0, 2.27) == pytest.approx(7.0, abs=0.05)
    assert reflex.max_coherent_span_days(hi, 42.0, 150.0, 2.27) == pytest.approx(7.0, abs=0.05)
    _, far = reflex.coherent_distance_range(42.0, 0.5, 180.0, 5.35)
    assert math.isinf(far)


def test_required_steps_match_audit_required_steps_table():
    assert reflex.required_velocity_step(7.0, 2.27) == pytest.approx(0.649, abs=0.001)
    assert reflex.required_velocity_step(14.0, 2.27) == pytest.approx(0.324, abs=0.001)
    assert reflex.required_angle_step_deg(7.0, 2.27, 100.0) == pytest.approx(0.372, abs=0.001)
    assert reflex.required_angle_step_deg(7.0, 2.27, 25.0) == pytest.approx(1.486, abs=0.001)
    assert reflex.required_angle_step_deg(14.0, 2.27, 170.0) == pytest.approx(0.109, abs=0.001)


def test_inclusive_sample_count_rounds_up_without_float_noise():
    assert reflex.inclusive_sample_count(150.0, 0.649) == 233
    assert reflex.inclusive_sample_count(1.0, 0.1) == 11
    assert reflex.inclusive_sample_count(0.0, 1.0) == 1


def test_tolerance_presets_scale_with_the_psf():
    assert reflex.tolerance_pixels("sigma", 5.35) == pytest.approx(2.272, abs=0.002)
    assert reflex.tolerance_pixels("half_fwhm", 5.35) == pytest.approx(2.675)
    assert reflex.tolerance_pixels("fwhm", 5.35) == pytest.approx(5.35)
    with pytest.raises(ValueError):
        reflex.tolerance_pixels("nope", 5.35)


@pytest.mark.parametrize(
    ("window", "elongation", "expected_nodes"),
    [
        (7.0, 150.0, (32.9, 40.7, 53.3, 77.4)),
        (7.0, 135.0, (32.3, 38.0, 46.2, 58.8, 81.1)),
        (14.0, 165.0, (31.4, 34.5, 38.4, 43.2, 49.5, 57.8, 69.6, 87.3)),
        (4.0, 150.0, (36.4, 63.2)),
    ],
)
def test_distance_ladder_matches_audit_table(window, elongation, expected_nodes):
    nodes = reflex.distance_ladder_nodes(30.0, 100.0, window, elongation, 2.27)
    assert len(nodes) == len(expected_nodes)
    for node, expected in zip(nodes, expected_nodes):
        assert node == pytest.approx(expected, abs=0.1)


def test_occupied_region_reproduces_audit_42au_row_for_ecliptic_objects():
    # occupied_region_by_node.csv: 42 au, d_true 35.7-50.4, elongation 180, i = 0 -> v_along 8.6..120.0.
    spec = reflex.CoverageSpec(42.0, 35.7, 50.4, inclination_max_deg=0.0, elongation_min_deg=180.0, elongation_max_deg=180.0)
    region = reflex.occupied_region(spec)
    assert region.v_along_min == pytest.approx(8.6, abs=0.2)
    assert region.v_along_max == pytest.approx(120.0, abs=0.2)
    assert region.v_perp_max == 0.0
    assert region.angle_max_deg == 0.0
    assert not region.retrograde_looking


def test_occupied_region_reproduces_audit_70au_row():
    # 70 au, d_true 59.5-84.0, elongation 180: v_along -6.1..65.3, cancellation 61.2, retrograde-looking.
    spec = reflex.CoverageSpec(70.0, 59.5, 84.0, inclination_max_deg=0.0, elongation_min_deg=180.0, elongation_max_deg=180.0)
    region = reflex.occupied_region(spec)
    assert region.v_along_min == pytest.approx(-6.1, abs=0.3)
    assert region.v_along_max == pytest.approx(65.3, abs=0.3)
    assert region.retrograde_looking
    assert region.speed_min == 0.0
    assert len(region.cancellation_distances_au) == 1
    assert region.cancellation_distances_au[0] == pytest.approx(61.0, abs=0.3)


def test_occupied_region_vperp_uses_the_nearest_distance():
    spec = reflex.CoverageSpec.symmetric(70.0, 5.0, inclination_max_deg=30.0)
    region = reflex.occupied_region(spec)
    assert region.v_perp_max == pytest.approx(reflex.cross_ecliptic_rate_max(65.0, 30.0), rel=1e-6)
    assert region.spec.half_width_au == 5.0
    assert region.spec.worst_elongation_deg == 135.0


def test_two_tier_reproduces_the_audit_span_scaled_count():
    # design_rules.py: polar_scaled(-60, 60, 20, 85, T) + polar_scaled(-20, 20, 75, 170, T), SIG 2.27, T = 4 -> 18,332.
    rules = reflex.DesignRules(min_angle_half_width_deg=0.0)
    slow = reflex._fan_tier("slow", 20.0, 85.0, 60.0, 4.0, 2.27, rules)
    fast = reflex._fan_tier("fast", 75.0, 170.0, 20.0, 4.0, 2.27, rules)
    assert slow.trajectories + fast.trajectories == 18_332
    single = reflex._fan_tier("fan", 20.0, 170.0, 60.0, 4.0, 2.27, rules)
    assert single.trajectories == 42_210


def test_box_sample_counts_reproduce_the_audit_cartesian_grid():
    # cart_scaled(20, 170, -50, 50, 4) with SIG 2.27 -> 12,060 trajectories.
    step = reflex.required_velocity_step(4.0, 2.27)
    assert reflex.inclusive_sample_count(150.0, step) * reflex.inclusive_sample_count(100.0, step) == 12_060


def test_recommendation_for_70au_plus_minus_5au():
    spec = reflex.CoverageSpec.symmetric(70.0, 5.0, inclination_max_deg=30.0)
    rec = reflex.recommend_search_configuration(spec, span_days=7.0)

    assert rec.tolerance_px == pytest.approx(2.675)
    assert rec.region.v_along_min == pytest.approx(9.8, abs=0.2)
    assert rec.region.v_along_max == pytest.approx(44.2, abs=0.2)
    assert rec.region.speed_min == pytest.approx(14.4, abs=0.2)
    assert rec.coherence.coverage_is_coherent
    assert rec.coherence.max_span_at_edges_days == pytest.approx(11.5, abs=0.2)
    assert rec.coherence.ladder_nodes_au == pytest.approx((69.6,), abs=0.2)
    kinds = [design.kind for design in rec.designs]
    assert kinds == ["single_fan", "two_tier", "velocity_box"]
    single, two_tier, box = rec.designs
    # The fan must not start at the production floor: the shell's slowest object runs at 14.4 px/d.
    assert single.tiers[0].velocity_min < 14.4
    assert single.tiers[0].velocity_max > 44.2
    assert two_tier.trajectories < single.trajectories
    assert box.box is not None and box.box.trajectories > box.box.unpadded_trajectories
    assert rec.recommended_kind == "two_tier"
    assert rec.recommended is two_tier
    assert rec.design("velocity_box") is box
    settings = {line.key: line.value for line in rec.settings}
    assert settings["results_per_pixel"] == 64
    assert settings["do_clustering"] is False
    assert settings["max_results"] == -1
    assert settings["stamp_type"] is None
    assert any("70 au node has no controlled cell" in w for w in rec.warnings)
    assert not any("longer than the coherent window" in w for w in rec.warnings)


def test_short_span_raises_results_per_pixel_and_long_span_warns():
    spec = reflex.CoverageSpec.symmetric(42.0, 5.0)
    short = reflex.recommend_search_configuration(spec, span_days=3.0)
    assert {l.key: l.value for l in short.settings}["results_per_pixel"] == 128
    long = reflex.recommend_search_configuration(spec, span_days=14.0)
    assert {l.key: l.value for l in long.settings}["results_per_pixel"] == 64
    assert any("longer than the coherent window" in w for w in long.warnings)
    assert any("past ~8 d" in w for w in long.warnings)
    assert len(long.coherence.ladder_nodes_au) > 1


def test_narrow_region_gets_a_single_fan():
    spec = reflex.CoverageSpec.symmetric(42.0, 1.0, inclination_max_deg=5.0)
    rec = reflex.recommend_search_configuration(spec, span_days=4.0)
    assert rec.recommended_kind == "single_fan"
    assert rec.recommended.tiers[0].angle_half_width_deg <= 45.0
    assert rec.recommended.tiers[0].angle_half_width_deg >= reflex.DEFAULT_MIN_ANGLE_HALF_WIDTH_DEG


def test_coverage_across_the_cancellation_distance_recommends_the_box():
    spec = reflex.CoverageSpec(70.0, 55.0, 75.0, inclination_max_deg=10.0)
    rec = reflex.recommend_search_configuration(spec, span_days=7.0)
    assert rec.region.retrograde_looking
    assert rec.recommended_kind == "velocity_box"
    assert rec.design("single_fan").tiers[0].angle_half_width_deg == 180.0
    assert any("cancellation distance" in w for w in rec.warnings)
    assert any("too slowly to be searched" in w for w in rec.warnings) or rec.recommended.velocity_floor == 0.0


def test_search_config_yaml_round_trips_and_carries_reasons():
    spec = reflex.CoverageSpec.symmetric(70.0, 5.0)
    rec = reflex.recommend_search_configuration(spec, span_days=7.0)
    design = rec.design("two_tier")
    text = reflex.search_config_yaml(rec, design, 1)
    loaded = yaml.safe_load(text)
    assert loaded["generator_config"]["name"] == "EclipticCenteredSearch"
    assert loaded["generator_config"]["angles"][2] == design.tiers[1].angle_samples
    assert loaded["generator_config"]["velocities"][:2] == pytest.approx(
        [design.tiers[1].velocity_min, design.tiers[1].velocity_max], abs=1e-3
    )
    assert loaded["generator_config"]["given_ecliptic"] is None
    assert loaded["results_per_pixel"] == 64
    assert loaded["coadds"] == []
    assert loaded["stamp_type"] is None
    assert "fast tier" in text.splitlines()[0]
    assert "# stamps were 96% of the result bytes" in text
    assert loaded == reflex.search_config_mapping(rec, design, 1)

    box_text = reflex.search_config_yaml(rec, rec.design("velocity_box"))
    box_loaded = yaml.safe_load(box_text)
    assert box_loaded["generator_config"]["name"] == "VelocityGridSearch"
    assert set(box_loaded["generator_config"]) == {"name", "vx_steps", "min_vx", "max_vx", "vy_steps", "min_vy", "max_vy"}


def test_report_and_zip_contain_every_design():
    spec = reflex.CoverageSpec.symmetric(42.0, 5.0)
    rec = reflex.recommend_search_configuration(spec, span_days=7.0)
    report = reflex.format_recommendation_report(rec)
    assert "Recommended:" in report
    assert "Two speed tiers" in report and "Velocity box" in report and "Single fan" in report
    assert "fixed fan holds" in report
    payload = reflex.build_recommendation_zip(rec)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
    assert "recommendation_report.txt" in names
    yaml_names = [n for n in names if n.endswith(".yaml")]
    assert len(yaml_names) == 1 + 2 + 1
    assert any("slow_tier" in n for n in yaml_names) and any("fast_tier" in n for n in yaml_names)


def test_plots_build_without_error():
    spec = reflex.CoverageSpec.symmetric(70.0, 5.0)
    rec = reflex.recommend_search_configuration(spec, span_days=7.0)
    import matplotlib.pyplot as plt

    for builder in (reflex.build_velocity_space_plot, reflex.build_coherence_plot):
        fig = builder(rec)
        assert fig.axes
        plt.close(fig)


def test_cli_writes_outputs(tmp_path, capsys):
    rec = reflex.run(["--node", "70", "--half-width", "5", "--span", "7", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert "KBMOD reflex-frame search recommendation" in out
    assert rec.spec.distance_min_au == 65.0
    written = sorted(p.name for p in tmp_path.iterdir())
    assert "recommendation_report.txt" in written
    assert "velocity_space.png" in written and "coherence.png" in written
    assert sum(name.endswith(".yaml") for name in written) == 4
    assert reflex.main(["--node", "0"]) == 1


def test_spec_validation():
    with pytest.raises(ValueError):
        reflex.CoverageSpec(42.0, 50.0, 40.0)
    with pytest.raises(ValueError):
        reflex.CoverageSpec(42.0, 0.5, 40.0)
    with pytest.raises(ValueError):
        reflex.CoverageSpec(42.0, 40.0, 44.0, elongation_min_deg=60.0)
    with pytest.raises(ValueError):
        reflex.DesignRules(tier_overlap_fraction=0.7)
