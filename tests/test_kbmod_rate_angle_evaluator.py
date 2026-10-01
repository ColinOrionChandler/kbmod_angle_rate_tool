from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

import kbmod_angle_rate_tool.kbmod_rate_angle_evaluator as evaluator


def test_default_ten_day_angle_count_uses_kbmod_grid():
    result = evaluator.evaluate_rate_angle_sampling(10.0)

    assert result.search_grid.angle_samples == 64
    assert result.search_grid.velocity_samples == 64
    assert result.velocity_min_pixels_per_day == 25.0
    assert result.velocity_max_pixels_per_day == 225.0
    assert result.velocity_min_arcsec_per_day == 5.0
    assert result.velocity_max_arcsec_per_day == 45.0
    assert result.primary.distance_arcsec == 450.0
    assert result.primary.distance_pixels == 2250.0
    assert result.primary.interval_count == 707
    assert result.primary.angle_count == 708
    assert result.primary.spacing_deg == pytest.approx(0.25464811853)
    assert result.typical_tno_ssb_velocity_arcsec_per_day == pytest.approx(14.4)
    assert result.typical_tno_ssb_velocity_pixels_per_day == pytest.approx(72.0)
    assert result.typical_tno_ssb_travel_arcsec == pytest.approx(144.0)
    assert result.typical_tno_ssb_travel_pixels == pytest.approx(720.0)
    assert result.typical_trojan_velocity_arcsec_per_day == pytest.approx(312.0)
    assert result.typical_trojan_velocity_pixels_per_day == pytest.approx(1560.0)
    assert result.typical_trojan_travel_arcsec == pytest.approx(3120.0)
    assert result.typical_trojan_travel_pixels == pytest.approx(15600.0)
    assert result.adjacent_angle_endpoint_separation_arcsec == pytest.approx(
        evaluator.chord_separation(450.0, result.search_grid.angle_spacing_deg)
    )
    assert result.adjacent_angle_endpoint_separation_pixels == pytest.approx(
        result.adjacent_angle_endpoint_separation_arcsec / result.pixel_scale_arcsec_per_pixel
    )
    assert result.typical_tno_adjacent_endpoint_separation_arcsec == pytest.approx(
        evaluator.chord_separation(144.0, result.search_grid.angle_spacing_deg)
    )
    full_trojan_chord_arcsec = evaluator.chord_separation(
        3120.0,
        result.search_grid.angle_spacing_deg,
    )
    clipped_trojan_chord_arcsec = evaluator.pixels_to_arcsec(
        evaluator.adjacent_endpoint_separation_for_distance_pixels(
            result,
            result.typical_trojan_travel_pixels,
            target_fraction=evaluator.TROJAN_ADJACENT_TARGET_FRACTION,
            clip_to_patch=True,
        ),
        result.pixel_scale_arcsec_per_pixel,
    )
    assert clipped_trojan_chord_arcsec < full_trojan_chord_arcsec
    assert result.typical_trojan_adjacent_endpoint_separation_arcsec == pytest.approx(
        clipped_trojan_chord_arcsec
    )
    assert result.right_edge_adjacent_angle_separation_pixels == pytest.approx(
        299.26131666055244
    )
    assert result.right_edge_adjacent_angle_separation_arcsec == pytest.approx(
        59.85226333211049
    )


def test_explicit_arcsec_velocity_range_preserves_old_calculation():
    result = evaluator.evaluate_rate_angle_sampling(
        10.0,
        velocity_range_arcsec_per_day=(20.0, 140.0),
    )

    assert result.velocity_min_pixels_per_day == 100.0
    assert result.velocity_max_pixels_per_day == 700.0
    assert result.primary.distance_arcsec == 1400.0
    assert result.primary.distance_pixels == 7000.0
    assert result.primary.interval_count == 2200
    assert result.primary.angle_count == 2201
    assert result.primary.spacing_deg == pytest.approx(0.08185112055)


def test_default_pixel_scale_conversions():
    result = evaluator.evaluate_rate_angle_sampling(10.0)

    assert result.seeing_pixels == 5.0
    assert result.velocity_min_pixels_per_day == 25.0
    assert result.velocity_max_pixels_per_day == 225.0
    assert result.primary.distance_pixels == 2250.0


def test_patch_diagonal_corner_to_corner_diagnostic():
    result = evaluator.evaluate_rate_angle_sampling(10.0)

    assert result.patch_diagonal_arcsec == pytest.approx(1697.05627485)
    assert result.patch_diagonal_pixels == pytest.approx(8485.28137424)
    assert result.patch_diagonal.angle_count == 2667
    assert result.patch_diagonal.spacing_deg == pytest.approx(0.06752372762)
    assert not result.max_motion_reaches_patch_diagonal


def test_strict_full_step_diagnostic():
    result = evaluator.evaluate_rate_angle_sampling(10.0)

    assert result.strict_full_step.angle_count == 1415
    assert result.strict_full_step.spacing_deg == pytest.approx(0.12732405927)


def test_parse_args_resolves_defaults():
    args = evaluator.parse_args(["10"])

    assert args.timespan_days == 10.0
    assert args.pixel_scale == evaluator.DEFAULT_PIXEL_SCALE_ARCSEC_PER_PIXEL
    assert args.seeing == evaluator.DEFAULT_SEEING_ARCSEC
    assert tuple(args.patch_size_arcmin) == evaluator.DEFAULT_PATCH_SIZE_ARCMIN
    assert args.plot_output is evaluator.DEFAULT_PLOT_OUTPUT is None
    assert args.no_report_file is False
    assert args.ecliptic_angle_deg is None
    assert args.search_grid == evaluator.DEFAULT_SEARCH_GRID


def test_default_plot_output_path_describes_integer_grid():
    result = evaluator.evaluate_rate_angle_sampling(10.0)

    assert evaluator.default_plot_output_path(result) == Path(
        "kbmod_rate_angle_evaluator_10_days_20X20arcmin_-90_to_90_"
        "eclipticOffsetDeg_64_angs_25_to_225_pixPerDay_64rateIntervals.png"
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["0"],
        ["10", "--pixel-scale", "0"],
        ["10", "--seeing", "-1"],
        ["10", "--patch-size-arcmin", "20", "0"],
        ["10", "--velocity-range-arcsec-per-day", "-1", "140"],
        ["10", "--velocity-range-arcsec-per-day", "140", "20"],
        ["10", "--velocity-range-arcsec-per-day", "0", "0"],
        ["10", "--velocity-range-pixels-per-day", "-1", "140"],
        ["10", "--velocity-range-pixels-per-day", "140", "20"],
        ["10", "--velocity-range-pixels-per-day", "0", "0"],
        [
            "10",
            "--velocity-range-pixels-per-day",
            "25",
            "225",
            "--velocity-range-arcsec-per-day",
            "5",
            "45",
        ],
        ["10", "--velocity-samples", "0"],
        ["10", "--angle-samples", "0"],
        ["10", "--angle-range-deg", "90", "-90"],
        ["10", "--angle-range-deg", "90", "90"],
        ["10", "--ecliptic-angle-deg", "nan"],
    ],
)
def test_parse_args_rejects_invalid_values(argv):
    with pytest.raises(SystemExit):
        evaluator.parse_args(argv)


def test_load_search_grid_from_kbmod_yaml_shape(tmp_path: Path):
    yaml_path = tmp_path / "search_config.yaml"
    yaml_path.write_text(
        """
generator_config:
  name: EclipticCenteredSearch
  angle_units: degree
  angles:
  - 90
  - -90
  - 64
  velocities:
  - 25.0
  - 225.0
  - 64
  velocity_units: pix / d
psf_val: 999
""",
        encoding="utf-8",
    )

    grid = evaluator.load_search_grid_from_kbmod_yaml(yaml_path)

    assert grid.angle_min_deg == -90.0
    assert grid.angle_max_deg == 90.0
    assert grid.angle_offset_min_deg == -90.0
    assert grid.angle_offset_max_deg == 90.0
    assert grid.angle_reference == evaluator.ANGLE_REFERENCE_ECLIPTIC_OFFSET
    assert grid.ecliptic_angle_deg is None
    assert grid.ecliptic_angle_source == evaluator.ECLIPTIC_ANGLE_SOURCE_RUNTIME_WCS
    assert grid.angle_image_min_deg is None
    assert grid.angle_image_max_deg is None
    assert grid.angle_samples == 64
    assert grid.velocity_min_pixels_per_day == 25.0
    assert grid.velocity_max_pixels_per_day == 225.0
    assert grid.velocity_samples == 64


def test_repo_example_yaml_loads():
    yaml_path = Path(__file__).parents[1] / "examples" / "search_config.yaml"

    grid = evaluator.load_search_grid_from_kbmod_yaml(yaml_path)

    assert grid == evaluator.DEFAULT_SEARCH_GRID


@pytest.mark.parametrize(
    ("angle_units", "given_ecliptic", "expected_deg"),
    [
        ("degree", 17.5, 17.5),
        ("radian", 0.25, pytest.approx(14.3239448783)),
    ],
)
def test_yaml_given_ecliptic_resolves_absolute_image_angle_bounds(
    tmp_path: Path,
    angle_units: str,
    given_ecliptic: float,
    expected_deg: float,
):
    yaml_path = tmp_path / "search_config.yaml"
    yaml_path.write_text(
        f"""
generator_config:
  name: EclipticCenteredSearch
  angle_units: {angle_units}
  angles: [-0.1, 0.2, 4]
  given_ecliptic: {given_ecliptic}
  velocities: [10.0, 20.0, 3]
  velocity_units: pix / d
""",
        encoding="utf-8",
    )

    grid = evaluator.load_search_grid_from_kbmod_yaml(yaml_path)

    assert grid.ecliptic_angle_deg == expected_deg
    assert grid.ecliptic_angle_source == evaluator.ECLIPTIC_ANGLE_SOURCE_GIVEN
    assert grid.angle_image_min_deg == pytest.approx(
        grid.ecliptic_angle_deg + grid.angle_offset_min_deg
    )
    assert grid.angle_image_max_deg == pytest.approx(
        grid.ecliptic_angle_deg + grid.angle_offset_max_deg
    )


def test_explicit_ecliptic_angle_overrides_runtime_derived_yaml_reference(tmp_path: Path):
    yaml_path = tmp_path / "search_config.yaml"
    yaml_path.write_text(
        """
generator_config:
  name: EclipticCenteredSearch
  angle_units: degree
  angles: [-30, 30, 11]
  given_ecliptic: null
  velocities: [10.0, 20.0, 3]
  velocity_units: pix / d
""",
        encoding="utf-8",
    )

    grid = evaluator.load_search_grid_from_kbmod_yaml(
        yaml_path,
        ecliptic_angle_deg=12.0,
    )

    assert grid.ecliptic_angle_deg == 12.0
    assert grid.ecliptic_angle_source == evaluator.ECLIPTIC_ANGLE_SOURCE_EXPLICIT
    assert grid.angle_image_min_deg == -18.0
    assert grid.angle_image_max_deg == 42.0


def test_explicit_ecliptic_angle_cannot_conflict_with_yaml_given_ecliptic(tmp_path: Path):
    yaml_path = tmp_path / "search_config.yaml"
    yaml_path.write_text(
        """
generator_config:
  name: EclipticCenteredSearch
  angle_units: degree
  angles: [-30, 30, 11]
  given_ecliptic: 7
  velocities: [10.0, 20.0, 3]
  velocity_units: pix / d
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="conflicts with generator_config.given_ecliptic"):
        evaluator.load_search_grid_from_kbmod_yaml(
            yaml_path,
            ecliptic_angle_deg=12.0,
        )


def test_yaml_overrides_builtins_and_cli_overrides_yaml(tmp_path: Path):
    yaml_path = tmp_path / "search_config.yaml"
    yaml_path.write_text(
        """
generator_config:
  name: EclipticCenteredSearch
  angle_units: degree
  angles: [-30, 30, 11]
  velocities: [10.0, 20.0, 3]
  velocity_units: pix / d
""",
        encoding="utf-8",
    )

    yaml_args = evaluator.parse_args(["10", "--kbmod-config-yaml", str(yaml_path)])
    assert yaml_args.search_grid.angle_min_deg == -30.0
    assert yaml_args.search_grid.angle_max_deg == 30.0
    assert yaml_args.search_grid.angle_samples == 11
    assert yaml_args.search_grid.velocity_min_pixels_per_day == 10.0
    assert yaml_args.search_grid.velocity_max_pixels_per_day == 20.0
    assert yaml_args.search_grid.velocity_samples == 3

    cli_args = evaluator.parse_args(
        [
            "10",
            "--kbmod-config-yaml",
            str(yaml_path),
            "--angle-range-deg",
            "-45",
            "45",
            "--angle-samples",
            "7",
            "--velocity-range-pixels-per-day",
            "1",
            "2",
            "--velocity-samples",
            "5",
        ]
    )
    assert cli_args.search_grid.angle_min_deg == -45.0
    assert cli_args.search_grid.angle_max_deg == 45.0
    assert cli_args.search_grid.angle_samples == 7
    assert cli_args.search_grid.velocity_min_pixels_per_day == 1.0
    assert cli_args.search_grid.velocity_max_pixels_per_day == 2.0
    assert cli_args.search_grid.velocity_samples == 5

    reference_args = evaluator.parse_args(
        [
            "10",
            "--kbmod-config-yaml",
            str(yaml_path),
            "--ecliptic-angle-deg",
            "12.5",
        ]
    )
    assert reference_args.search_grid.ecliptic_angle_deg == 12.5
    assert reference_args.search_grid.angle_image_min_deg == -17.5
    assert reference_args.search_grid.angle_image_max_deg == 42.5


def test_run_prints_report_without_plot(capsys, tmp_path: Path):
    output = tmp_path / "plot.png"
    result = evaluator.run(["10", "--no-plot", "--plot-output", str(output)])

    captured = capsys.readouterr()
    assert result.search_grid.angle_samples == 64
    assert result.primary.angle_count == 708
    assert "chosen angles: 64" in captured.out
    assert "inclusive angles: 708" in captured.out
    assert "configured angle-offset range (from ecliptic): -90 to 90 deg" in captured.out
    assert "angle reference: offsets from KBMOD's per-field ecliptic direction" in captured.out
    assert "absolute image-angle range: unavailable" in captured.out
    assert "Typical TNO SSB reference" in captured.out
    assert "  rate: 0.6 arcsec/hour (72 px/day)" in captured.out
    assert "adjacent chosen-angle endpoint separation at reference travel: 35.9 px (7.18 arcsec)" in captured.out
    assert "Typical Trojan reference" in captured.out
    assert "  rate: 13 arcsec/hour (1560 px/day)" in captured.out
    assert "reference compared to configured maximum: 0.32x" in captured.out
    assert "reference compared to configured maximum: 6.93333x" in captured.out
    assert "Configured maximum travel adjacent endpoints" in captured.out
    assert "  separation: 112 px (22.4 arcsec)" in captured.out
    assert "  right-edge spanned angle: 299 px (59.9 arcsec)" in captured.out
    assert "patch corner-to-corner angles: 2667" in captured.out
    assert "strict full-step/no-half-step angles: 1415" in captured.out
    assert "Plots: wrote" not in captured.out
    assert "Report: wrote" in captured.out


def test_run_writes_plot(capsys, tmp_path: Path):
    output = tmp_path / "plot.png"
    pdf_output = tmp_path / "plot.pdf"
    result = evaluator.run(["10", "--plot-output", str(output)])

    captured = capsys.readouterr()
    assert result.search_grid.angle_samples == 64
    assert "Plots: wrote" in captured.out
    assert f"  {output}" in captured.out
    assert f"  {pdf_output}" in captured.out
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert output.stat().st_size > 1000
    assert pdf_output.read_bytes().startswith(b"%PDF")
    assert pdf_output.stat().st_size > 1000


def test_run_writes_report_file(tmp_path: Path, capsys):
    output = tmp_path / "plot.png"
    report = tmp_path / "plot_report.txt"
    result = evaluator.run(["10", "--plot-output", str(output)])

    captured = capsys.readouterr()
    assert result.search_grid.angle_samples == 64
    assert "Report: wrote" in captured.out
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "KBMOD rate/angle evaluation" in text
    assert "PSF-needed angle coverage" in text
    assert "Typical TNO SSB reference" in text
    assert "  rate: 0.6 arcsec/hour (72 px/day)" in text
    assert "adjacent chosen-angle endpoint separation at reference travel: 35.9 px (7.18 arcsec)" in text
    assert "Typical Trojan reference" in text
    assert "  rate: 13 arcsec/hour (1560 px/day)" in text
    assert "Configured maximum travel adjacent endpoints" in text
    assert "  separation: 112 px (22.4 arcsec)" in text
    assert "  right-edge spanned angle: 299 px (59.9 arcsec)" in text


def test_build_search_plot_reference_toggles_control_legend_labels():
    import matplotlib.pyplot as plt

    result = evaluator.evaluate_rate_angle_sampling(10.0)

    default_fig = evaluator.build_search_plot(result)
    try:
        assert "relative to ecliptic" in default_fig.axes[0].get_title()
        assert "ecliptic direction = +x" in default_fig.axes[0].get_xlabel()
        assert any(
            "angle offsets from ecliptic" in text.get_text()
            for text in default_fig.axes[0].texts
        )
        _, default_labels = default_fig.axes[0].get_legend_handles_labels()
        assert any("TNO SSB" in label for label in default_labels)
        assert any("Trojan" in label for label in default_labels)
    finally:
        plt.close(default_fig)

    toggled_fig = evaluator.build_search_plot(
        result,
        show_tno_reference=False,
        show_trojan_reference=False,
    )
    try:
        _, toggled_labels = toggled_fig.axes[0].get_legend_handles_labels()
        assert not any("TNO SSB" in label for label in toggled_labels)
        assert not any("Trojan" in label for label in toggled_labels)
        assert any("configured min" in label for label in toggled_labels)
        assert any("configured max" in label for label in toggled_labels)
    finally:
        plt.close(toggled_fig)


def test_format_report_reference_toggles_do_not_change_core_counts():
    result = evaluator.evaluate_rate_angle_sampling(10.0)

    text = evaluator.format_report(
        result,
        show_tno_reference=False,
        show_trojan_reference=False,
    )

    assert "Typical TNO SSB reference" not in text
    assert "Typical Trojan reference" not in text
    assert "inclusive angles: 708" in text
    assert "Configured maximum travel adjacent endpoints" in text


def test_format_report_includes_known_absolute_image_angle_range():
    grid = evaluator.SearchGridConfig(
        angle_min_deg=-30.0,
        angle_max_deg=30.0,
        angle_samples=7,
        velocity_min_pixels_per_day=25.0,
        velocity_max_pixels_per_day=225.0,
        velocity_samples=64,
        ecliptic_angle_deg=12.0,
        ecliptic_angle_source=evaluator.ECLIPTIC_ANGLE_SOURCE_GIVEN,
    )
    result = evaluator.evaluate_rate_angle_sampling(1.0, search_grid=grid)

    text = evaluator.format_report(result)

    assert "configured angle-offset range (from ecliptic): -30 to 30 deg" in text
    assert "ecliptic image angle: 12 deg (source: generator_config.given_ecliptic)" in text
    assert "absolute image-angle range: -18 to 42 deg" in text


def test_build_export_zip_contains_plot_pdf_and_report_text():
    result = evaluator.evaluate_rate_angle_sampling(10.0)

    zip_bytes = evaluator.build_export_zip(
        result,
        show_tno_reference=False,
        show_trojan_reference=True,
        stem="example_export",
    )

    with zipfile.ZipFile(io.BytesIO(zip_bytes), mode="r") as archive:
        assert sorted(archive.namelist()) == [
            "example_export.pdf",
            "example_export.png",
            "example_export_report.txt",
        ]
        assert archive.read("example_export.png").startswith(b"\x89PNG\r\n\x1a\n")
        assert archive.read("example_export.pdf").startswith(b"%PDF")
        report_text = archive.read("example_export_report.txt").decode("utf-8")
    assert "KBMOD rate/angle evaluation" in report_text
    assert "Typical TNO SSB reference" not in report_text
    assert "Typical Trojan reference" in report_text


def test_run_writes_config_descriptive_default_plot_and_report(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = evaluator.run(["10"])

    output = tmp_path / (
        "kbmod_rate_angle_evaluator_10_days_20X20arcmin_-90_to_90_"
        "eclipticOffsetDeg_64_angs_25_to_225_pixPerDay_64rateIntervals.png"
    )
    pdf_output = tmp_path / (
        "kbmod_rate_angle_evaluator_10_days_20X20arcmin_-90_to_90_"
        "eclipticOffsetDeg_64_angs_25_to_225_pixPerDay_64rateIntervals.pdf"
    )
    report = tmp_path / (
        "kbmod_rate_angle_evaluator_10_days_20X20arcmin_-90_to_90_"
        "eclipticOffsetDeg_64_angs_25_to_225_pixPerDay_64rateIntervals_report.txt"
    )
    assert result.search_grid.angle_samples == 64
    assert output.exists()
    assert pdf_output.exists()
    assert report.exists()


def test_no_report_file_disables_report_output(tmp_path: Path, capsys):
    output = tmp_path / "plot.png"
    report = tmp_path / "plot_report.txt"
    evaluator.run(["10", "--plot-output", str(output), "--no-report-file"])

    captured = capsys.readouterr()
    assert "Report: wrote" not in captured.out
    assert not report.exists()
