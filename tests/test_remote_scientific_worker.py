from __future__ import annotations

from pathlib import Path

import pytest

from lyra.models import BandPlotSettings, DosPlotSettings
from lyra.remote.scientific_worker import (
    build_band_settings,
    build_dos_settings,
    cache_namespace,
    cancellation_requested,
)


def test_remote_settings_are_bounded_and_lite_by_default() -> None:
    band = build_band_settings({})
    dos = build_dos_settings({})

    assert isinstance(band, BandPlotSettings)
    assert isinstance(dos, DosPlotSettings)
    assert (band.energy_range.minimum, band.energy_range.maximum) == (-5.0, 5.0)
    assert band.export.formats == ("png",)
    assert dos.export.formats == ("png",)


def test_remote_settings_accept_projection_channels_and_style() -> None:
    settings = build_band_settings(
        {
            "energy_range": [-2, 3],
            "mode": "fatband",
            "projection_channels": [
                {"element": "Ta", "orbital": "d", "colour": "#e53935"}
            ],
            "font_size": 18,
            "line_width": 2.5,
        }
    )

    assert settings.projection_channels[0].element == "Ta"
    assert settings.projection_channels[0].orbital == "d"
    assert settings.export.font_size == 18
    assert settings.export.line_width == 2.5


@pytest.mark.parametrize(
    "options",
    [
        {"energy_range": [1, -1]},
        {"mode": "rgb", "projection_channels": [{"element": "A"}] * 4},
        {"projection_channels": [{"element": "../bad"}]},
        {"font_size": 1000},
    ],
)
def test_remote_settings_reject_unsafe_or_unbounded_values(options: dict) -> None:
    with pytest.raises(ValueError):
        build_band_settings(options)


def test_cache_namespace_is_outside_source_and_fingerprint_scoped(tmp_path: Path) -> None:
    source = tmp_path / "calc"
    source.mkdir()
    root = tmp_path / "remote-cache"

    namespace = cache_namespace(root, "a" * 64)

    assert namespace == root.resolve() / "data" / ("a" * 64)
    assert not namespace.is_relative_to(source)


def test_cancellation_requires_matching_opaque_token(tmp_path: Path) -> None:
    token = "a" * 24
    (tmp_path / "cancel.json").write_text(
        '{"schema_version":1,"cancel_token":"wrong"}', encoding="utf-8"
    )
    assert cancellation_requested(tmp_path, token) is False
    (tmp_path / "cancel.json").write_text(
        '{"schema_version":1,"cancel_token":"' + token + '"}', encoding="utf-8"
    )
    assert cancellation_requested(tmp_path, token) is True
