from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lyra.models import (
    BandPlotSettings,
    CombinedPlotSettings,
    DosPlotSettings,
    GeneratedResult,
    ParsedBandData,
    ParsedDosData,
)
from lyra.plotting.band import draw_band_axes
from lyra.plotting.dos import draw_dos_axes
from lyra.plotting.style import publication_rc


def build_combined_figure(
    band_data: ParsedBandData,
    dos_data: ParsedDosData,
    settings: CombinedPlotSettings,
):
    settings.validate()
    width = settings.export.width_mm / 25.4
    reference = (
        dos_data.efermi
        if settings.reference_energy is None
        else settings.reference_energy
    )
    band_settings = BandPlotSettings(
        energy_range=settings.energy_range,
        reference_energy=reference,
        mode=settings.band_mode,
        projection_channels=settings.band_projection_channels,
        complete_band_colour=settings.band_complete_colour,
        fat_scale=settings.band_fat_scale,
        export=settings.export,
    )
    dos_settings = DosPlotSettings(
        energy_range=settings.energy_range,
        density_range=settings.dos_density_range,
        reference_energy=reference,
        projection_channels=settings.dos_projection_channels,
        plot_total=settings.plot_total_dos,
        export=settings.export,
    )
    with plt.rc_context(publication_rc(settings.export)):
        figure, (band_axes, dos_axes) = plt.subplots(
            1,
            2,
            figsize=(width, width * 0.63),
            sharey=True,
            layout="constrained",
            gridspec_kw={"width_ratios": (settings.panel_ratio, 1.0), "wspace": 0.06},
        )
        draw_band_axes(band_axes, band_data, band_settings, reference)
        draw_dos_axes(
            dos_axes,
            dos_data,
            dos_settings,
            reference,
            energy_axis="y",
            legend_outside=True,
        )
        dos_axes.tick_params(axis="y", labelleft=False)
    return figure


def render_combined(
    band_data: ParsedBandData,
    dos_data: ParsedDosData,
    settings: CombinedPlotSettings,
    output_directory: str | Path,
) -> GeneratedResult:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    figure = build_combined_figure(band_data, dos_data, settings)
    outputs = []
    for image_format in settings.export.formats:
        path = output_directory / f"{settings.export.basename}.{image_format}"
        figure.savefig(path, dpi=settings.export.dpi, bbox_inches="tight")
        outputs.append(path)
    plt.close(figure)
    preview = next((path for path in outputs if path.suffix.lower() == ".png"), outputs[0])
    return GeneratedResult(
        tuple(outputs), preview, "Combined Band and DOS plot generated successfully."
    )
