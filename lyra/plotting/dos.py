from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lyra.models import DosPlotSettings, GeneratedResult, ParsedDosData
from lyra.plotting.style import publication_rc


def _projection_key(channel) -> str:
    return channel.element if not channel.orbital else f"{channel.element}:{channel.orbital}"


def _signed_density(spin_name, values):
    return -values if spin_name == "down" else values


def _automatic_density_limits(data, settings, shifted_energies):
    visible = (
        np.isfinite(shifted_energies)
        & (shifted_energies >= settings.energy_range.minimum)
        & (shifted_energies <= settings.energy_range.maximum)
    )
    curves = []
    if settings.plot_total:
        curves.extend(
            _signed_density(spin_name, np.asarray(values, dtype=float))
            for spin_name, values in data.densities.items()
        )
    for channel in settings.projection_channels:
        key = _projection_key(channel)
        if key not in data.projections:
            raise ValueError(f"Projection channel {key} is not available in DOS data.")
        curves.extend(
            _signed_density(spin_name, np.asarray(values, dtype=float))
            for spin_name, values in data.projections[key].items()
        )
    finite_visible = [
        values[visible & np.isfinite(values)]
        for values in curves
        if values.shape == shifted_energies.shape
    ]
    finite_visible = [values for values in finite_visible if values.size]
    if not finite_visible:
        return 0.0, 1.0
    values = np.concatenate(finite_visible)
    minimum = min(0.0, float(np.min(values)))
    maximum = max(0.0, float(np.max(values)))
    if minimum == maximum:
        return (minimum - 1.0, maximum + 1.0) if minimum else (0.0, 1.0)
    padding = (maximum - minimum) * 0.05
    return (
        minimum - padding if minimum < 0.0 else 0.0,
        maximum + padding if maximum > 0.0 else 0.0,
    )


def draw_dos_axes(
    axes,
    data: ParsedDosData,
    settings: DosPlotSettings,
    reference_energy: float | None = None,
    energy_axis: str = "x",
    legend_outside: bool = False,
):
    settings.energy_range.validate()
    if settings.density_range is not None:
        settings.density_range.validate()
    if energy_axis not in {"x", "y"}:
        raise ValueError("DOS energy axis must be either 'x' or 'y'.")
    reference = data.efermi if reference_energy is None else reference_energy
    shifted_energies = data.energies - reference
    density_limits = (
        (
            settings.density_range.minimum,
            settings.density_range.maximum,
        )
        if settings.density_range is not None
        else _automatic_density_limits(data, settings, shifted_energies)
    )

    def plot_curve(density, energies, **kwargs):
        if energy_axis == "x":
            axes.plot(energies, density, **kwargs)
        else:
            axes.plot(density, energies, **kwargs)

    if settings.plot_total:
        for spin_name, values in data.densities.items():
            label = "Total" if spin_name == "total" else f"Total {spin_name}"
            plot_curve(
                _signed_density(spin_name, values),
                shifted_energies,
                color="#172033",
                linestyle="--" if spin_name == "down" else "-",
                linewidth=settings.export.line_width * 1.15,
                label=label,
            )
    for channel in settings.projection_channels:
        key = _projection_key(channel)
        if key not in data.projections:
            raise ValueError(f"Projection channel {key} is not available in DOS data.")
        for spin_name, values in data.projections[key].items():
            base_label = (
                channel.element if not channel.orbital else f"{channel.element}-{channel.orbital}"
            )
            label = base_label if spin_name == "total" else f"{base_label} {spin_name}"
            plot_curve(
                _signed_density(spin_name, values),
                shifted_energies,
                color=channel.colour,
                linestyle="--" if spin_name == "down" else "-",
                linewidth=settings.export.line_width,
                label=label,
            )
    if energy_axis == "x":
        axes.axvline(0.0, color="#20242b", linewidth=0.65, linestyle="--")
        axes.axhline(0.0, color="#20242b", linewidth=0.55)
        axes.set_xlim(settings.energy_range.minimum, settings.energy_range.maximum)
        axes.set_xlabel("Energy - $E_F$ (eV)")
        axes.set_ylabel("DOS (states/eV)")
        axes.set_ylim(*density_limits)
    else:
        axes.axhline(0.0, color="#20242b", linewidth=0.65, linestyle="--")
        axes.axvline(0.0, color="#20242b", linewidth=0.55)
        axes.set_ylim(settings.energy_range.minimum, settings.energy_range.maximum)
        axes.set_xlabel("DOS (states/eV)")
        axes.set_xlim(*density_limits)
    if settings.plot_total or settings.projection_channels:
        legend_options = {"frameon": False, "ncol": 1 if energy_axis == "y" else 2}
        if legend_outside:
            legend_options.update(
                loc="center left",
                bbox_to_anchor=(1.04, 0.5),
                borderaxespad=0.0,
            )
        axes.legend(**legend_options)


def build_dos_figure(data: ParsedDosData, settings: DosPlotSettings):
    settings.energy_range.validate()
    width = settings.export.width_mm / 25.4
    with plt.rc_context(publication_rc(settings.export)):
        figure, axes = plt.subplots(figsize=(width, width * 0.72))
        draw_dos_axes(axes, data, settings, settings.reference_energy)
        figure.tight_layout()
    return figure


def render_dos(
    data: ParsedDosData,
    settings: DosPlotSettings,
    output_directory: str | Path,
) -> GeneratedResult:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    figure = build_dos_figure(data, settings)
    outputs = []
    for image_format in settings.export.formats:
        path = output_directory / f"{settings.export.basename}.{image_format}"
        figure.savefig(path, dpi=settings.export.dpi, bbox_inches="tight")
        outputs.append(path)
    plt.close(figure)
    preview = next((path for path in outputs if path.suffix.lower() == ".png"), outputs[0])
    return GeneratedResult(tuple(outputs), preview, "DOS plot generated successfully.")
