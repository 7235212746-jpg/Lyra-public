from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
import numpy as np

from lyra.adapters.common import normalise_orbital, orbital_group
from lyra.models import BandPlotSettings, GeneratedResult, ParsedBandData
from lyra.plotting.style import publication_rc


SPIN_STYLES = {
    "total": ("#2457d6", "-"),
    "up": ("#d94841", "-"),
    "down": ("#2457d6", "--"),
}


def _projection_key(channel) -> str:
    return channel.element if not channel.orbital else f"{channel.element}:{channel.orbital}"


def _projection_values(data, channel, spin_name):
    key = _projection_key(channel)
    if key not in data.projections:
        raise ValueError(f"Projection channel {key} is not available in the parsed data.")
    channels = data.projections[key]
    if spin_name in channels:
        return channels[spin_name]
    if "total" in channels:
        return channels["total"]
    raise ValueError(f"Projection channel {key} does not contain spin {spin_name}.")


def _projection_total_values(data, spin_name, selected):
    if spin_name in data.projection_total:
        return data.projection_total[spin_name]
    if "total" in data.projection_total:
        return data.projection_total["total"]
    return np.sum(np.stack(selected, axis=0), axis=0)


def _canonical_projection_scope(channel):
    orbital = normalise_orbital(channel.orbital) if channel.orbital else None
    return channel.element, orbital


def _validate_rgb_channels(channels):
    orbital_groups = {"s", "p", "d", "f"}
    for first_index, first in enumerate(channels):
        first_element, first_orbital = _canonical_projection_scope(first)
        for second_index in range(first_index + 1, len(channels)):
            second = channels[second_index]
            second_element, second_orbital = _canonical_projection_scope(second)
            if first_element != second_element:
                continue
            overlaps = (
                first_orbital is None
                or second_orbital is None
                or first_orbital == second_orbital
                or (
                    first_orbital in orbital_groups
                    and orbital_group(second_orbital) == first_orbital
                )
                or (
                    second_orbital in orbital_groups
                    and orbital_group(first_orbital) == second_orbital
                )
            )
            if overlaps:
                first_label = _projection_key(first)
                second_label = _projection_key(second)
                raise ValueError(
                    f"RGB channels {first_label} (channel {first_index + 1}) and "
                    f"{second_label} (channel {second_index + 1}) overlap; "
                    "RGB channels must not overlap."
                )


def _fat_display_indices(point_count, segment_ranges, spin_name):
    selected = np.zeros(point_count, dtype=bool)
    for start, stop in segment_ranges:
        start = max(0, int(start))
        stop = min(point_count, int(stop))
        if stop > start:
            selected[start:stop] = True
    return np.flatnonzero(selected)


def _draw_standard(
    axes,
    data,
    settings,
    reference,
    muted=False,
    include_single_points=True,
):
    for spin_name, matrix in data.energies.items():
        colour, linestyle = SPIN_STYLES.get(spin_name, ("#2457d6", "-"))
        if spin_name == "total":
            colour = settings.complete_band_colour
        if muted:
            colour = settings.complete_band_colour
        segments = []
        labels = []
        for band_index in range(matrix.shape[1]):
            shifted = matrix[:, band_index] - reference
            for segment_index, (start, stop) in enumerate(data.segment_ranges):
                if stop - start < (1 if include_single_points else 2):
                    continue
                segments.append(
                    np.column_stack((data.distances[start:stop], shifted[start:stop]))
                )
                labels.append(f"_band_{spin_name}_{band_index}_{segment_index}")
        if not segments:
            continue
        collection = LineCollection(
            segments,
            colors=colour,
            linestyles=linestyle,
            linewidths=settings.export.line_width * (0.6 if muted else 1.0),
            alpha=1.0,
            capstyle="projecting" if linestyle == "-" else "butt",
            joinstyle="round",
        )
        collection._lyra_band_labels = tuple(labels)
        collection._lyra_layer = "complete-band"
        axes.add_collection(collection)


def _draw_rgb(axes, data, settings, reference):
    channels = settings.projection_channels
    if not 1 <= len(channels) <= 3:
        raise ValueError("RGB projected Band requires one to at most three channels.")
    _validate_rgb_channels(channels)
    channel_colours = np.array([to_rgb(channel.colour) for channel in channels])
    spin_names = {name for name in data.energies if name in {"up", "down"}}
    show_spin_guides = len(spin_names) > 1
    guide_width = settings.export.line_width * 0.8
    legend_handles = []
    _draw_standard(
        axes,
        data,
        settings,
        reference,
        muted=True,
        include_single_points=False,
    )
    for spin_name, matrix in data.energies.items():
        spin_colour, linestyle = SPIN_STYLES.get(spin_name, ("#2457d6", "-"))
        projected = [_projection_values(data, channel, spin_name) for channel in channels]
        projection_total = _projection_total_values(data, spin_name, projected)
        all_line_segments = []
        all_colours = []
        labels = []
        edge_labels = []
        path_slices = []
        guide_paths = []
        edge_offset = 0
        for band_index in range(matrix.shape[1]):
            shifted = matrix[:, band_index] - reference
            for segment_index, (start, stop) in enumerate(data.segment_ranges):
                x = data.distances[start:stop]
                y = shifted[start:stop]
                points = np.column_stack((x, y))
                if len(points) < 2:
                    continue
                segment_lines = np.stack((points[:-1], points[1:]), axis=1)
                weights = np.stack(
                    [values[start:stop, band_index] for values in projected], axis=1
                )
                segment_weights = (weights[:-1] + weights[1:]) / 2.0
                total_weights = projection_total[start:stop, band_index]
                totals = ((total_weights[:-1] + total_weights[1:]) / 2.0)[:, None]
                fractions = np.divide(
                    np.maximum(segment_weights, 0.0),
                    np.maximum(totals, 0.0),
                    out=np.zeros_like(segment_weights),
                    where=totals > 1e-12,
                )
                selected_fraction = np.clip(
                    np.sum(fractions, axis=1, keepdims=True), 0.0, 1.0
                )
                mixed_colours = np.divide(
                    fractions @ channel_colours,
                    selected_fraction,
                    out=np.zeros((len(fractions), 3), dtype=float),
                    where=selected_fraction > 1e-12,
                )
                segment_colours = np.column_stack(
                    (np.clip(mixed_colours, 0.0, 1.0), selected_fraction[:, 0])
                )
                all_line_segments.append(segment_lines)
                all_colours.append(segment_colours)
                label = f"_band_{spin_name}_{band_index}_{segment_index}"
                edge_count = len(segment_lines)
                labels.append(label)
                edge_labels.extend((label,) * edge_count)
                path_slices.append((label, edge_offset, edge_offset + edge_count))
                guide_paths.append((x, y, label))
                edge_offset += edge_count
        if not all_line_segments:
            continue
        collection = LineCollection(
            np.concatenate(all_line_segments),
            colors=np.concatenate(all_colours),
            linewidths=settings.export.line_width * 1.35,
        )
        collection._lyra_band_labels = tuple(labels)
        collection._lyra_band_edge_labels = tuple(edge_labels)
        collection._lyra_band_path_slices = tuple(path_slices)
        collection._lyra_layer = "projection"
        axes.add_collection(collection)
        if show_spin_guides and spin_name in spin_names:
            for x, y, label in guide_paths:
                axes.plot(
                    x,
                    y,
                    color=spin_colour,
                    linestyle=linestyle,
                    linewidth=guide_width,
                    label=f"_band_spin_guide_{label}",
                )
            legend_handles.append(
                Line2D(
                    [],
                    [],
                    color=spin_colour,
                    linestyle=linestyle,
                    linewidth=guide_width,
                    label=f"Spin {spin_name}",
                )
            )
    if legend_handles:
        axes.legend(handles=legend_handles, frameon=False)


def _draw_fat(axes, data, settings, reference):
    if not settings.projection_channels:
        raise ValueError("Fat-band mode requires at least one projection channel.")
    _draw_standard(axes, data, settings, reference, muted=True)
    for spin_name, matrix in data.energies.items():
        if matrix.shape[1] == 0:
            continue
        display_indices = _fat_display_indices(
            len(data.distances),
            data.segment_ranges,
            spin_name,
        )
        if not len(display_indices):
            continue
        for channel in settings.projection_channels:
            weights = _projection_values(data, channel, spin_name)
            axes.scatter(
                np.tile(data.distances[display_indices], matrix.shape[1]),
                (matrix[display_indices] - reference).T.reshape(-1),
                s=(
                    np.maximum(weights[display_indices], 0.0)
                    * settings.fat_scale
                ).T.reshape(-1),
                color=channel.colour,
                alpha=0.55,
                linewidths=0,
            )


def draw_band_axes(
    axes,
    data: ParsedBandData,
    settings: BandPlotSettings,
    reference_energy: float | None = None,
    show_ylabel: bool = True,
):
    settings.energy_range.validate()
    reference = (
        data.efermi if reference_energy is None else reference_energy
    )
    if settings.mode == "standard":
        _draw_standard(axes, data, settings, reference)
    elif settings.mode == "rgb":
        _draw_rgb(axes, data, settings, reference)
    elif settings.mode == "fat":
        _draw_fat(axes, data, settings, reference)
    else:
        raise ValueError(f"Unsupported Band rendering mode: {settings.mode}")
    for position in data.tick_positions:
        axes.axvline(position, color="#20242b", linewidth=0.55, zorder=0)
    axes.axhline(0.0, color="#20242b", linewidth=0.65, linestyle="--", zorder=0)
    axes.set_xticks(data.tick_positions, data.tick_labels)
    axes.set_xlim(float(data.distances[0]), float(data.distances[-1]))
    axes.set_ylim(settings.energy_range.minimum, settings.energy_range.maximum)
    axes.set_xlabel("k path")
    if show_ylabel:
        axes.set_ylabel("Energy - $E_F$ (eV)")


def build_band_figure(data: ParsedBandData, settings: BandPlotSettings):
    settings.energy_range.validate()
    width = settings.export.width_mm / 25.4
    reference = data.efermi if settings.reference_energy is None else settings.reference_energy
    with plt.rc_context(publication_rc(settings.export)):
        figure, axes = plt.subplots(figsize=(width, width * 0.72))
        draw_band_axes(axes, data, settings, reference)
        figure.tight_layout()
    return figure


def render_band(
    data: ParsedBandData,
    settings: BandPlotSettings,
    output_directory: str | Path,
) -> GeneratedResult:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    figure = build_band_figure(data, settings)
    outputs = []
    for image_format in settings.export.formats:
        path = output_directory / f"{settings.export.basename}.{image_format}"
        figure.savefig(path, dpi=settings.export.dpi, bbox_inches="tight")
        outputs.append(path)
    plt.close(figure)
    preview = next((path for path in outputs if path.suffix.lower() == ".png"), outputs[0])
    return GeneratedResult(tuple(outputs), preview, "Band plot generated successfully.")
