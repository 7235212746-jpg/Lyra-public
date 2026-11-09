from __future__ import annotations

from importlib import import_module
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from lyra.models import (
    FermiPlotSettings,
    FermiSurfaceMesh,
    GeneratedResult,
    ParsedFermiData,
)
from lyra.plotting.style import publication_rc


SPIN_COLOURS = {"down": "#d94841"}


def _marching_cubes():
    try:
        return import_module("skimage.measure").marching_cubes
    except ImportError as exc:
        raise RuntimeError(
            "Fermi-surface rendering requires scikit-image."
        ) from exc


def _selected_spins(data: ParsedFermiData, settings: FermiPlotSettings):
    if settings.spin_channel == "all":
        return tuple(data.energies)
    if settings.spin_channel not in data.energies:
        raise ValueError(
            f"Spin channel {settings.spin_channel!r} is unavailable in this calculation."
        )
    return (settings.spin_channel,)


def _periodic_volume(volume: np.ndarray) -> np.ndarray:
    return np.pad(volume, ((0, 1), (0, 1), (0, 1)), mode="wrap")


def _extended_axis(axis: np.ndarray) -> np.ndarray:
    return np.concatenate((axis, (axis[0] + 1.0,)))


def _vertices_to_cartesian(
    vertices: np.ndarray,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    reciprocal_lattice: np.ndarray,
) -> np.ndarray:
    fractional = np.empty_like(vertices, dtype=float)
    for dimension, axis in enumerate(axes):
        extended = _extended_axis(axis)
        fractional[:, dimension] = np.interp(
            vertices[:, dimension],
            np.arange(len(extended), dtype=float),
            extended,
        )
        fractional[:, dimension] -= axis[0] + 0.5
    return fractional @ reciprocal_lattice


def extract_fermi_meshes(
    data: ParsedFermiData,
    settings: FermiPlotSettings,
) -> tuple[FermiSurfaceMesh, ...]:
    settings.validate()
    level = data.efermi + settings.energy_offset
    marching_cubes = _marching_cubes()
    meshes = []
    for spin in _selected_spins(data, settings):
        grid = data.energies[spin]
        for band_index in range(grid.shape[-1]):
            volume = _periodic_volume(grid[..., band_index])
            if not float(np.nanmin(volume)) < level < float(np.nanmax(volume)):
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=DeprecationWarning,
                    module=r"skimage\.measure\._marching_cubes_lewiner",
                )
                vertices, faces, _normals, _values = marching_cubes(
                    volume,
                    level=level,
                )
            meshes.append(
                FermiSurfaceMesh(
                    vertices=_vertices_to_cartesian(
                        vertices,
                        data.axes,
                        data.reciprocal_lattice,
                    ),
                    faces=np.asarray(faces, dtype=np.int32),
                    spin=spin,
                    band_index=band_index,
                )
            )
    if not meshes:
        raise ValueError(
            "No band crosses the selected Fermi-surface energy on this k-point grid."
        )
    return tuple(meshes)


def _cell_corners(reciprocal_lattice: np.ndarray) -> np.ndarray:
    fractional = np.array(
        [
            [x, y, z]
            for x in (-0.5, 0.5)
            for y in (-0.5, 0.5)
            for z in (-0.5, 0.5)
        ],
        dtype=float,
    )
    return fractional @ reciprocal_lattice


def _draw_cell(axes, reciprocal_lattice: np.ndarray) -> np.ndarray:
    corners = _cell_corners(reciprocal_lattice)
    for index, first in enumerate(corners):
        for second_index in range(index + 1, len(corners)):
            second = corners[second_index]
            if np.count_nonzero(
                np.abs(
                    np.array(
                        [
                            (index // 4) % 2,
                            (index // 2) % 2,
                            index % 2,
                        ]
                    )
                    - np.array(
                        [
                            (second_index // 4) % 2,
                            (second_index // 2) % 2,
                            second_index % 2,
                        ]
                    )
                )
            ) == 1:
                axes.plot(
                    *np.vstack((first, second)).T,
                    color="#66758a",
                    linewidth=0.65,
                    alpha=0.75,
                )
    return corners


def _surface_figure(data: ParsedFermiData, settings: FermiPlotSettings):
    width = settings.export.width_mm / 25.4
    figure = plt.figure(figsize=(width, width * 0.86))
    axes = figure.add_subplot(111, projection="3d")
    meshes = extract_fermi_meshes(data, settings)
    legend = {}
    for mesh in meshes:
        colour = SPIN_COLOURS.get(mesh.spin, settings.colour)
        collection = Poly3DCollection(
            mesh.vertices[mesh.faces],
            facecolor=colour,
            edgecolor="none",
            alpha=0.78,
        )
        axes.add_collection3d(collection)
        legend[mesh.spin] = colour
    corners = _draw_cell(axes, data.reciprocal_lattice)
    minimum = corners.min(axis=0)
    maximum = corners.max(axis=0)
    axes.set_xlim(minimum[0], maximum[0])
    axes.set_ylim(minimum[1], maximum[1])
    axes.set_zlim(minimum[2], maximum[2])
    axes.set_box_aspect(np.maximum(maximum - minimum, 1e-12))
    axes.view_init(elev=settings.elevation, azim=settings.azimuth)
    axes.set_xlabel("$k_x$")
    axes.set_ylabel("$k_y$")
    axes.set_zlabel("$k_z$")
    axes.grid(False)
    axes.set_title(f"$E - E_F$ = {settings.energy_offset:.3f} eV")
    if len(legend) > 1:
        axes.legend(
            handles=[
                Patch(facecolor=colour, label=spin)
                for spin, colour in legend.items()
            ],
            frameon=False,
        )
    return figure


def _slice_plane(
    volume: np.ndarray,
    axis_values: np.ndarray,
    position: float,
    axis: int,
) -> np.ndarray:
    extended = _extended_axis(axis_values)
    coordinate = position + axis_values[0] + 0.5
    fractional_index = np.interp(
        coordinate,
        extended,
        np.arange(len(extended), dtype=float),
    )
    base_index = int(np.floor(fractional_index))
    lower = base_index % len(axis_values)
    upper = (lower + 1) % len(axis_values)
    fraction = fractional_index - base_index
    return (
        np.take(volume, lower, axis=axis) * (1.0 - fraction)
        + np.take(volume, upper, axis=axis) * fraction
    )


def _slice_cartesian_coordinates(
    data: ParsedFermiData,
    slice_axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    remaining = tuple(index for index in range(3) if index != slice_axis)
    fractional_axes = [
        _extended_axis(data.axes[index]) - data.axes[index][0] - 0.5
        for index in remaining
    ]
    first, second = np.meshgrid(*fractional_axes, indexing="ij")
    first_vector = np.asarray(data.reciprocal_lattice[remaining[0]], dtype=float)
    second_vector = np.asarray(data.reciprocal_lattice[remaining[1]], dtype=float)
    first_norm = float(np.linalg.norm(first_vector))
    if first_norm <= 0.0:
        raise ValueError("Reciprocal lattice contains a zero in-plane vector.")
    horizontal = first_vector / first_norm
    perpendicular = second_vector - np.dot(second_vector, horizontal) * horizontal
    perpendicular_norm = float(np.linalg.norm(perpendicular))
    if perpendicular_norm <= 0.0:
        raise ValueError("Reciprocal slice vectors must be linearly independent.")
    vertical = perpendicular / perpendicular_norm
    cartesian = (
        first[..., None] * first_vector
        + second[..., None] * second_vector
    )
    return cartesian @ horizontal, cartesian @ vertical


def _slice_figure(data: ParsedFermiData, settings: FermiPlotSettings):
    width = settings.export.width_mm / 25.4
    figure, axes = plt.subplots(figsize=(width, width * 0.82))
    level = data.efermi + settings.energy_offset
    remaining = tuple(index for index in range(3) if index != settings.slice_axis)
    x, y = _slice_cartesian_coordinates(data, settings.slice_axis)
    contour_count = 0
    for spin in _selected_spins(data, settings):
        colour = SPIN_COLOURS.get(spin, settings.colour)
        grid = data.energies[spin]
        for band_index in range(grid.shape[-1]):
            plane = _slice_plane(
                grid[..., band_index],
                data.axes[settings.slice_axis],
                settings.slice_position,
                settings.slice_axis,
            )
            plane = np.pad(plane, ((0, 1), (0, 1)), mode="wrap")
            if not float(np.nanmin(plane)) < level < float(np.nanmax(plane)):
                continue
            axes.contour(
                x,
                y,
                plane,
                levels=(level,),
                colors=(colour,),
                linewidths=settings.export.line_width * 1.3,
            )
            contour_count += 1
    if contour_count == 0:
        plt.close(figure)
        raise ValueError(
            "No band crosses the selected energy on this Fermi-surface slice."
        )
    axes.set_xlabel("$k_{\\parallel,1}$ ($\\AA^{-1}$)")
    axes.set_ylabel("$k_{\\parallel,2}$ ($\\AA^{-1}$)")
    axes.set_aspect("equal", adjustable="box")
    axes.set_title(
        f"$q_{settings.slice_axis + 1}$ = {settings.slice_position:.3f}, "
        f"$E - E_F$ = {settings.energy_offset:.3f} eV"
    )
    figure.tight_layout()
    return figure


def build_fermi_figure(
    data: ParsedFermiData,
    settings: FermiPlotSettings,
):
    settings.validate()
    with plt.rc_context(publication_rc(settings.export)):
        figure = (
            _surface_figure(data, settings)
            if settings.view_mode == "surface"
            else _slice_figure(data, settings)
        )
        figure.tight_layout()
    return figure


def render_fermi(
    data: ParsedFermiData,
    settings: FermiPlotSettings,
    output_directory: str | Path,
) -> GeneratedResult:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    figure = build_fermi_figure(data, settings)
    outputs = []
    for image_format in settings.export.formats:
        path = output_directory / f"{settings.export.basename}.{image_format}"
        figure.savefig(path, dpi=settings.export.dpi, bbox_inches="tight")
        outputs.append(path)
    plt.close(figure)
    preview = next(
        (path for path in outputs if path.suffix.lower() == ".png"),
        outputs[0],
    )
    return GeneratedResult(
        tuple(outputs),
        preview,
        "Fermi-surface plot generated successfully.",
    )
