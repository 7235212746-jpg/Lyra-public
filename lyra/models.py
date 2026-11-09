from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping

import numpy as np


class ReadMode(str, Enum):
    LITE = "lite"
    FULL = "full"


class SpinKind(str, Enum):
    UNKNOWN = "unknown"
    NON_SPIN = "non_spin"
    COLLINEAR = "collinear"
    SOC = "soc"


@dataclass(frozen=True)
class CapabilityReport:
    folder: Path
    mode: ReadMode
    detected_files: tuple[str, ...]
    used_files: tuple[str, ...]
    ignored_files: tuple[str, ...]
    missing_files: tuple[str, ...]
    features: Mapping[str, bool]
    disabled_reasons: Mapping[str, str] = field(default_factory=dict)
    calculation_type: SpinKind = SpinKind.UNKNOWN

    def is_available(self, feature: str) -> bool:
        return bool(self.features.get(feature, False))

    def reason(self, feature: str) -> str:
        return self.disabled_reasons.get(feature, "")


@dataclass(frozen=True)
class ProjectionChannel:
    element: str
    orbital: str | None = None
    colour: str = "#3974d8"


@dataclass(frozen=True)
class PlotRange:
    minimum: float
    maximum: float

    def validate(self) -> None:
        if self.minimum >= self.maximum:
            raise ValueError("Plot range minimum must be smaller than maximum.")


@dataclass(frozen=True)
class ExportSettings:
    basename: str
    formats: tuple[str, ...] = ("png", "pdf", "svg")
    dpi: int = 600
    width_mm: float = 180.0
    font_family: str = "Arial"
    font_size: float = 13.0
    line_width: float = 1.0


@dataclass(frozen=True)
class QualityIssue:
    severity: str
    code: str
    summary: str
    evidence: str
    recommendation: str


@dataclass(frozen=True)
class QualityMetric:
    name: str
    value: str
    unit: str = ""
    interpretation: str = ""


@dataclass(frozen=True)
class AnalysisQualityReport:
    verdict: str
    method: str
    metrics: tuple[QualityMetric, ...] = ()
    issues: tuple[QualityIssue, ...] = ()
    sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class GeneratedResult:
    output_files: tuple[Path, ...]
    preview_file: Path
    message: str
    source_identities: tuple[tuple[Path, "CacheIdentity"], ...] = ()
    quality_report: AnalysisQualityReport | None = None
    report_files: tuple[Path, ...] = ()


@dataclass(frozen=True)
class ParsedStructure:
    lattice: np.ndarray
    atom_elements: tuple[str, ...]
    fractional_positions: np.ndarray
    source: str

    def __post_init__(self) -> None:
        atom_elements = tuple(self.atom_elements)
        lattice = np.asarray(self.lattice, dtype=float)
        positions = np.asarray(self.fractional_positions, dtype=float)
        if lattice.shape != (3, 3) or not np.all(np.isfinite(lattice)):
            raise ValueError("lattice must be a finite 3x3 array.")
        if abs(float(np.linalg.det(lattice))) <= 1e-12:
            raise ValueError("lattice must have non-zero volume.")
        if positions.shape != (len(atom_elements), 3) or not np.all(np.isfinite(positions)):
            raise ValueError("fractional_positions must be finite and match atom_elements.")
        if not all(isinstance(element, str) and element for element in atom_elements):
            raise ValueError("atom_elements must contain non-empty element symbols.")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source must be a non-empty string.")
        lattice = lattice.copy()
        positions = np.mod(positions, 1.0)
        positions[np.isclose(positions, 1.0, atol=1e-12, rtol=0.0)] = 0.0
        positions = positions.copy()
        lattice.setflags(write=False)
        positions.setflags(write=False)
        object.__setattr__(self, "lattice", lattice)
        object.__setattr__(self, "atom_elements", atom_elements)
        object.__setattr__(self, "fractional_positions", positions)


@dataclass(frozen=True)
class ParsedBandData:
    folder: Path
    kpoints: np.ndarray
    distances: np.ndarray
    segment_ranges: tuple[tuple[int, int], ...]
    tick_positions: tuple[float, ...]
    tick_labels: tuple[str, ...]
    energies: Mapping[str, np.ndarray]
    efermi: float
    spin_kind: SpinKind
    elements: tuple[str, ...]
    occupations: Mapping[str, np.ndarray] = field(default_factory=dict)
    projections: Mapping[str, Mapping[str, np.ndarray]] = field(default_factory=dict)
    orbitals: tuple[str, ...] = ()
    projection_total: Mapping[str, np.ndarray] = field(default_factory=dict)
    structure: ParsedStructure | None = None


@dataclass(frozen=True)
class BandStateSelection:
    spin: str
    kpoint_index: int
    band_index: int

    def __post_init__(self) -> None:
        if not self.spin:
            raise ValueError("Band-state spin name must not be empty.")
        if self.kpoint_index < 0 or self.band_index < 0:
            raise ValueError("Band-state indices must be non-negative.")


@dataclass(frozen=True)
class BandElementIndex:
    spin_names: tuple[str, ...]
    elements: tuple[str, ...]
    orbitals: tuple[str, ...]
    kpoints: np.ndarray
    distances: np.ndarray
    segment_ranges: tuple[tuple[int, int], ...]
    tick_positions: tuple[float, ...]
    tick_labels: tuple[str, ...]
    energies: Mapping[str, np.ndarray]
    reference_energy: float
    raw_element_totals: np.ndarray
    raw_projection_totals: np.ndarray
    relative_element_percentages: np.ndarray
    structure: ParsedStructure | None = None


@dataclass(frozen=True)
class BandStateDetail:
    selection: BandStateSelection
    elements: tuple[str, ...]
    orbitals: tuple[str, ...]
    band_indices: tuple[int, ...]
    raw_orbital_weights: np.ndarray
    relative_orbital_percentages: np.ndarray
    raw_projection_total: float
    spin_components: np.ndarray | None = None
    whole_band: bool = False


@dataclass(frozen=True)
class BandPlotSettings:
    energy_range: PlotRange
    reference_energy: float | None = None
    mode: str = "standard"
    projection_channels: tuple[ProjectionChannel, ...] = ()
    complete_band_colour: str = "#000000"
    fat_scale: float = 120.0
    export: ExportSettings = field(
        default_factory=lambda: ExportSettings(basename="lyra_band")
    )


@dataclass(frozen=True)
class ParsedDosData:
    folder: Path
    energies: np.ndarray
    densities: Mapping[str, np.ndarray]
    efermi: float
    spin_kind: SpinKind
    elements: tuple[str, ...]
    projections: Mapping[str, Mapping[str, np.ndarray]] = field(default_factory=dict)
    orbitals: tuple[str, ...] = ()
    structure: ParsedStructure | None = None


@dataclass(frozen=True)
class DosPlotSettings:
    energy_range: PlotRange
    reference_energy: float | None = None
    projection_channels: tuple[ProjectionChannel, ...] = ()
    plot_total: bool = True
    export: ExportSettings = field(
        default_factory=lambda: ExportSettings(basename="lyra_dos")
    )
    density_range: PlotRange | None = None


@dataclass(frozen=True)
class CombinedPlotSettings:
    energy_range: PlotRange
    panel_ratio: float = 3.0
    reference_energy: float | None = None
    band_mode: str = "standard"
    band_projection_channels: tuple[ProjectionChannel, ...] = ()
    band_complete_colour: str = "#000000"
    band_fat_scale: float = 120.0
    dos_projection_channels: tuple[ProjectionChannel, ...] = ()
    plot_total_dos: bool = True
    export: ExportSettings = field(
        default_factory=lambda: ExportSettings(basename="lyra_band_dos")
    )
    dos_density_range: PlotRange | None = None

    def validate(self) -> None:
        self.energy_range.validate()
        if self.dos_density_range is not None:
            self.dos_density_range.validate()
        if self.panel_ratio <= 0:
            raise ValueError("Combined panel ratio must be greater than zero.")


@dataclass(frozen=True)
class ParsedFermiData:
    folder: Path
    axes: tuple[np.ndarray, np.ndarray, np.ndarray]
    energies: Mapping[str, np.ndarray]
    efermi: float
    reciprocal_lattice: np.ndarray
    spin_kind: SpinKind
    elements: tuple[str, ...] = ()

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        return tuple(len(axis) for axis in self.axes)


@dataclass(frozen=True)
class FermiSurfaceMesh:
    vertices: np.ndarray
    faces: np.ndarray
    spin: str
    band_index: int


@dataclass(frozen=True)
class FermiPlotSettings:
    energy_offset: float = 0.0
    view_mode: str = "surface"
    spin_channel: str = "all"
    slice_axis: int = 2
    slice_position: float = 0.0
    elevation: float = 24.0
    azimuth: float = -55.0
    colour: str = "#3974d8"
    export: ExportSettings = field(
        default_factory=lambda: ExportSettings(basename="lyra_fermi_surface")
    )

    def validate(self) -> None:
        if not np.isfinite(self.energy_offset):
            raise ValueError("Fermi-surface energy offset must be finite.")
        if self.view_mode not in {"surface", "slice"}:
            raise ValueError("Fermi view mode must be 'surface' or 'slice'.")
        if self.spin_channel not in {"all", "total", "up", "down"}:
            raise ValueError("Unsupported Fermi-surface spin channel.")
        if self.slice_axis not in {0, 1, 2}:
            raise ValueError("Fermi slice axis must be 0, 1, or 2.")
        if not -0.5 <= self.slice_position <= 0.5:
            raise ValueError("Fermi slice position must be between -0.5 and 0.5.")
