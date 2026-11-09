from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4
from zipfile import BadZipFile

import numpy as np

from lyra.models import ParsedBandData, ParsedDosData, ParsedStructure, ReadMode, SpinKind
from lyra.projection_store import BandProjectionStore, DosProjectionStore


CACHE_SCHEMA_VERSION = 1
_CACHE_ROOT_NAME = ".lyra-cache"
_BAND_CACHE_NAME = "band-projection-v1"
_DOS_CACHE_NAME = "dos-projection-v1"
_MANIFEST_NAME = "manifest.json"
_PROJECTIONS_NAME = "projections.npy"
_SPIN_COMPONENTS_NAME = "spin_components.npy"
_METADATA_NAME = "energies.npz"
_DOS_METADATA_NAME = "dos.npz"
_GENERATIONS_NAME = "generations"
_INDEX_NAME = "index.json"
_REBUILD_TRANSACTION_NAME = "rebuild-transaction.json"
_TEMP_PREFIX = ".tmp-"


@dataclass(frozen=True)
class CacheSource:
    filename: str
    size: int
    mtime_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.filename, str):
            raise ValueError("Cache source filename must be a non-empty string.")
        filename = Path(self.filename).name
        if not filename:
            raise ValueError("Cache source filename must be a non-empty string.")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
            raise ValueError("Cache source size must be a non-negative integer.")
        if isinstance(self.mtime_ns, bool) or not isinstance(self.mtime_ns, int) or self.mtime_ns < 0:
            raise ValueError("Cache source mtime_ns must be a non-negative integer.")
        object.__setattr__(self, "filename", filename)

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> CacheSource:
        required = {"filename", "size", "mtime_ns"}
        if not isinstance(source, Mapping) or set(source) != required:
            raise ValueError("Cache source records require filename, size, and mtime_ns.")
        return cls(source["filename"], source["size"], source["mtime_ns"])


@dataclass(frozen=True)
class CacheIdentity:
    schema_version: int
    parser_version: str
    read_mode: str
    data_kind: str
    sources: tuple[CacheSource, ...]

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int) or self.schema_version < 1:
            raise ValueError("Cache identity schema_version must be a positive integer.")
        if not isinstance(self.parser_version, str) or not self.parser_version.strip():
            raise ValueError("Cache identity parser_version must be a non-empty string.")
        if self.read_mode not in {mode.value for mode in ReadMode}:
            raise ValueError("Cache identity read_mode is invalid.")
        if self.data_kind not in {"band", "dos", "fermi"}:
            raise ValueError("Cache identity data_kind must be 'band', 'dos', or 'fermi'.")
        if not self.sources or not all(isinstance(source, CacheSource) for source in self.sources):
            raise ValueError("Cache identity sources must contain CacheSource records.")
        if len({source.filename for source in self.sources}) != len(self.sources):
            raise ValueError("Cache identity source filenames must be unique.")
        object.__setattr__(self, "sources", tuple(sorted(self.sources, key=lambda source: source.filename)))

    @classmethod
    def from_mapping(cls, identity: Mapping[str, Any]) -> CacheIdentity:
        required = {"schema_version", "parser_version", "read_mode", "data_kind", "sources"}
        if set(identity) != required:
            raise ValueError(
                "Cache identity requires schema_version, parser_version, read_mode, data_kind, and sources."
            )
        sources = identity["sources"]
        if isinstance(sources, str | bytes) or not isinstance(sources, (tuple, list)):
            raise ValueError("Cache identity sources must be a non-empty sequence.")
        return cls(
            schema_version=identity["schema_version"],
            parser_version=identity["parser_version"],
            read_mode=identity["read_mode"],
            data_kind=identity["data_kind"],
            sources=tuple(
                source if isinstance(source, CacheSource) else CacheSource.from_mapping(source)
                for source in sources
            ),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "parser_version": self.parser_version,
            "read_mode": self.read_mode,
            "data_kind": self.data_kind,
            "sources": [
                {"filename": source.filename, "size": source.size, "mtime_ns": source.mtime_ns}
                for source in self.sources
            ],
        }


class ProjectionCache:
    """Folder-local persistence for normalized projection stores."""

    def __init__(self, *, root: str | Path | None = None) -> None:
        self._pending_rebuilds: dict[tuple[Path, str], str] = {}
        self._external_root = Path(root).expanduser().resolve() if root is not None else None

    def load_band(
        self,
        folder: str | Path,
        identity: CacheIdentity | Mapping[str, Any],
        *,
        mmap: bool = False,
    ) -> tuple[ParsedBandData, BandProjectionStore | None] | None:
        expected_identity = _coerce_identity(identity, "band")
        resolved_folder = Path(folder).resolve()
        namespace = self._band_directory(resolved_folder)
        for attempt in range(2):
            generation = self._read_generation_index(namespace)
            if generation is None:
                return None
            loaded = self._load_band_directory(
                namespace / _GENERATIONS_NAME / generation,
                resolved_folder,
                expected_identity,
                mmap=mmap,
            )
            if loaded is not None:
                return loaded
            if attempt or generation == self._read_generation_index(namespace):
                return None
        return None

    def save_band(
        self,
        folder: str | Path,
        identity: CacheIdentity | Mapping[str, Any],
        base_data: ParsedBandData,
        store: BandProjectionStore | None,
        *,
        retain_previous: bool = False,
    ) -> None:
        expected_identity = _coerce_identity(identity, "band")
        resolved_folder = Path(folder).resolve()
        self._validate_save_inputs(resolved_folder, base_data, store)
        namespace = self._band_directory(resolved_folder)
        self.cleanup_incomplete(resolved_folder, "band")
        generation_root = namespace / _GENERATIONS_NAME
        generation_root.mkdir(parents=True, exist_ok=True)
        self._cleanup_temporary_directories(generation_root)
        self._cleanup_temporary_index_files(namespace)

        generation_name = f"generation-{uuid4().hex}"
        temporary_directory = generation_root / f"{_TEMP_PREFIX}{generation_name}"
        generation_directory = generation_root / generation_name
        temporary_directory.mkdir()
        try:
            self._write_band_directory(
                temporary_directory,
                expected_identity,
                resolved_folder,
                base_data,
                store,
            )
            validated = self._load_band_directory(
                temporary_directory,
                resolved_folder,
                expected_identity,
            )
            if validated is None:
                raise RuntimeError("Projection cache validation failed.")
            del validated
            os.replace(temporary_directory, generation_directory)
            if retain_previous:
                self._write_rebuild_transaction(
                    namespace,
                    self._read_generation_index(namespace),
                    generation_name,
                )
                self._pending_rebuilds[(resolved_folder, "band")] = generation_name
            else:
                self._write_generation_index(namespace, generation_name)
                self._cleanup_unreferenced_generations(namespace, generation_name)
        finally:
            if temporary_directory.exists():
                shutil.rmtree(temporary_directory, ignore_errors=True)
            self._cleanup_temporary_directories(generation_root)
            self._cleanup_temporary_index_files(namespace)
            if not retain_previous:
                self._cleanup_unreferenced_generations(
                    namespace,
                    self._read_generation_index(namespace),
                )

    def load_dos(
        self,
        folder: str | Path,
        identity: CacheIdentity | Mapping[str, Any],
    ) -> tuple[ParsedDosData, DosProjectionStore | None] | None:
        expected_identity = _coerce_identity(identity, "dos")
        resolved_folder = Path(folder).resolve()
        namespace = self._dos_directory(resolved_folder)
        for attempt in range(2):
            generation = self._read_generation_index(namespace)
            if generation is None:
                return None
            loaded = self._load_dos_directory(
                namespace / _GENERATIONS_NAME / generation,
                resolved_folder,
                expected_identity,
            )
            if loaded is not None:
                return loaded
            if attempt or generation == self._read_generation_index(namespace):
                return None
        return None

    def band_status(
        self,
        folder: str | Path,
        identity: CacheIdentity | Mapping[str, Any],
    ) -> str:
        expected_identity = _coerce_identity(identity, "band")
        namespace = self._band_directory(Path(folder).resolve())
        generation = self._read_generation_index(namespace)
        if generation is None:
            return "missing" if not namespace.exists() else "stale"
        loaded = self._load_band_directory(
            namespace / _GENERATIONS_NAME / generation,
            Path(folder).resolve(),
            expected_identity,
        )
        return "valid" if loaded is not None else "stale"

    def dos_status(
        self,
        folder: str | Path,
        identity: CacheIdentity | Mapping[str, Any],
    ) -> str:
        expected_identity = _coerce_identity(identity, "dos")
        namespace = self._dos_directory(Path(folder).resolve())
        generation = self._read_generation_index(namespace)
        if generation is None:
            return "missing" if not namespace.exists() else "stale"
        loaded = self._load_dos_directory(
            namespace / _GENERATIONS_NAME / generation,
            Path(folder).resolve(),
            expected_identity,
        )
        return "valid" if loaded is not None else "stale"

    def save_dos(
        self,
        folder: str | Path,
        identity: CacheIdentity | Mapping[str, Any],
        base_data: ParsedDosData,
        store: DosProjectionStore | None,
        *,
        retain_previous: bool = False,
    ) -> None:
        expected_identity = _coerce_identity(identity, "dos")
        resolved_folder = Path(folder).resolve()
        self._validate_dos_save_inputs(base_data, store)
        namespace = self._dos_directory(resolved_folder)
        self.cleanup_incomplete(resolved_folder, "dos")
        generation_root = namespace / _GENERATIONS_NAME
        generation_root.mkdir(parents=True, exist_ok=True)
        self._cleanup_temporary_directories(generation_root)
        self._cleanup_temporary_index_files(namespace)

        generation_name = f"generation-{uuid4().hex}"
        temporary_directory = generation_root / f"{_TEMP_PREFIX}{generation_name}"
        generation_directory = generation_root / generation_name
        temporary_directory.mkdir()
        try:
            self._write_dos_directory(
                temporary_directory,
                expected_identity,
                resolved_folder,
                base_data,
                store,
            )
            validated = self._load_dos_directory(
                temporary_directory,
                resolved_folder,
                expected_identity,
            )
            if validated is None:
                raise RuntimeError("DOS projection cache validation failed.")
            del validated
            os.replace(temporary_directory, generation_directory)
            if retain_previous:
                self._write_rebuild_transaction(
                    namespace,
                    self._read_generation_index(namespace),
                    generation_name,
                )
                self._pending_rebuilds[(resolved_folder, "dos")] = generation_name
            else:
                self._write_generation_index(namespace, generation_name)
                self._cleanup_unreferenced_generations(namespace, generation_name)
        finally:
            if temporary_directory.exists():
                shutil.rmtree(temporary_directory, ignore_errors=True)
            self._cleanup_temporary_directories(generation_root)
            self._cleanup_temporary_index_files(namespace)
            if not retain_previous:
                self._cleanup_unreferenced_generations(
                    namespace,
                    self._read_generation_index(namespace),
                )

    def active_generation(self, folder: str | Path, data_kind: str) -> str | None:
        namespace = self._namespace(Path(folder).resolve(), data_kind)
        return self._read_generation_index(namespace) if namespace is not None else None

    def promote_rebuild(
        self,
        folder: str | Path,
        data_kind: str,
    ) -> None:
        resolved_folder = Path(folder).resolve()
        namespace = self._namespace(resolved_folder, data_kind)
        if namespace is None:
            return
        transaction = self._read_rebuild_transaction(namespace)
        pending = self._pending_rebuilds.get((resolved_folder, data_kind))
        if pending is None and transaction is not None:
            pending = transaction[1]
        if pending is not None:
            self._write_generation_index(namespace, pending)

    def finish_rebuild(
        self,
        folder: str | Path,
        data_kind: str,
        previous_generation: str | None,
        *,
        success: bool,
    ) -> None:
        namespace = self._namespace(Path(folder).resolve(), data_kind)
        if namespace is None:
            return
        resolved_folder = Path(folder).resolve()
        transaction = self._read_rebuild_transaction(namespace)
        pending = self._pending_rebuilds.pop((resolved_folder, data_kind), None)
        if transaction is not None:
            recorded_previous, recorded_pending = transaction
            previous_generation = recorded_previous
            pending = pending or recorded_pending
        if success:
            if pending is not None and self._read_generation_index(namespace) != pending:
                self._write_generation_index(namespace, pending)
        elif previous_generation is None:
            (namespace / _INDEX_NAME).unlink(missing_ok=True)
        else:
            self._write_generation_index(namespace, previous_generation)
        (namespace / _REBUILD_TRANSACTION_NAME).unlink(missing_ok=True)
        self._cleanup_unreferenced_generations(
            namespace,
            self._read_generation_index(namespace),
        )

    def cleanup_incomplete(self, folder: str | Path, data_kind: str) -> None:
        namespace = self._namespace(Path(folder).resolve(), data_kind)
        if namespace is None:
            return
        resolved_folder = Path(folder).resolve()
        transaction = self._read_rebuild_transaction(namespace)
        if transaction is not None:
            previous_generation, _pending = transaction
            if previous_generation is None:
                (namespace / _INDEX_NAME).unlink(missing_ok=True)
            else:
                self._write_generation_index(namespace, previous_generation)
            (namespace / _REBUILD_TRANSACTION_NAME).unlink(missing_ok=True)
        self._pending_rebuilds.pop((resolved_folder, data_kind), None)
        generation_root = namespace / _GENERATIONS_NAME
        self._cleanup_temporary_directories(generation_root)
        self._cleanup_temporary_index_files(namespace)
        self._cleanup_unreferenced_generations(
            namespace,
            self._read_generation_index(namespace),
        )

    @staticmethod
    def _read_rebuild_transaction(namespace: Path) -> tuple[str | None, str] | None:
        try:
            payload = json.loads(
                (namespace / _REBUILD_TRANSACTION_NAME).read_text(encoding="utf-8")
            )
            if set(payload) != {"pending", "previous"}:
                return None
            previous = payload["previous"]
            pending = payload["pending"]
            if previous is not None and (
                not isinstance(previous, str)
                or not previous.startswith("generation-")
                or Path(previous).name != previous
            ):
                return None
            if (
                not isinstance(pending, str)
                or not pending.startswith("generation-")
                or Path(pending).name != pending
            ):
                return None
            return previous, pending
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    @staticmethod
    def _write_rebuild_transaction(
        namespace: Path,
        previous: str | None,
        pending: str,
    ) -> None:
        temporary = namespace / f"{_TEMP_PREFIX}rebuild-{uuid4().hex}.json"
        try:
            temporary.write_text(
                json.dumps(
                    {"pending": pending, "previous": previous},
                    ensure_ascii=True,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            os.replace(temporary, namespace / _REBUILD_TRANSACTION_NAME)
        finally:
            temporary.unlink(missing_ok=True)

    def _namespace(self, folder: Path, data_kind: str) -> Path | None:
        try:
            return {
                "band": self._band_directory(folder),
                "dos": self._dos_directory(folder),
                "fermi": None,
            }[data_kind]
        except KeyError as exc:
            raise ValueError("Cache data_kind must be 'band', 'dos', or 'fermi'.") from exc

    def clear(self, folder: str | Path, data_kind: str | None = None) -> None:
        cache_root = self._cache_root(Path(folder).resolve())
        if data_kind is not None:
            namespace = {
                "band": self._band_directory(Path(folder).resolve()),
                "dos": self._dos_directory(Path(folder).resolve()),
                "fermi": None,
            }.get(data_kind)
            if data_kind not in {"band", "dos", "fermi"}:
                raise ValueError("Cache data_kind must be 'band', 'dos', or 'fermi'.")
            if namespace is not None:
                if namespace.is_dir():
                    shutil.rmtree(namespace)
                elif namespace.exists():
                    namespace.unlink()
            previews = cache_root / "previews"
            if previews.is_dir():
                shutil.rmtree(previews)
            elif previews.exists():
                previews.unlink()
            return
        if cache_root.is_dir():
            shutil.rmtree(cache_root)
        elif cache_root.exists():
            cache_root.unlink()

    def _cache_root(self, folder: Path) -> Path:
        return self._external_root or (folder / _CACHE_ROOT_NAME)

    def _band_directory(self, folder: Path) -> Path:
        return self._cache_root(folder) / _BAND_CACHE_NAME

    def _dos_directory(self, folder: Path) -> Path:
        return self._cache_root(folder) / _DOS_CACHE_NAME

    def _write_band_directory(
        self,
        directory: Path,
        identity: CacheIdentity,
        folder: Path,
        base_data: ParsedBandData,
        store: BandProjectionStore | None,
    ) -> None:
        spin_names = tuple(base_data.energies)
        files = {"metadata": _METADATA_NAME}
        store_metadata = None
        if store is not None:
            np.save(directory / _PROJECTIONS_NAME, store.tensor, allow_pickle=False)
            files["projections"] = _PROJECTIONS_NAME
            store_metadata = {
                "dtype": store.tensor.dtype.name,
                "elements": list(store.elements),
                "orbitals": list(store.orbitals),
                "shape": list(store.tensor.shape),
                "spin_names": list(store.spin_names),
            }
            if store.spin_components is not None:
                np.save(
                    directory / _SPIN_COMPONENTS_NAME,
                    store.spin_components,
                    allow_pickle=False,
                )
                files["spin_components"] = _SPIN_COMPONENTS_NAME
                store_metadata["spin_components_shape"] = list(
                    store.spin_components.shape
                )
        energies = np.stack([np.asarray(base_data.energies[spin]) for spin in spin_names])
        array_metadata = {
            "kpoints": np.asarray(base_data.kpoints),
            "distances": np.asarray(base_data.distances),
            "energies": energies,
        }
        if base_data.occupations:
            array_metadata["occupations"] = np.stack(
                [np.asarray(base_data.occupations[spin]) for spin in spin_names]
            )
        structure_metadata = None
        if base_data.structure is not None:
            array_metadata.update(
                structure_lattice=np.asarray(base_data.structure.lattice),
                structure_positions=np.asarray(
                    base_data.structure.fractional_positions
                ),
            )
            structure_metadata = {
                "atom_elements": list(base_data.structure.atom_elements),
                "source": base_data.structure.source,
            }
        np.savez(directory / _METADATA_NAME, **array_metadata)
        manifest = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "identity": identity.to_json(),
            "files": files,
            "store": store_metadata,
            "band": {
                "efermi": float(base_data.efermi),
                "elements": list(base_data.elements),
                "energy_spins": list(spin_names),
                "folder": str(folder),
                "orbitals": list(base_data.orbitals),
                "segment_ranges": [list(value) for value in base_data.segment_ranges],
                "spin_kind": base_data.spin_kind.value,
                "tick_labels": list(base_data.tick_labels),
                "tick_positions": list(base_data.tick_positions),
                "structure": structure_metadata,
            },
        }
        (directory / _MANIFEST_NAME).write_text(
            json.dumps(manifest, ensure_ascii=True, sort_keys=True), encoding="utf-8"
        )

    def _load_band_directory(
        self,
        directory: Path,
        folder: Path,
        expected_identity: CacheIdentity,
        *,
        mmap: bool = False,
    ) -> tuple[ParsedBandData, BandProjectionStore | None] | None:
        try:
            manifest = json.loads((directory / _MANIFEST_NAME).read_text(encoding="utf-8"))
            if manifest["cache_schema_version"] != CACHE_SCHEMA_VERSION:
                return None
            if manifest["identity"] != expected_identity.to_json():
                return None
            store_metadata = manifest["store"]
            files = manifest["files"]
            if store_metadata is None:
                if files != {"metadata": _METADATA_NAME}:
                    return None
                store = None
            else:
                expected_files = {
                    "metadata": _METADATA_NAME,
                    "projections": _PROJECTIONS_NAME,
                }
                component_shape = store_metadata.get("spin_components_shape")
                if component_shape is not None:
                    expected_files["spin_components"] = _SPIN_COMPONENTS_NAME
                if files != expected_files:
                    return None
                spin_names = tuple(store_metadata["spin_names"])
                store_elements = tuple(store_metadata["elements"])
                store_orbitals = tuple(store_metadata["orbitals"])
                tensor = np.load(
                    directory / _PROJECTIONS_NAME,
                    allow_pickle=False,
                    mmap_mode="r" if mmap else None,
                )
                if (
                    tensor.dtype != np.dtype(np.float32)
                    or tuple(tensor.shape) != tuple(store_metadata["shape"])
                    or store_metadata["dtype"] != "float32"
                ):
                    return None
                spin_components = None
                if component_shape is not None:
                    spin_components = np.load(
                        directory / _SPIN_COMPONENTS_NAME,
                        allow_pickle=False,
                        mmap_mode="r" if mmap else None,
                    )
                    if (
                        spin_components.dtype != np.dtype(np.float32)
                        or tuple(spin_components.shape) != tuple(component_shape)
                    ):
                        return None
                store = BandProjectionStore(
                    spin_names,
                    store_elements,
                    store_orbitals,
                    tensor,
                    spin_components=spin_components,
                )

            with (directory / _METADATA_NAME).open("rb") as metadata_file:
                with np.load(metadata_file, allow_pickle=False) as metadata:
                    metadata_names = set(metadata.files)
                    basic_names = {"distances", "energies", "kpoints"}
                    occupation_names = {"occupations"}
                    structure_names = {"structure_lattice", "structure_positions"}
                    if frozenset(metadata_names) not in {
                        frozenset(basic_names),
                        frozenset(basic_names | occupation_names),
                        frozenset(basic_names | structure_names),
                        frozenset(basic_names | occupation_names | structure_names),
                    }:
                        return None
                    kpoints = metadata["kpoints"]
                    distances = metadata["distances"]
                    stacked_energies = metadata["energies"]
                    stacked_occupations = (
                        np.asarray(metadata["occupations"], dtype=float)
                        if "occupations" in metadata_names
                        else None
                    )
                    structure_arrays = (
                        (
                            np.asarray(metadata["structure_lattice"], dtype=float),
                            np.asarray(metadata["structure_positions"], dtype=float),
                        )
                        if structure_names <= metadata_names
                        else None
                    )
            band_metadata = manifest["band"]
            spin_names = tuple(band_metadata["energy_spins"])
            expected_energy_shape = tuple(stacked_energies.shape[1:])
            if (
                len(expected_energy_shape) != 2
                or not all(expected_energy_shape)
                or stacked_energies.shape != (len(spin_names), *expected_energy_shape)
                or distances.shape != (expected_energy_shape[0],)
                or kpoints.ndim != 2
                or kpoints.shape[0] != expected_energy_shape[0]
                or (
                    stacked_occupations is not None
                    and stacked_occupations.shape != stacked_energies.shape
                )
            ):
                return None

            if store is not None and store.tensor.shape[3:] != expected_energy_shape:
                return None
            elements = tuple(band_metadata.get("elements", store.elements if store else ()))
            orbitals = tuple(band_metadata.get("orbitals", store.orbitals if store else ()))
            if store is not None and (
                store.spin_names != spin_names
                or store.elements != elements
                or store.orbitals != orbitals
            ):
                return None
            segment_ranges = tuple(tuple(int(value) for value in pair) for pair in band_metadata["segment_ranges"])
            tick_positions = tuple(float(value) for value in band_metadata["tick_positions"])
            tick_labels = tuple(str(value) for value in band_metadata["tick_labels"])
            if len(tick_positions) != len(tick_labels):
                return None
            structure = None
            structure_metadata = band_metadata.get("structure")
            if structure_arrays is not None:
                if not isinstance(structure_metadata, dict):
                    return None
                structure = ParsedStructure(
                    lattice=structure_arrays[0],
                    atom_elements=tuple(structure_metadata["atom_elements"]),
                    fractional_positions=structure_arrays[1],
                    source=str(structure_metadata["source"]),
                )
            data = ParsedBandData(
                folder=folder,
                kpoints=kpoints,
                distances=distances,
                segment_ranges=segment_ranges,
                tick_positions=tick_positions,
                tick_labels=tick_labels,
                energies={
                    spin_name: stacked_energies[index]
                    for index, spin_name in enumerate(spin_names)
                },
                occupations=(
                    {
                        spin_name: stacked_occupations[index]
                        for index, spin_name in enumerate(spin_names)
                    }
                    if stacked_occupations is not None
                    else {}
                ),
                efermi=float(band_metadata["efermi"]),
                spin_kind=SpinKind(band_metadata["spin_kind"]),
                elements=elements,
                orbitals=orbitals,
                structure=structure,
            )
            return data, store
        except (
            KeyError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            BadZipFile,
        ):
            return None

    def _write_dos_directory(
        self,
        directory: Path,
        identity: CacheIdentity,
        folder: Path,
        base_data: ParsedDosData,
        store: DosProjectionStore | None,
    ) -> None:
        spin_names = tuple(base_data.densities)
        files = {"metadata": _DOS_METADATA_NAME}
        store_metadata = None
        if store is not None:
            np.save(directory / _PROJECTIONS_NAME, store.tensor, allow_pickle=False)
            files["projections"] = _PROJECTIONS_NAME
            store_metadata = {
                "dtype": store.tensor.dtype.name,
                "elements": list(store.elements),
                "orbitals": list(store.orbitals),
                "shape": list(store.tensor.shape),
                "spin_names": list(store.spin_names),
            }
        densities = np.stack(
            [np.asarray(base_data.densities[spin]) for spin in spin_names]
        )
        np.savez(
            directory / _DOS_METADATA_NAME,
            energies=np.asarray(base_data.energies),
            densities=densities,
        )
        manifest = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "identity": identity.to_json(),
            "files": files,
            "store": store_metadata,
            "dos": {
                "density_spins": list(spin_names),
                "efermi": float(base_data.efermi),
                "elements": list(base_data.elements),
                "folder": str(folder),
                "orbitals": list(base_data.orbitals),
                "spin_kind": base_data.spin_kind.value,
            },
        }
        (directory / _MANIFEST_NAME).write_text(
            json.dumps(manifest, ensure_ascii=True, sort_keys=True), encoding="utf-8"
        )

    def _load_dos_directory(
        self,
        directory: Path,
        folder: Path,
        expected_identity: CacheIdentity,
    ) -> tuple[ParsedDosData, DosProjectionStore | None] | None:
        try:
            manifest = json.loads((directory / _MANIFEST_NAME).read_text(encoding="utf-8"))
            if manifest["cache_schema_version"] != CACHE_SCHEMA_VERSION:
                return None
            if manifest["identity"] != expected_identity.to_json():
                return None
            store_metadata = manifest["store"]
            files = manifest["files"]
            if store_metadata is None:
                if files != {"metadata": _DOS_METADATA_NAME}:
                    return None
                store = None
            else:
                if files != {
                    "metadata": _DOS_METADATA_NAME,
                    "projections": _PROJECTIONS_NAME,
                }:
                    return None
                spin_names = tuple(store_metadata["spin_names"])
                store_elements = tuple(store_metadata["elements"])
                store_orbitals = tuple(store_metadata["orbitals"])
                tensor = np.load(directory / _PROJECTIONS_NAME, allow_pickle=False)
                if (
                    tensor.dtype != np.dtype(np.float32)
                    or tuple(tensor.shape) != tuple(store_metadata["shape"])
                    or store_metadata["dtype"] != "float32"
                ):
                    return None
                store = DosProjectionStore(
                    spin_names, store_elements, store_orbitals, tensor
                )

            with (directory / _DOS_METADATA_NAME).open("rb") as metadata_file:
                with np.load(metadata_file, allow_pickle=False) as metadata:
                    if set(metadata.files) != {"densities", "energies"}:
                        return None
                    energies = metadata["energies"]
                    stacked_densities = metadata["densities"]
            dos_metadata = manifest["dos"]
            spin_names = tuple(dos_metadata["density_spins"])
            energy_count = energies.size
            if energies.shape != (energy_count,) or stacked_densities.shape != (
                len(spin_names),
                energy_count,
            ):
                return None

            if store is not None and store.tensor.shape[3] != energy_count:
                return None
            elements = tuple(dos_metadata.get("elements", store.elements if store else ()))
            orbitals = tuple(dos_metadata.get("orbitals", store.orbitals if store else ()))
            if store is not None and (
                store.spin_names != spin_names
                or store.elements != elements
                or store.orbitals != orbitals
            ):
                return None
            data = ParsedDosData(
                folder=folder,
                energies=energies,
                densities={
                    spin_name: stacked_densities[index]
                    for index, spin_name in enumerate(spin_names)
                },
                efermi=float(dos_metadata["efermi"]),
                spin_kind=SpinKind(dos_metadata["spin_kind"]),
                elements=elements,
                orbitals=orbitals,
            )
            return data, store
        except (
            KeyError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            BadZipFile,
        ):
            return None

    @staticmethod
    def _validate_save_inputs(
        folder: Path,
        base_data: ParsedBandData,
        store: BandProjectionStore | None,
    ) -> None:
        spin_names = tuple(base_data.energies)
        if not spin_names:
            raise ValueError("Band data must contain at least one energy channel.")
        energy_shape = np.asarray(base_data.energies[spin_names[0]]).shape
        payload_arrays = (
            base_data.kpoints,
            base_data.distances,
            *base_data.energies.values(),
            *base_data.occupations.values(),
        )
        if store is not None:
            payload_arrays = (
                store.tensor,
                *((store.spin_components,) if store.spin_components is not None else ()),
                *payload_arrays,
            )
        if any(np.asarray(values).dtype.hasobject for values in payload_arrays):
            raise ValueError("Cache payload arrays must not use object dtype.")
        if any(
            np.asarray(base_data.energies[spin]).shape != energy_shape
            for spin in spin_names
        ):
            raise ValueError("Band data energy channels must have matching shapes.")
        if base_data.occupations and (
            set(base_data.occupations) != set(spin_names)
            or any(
                np.asarray(base_data.occupations[spin]).shape != energy_shape
                for spin in spin_names
            )
        ):
            raise ValueError("Band occupations must match every energy channel and shape.")
        if len(energy_shape) != 2 or not all(energy_shape):
            raise ValueError("Band data energy axes must be non-empty.")
        if store is None:
            if base_data.projections:
                raise ValueError("Band projection data requires a projection store.")
            return
        if spin_names != store.spin_names:
            raise ValueError("Band data energy channels do not match the projection store.")
        if tuple(base_data.elements) != store.elements or tuple(base_data.orbitals) != store.orbitals:
            raise ValueError("Band data metadata does not match the projection store.")
        if energy_shape != store.tensor.shape[3:]:
            raise ValueError("Band data energy shape does not match the projection store.")

    @staticmethod
    def _validate_dos_save_inputs(
        base_data: ParsedDosData,
        store: DosProjectionStore | None,
    ) -> None:
        spin_names = tuple(base_data.densities)
        if not spin_names:
            raise ValueError("DOS data must contain at least one density channel.")
        payload_arrays = (base_data.energies, *base_data.densities.values())
        if store is not None:
            payload_arrays = (store.tensor, *payload_arrays)
        if any(np.asarray(values).dtype.hasobject for values in payload_arrays):
            raise ValueError("Cache payload arrays must not use object dtype.")
        energy_count = np.asarray(base_data.energies).size
        if np.asarray(base_data.energies).shape != (energy_count,) or any(
            np.asarray(base_data.densities[spin]).shape != (energy_count,)
            for spin in spin_names
        ):
            raise ValueError("DOS data shape does not match the projection store.")
        if store is None:
            if base_data.projections:
                raise ValueError("DOS projection data requires a projection store.")
            return
        if spin_names != store.spin_names:
            raise ValueError("DOS density channels do not match the projection store.")
        if tuple(base_data.elements) != store.elements or tuple(base_data.orbitals) != store.orbitals:
            raise ValueError("DOS data metadata does not match the projection store.")
        if store.tensor.shape[3] != energy_count:
            raise ValueError("DOS data shape does not match the projection store.")

    @staticmethod
    def _read_generation_index(namespace: Path) -> str | None:
        try:
            index = json.loads((namespace / _INDEX_NAME).read_text(encoding="utf-8"))
            generation = index["generation"]
            if set(index) != {"generation"} or not isinstance(generation, str):
                return None
            if not generation.startswith("generation-") or Path(generation).name != generation:
                return None
            return generation
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    @staticmethod
    def _write_generation_index(namespace: Path, generation: str) -> None:
        temporary_index = namespace / f"{_TEMP_PREFIX}index-{uuid4().hex}.json"
        try:
            temporary_index.write_text(
                json.dumps({"generation": generation}, ensure_ascii=True, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(temporary_index, namespace / _INDEX_NAME)
        finally:
            temporary_index.unlink(missing_ok=True)

    @staticmethod
    def _cleanup_unreferenced_generations(namespace: Path, referenced_generation: str | None) -> None:
        generation_root = namespace / _GENERATIONS_NAME
        if not generation_root.is_dir():
            return
        for directory in generation_root.iterdir():
            if (
                directory.is_dir()
                and directory.name.startswith("generation-")
                and directory.name != referenced_generation
            ):
                shutil.rmtree(directory, ignore_errors=True)

    @staticmethod
    def _cleanup_temporary_directories(generation_root: Path) -> None:
        for directory in generation_root.glob(f"{_TEMP_PREFIX}*"):
            if directory.is_dir():
                shutil.rmtree(directory, ignore_errors=True)
            elif directory.exists():
                directory.unlink(missing_ok=True)

    @staticmethod
    def _cleanup_temporary_index_files(namespace: Path) -> None:
        for index in namespace.glob(f"{_TEMP_PREFIX}index-*.json"):
            index.unlink(missing_ok=True)

def _coerce_identity(
    identity: CacheIdentity | Mapping[str, Any], expected_kind: str
) -> CacheIdentity:
    if isinstance(identity, CacheIdentity):
        result = identity
    else:
        if not isinstance(identity, Mapping):
            raise ValueError("Cache identity must be a CacheIdentity or mapping.")
        result = CacheIdentity.from_mapping(identity)
    if result.data_kind != expected_kind:
        raise ValueError(
            f"Cache identity data_kind must be {expected_kind!r} for this operation."
        )
    return result
