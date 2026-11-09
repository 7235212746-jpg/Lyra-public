from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any

from lyra.adapters.doscar import DoscarAdapter
from lyra.adapters.eigenval import EigenvalAdapter
from lyra.adapters.fermi import FermiAdapter
from lyra.adapters.procar import ProcarAdapter
from lyra.adapters.vasprun import VasprunAdapter
from lyra.cache import CACHE_SCHEMA_VERSION, CacheIdentity, CacheSource, ProjectionCache
from lyra.models import (
    BandElementIndex,
    BandStateDetail,
    BandStateSelection,
    ParsedBandData,
    ParsedDosData,
    ParsedFermiData,
    ProjectionChannel,
    ReadMode,
)
from lyra.band_inspector import (
    band_state_detail,
    build_element_index,
    whole_band_detail,
)
from lyra.projection_store import BandProjectionStore, DosProjectionStore
from lyra.paired_band import (
    PairedBandSession,
    assess_index_compatibility,
    build_standard_index,
)


PARSER_VERSION = "0.2.2"
FULL_BAND_PARSER_VERSION = "0.1.5"
ProgressCallback = Callable[[dict[str, Any]], None]
DEFAULT_HOT_MEMORY_BUDGET = int(1.5 * 1024**3)


class ProjectionDataService:
    """Resolve normalized projection data through memory, disk, then source."""

    def __init__(
        self,
        *,
        cache: ProjectionCache | None = None,
        eigenval_loader: Callable[[str | Path], ParsedBandData] = EigenvalAdapter.load,
        procar_loader: Callable[..., BandProjectionStore] = ProcarAdapter.load_store,
        vasprun_band_loader: Callable[[str | Path], ParsedBandData] = VasprunAdapter.load_band,
        vasprun_dos_loader: Callable[[str | Path], ParsedDosData] = VasprunAdapter.load_dos,
        doscar_loader: Callable[[str | Path], ParsedDosData] = DoscarAdapter.load,
        fermi_lite_loader: Callable[[str | Path], ParsedFermiData] = FermiAdapter.load_lite,
        fermi_full_loader: Callable[[str | Path], ParsedFermiData] = FermiAdapter.load_full,
        hot_memory_budget: int = DEFAULT_HOT_MEMORY_BUDGET,
    ) -> None:
        if isinstance(hot_memory_budget, bool) or not isinstance(hot_memory_budget, int):
            raise TypeError("Hot-memory budget must be an integer number of bytes.")
        if hot_memory_budget < 0:
            raise ValueError("Hot-memory budget must not be negative.")
        self.cache = cache or ProjectionCache()
        self.eigenval_loader = eigenval_loader
        self.procar_loader = procar_loader
        self.vasprun_band_loader = vasprun_band_loader
        self.vasprun_dos_loader = vasprun_dos_loader
        self.doscar_loader = doscar_loader
        self.fermi_lite_loader = fermi_lite_loader
        self.fermi_full_loader = fermi_full_loader
        self.hot_memory_budget = hot_memory_budget
        self._hot_order: OrderedDict[tuple[int, object], dict] = OrderedDict()
        self._band_memory: dict[
            tuple[Path, CacheIdentity],
            tuple[ParsedBandData, BandProjectionStore | None],
        ] = {}
        self._dos_memory: dict[
            tuple[Path, CacheIdentity],
            tuple[ParsedDosData, DosProjectionStore | None],
        ] = {}
        self._fermi_memory: dict[
            tuple[Path, CacheIdentity],
            ParsedFermiData,
        ] = {}
        self._rebuild_memory: dict[tuple[Path, str], tuple[dict, object, object]] = {}

    def load_fermi(
        self,
        folder: str | Path,
        mode: ReadMode | str,
        progress: ProgressCallback | None = None,
        *,
        identity: CacheIdentity | None = None,
        rebuild: bool = False,
    ) -> ParsedFermiData:
        resolved_folder = Path(folder).resolve()
        read_mode = ReadMode(mode)
        identity = identity or self.fermi_identity(
            resolved_folder,
            read_mode,
            progress,
        )
        key = (resolved_folder, identity)
        memory = None if rebuild else self._memory_get(self._fermi_memory, key)
        if memory is not None:
            self._emit(progress, "memory-hit", "Using in-memory Fermi-surface data.")
            return memory
        self._emit(progress, "source-parse", "Reading uniform Fermi-surface grid.")
        self._emit(
            progress,
            "cache-not-used",
            "Fermi-surface source data is retained in worker memory.",
        )
        loader = (
            self.fermi_lite_loader
            if read_mode is ReadMode.LITE
            else self.fermi_full_loader
        )
        parsed = loader(resolved_folder)
        if rebuild:
            self._stage_rebuild_memory(self._fermi_memory, key, parsed)
        else:
            self._remember(self._fermi_memory, key, parsed)
        return parsed

    def load_band(
        self,
        folder: str | Path,
        mode: ReadMode | str,
        channels: Iterable[ProjectionChannel] = (),
        progress: ProgressCallback | None = None,
        *,
        identity: CacheIdentity | None = None,
        rebuild: bool = False,
    ) -> ParsedBandData:
        resolved_folder = Path(folder).resolve()
        read_mode = ReadMode(mode)
        selected_channels = tuple(channels)

        if read_mode is ReadMode.LITE and not selected_channels:
            identity = identity or self.band_identity(
                resolved_folder, read_mode, selected_channels, progress
            )
            memory = (
                None
                if rebuild
                else self._memory_get(self._band_memory, (resolved_folder, identity))
            )
            if memory is not None:
                self._emit(progress, "memory-hit", "Using in-memory Lite Band data.")
                return memory[0]
            self._emit(progress, "source-parse", "Reading Lite Band sources.")
            self._emit(progress, "cache-not-used", "Lite standard Band reads sources directly.")
            parsed = self.eigenval_loader(resolved_folder)
            if rebuild:
                self._stage_rebuild_memory(
                    self._band_memory,
                    (resolved_folder, identity),
                    (parsed, None),
                )
            else:
                self._remember(
                    self._band_memory, (resolved_folder, identity), (parsed, None)
                )
            return parsed

        if read_mode is ReadMode.LITE:
            identity = identity or self.band_identity(
                resolved_folder, read_mode, selected_channels, progress
            )
        else:
            identity = identity or self.band_identity(
                resolved_folder, read_mode, selected_channels, progress
            )
        base_data, store = self._load_projected_band_bundle(
            resolved_folder,
            read_mode,
            identity,
            progress,
            rebuild=rebuild,
        )
        if store is None:
            if selected_channels:
                raise ValueError(
                    "Band projection data is unavailable: vasprun does not contain "
                    "projected eigenvalues. Re-run VASP with projection output enabled."
                )
            return base_data
        return store.materialize(base_data, selected_channels)

    def prepare_band_inspector(
        self,
        folder: str | Path,
        mode: ReadMode | str,
        progress: ProgressCallback | None = None,
        *,
        identity: CacheIdentity | None = None,
        reference_energy: float | None = None,
        display_channels: tuple[ProjectionChannel, ...] = (),
    ) -> tuple[BandElementIndex, CacheIdentity, ParsedBandData]:
        resolved_folder = Path(folder).resolve()
        read_mode = ReadMode(mode)
        if identity is not None and not self.identity_matches(
            resolved_folder,
            identity,
        ):
            raise ValueError(
                "Band projection sources changed; the inspector cache identity is stale."
            )
        identity = identity or self.band_identity(
            resolved_folder,
            read_mode,
            (ProjectionChannel("__all_projection_data__"),),
            progress,
        )
        base_data, store = self._load_projected_band_bundle(
            resolved_folder,
            read_mode,
            identity,
            progress,
        )
        if store is None:
            raise ValueError("Band projection data is unavailable for analysis.")
        display_data = store.materialize(base_data, display_channels)
        return (
            build_element_index(base_data, store, reference_energy),
            identity,
            display_data,
        )

    def prepare_band_zoom(
        self,
        folder: str | Path,
        mode: ReadMode | str,
        settings,
        identity: CacheIdentity,
        progress: ProgressCallback | None = None,
    ) -> dict[str, object]:
        resolved_folder = Path(folder).resolve()
        if not self.identity_matches(resolved_folder, identity):
            raise ValueError(
                "Band sources changed; regenerate the preview before local zoom."
            )
        channels = (
            tuple(settings.projection_channels)
            if settings.mode != "standard"
            else ()
        )
        data = self.load_band(
            resolved_folder,
            mode,
            channels,
            progress,
            identity=identity,
        )
        return {
            "data": data,
            "settings": settings,
            "identity": identity,
        }

    def prepare_paired_band(
        self,
        left_folder: str | Path,
        left_mode: ReadMode | str,
        right_folder: str | Path,
        right_mode: ReadMode | str,
        progress: ProgressCallback | None = None,
    ) -> PairedBandSession:
        left_path = Path(left_folder).resolve()
        right_path = Path(right_folder).resolve()

        def side_progress(side: str):
            if progress is None:
                return None

            def emit(event: dict[str, Any]) -> None:
                progress({**event, "side": side})

            return emit

        def prepare_side(path: Path, mode: ReadMode | str, side: str):
            try:
                index, identity, _display = self.prepare_band_inspector(
                    path, mode, side_progress(side)
                )
                return index, identity, True
            except FileNotFoundError as exc:
                if "PROCAR" not in str(exc) or ReadMode(mode) is not ReadMode.LITE:
                    raise
                identity = self.band_identity(path, mode, (), side_progress(side))
                data = self.load_band(
                    path, mode, (), side_progress(side), identity=identity
                )
                return build_standard_index(data), identity, False
            except ValueError as exc:
                if "projection data is unavailable" not in str(exc).lower():
                    raise
                identity = self.band_identity(path, mode, (), side_progress(side))
                data = self.load_band(
                    path, mode, (), side_progress(side), identity=identity
                )
                return build_standard_index(data), identity, False

        left_index, left_identity, left_detail = prepare_side(
            left_path, left_mode, "left"
        )
        right_index, right_identity, right_detail = prepare_side(
            right_path, right_mode, "right"
        )
        return PairedBandSession(
            left=left_index,
            right=right_index,
            left_identity=left_identity,
            right_identity=right_identity,
            left_folder=left_path,
            right_folder=right_path,
            left_mode=ReadMode(left_mode).value,
            right_mode=ReadMode(right_mode).value,
            compatibility=assess_index_compatibility(left_index, right_index),
            left_detail_available=left_detail,
            right_detail_available=right_detail,
        )

    def query_band_state(
        self,
        folder: str | Path,
        mode: ReadMode | str,
        selection: BandStateSelection,
        progress: ProgressCallback | None = None,
        *,
        identity: CacheIdentity | None = None,
        whole_band: bool = False,
        band_indices: tuple[int, ...] | None = None,
    ) -> BandStateDetail:
        resolved_folder = Path(folder).resolve()
        read_mode = ReadMode(mode)
        if identity is not None and not self.identity_matches(
            resolved_folder,
            identity,
        ):
            raise ValueError(
                "Band projection sources changed; the inspector cache identity is stale."
            )
        identity = identity or self.band_identity(
            resolved_folder,
            read_mode,
            (ProjectionChannel("__all_projection_data__"),),
            progress,
        )
        base_data, store = self._load_projected_band_bundle(
            resolved_folder,
            read_mode,
            identity,
            progress,
        )
        if store is None:
            raise ValueError("Band projection data is unavailable for analysis.")
        detail_builder = whole_band_detail if whole_band else band_state_detail
        return detail_builder(
            base_data,
            store,
            selection,
            band_indices=band_indices,
        )

    def query_cached_band_state(
        self,
        folder: str | Path,
        identity: CacheIdentity,
        selection: BandStateSelection,
        *,
        whole_band: bool = False,
        band_indices: tuple[int, ...] | None = None,
    ) -> BandStateDetail:
        """Read one inspector selection directly from the complete folder cache."""
        resolved_folder = Path(folder).resolve()
        if not self.identity_matches(resolved_folder, identity):
            raise ValueError(
                "Band projection sources changed; rebuild the analysis cache."
            )
        cached = self.cache.load_band(resolved_folder, identity, mmap=True)
        if cached is None or cached[1] is None:
            raise ValueError(
                "Complete Band projection cache is unavailable; rebuild the cache first."
            )
        base_data, store = cached
        try:
            detail_builder = whole_band_detail if whole_band else band_state_detail
            return detail_builder(
                base_data,
                store,
                selection,
                band_indices=band_indices,
            )
        finally:
            for array in (store.tensor, store.spin_components):
                mapping = getattr(array, "_mmap", None)
                if mapping is not None:
                    mapping.close()

    def load_dos(
        self,
        folder: str | Path,
        mode: ReadMode | str,
        channels: Iterable[ProjectionChannel] = (),
        progress: ProgressCallback | None = None,
        *,
        identity: CacheIdentity | None = None,
        rebuild: bool = False,
    ) -> ParsedDosData:
        resolved_folder = Path(folder).resolve()
        read_mode = ReadMode(mode)
        selected_channels = tuple(channels)

        if read_mode is ReadMode.LITE:
            identity = identity or self.dos_identity(
                resolved_folder, read_mode, selected_channels, progress
            )
            memory = (
                None
                if rebuild
                else self._memory_get(self._dos_memory, (resolved_folder, identity))
            )
            if memory is not None:
                self._emit(progress, "memory-hit", "Using in-memory Lite DOS data.")
                return memory[0]
            self._emit(progress, "source-parse", "Reading Lite DOS sources.")
            self._emit(progress, "cache-not-used", "Lite DOS reads sources directly.")
            parsed = self.doscar_loader(resolved_folder)
            if rebuild:
                self._stage_rebuild_memory(
                    self._dos_memory,
                    (resolved_folder, identity),
                    (parsed, None),
                )
            else:
                self._remember(
                    self._dos_memory, (resolved_folder, identity), (parsed, None)
                )
            return parsed

        identity = identity or self.dos_identity(
            resolved_folder, read_mode, selected_channels, progress
        )

        def source_loader() -> tuple[ParsedDosData, DosProjectionStore | None]:
            parsed = self.vasprun_dos_loader(resolved_folder)
            store = DosProjectionStore.from_parsed(parsed) if parsed.projections else None
            return replace(parsed, folder=resolved_folder, projections={}), store

        base_data, store = self._resolve_dos(
            resolved_folder, identity, source_loader, progress, rebuild=rebuild
        )
        if store is None:
            if selected_channels:
                raise ValueError(
                    "DOS projection data is unavailable: vasprun does not contain "
                    "projected DOS. Re-run VASP with projection output enabled."
                )
            return base_data
        return store.materialize(base_data, selected_channels)

    def clear(self, folder: str | Path, data_kind: str | None = None) -> None:
        resolved_folder = Path(folder).resolve()
        if data_kind in {None, "band"}:
            self._forget_folder(self._band_memory, resolved_folder)
        if data_kind in {None, "dos"}:
            self._forget_folder(self._dos_memory, resolved_folder)
        if data_kind in {None, "fermi"}:
            self._forget_folder(self._fermi_memory, resolved_folder)
        for rebuild_key in tuple(self._rebuild_memory):
            if rebuild_key[0] == resolved_folder and data_kind in {
                None,
                rebuild_key[1],
            }:
                del self._rebuild_memory[rebuild_key]
        self.cache.clear(resolved_folder, data_kind)

    def begin_rebuild(self, folder: str | Path, data_kind: str) -> str | None:
        resolved_folder = Path(folder).resolve()
        self._rebuild_memory.pop((resolved_folder, data_kind), None)
        return self.cache.active_generation(resolved_folder, data_kind)

    def promote_rebuild(self, folder: str | Path, data_kind: str) -> None:
        self.cache.promote_rebuild(folder, data_kind)

    def finish_rebuild(
        self,
        folder: str | Path,
        data_kind: str,
        previous_generation: str | None,
        *,
        success: bool,
    ) -> None:
        resolved_folder = Path(folder).resolve()
        staged = self._rebuild_memory.pop((resolved_folder, data_kind), None)
        self.cache.finish_rebuild(
            resolved_folder,
            data_kind,
            previous_generation,
            success=success,
        )
        if success and staged is not None:
            memory, key, value = staged
            self._remember(memory, key, value)

    def band_identity(
        self,
        folder: str | Path,
        mode: ReadMode | str,
        channels: Iterable[ProjectionChannel] = (),
        progress: ProgressCallback | None = None,
    ) -> CacheIdentity:
        resolved_folder = Path(folder).resolve()
        read_mode = ReadMode(mode)
        filenames = (
            ("EIGENVAL", "KPOINTS", "POSCAR", "OUTCAR")
            if read_mode is ReadMode.LITE and not tuple(channels)
            else ("EIGENVAL", "KPOINTS", "POSCAR", "OUTCAR", "PROCAR")
            if read_mode is ReadMode.LITE
            else (self._xml_name(resolved_folder), "KPOINTS")
        )
        return self._identity(resolved_folder, read_mode, "band", filenames, progress)

    def dos_identity(
        self,
        folder: str | Path,
        mode: ReadMode | str,
        channels: Iterable[ProjectionChannel] = (),
        progress: ProgressCallback | None = None,
    ) -> CacheIdentity:
        resolved_folder = Path(folder).resolve()
        read_mode = ReadMode(mode)
        filenames = (
            ("DOSCAR", "POSCAR", "OUTCAR")
            if read_mode is ReadMode.LITE
            else (self._xml_name(resolved_folder),)
        )
        return self._identity(resolved_folder, read_mode, "dos", filenames, progress)

    def fermi_identity(
        self,
        folder: str | Path,
        mode: ReadMode | str,
        progress: ProgressCallback | None = None,
    ) -> CacheIdentity:
        resolved_folder = Path(folder).resolve()
        read_mode = ReadMode(mode)
        filenames = (
            ("EIGENVAL", "KPOINTS", "POSCAR", "OUTCAR")
            if read_mode is ReadMode.LITE
            else (self._xml_name(resolved_folder),)
        )
        return self._identity(
            resolved_folder,
            read_mode,
            "fermi",
            filenames,
            progress,
        )

    @classmethod
    def identity_matches(cls, folder: str | Path, identity: CacheIdentity) -> bool:
        try:
            current = cls._identity(
                Path(folder).resolve(),
                ReadMode(identity.read_mode),
                identity.data_kind,
                tuple(source.filename for source in identity.sources),
                None,
            )
        except (FileNotFoundError, OSError, ValueError):
            return False
        return current == identity

    def _resolve_band(
        self,
        folder: Path,
        identity: CacheIdentity,
        source_loader: Callable[
            [], tuple[ParsedBandData, BandProjectionStore | None]
        ],
        progress: ProgressCallback | None,
        *,
        rebuild: bool = False,
    ) -> tuple[ParsedBandData, BandProjectionStore | None]:
        key = (folder, identity)
        memory = None if rebuild else self._memory_get(self._band_memory, key)
        if memory is not None:
            self._emit(progress, "memory-hit", "Using in-memory Band projection data.")
            return memory
        cached = None if rebuild else self.cache.load_band(folder, identity)
        if cached is not None:
            self._remember(self._band_memory, key, cached)
            self._emit(progress, "cache-hit", "Using folder Band projection cache.")
            return cached
        self._emit(
            progress,
            "cache-miss",
            "Band projection cache is missing or stale.",
            cache_state=self.cache.band_status(folder, identity),
        )
        self._emit(progress, "source-parse", "Parsing Band projection sources.")
        loaded = source_loader()
        if rebuild:
            self._stage_rebuild_memory(self._band_memory, key, loaded)
        else:
            self._remember(self._band_memory, key, loaded)
        self._save_band(
            folder, identity, loaded, progress, retain_previous=rebuild
        )
        return loaded

    def _load_projected_band_bundle(
        self,
        folder: Path,
        mode: ReadMode,
        identity: CacheIdentity,
        progress: ProgressCallback | None,
        *,
        rebuild: bool = False,
    ) -> tuple[ParsedBandData, BandProjectionStore | None]:
        if mode is ReadMode.LITE:
            def source_loader() -> tuple[ParsedBandData, BandProjectionStore]:
                base_data = self.eigenval_loader(folder)
                store = self.procar_loader(folder, base_data, progress=progress)
                return replace(
                    base_data,
                    folder=folder,
                    elements=store.elements,
                    orbitals=store.orbitals,
                    projections={},
                ), store
        else:
            def source_loader() -> tuple[ParsedBandData, BandProjectionStore | None]:
                parsed = self.vasprun_band_loader(folder)
                store = (
                    BandProjectionStore.from_parsed(parsed)
                    if parsed.projections
                    else None
                )
                return replace(parsed, folder=folder, projections={}), store
        return self._resolve_band(
            folder, identity, source_loader, progress, rebuild=rebuild
        )

    def _resolve_dos(
        self,
        folder: Path,
        identity: CacheIdentity,
        source_loader: Callable[[], tuple[ParsedDosData, DosProjectionStore | None]],
        progress: ProgressCallback | None,
        *,
        rebuild: bool = False,
    ) -> tuple[ParsedDosData, DosProjectionStore | None]:
        key = (folder, identity)
        memory = None if rebuild else self._memory_get(self._dos_memory, key)
        if memory is not None:
            self._emit(progress, "memory-hit", "Using in-memory DOS projection data.")
            return memory
        cached = None if rebuild else self.cache.load_dos(folder, identity)
        if cached is not None:
            self._remember(self._dos_memory, key, cached)
            self._emit(progress, "cache-hit", "Using folder DOS projection cache.")
            return cached
        self._emit(
            progress,
            "cache-miss",
            "DOS projection cache is missing or stale.",
            cache_state=self.cache.dos_status(folder, identity),
        )
        self._emit(progress, "source-parse", "Parsing DOS projection sources.")
        loaded = source_loader()
        if rebuild:
            self._stage_rebuild_memory(self._dos_memory, key, loaded)
        else:
            self._remember(self._dos_memory, key, loaded)
        self._save_dos(
            folder, identity, loaded, progress, retain_previous=rebuild
        )
        return loaded

    def _save_band(
        self,
        folder: Path,
        identity: CacheIdentity,
        loaded: tuple[ParsedBandData, BandProjectionStore | None],
        progress: ProgressCallback | None,
        *,
        retain_previous: bool = False,
    ) -> None:
        self._emit(progress, "cache-write", "Writing Band projection cache.")
        try:
            self.cache.save_band(
                folder, identity, *loaded, retain_previous=retain_previous
            )
            self._emit(progress, "cache-written", "Band projection cache is ready.")
        except OSError as exc:
            self._emit(
                progress,
                "cache-write-failed",
                f"Folder cache could not be written; using memory only: {exc}",
            )

    def _save_dos(
        self,
        folder: Path,
        identity: CacheIdentity,
        loaded: tuple[ParsedDosData, DosProjectionStore | None],
        progress: ProgressCallback | None,
        *,
        retain_previous: bool = False,
    ) -> None:
        self._emit(progress, "cache-write", "Writing DOS projection cache.")
        try:
            self.cache.save_dos(
                folder, identity, *loaded, retain_previous=retain_previous
            )
            self._emit(progress, "cache-written", "DOS projection cache is ready.")
        except OSError as exc:
            self._emit(
                progress,
                "cache-write-failed",
                f"Folder cache could not be written; using memory only: {exc}",
            )

    def _remember(self, memory: dict, key: tuple[Path, CacheIdentity], value) -> None:
        folder, identity = key
        stale_keys = [
            candidate
            for candidate in memory
            if candidate[0] == folder
            and getattr(candidate[1], "data_kind", candidate[1])
            == getattr(identity, "data_kind", identity)
        ]
        for stale_key in stale_keys:
            del memory[stale_key]
            self._hot_order.pop((id(memory), stale_key), None)
        memory[key] = value
        self._touch_hot_entry(memory, key)
        self._evict_hot_memory()

    def _memory_get(self, memory: dict, key):
        value = memory.get(key)
        if value is not None:
            self._touch_hot_entry(memory, key)
        return value

    def _stage_rebuild_memory(self, memory: dict, key, value) -> None:
        folder, identity = key
        self._rebuild_memory[(folder, identity.data_kind)] = (memory, key, value)

    def _forget_folder(self, memory: dict, folder: Path) -> None:
        for key in tuple(memory):
            if key[0] == folder:
                self._hot_order.pop((id(memory), key), None)
                del memory[key]

    def _touch_hot_entry(self, memory: dict, key) -> None:
        marker = (id(memory), key)
        self._hot_order.pop(marker, None)
        if key in memory:
            self._hot_order[marker] = memory

    def _prune_hot_order(self) -> None:
        for marker, memory in tuple(self._hot_order.items()):
            if marker[1] not in memory:
                del self._hot_order[marker]

    def _evict_hot_memory(self) -> None:
        self._prune_hot_order()
        while self._resident_hot_bytes() > self.hot_memory_budget and self._hot_order:
            marker, memory = self._hot_order.popitem(last=False)
            memory.pop(marker[1], None)

    def _resident_hot_bytes(self) -> int:
        seen: set[int] = set()
        return sum(
            self._estimate_hot_bytes(memory[key], seen)
            for (_memory_id, key), memory in self._hot_order.items()
            if key in memory
        )

    @classmethod
    def _estimate_hot_bytes(cls, value, seen: set[int] | None = None) -> int:
        import numpy as np
        import sys

        if seen is None:
            seen = set()
        identity = id(value)
        if identity in seen:
            return 0
        seen.add(identity)
        if isinstance(value, np.ndarray):
            return int(value.nbytes)
        if isinstance(value, (bytes, bytearray, memoryview)):
            return len(value)
        if isinstance(value, str):
            return len(value.encode("utf-8"))
        if isinstance(value, Mapping):
            return sys.getsizeof(value) + sum(
                cls._estimate_hot_bytes(key, seen)
                + cls._estimate_hot_bytes(item, seen)
                for key, item in value.items()
            )
        if is_dataclass(value) and not isinstance(value, type):
            return sys.getsizeof(value) + sum(
                cls._estimate_hot_bytes(getattr(value, field.name), seen)
                for field in fields(value)
            )
        if isinstance(value, Sequence):
            return sys.getsizeof(value) + sum(
                cls._estimate_hot_bytes(item, seen) for item in value
            )
        if hasattr(value, "__dict__"):
            return cls._estimate_hot_bytes(vars(value), seen)
        return sys.getsizeof(value)

    @classmethod
    def _identity(
        cls,
        folder: Path,
        mode: ReadMode,
        data_kind: str,
        filenames: tuple[str, ...],
        progress: ProgressCallback | None,
    ) -> CacheIdentity:
        cls._emit(progress, "validate", f"Validating {data_kind} source identity.")
        sources = []
        for filename in filenames:
            path = folder / filename
            try:
                stat = path.stat()
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"Required {data_kind} source {filename} was not found in {folder}."
                ) from exc
            if not path.is_file():
                raise FileNotFoundError(
                    f"Required {data_kind} source {filename} is not a file in {folder}."
                )
            sources.append(CacheSource(filename, stat.st_size, stat.st_mtime_ns))
        return CacheIdentity(
            schema_version=CACHE_SCHEMA_VERSION,
            parser_version=cls._parser_version(mode, data_kind),
            read_mode=mode.value,
            data_kind=data_kind,
            sources=tuple(sources),
        )

    @staticmethod
    def _parser_version(mode: ReadMode, data_kind: str) -> str:
        if mode is ReadMode.FULL and data_kind == "band":
            return FULL_BAND_PARSER_VERSION
        return PARSER_VERSION

    @staticmethod
    def _xml_name(folder: Path) -> str:
        for name in ("vasprun.xml", "vasprun.xml.gz"):
            if (folder / name).is_file():
                return name
        raise FileNotFoundError(
            f"No vasprun.xml or vasprun.xml.gz was found in {folder}."
        )

    @staticmethod
    def _emit(
        progress: ProgressCallback | None,
        phase: str,
        detail: str,
        current: int = 0,
        total: int = 0,
        **extra: Any,
    ) -> None:
        if progress is not None:
            progress(
                {
                    "phase": phase,
                    "current": current,
                    "total": total,
                    "detail": detail,
                    **extra,
                }
            )
