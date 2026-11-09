from __future__ import annotations

from PySide6.QtCore import (
    QByteArray,
    QAbstractListModel,
    QModelIndex,
    Property,
    Qt,
    Signal,
    Slot,
)

from lyra.models import ProjectionChannel


COLOURS = (
    "#3974d8",
    "#d94841",
    "#1f9d72",
    "#e18727",
    "#8b5cf6",
    "#0891b2",
    "#db2777",
    "#7c8b2a",
)


class ProjectionChannelModel(QAbstractListModel):
    changed = Signal()

    EnabledRole = Qt.UserRole + 1
    ElementRole = Qt.UserRole + 2
    OrbitalRole = Qt.UserRole + 3
    ColourRole = Qt.UserRole + 4

    _ROLE_NAMES = {
        EnabledRole: QByteArray(b"selected"),
        ElementRole: QByteArray(b"element"),
        OrbitalRole: QByteArray(b"orbital"),
        ColourRole: QByteArray(b"colour"),
    }
    _ROLE_KEYS = {
        EnabledRole: "enabled",
        ElementRole: "element",
        OrbitalRole: "orbital",
        ColourRole: "colour",
    }

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._rows: list[dict[str, object]] = []

    def roleNames(self):
        return self._ROLE_NAMES

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self._rows):
            return None
        key = self._ROLE_KEYS.get(role)
        return self._rows[index.row()].get(key) if key else None

    @Property(int, notify=changed)
    def count(self) -> int:
        return len(self._rows)

    @Property(bool, notify=changed)
    def rgbCompatible(self) -> bool:
        return sum(bool(row["enabled"]) for row in self._rows) <= 3

    @Slot(str, str, str)
    def add(
        self,
        element: str = "全部元素",
        orbital: str = "全部",
        colour: str = "",
    ) -> None:
        row = len(self._rows)
        if not colour:
            colour = COLOURS[row % len(COLOURS)]
        self.beginInsertRows(QModelIndex(), row, row)
        self._rows.append(
            {
                "enabled": True,
                "element": element or "全部元素",
                "orbital": orbital or "全部",
                "colour": colour,
            }
        )
        self.endInsertRows()
        self.changed.emit()

    @Slot(int)
    def remove(self, row: int) -> None:
        if not 0 <= row < len(self._rows):
            return
        self.beginRemoveRows(QModelIndex(), row, row)
        self._rows.pop(row)
        self.endRemoveRows()
        self.changed.emit()

    @Slot(int, str, "QVariant")
    def setField(self, row: int, field: str, value) -> None:
        if not 0 <= row < len(self._rows) or field not in {
            "enabled",
            "element",
            "orbital",
            "colour",
        }:
            return
        if field == "enabled":
            value = bool(value)
        else:
            value = str(value)
        if self._rows[row][field] == value:
            return
        self._rows[row][field] = value
        index = self.index(row, 0)
        role = next(role for role, key in self._ROLE_KEYS.items() if key == field)
        self.dataChanged.emit(index, index, [role])
        self.changed.emit()

    def to_channels(self, elements: tuple[str, ...]) -> tuple[ProjectionChannel, ...]:
        channels: list[ProjectionChannel] = []
        for row in self._rows:
            if not row["enabled"]:
                continue
            selected = (
                elements
                if row["element"] == "全部元素"
                else (str(row["element"]),)
            )
            orbital = None if row["orbital"] == "全部" else str(row["orbital"])
            channels.extend(
                ProjectionChannel(element, orbital, str(row["colour"]))
                for element in selected
            )
        return tuple(channels)

    def snapshot(self) -> list[dict[str, object]]:
        return [dict(row) for row in self._rows]

    def restore(self, rows) -> None:
        if not isinstance(rows, list):
            return
        clean = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            clean.append(
                {
                    "enabled": bool(row.get("enabled", True)),
                    "element": str(row.get("element") or "全部元素"),
                    "orbital": str(row.get("orbital") or "全部"),
                    "colour": str(row.get("colour") or COLOURS[len(clean) % len(COLOURS)]),
                }
            )
        self.beginResetModel()
        self._rows = clean
        self.endResetModel()
        self.changed.emit()
