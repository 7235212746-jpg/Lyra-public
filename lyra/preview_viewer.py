from __future__ import annotations

from pathlib import Path
import math
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from lyra.theme import COLORS, FONTS


MAX_RENDER_PIXELS = 24_000_000


def clamp_scale(
    scale: float,
    minimum: float = 0.05,
    maximum: float = 8.0,
) -> float:
    return max(minimum, min(maximum, scale))


def fit_scale(
    image_size: tuple[int, int],
    viewport_size: tuple[int, int],
) -> float:
    image_width, image_height = image_size
    viewport_width, viewport_height = viewport_size
    if min(image_width, image_height, viewport_width, viewport_height) <= 0:
        return 1.0
    return min(1.0, viewport_width / image_width, viewport_height / image_height)


class PreviewImageViewer(tk.Toplevel):
    def __init__(self, parent: tk.Misc, image_path: str | Path) -> None:
        super().__init__(parent)
        self.image_path = Path(image_path)
        with Image.open(self.image_path) as source:
            self._source_image = source.convert("RGB")

        self.title(f"Lyra - {self.image_path.name}")
        self.geometry("1100x760")
        self.minsize(640, 420)
        self.configure(bg=COLORS["background"])
        self.transient(parent)

        self.scale = 1.0
        self._origin = (0.0, 0.0)
        self._drag_start: tuple[int, int, float, float] | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._image_item: int | None = None
        self._fit_mode = True
        self._max_scale = max(
            1.0,
            min(
                8.0,
                math.sqrt(
                    MAX_RENDER_PIXELS
                    / (self._source_image.width * self._source_image.height)
                ),
            ),
        )

        self._build_layout()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after_idle(self.fit_to_window)

    def _build_layout(self) -> None:
        toolbar = ttk.Frame(self, padding=(10, 8))
        toolbar.pack(fill="x")
        ttk.Label(
            toolbar,
            text=self.image_path.name,
            font=FONTS["body"],
        ).pack(side="left")
        ttk.Button(
            toolbar,
            text="-",
            width=3,
            command=lambda: self._zoom_at(1 / 1.2),
        ).pack(side="right", padx=(4, 0))
        ttk.Button(
            toolbar,
            text="+",
            width=3,
            command=lambda: self._zoom_at(1.2),
        ).pack(side="right", padx=(4, 0))
        ttk.Button(
            toolbar,
            text="适合窗口",
            command=self.fit_to_window,
        ).pack(side="right", padx=(4, 0))
        ttk.Button(toolbar, text="关闭", command=self.close).pack(side="right")

        self.canvas = tk.Canvas(
            self,
            background=COLORS["panel_alt"],
            borderwidth=0,
            highlightthickness=0,
            cursor="fleur",
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<MouseWheel>", self._zoom)
        self.canvas.bind("<ButtonPress-1>", self._start_pan)
        self.canvas.bind("<B1-Motion>", self._pan)
        self.canvas.bind("<Double-Button-1>", self._fit_event)

    def fit_to_window(self) -> None:
        if not self.winfo_exists():
            return
        self.update_idletasks()
        viewport = (
            max(1, self.canvas.winfo_width() - 24),
            max(1, self.canvas.winfo_height() - 24),
        )
        self.scale = fit_scale(self._source_image.size, viewport)
        rendered_size = self._rendered_size()
        self._origin = (
            (self.canvas.winfo_width() - rendered_size[0]) / 2,
            (self.canvas.winfo_height() - rendered_size[1]) / 2,
        )
        self._fit_mode = True
        self._render()

    def _fit_event(self, _event: tk.Event) -> str:
        self.fit_to_window()
        return "break"

    def _on_canvas_configure(self, _event: tk.Event) -> None:
        if self._fit_mode:
            self.fit_to_window()

    def _zoom(self, event: tk.Event) -> str:
        if event.delta == 0:
            return "break"
        factor = 1.2 if event.delta > 0 else 1 / 1.2
        self._zoom_at(factor, event.x, event.y)
        return "break"

    def _zoom_at(
        self,
        factor: float,
        pointer_x: float | None = None,
        pointer_y: float | None = None,
    ) -> None:
        old_scale = self.scale
        new_scale = clamp_scale(
            old_scale * factor,
            minimum=0.05,
            maximum=self._max_scale,
        )
        if new_scale == old_scale:
            return

        x = self.canvas.winfo_width() / 2 if pointer_x is None else pointer_x
        y = self.canvas.winfo_height() / 2 if pointer_y is None else pointer_y
        image_x = (x - self._origin[0]) / old_scale
        image_y = (y - self._origin[1]) / old_scale
        self._origin = (
            x - image_x * new_scale,
            y - image_y * new_scale,
        )
        self.scale = new_scale
        self._fit_mode = False
        self._render()

    def _start_pan(self, event: tk.Event) -> None:
        self._drag_start = (
            event.x,
            event.y,
            self._origin[0],
            self._origin[1],
        )

    def _pan(self, event: tk.Event) -> None:
        if self._drag_start is None:
            return
        start_x, start_y, origin_x, origin_y = self._drag_start
        self._origin = (
            origin_x + event.x - start_x,
            origin_y + event.y - start_y,
        )
        self._fit_mode = False
        if self._image_item is not None:
            self.canvas.coords(self._image_item, *self._origin)

    def _rendered_size(self) -> tuple[int, int]:
        return (
            max(1, round(self._source_image.width * self.scale)),
            max(1, round(self._source_image.height * self.scale)),
        )

    def _render(self) -> None:
        rendered = self._source_image.resize(
            self._rendered_size(),
            Image.Resampling.LANCZOS,
        )
        self._photo = ImageTk.PhotoImage(rendered, master=self)
        if self._image_item is None:
            self._image_item = self.canvas.create_image(
                *self._origin,
                image=self._photo,
                anchor="nw",
            )
        else:
            self.canvas.coords(self._image_item, *self._origin)
            self.canvas.itemconfigure(self._image_item, image=self._photo)

    def close(self) -> None:
        self._photo = None
        self._source_image.close()
        self.destroy()
