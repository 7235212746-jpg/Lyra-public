import tempfile
import tkinter as tk
import unittest
from pathlib import Path

from PIL import Image

from lyra.preview_viewer import PreviewImageViewer, clamp_scale, fit_scale


class PreviewViewerGeometryTests(unittest.TestCase):
    def test_fit_scale_contains_image_without_upscaling(self):
        self.assertEqual(fit_scale((1200, 800), (600, 600)), 0.5)
        self.assertEqual(fit_scale((400, 300), (1000, 800)), 1.0)

    def test_fit_scale_handles_unavailable_viewport(self):
        self.assertEqual(fit_scale((1200, 800), (0, 0)), 1.0)

    def test_clamp_scale_enforces_viewer_bounds(self):
        self.assertEqual(clamp_scale(0.001), 0.05)
        self.assertEqual(clamp_scale(2.0), 2.0)
        self.assertEqual(clamp_scale(20.0), 8.0)


class PreviewImageViewerTests(unittest.TestCase):
    def test_viewer_binds_zoom_pan_and_fit_interactions(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "preview.png"
            Image.new("RGB", (1200, 800), "white").save(image_path)
            root = tk.Tk()
            root.withdraw()
            try:
                viewer = PreviewImageViewer(root, image_path)
                viewer.geometry("700x500+0+0")
                viewer.update()
                viewer.fit_to_window()

                self.assertLessEqual(viewer.scale, 1.0)
                self.assertTrue(viewer.canvas.bind("<MouseWheel>"))
                self.assertTrue(viewer.canvas.bind("<B1-Motion>"))
                self.assertTrue(viewer.canvas.bind("<Double-Button-1>"))

                original_scale = viewer.scale
                event = type("WheelEvent", (), {"delta": 120, "x": 200, "y": 150})()
                viewer._zoom(event)
                self.assertGreater(viewer.scale, original_scale)
            finally:
                root.destroy()


if __name__ == "__main__":
    unittest.main()
