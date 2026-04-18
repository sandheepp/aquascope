"""
Frame enhancement pipeline for AquaScope.

Applies CLAHE contrast normalisation, LAB colour-space saturation boost,
and a sharpening kernel — all in a single colour-space round-trip.
"""

import cv2
import numpy as np


class FrameEnhancer:
    """Stateful enhancer that reuses a single CLAHE object across frames."""

    def __init__(self, clip_limit: float = 3.0, tile_grid: tuple[int, int] = (8, 8)):
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """Return an enhanced copy of *frame*.

        Steps:
        1. CLAHE on the L channel (luminance contrast).
        2. Saturation boost on the a/b channels (colour vibrancy).
        3. Unsharp-mask-style sharpening kernel.
        """
        # 1 & 2 — single LAB round-trip
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
        ab = lab[:, :, 1:].astype(np.float32)
        lab[:, :, 1:] = np.clip((ab - 128) * 1.4 + 128, 0, 255).astype(np.uint8)
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 3 — sharpening kernel (single filter pass)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        frame = cv2.filter2D(frame, -1, kernel)

        return frame
