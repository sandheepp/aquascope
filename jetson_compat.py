"""
Jetson Orin Nano compatibility shims.

MUST be imported before torch, torchvision, or ultralytics.
Fixes two issues with the Jetson nv24.08 PyTorch build:

1. Qt crash when no X display is present (sets QT_QPA_PLATFORM=offscreen).
2. torchvision C++ extension not compiled → torchvision::nms missing from
   the dispatch registry, causing _meta_registrations to crash on import.
   Fix: register a pure-Python NMS + patch torchvision.ops.nms.
"""

import os

# ── 1. Qt display fix ─────────────────────────────────────
if not os.environ.get("DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ── 2. torchvision::nms dispatch stub ────────────────────
import torch

try:
    torch.ops.torchvision.nms  # already registered — C++ ext loaded, nothing to do
except AttributeError:
    _tv_lib = torch.library.Library("torchvision", "DEF")
    _tv_lib.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")

    def _nms_cpu(dets: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
        if dets.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort(descending=True)
        keep = []
        while order.numel() > 0:
            i = order[0].item()
            keep.append(i)
            if order.numel() == 1:
                break
            xx1 = x1[order[1:]].clamp(min=x1[i].item())
            yy1 = y1[order[1:]].clamp(min=y1[i].item())
            xx2 = x2[order[1:]].clamp(max=x2[i].item())
            yy2 = y2[order[1:]].clamp(max=y2[i].item())
            inter = (xx2 - xx1).clamp(0) * (yy2 - yy1).clamp(0)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[1:][iou <= iou_threshold]
        return torch.tensor(keep, dtype=torch.long)

    torch.library.impl(_tv_lib, "nms", "CPU")(_nms_cpu)
    torch.library.impl(_tv_lib, "nms", "CUDA")(
        lambda d, s, t: _nms_cpu(d.cpu(), s.cpu(), t)
    )

# ── 3. Patch torchvision.ops.nms (bypasses C++ extension guard) ──
import torchvision.ops
import torchvision.ops.boxes as _tv_boxes


def _nms_via_dispatch(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    return torch.ops.torchvision.nms(boxes, scores, float(iou_threshold))


torchvision.ops.nms = _nms_via_dispatch
_tv_boxes.nms = _nms_via_dispatch
