"""Lazy-import replacement for depth_anything_3/utils/export/__init__.py

The stock depth_anything_3 package eagerly imports pycolmap and gsplat, which
have no ARM64 builds. InSpatio only uses depth estimation, not 3D export.
This replacement makes those imports lazy so the package loads cleanly.

Install: copy this file to <site-packages>/depth_anything_3/utils/export/__init__.py
"""
from depth_anything_3.specs import Prediction

try:
    from .depth_vis import export_to_depth_vis
except ImportError:
    export_to_depth_vis = None

try:
    from .feat_vis import export_to_feat_vis
except ImportError:
    export_to_feat_vis = None

try:
    from .npz import export_to_mini_npz, export_to_npz
except ImportError:
    export_to_mini_npz = None
    export_to_npz = None


def _lazy_import(name):
    try:
        if name == "gs":
            from depth_anything_3.utils.export.gs import export_to_gs_ply, export_to_gs_video
            return export_to_gs_ply, export_to_gs_video
        elif name == "colmap":
            from .colmap import export_to_colmap
            return export_to_colmap
        elif name == "glb":
            from .glb import export_to_glb
            return export_to_glb
    except ImportError:
        return None


def export(prediction: Prediction, export_format: str, export_dir: str, **kwargs):
    if "-" in export_format:
        for fmt in export_format.split("-"):
            export(prediction, fmt, export_dir, **kwargs)
        return

    if export_format == "depth_vis" and export_to_depth_vis:
        export_to_depth_vis(prediction, export_dir)
    elif export_format == "npz" and export_to_npz:
        export_to_npz(prediction, export_dir)
    elif export_format == "mini_npz" and export_to_mini_npz:
        export_to_mini_npz(prediction, export_dir)
    elif export_format == "feat_vis" and export_to_feat_vis:
        export_to_feat_vis(prediction, export_dir, **kwargs.get(export_format, {}))
    elif export_format in ("gs_ply", "gs_video", "colmap", "glb"):
        result = _lazy_import(
            export_format.split("_")[0] if "gs" in export_format else export_format
        )
        if result is None:
            raise ImportError(
                f"Export format '{export_format}' requires dependencies unavailable on ARM64"
            )
    else:
        raise ValueError(f"Unsupported export format: {export_format}")


__all__ = [export]
