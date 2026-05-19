"""Export trained-run checkpoints to ONNX. Run locally; no GPU needed.

For each ``runs/<name>/checkpoint_best.pt`` produced by ``src.train`` (or by
notebooks/06_assignment_colab.ipynb), this script:

* reloads the saved TrainConfig and model weights,
* exports the model to ``runs/<name>/model.onnx`` (opset 18, dynamic
  batch + spatial dims),
* verifies the export against PyTorch via ``onnxruntime`` and prints the
  max absolute difference (should be ``<1e-3`` for fp32).

Usage:

    python scripts/export_runs_onnx.py
    python scripts/export_runs_onnx.py --runs vgg16_pretrained custom_lwir
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

# Repo-root on sys.path so ``from src...`` works when run from anywhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.unet import build_unet_vgg16  # noqa: E402
from src.models.custom_lwir_unet import build_custom_lwir_unet  # noqa: E402

logger = logging.getLogger("export_runs_onnx")
NUM_CLASSES = 7

DEFAULT_RUNS_DIR = _REPO_ROOT / "runs"
DEFAULT_RUN_NAMES = ["vgg16_pretrained", "vgg16_scratch", "custom_lwir"]
INPUT_H, INPUT_W, INPUT_C = 512, 640, 1
OPSET = 18


def _build_model_from_saved_cfg(saved_cfg: dict) -> torch.nn.Module:
    """Construct the right model architecture from the saved config dict.

    Works with both flavours of TrainConfig we've used (the leaner one from
    notebooks/06_assignment_colab.ipynb and the fuller one from src.train) —
    we only read the fields we actually need to pick + parameterise the
    builder.
    """
    model_name = saved_cfg["model"]
    if model_name == "vgg16":
        return build_unet_vgg16(
            num_classes=NUM_CLASSES, in_channels=1,
            encoder_weights=saved_cfg.get("encoder_weights"),
        )
    if model_name == "custom_lwir":
        return build_custom_lwir_unet(
            num_classes=NUM_CLASSES, in_channels=1,
            stem_channels=saved_cfg.get("stem_channels", 48),
            transformer_layers=saved_cfg.get("transformer_layers", 2),
            use_aux_heads=saved_cfg.get("use_aux_heads", True),
        )
    raise ValueError(f"unknown model {model_name!r}")


def _inline_external_data(onnx_path: Path) -> None:
    """Fold a sibling ``.onnx.data`` weight blob back into the .onnx file.

    torch.onnx.export (dynamo backend) writes weights to an external sidecar
    by default. For downloads / deployment we want a single self-contained
    file, so we read the external initializers and rewrite them as inline
    tensors, then delete the sidecar.
    """
    import onnx
    model = onnx.load(str(onnx_path), load_external_data=True)
    for tensor in model.graph.initializer:
        if (tensor.HasField("data_location")
                and tensor.data_location == onnx.TensorProto.EXTERNAL):
            tensor.data_location = onnx.TensorProto.DEFAULT
            del tensor.external_data[:]
    onnx.save(model, str(onnx_path))
    data_path = onnx_path.with_suffix(".onnx.data")
    if data_path.exists():
        data_path.unlink()


def _find_best_checkpoint(run_dir: Path) -> Path | None:
    """Find the best-by-val-mIoU checkpoint in ``run_dir``.

    Prefers the canonical ``checkpoint_best.pt`` produced by ``run_training``.
    Falls back to any ``*checkpoint_best*.pt`` in the directory (handles
    user-renamed files like ``custom_checkpoint_best.pt`` downloaded from
    Kaggle's output panel).
    """
    canonical = run_dir / "checkpoint_best.pt"
    if canonical.exists():
        return canonical
    candidates = sorted(run_dir.glob("*checkpoint_best*.pt"))
    if not candidates:
        return None
    if len(candidates) > 1:
        logger.warning("[%s] %d 'checkpoint_best' files found; using %s",
                       run_dir.name, len(candidates), candidates[0].name)
    return candidates[0]


@torch.no_grad()
def export_one(run_dir: Path) -> Path | None:
    ckpt_path = _find_best_checkpoint(run_dir)
    if ckpt_path is None:
        logger.warning("[%s] skipped — no checkpoint_best.pt", run_dir.name)
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = _build_model_from_saved_cfg(ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()  # aux heads inactive; matches deployment graph

    onnx_path = run_dir / "model.onnx"
    dummy = torch.zeros(1, INPUT_C, INPUT_H, INPUT_W)
    dynamic_axes = {
        "input":  {0: "batch", 2: "height", 3: "width"},
        "logits": {0: "batch", 2: "height", 3: "width"},
    }
    logger.info("[%s] exporting to %s", run_dir.name, onnx_path)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["logits"],
        opset_version=OPSET, dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    # torch.onnx.export with the dynamo backend externalises weights to a
    # sibling ``model.onnx.data`` file by default. For deployment / download we
    # want a single self-contained .onnx, so fold the weights back inline
    # and remove the sidecar.
    _inline_external_data(onnx_path)

    # Verify the export.
    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    pt_out = model(dummy).numpy()
    ort_out = sess.run(["logits"], {"input": dummy.numpy()})[0]
    max_diff = float(np.abs(pt_out - ort_out).max())
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    logger.info(
        "[%s] OK  size=%.1f MB  max|pt-onnx|=%.2e  out_shape=%s",
        run_dir.name, size_mb, max_diff, tuple(ort_out.shape),
    )
    return onnx_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--runs-dir", type=Path, default=DEFAULT_RUNS_DIR,
        help=f"Directory containing the run subdirs (default {DEFAULT_RUNS_DIR}).",
    )
    p.add_argument(
        "--runs", nargs="+", default=DEFAULT_RUN_NAMES,
        help="Run subdirectory names to export (default: the three assignment configs).",
    )
    return p


def main() -> int:
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args()

    written = []
    for name in args.runs:
        out = export_one(args.runs_dir / name)
        if out is not None:
            written.append(out)
    print(f"\nWrote {len(written)} ONNX files:")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
