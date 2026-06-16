from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from torch import nn

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from alphazero_gomoku.utils import load_model


class WebPolicyValueNet(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        policy_logits, _, value, _ = self.model(board)
        return policy_logits.float(), value.float()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def split_file(path: Path, out_dir: Path, chunk_size: int) -> list[dict]:
    chunks: list[dict] = []
    stem = path.name
    for old in out_dir.glob(f"{stem}.part*"):
        old.unlink()
    with path.open("rb") as src:
        index = 0
        while True:
            data = src.read(chunk_size)
            if not data:
                break
            chunk_name = f"{stem}.part{index:02d}"
            chunk_path = out_dir / chunk_name
            chunk_path.write_bytes(data)
            chunks.append(
                {
                    "file": chunk_name,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
            index += 1
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the PyTorch Gomoku checkpoint for the static GitHub Pages app."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/checkpoints/v1-old-best/gomoku10_best.pt"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Per-model output directory, e.g. docs/assets/models/v3. "
            "Required so a new export cannot silently overwrite another model "
            "(such as v1)."
        ),
    )
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--chunk-mib", type=int, default=24)
    parser.add_argument(
        "--fp16",
        action="store_true",
        help=(
            "Convert the ONNX weights to float16 (keeps float32 graph I/O so the "
            "browser worker needs no change). Roughly halves the download and lets "
            "the WebGPU backend run fp16 kernels."
        ),
    )
    parser.add_argument(
        "--keep-onnx",
        action="store_true",
        help="Keep the full ONNX file beside the GitHub-friendly chunks.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    if out_dir is None:
        if not args.model_id:
            parser.error(
                "--out-dir is required (e.g. docs/assets/models/v3), "
                "or pass --model-id to derive docs/assets/models/<model-id>."
            )
        out_dir = Path("docs/assets/models") / args.model_id
    # Effective model id: explicit --model-id, else derived from the target dir
    # name. This keeps the overwrite guard live even when --out-dir is given
    # without --model-id, and avoids writing a null "id" into the manifest.
    model_id = args.model_id or out_dir.name
    # Guard against silently overwriting a different model's chunks.
    existing_manifest = out_dir / "manifest.json"
    if existing_manifest.exists():
        try:
            prior = json.loads(existing_manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            prior = {}
        prior_id = prior.get("id")
        if prior_id and prior_id != model_id:
            parser.error(
                f"refusing to overwrite existing model '{prior_id}' in {out_dir} "
                f"with model-id '{model_id}'. Choose a dedicated --out-dir."
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "gomoku10_best.onnx"

    model, cfg = load_model(args.checkpoint, "cpu")
    try:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(args.checkpoint, map_location="cpu")
    wrapped = WebPolicyValueNet(model).eval()
    board_size = int(cfg.get("board_size", 10))
    dummy = torch.zeros((1, 2, board_size, board_size), dtype=torch.float32)

    torch.onnx.export(
        wrapped,
        dummy,
        onnx_path,
        input_names=["board"],
        output_names=["policy_logits", "value"],
        dynamic_axes={
            "board": {0: "batch"},
            "policy_logits": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )

    precision = "float32"
    if args.fp16:
        import onnx
        # onnxruntime's converter (vs onnxconverter_common) handles the value
        # head's internal Cast node cleanly; the latter produces a graph ORT
        # refuses to load. keep_io_types=True leaves "board"/outputs as float32
        # so engine.worker.js still feeds and reads float32 tensors unchanged.
        from onnxruntime.transformers.float16 import convert_float_to_float16

        model_fp16 = convert_float_to_float16(onnx.load(str(onnx_path)), keep_io_types=True)
        onnx.save(model_fp16, str(onnx_path))
        precision = "float16"
        print("converted ONNX weights to float16 (fp32 graph I/O preserved)")

    chunk_size = args.chunk_mib * 1024 * 1024
    chunks = split_file(onnx_path, out_dir, chunk_size)
    manifest = {
        "format": "onnx",
        "id": model_id,
        "label": args.model_label or model_id,
        "model": onnx_path.name,
        "storage": "chunks",
        "precision": precision,
        "bytes": onnx_path.stat().st_size,
        "sha256": sha256_file(onnx_path),
        "chunkSize": chunk_size,
        "chunks": chunks,
        "checkpoint": str(args.checkpoint),
        "checkpointIteration": payload.get("iteration"),
        "config": {
            "board_size": int(cfg.get("board_size", 10)),
            "win_length": int(cfg.get("win_length", 5)),
            "channels": int(cfg.get("channels", 64)),
            "residual_blocks": int(cfg.get("residual_blocks", 4)),
            "policy_channels": int(cfg.get("policy_channels", 2)),
            "value_channels": int(cfg.get("value_channels", 1)),
            "value_hidden": int(cfg.get("value_hidden", 128)),
            "mcts_batch_size": int(cfg.get("mcts_batch_size", 16)),
            "mcts_c_puct": float(cfg.get("mcts_c_puct", 1.5)),
            "mcts_root_policy_temp": float(cfg.get("mcts_root_policy_temp", 1.0)),
            "mcts_dynamic_cpuct": bool(cfg.get("mcts_dynamic_cpuct", False)),
            "mcts_fpu_reduction": float(cfg.get("mcts_fpu_reduction", 0.0) or 0.0),
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"exported {onnx_path} ({manifest['bytes']:,} bytes)")
    print(f"wrote {len(chunks)} chunks of at most {args.chunk_mib} MiB")
    if args.catalog:
        catalog_path = args.catalog
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        catalog = {"defaultModel": args.model_id, "models": []}
        if catalog_path.exists():
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        manifest_ref = os.path.relpath(out_dir / "manifest.json", catalog_path.parent).replace("\\", "/")
        entry = {
            "id": args.model_id or out_dir.name,
            "label": args.model_label or args.model_id or out_dir.name,
            "manifest": manifest_ref,
            "iteration": payload.get("iteration"),
            "bytes": manifest["bytes"],
            "arch": {
                "channels": manifest["config"]["channels"],
                "residual_blocks": manifest["config"]["residual_blocks"],
            },
        }
        models = [item for item in catalog.get("models", []) if item.get("id") != entry["id"]]
        models.append(entry)
        catalog["models"] = models
        if not catalog.get("defaultModel"):
            catalog["defaultModel"] = entry["id"]
        catalog_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"updated catalog {catalog_path}")
    if not args.keep_onnx:
        onnx_path.unlink()
        print(f"removed full ONNX file; browser app will load {len(chunks)} chunks")


if __name__ == "__main__":
    main()
