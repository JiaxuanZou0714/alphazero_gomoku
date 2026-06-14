from __future__ import annotations

import argparse
import hashlib
import json
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
        policy_logits, _, value = self.model(board)
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
        default=Path("outputs/checkpoints/a100-4-prod-v3/gomoku10_best.pt"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/assets/model"),
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--chunk-mib", type=int, default=24)
    parser.add_argument(
        "--keep-onnx",
        action="store_true",
        help="Keep the full ONNX file beside the GitHub-friendly chunks.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "gomoku10_best.onnx"

    model, cfg = load_model(args.checkpoint, "cpu")
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

    chunk_size = args.chunk_mib * 1024 * 1024
    chunks = split_file(onnx_path, out_dir, chunk_size)
    manifest = {
        "format": "onnx",
        "model": onnx_path.name,
        "storage": "chunks",
        "bytes": onnx_path.stat().st_size,
        "sha256": sha256_file(onnx_path),
        "chunkSize": chunk_size,
        "chunks": chunks,
        "config": {
            "board_size": int(cfg.get("board_size", 10)),
            "win_length": int(cfg.get("win_length", 5)),
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
    if not args.keep_onnx:
        onnx_path.unlink()
        print(f"removed full ONNX file; browser app will load {len(chunks)} chunks")


if __name__ == "__main__":
    main()
