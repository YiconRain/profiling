#!/usr/bin/env python3
"""Download Qwen3 / Qwen3.5 models used by the kernel-launch experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Qwen3 series. Dense 0.6B/1.7B/8B plus Qwen3-30B-A3B, the small Qwen3 MoE
# (30B total / 3B active) that fits on a single 80GB H100 (the 35B Qwen3.5 MoE
# does not).
QWEN3_SPECS = {
    "Qwen3-0.6B": "Qwen/Qwen3-0.6B",
    "Qwen3-1.7B": "Qwen/Qwen3-1.7B",
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Qwen3-30B-A3B": "Qwen/Qwen3-30B-A3B",
}

# Qwen3.5 series (dense models used in the kernel-launch experiments).
# The smallest Qwen3.5 MoE (35B-A3B, ~70GB bf16) does not fit a single 80GB
# H100, so the MoE case uses Qwen3-30B-A3B instead (see QWEN3_SPECS).
QWEN35_SPECS = {
    "Qwen3.5-0.8B": "Qwen/Qwen3.5-0.8B",
    "Qwen3.5-2B": "Qwen/Qwen3.5-2B",
    "Qwen3.5-4B": "Qwen/Qwen3.5-4B",
    "Qwen3.5-9B": "Qwen/Qwen3.5-9B",
    "Qwen3.5-27B": "Qwen/Qwen3.5-27B",
}

SERIES = {
    "qwen3": QWEN3_SPECS,
    "qwen3.5": QWEN35_SPECS,
}

# Combined lookup: every known alias -> Hugging Face repo id.
ALL_SPECS = {**QWEN3_SPECS, **QWEN35_SPECS}


def project_root() -> Path:
    # This file lives at <root>/envs/download_models.py.
    return Path(__file__).resolve().parents[1]


def default_model_dir() -> Path:
    return project_root() / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Qwen3 or Qwen3.5 models from Hugging Face into a local "
            "model directory. With no selection flag, the full Qwen3.5 series "
            "is downloaded (the default for the kernel-launch experiments)."
        )
    )
    parser.add_argument(
        "--series",
        choices=[*SERIES.keys(), "all"],
        help="Download an entire model series: qwen3, qwen3.5, or all.",
    )
    parser.add_argument(
        "--model",
        help=f"Download a single model by alias. Choices: {', '.join(ALL_SPECS)}",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Explicit list of model aliases to download (overrides --series/--model).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=default_model_dir(),
        help="Directory that will contain one subdirectory per model. Default: %(default)s",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face revision, branch, tag, or commit to download.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. Usually not needed for public Qwen models.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Call snapshot_download even if the local model directory looks complete.",
    )
    return parser.parse_args()


def resolve_aliases(args: argparse.Namespace) -> list[str]:
    aliases: list[str] = []

    if args.models:
        aliases.extend(args.models)
    if args.model:
        aliases.append(args.model)
    if args.series:
        if args.series == "all":
            aliases.extend(ALL_SPECS)
        else:
            aliases.extend(SERIES[args.series])

    # Default target for the current experiments: the whole Qwen3.5 series.
    if not aliases:
        aliases.extend(QWEN35_SPECS)

    deduped, seen = [], set()
    for alias in aliases:
        if alias not in ALL_SPECS:
            valid = ", ".join(ALL_SPECS)
            raise SystemExit(f"Unknown model alias {alias!r}. Valid choices: {valid}")
        if alias not in seen:
            deduped.append(alias)
            seen.add(alias)
    return deduped


def model_looks_complete(model_path: Path) -> bool:
    if not model_path.is_dir():
        return False

    has_config = (model_path / "config.json").is_file()
    has_tokenizer = any(
        (model_path / name).is_file()
        for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "vocab.json")
    )
    has_weights = any(model_path.glob("*.safetensors")) or any(model_path.glob("*.bin"))
    has_index = any(model_path.glob("*.safetensors.index.json")) or any(
        model_path.glob("*.bin.index.json")
    )
    return has_config and has_tokenizer and (has_weights or has_index)


def download_model(alias: str, repo_id: str, model_dir: Path, revision, token, force) -> None:
    local_dir = model_dir / alias
    if model_looks_complete(local_dir) and not force:
        print(f"[skip] {alias}: found existing model at {local_dir}")
        return

    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] {repo_id} -> {local_dir}")
    snapshot_download(repo_id=repo_id, revision=revision, local_dir=str(local_dir), token=token)
    if not model_looks_complete(local_dir):
        raise SystemExit(f"Download completed, but {local_dir} does not look complete.")
    print(f"[ok] {alias}: ready at {local_dir}")


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    for alias in resolve_aliases(args):
        download_model(alias, ALL_SPECS[alias], model_dir, args.revision, args.token, args.force)

    print(f"Models directory: {model_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
