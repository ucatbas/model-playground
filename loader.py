"""
Model loader – auto-detects model type from HuggingFace and loads it.

Handles:
  - Standard HF pipeline models (text-generation, image-classification, etc.)
  - Custom models with their own code (like model.py + pytorch_model.pt)
  - GGUF / llama.cpp models
  - Diffusion models
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from rich.console import Console

console = Console()

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def parse_model_id(url_or_id: str) -> str:
    """Extract 'org/model' from a HuggingFace URL or pass through an ID."""
    url_or_id = url_or_id.strip().rstrip("/")
    # https://huggingface.co/org/model  or  https://huggingface.co/org/model/tree/main
    m = re.match(r"https?://huggingface\.co/([^/]+/[^/]+?)(?:/.*)?$", url_or_id)
    if m:
        return m.group(1)
    # Already an ID like "meta-llama/Llama-3.1-8B-Instruct"
    if "/" in url_or_id:
        return url_or_id
    raise ValueError(
        f"Can't parse model ID from: {url_or_id}\n"
        "Expected a HuggingFace URL or 'org/model-name' format."
    )


def get_model_info(model_id: str) -> dict[str, Any]:
    """Fetch model metadata from HuggingFace Hub API."""
    api = HfApi()
    try:
        info = api.model_info(model_id)
    except Exception as e:
        console.print(f"[red]Could not fetch model info for '{model_id}': {e}[/red]")
        return {}

    # Gather useful info
    files = [s.rfilename for s in (info.siblings or [])]
    tags = info.tags or []
    pipeline_tag = info.pipeline_tag or ""
    library = info.library_name or ""

    return {
        "model_id": model_id,
        "pipeline_tag": pipeline_tag,
        "library": library,
        "tags": tags,
        "files": files,
        "card_data": info.card_data.__dict__ if info.card_data else {},
    }


def detect_model_type(info: dict[str, Any]) -> str:
    """
    Classify the model into a loading strategy:
      - 'pipeline'      → standard transformers pipeline
      - 'custom'        → has model.py or custom inference code
      - 'diffusion'     → Stable Diffusion / Flux / etc.
      - 'gguf'          → quantised GGUF file
      - 'unknown'       → manual setup needed
    """
    files = info.get("files", [])
    pipeline_tag = info.get("pipeline_tag", "")
    library = info.get("library", "")
    tags = info.get("tags", [])

    # GGUF files
    if any(f.endswith(".gguf") for f in files):
        return "gguf"

    # Diffusion models
    if library == "diffusers" or "diffusers" in tags or pipeline_tag in (
        "text-to-image", "image-to-image",
    ):
        return "diffusion"

    # Custom code models (has model.py or inference script)
    if "model.py" in files or "inference.py" in files:
        return "custom"

    # Standard transformers pipeline
    if pipeline_tag:
        return "pipeline"

    # Fallback – if it has config.json it's probably transformers
    if "config.json" in files:
        return "pipeline"

    return "unknown"


def download_model(model_id: str, info: dict[str, Any] | None = None) -> Path:
    """Download all model files into models/<model_name>/."""
    safe_name = model_id.replace("/", "__")
    local_dir = MODELS_DIR / safe_name

    if local_dir.exists() and any(local_dir.iterdir()):
        console.print(f"  [green]✓[/green] Already downloaded: {local_dir}")
        return local_dir

    console.print(f"  [cyan]↓[/cyan] Downloading {model_id} → {local_dir}")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        ignore_patterns=["*.md", ".gitattributes"],
    )
    console.print(f"  [green]✓[/green] Download complete")
    return local_dir


def load_pipeline_model(model_id: str, info: dict[str, Any], device: str = "auto"):
    """Load a standard HuggingFace pipeline model."""
    from transformers import pipeline as hf_pipeline

    tag = info.get("pipeline_tag", "")
    console.print(f"  [cyan]⏳[/cyan] Loading pipeline: [bold]{tag}[/bold]")

    kwargs: dict[str, Any] = {}
    if device == "auto":
        kwargs["device_map"] = "auto"
    elif device != "cpu":
        kwargs["device"] = device

    pipe = hf_pipeline(tag, model=model_id, **kwargs)
    console.print(f"  [green]✓[/green] Pipeline ready")
    return pipe


def load_custom_model(model_id: str, info: dict[str, Any], local_dir: Path):
    """
    Load a model that ships its own model.py / inference code.
    Downloads the repo and imports the custom module.
    """
    files = info.get("files", [])

    # Find the custom module
    custom_file = None
    for candidate in ["model.py", "inference.py", "predict.py"]:
        if candidate in files:
            custom_file = candidate
            break

    if not custom_file:
        console.print("[yellow]⚠ No model.py found — falling back to pipeline[/yellow]")
        return load_pipeline_model(model_id, info)

    module_path = local_dir / custom_file
    console.print(f"  [cyan]⏳[/cyan] Loading custom module: {custom_file}")

    # Add model dir to sys.path so imports within model.py work
    model_dir_str = str(local_dir)
    if model_dir_str not in sys.path:
        sys.path.insert(0, model_dir_str)

    spec = importlib.util.spec_from_file_location("custom_model", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    console.print(f"  [green]✓[/green] Module loaded")
    console.print(f"\n  [bold]Available classes/functions:[/bold]")
    exports = [
        name for name in dir(module)
        if not name.startswith("_")
        and (callable(getattr(module, name)) or isinstance(getattr(module, name), type))
    ]
    for name in exports:
        obj = getattr(module, name)
        kind = "class" if isinstance(obj, type) else "function"
        console.print(f"    • {name} ({kind})")

    return module


def load_diffusion_model(model_id: str, info: dict[str, Any], device: str = "auto"):
    """Load a diffusion model via the diffusers library."""
    try:
        from diffusers import AutoPipelineForText2Image
    except ImportError:
        console.print("[red]✗ diffusers not installed. Run: pip install diffusers[/red]")
        return None

    console.print(f"  [cyan]⏳[/cyan] Loading diffusion pipeline…")
    import torch
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    )
    if device == "auto":
        if hasattr(pipe, "to") and torch.backends.mps.is_available():
            pipe = pipe.to("mps")
        elif hasattr(pipe, "to") and torch.cuda.is_available():
            pipe = pipe.to("cuda")
    elif device != "cpu":
        pipe = pipe.to(device)

    console.print(f"  [green]✓[/green] Diffusion pipeline ready")
    return pipe


def load_model(url_or_id: str, device: str = "auto"):
    """
    Main entry point. Give it a HuggingFace URL or model ID,
    it figures out what kind of model it is and loads it.

    Returns (model_or_pipeline, info_dict, model_type_str)
    """
    model_id = parse_model_id(url_or_id)
    console.print(f"\n[bold]🔍 Model:[/bold] {model_id}")

    # 1. Fetch metadata
    info = get_model_info(model_id)
    if not info:
        console.print("[yellow]Could not fetch metadata, will attempt download anyway[/yellow]")
        info = {"model_id": model_id, "files": [], "pipeline_tag": "", "tags": []}

    model_type = detect_model_type(info)
    console.print(f"[bold]📦 Type:[/bold]  {model_type}")
    console.print(f"[bold]🏷  Task:[/bold]  {info.get('pipeline_tag', 'n/a')}")
    console.print(f"[bold]📚 Lib:[/bold]   {info.get('library', 'n/a')}")

    # 2. Load based on type
    if model_type == "pipeline":
        model = load_pipeline_model(model_id, info, device=device)
        return model, info, model_type

    if model_type == "custom":
        local_dir = download_model(model_id, info)
        module = load_custom_model(model_id, info, local_dir)
        return module, info, model_type

    if model_type == "diffusion":
        model = load_diffusion_model(model_id, info, device=device)
        return model, info, model_type

    if model_type == "gguf":
        local_dir = download_model(model_id, info)
        gguf_files = [f for f in info.get("files", []) if f.endswith(".gguf")]
        console.print(f"\n  [bold]GGUF files downloaded to:[/bold] {local_dir}")
        for f in gguf_files:
            console.print(f"    • {f}")
        console.print(
            "\n  Load with llama-cpp-python:\n"
            f"    from llama_cpp import Llama\n"
            f'    llm = Llama(model_path="{local_dir / gguf_files[0]}")\n'
        )
        return local_dir, info, model_type

    # Unknown
    local_dir = download_model(model_id, info)
    console.print(f"\n  [yellow]⚠ Unknown model type. Files downloaded to:[/yellow] {local_dir}")
    console.print(f"  [dim]Files: {', '.join(info.get('files', [])[:15])}[/dim]")
    return local_dir, info, model_type
