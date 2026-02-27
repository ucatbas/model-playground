#!/usr/bin/env python3
"""
run.py – Your one-command model tester.

Usage:
  python run.py https://huggingface.co/Bombek1/ai-image-detector-siglip-dinov2
  python run.py meta-llama/Llama-3.1-8B-Instruct
  python run.py info https://huggingface.co/some/model
  python run.py download https://huggingface.co/some/model
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from loader import parse_model_id, get_model_info, detect_model_type, download_model, load_model

console = Console()

SAMPLE_INPUTS = {
    "text-generation": "Explain quantum computing in simple terms.",
    "text2text-generation": "Translate to French: Hello, how are you?",
    "summarization": "Summarize: Machine learning is a subset of AI that focuses on building systems that learn from data.",
    "question-answering": {"question": "What is Python?", "context": "Python is a programming language created by Guido van Rossum in 1991."},
    "fill-mask": "The capital of France is [MASK].",
    "sentiment-analysis": "I absolutely love this product, it changed my life!",
    "text-classification": "I absolutely love this product, it changed my life!",
    "token-classification": "Hugging Face Inc. is a company based in New York City.",
    "translation": "Hello, how are you today?",
    "zero-shot-classification": "I love playing soccer on weekends",
    "image-classification": "samples/sample.jpg",
    "object-detection": "samples/sample.jpg",
    "image-segmentation": "samples/sample.jpg",
    "text-to-image": "A beautiful sunset over mountains, digital art",
    "automatic-speech-recognition": "samples/sample.wav",
    "feature-extraction": "This is a test sentence for embedding.",
}


# ── Commands ─────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """AI Model Playground – test any HuggingFace model locally."""
    pass


@cli.command("test")
@click.argument("model")
@click.option("--device", "-d", default="auto", help="Device: auto, cpu, cuda, mps")
@click.option("--input", "-i", "user_input", default=None, help="Custom input (text, file path, etc.)")
def test_cmd(model, device, user_input):
    """Load and test a model (default command)."""
    _run_model(model, device, user_input)


@cli.command("info")
@click.argument("model")
def info_cmd(model):
    """Show model info without downloading."""
    model_id = parse_model_id(model)
    info = get_model_info(model_id)
    model_type = detect_model_type(info)

    table = Table(title=f"Model: {model_id}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Pipeline Tag", info.get("pipeline_tag", "n/a"))
    table.add_row("Library", info.get("library", "n/a"))
    table.add_row("Detected Type", model_type)
    table.add_row("Tags", ", ".join(info.get("tags", [])[:10]))

    files = info.get("files", [])
    table.add_row("Files", f"{len(files)} files")
    for f in files[:20]:
        table.add_row("", f"  {f}")
    if len(files) > 20:
        table.add_row("", f"  … and {len(files) - 20} more")

    console.print(table)


@cli.command("download")
@click.argument("model")
def download_cmd(model):
    """Download model files without loading."""
    model_id = parse_model_id(model)
    info = get_model_info(model_id)
    local_dir = download_model(model_id, info)
    console.print(f"\n[green]✓ Downloaded to: {local_dir}[/green]")


# ── Core logic ───────────────────────────────────────────────────────────────

def _run_model(model: str, device: str, user_input: str | None):
    result, info, model_type = load_model(model, device=device)

    if result is None:
        return

    pipeline_tag = info.get("pipeline_tag", "")

    if model_type == "pipeline" and result is not None:
        inp = user_input or SAMPLE_INPUTS.get(pipeline_tag)
        if inp:
            console.print(f"\n[bold]▶ Running with sample input:[/bold]")
            console.print(f"  [dim]{str(inp)[:200]}[/dim]\n")

            if isinstance(inp, str) and inp.endswith((".jpg", ".png", ".jpeg", ".webp")):
                inp_path = Path(inp)
                if not inp_path.exists():
                    _ensure_sample_image(inp_path)
                if inp_path.exists():
                    from PIL import Image
                    inp = Image.open(inp_path)
                else:
                    console.print(f"[yellow]⚠ Sample image not found: {inp}[/yellow]")
                    console.print(f"  Place an image at [bold]{inp}[/bold] and re-run")
                    return

            try:
                output = result(inp)
                console.print("[bold green]Result:[/bold green]")
                _pretty_print(output)
            except Exception as e:
                console.print(f"[red]Error during inference: {e}[/red]")
        else:
            console.print(f"\n[yellow]No sample input for task '{pipeline_tag}'.[/yellow]")
            console.print(f"  Re-run with: python run.py test '{model}' -i 'your input here'")

    elif model_type == "custom":
        _print_custom_usage(info, result)

    console.print(f"\n[dim]💡 Tip: Open [bold]playground.ipynb[/bold] for interactive exploration[/dim]")


def _ensure_sample_image(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        console.print(f"  [cyan]↓[/cyan] Downloading sample image…")
        import requests
        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
        try:
            r = requests.get(url, timeout=15)
            path.write_bytes(r.content)
            console.print(f"  [green]✓[/green] Saved to {path}")
        except Exception:
            console.print(f"  [yellow]⚠ Could not download sample image[/yellow]")


def _pretty_print(output):
    if isinstance(output, list):
        for item in output[:10]:
            if isinstance(item, dict):
                parts = []
                for k, v in item.items():
                    if isinstance(v, float):
                        parts.append(f"{k}: {v:.4f}")
                    else:
                        parts.append(f"{k}: {v}")
                console.print(f"  {' | '.join(parts)}")
            else:
                console.print(f"  {item}")
        if len(output) > 10:
            console.print(f"  … and {len(output) - 10} more")
    elif isinstance(output, dict):
        for k, v in output.items():
            if isinstance(v, float):
                console.print(f"  {k}: {v:.4f}")
            elif isinstance(v, str) and len(v) > 300:
                console.print(f"  {k}: {v[:300]}…")
            else:
                console.print(f"  {k}: {v}")
    elif isinstance(output, str):
        console.print(f"  {output[:500]}")
    else:
        console.print(f"  {output}")


def _print_custom_usage(info: dict, module):
    model_id = info.get("model_id", "")
    safe_name = model_id.replace("/", "__")
    local_dir = Path("models") / safe_name

    console.print(f"\n[bold]📝 This model has custom code. Quick start:[/bold]\n")
    console.print(f'  [cyan]# In a notebook or script:[/cyan]')
    console.print(f'  from loader import load_model')
    console.print(f'  module, info, _ = load_model("{model_id}")')
    console.print(f'')
    console.print(f'  [cyan]# Or import the custom code directly:[/cyan]')
    console.print(f'  import sys')
    console.print(f'  sys.path.insert(0, "{local_dir}")')

    files = info.get("files", [])
    if "model.py" in files:
        console.print(f'  from model import *')
    console.print()


# ── Allow `python run.py <url>` as shortcut for `python run.py test <url>` ──
def main():
    """Handle the case where user just passes a model URL without 'test' subcommand."""
    args = sys.argv[1:]

    # If first arg looks like a model URL/ID (not a known command), insert "test"
    known_commands = {"test", "info", "download", "--help", "-h"}
    if args and args[0] not in known_commands and not args[0].startswith("-"):
        sys.argv.insert(1, "test")

    cli()


if __name__ == "__main__":
    main()
