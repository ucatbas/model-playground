#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# setup.sh – one-command environment setup for the AI Model Playground
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${GREEN}✓${NC} $*"; }
warn()  { echo -e "${YELLOW}⚠${NC} $*"; }

cd "$(dirname "$0")"

echo -e "\n${BOLD}🧪 AI Model Playground – Setup${NC}\n"

# ── 1. Virtual environment ──────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment…"
    python3 -m venv .venv
    info "Created .venv"
else
    info ".venv already exists"
fi

source .venv/bin/activate
info "Activated .venv ($(python3 --version))"

# ── 2. Core dependencies ────────────────────────────────────────────────────
echo ""
echo "Installing dependencies (this may take a few minutes on first run)…"
pip install --upgrade pip -q
pip install -e . -q
info "Core dependencies installed"

# ── 3. Create dirs ──────────────────────────────────────────────────────────
mkdir -p models samples outputs
info "Created models/ samples/ outputs/"

# ── 4. Optional extras ──────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Optional extras:${NC}"

read -p "  Install diffusers (image generation)? [y/N] " -n 1 -r; echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install -e ".[diffusion]" -q
    info "Diffusers installed"
fi

read -p "  Install audio support (whisper, etc.)? [y/N] " -n 1 -r; echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install -e ".[audio]" -q
    info "Audio packages installed"
fi

read -p "  Install bitsandbytes (4-bit quantisation)? [y/N] " -n 1 -r; echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install -e ".[quantize]" -q
    info "bitsandbytes installed"
fi

# ── 5. Done ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}✅ Setup complete!${NC}\n"
echo "  source .venv/bin/activate"
echo ""
echo "  # CLI: test any model"
echo "  python run.py https://huggingface.co/Bombek1/ai-image-detector-siglip-dinov2"
echo "  python run.py google/vit-base-patch16-224"
echo "  python run.py info meta-llama/Llama-3.1-8B-Instruct"
echo ""
echo "  # Notebook: interactive exploration"
echo "  jupyter notebook playground.ipynb"
echo ""
