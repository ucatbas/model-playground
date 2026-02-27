# AI Model Playground

A personal environment for quickly testing **any** open-source AI model from HuggingFace (or anywhere else). Paste a link → run → see results.

## Quick Start

```bash
chmod +x setup.sh && ./setup.sh   # one-time setup
source .venv/bin/activate

# Test any model from a URL or ID
python run.py https://huggingface.co/Bombek1/ai-image-detector-siglip-dinov2
python run.py google/vit-base-patch16-224
python run.py info microsoft/Phi-3-mini-4k-instruct
```

Or open [playground.ipynb](playground.ipynb) for interactive exploration.

## How It Works

1. **Paste a model URL or ID** → the loader auto-detects what kind of model it is
2. **Downloads** the model files (cached in `models/`)  
3. **Loads** using the right approach (HF pipeline, AutoModel, or custom code)
4. **Runs inference** with a sample input

### Supported Model Types

| Type | Example | What happens |
|------|---------|-------------|
| Image classification | `google/vit-base-patch16-224` | Loads with AutoModelForImageClassification, runs on sample image |
| Text generation | `microsoft/Phi-3-mini-4k-instruct` | Loads with AutoModelForCausalLM, generates text |
| Text classification | `distilbert-base-uncased-finetuned-sst-2-english` | Loads with AutoModelForSequenceClassification |
| Object detection | `facebook/detr-resnet-50` | Loads via pipeline |
| Custom models | `Bombek1/ai-image-detector-siglip-dinov2` | Downloads repo + model.py, imports the module |
| Diffusion | `stabilityai/stable-diffusion-xl-base-1.0` | Loads via diffusers |
| GGUF | `TheBloke/Llama-2-7B-GGUF` | Downloads .gguf file, shows llama.cpp usage |
| Any other | `some/model` | Downloads files, you use them manually |

## Usage

### CLI

```bash
# Load & run a model (URL is auto-detected as 'test' command)
python run.py https://huggingface.co/some/model
python run.py google/vit-base-patch16-224

# Custom input
python run.py google/vit-base-patch16-224 -i samples/my_photo.jpg
python run.py microsoft/Phi-3-mini-4k-instruct -i "Write a haiku about coding"

# Just inspect a model (no download)
python run.py info https://huggingface.co/some/model

# Download without loading
python run.py download some/model
```

### Notebook

Open `playground.ipynb` and change the `MODEL_ID` variable:

```python
MODEL_ID = "Bombek1/ai-image-detector-siglip-dinov2"  # ← change this
model, processor, info = load_any_model(MODEL_ID)
```

Then run the cells. The notebook handles everything: loading, inference, and visualisation.

## Files

```
├── run.py              # CLI – python run.py <url> to test any model
├── loader.py           # Auto-detects model type & loads it
├── playground.ipynb    # Interactive notebook for exploration
├── setup.sh            # One-command environment setup
├── pyproject.toml      # Dependencies
├── models/             # Downloaded models (gitignored)
├── samples/            # Sample input files (images, audio)
└── outputs/            # Saved results
```

## Adding Custom Model Support

If a model has unusual code that doesn't fit the auto-detection, you can always:

```python
from loader import download_model, get_model_info, parse_model_id

model_id = parse_model_id("https://huggingface.co/some/model")
info = get_model_info(model_id)
local_dir = download_model(model_id, info)

# Now use the files in local_dir however you need
import sys
sys.path.insert(0, str(local_dir))
from model import YourClass
```

## Requirements

- Python 3.10+
- ~4GB disk for a typical model
- GPU optional but recommended for large models
