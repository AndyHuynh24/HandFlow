# HandFlow - Complete Project Guide

A gesture recognition ML project using deep learning. This guide covers everything from setup to training to experiments.

---

  # Basic benchmark (all architectures)                      
  python scripts/benchmark_inference.py                      
                                                             
  # More iterations for accurate results                     
  python scripts/benchmark_inference.py --iterations 500     
  --warmup 50                                                
                                                             
  # Include TFLite comparison                                
  python scripts/benchmark_inference.py --include-tflite     
                                                             
  # Benchmark specific architectures only                    
  python scripts/benchmark_inference.py --architectures lstm 
  tcn transformer                                            
                                                             
  # Different batch size                                     
  python scripts/benchmark_inference.py --batch-size 8       
                                                             
  Output will show a table with:                             
  - Mean/Std/Min/P95 inference times in milliseconds         
  - Parameter count for each model                           
  - Fastest model summary at the end     

## Table of Contents

1. [Quick Start](#quick-start)
2. [Project Overview](#project-overview)
3. [Environment Setup](#environment-setup)
4. [Project Structure](#project-structure)
5. [Configuration](#configuration)
6. [Training Pipeline](#training-pipeline)
7. [Model Export](#model-export)
8. [Experiment Tracking with W&B](#experiment-tracking-with-wb)
9. [Testing](#testing)
10. [Running the App](#running-the-app)

---

## Quick Start

```bash
# 1. Create environment
conda create -n handflow python=3.10 -y
conda activate handflow

# 2. Install
cd HandFlow
pip install -e ".[dev]"

# 3. Train (if you have data)
python scripts/train.py --hand right --architecture gru --epochs 100

# 4. Run
handflow run
```

---

## Project Overview

### What It Does

HandFlow recognizes hand gestures using computer vision and maps them to computer actions:

```
Camera → MediaPipe → Feature Engineering → Neural Network → Action
  📷        ✋              🔧                  🧠           ⌨️
```

### Key Features

| Feature | Description |
|---------|-------------|
| **4 Model Architectures** | LSTM, GRU, CNN1D, Transformer |
| **Feature Engineering** | Velocity, acceleration, finger angles |
| **Data Augmentation** | On-the-fly during training |
| **Experiment Tracking** | Weights & Biases integration |
| **Model Quantization** | TFLite for faster inference |

---

## Environment Setup

### Option A: Conda (Recommended)

```bash
# Create environment
conda create -n handflow python=3.10 -y
conda activate handflow

# Install package
pip install -e ".[dev]"

# macOS only: Install Quartz bindings
pip install pyobjc-framework-Quartz
```

### Option B: Python venv

```bash
# Create virtual environment
python3 -m venv .venv

# Activate
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Install
pip install -e ".[dev]"
```

### Verify Installation

```bash
handflow info
```

You should see:
```
🖐️ HandFlow Configuration
Version: 1.0.0
Architecture: gru
...
```

---

## Project Structure

```
HandFlow/
├── config/                  # Configuration files
│   ├── config.yaml         # Main settings (model, training, features)
│   ├── gestures.yaml       # Gesture definitions
│   └── logging.yaml        # Logging configuration
│
├── data/                    # Data (not in git - use DVC)
│   ├── raw/MP_Data/        # Raw keypoint sequences
│   └── processed/          # Preprocessed data (cached)
│
├── models/                  # Trained models
│   ├── right_action.h5     # Right hand model
│   └── left_action.h5      # Left hand model
│
├── src/handflow/            # Main Python package
│   ├── data/               # Data loading & augmentation
│   ├── features/           # Keypoint extraction & engineering
│   ├── models/             # Architectures, training, inference
│   ├── actions/            # Mouse control, action execution
│   └── utils/              # Configuration & logging
│
├── scripts/                 # Standalone scripts
│   ├── train.py            # Training script
│   ├── preprocess.py       # Data preprocessing
│   └── quantize_model.py   # TFLite conversion
│
└── tests/                   # Test suite
```

---

## Configuration

All settings are in `config/config.yaml`:

### Model Settings

```yaml
model:
  architecture: gru      # lstm, gru, cnn1d, transformer
  sequence_length: 16    # frames per sequence
  hidden_units: 128      # neurons in hidden layers
  dropout: 0.3           # regularization
  num_classes: 8         # number of gestures
```

### Feature Engineering

```yaml
features:
  velocity: true         # +84 dimensions (motion)
  acceleration: true     # +84 dimensions (acceleration)
  finger_angles: true    # +15 dimensions (joint angles)
  hand_bbox_size: true   # +4 dimensions (hand size)
```

**Total dimensions with all features:** 84 (base) + 84 + 84 + 15 + 4 = **271**

### Training Settings

```yaml
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  early_stopping_patience: 15
  validation_split: 0.2
```

---

## Training Pipeline

### Step 1: Prepare Data

Your data should be in this structure:
```
data/raw/MP_Data/
├── none/           # gesture name
│   ├── 0/          # sequence 0
│   │   ├── 0.npy   # frame 0 (keypoints)
│   │   ├── 1.npy   # frame 1
│   │   └── ...
│   ├── 1/          # sequence 1
│   └── ...
├── swiperight/
└── ...
```

Each `.npy` file contains 84 values (21 landmarks × 4 values: x, y, z, visibility).

### Step 2: Train

```bash
# Basic training
python scripts/train.py --hand right --architecture gru --epochs 100

# Full options
python scripts/train.py \
    --hand right \
    --architecture gru \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --use-wandb           # Enable experiment tracking
```

### What Happens During Training

1. **Data Loading**: Loads raw `.npy` files or cached `.npz` files
2. **Feature Engineering**: Adds velocity, angles, etc.
3. **Data Augmentation**: Applied on-the-fly (noise, time warp, rotation)
4. **Training**: With early stopping and learning rate reduction
5. **Logging**: Metrics sent to Weights & Biases
6. **Saving**: Model saved to `models/right_action.h5`

### Available Architectures

| Architecture | Best For | Speed |
|--------------|----------|-------|
| `gru` | General use (recommended) | Fast |
| `lstm` | Complex temporal patterns | Medium |
| `cnn1d` | Edge deployment | Fastest |
| `transformer` | Large datasets | Slow |

---

## Model Export

### Quantize to TFLite

Convert your model for faster inference:

```bash
python scripts/quantize_model.py \
    --input models/right_action.h5 \
    --output models/right_action.tflite \
    --quantization dynamic
```

**Results:**
- Original: ~2.5 MB
- Quantized: ~0.6 MB (75% smaller!)
- Inference: 2-3x faster

### Quantization Options

| Type | Size Reduction | Accuracy Impact |
|------|---------------|-----------------|
| `dynamic` | ~75% | Minimal |
| `float16` | ~50% | None |
| `int8` | ~75% | Slight |

---

## Experiment Tracking with W&B

Weights & Biases (W&B) helps you track experiments, compare models, and visualize training.

### Step 1: Create a W&B Account

1. Go to [wandb.ai](https://wandb.ai)
2. Click **Sign Up** (free for personal use)
3. Verify your email

### Step 2: Get Your API Key

1. After logging in, go to [wandb.ai/authorize](https://wandb.ai/authorize)
2. Copy your API key (looks like: `abc123def456...`)

### Step 3: Login from Terminal

```bash
wandb login
```

Paste your API key when prompted. This saves it to `~/.netrc` so you only do this once.

### Step 4: Train with W&B Enabled

```bash
python scripts/train.py --hand right --use-wandb
```

### What Gets Logged

When you run training with W&B:

| Metric | Description |
|--------|-------------|
| `train/loss` | Training loss per epoch |
| `train/accuracy` | Training accuracy |
| `val/loss` | Validation loss |
| `val/accuracy` | Validation accuracy |
| `learning_rate` | Current learning rate |
| Config | All hyperparameters |

### Step 5: View Your Experiments

1. Go to [wandb.ai](https://wandb.ai)
2. Find your project (default: `handflow`)
3. Click on a run to see:
   - **Charts**: Loss & accuracy over time
   - **Config**: All hyperparameters used
   - **System**: GPU usage, runtime

### Comparing Experiments

The power of W&B is comparing runs:

1. Train with different settings:
   ```bash
   python scripts/train.py --hand right --architecture gru --epochs 100 --use-wandb
   python scripts/train.py --hand right --architecture lstm --epochs 100 --use-wandb
   python scripts/train.py --hand right --architecture transformer --epochs 100 --use-wandb
   ```

2. In W&B dashboard, select multiple runs and compare:
   - Which architecture trained faster?
   - Which achieved better accuracy?
   - Which one overfit?

### Disable W&B

If you want to train without logging:

```bash
python scripts/train.py --hand right --no-wandb
```

### W&B Tips

- **Run names**: W&B auto-generates names like `stellar-river-42`
- **Tags**: Add tags to organize experiments
- **Notes**: Add notes describing what you changed
- **Groups**: Group related experiments together

---

## Testing

### Run All Tests

```bash
pytest tests/ -v -o "addopts="
```

### Run Specific Tests

```bash
# Config tests
pytest tests/unit/test_config.py -v -o "addopts="

# Architecture tests
pytest tests/unit/test_architectures.py -v -o "addopts="
```

### What Gets Tested

- ✅ Configuration loading and validation
- ✅ All 4 model architectures build correctly
- ✅ Models can do forward pass
- ✅ Feature engineering dimensions match

---

## Running the App

### Start Gesture Detection

```bash
handflow run
```

Or with options:
```bash
handflow run --camera 1 --mappings gesture_mapping.json
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `handflow info` | Show configuration |
| `handflow run` | Start gesture detection |
| `handflow train` | Train a model |

---

## Common Issues

### "ModuleNotFoundError: No module named 'handflow'"

Install the package:
```bash
pip install -e .
```

### "Camera not found"

Try a different camera index:
```bash
handflow run --camera 1
```

### W&B "API key not found"

Login again:
```bash
wandb login
```

### Tests fail with coverage error

Run without coverage:
```bash
pytest tests/ -v -o "addopts="
```

---

## Next Steps

1. **Collect your own gesture data** using the data collection scripts
2. **Try different architectures** and compare in W&B
3. **Tune hyperparameters** (learning rate, hidden units, dropout)
4. **Add new gestures** by collecting more data
5. **Deploy** using the TFLite quantized model

Good luck! 🖐️
