# Experiment Tracking Guide

A comprehensive guide to tracking ML experiments using **Weights & Biases (W&B)** and **TensorBoard** in HandFlow and other ML projects.

---
  Quick Reference                   
  Tool: TensorBoard                 
  Command: tensorboard              
    --logdir=logs/tensorboard       
  View At: http://localhost:6006  

## Table of Contents

1. [What is Experiment Tracking?](#what-is-experiment-tracking)
2. [Weights & Biases (W&B)](#weights--biases-wb)
   - [What is W&B?](#what-is-wb)
   - [Installation & Setup](#installation--setup)
   - [How to Use](#how-to-use-wb)
   - [Dashboard Overview](#dashboard-overview)
   - [Key Features](#key-features)
3. [TensorBoard](#tensorboard)
   - [What is TensorBoard?](#what-is-tensorboard)
   - [Installation & Setup](#tensorboard-installation--setup)
   - [How to Launch](#how-to-launch-tensorboard)
   - [Dashboard Overview](#tensorboard-dashboard)
4. [HandFlow Integration](#handflow-integration)
   - [Configuration](#configuration)
   - [Training with Tracking](#training-with-tracking)
   - [Reusing in Other Projects](#reusing-in-other-projects)
5. [Comparison: W&B vs TensorBoard](#comparison-wb-vs-tensorboard)
6. [Best Practices](#best-practices)

---

## What is Experiment Tracking?

Experiment tracking is the practice of **recording everything about your ML experiments** so you can:

- **Compare runs**: See how different hyperparameters affect performance
- **Reproduce results**: Know exactly what settings produced your best model
- **Debug issues**: Identify when and why training went wrong
- **Collaborate**: Share results with teammates
- **Version control**: Track model versions alongside code changes

Without tracking, you'll end up with dozens of models and no idea which one is best or how it was trained.

---

## Weights & Biases (W&B)

### What is W&B?

**Weights & Biases** is a cloud-based MLOps platform that automatically tracks:

- **Hyperparameters**: Learning rate, batch size, architecture, etc.
- **Metrics**: Loss, accuracy, F1-score over time
- **System metrics**: GPU/CPU usage, memory, network
- **Model artifacts**: Save and version your trained models
- **Code**: Automatic git commit tracking

**Key benefit**: Your experiments are stored in the cloud - accessible from anywhere, shareable with teammates.

### Installation & Setup

#### Step 1: Install W&B

```bash
pip install wandb
```

#### Step 2: Create an Account

1. Go to [wandb.ai](https://wandb.ai)
2. Click "Sign Up" (free for individuals and academics)
3. Create your account (you can use GitHub/Google)

#### Step 3: Get Your API Key

1. After logging in, go to [wandb.ai/authorize](https://wandb.ai/authorize)
2. Copy your API key (it looks like: `abc123def456...`)

#### Step 4: Login from Terminal

```bash
wandb login
```

When prompted, paste your API key. This saves it locally so you don't need to enter it again.

**Alternative**: Set as environment variable:
```bash
export WANDB_API_KEY="your-api-key-here"
```

#### Step 5: Verify Installation

```python
import wandb
wandb.login()
print("W&B is ready!")
```

### How to Use W&B

#### Basic Usage (Standalone)

```python
import wandb

# 1. Start a new run
wandb.init(
    project="my-project",          # Groups runs together
    name="experiment-1",           # Name for this specific run
    config={                       # Hyperparameters to track
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 32,
        "architecture": "transformer"
    }
)

# 2. Training loop
for epoch in range(100):
    loss = train_one_epoch()
    accuracy = evaluate()

    # Log metrics - they appear in real-time on the dashboard
    wandb.log({
        "loss": loss,
        "accuracy": accuracy,
        "epoch": epoch
    })

# 3. Save your model as an artifact
artifact = wandb.Artifact("trained-model", type="model")
artifact.add_file("model.h5")
wandb.log_artifact(artifact)

# 4. End the run
wandb.finish()
```

#### With Keras (Automatic Logging)

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger

# Start run
wandb.init(project="my-project", config={"epochs": 100})

# Train with automatic logging
model.fit(
    x_train, y_train,
    epochs=100,
    callbacks=[WandbMetricsLogger()]  # Logs every epoch automatically
)

wandb.finish()
```

### Dashboard Overview

After running your training script, go to [wandb.ai](https://wandb.ai) and click your project. You'll see:

#### 1. **Runs Table**
Shows all your experiments with key metrics. Click column headers to sort.

| Run Name | Loss | Accuracy | LR | Epochs | Duration |
|----------|------|----------|-----|--------|----------|
| run-001  | 0.23 | 94.2%    | 0.001 | 100  | 45m      |
| run-002  | 0.31 | 91.8%    | 0.01  | 100  | 42m      |

#### 2. **Charts**
Real-time line charts showing metrics over time. Compare multiple runs by selecting them.

#### 3. **System Metrics**
GPU utilization, memory usage, CPU load - helps identify bottlenecks.

#### 4. **Config Comparison**
Side-by-side comparison of hyperparameters between runs.

### Key Features

#### Logging Different Data Types

```python
# Scalars (numbers)
wandb.log({"loss": 0.5, "accuracy": 0.92})

# Images
wandb.log({"predictions": wandb.Image(image_array, caption="Epoch 10")})

# Tables (for confusion matrices, etc.)
table = wandb.Table(columns=["Predicted", "Actual", "Correct"])
table.add_data("cat", "cat", True)
wandb.log({"predictions_table": table})

# Histograms
wandb.log({"weights": wandb.Histogram(model_weights)})
```

#### Sweeps (Hyperparameter Search)

```python
# Define search space
sweep_config = {
    "method": "bayes",  # or "random", "grid"
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"min": 0.0001, "max": 0.01},
        "batch_size": {"values": [16, 32, 64]},
        "hidden_units": {"values": [64, 128, 256]}
    }
}

# Create sweep
sweep_id = wandb.sweep(sweep_config, project="my-project")

# Run agent (will try different combinations)
wandb.agent(sweep_id, function=train)
```

---

## TensorBoard

### What is TensorBoard?

**TensorBoard** is TensorFlow's visualization toolkit. It runs **locally** on your machine (no cloud account needed) and provides:

- **Scalar metrics**: Loss and accuracy curves
- **Histograms**: Weight and gradient distributions
- **Graphs**: Model architecture visualization
- **Images**: Visualize predictions
- **Profiling**: Performance analysis

**Key benefit**: No account needed, completely local, works offline.

### TensorBoard Installation & Setup

TensorBoard comes with TensorFlow:

```bash
pip install tensorflow  # Includes TensorBoard
# or install separately:
pip install tensorboard
```

Verify installation:
```bash
tensorboard --version
```

### How to Launch TensorBoard

#### Step 1: Run Training (creates log files)

When you train with HandFlow, logs are automatically saved to `logs/tensorboard/`.

Or manually in your code:

```python
from tensorflow import keras

# Create TensorBoard callback
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir="logs/tensorboard/my-run",
    histogram_freq=1,      # Log weight histograms every epoch
    write_graph=True,      # Visualize model graph
    update_freq="epoch"    # Log every epoch
)

# Train with callback
model.fit(
    x_train, y_train,
    epochs=100,
    callbacks=[tensorboard_callback]
)
```

#### Step 2: Launch TensorBoard Server

Open a terminal and run:

```bash
# Basic launch
tensorboard --logdir=logs/tensorboard

# Specify port (default is 6006)
tensorboard --logdir=logs/tensorboard --port=6007

# Bind to all interfaces (for remote access)
tensorboard --logdir=logs/tensorboard --bind_all
```

#### Step 3: Open in Browser

Go to: **http://localhost:6006**

### TensorBoard Dashboard

#### Scalars Tab
- Line charts for loss, accuracy, learning rate
- Compare multiple runs by selecting them in the left sidebar
- Use the smoothing slider to reduce noise

#### Graphs Tab
- Interactive visualization of your model architecture
- Click nodes to see layer details

#### Histograms Tab
- Weight and bias distributions over training
- Helps identify vanishing/exploding gradients

#### Images Tab
- Visualize sample predictions
- Useful for debugging image models

### TensorBoard Commands Reference

```bash
# Launch with specific log directory
tensorboard --logdir=/path/to/logs

# Compare multiple experiments
tensorboard --logdir=run1:logs/run1,run2:logs/run2

# Launch in background
nohup tensorboard --logdir=logs &

# Stop TensorBoard
# Find process ID
lsof -i :6006
# Kill it
kill -9 <PID>
```

---

## HandFlow Integration

### Configuration

The tracking settings are in `config/config.yaml`:

```yaml
# Experiment Tracking Configuration
tracking:
  enabled: true                       # Master switch

  # Weights & Biases
  wandb:
    enabled: false                    # Set to true to enable
    project: "handflow-gestures"      # Your project name
    entity: null                      # Your username/team

  # TensorBoard
  tensorboard:
    enabled: true                     # Enabled by default
    log_dir: "logs/tensorboard"
    histogram_freq: 1
```

### Training with Tracking

#### TensorBoard Only (Default)

```bash
python scripts/train.py --architecture transformer --epochs 100
```

After training, view results:
```bash
tensorboard --logdir=logs/tensorboard
# Open http://localhost:6006
```

#### With W&B Enabled

```bash
# Enable via CLI flag
python scripts/train.py --architecture transformer --use-wandb

# Or enable in config.yaml (set wandb.enabled: true)
```

### Reusing in Other Projects

The `ExperimentTracker` is designed to be project-agnostic. Copy these files to any project:

```
src/handflow/utils/experiment_tracker.py  # Main tracker module
```

Usage in any project:

```python
from experiment_tracker import create_tracker, TrackingConfig

# Quick setup
tracker = create_tracker(
    project="my-other-project",
    use_wandb=True,
    use_tensorboard=True
)

# Start a run
tracker.start_run("experiment-1", config={"lr": 0.001})

# Log metrics
tracker.log_metrics({"loss": 0.5, "accuracy": 0.92})

# Get Keras callbacks for automatic logging
callbacks = tracker.get_keras_callbacks()
model.fit(x, y, callbacks=callbacks)

# Finish
tracker.finish()
```

Or use the full configuration:

```python
from experiment_tracker import ExperimentTracker, TrackingConfig

config = TrackingConfig(
    enabled=True,
    wandb_enabled=True,
    wandb_project="my-project",
    wandb_entity="my-team",
    tensorboard_enabled=True,
    tensorboard_log_dir="logs/tb",
    log_model=True,
)

tracker = ExperimentTracker(config)
```

---

## Comparison: W&B vs TensorBoard

| Feature | W&B | TensorBoard |
|---------|-----|-------------|
| **Storage** | Cloud (wandb.ai) | Local files |
| **Account Required** | Yes (free tier available) | No |
| **Offline Use** | Limited | Full support |
| **Team Collaboration** | Built-in | Manual sharing |
| **Model Versioning** | Built-in artifacts | Not built-in |
| **Hyperparameter Sweeps** | Built-in | Requires plugins |
| **System Metrics** | Automatic | Manual |
| **Setup Complexity** | Moderate | Simple |
| **Cost** | Free tier + paid | Free |

### When to Use Each

**Use W&B when:**
- Working in a team
- Need to compare many experiments
- Want automatic hyperparameter sweeps
- Need model versioning/registry
- Want to access experiments from any device

**Use TensorBoard when:**
- Working solo
- Need quick local visualization
- Working offline or in restricted environments
- Don't want to create accounts
- Need detailed weight/gradient histograms

**Use Both when:**
- Want the best of both worlds
- W&B for collaboration and sweeps
- TensorBoard for detailed local debugging

---

## Best Practices

### 1. Name Your Runs Meaningfully

```python
# Bad
run_name = "run_001"

# Good
run_name = f"transformer_lr{lr}_bs{batch_size}_{datetime.now():%Y%m%d}"
```

### 2. Log Everything Relevant

```python
config = {
    "architecture": "transformer",
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "optimizer": "adam",
    "dropout": 0.2,
    "data_augmentation": True,
    "training_samples": len(x_train),
    "git_commit": get_git_commit(),  # Track code version
}
```

### 3. Use Tags for Organization

```python
wandb.init(
    project="my-project",
    tags=["baseline", "v2", "augmentation"],  # Filterable in UI
    notes="Testing new augmentation strategy"
)
```

### 4. Save Model Artifacts

```python
# W&B
artifact = wandb.Artifact("model", type="model")
artifact.add_file("best_model.h5")
wandb.log_artifact(artifact)

# TensorBoard: Use ModelCheckpoint callback
keras.callbacks.ModelCheckpoint("checkpoints/best.h5", save_best_only=True)
```

### 5. Clean Up Old Runs

```bash
# W&B: Delete via UI or API
wandb.Api().run("user/project/run_id").delete()

# TensorBoard: Delete log directories
rm -rf logs/tensorboard/old_run/
```

### 6. Monitor System Resources

W&B tracks GPU/CPU automatically. For TensorBoard, use profiler:

```python
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir="logs",
    profile_batch="10,20"  # Profile batches 10-20
)
```

---

## Troubleshooting

### W&B Issues

**"wandb: Network error"**
```bash
# Check internet connection, or use offline mode
wandb offline
python train.py
wandb sync  # Upload later
```

**"wandb: Error logging"**
```bash
# Clear cache
rm -rf ~/.wandb/
wandb login
```

### TensorBoard Issues

**"No dashboards are active"**
- Ensure log directory has event files
- Check path: `ls logs/tensorboard/*/events.*`

**Port already in use**
```bash
# Find and kill existing process
lsof -i :6006
kill -9 <PID>
tensorboard --logdir=logs
```

**High memory usage**
```bash
# Limit samples displayed
tensorboard --logdir=logs --samples_per_plugin=images=10
```

---

## Quick Reference

### Start Tracking in 30 Seconds

```python
# Option 1: W&B only
import wandb
wandb.init(project="quick-test")
wandb.log({"metric": 0.5})
wandb.finish()

# Option 2: TensorBoard only
callback = keras.callbacks.TensorBoard("logs/quick")
model.fit(x, y, callbacks=[callback])
# Then: tensorboard --logdir=logs

# Option 3: HandFlow tracker (both)
from handflow.utils import create_tracker
tracker = create_tracker(project="test", use_wandb=True, use_tensorboard=True)
tracker.start_run("run-1", {"lr": 0.001})
tracker.log_metrics({"loss": 0.5})
tracker.finish()
```

---

## Resources

- **W&B Documentation**: [docs.wandb.ai](https://docs.wandb.ai)
- **W&B Examples**: [github.com/wandb/examples](https://github.com/wandb/examples)
- **TensorBoard Guide**: [tensorflow.org/tensorboard](https://www.tensorflow.org/tensorboard)
- **TensorBoard GitHub**: [github.com/tensorflow/tensorboard](https://github.com/tensorflow/tensorboard)
