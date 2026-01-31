# Machine Learning Engineering Guide

A comprehensive guide covering experiment tracking, model evaluation, architecture design, and user personalization for gesture recognition systems. This document captures practical ML engineering insights applicable to HandFlow and similar projects.

---

## Table of Contents

1. [Experiment Tracking](#1-experiment-tracking)
   - [Why Track Experiments?](#why-track-experiments)
   - [TensorBoard (Local)](#tensorboard-local)
   - [Weights & Biases (Cloud)](#weights--biases-cloud)
   - [What to Log](#what-to-log)
   - [Implementation](#tracking-implementation)

2. [Model Evaluation](#2-model-evaluation)
   - [Industry-Standard Metrics](#industry-standard-metrics)
   - [Confusion Matrix Analysis](#confusion-matrix-analysis)
   - [When to Use Which Metric](#when-to-use-which-metric)

3. [Neural Network Architectures](#3-neural-network-architectures)
   - [Architecture Comparison](#architecture-comparison-for-gesture-recognition)
   - [Residual Connections](#residual-connections)
   - [TCN (Temporal Convolutional Network)](#tcn-temporal-convolutional-network)

4. [Transfer Learning & User Calibration](#4-transfer-learning--user-calibration)
   - [What is Transfer Learning?](#what-is-transfer-learning)
   - [Freeze Strategies](#freeze-strategies)
   - [Implementation Guide](#calibration-implementation)

5. [Model Formats: .keras vs TFLite](#5-model-formats-keras-vs-tflite)

6. [Best Practices](#6-best-practices)

---

## 1. Experiment Tracking

### Why Track Experiments?

Without tracking, you'll end up with dozens of models and no idea which one is best or how it was trained.

**Experiment tracking records:**
- Hyperparameters (learning rate, batch size, architecture)
- Metrics over time (loss, accuracy per epoch)
- System resources (GPU/CPU usage, memory)
- Model artifacts (checkpoints, final model)
- Code version (git commit)

**Benefits:**
| Benefit | Description |
|---------|-------------|
| Compare runs | See how different hyperparameters affect performance |
| Reproduce results | Know exactly what settings produced your best model |
| Debug issues | Identify when and why training went wrong |
| Collaborate | Share results with teammates |
| Version control | Track model versions alongside code changes |

---

### TensorBoard (Local)

TensorBoard is TensorFlow's visualization toolkit. Runs locally, no account needed.

**Setup:**
```bash
pip install tensorflow  # Includes TensorBoard
```

**Usage:**
```python
from tensorflow import keras

# Add callback during training
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir="logs/tensorboard/my-run",
    histogram_freq=1,      # Log weight histograms every epoch
    write_graph=True,      # Visualize model graph
)

model.fit(x_train, y_train, callbacks=[tensorboard_callback])
```

**View Results:**
```bash
# Terminal 1: Run training
python scripts/train.py

# Terminal 2: Launch TensorBoard
tensorboard --logdir=logs/tensorboard

# Open browser: http://localhost:6006
```

**What TensorBoard Shows:**
- **Scalars**: Loss/accuracy curves over time
- **Graphs**: Model architecture visualization
- **Histograms**: Weight distributions (detect vanishing gradients)
- **Images**: Sample predictions (for image models)

**Storage:** Local files in `logs/tensorboard/<run_name>/`

---

### Weights & Biases (Cloud)

W&B is a cloud-based MLOps platform with team collaboration features.

**Setup:**
```bash
pip install wandb
wandb login  # Paste API key from wandb.ai/authorize
```

**Usage:**
```python
import wandb
from wandb.integration.keras import WandbMetricsLogger

# Start run
wandb.init(
    project="my-project",
    name="experiment-1",
    config={"lr": 0.001, "epochs": 100}
)

# Train with automatic logging
model.fit(x_train, y_train, callbacks=[WandbMetricsLogger()])

wandb.finish()
```

**View Results:**
- URL printed in terminal: `wandb: Run page: https://wandb.ai/...`
- Or go to wandb.ai and select your project

**What W&B Provides:**
- Real-time metrics visualization
- Hyperparameter comparison across runs
- Automatic system metrics (GPU, CPU, memory)
- Model artifact versioning
- Team collaboration
- Hyperparameter sweeps (automatic tuning)

**Storage:** Cloud (wandb.ai) + local cache in `wandb/` folder

---

### What to Log

**Essential (always log):**
```python
config = {
    # Model
    "architecture": "tcn",
    "hidden_units": 128,
    "num_layers": 3,
    "dropout": 0.2,

    # Training
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "optimizer": "adam",

    # Data
    "training_samples": len(x_train),
    "validation_samples": len(x_val),
    "sequence_length": 12,
    "num_classes": 11,
}
```

**Per-Epoch Metrics:**
```python
wandb.log({
    "loss": loss,
    "accuracy": accuracy,
    "val_loss": val_loss,
    "val_accuracy": val_accuracy,
    "learning_rate": current_lr,  # If using LR scheduling
})
```

**End of Training:**
```python
# Confusion matrix
wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        y_true=y_true,
        preds=y_pred,
        class_names=class_names
    )
})

# Per-class metrics
wandb.log({
    "f1_per_class": wandb.plot.bar(table, "class", "f1_score")
})
```

---

### Tracking Implementation

HandFlow uses a unified `ExperimentTracker` that supports both backends:

```python
from handflow.utils import create_tracker

# Create tracker
tracker = create_tracker(
    project="handflow-gestures",
    use_wandb=True,      # Enable W&B
    use_tensorboard=True  # Enable TensorBoard
)

# Start run
tracker.start_run("experiment-1", config={"lr": 0.001})

# Log metrics
tracker.log_metrics({"loss": 0.5, "accuracy": 0.92})

# Log confusion matrix (W&B only)
tracker.log_confusion_matrix(y_true, y_pred, class_names)

# Get Keras callbacks for automatic logging
callbacks = tracker.get_keras_callbacks()
model.fit(x, y, callbacks=callbacks)

# Finish
tracker.finish()
```

**Configuration (config.yaml):**
```yaml
tracking:
  enabled: true

  wandb:
    enabled: false                # Set true to enable
    project: "handflow-gestures"

  tensorboard:
    enabled: true                 # Enabled by default
    log_dir: "logs/tensorboard"
```

---

## 2. Model Evaluation

### Industry-Standard Metrics

| Metric | Formula | What It Measures | Range |
|--------|---------|------------------|-------|
| **Accuracy** | Correct / Total | Overall correctness | 0-1 |
| **Precision** | TP / (TP + FP) | Of predicted positives, % actually correct | 0-1 |
| **Recall** | TP / (TP + FN) | Of actual positives, % we found | 0-1 |
| **F1-Score** | 2 * (P * R) / (P + R) | Harmonic mean of P and R | 0-1 |
| **Cohen's Kappa** | (Acc - Expected) / (1 - Expected) | Agreement beyond chance | -1 to 1 |
| **MCC** | See formula | Correlation for imbalanced data | -1 to 1 |
| **ROC-AUC** | Area under ROC curve | Ranking quality | 0-1 |
| **Top-K Accuracy** | Correct in top K / Total | Correct answer in top K predictions | 0-1 |

**Averaging Methods (for multi-class):**

| Method | Description | When to Use |
|--------|-------------|-------------|
| **Macro** | Average of per-class metrics | All classes equally important |
| **Micro** | Global TP/FP/FN counts | Large dataset, class imbalance OK |
| **Weighted** | Weighted by class support | Account for class imbalance |

---

### Confusion Matrix Analysis

```
              Predicted
            A    B    C    D
        ┌────┬────┬────┬────┐
      A │ 45 │  2 │  1 │  0 │  ← 45 correct, 3 errors
Actual  ├────┼────┼────┼────┤
      B │  3 │ 38 │  5 │  2 │  ← Often confused with C
        ├────┼────┼────┼────┤
      C │  0 │  4 │ 42 │  1 │  ← Often confused with B
        ├────┼────┼────┼────┤
      D │  1 │  0 │  2 │ 50 │  ← Best performing
        └────┴────┴────┴────┘
```

**What to Look For:**
- **Diagonal**: Correct predictions (should be high)
- **Off-diagonal**: Confusion between classes
- **Symmetric confusion**: B↔C suggests similar features
- **One-way confusion**: A→B but not B→A suggests class overlap

**Normalized Confusion Matrix:**
Divide each row by row sum to show percentages:
```python
cm_normalized = cm / cm.sum(axis=1, keepdims=True)
```

---

### When to Use Which Metric

| Scenario | Primary Metric | Why |
|----------|---------------|-----|
| **Balanced classes** | Accuracy, F1-Macro | All classes matter equally |
| **Imbalanced classes** | F1-Weighted, MCC | Account for class sizes |
| **False positives costly** | Precision | Minimize false alarms |
| **False negatives costly** | Recall | Don't miss positives |
| **Ranking matters** | ROC-AUC | Quality of probability ordering |
| **Top choices matter** | Top-K Accuracy | Useful for recommendations |

**For Gesture Recognition:**
- Use **F1-Macro** (all gestures equally important)
- Monitor **per-class F1** (find weak gestures)
- Use **confusion matrix** (understand gesture similarity)

---

## 3. Neural Network Architectures

### Architecture Comparison for Gesture Recognition

For short sequences (12 frames) with hand landmarks (84 features):

| Rank | Architecture | Pros | Cons | Inference Speed |
|------|--------------|------|------|-----------------|
| 1 | **TCN** | Multi-scale patterns, parallelizable, good for short sequences | Slightly more complex | Fast |
| 2 | **1D CNN** | Simple, fastest inference, easy to deploy | Less temporal context | Fastest |
| 3 | **GRU** | Good temporal modeling | Sequential (slower) | Medium |
| 4 | **Transformer** | Powerful attention | Overkill for 12 frames, expensive | Slow |
| 5 | **LSTM** | Classic choice | Outdated, slower than GRU | Medium |

**Recommendation:** TCN or 1D CNN for gesture recognition.

---

### Residual Connections

**The Problem:**
Deep networks suffer from vanishing gradients - gradients become tiny as they backpropagate through many layers.

```
Without Residual:
Input ──► Layer1 ──► Layer2 ──► Layer3 ──► Output
              │          │          │
          gradient   gradient   gradient
          shrinks    shrinks    shrinks
              ▼          ▼          ▼
           0.1        0.01       0.001  ← Vanishing!
```

**The Solution:**
Add skip connections that let gradients flow directly:

```
With Residual:
Input ─────────────────────────────┐
   │                               │
   ├──► Layer1 ──► Layer2 ──► (+) ◄┘
   │                           │
   │    Gradient flows         │ Gradient also flows
   │    through layers         │ directly through skip!
   │         ▼                 ▼
   │        0.01      +       1.0   ← Preserved!
```

**Implementation:**
```python
def residual_block(x, filters):
    residual = x  # Save input

    # Transform
    x = Conv1D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv1D(filters, 1)(x)
    x = BatchNormalization()(x)

    # Add skip connection
    x = Add()([x, residual])  # ← Key line
    x = ReLU()(x)

    return x
```

**Benefits:**
| Benefit | Explanation |
|---------|-------------|
| No vanishing gradients | Gradients flow through skip connection |
| Learns refinements | Layer learns `F(x) = output - input` (the difference) |
| Faster convergence | Easier optimization landscape |
| Deeper networks | Can stack more layers without degradation |

---

### TCN (Temporal Convolutional Network)

TCN uses dilated convolutions to capture patterns at multiple time scales:

```
Dilation = 1:  [■ ■ ■]           ← Sees 3 adjacent frames
Dilation = 2:  [■ · ■ · ■]       ← Sees 5 frames (skips 1)
Dilation = 4:  [■ · · · ■ · · · ■] ← Sees 9 frames (skips 3)
```

**Why Dilated Convolutions?**
- Cover longer time spans without more parameters
- Exponentially increasing receptive field
- Parallelizable (unlike RNNs)

**HandFlow TCN Architecture:**
```
Input (12 frames, 84 features)
    │
    ├──► Conv1D 1x1 (project to 128 dims)
    │
    ├──► Residual Block (dilation=1)  ← Local patterns
    ├──► Residual Block (dilation=2)  ← Medium patterns
    ├──► Residual Block (dilation=4)  ← Global patterns
    │
    ├──► GlobalAvgPool + GlobalMaxPool (concatenated)
    │
    ├──► Dense(64) + Dropout
    └──► Dense(11) + Softmax
```

**Receptive Field Calculation:**
With dilations [1, 2, 4] and kernel size 3:
- Block 1: 3 frames
- Block 2: 3 + 2*(3-1) = 7 frames
- Block 3: 7 + 4*(3-1) = 15 frames > 12 (covers full sequence)

---

## 4. Transfer Learning & User Calibration

### What is Transfer Learning?

Using knowledge from a pre-trained model to improve learning on a new task.

```
┌─────────────────────────────────────────────────────────────┐
│                    PRE-TRAINED MODEL                        │
│  (Trained on many users, general gesture patterns)          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼ Transfer knowledge
┌─────────────────────────────────────────────────────────────┐
│                  USER CALIBRATION                           │
│  1. User performs each gesture 5-10 times                   │
│  2. Collect samples with their hand/camera/lighting         │
│  3. Fine-tune ONLY the last layers                          │
│  4. Save personalized model                                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  PERSONALIZED MODEL                         │
│  (Adapted to this user's hand shape, camera angle, etc.)    │
└─────────────────────────────────────────────────────────────┘
```

**Why It Works:**
- Early layers learn **general features** (motion patterns, joint relationships)
- Later layers learn **task-specific mappings** (gesture classification)
- General features transfer across users; only mappings need adjustment

**Why Personalization Helps Gesture Recognition:**

| Variation | Impact on Recognition |
|-----------|----------------------|
| Hand size | Landmark positions scale differently |
| Finger proportions | Gesture shapes vary |
| Camera angle | Same gesture looks different |
| Lighting | MediaPipe landmark accuracy varies |
| Gesture speed | Fast vs slow performers |
| Gesture style | Wide vs narrow movements |

---

### Freeze Strategies

**Key Insight:** Don't train the whole model with few samples - you'll overwrite good features.

```
┌─────────────────────────────────────────────────────────────┐
│  FREEZE THESE (learned general patterns)                    │
│  ─────────────────────────────────────────                  │
│  • Input projection                                         │
│  • Early conv blocks (low-level features)                   │
│  • Mid conv blocks (motion patterns)                        │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│  TRAIN THESE (adapt to user)                                │
│  ────────────────────────────                               │
│  • Last conv block (high-level, task-specific)              │
│  • Dense layers (classification)                            │
└─────────────────────────────────────────────────────────────┘
```

**Strategy Options:**

| Strategy | Frozen | Trained | Samples Needed | Adaptation Level |
|----------|--------|---------|----------------|------------------|
| `head_only` | All except last Dense | 1 layer | 5-10 total | Minimal |
| `last_block` | First 60% | Last 40% | 5-10 per class | **Recommended** |
| `half` | First 50% | Last 50% | 20+ per class | Maximum |

**How to Freeze Layers:**
```python
# Load pre-trained model
model = keras.models.load_model("models/hand_action.keras")

# Freeze early layers
for layer in model.layers[:-3]:  # All except last 3
    layer.trainable = False

# Compile with LOW learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # Very low!
    loss="categorical_crossentropy"
)

# Train on user data
model.fit(user_samples, user_labels, epochs=10)
```

---

### Calibration Implementation

**Quick Usage:**
```python
from handflow.models import quick_calibrate

model, result = quick_calibrate(
    base_model_path="models/hand_action.keras",
    user_samples=samples,      # (N, 12, 84)
    user_labels=labels,        # (N,)
    class_names=["none", "swipe", ...],
    freeze_strategy="last_block"
)

print(f"Improvement: {result.improvement:+.2%}")

if result.success:
    model.save("models/user_model.keras")
```

**Interactive Calibration (for UI):**
```python
from handflow.models import UserCalibrator, CalibrationConfig

config = CalibrationConfig(
    min_samples_per_gesture=5,
    freeze_strategy="last_block",
    learning_rate=1e-5
)

calibrator = UserCalibrator(
    base_model_path="models/hand_action.keras",
    class_names=gesture_list,
    config=config
)

# During calibration UI
calibrator.add_sample(sequence, gesture_name="swipe_left")

# Check progress
counts = calibrator.get_sample_counts()
ready, message = calibrator.is_ready()

# Calibrate
result = calibrator.calibrate()
if result.success:
    calibrator.save("models/user_profiles/user123.keras")
```

**Recommended Settings:**

| Setting | Value | Reason |
|---------|-------|--------|
| Samples per gesture | 5-10 | Enough for adaptation, not tedious |
| Learning rate | 1e-5 | Very low to preserve features |
| Epochs | 10-20 | Few epochs, early stopping |
| Batch size | 4 | Small for small dataset |
| Freeze strategy | `last_block` | Good balance |

---

## 5. Model Formats: .keras vs TFLite

| Aspect | .keras | TFLite (.tflite) |
|--------|--------|------------------|
| **File size** | Larger | ~4x smaller (with int8 quantization) |
| **Inference speed** | Normal | Faster on CPU/mobile |
| **Platform** | Python/TensorFlow | Android, iOS, embedded, Python |
| **GPU support** | Full CUDA | Limited (delegates) |
| **Quantization** | Requires extra steps | int8, float16 built-in |
| **Retraining** | Easy | Not possible |
| **Op support** | All TensorFlow ops | Limited (some layers fail) |
| **Dependencies** | TensorFlow (~500MB) | TFLite runtime (~5MB) |

**When to Use Each:**

| Use Case | Format | Reason |
|----------|--------|--------|
| Development | .keras | Easy to debug, retrain, modify |
| Desktop app | .keras | Full features, retraining for calibration |
| Mobile app | TFLite | Small size, fast inference |
| Raspberry Pi | TFLite | Memory constrained |
| Edge devices | TFLite | Optimized for ARM |

**For HandFlow (desktop-only):** Use `.keras` - supports user calibration and full features.

**Converting to TFLite (if needed later):**
```python
import tensorflow as tf

# Load Keras model
model = tf.keras.models.load_model("model.keras")

# Convert
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Enable quantization
tflite_model = converter.convert()

# Save
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
```

---

## 6. Best Practices

### Experiment Tracking

1. **Name runs meaningfully:**
   ```python
   run_name = f"{architecture}_lr{lr}_bs{batch}_{timestamp}"
   ```

2. **Log everything relevant** - you can always ignore data, but can't recover what wasn't logged

3. **Use tags for organization:**
   ```python
   wandb.init(tags=["baseline", "v2", "augmentation"])
   ```

4. **Compare runs systematically** - use the same validation set

### Model Training

1. **Start simple, add complexity** - begin with 1D CNN, try TCN if needed

2. **Use early stopping** - prevents overfitting, saves time

3. **Monitor validation metrics** - training metrics can be misleading

4. **Save checkpoints** - recover from crashes

### Transfer Learning

1. **Always freeze early layers** - they contain valuable general features

2. **Use very low learning rate** - 1e-5 or lower for fine-tuning

3. **Validate improvement** - don't save if accuracy decreased

4. **Keep original model** - as fallback if calibration fails

### Evaluation

1. **Look at per-class metrics** - overall accuracy hides problems

2. **Analyze confusion matrix** - understand which classes are confused

3. **Test on held-out data** - never evaluate on training data

4. **Consider real-world conditions** - test with different users, lighting, etc.

---

## Quick Reference

### Commands

```bash
# Training with TensorBoard
python scripts/train.py --architecture tcn
tensorboard --logdir=logs/tensorboard
# Open http://localhost:6006

# Training with W&B
python scripts/train.py --use-wandb
# View at wandb.ai

# Evaluation
python scripts/evaluate.py --save-report
# Reports saved to reports/evaluation/
```

### Key Files

| File | Purpose |
|------|---------|
| `src/handflow/utils/experiment_tracker.py` | Unified tracking (W&B + TensorBoard) |
| `src/handflow/models/trainer.py` | Training loop with tracking |
| `src/handflow/models/calibration.py` | User personalization |
| `src/handflow/models/architectures.py` | Neural network definitions |
| `scripts/train.py` | Training script |
| `scripts/evaluate.py` | Evaluation with all metrics |
| `config/config.yaml` | Configuration |

---

## Further Reading

- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard)
- [Weights & Biases Docs](https://docs.wandb.ai)
- [TCN Paper: "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"](https://arxiv.org/abs/1803.01271)
- [ResNet Paper: "Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385)
- [Transfer Learning Survey](https://arxiv.org/abs/1911.02685)
