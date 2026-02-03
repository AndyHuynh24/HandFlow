# Resume Corrections & Notes

> Reference document for HandFlow project resume bullet points

---

## Original Resume Bullets (Your Version)

```
• Built an end-to-end interaction pipeline integrating real-time hand-gesture recognition (intent)
  with ArUco-based homography calibration (spatial mapping) to enable free-space gestures, virtual
  touchscreens on non-touch display, and 24-button paper macropads (ordinary paper with printed
  buttons), mapped directly to OS-level controls.

• Designed and trained a custom Temporal CNN with velocity-gated attention to suppress idle-hand
  noise, achieving 98.9% accuracy (+6% vs. LSTM baselines) using 88 engineered geometric features;
  applied online geometric augmentation during training and tracked experiments, ablations, and
  hyperparameters with Weights & Biases.

• Developed a robust spatial calibration pipeline with ArUco markers, including missing-marker
  estimation and homography recovery, ensuring stable touch mapping under partial marker occlusion
  and camera jitter.

• Optimized for consumer-grade real-time deployment by building a multithreaded inference pipeline,
  achieving 20 FPS with <6 ms end-to-end latency on standard CPU.
```

---

## Issues Found (Code Review)

| Claim | What Code Shows | Status |
|-------|-----------------|--------|
| "24-button paper macropads" | 8-button (4×2 grid) in `macropad_detector.py` | **Needs correction** |
| "98.9% accuracy (+6% vs. LSTM)" | Not found in code - need W&B logs | **Verify from experiments** |
| "<6 ms end-to-end latency" | Inference: 2-5ms, Total pipeline: ~20-30ms | **Needs correction** |
| "multithreaded inference pipeline" | Only action execution is async | **Needs correction** |
| "velocity-gated attention" | `SoftMotionWeighting` layer confirmed | **Accurate** |
| "88 engineered geometric features" | Confirmed in `feature_engineer.py` | **Accurate** |
| "online geometric augmentation" | Confirmed in `augmentation.py` | **Accurate** |
| "Weights & Biases tracking" | Confirmed in codebase | **Accurate** |
| "missing-marker estimation" | Confirmed in `aruco_screen_detector.py` | **Accurate** |
| "homography recovery" | Confirmed | **Accurate** |

---

## Corrected Resume Bullets

### Option 1: Conservative (Verified Claims Only)

```
• Built an end-to-end interaction pipeline integrating real-time hand-gesture recognition
  with ArUco-based homography calibration to enable free-space gestures, virtual touchscreens
  on non-touch displays, and configurable paper macro pads mapped to OS-level controls.

• Designed and trained a Temporal CNN with velocity-gated attention (SoftMotionWeighting) to
  suppress idle-hand noise using 88 engineered geometric features; applied online geometric
  augmentation and tracked experiments with Weights & Biases.

• Developed a robust spatial calibration pipeline with ArUco markers, including missing-marker
  estimation via homography recovery, ensuring stable touch mapping under partial occlusion.

• Optimized for consumer-grade deployment with TFLite quantization, achieving ~25 FPS with
  ~5ms inference latency on CPU, using smoothing filters for stable cursor tracking.
```

### Option 2: With Accuracy (If You Can Verify from W&B)

```
• Built an end-to-end interaction pipeline integrating real-time hand-gesture recognition
  with ArUco-based homography calibration to enable free-space gestures, virtual touchscreens
  on non-touch displays, and configurable paper macro pads mapped to OS-level controls.

• Designed and trained a Temporal CNN with velocity-gated attention to suppress idle-hand
  noise, achieving [XX]% accuracy (+[X]% vs. LSTM baselines) using 88 engineered geometric
  features; applied online geometric augmentation and tracked experiments with Weights & Biases.

• Developed a robust spatial calibration pipeline with ArUco markers, including missing-marker
  estimation and homography recovery, ensuring stable touch mapping under partial occlusion
  and camera jitter.

• Optimized for consumer-grade deployment with TFLite quantization, achieving ~25 FPS with
  sub-10ms inference latency on standard CPU.
```

---

## Detailed Corrections

### 1. Macro Pad Buttons

**Original:** "24-button paper macropads"

**Code Reality:**
- `macropad_detector.py` line 88-89: `GRID_COLS = 4`, `GRID_ROWS = 2` = **8 buttons**
- Supports up to 12 different macro pad sets (IDs 12+)

**Options:**
- "8-button paper macro pads" (accurate)
- "configurable paper macro pads" (flexible)
- "paper macro pads with expandable button layouts" (future-looking)

---

### 2. Accuracy Metrics

**Original:** "98.9% accuracy (+6% vs. LSTM baselines)"

**Code Reality:** No accuracy metrics found in source code. This would be in:
- Weights & Biases dashboard
- Training logs in `logs/` directory
- Jupyter notebook outputs

**Action Required:**
1. Check your W&B project for the actual numbers
2. Look in `logs/` for tensorboard logs
3. If you have the numbers, use them; if not, remove the specific claim

---

### 3. Latency Claims

**Original:** "<6 ms end-to-end latency"

**Code Reality:**
- TFLite inference: **2-5ms** (this is accurate)
- MediaPipe detection: **~13ms**
- Feature engineering: **2-3ms**
- Total per-frame: **~20-30ms**

**Correction Options:**
- "~5ms model inference latency" (accurate for just the model)
- "sub-10ms inference latency" (safe claim for model)
- "~25ms end-to-end frame processing" (full pipeline)

---

### 4. Multithreading

**Original:** "multithreaded inference pipeline"

**Code Reality:**
- Main detection loop is **single-threaded**
- Action execution uses **async/background threads**
- Not a fully multithreaded inference pipeline

**Correction Options:**
- "asynchronous action execution pipeline"
- "optimized inference pipeline with async action dispatch"
- Remove "multithreaded" entirely

---

## Verified Technical Details (Safe to Claim)

| Feature | Details |
|---------|---------|
| Model Architecture | Temporal CNN (TCN) with dilated causal convolutions |
| Velocity Gating | `SoftMotionWeighting` layer - sigmoid-weighted by motion magnitude |
| Feature Count | 88 features (63 positions + 5 distances + 9 raw + 6 velocity + 5 angles) |
| Sequence Length | 12 frames (~400ms at 30 FPS) |
| Supported Architectures | TCN, LSTM, GRU, CNN1D, Transformer |
| ArUco System | 4-corner screen calibration + 8-marker macro pad |
| Missing Marker Recovery | Estimates 4th corner from 3 visible markers |
| Quantization | TFLite dynamic/int8 quantization |
| Model Size | ~2.5MB (full) → ~0.6MB (quantized) |
| Augmentation Types | Noise, time warp, dropout, scaling, rotation, Z-axis transforms |
| Gesture Classes | 11 classes (none + 10 gestures) |
| Experiment Tracking | Weights & Biases integration |

---

## Action Items

- [ ] Verify accuracy from W&B logs
- [ ] Update macro pad button count (8, not 24)
- [ ] Clarify latency claim (inference vs. end-to-end)
- [ ] Remove or clarify "multithreaded" claim
- [ ] Add demo video link to README

---

*Generated from codebase analysis on 2024*
