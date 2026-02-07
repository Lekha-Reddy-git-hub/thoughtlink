# ThoughtLink: Brain-to-Robot Control System

**Hack Nation 2026 -- Challenge 9 (ThoughtLink)**

A brain-computer interface (BCI) that decodes EEG brain signals into robot movement commands, driving a Unitree G1 humanoid robot in MuJoCo simulation.

## Architecture

```
  Raw EEG (.npz)
       |
       v
  +-----------+     +-----------+     +-----------+
  | Bandpass  |---->| Active    |---->| Normalize |
  | 8-30 Hz   |     | Segment   |     | Per Chan  |
  +-----------+     +-----------+     +-----------+
       |
       v
  +-----------+     +-----------+     +-----------+
  | Window    |---->| Feature   |---->| 69 feats  |
  | 1s/0.5s   |     | Extract   |     | PSD+Stat  |
  +-----------+     +-----------+     +-----------+
       |
       v
  +===========+     +===========+
  | Stage 1   |---->| Stage 2   |    (if active)
  | Rest vs   |     | Direction |
  | Active    |     | 3-class   |
  | RF 79.4%  |     | RF 49.2%  |
  +===========+     +===========+
       |                 |
       v                 v
  +-----------+     +-----------+     +-----------+
  | Confidence|---->| Majority  |---->| Hysteresis|
  | Gate      |     | Vote (5)  |     | Filter(3) |
  +-----------+     +-----------+     +-----------+
       |
       v
  +===========+
  | BRI API   |  Action: FORWARD / LEFT / RIGHT / STOP
  | G1 Robot  |  MuJoCo Simulation
  +===========+
```

## Quick Start

```bash
# Clone and setup
git clone <repo>
cd thoughtlink
pip install uv
uv venv --python 3.12
uv sync

# Download dataset (~536 MB)
uv run python src/download.py

# Run full pipeline (preprocessing + training)
uv run python src/preprocess.py
uv run python src/features.py
uv run python src/stage1_binary.py
uv run python src/stage2_direction.py

# Run demo (launches MuJoCo G1 humanoid)
uv run python demo/run_demo.py data/0b2dbd41-10.npz
```

## Results

| Metric | Value |
|--------|-------|
| Stage 1 Accuracy (Rest vs Active) | 79.4% (cross-subject) |
| Stage 2 Accuracy (3-class direction) | 49.2% (cross-subject, random=33%) |
| Inference Latency (per window) | 31 ms avg, 73 ms max (headless) |
| Inference Latency (with MuJoCo) | 146 ms avg |
| Flicker Reduction (smoothing) | 92.9% |
| Total EEG Files | 900 (865 usable) |
| Subjects | 6 unique |
| EEG Channels | 6 (4 frontal + 2 midline) |
| Classes | Relax, Left Fist, Right Fist, Both Fists, Tongue Tapping |
| Robot Actions | STOP, LEFT, RIGHT, FORWARD |

## Dataset

Source: [KernelCo/robot_control](https://huggingface.co/datasets/KernelCo/robot_control) on HuggingFace

- 900 recordings, 15 seconds each at 500 Hz
- 6 EEG channels: AFF6, AFp2, AFp1, AFF5, FCz, CPz
- 5 labels: Both Fists, Left Fist, Relax, Right Fist, Tongue Tapping
- 6 subjects, 20 sessions

## Pipeline Components

| Module | Description |
|--------|-------------|
| `src/download.py` | Dataset download and validation |
| `src/preprocess.py` | Bandpass filtering, segmentation, windowing |
| `src/features.py` | PSD and statistical feature extraction (69 features) |
| `src/stage1_binary.py` | Rest vs Active classifier (RandomForest) |
| `src/stage2_direction.py` | Direction classifier (RandomForest, 3-class) |
| `src/smoothing.py` | MajorityVote + ConfidenceGate + Hysteresis |
| `src/pipeline.py` | End-to-end ThoughtLinkPipeline class |
| `src/integration.py` | BRI robot integration (MuJoCo sim + fallback) |
| `src/visualize.py` | All visualization code |
| `demo/run_demo.py` | Single-command demo entry point |

## Visualizations

All saved in `results/`:
- `psd_verification.png` - Bandpass filter verification
- `psd_per_class.png` - PSD per class for all channels
- `feature_tsne.png` - t-SNE of feature space
- `stage1_confusion.png` - Stage 1 confusion matrix
- `stage2_confusion.png` - Stage 2 confusion matrix
- `channel_importance.png` - Feature importance by channel
- `temporal_timeline.png` - Confidence and action timeline
- `smoothing_comparison.png` - Raw vs smoothed predictions
- `channel_layout.png` - EEG channel positions on head
- `cross_subject_accuracy.png` - Per-subject accuracy

## Limitations

- **6 channels** vs 64+ in research BCI systems
- **Frontal-heavy montage** (AFF6, AFp2, AFp1, AFF5) -- not ideal for motor cortex signals
- **Cross-subject generalization** is inherently harder than within-subject
- Direction discrimination (left vs right) is weak because motor cortex channels (C3, C4) are missing
- 4-class direction was below chance (27.9%), fell back to 3-class (49.2%)

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design decisions.

## Robot Integration

Uses the [Nabla7/brain-robot-interface](https://github.com/Nabla7/brain-robot-interface) framework:
- Unitree G1 humanoid in MuJoCo simulation
- ONNX walking policy for locomotion
- BCI commands sent via `Controller.set_action()` at >2 Hz
- Fallback to matplotlib 2D visualizer if MuJoCo unavailable
