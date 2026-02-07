# ThoughtLink: Brain-to-Robot Control System

**Hack Nation 2026 -- Challenge 9 (ThoughtLink)**

A brain-computer interface (BCI) that decodes EEG brain signals into 5 robot movement commands (FORWARD, BACKWARD, LEFT, RIGHT, STOP), driving a Unitree G1 humanoid robot in MuJoCo simulation.

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
  | Active    |     | 4-class   |
  | RF 79.4%  |     | SVM 27.9% |
  +===========+     +===========+
       |                 |
       v                 v
  +-----------+     +-----------+     +-----------+
  | Confidence|---->| Majority  |---->| Hysteresis|
  | Gate      |     | Vote (5)  |     | Filter(3) |
  +-----------+     +-----------+     +-----------+
       |
       v
  +===========+     +=============+
  | Phase     |     | BRI API     |
  | Detection |---->| G1 Robot    |
  | I/S/R     |     | MuJoCo Sim  |
  +===========+     +=============+
       Action: FORWARD / BACKWARD / LEFT / RIGHT / STOP
       Phase:  INITIATION / SUSTAINED / RELEASE
```

## Quick Start

```bash
# Clone and install dependencies
git clone <repo>
cd thoughtlink
pip install uv
uv venv --python 3.12
uv sync

# One-command setup: downloads data, trains models, clones BRI
uv run python setup_project.py

# THE demo (self-explanatory 6-panel visual)
uv run python demo/full_demo.py

# Other demos
uv run python demo/run_all.py              # All 5 actions with MuJoCo
uv run python demo/override_demo.py        # Autonomous override scenario
uv run python demo/scalability_demo.py     # Fleet scalability test
uv run python demo/run_demo.py data/0b2dbd41-10.npz  # Single file
```

`setup_project.py` handles everything: downloading the 900 EEG recordings from HuggingFace
(~536 MB), preprocessing, feature extraction, training both classifiers, and
cloning the brain-robot-interface repo for MuJoCo simulation. Takes ~5-10 minutes.

<details>
<summary>Manual setup (step by step)</summary>

```bash
uv run python src/download.py        # Download dataset
uv run python src/preprocess.py      # Bandpass filter + windowing
uv run python src/features.py        # Extract 69 features per window
uv run python src/stage1_binary.py   # Train Rest vs Active classifier
uv run python src/stage2_direction.py # Train 4-class direction classifier
git clone https://github.com/Nabla7/brain-robot-interface.git  # For MuJoCo
```
</details>

## Results

| Metric | Value |
|--------|-------|
| Stage 1 Accuracy (Rest vs Active) | 79.4% (cross-subject, RF) |
| Stage 2 Accuracy (4-class direction) | 27.9% (cross-subject, SVM, random=25%) |
| MLP Baseline (raw EEG) | 79.1% / 24.3% (Stage 1 / Stage 2) |
| Inference Latency (per window) | 31 ms avg (headless) |
| Flicker Reduction (smoothing) | 92.9% |
| Phase Detection | INITIATION / SUSTAINED / RELEASE |
| Scalability | ~35 robots/core at 1 Hz |
| TD-NIRS Fusion | No benefit (zero-variance features) |
| Total EEG Files | 900 (865 usable) |
| Subjects | 6 unique |
| EEG Channels | 6 (4 frontal + 2 midline) |
| Brain Signal Classes | Relax, Left Fist, Right Fist, Both Fists, Tongue Tapping |
| Robot Actions | STOP, LEFT, RIGHT, FORWARD, BACKWARD |

## Dataset

Source: [KernelCo/robot_control](https://huggingface.co/datasets/KernelCo/robot_control) on HuggingFace

- 900 recordings, 15 seconds each at 500 Hz
- 6 EEG channels: AFF6, AFp2, AFp1, AFF5, FCz, CPz
- 5 labels: Both Fists, Left Fist, Relax, Right Fist, Tongue Tapping
- 6 subjects, 20 sessions
- TD-NIRS data: feature_moments (72, 40, 3, 2, 3) -- explored, not used (zero-variance)

## Pipeline Components

| Module | Description |
|--------|-------------|
| `src/download.py` | Dataset download and validation |
| `src/preprocess.py` | Bandpass filtering, segmentation, windowing |
| `src/features.py` | PSD and statistical feature extraction (69 features) |
| `src/stage1_binary.py` | Rest vs Active classifier (RandomForest) |
| `src/stage2_direction.py` | Direction classifier (4-class, SVM) |
| `src/smoothing.py` | MajorityVote + ConfidenceGate + Hysteresis |
| `src/pipeline.py` | End-to-end ThoughtLinkPipeline with phase detection |
| `src/integration.py` | BRI robot integration (MuJoCo sim + fallback) |
| `src/cnn_baseline.py` | MLP vs RF accuracy/latency comparison |
| `src/explore_nirs.py` | TD-NIRS multimodal fusion exploration |
| `src/temporal_embedding.py` | Intent evolution PCA trajectories |
| `src/visualize.py` | All visualization code |

## Demo Scripts

| Script | Description |
|--------|-------------|
| `demo/full_demo.py` | Self-explanatory 6-panel visual demo (THE demo for judges) |
| `demo/run_all.py` | All 5 robot actions back-to-back with MuJoCo |
| `demo/override_demo.py` | Autonomous-to-override scenario |
| `demo/scalability_demo.py` | Fleet scalability (10 robots, 5 rounds) |
| `demo/run_demo.py` | Single-file demo entry point |
| `demo/demo_video.py` | Generate demo_video.gif and summary figures |

## Visualizations

All saved in `results/`:
- `cnn_vs_rf.png` - MLP vs Random Forest accuracy and latency comparison
- `intent_evolution.png` - PCA trajectories showing intent evolution over time
- `nirs_exploration.txt` - TD-NIRS multimodal fusion analysis
- `stage1_confusion.png` - Stage 1 confusion matrix
- `stage2_confusion.png` - Stage 2 confusion matrix (4-class)
- `psd_per_class.png` - PSD per class for all channels
- `feature_tsne.png` - t-SNE of feature space
- `channel_importance.png` - Feature importance by channel
- `temporal_timeline.png` - Confidence and action timeline
- `smoothing_comparison.png` - Raw vs smoothed predictions
- `cross_subject_accuracy.png` - Per-subject accuracy

## Key Design Decisions

**Why 4-class direction at 27.9%?** The challenge explicitly asks for "left, right, forward, and backward." We keep the honest 4-class model (above 25% random chance) rather than a 3-class shortcut. The two-stage pipeline compensates: Stage 1 (79.4% binary) handles the critical rest-vs-active detection, while temporal smoothing (92.9% flicker reduction) stabilizes noisy direction estimates.

**Why RF over MLP?** The MLP on raw EEG (3000 inputs) achieves comparable Stage 1 accuracy (79.1%) and is 49x faster (0.3ms vs 13.3ms). But RF with hand-crafted features wins on 4-class direction (27.9% vs 24.3%) -- domain-specific PSD features capture motor imagery better across subjects.

**Why no TD-NIRS?** All 1728 NIRS features have zero variance across recordings. The hemodynamic response is likely too slow for the task windows or the sensors are too far from motor cortex. EEG alone is both necessary and sufficient.

**Phase Detection:** Each output is tagged INITIATION (STOP->active), SUSTAINED (same action), or RELEASE (active->STOP), addressing the challenge's "phase-aware modeling" direction.

## Limitations

- **6 channels** vs 64+ in research BCI systems
- **Frontal-heavy montage** (AFF6, AFp2, AFp1, AFF5) -- not ideal for motor cortex signals
- **Cross-subject generalization** is inherently harder than within-subject
- Left/right discrimination is weak because motor cortex channels (C3, C4) are missing
- 4-class direction accuracy (27.9%) is modest but above chance (25%)
- Tongue Tapping -> BACKWARD mapping is a convention; both are "novel" motor imagery

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design decisions.

## Robot Integration

Uses the [Nabla7/brain-robot-interface](https://github.com/Nabla7/brain-robot-interface) framework:
- Unitree G1 humanoid in MuJoCo simulation
- ONNX walking policy for locomotion
- 5 actions: FORWARD, BACKWARD, LEFT, RIGHT, STOP
- BCI commands sent via `Controller.set_action()` at >2 Hz
- Camera tracking follows robot position
- Fallback to matplotlib 2D visualizer if MuJoCo unavailable
