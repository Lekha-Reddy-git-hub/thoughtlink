# ThoughtLink: From Brain to Robot, at the Speed of Thought

**Hack Nation 2026 | Challenge 9 | VC Track supported by Kernel & Dimensional**

---

Imagine a factory with 1,000 humanoid robots operating simultaneously. Each robot performs assembly, inspection, navigation, or logistics. Most of the time, autonomy works. But when something unexpected happens — an obstacle, a misplaced object, a failed task — the system needs human judgment, immediately.

Today, that judgment flows through keyboards, dashboards, and manual overrides. This does not scale.

The humanoid robot market is projected to reach $38B by 2035 (Goldman Sachs). Boston Dynamics, Figure AI, Tesla Optimus, and Unitree are deploying humanoids into warehouses, factories, and construction sites. As fleets scale from 10 to 1,000, the bottleneck shifts from robot capability to human oversight bandwidth.

Current interfaces — dashboards, joysticks, VR controllers — require one operator per robot. Brain-computer interfaces collapse that ratio: one human, many robots, intervening only when autonomy fails.

ThoughtLink explores a non-invasive, immediately deployable path: decoding human intent from brain signals and mapping it directly to robot instructions. Not teleoperation. Not automation. Intent infrastructure for intelligent machines — from brain to robot, at the speed of thought.

---

## What We Built

A complete brain-to-robot pipeline that:

- **Decodes 5 types of motor imagery** from EEG brain signals (Left Fist, Right Fist, Both Fists, Tongue Tapping, Relax)
- **Maps them to 5 robot actions** (LEFT, RIGHT, FORWARD, BACKWARD, STOP)
- **Drives a Unitree G1 humanoid robot** in MuJoCo physics simulation
- **Runs at 24ms average latency** (real-time capable)
- **Scales to 35+ simultaneous robots** per CPU core
- **Detects intent phases:** INITIATION, SUSTAINED, RELEASE

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
  | Stage 1   |---->| Stage 2   |    (only if active)
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

---

## The Demo

Our demo shows the core ThoughtLink vision: one human supervising a fleet of autonomous robots, intervening only when needed.

When you run `demo/full_demo.py`, two windows open side by side:

**Left window:** A Unitree G1 humanoid robot in MuJoCo 3D physics simulation. It walks, turns, and stops based on decoded brain signals.

**Right window:** A fleet operations panel showing:
- A live 2D map of 10 robots operating autonomously
- When one robot gets stuck (flashes red), the operator sends an override
- Click buttons (LEFT FIST, RIGHT FIST, BOTH FISTS, TONGUE TAP, RELAX) to trigger real EEG decoding
- Or type natural language commands: "turn left", "go forward", "stop"
- Every command loads a real pre-recorded brain signal, processes it through the full pipeline, and sends the decoded action
- Live metrics update: latency, confidence, pipeline stages, phase detection

This is not remote control. The operator doesn't drive each robot. 9 robots run autonomously while the operator intervenes on the 1 that needs help. One thought, one override, back to autonomous.

Press **A** for auto-play mode (cycles through all 5 brain signals automatically). Press **Q** to quit.

---

## How It Works

Raw EEG enters as a 6-channel time series from a Kernel Flow headset. We bandpass filter to 8-30 Hz (the motor imagery band — mu rhythm at ~10 Hz and beta rhythm at ~20 Hz), extract the active segment using stimulus timing, then window into 1-second chunks with 50% overlap.

Each window gets 69 features: power spectral density across theta/alpha/beta bands (24 features), statistical moments like variance, kurtosis, and zero crossings (42 features), and cross-channel asymmetry measures (3 features). These feed a two-stage hierarchical classifier:

**Stage 1 — Is the person thinking about movement?**

A Random Forest separates rest from active motor imagery. 79.4% accuracy, cross-subject. This is the critical gate: it prevents false triggers when the operator is not intending to intervene. In a fleet of 1,000 robots, a false "FORWARD" command sent during operator rest would be catastrophic.

**Stage 2 — Which direction?**

An SVM classifies the active signal into 4 directions: LEFT (Left Fist), RIGHT (Right Fist), FORWARD (Both Fists), BACKWARD (Tongue Tapping). 27.9% cross-subject accuracy (above 25% chance). The challenge asks for all four directions, so we kept the honest 4-class model rather than merging classes for a higher number.

**Temporal smoothing** eliminates noise: majority voting over 5 windows, confidence gating (0.6 threshold for Stage 1, 0.4 for Stage 2), and hysteresis filtering requiring 3 consecutive identical predictions to switch. Result: 92.9% reduction in action flicker. For robot control, command stability matters more than raw per-window accuracy.

**Phase detection** tags every output as INITIATION (rest to movement), SUSTAINED (continuing), or RELEASE (movement to rest). This captures intent transitions, not just static predictions — the system knows when an operator first engages, how long they sustain intent, and when they disengage.

The two-stage hierarchical design directly addresses the challenge's "Hierarchical Intent Models" direction. Phase detection addresses "Phase-Aware Modeling." Temporal smoothing addresses "Temporal Intent Decoding."

---

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
| Total EEG Files | 900 (865 usable after filtering) |
| Subjects | 6 unique |
| EEG Channels | 6 (4 frontal + 2 midline) |
| Brain Signal Classes | Relax, Left Fist, Right Fist, Both Fists, Tongue Tapping |
| Robot Actions | STOP, LEFT, RIGHT, FORWARD, BACKWARD |

All metrics are cross-subject: trained on 5 subjects, tested on 1 held out. No within-subject shortcuts. This is the harder generalization problem — the system must work for a user it has never seen before, without calibration.

---

## What We Explored

### Classical ML vs Neural Networks

We compared Random Forest with hand-crafted PSD features against an MLP on raw EEG (3,000 input dimensions). RF wins on direction accuracy (27.9% vs 24.3%) because domain-specific spectral features capture motor imagery patterns better across subjects. MLP is 49x faster per inference (0.3ms vs 13.3ms), but speed isn't the bottleneck when total pipeline latency is 31ms. A fast, simple model that works beats a slow, complex model that doesn't.

| Model | Stage 1 | Stage 2 | Latency |
|-------|---------|---------|---------|
| RF (69 features) | 79.4% | 27.9% | 13.3ms |
| MLP (raw 3000) | 79.1% | 24.3% | 0.3ms |

### Multimodal Fusion: EEG + TD-NIRS

The Kernel Flow dataset includes TD-NIRS (time-domain near-infrared spectroscopy) measuring brain blood flow alongside EEG. We tested multimodal fusion. All 1,728 NIRS features had zero variance across recordings — the hemodynamic response is too slow for the 3-12 second motor imagery windows, or the sensors are positioned too far from motor cortex. Adding NIRS hurt accuracy (Stage 1: 79.4% to 76.0%). EEG alone is necessary and sufficient for this task.

### Intent Evolution Over Time

We projected windowed features into 2D via PCA to visualize how brain activity evolves within a single recording. Different intent types follow distinct trajectories through feature space. Rest states cluster tightly; motor imagery states spread along the first principal component (45.5% variance explained). This supports the phase detection approach: transitions between clusters mark INITIATION and RELEASE events.

---

## Quick Start

```bash
# Clone and install dependencies
git clone https://github.com/Lekha-Reddy-git-hub/thoughtlink.git
cd thoughtlink
pip install uv
uv venv --python 3.12
uv sync

# One-command setup: downloads data, trains models, clones BRI
uv run python setup_project.py

# THE demo: multi-robot orchestration + MuJoCo G1
uv run python demo/full_demo.py

# Other demos
uv run python demo/run_all.py              # All 5 actions with MuJoCo
uv run python demo/override_demo.py        # Autonomous override scenario
uv run python demo/scalability_demo.py     # Fleet scalability test
uv run python demo/run_demo.py data/0b2dbd41-10.npz  # Single file
```

`setup_project.py` handles everything: downloading the 900 EEG recordings from HuggingFace (~536 MB), preprocessing, feature extraction, training both classifiers, and cloning the brain-robot-interface repo for MuJoCo simulation. Takes ~5-10 minutes.

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

---

## Demo Scripts

| Script | Description |
|--------|-------------|
| `demo/full_demo.py` | Multi-robot orchestration: 10-robot fleet view + MuJoCo G1 humanoid + chat/button control |
| `demo/run_all.py` | All 5 brain signals decoded back-to-back with MuJoCo |
| `demo/override_demo.py` | Autonomous-to-human-override transition scenario |
| `demo/scalability_demo.py` | Fleet scalability test: 10 robots, 38 robots/core throughput |
| `demo/run_demo.py` | Single file decode + robot control |
| `demo/demo_video.py` | Generate animated GIF and summary figures |

---

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

---

## Limitations and Open Research Questions

- **Can cross-subject BCI generalize with only 6 frontal channels?** We show it partially can: 79.4% binary gate, 27.9% direction. Research-grade BCIs use 64+ channels with motor cortex coverage (C3, C4).

- **Is motor cortex electrode placement required for left/right discrimination?** Our frontal-heavy montage (AFF6, AFp2, AFp1, AFF5) struggles with lateral discrimination. Adding C3/C4 would likely improve direction accuracy significantly.

- **When does model complexity help?** For this dataset size (900 recordings, 6 subjects), it doesn't. Random Forest with domain features beats MLP on raw EEG. Transformers and CNNs need more data to justify their parameter count.

- **Can TD-NIRS contribute to fast motor imagery decoding?** Not with this sensor configuration. The zero-variance features suggest the hemodynamic response is too slow or spatially misaligned for motor cortex.

- **Tongue Tapping as BACKWARD:** This mapping is a convention. Both are "novel" motor imagery classes. A dedicated "backward" mental task might improve decoding.

---

## Dataset

Source: [KernelCo/robot_control](https://huggingface.co/datasets/KernelCo/robot_control) on HuggingFace

- 900 recordings, 15 seconds each at 500 Hz
- 6 EEG channels: AFF6, AFp2, AFp1, AFF5, FCz, CPz
- 5 labels: Both Fists, Left Fist, Relax, Right Fist, Tongue Tapping
- 6 subjects, 20 sessions
- TD-NIRS data: feature_moments (72, 40, 3, 2, 3) — explored, not used (zero-variance across recordings)

---

## Built With

Python 3.12 | scikit-learn | scipy | MuJoCo | Unitree G1 | Kernel Flow EEG | HuggingFace Datasets | [brain-robot-interface](https://github.com/Nabla7/brain-robot-interface) (Nabla7)

---

## Acknowledgments

Built for the **ThoughtLink Challenge** at Hack Nation 2026, sponsored by **Kernel** and **Dimensional**.

**Dataset:** 900 EEG recordings collected using a Kernel Flow headset, provided by [KernelCo on HuggingFace](https://huggingface.co/datasets/KernelCo/robot_control).

**Kernel Flow** is a non-invasive neural interface that records TD-NIRS and EEG simultaneously from 40+ modules across the scalp. In this challenge, we used the 6-channel EEG data (AFF6, AFp2, AFp1, AFF5, FCz, CPz) sampled at 500 Hz.

**Simulation:** Unitree G1 humanoid via [brain-robot-interface](https://github.com/Nabla7/brain-robot-interface) by Nabla7.

**Challenge:** *ThoughtLink — From Brain to Robot* explores the missing intelligence layer between human intent and robotic action. If successful, one human can supervise dozens — or hundreds — of robots, intervening only where context and judgment are required.

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design decisions and data flow.
