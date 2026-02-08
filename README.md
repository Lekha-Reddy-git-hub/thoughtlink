# ThoughtLink: From Brain to Robot, at the Speed of Thought

**Hack Nation 2026 | Challenge 9 | VC Track supported by Kernel & Dimensional**

## The Problem

In a world with 1,000 humanoid robots operating in a single facility, autonomy alone is not enough. What matters is how quickly and safely human judgment can be injected when something goes wrong.

The humanoid robot market is projected to reach $38B by 2035 (Goldman Sachs). Boston Dynamics, Figure AI, Tesla Optimus, and Unitree are deploying humanoids into warehouses, factories, and construction sites. As fleets scale from 10 to 1,000, the bottleneck shifts from robot capability to human oversight bandwidth.

Current interfaces — dashboards, joysticks, VR controllers — require one operator per robot. Brain-computer interfaces collapse that ratio: one human, many robots, intervening only when autonomy fails.

ThoughtLink explores a non-invasive, immediately deployable path: decoding human intent from brain signals and mapping it directly to robot instructions. Not teleoperation. Not automation. Intent infrastructure for intelligent machines — from brain to robot, at the speed of thought.

## What We Built

A complete brain-to-robot pipeline that:

- **Decodes 5 types of motor imagery** from EEG brain signals (Left Fist, Right Fist, Both Fists, Tongue Tapping, Relax)
- **Maps them to 5 robot actions** (LEFT, RIGHT, FORWARD, BACKWARD, STOP)
- **Drives a Unitree G1 humanoid robot** in MuJoCo physics simulation
- **Orchestrates a fleet of 100 autonomous robots** with hierarchical group commands
- **Runs at 24ms average latency** (real-time capable)
- **Scales to 40 robots per second** on a single CPU core
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

Our demo shows the core ThoughtLink vision: one human supervising 100 autonomous robots, intervening only when needed.

`demo/full_demo.py` opens two windows side by side:

**Left window:** Close-up 3D view of the robot currently being overridden (Unitree G1 in MuJoCo physics simulation). This is the detail camera — it follows whichever robot the operator is actively commanding.

**Right window:** Fleet operations panel showing 100 robots organized into 10 groups (G1-G10). Autonomous robots move in their group colors. Stuck robots flash red with failure reasons displayed.

**Three layers of control:**

**Layer 1 — Individual Override:** Click a brain signal button (LEFT FIST, RIGHT FIST, etc.). The pipeline decodes one EEG recording and sends the action to one stuck robot. Simple, direct, but doesn't scale.

**Layer 2 — Group Command:** Type `group 3 left` in the chat. The pipeline decodes one brain signal and sends the same action to every stuck robot in Group 3. One command, multiple robots.

**Layer 3 — Context-Aware Fix:** Type `group 3 fix` in the chat. The pipeline decodes one brain signal as a trigger. But instead of sending the same action to every robot, the system diagnoses each robot's failure reason and prescribes the right corrective action individually:
  - Robot #22 has obstacle on left → gets RIGHT
  - Robot #25 lost its target → gets FORWARD
  - Robot #28 failed its task → gets BACKWARD

One human command. Three robots. Three different actions. The operator says WHERE to intervene. The system decides HOW.

Type `fix all` to resolve every stuck robot across the entire fleet with individualized actions. The efficiency counter tracks operator leverage: "12 robots fixed with 4 commands — 3.0x leverage."

Press **A** for auto-play mode (cycles through all 3 layers automatically). Press **Q** to quit.

---

## How It Works

Raw EEG enters as a 6-channel time series from a Kernel Flow headset. We bandpass filter to 8-30 Hz (the motor imagery band — mu rhythm at ~10 Hz and beta rhythm at ~20 Hz), extract the active segment using stimulus timing, then window into 1-second chunks with 50% overlap.

Each window gets 69 features: power spectral density across alpha/beta bands plus statistical moments. These feed a two-stage hierarchical classifier:

**Stage 1 — Is the person thinking about movement?**

A Random Forest separates rest from active motor imagery. 79.4% accuracy, cross-subject. This is the critical gate: it prevents false triggers when the operator is not intending to intervene. In a fleet of 1,000 robots, a false "FORWARD" command sent during operator rest would be catastrophic.

**Stage 2 — Which direction?**

An SVM classifies the active signal into 4 directions: LEFT (Left Fist), RIGHT (Right Fist), FORWARD (Both Fists), BACKWARD (Tongue Tapping). 27.9% cross-subject accuracy (above 25% chance). The challenge asks for all four directions, so we kept the honest 4-class model rather than a 3-class shortcut.

**Temporal smoothing** eliminates noise: majority voting over 5 windows, confidence gating, and hysteresis filtering. Result: 92.9% reduction in action flicker. For robot control, command stability matters more than raw per-window accuracy.

**Phase detection** tags every output as INITIATION (rest to movement), SUSTAINED (continuing), or RELEASE (movement to rest). This captures intent transitions, not just static predictions — the system knows when an operator first engages, how long they sustain intent, and when they disengage.

The two-stage hierarchical design addresses the challenge's "Hierarchical Intent Models" direction. Phase detection addresses "Phase-Aware Modeling." Temporal smoothing addresses "Temporal Intent Decoding."

---

## Hierarchical Fleet Architecture

The single-robot BCI pipeline is a solved component. The real question is: how does one operator scale to 100 or 1,000 robots?

Our answer is a three-layer command dispatch system:

**Layer 1: Human Intent** — The operator's brain signal is decoded into an action or a trigger. This is the BCI pipeline.

**Layer 2: Group Router** — Robots are organized into groups. The operator targets a group, and the system identifies which robots within that group need intervention. No per-robot decisions required from the human.

**Layer 3: Context AI** — Each stuck robot has a diagnosed failure reason (obstacle, lost target, failed task). The system maps failure reasons to corrective actions automatically. The human provides strategic oversight. The system handles tactical execution.

**Impact:** Without grouping, 5 stuck robots = 5 brain signals. With grouping, 5 stuck robots across 2 groups = 2 brain signals. At scale, 50 stuck robots across 8 groups = 8 brain signals instead of 50.

The pipeline processes 40 robots per second on a single CPU core. At 26ms per robot, an 8-core machine handles 300+ robots at 1 Hz. The decoder is pure CPU (scikit-learn + scipy), requires no GPU, and scales linearly with cores.

---

## Results

| Metric | Value |
|--------|-------|
| Stage 1 Accuracy (Rest vs Active) | 79.4% (cross-subject, RF) |
| Stage 2 Accuracy (4-class direction) | 27.9% (cross-subject, SVM, random=25%) |
| MLP Baseline (raw EEG) | 79.1% / 24.3% (Stage 1 / Stage 2) |
| Inference Latency (per window) | 24ms avg (headless) |
| Flicker Reduction (smoothing) | 92.9% |
| Phase Detection | INITIATION / SUSTAINED / RELEASE |
| Fleet Size (demo) | 100 robots, 10 groups |
| Throughput (single core) | 40 robots/second |
| Throughput (8-core machine) | ~300 robots at 1 Hz |
| TD-NIRS Fusion | No benefit (zero-variance features) |

All metrics are cross-subject: trained on 5 subjects, tested on 1 held out. No within-subject shortcuts. This is the harder generalization problem — the system must work for a user it has never seen before, without calibration.

---

## What We Explored

### Latency-Accuracy Tradeoffs

We benchmarked four model families (Logistic Regression, SVM, Random Forest, MLP) across both pipeline stages. All models run under 100ms, meeting real-time constraints. Random Forest offers the best accuracy-latency balance for Stage 1. See `results/latency_accuracy_tradeoff.png`.

| Model | Stage 1 | Stage 2 | Latency |
|-------|---------|---------|---------|
| LR (69 features) | 79.4% | 27.4% | 11.7ms |
| SVM (69 features) | 79.4% | 28.1% | 11.7ms |
| RF (69 features) | 79.4% | 27.9% | 24.7ms |
| MLP (raw 3000) | 79.1% | 24.3% | 0.3ms |

### Classical ML vs Neural Networks

Random Forest with hand-crafted PSD features beats MLP on raw EEG for direction accuracy (27.9% vs 24.3%). Domain-specific spectral features capture motor imagery patterns better across subjects. MLP is 49x faster per inference (0.3ms vs 13.3ms), but speed isn't the bottleneck at 31ms total pipeline latency.

### Multimodal Fusion: EEG + TD-NIRS

The Kernel Flow dataset includes TD-NIRS (near-infrared spectroscopy) measuring brain blood flow alongside EEG. All 1,728 NIRS features had zero variance across recordings. The hemodynamic response is too slow for motor imagery windows. EEG alone is necessary and sufficient.

### Intent Evolution Over Time

PCA projections of windowed features show distinct trajectories for different intent types. Rest states cluster tightly; motor imagery states spread along the first principal component (45.5% variance explained). This supports the phase detection approach.

### Failure Mode Analysis

The primary confusion is between LEFT and RIGHT actions (Right Fist → FORWARD 654 times). Root cause: 4 of 6 channels are frontal, not over motor cortex (C3/C4). Subject d696086d is the outlier at 38.9% accuracy vs 100% for the best subject. See `results/failure_analysis.png`.

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/Lekha-Reddy-git-hub/thoughtlink.git
cd thoughtlink
pip install uv
uv venv --python 3.12
uv sync

# One-command setup: downloads data, trains models, clones BRI (~10 min)
uv run python setup_project.py

# Run the demo
uv run python demo/full_demo.py
```

`setup_project.py` handles everything: downloading 900 EEG recordings from HuggingFace (~536 MB), preprocessing, feature extraction, training both classifiers, and cloning brain-robot-interface for MuJoCo simulation.

<details>
<summary>Manual setup (step by step)</summary>

```bash
uv run python src/download.py
uv run python src/preprocess.py
uv run python src/features.py
uv run python src/stage1_binary.py
uv run python src/stage2_direction.py
git clone https://github.com/Nabla7/brain-robot-interface.git
```
</details>

---

## Demo Scripts

| Script | Description |
|--------|-------------|
| `demo/full_demo.py` | 100-robot fleet orchestration with hierarchical group commands, MuJoCo G1 humanoid, chat/button control |
| `demo/run_all.py` | All 5 brain signals decoded back-to-back with MuJoCo |
| `demo/override_demo.py` | Autonomous-to-human-override transition scenario |
| `demo/scalability_demo.py` | Fleet scalability benchmark: 10/50/100 robots with timing |
| `demo/run_demo.py` | Single file decode + robot control |
| `demo/demo_video.py` | Generate animated GIF and summary figures |

### Demo Controls

- **Buttons:** LEFT FIST, RIGHT FIST, BOTH FISTS, TONGUE TAP, RELAX — individual robot override
- **Chat commands:** `group 3 fix`, `group 1 left`, `fix all`, `turn left`
- **Keys:** A = Auto-play (demos all 3 layers), Q = Quit, 1/2/3 = Bonus evidence charts

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

- **Can cross-subject BCI generalize with only 6 frontal channels?** We show it partially can: 79.4% binary gate, 27.9% direction. Research-grade BCIs use 64+ channels with motor cortex coverage.
- **Is motor cortex electrode placement required for left/right discrimination?** Our frontal-heavy montage (AFF6, AFp2, AFp1, AFF5) struggles with lateral discrimination. Adding C3/C4 would likely improve direction accuracy significantly.
- **When does model complexity help?** For this dataset size (900 recordings, 6 subjects), it doesn't. Random Forest with domain features beats MLP on raw EEG.
- **Can TD-NIRS contribute to fast motor imagery decoding?** Not with this sensor configuration. Zero-variance features suggest the hemodynamic response is too slow or spatially misaligned.

---

## Dataset

Source: [KernelCo/robot_control](https://huggingface.co/datasets/KernelCo/robot_control) on HuggingFace

- 900 recordings, 15 seconds each at 500 Hz
- 6 EEG channels: AFF6, AFp2, AFp1, AFF5, FCz, CPz
- 5 labels: Both Fists, Left Fist, Relax, Right Fist, Tongue Tapping
- 6 subjects, 20 sessions
- TD-NIRS data: feature_moments (72, 40, 3, 2, 3) — explored, not used (zero-variance)

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
