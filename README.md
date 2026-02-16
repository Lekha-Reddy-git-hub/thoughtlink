# ThoughtLink: Brain-to-Robot Fleet Orchestration

**Hack Nation 2026 | Challenge 9 | Kernel x Dimensional**

BCI systems control one robot. ThoughtLink controls 100.

We decode EEG brain signals into robot movement commands and route them through a three-layer fleet orchestration system. One operator, 100 autonomous Unitree G1 humanoids, intervening only when robots get stuck.

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
  +===========+
  | Fleet     |  100 robots, 10 groups
  | Router    |  Context-aware dispatch
  +===========+
```

## Quick Start

```bash
git clone https://github.com/Lekha-Reddy-git-hub/thoughtlink.git
cd thoughtlink
pip install uv && uv venv --python 3.12 && uv sync

# downloads data, trains models, clones BRI (~10 min)
uv run python setup_project.py

# run the fleet demo
uv run python demo/full_demo.py
```

<details>
<summary>manual setup</summary>

```bash
uv run python src/download.py
uv run python src/preprocess.py
uv run python src/features.py
uv run python src/stage1_binary.py
uv run python src/stage2_direction.py
git clone https://github.com/Nabla7/brain-robot-interface.git
```
</details>

## The Demo

`demo/full_demo.py` opens two windows:

**Left:** MuJoCo G1 humanoid close-up (follows whichever robot is being overridden).

**Right:** Fleet operations panel. 100 robots in 10 groups. Autonomous robots move in group colors. Stuck robots flash red with diagnosed failure reasons.

Three layers of control:

**Layer 1 (individual):** Click a brain signal button. Pipeline decodes one EEG recording, sends the action to one stuck robot. Direct but doesn't scale.

**Layer 2 (group):** Type `group 3 left`. Sends LEFT to every stuck robot in Group 3. One command, multiple robots.

**Layer 3 (context-aware):** Type `group 3 fix`. System diagnoses each robot's failure and prescribes the right corrective action per robot:
  - R22 has obstacle on left -> RIGHT
  - R25 lost its target -> FORWARD
  - R28 failed its task -> BACKWARD

One command, three robots, three different actions. The operator says WHERE. The system decides HOW.

`fix all` resolves every stuck robot fleet-wide. Press **A** for auto-play, **Q** to quit.

## Results

| Metric | Value |
|--------|-------|
| Stage 1 (Rest vs Active) | 79.4% cross-subject |
| Stage 2 (4-class direction) | 27.9% cross-subject (chance = 25%) |
| Inference latency | 24ms avg headless |
| Flicker reduction | 92.9% |
| Fleet size | 100 robots, 10 groups |
| Throughput | ~40 robots/sec per core |
| TD-NIRS fusion | no benefit (zero-variance features) |

All metrics are cross-subject (train on 5, test on 1 held out). No within-subject shortcuts.

## Why These Design Choices

**Two-stage classifier instead of flat 5-class:** False triggers (sending FORWARD during rest) are worse than missed commands. The binary gate (79.4%) prevents the direction classifier from running on rest data.

**4-class direction at 27.9%:** Modest but honest. The challenge asks for all four directions. We kept 4-class rather than dropping to 3-class. Cross-subject generalization with 6 frontal channels is inherently hard since motor cortex channels (C3/C4) are missing.

**Smoothing over raw accuracy:** For robot control, command stability matters more than per-window accuracy. Majority voting (5 windows) + confidence gating + hysteresis (3 consecutive) = 92.9% flicker reduction.

**RF over neural nets:** Random Forest with hand-crafted PSD features beats MLP on raw EEG for direction (27.9% vs 24.3%). Domain features capture motor imagery better cross-subject.

**TD-NIRS didn't help:** All 1,728 NIRS features had zero variance. Hemodynamic response is too slow for motor imagery windows.

## Fleet Architecture

Scaling from 1 to 100+ robots:

| Layer | Command | Effect |
|-------|---------|--------|
| Layer 1 | click button | 1 action -> 1 robot |
| Layer 2 | `group 3 left` | 1 action -> all stuck in G3 |
| Layer 3 | `group 3 fix` | 1 trigger -> N individualized actions |
| Layer 3 | `fix all` | 1 trigger -> all stuck robots fleet-wide |

Context-aware dispatch maps failure reason to corrective action: obstacle_left -> RIGHT, lost_target -> FORWARD, failed_task -> BACKWARD, unknown -> STOP. The BCI decoder runs once per decision cycle regardless of fleet size. Group routing is O(N) on pre-decoded intent.

At 26ms per robot, an 8-core machine handles 300+ robots at 1 Hz. Pure CPU, no GPU.

## Dataset

[KernelCo/robot_control](https://huggingface.co/datasets/KernelCo/robot_control) on HuggingFace. 900 recordings, 15s each at 500 Hz. 6 EEG channels (AFF6, AFp2, AFp1, AFF5, FCz, CPz). 5 labels. 6 subjects, 20 sessions.

## Modules

| Module | What it does |
|--------|-------------|
| `src/download.py` | dataset download + validation |
| `src/preprocess.py` | bandpass filter, segment, window |
| `src/features.py` | PSD + statistical features (69 per window) |
| `src/stage1_binary.py` | rest vs active classifier (RF) |
| `src/stage2_direction.py` | direction classifier (SVM, 4-class) |
| `src/smoothing.py` | majority vote + confidence gate + hysteresis |
| `src/pipeline.py` | end-to-end ThoughtLinkPipeline |
| `src/integration.py` | BRI robot integration (MuJoCo + fallback) |
| `src/cnn_baseline.py` | MLP vs RF comparison |
| `src/explore_nirs.py` | TD-NIRS multimodal fusion test |
| `src/temporal_embedding.py` | intent evolution PCA trajectories |
| `demo/full_demo.py` | 100-robot fleet demo |
| `demo/run_demo.py` | single file decode + robot |
| `demo/scalability_demo.py` | fleet scaling benchmark |

## Limitations

- 6 channels vs 64+ in research BCI systems
- 4 of 6 channels are frontal, not over motor cortex (C3/C4 missing)
- cross-subject generalization without calibration is fundamentally hard
- 4-class direction at 27.9% is above chance but not reliable for fine control
- offline processing (pre-recorded .npz, not live EEG stream)

See [ARCHITECTURE.md](ARCHITECTURE.md) for full design decisions.

## Built With

Python 3.12, scikit-learn, scipy, MuJoCo, Unitree G1, Kernel Flow EEG, [brain-robot-interface](https://github.com/Nabla7/brain-robot-interface) (Nabla7)

## Acknowledgments

Built at Hack Nation 2026. Dataset from [KernelCo](https://huggingface.co/datasets/KernelCo/robot_control). Simulation via [Nabla7/brain-robot-interface](https://github.com/Nabla7/brain-robot-interface).
