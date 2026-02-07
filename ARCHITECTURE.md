# ThoughtLink Architecture

## Design Decisions

### 1. Hierarchical Classification (Binary then Direction)

Instead of a flat 5-class classifier, we use a two-stage approach:

- **Stage 1 (Binary):** Rest vs Active -- catches the "no command" state with high confidence
- **Stage 2 (Direction):** Only runs when Stage 1 predicts Active -- reduces false triggers

This design matters for robot control because:
- False triggers (accidentally sending FORWARD when user is resting) are worse than missed commands
- The binary gate prevents the direction classifier from running on rest data
- Stage 2 sees cleaner training signal since it only trains on active samples

### 2. Bandpass Filter: 8-30 Hz

We filter to the mu (8-13 Hz) and beta (13-30 Hz) frequency bands because:
- **Mu rhythm** (~10 Hz) is suppressed during motor imagery and execution
- **Beta rhythm** (~20 Hz) shows event-related desynchronization during movement
- Frequencies below 8 Hz (delta, theta) contain mostly eye/movement artifacts
- Frequencies above 30 Hz contain mostly EMG noise with 6-channel consumer EEG

### 3. Cross-Subject Validation

All evaluation uses **leave-subject-out** splits:
- No subject appears in both training and test sets
- This tests generalization to new users without calibration
- Within-subject accuracy would be higher but unrealistic for a deployed system
- Cross-subject is the honest metric: the system works for anyone, not just trained users

### 4. Temporal Smoothing Over Raw Accuracy

For robot control, **command stability > raw accuracy**:

| Component | Purpose | Effect |
|-----------|---------|--------|
| MajorityVote(5) | Sliding window vote over 5 recent predictions | Removes transient flips |
| ConfidenceGate | Threshold-based gating on classifier probabilities | Prevents low-confidence actions |
| HysteresisFilter(3) | Requires 3 consecutive identical predictions to switch | Prevents oscillation |

Combined effect: 92.9% reduction in command flickering.

### 5. Channel Layout Analysis

The 6 EEG channels are:
- **AFF6** (right anterior frontal)
- **AFp2** (right anterior frontopolar)
- **AFp1** (left anterior frontopolar)
- **AFF5** (left anterior frontal)
- **FCz** (midline frontocentral)
- **CPz** (midline centroparietal)

Key observations:
- 4 of 6 channels are frontal/prefrontal -- **not** over motor cortex
- Motor imagery is best captured at C3/C4 (over left/right motor cortex), which are absent
- **FCz and CPz** are closest to motor areas and likely the most informative
- Left-right asymmetry (AFF5/AFp1 vs AFF6/AFp2) provides some laterality signal
- Feature importance analysis confirmed FCz and CPz dominate

### 6. 3-Class Fallback

The original 4-class direction problem (Left Fist, Right Fist, Both Fists, Tongue Tapping) achieved only 27.9% accuracy cross-subject (random = 25%). We merged:
- "Tongue Tapping" into "Both Fists" (both become FORWARD)

This gave a 3-class problem with 49.2% accuracy (random = 33%). The merge is natural since both actions map to "go forward" for robot control.

### 7. Feature Engineering

69 features per 1-second window:
- **24 PSD features**: 4 band powers (theta, alpha, beta, alpha/beta ratio) x 6 channels
- **42 statistical features**: 7 stats (variance, MAV, RMS, peak, kurtosis, skewness, zero crossings) x 6 channels
- **3 cross-channel features**: Left-right asymmetry (2) + midline difference (1)

The alpha/beta ratio captures mu suppression during motor imagery. Zero crossings approximate signal frequency. Cross-channel asymmetry helps distinguish left vs right imagery.

## Honest Limitations

1. **6 channels vs 64+**: Research BCI systems use 64-256 channels. Our 6-channel system has limited spatial resolution, especially for motor imagery which relies on channels over sensorimotor cortex (C3, Cz, C4).

2. **Frontal-heavy montage**: The Kernel Flow headset prioritizes frontal coverage. Motor imagery signals are strongest at central electrodes (C3, C4, Cz), which are not directly measured.

3. **Cross-subject difficulty**: Without per-user calibration, accuracy is fundamentally limited. Different people have different brain signal patterns, scalp thicknesses, and hair densities.

4. **Direction classification is weak**: 49.2% on 3 classes (vs 33% random) is statistically significant but not practically reliable. The left vs right distinction requires channels over left/right motor cortex.

5. **Offline processing**: We process pre-recorded .npz files, not live EEG streams. Real-time BCI would need streaming data, online artifact rejection, and tighter latency constraints.

## Judging Criteria Mapping

| Criterion | Our Approach |
|-----------|-------------|
| **Decoding Accuracy** | 79.4% binary (cross-subject), 49.2% 3-class direction |
| **Inference Latency** | 31ms headless, 146ms with MuJoCo (per window) |
| **Temporal Stability** | MajorityVote + Hysteresis + ConfidenceGate = 92.9% flicker reduction |
| **False Trigger Rate** | Binary gate prevents active commands during rest |
| **Scalability** | Cross-subject design, no per-user calibration needed |
| **Demo Clarity** | MuJoCo G1 humanoid sim, with matplotlib fallback |
| **Interpretability** | t-SNE, PSD plots, channel importance, confusion matrices |

## Data Flow

```
.npz file
  -> feature_eeg (7499, 6) at 500 Hz
  -> bandpass_filter 8-30 Hz
  -> extract active segment (samples 1500 to 1500+duration*500)
  -> normalize per channel (zero-mean, unit-variance)
  -> segment 1s windows, 0.5s overlap
  -> extract 69 features per window
  -> Stage 1: RandomForest -> P(active)
  -> if active: Stage 2: RandomForest -> direction class
  -> ConfidenceGate -> raw action string
  -> MajorityVote(5) -> smoothed action
  -> Hysteresis(3) -> final action
  -> BRI Controller.set_action(Action.FORWARD/LEFT/RIGHT/STOP)
  -> G1 humanoid moves in MuJoCo
```
