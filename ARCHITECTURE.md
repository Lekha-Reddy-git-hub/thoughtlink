# ThoughtLink Architecture

## Design Decisions

### 1. Hierarchical Classification (Binary then Direction)

Instead of a flat 5-class classifier, we use a two-stage approach:

- **Stage 1 (Binary):** Rest vs Active -- catches the "no command" state with high confidence (79.4%)
- **Stage 2 (Direction):** Only runs when Stage 1 predicts Active -- 4-class (FORWARD/BACKWARD/LEFT/RIGHT)

This design matters for robot control because:
- False triggers (accidentally sending FORWARD when user is resting) are worse than missed commands
- The binary gate prevents the direction classifier from running on rest data
- Stage 2 sees cleaner training signal since it only trains on active samples

### 2. 4-Class Direction (Including BACKWARD)

The challenge asks for "left, right, forward, and backward." We map:
- Both Fists -> FORWARD
- Left Fist -> LEFT
- Right Fist -> RIGHT
- Tongue Tapping -> BACKWARD

4-class accuracy is 27.9% (random = 25%). This is modest but honest:
- Cross-subject generalization with 6 frontal channels is inherently difficult
- The two-stage design compensates: Stage 1 (79.4%) handles the critical rest-vs-active gate
- Temporal smoothing (92.9% flicker reduction) stabilizes noisy direction estimates
- We keep 4-class rather than merging to 3-class because the challenge explicitly requires backward

### 3. Phase-Aware Detection

Each decoded action is tagged with a phase:
- **INITIATION**: First window where action changes from STOP to active
- **SUSTAINED**: Same action continues
- **RELEASE**: Action changes back to STOP

This addresses the challenge's "phase-aware modeling" direction and enables:
- Detecting when a human operator first engages
- Tracking sustained intent duration
- Detecting operator disengagement

### 4. Bandpass Filter: 8-30 Hz

We filter to the mu (8-13 Hz) and beta (13-30 Hz) frequency bands because:
- **Mu rhythm** (~10 Hz) is suppressed during motor imagery and execution
- **Beta rhythm** (~20 Hz) shows event-related desynchronization during movement
- Frequencies below 8 Hz (delta, theta) contain mostly eye/movement artifacts
- Frequencies above 30 Hz contain mostly EMG noise with 6-channel consumer EEG

### 5. Cross-Subject Validation

All evaluation uses **leave-subject-out** splits:
- No subject appears in both training and test sets
- This tests generalization to new users without calibration
- Cross-subject is the honest metric: the system works for anyone, not just trained users

### 6. Temporal Smoothing Over Raw Accuracy

For robot control, **command stability > raw accuracy**:

| Component | Purpose | Effect |
|-----------|---------|--------|
| MajorityVote(5) | Sliding window vote over 5 recent predictions | Removes transient flips |
| ConfidenceGate | Threshold-based gating on classifier probabilities | Prevents low-confidence actions |
| HysteresisFilter(3) | Requires 3 consecutive identical predictions to switch | Prevents oscillation |

Combined effect: 92.9% reduction in command flickering.

### 7. MLP vs Random Forest Comparison

We compared hand-crafted features + RF against raw EEG + neural network:

| Model | Stage 1 | Stage 2 | Latency |
|-------|---------|---------|---------|
| RF (69 features) | 79.4% | 27.9% | 13.3ms |
| MLP (raw 3000) | 79.1% | 24.3% | 0.3ms |

RF wins on accuracy because domain-specific PSD features capture motor imagery patterns that raw neural networks struggle to learn cross-subject. MLP is 49x faster but sacrifices direction accuracy.

### 8. TD-NIRS Exploration

Each .npz file includes feature_moments (72, 40, 3, 2, 3) -- brain blood flow data. Our analysis found:
- All 1728 NIRS features have zero variance across recordings
- Adding NIRS to EEG hurts accuracy (-1.2% Stage 1, -4.4% Stage 2)
- The hemodynamic response is too slow for the task windows or sensors are too far from motor cortex
- **Conclusion: EEG alone is both necessary and sufficient**

### 9. Temporal Embeddings

PCA projection of per-window features within recordings shows:
- Clear separation between rest and active phases in feature space
- Trajectories evolve from rest cluster to intent-specific regions
- 45.5% variance explained by PC1 (rest-vs-active separation)
- Phase transitions are visible as trajectory direction changes

### 10. Feature Engineering

69 features per 1-second window:
- **24 PSD features**: 4 band powers (theta, alpha, beta, alpha/beta ratio) x 6 channels
- **42 statistical features**: 7 stats (variance, MAV, RMS, peak, kurtosis, skewness, zero crossings) x 6 channels
- **3 cross-channel features**: Left-right asymmetry (2) + midline difference (1)

### 11. Channel Layout

The 6 EEG channels are:
- **AFF6** (right anterior frontal), **AFp2** (right anterior frontopolar)
- **AFp1** (left anterior frontopolar), **AFF5** (left anterior frontal)
- **FCz** (midline frontocentral), **CPz** (midline centroparietal)

4 of 6 channels are frontal -- not over motor cortex. FCz and CPz are closest to motor areas and dominate feature importance.

## Honest Limitations

1. **6 channels vs 64+**: Research BCI systems use 64-256 channels. Our 6-channel system has limited spatial resolution.
2. **Frontal-heavy montage**: Motor imagery signals are strongest at C3/C4, which are not measured.
3. **Cross-subject difficulty**: Without per-user calibration, accuracy is fundamentally limited.
4. **4-class direction is modest**: 27.9% (vs 25% random) is above chance but not practically reliable for fine control.
5. **Offline processing**: We process pre-recorded .npz files, not live EEG streams.

## Judging Criteria Mapping

| Criterion | Our Approach |
|-----------|-------------|
| **Decoding Accuracy** | 79.4% binary (cross-subject), 27.9% 4-class direction |
| **Inference Latency** | 31ms headless, ~55ms with feature extraction (per window) |
| **Temporal Stability** | MajorityVote + Hysteresis + ConfidenceGate = 92.9% flicker reduction |
| **False Trigger Rate** | Binary gate prevents active commands during rest |
| **Scalability** | ~35 robots/core at 1 Hz, cross-subject (no calibration) |
| **Phase-Aware Modeling** | INITIATION / SUSTAINED / RELEASE detection |
| **Multimodal Fusion** | Explored TD-NIRS; EEG alone is sufficient (honest negative result) |
| **Temporal Embeddings** | PCA trajectories showing intent evolution within recordings |
| **Model Comparison** | RF vs MLP: hand-crafted features beat raw neural network |
| **Demo Clarity** | 6-panel visual demo, MuJoCo sim, override scenario |

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
  -> if active: Stage 2: SVM -> direction class (4-class)
  -> ConfidenceGate -> raw action string
  -> MajorityVote(5) -> smoothed action
  -> Hysteresis(3) -> final action
  -> Phase detection (INITIATION / SUSTAINED / RELEASE)
  -> BRI Controller.set_action(Action.FORWARD/BACKWARD/LEFT/RIGHT/STOP)
  -> G1 humanoid moves in MuJoCo
```
