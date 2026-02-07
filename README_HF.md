---
title: ThoughtLink - Brain-to-Robot Intent Decoder
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: demo/app.py
pinned: false
license: mit
---

# ThoughtLink: Brain-to-Robot Intent Decoder

**Hack Nation 2026 | Challenge 9**

Real-time BCI pipeline that decodes 5 motor imagery classes from 6-channel EEG into robot directional commands (LEFT, RIGHT, FORWARD, BACKWARD, STOP).

## Features

- **Two-stage classification**: Binary rest/active gate (79.4%) + 4-class direction decoder
- **Temporal smoothing**: MajorityVote + ConfidenceGate + Hysteresis = 92.9% flicker reduction
- **<100ms latency**: Real-time capable on CPU (no GPU required)
- **Cross-subject**: Works on unseen users without calibration

## Tabs

1. **Live Demo** — Select a signal type, run the full pipeline, see EEG, robot direction, and metrics
2. **Model Comparison** — RF vs MLP analysis
3. **Analysis** — Confusion matrices, failure analysis, temporal embeddings
4. **Scalability** — Live test: how many robots can one pipeline instance serve?
5. **Architecture** — Full design document
