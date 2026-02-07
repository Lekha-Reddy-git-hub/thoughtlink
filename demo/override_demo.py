"""
ThoughtLink Override Demo -- Autonomous-to-Human-Override-to-Autonomous
=======================================================================
Demonstrates the core ThoughtLink concept in MuJoCo:
  Robots operate autonomously. When one needs help, a human operator
  overrides with a single brain signal. The robot acts, and autonomy
  resumes. One human thought, one robot correction.

Usage:  python demo/override_demo.py
"""
import sys
import os
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

LEFT_FIST_FILE = os.path.join(project_root, "data", "2562e7bd-14.npz")
RELAX_FILE = os.path.join(project_root, "data", "2161ecb6-12.npz")


def banner(text, char="=", width=70):
    print()
    print(char * width)
    print(f"  {text}")
    print(char * width)
    print()


def send_action_for(ctrl, action, seconds, viewer_data, label=""):
    """Send an action continuously for a duration, tracking the camera."""
    viewer, data = viewer_data
    t_start = time.time()
    last_printed = -1
    while time.time() - t_start < seconds:
        ctrl.set_action(action)
        if data is not None and viewer is not None and viewer.is_running():
            viewer.cam.lookat[0] = float(data.qpos[0])
            viewer.cam.lookat[1] = float(data.qpos[1])
            viewer.cam.lookat[2] = 0.8
        elapsed = time.time() - t_start
        sec_int = int(elapsed)
        if label and sec_int > last_printed:
            last_printed = sec_int
            print(f"    [{sec_int:2d}s / {seconds:.0f}s]  {label}")
        time.sleep(0.05)


def main():
    # -- Setup --
    bri_src = os.path.join(project_root, "brain-robot-interface", "src")
    sys.path.insert(0, bri_src)
    from bri import Action, Controller
    from pipeline import ThoughtLinkPipeline

    bundle_dir = os.path.join(project_root, "brain-robot-interface", "bundles", "g1_mjlab")

    banner("ThoughtLink: Autonomous Override Demo")
    print("  Scenario: Robot #47 is walking autonomously when it")
    print("  encounters an obstacle. A human operator overrides")
    print("  with a brain signal to steer it LEFT, then disengages.")
    print("  The robot resumes autonomous navigation.")
    print()

    # -- Start MuJoCo --
    ctrl = Controller(
        backend="sim", hold_s=1.0,
        forward_speed=0.4, yaw_rate=1.0, smooth_alpha=0.3,
        bundle_dir=bundle_dir,
    )
    print("  Starting MuJoCo simulation...")
    ctrl.start()

    viewer = ctrl._backend._viewer
    data = ctrl._backend._data
    vd = (viewer, data)

    # Camera setup
    if viewer is not None:
        viewer.cam.distance = 5.0
        viewer.cam.azimuth = 180.0
        viewer.cam.elevation = -25.0
        if data is not None:
            viewer.cam.lookat[0] = float(data.qpos[0])
            viewer.cam.lookat[1] = float(data.qpos[1])
            viewer.cam.lookat[2] = 0.8

    print("  MuJoCo viewer ready.\n")
    time.sleep(1.0)

    t_demo_start = time.time()

    # ============================================================
    # PHASE 1: Autonomous forward navigation (6 seconds)
    # ============================================================
    banner("AUTONOMOUS MODE -- Robot #47 navigating normally", "-")
    auto1_start = time.time()
    send_action_for(ctrl, Action.FORWARD, 6.0, vd, label="FORWARD (autonomous)")
    auto1_dur = time.time() - auto1_start

    # ============================================================
    # PHASE 2: Anomaly detected, request human guidance
    # ============================================================
    banner("ANOMALY DETECTED -- Robot #47 encountered obstacle", "!")
    print("  Requesting human operator guidance...")
    print("  Robot is STOPPED, awaiting brain signal input.")
    ctrl.set_action(Action.STOP)
    send_action_for(ctrl, Action.STOP, 2.0, vd, label="STOP (awaiting operator)")

    # ============================================================
    # PHASE 3: Decode Left Fist brain signal -> LEFT override
    # ============================================================
    banner("BRAIN SIGNAL RECEIVED -- Decoding operator intent...", "-")
    print(f"  EEG source: {os.path.basename(LEFT_FIST_FILE)}")
    print(f"  Ground truth label: Left Fist")
    print()

    pipeline = ThoughtLinkPipeline()
    pipeline.load_models(
        os.path.join(project_root, "models", "stage1_binary.pkl"),
        os.path.join(project_root, "models", "stage2_direction.pkl"),
    )

    # Decode and drive the robot with the brain signal
    override_start = time.time()
    decoded_actions = []
    decoded_confs = []
    decoded_lats = []

    for action_str, conf, lat, phase in pipeline.process_file(LEFT_FIST_FILE):
        bri_action = {"FORWARD": Action.FORWARD, "LEFT": Action.LEFT,
                      "RIGHT": Action.RIGHT, "STOP": Action.STOP}[action_str]
        ctrl.set_action(bri_action)
        decoded_actions.append(action_str)
        decoded_confs.append(conf)
        decoded_lats.append(lat)
        print(f"    Decoded: {action_str:8s} [{phase:10s}]  conf={conf:.2f}  lat={lat:.1f}ms")

        # Pace at real-time (0.5s per window) + camera tracking
        sleep_t = max(0, 0.5 - lat / 1000)
        t0 = time.time()
        while time.time() - t0 < sleep_t:
            ctrl.set_action(bri_action)
            if data is not None and viewer is not None and viewer.is_running():
                viewer.cam.lookat[0] = float(data.qpos[0])
                viewer.cam.lookat[1] = float(data.qpos[1])
                viewer.cam.lookat[2] = 0.8
            time.sleep(0.05)

    override_dur = time.time() - override_start

    from collections import Counter
    action_dist = Counter(decoded_actions)
    dominant = action_dist.most_common(1)[0][0]
    avg_conf = sum(decoded_confs) / len(decoded_confs)
    avg_lat = sum(decoded_lats) / len(decoded_lats)

    print()
    print(f"  HUMAN OVERRIDE: {dominant} "
          f"(confidence: {avg_conf:.2f}, latency: {avg_lat:.0f}ms)")
    print(f"  Action distribution: {dict(action_dist)}")

    # ============================================================
    # PHASE 4: Decode Relax brain signal -> operator disengages
    # ============================================================
    banner("CHECKING OPERATOR STATE -- Decoding disengage signal...", "-")
    print(f"  EEG source: {os.path.basename(RELAX_FILE)}")
    print(f"  Ground truth label: Relax")
    print()

    pipeline2 = ThoughtLinkPipeline()
    pipeline2.load_models(
        os.path.join(project_root, "models", "stage1_binary.pkl"),
        os.path.join(project_root, "models", "stage2_direction.pkl"),
    )

    relax_actions = []
    for action_str, conf, lat, phase in pipeline2.process_file(RELAX_FILE):
        bri_action = {"FORWARD": Action.FORWARD, "LEFT": Action.LEFT,
                      "RIGHT": Action.RIGHT, "STOP": Action.STOP}[action_str]
        ctrl.set_action(bri_action)
        relax_actions.append(action_str)
        # Fast-forward (no real-time pacing for the check)
        if data is not None and viewer is not None and viewer.is_running():
            viewer.cam.lookat[0] = float(data.qpos[0])
            viewer.cam.lookat[1] = float(data.qpos[1])
            viewer.cam.lookat[2] = 0.8
        time.sleep(0.1)

    relax_dist = Counter(relax_actions)
    relax_dominant = relax_dist.most_common(1)[0][0]
    print(f"  Decoded: {relax_dominant} ({dict(relax_dist)})")

    banner("OPERATOR DISENGAGED -- Resuming autonomous navigation", "-")

    # ============================================================
    # PHASE 5: Resume autonomous forward navigation (5 seconds)
    # ============================================================
    auto2_start = time.time()
    send_action_for(ctrl, Action.FORWARD, 5.0, vd, label="FORWARD (autonomous)")
    auto2_dur = time.time() - auto2_start

    total_auto = auto1_dur + auto2_dur
    total_time = time.time() - t_demo_start

    # ============================================================
    # Summary
    # ============================================================
    ctrl.set_action(Action.STOP)
    time.sleep(0.5)
    ctrl.stop()

    banner("OVERRIDE CYCLE COMPLETE")
    print(f"  Phase 1 (autonomous FORWARD):  {auto1_dur:.1f}s")
    print(f"  Phase 2 (anomaly + pause):      2.0s")
    print(f"  Phase 3 (human override LEFT):  {override_dur:.1f}s")
    print(f"  Phase 4 (disengage check):      quick")
    print(f"  Phase 5 (autonomous FORWARD):  {auto2_dur:.1f}s")
    print()
    print(f"  Human intervention:  {override_dur:.1f}s")
    print(f"  Total autonomous:    {total_auto:.1f}s")
    print(f"  Total demo time:     {total_time:.1f}s")
    print()
    print("  This demonstrates intent-level control -- the operator")
    print("  thinks LEFT once, the robot turns, and autonomy resumes.")
    print("  No continuous manual control needed. One brain signal")
    print("  corrects the robot's path, then the system self-recovers.")
    print()
    print(f"  Brain decode stats:")
    print(f"    Override action:  {dominant} (decoded from Left Fist EEG)")
    print(f"    Avg confidence:   {avg_conf:.2f}")
    print(f"    Avg latency:      {avg_lat:.0f}ms")
    print(f"    Disengage signal: {relax_dominant} (decoded from Relax EEG)")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
