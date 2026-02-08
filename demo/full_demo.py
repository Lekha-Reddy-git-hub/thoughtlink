"""
ThoughtLink: Multi-Robot Orchestration Demo (100 Robots)
=========================================================
100 robots in 10 groups. Multiple robots get stuck simultaneously.
The operator uses hierarchical commands:
  Layer 1: Button click -> override ONE stuck robot
  Layer 2: "group 3 left" -> send LEFT to all stuck robots in Group 3
  Layer 3: "group 3 fix" -> individualized actions per robot's failure reason
  Layer 3+: "fix all" -> individualized actions for ALL stuck robots

Window 1 (left):  MuJoCo G1 humanoid
Window 2 (right): Tkinter control panel -- fleet view, metrics, controls

Controls:
  Click a signal button to override one stuck robot.
  Type "group 3 left" to send LEFT to all stuck in Group 3.
  Type "group 3 fix" for context-aware individualized overrides.
  Type "fix all" for fleet-wide context-aware override.
  Press A for auto-play (showcases all 3 layers).
  Press Q to quit. Press 1/2/3 for evidence charts.

Usage: python demo/full_demo.py
"""
import sys
import os
import time
import random
import math
import re
import threading
import tkinter as tk
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from collections import Counter

from pipeline import ThoughtLinkPipeline

# =====================================================================
#  CONSTANTS
# =====================================================================

BG = "#0d1117"
BG2 = "#161b22"
BORDER = "#30363d"
TEXT = "#c9d1d9"
DIM = "#8b949e"
GREEN = "#34a853"
RED = "#ea4335"
BLUE = "#4285f4"
YELLOW = "#fbbc04"
GRAY = "#9aa0a6"

ACTION_COLORS = {
    "LEFT": BLUE, "RIGHT": RED, "FORWARD": GREEN,
    "BACKWARD": YELLOW, "STOP": GRAY,
}
CHANNEL_NAMES = ["AFF6", "AFp2", "AFp1", "AFF5", "FCz", "CPz"]
CHANNEL_COLORS = ["#4285f4", "#ea4335", "#fbbc04", "#34a853", "#ff6d00", "#ab47bc"]
INTENT_MAP = {
    "Left Fist": "LEFT", "Right Fist": "RIGHT",
    "Both Fists": "FORWARD", "Tongue Tapping": "BACKWARD",
    "Relax": "STOP",
}

DEMO_FILES = {
    "Left Fist":      "2562e7bd-14.npz",
    "Right Fist":     "0b2dbd41-34.npz",
    "Both Fists":     "4787dfb9-10.npz",
    "Tongue Tapping": "0b2dbd41-16.npz",
    "Relax":          "2161ecb6-12.npz",
}

NUM_ROBOTS = 100
NUM_GROUPS = 10
ROBOTS_PER_GROUP = 10

GROUP_COLORS = {
    1: "#4CAF50", 2: "#2196F3", 3: "#FF9800", 4: "#9C27B0", 5: "#00BCD4",
    6: "#FFEB3B", 7: "#E91E63", 8: "#8BC34A", 9: "#FF5722", 10: "#FFFFFF",
}

FAILURE_ACTIONS = {
    "obstacle_left": "RIGHT",
    "obstacle_right": "LEFT",
    "lost_target": "FORWARD",
    "failed_task": "BACKWARD",
    "unknown": "STOP",
}
FAILURE_REASONS = list(FAILURE_ACTIONS.keys())

CHAT_DIRECTION_MAP = {
    "left": "Left Fist", "right": "Right Fist", "forward": "Both Fists",
    "backward": "Tongue Tapping", "back": "Tongue Tapping", "stop": "Relax",
}

# Longest-first so "turn left" matches before "left"
CHAT_COMMANDS = [
    ("turn left",  "Left Fist"),
    ("turn right", "Right Fist"),
    ("go forward", "Both Fists"),
    ("forward",    "Both Fists"),
    ("backward",   "Tongue Tapping"),
    ("reverse",    "Tongue Tapping"),
    ("walk",       "Both Fists"),
    ("left",       "Left Fist"),
    ("right",      "Right Fist"),
    ("back",       "Tongue Tapping"),
    ("stop",       "Relax"),
    ("halt",       "Relax"),
    ("go",         "Both Fists"),
]

DATA_DIR = os.path.join(project_root, "data")
MODEL_DIR = os.path.join(project_root, "models")
RESULTS_DIR = os.path.join(project_root, "results")
FONT = "Segoe UI"

EVIDENCE_CHARTS = {
    "1": ("latency_accuracy_tradeoff.png", "Latency vs Accuracy: 4 Model Comparison"),
    "2": ("scalability_100.png",           "Fleet Scalability: 10 to 100 Robots"),
    "3": ("failure_analysis.png",          "Failure Mode Analysis"),
}

# =====================================================================
#  LOAD PIPELINE MODELS (once, shared across requests)
# =====================================================================

print("Loading pipeline models...", end="", flush=True)
PIPELINE = ThoughtLinkPipeline()
PIPELINE.load_models(
    os.path.join(MODEL_DIR, "stage1_binary.pkl"),
    os.path.join(MODEL_DIR, "stage2_direction.pkl"),
)
print(" done")


# =====================================================================
#  FLEET ROBOT
# =====================================================================

class FleetRobot:
    """One robot in the 2D fleet view. Belongs to a group with a bounded region."""

    def __init__(self, robot_id, group_id):
        self.id = robot_id
        self.group_id = group_id

        # Region in percentage coords (0-100 canvas space)
        col = (group_id - 1) % 5
        row = (group_id - 1) // 5
        self.region = (
            col * 20 + 1.5,         # x_min
            (col + 1) * 20 - 1.5,   # x_max
            row * 50 + 5,           # y_min (room for group label)
            (row + 1) * 50 - 1.5,   # y_max
        )

        self.x = random.uniform(self.region[0] + 1, self.region[1] - 1)
        self.y = random.uniform(self.region[2] + 1, self.region[3] - 1)
        self.heading = random.uniform(0, 360)
        self.speed = random.uniform(0.08, 0.15)
        self.state = "autonomous"          # autonomous | stuck | override
        self.dir_timer = random.uniform(2, 4)
        self.flash_tick = 0
        self.override_timer = 0.0
        self.override_action = None
        self.failure_reason = None

    def update(self, dt):
        self.flash_tick += 1
        x_min, x_max, y_min, y_max = self.region

        if self.state == "autonomous":
            rad = math.radians(self.heading)
            self.x += self.speed * math.cos(rad) * dt * 30
            self.y += self.speed * math.sin(rad) * dt * 30
            # Bounce within group region
            if self.x < x_min or self.x > x_max:
                self.heading = 180 - self.heading
                self.x = max(x_min, min(x_max, self.x))
            if self.y < y_min or self.y > y_max:
                self.heading = -self.heading
                self.y = max(y_min, min(y_max, self.y))
            self.dir_timer -= dt
            if self.dir_timer <= 0:
                self.heading += random.uniform(-90, 90)
                self.dir_timer = random.uniform(2, 4)

        elif self.state == "override":
            if self.override_action and self.override_action != "STOP":
                dx, dy = {"LEFT": (-1, 0), "RIGHT": (1, 0),
                          "FORWARD": (0, -1), "BACKWARD": (0, 1),
                          }.get(self.override_action, (0, 0))
                self.x += dx * 0.3 * dt * 30
                self.y += dy * 0.3 * dt * 30
                self.x = max(x_min, min(x_max, self.x))
                self.y = max(y_min, min(y_max, self.y))
            self.override_timer -= dt
            if self.override_timer <= 0:
                self.state = "autonomous"
                self.override_action = None
                self.failure_reason = None
                self.dir_timer = random.uniform(2, 4)

    def get_color(self):
        if self.state == "autonomous":
            return GROUP_COLORS.get(self.group_id, "#FFFFFF")
        elif self.state == "stuck":
            return RED if self.flash_tick % 30 < 15 else "#3d1111"
        elif self.state == "override":
            if self.override_timer > 2.0 and self.flash_tick % 10 < 5:
                return "#ffffff"
            return ACTION_COLORS.get(self.override_action, BLUE)
        return GRAY


# =====================================================================
#  MUJOCO RUNNER (optional second window)
# =====================================================================

class MuJoCoRunner:
    """Wraps the BRI Controller for MuJoCo G1 humanoid."""

    def __init__(self):
        self.ctrl = None
        self.available = False
        self.action_map = {}

    def start(self):
        try:
            bri_src = os.path.join(project_root, "brain-robot-interface", "src")
            if bri_src not in sys.path:
                sys.path.insert(0, bri_src)
            from bri import Action, Controller

            bundle_dir = os.path.join(
                project_root, "brain-robot-interface", "bundles", "g1_mjlab")
            self.ctrl = Controller(
                backend="sim", hold_s=1.0,
                forward_speed=0.4, yaw_rate=1.0,
                smooth_alpha=0.3, bundle_dir=bundle_dir,
            )
            self.ctrl.start()

            # Camera setup
            viewer = self.ctrl._backend._viewer
            if viewer:
                viewer.cam.distance = 5.0
                viewer.cam.azimuth = 180.0
                viewer.cam.elevation = -25.0
                data = self.ctrl._backend._data
                if data:
                    viewer.cam.lookat[0] = float(data.qpos[0])
                    viewer.cam.lookat[1] = float(data.qpos[1])
                    viewer.cam.lookat[2] = 0.8

            self.action_map = {
                "FORWARD": Action.FORWARD, "BACKWARD": Action.BACKWARD,
                "LEFT": Action.LEFT, "RIGHT": Action.RIGHT,
                "STOP": Action.STOP,
            }
            self.available = True
            print("  MuJoCo G1 humanoid started.")
        except Exception as e:
            print(f"  MuJoCo unavailable: {e}")
            print("  Running control panel only.")

    def set_action(self, action_str):
        if not self.available:
            return
        action = self.action_map.get(action_str)
        if action is not None:
            self.ctrl.set_action(action)

    def update_camera(self):
        if not self.available:
            return
        try:
            data = self.ctrl._backend._data
            viewer = self.ctrl._backend._viewer
            if data and viewer and viewer.is_running():
                viewer.cam.lookat[0] = float(data.qpos[0])
                viewer.cam.lookat[1] = float(data.qpos[1])
                viewer.cam.lookat[2] = 0.8
            elif viewer and not viewer.is_running():
                self.available = False
        except Exception:
            pass

    def stop(self):
        if self.available:
            try:
                self.ctrl.stop()
            except Exception:
                pass


# =====================================================================
#  CONTROL PANEL (tkinter window)
# =====================================================================

class ControlPanel:
    """Tkinter window: 100-robot fleet, hierarchical overrides, metrics."""

    def __init__(self, root, mujoco_runner):
        self.root = root
        self.mujoco = mujoco_runner if mujoco_runner and mujoco_runner.available else None

        # Fleet: 100 robots in 10 groups
        self.robots = []
        for i in range(NUM_ROBOTS):
            group = i // ROBOTS_PER_GROUP + 1
            self.robots.append(FleetRobot(i, group))

        # Pipeline state
        self._pipeline_running = False
        self._pipeline_result = None
        self._decode_cmd_type = "single"
        self._decode_group = None
        self._decode_label = None
        self._decode_target = None
        self._current_mujoco_action = "FORWARD"

        # Efficiency tracking
        self.total_robots_overridden = 0
        self.total_commands_issued = 0

        # Auto-play state
        self.autoplay = False
        self._autoplay_step = 0
        self._autoplay_timer = 0.0
        self._autoplay_retry = False

        # Track display state to avoid redundant updates
        self._last_stuck_ids = frozenset()

        # Build UI
        self._build_ui()

        # Start loops
        self._last_time = time.time()
        self._update_fleet()
        self.root.after(8000, self._make_robots_stuck)

        # Send default FORWARD to MuJoCo
        if self.mujoco:
            self.mujoco.set_action("FORWARD")

        # Key bindings
        self.root.bind("<Key>", self._on_key)

    # -----------------------------------------------------------------
    #  UI BUILDING
    # -----------------------------------------------------------------

    def _build_ui(self):
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill=tk.BOTH, expand=True)

        tk.Label(main, text="ThoughtLink: 100-Robot Fleet Orchestration",
                 font=(FONT, 14, "bold"), fg=TEXT, bg=BG).pack(pady=(5, 0))
        tk.Label(main, text="Left: Close-up of active override | Right: Fleet operations",
                 font=(FONT, 9), fg=DIM, bg=BG).pack(pady=(0, 2))

        self._build_fleet(main)
        self._build_override(main)
        self._build_metrics(main)
        self._build_controls(main)
        self._build_chat(main)
        self._build_eeg(main)

    def _build_fleet(self, parent):
        """SECTION 1: Fleet overview with 100 robots in 10 groups."""
        frame = tk.Frame(parent, bg=BG2,
                         highlightbackground=BORDER, highlightthickness=1)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.fleet_counter = tk.Label(
            frame, text="Fleet: 100 | Autonomous: 100 | Stuck: 0 | Groups affected: 0",
            font=(FONT, 12, "bold"), fg=TEXT, bg=BG2)
        self.fleet_counter.pack(pady=(4, 0))

        self.canvas = tk.Canvas(frame, bg=BG2, highlightthickness=0, height=300)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Create canvas items for each robot
        self._ovals = []
        self._robot_labels = []
        for robot in self.robots:
            ov = self.canvas.create_oval(0, 0, 8, 8,
                fill=GROUP_COLORS.get(robot.group_id, "#FFF"), outline="")
            lb = self.canvas.create_text(0, 0, text=f"{robot.id:02d}",
                fill="white", font=(FONT, 7, "bold"), state="hidden")
            self._ovals.append(ov)
            self._robot_labels.append(lb)

        # Group labels (G1-G10)
        self._group_labels = []
        for g in range(1, NUM_GROUPS + 1):
            gl = self.canvas.create_text(0, 0, text=f"G{g}",
                fill=GROUP_COLORS.get(g, "#FFF"), font=(FONT, 10, "bold"))
            self._group_labels.append(gl)

        tk.Label(frame, text="~33 robots/core at 1Hz | 10 groups x 10 robots",
                 font=(FONT, 9), fg=DIM, bg=BG2).pack(pady=(0, 4))

    def _build_override(self, parent):
        """SECTION 2: Override status (supports multi-robot display)."""
        frame = tk.Frame(parent, bg=BG2,
                         highlightbackground=BORDER, highlightthickness=1,
                         height=220)
        frame.pack(fill=tk.X, padx=8, pady=4)
        frame.pack_propagate(False)

        self._ov_status = tk.Label(
            frame, text="ALL ROBOTS AUTONOMOUS",
            font=(FONT, 16, "bold"), fg="#1a4d2e", bg=BG2)
        self._ov_status.pack(pady=(8, 0))

        self._ov_action = tk.Label(
            frame, text="", font=(FONT, 28, "bold"), fg=GRAY, bg=BG2)
        self._ov_action.pack()

        self._ov_detail = tk.Label(
            frame, text="", font=("Consolas", 9), fg=DIM, bg=BG2,
            justify="left", anchor="w")
        self._ov_detail.pack(fill=tk.X, padx=15, pady=(0, 4))

        self._mujoco_label = tk.Label(
            frame, text="MuJoCo view: Fleet patrol (autonomous)",
            font=(FONT, 9), fg=DIM, bg=BG2)
        self._mujoco_label.pack(pady=(0, 6))

    def _build_metrics(self, parent):
        """SECTION 3: Live metrics + efficiency counter."""
        frame = tk.Frame(parent, bg=BG2,
                         highlightbackground=BORDER, highlightthickness=1,
                         height=140)
        frame.pack(fill=tk.X, padx=8, pady=4)
        frame.pack_propagate(False)

        inner = tk.Frame(frame, bg=BG2)
        inner.pack(fill=tk.X, padx=15, pady=8)

        # Row 1: Latency + Confidence
        r1 = tk.Frame(inner, bg=BG2)
        r1.pack(fill=tk.X)

        tk.Label(r1, text="Latency:", font=(FONT, 11), fg=DIM,
                 bg=BG2).pack(side=tk.LEFT)
        self._m_lat = tk.Label(r1, text="--", font=(FONT, 20, "bold"),
                               fg=DIM, bg=BG2)
        self._m_lat.pack(side=tk.LEFT, padx=(5, 30))

        tk.Label(r1, text="Confidence:", font=(FONT, 11), fg=DIM,
                 bg=BG2).pack(side=tk.LEFT)
        self._m_conf = tk.Label(r1, text="--", font=(FONT, 20, "bold"),
                                fg=DIM, bg=BG2)
        self._m_conf.pack(side=tk.LEFT, padx=5)

        self._conf_bar = tk.Canvas(r1, width=100, height=16, bg="#21262d",
                                    highlightthickness=0)
        self._conf_bar.pack(side=tk.LEFT, padx=5)
        self._conf_fill = self._conf_bar.create_rectangle(
            0, 0, 0, 16, fill=GREEN, outline="")

        # Row 2: Pipeline + Windows
        r2 = tk.Frame(inner, bg=BG2)
        r2.pack(fill=tk.X, pady=(5, 0))

        tk.Label(r2, text="Pipeline:", font=(FONT, 11), fg=DIM,
                 bg=BG2).pack(side=tk.LEFT)
        self._m_pipe = tk.Label(r2, text="--", font=(FONT, 11, "bold"),
                                fg=DIM, bg=BG2)
        self._m_pipe.pack(side=tk.LEFT, padx=(5, 30))

        tk.Label(r2, text="Windows:", font=(FONT, 11), fg=DIM,
                 bg=BG2).pack(side=tk.LEFT)
        self._m_win = tk.Label(r2, text="--", font=(FONT, 11, "bold"),
                               fg=DIM, bg=BG2)
        self._m_win.pack(side=tk.LEFT, padx=5)

        # Row 3: Efficiency
        r3 = tk.Frame(inner, bg=BG2)
        r3.pack(fill=tk.X, pady=(5, 0))

        self._m_eff = tk.Label(r3, text="Session: Awaiting first command...",
                               font=(FONT, 10), fg=DIM, bg=BG2)
        self._m_eff.pack(side=tk.LEFT)

    def _build_controls(self, parent):
        """SECTION 4: Brain signal buttons."""
        frame = tk.Frame(parent, bg=BG2,
                         highlightbackground=BORDER, highlightthickness=1,
                         height=85)
        frame.pack(fill=tk.X, padx=8, pady=4)
        frame.pack_propagate(False)

        btn_row = tk.Frame(frame, bg=BG2)
        btn_row.pack(expand=True)

        buttons = [
            ("LEFT FIST",  "Left Fist",      BLUE,   "white"),
            ("RIGHT FIST", "Right Fist",     RED,    "white"),
            ("BOTH FISTS", "Both Fists",     GREEN,  "white"),
            ("TONGUE TAP", "Tongue Tapping", YELLOW, "#000"),
            ("RELAX",      "Relax",          GRAY,   "white"),
        ]
        for text, label, color, fg in buttons:
            btn = tk.Button(
                btn_row, text=text, font=(FONT, 10, "bold"),
                bg=color, fg=fg, activebackground=color, activeforeground=fg,
                relief=tk.FLAT, padx=12, pady=8, cursor="hand2",
                command=lambda l=label: self._on_signal(l))
            btn.pack(side=tk.LEFT, padx=4)

    def _build_chat(self, parent):
        """SECTION 5: Chat input."""
        frame = tk.Frame(parent, bg=BG2,
                         highlightbackground=BORDER, highlightthickness=1,
                         height=75)
        frame.pack(fill=tk.X, padx=8, pady=4)
        frame.pack_propagate(False)

        row = tk.Frame(frame, bg=BG2)
        row.pack(fill=tk.X, padx=10, pady=(10, 2))

        self._chat = tk.Entry(
            row, font=(FONT, 11), bg="#21262d", fg=DIM,
            insertbackground=TEXT, relief=tk.FLAT, bd=5)
        self._chat_placeholder = True
        self._chat.insert(0, "Try: 'group 3 fix', 'group 1 left', 'fix all', or 'turn left'")
        self._chat.bind("<FocusIn>", self._on_chat_focus)
        self._chat.bind("<FocusOut>", self._on_chat_blur)
        self._chat.bind("<Return>", lambda e: self._on_chat_submit())
        self._chat.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Button(row, text="Send", font=(FONT, 10, "bold"),
                  bg=BLUE, fg="white", relief=tk.FLAT, padx=12,
                  cursor="hand2",
                  command=self._on_chat_submit).pack(side=tk.LEFT, padx=(8, 0))

        self._chat_result = tk.Label(
            frame, text="", font=(FONT, 9), fg=DIM, bg=BG2, anchor="w")
        self._chat_result.pack(fill=tk.X, padx=10)

    def _build_eeg(self, parent):
        """SECTION 6: EEG preview + key hints."""
        frame = tk.Frame(parent, bg=BG2,
                         highlightbackground=BORDER, highlightthickness=1,
                         height=130)
        frame.pack(fill=tk.X, padx=8, pady=(4, 8))
        frame.pack_propagate(False)

        self._eeg_fig = Figure(figsize=(6, 1.0), dpi=85)
        self._eeg_fig.patch.set_facecolor(BG2)
        self._eeg_ax = self._eeg_fig.add_subplot(111)
        self._eeg_ax.set_facecolor(BG2)
        self._eeg_ax.set_xticks([])
        self._eeg_ax.set_yticks([])
        for spine in self._eeg_ax.spines.values():
            spine.set_color(BORDER)

        self._eeg_canvas = FigureCanvasTkAgg(self._eeg_fig, master=frame)
        self._eeg_canvas.get_tk_widget().pack(fill=tk.X, padx=10, pady=(5, 0))
        self._eeg_canvas.draw()

        tk.Label(frame,
                 text="Keys: A=Auto-play | Q=Quit | 1=Latency | 2=Scalability | 3=Failures",
                 font=(FONT, 9), fg=DIM, bg=BG2).pack(pady=(2, 5))

    # -----------------------------------------------------------------
    #  FLEET ANIMATION (30 fps)
    # -----------------------------------------------------------------

    def _update_fleet(self):
        now = time.time()
        dt = min(now - self._last_time, 0.1)
        self._last_time = now

        # Update robot positions
        for robot in self.robots:
            robot.update(dt)

        # Canvas dimensions
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            self.root.after(33, self._update_fleet)
            return

        # Redraw robots
        auto_n = 0
        stuck_n = 0
        stuck_groups = set()
        current_stuck_ids = set()

        for i, robot in enumerate(self.robots):
            cx = robot.x / 100 * cw
            cy = robot.y / 100 * ch
            r = 5 if robot.state == "stuck" else 4
            self.canvas.coords(self._ovals[i], cx - r, cy - r, cx + r, cy + r)
            self.canvas.itemconfig(self._ovals[i], fill=robot.get_color())

            # Show label only for stuck robots
            if robot.state == "stuck":
                self.canvas.coords(self._robot_labels[i], cx + 10, cy)
                self.canvas.itemconfig(self._robot_labels[i], state="normal")
                stuck_n += 1
                stuck_groups.add(robot.group_id)
                current_stuck_ids.add(robot.id)
            else:
                self.canvas.itemconfig(self._robot_labels[i], state="hidden")
                if robot.state == "autonomous":
                    auto_n += 1

        # Group labels
        for g_idx in range(NUM_GROUPS):
            g = g_idx + 1
            col = (g - 1) % 5
            row = (g - 1) // 5
            gx = (col * 20 + (col + 1) * 20) / 2 / 100 * cw
            gy = (row * 50 + 3) / 100 * ch
            self.canvas.coords(self._group_labels[g_idx], gx, gy)

        # Fleet counter
        self.fleet_counter.config(
            text=f"Fleet: {NUM_ROBOTS} | Autonomous: {auto_n} | "
                 f"Stuck: {stuck_n} | Groups affected: {len(stuck_groups)}")

        # Update override display when state changes
        current_stuck_frozen = frozenset(current_stuck_ids)
        any_override = any(r.state == "override" for r in self.robots)
        if not any_override and current_stuck_frozen != self._last_stuck_ids:
            self._last_stuck_ids = current_stuck_frozen
            if stuck_n > 0:
                self._update_stuck_display()
            else:
                self._show_all_autonomous()

        # MuJoCo: re-send action + track camera
        if self.mujoco:
            self.mujoco.set_action(self._current_mujoco_action)
            self.mujoco.update_camera()

        # Auto-play
        if self.autoplay and not self._pipeline_running:
            self._autoplay_tick(dt)

        self.root.after(33, self._update_fleet)

    def _make_robots_stuck(self):
        """Every 8 seconds, make 3-6 robots stuck across 2-3 groups."""
        n_groups = random.randint(2, 3)
        groups = random.sample(range(1, NUM_GROUPS + 1), min(n_groups, NUM_GROUPS))

        new_stuck = 0
        for g in groups:
            autonomous = [r for r in self.robots
                          if r.group_id == g and r.state == "autonomous"]
            if not autonomous:
                continue
            n = random.randint(1, min(3, len(autonomous)))
            for robot in random.sample(autonomous, n):
                robot.state = "stuck"
                robot.flash_tick = 0
                robot.failure_reason = random.choice(FAILURE_REASONS)
                new_stuck += 1

        if new_stuck > 0:
            any_override = any(r.state == "override" for r in self.robots)
            if not any_override:
                self._update_stuck_display()

        self.root.after(8000, self._make_robots_stuck)

    def _update_stuck_display(self):
        """Update override status to show currently stuck robots."""
        stuck = [r for r in self.robots if r.state == "stuck"]
        if not stuck:
            self._show_all_autonomous()
            return

        by_group = {}
        for r in stuck:
            by_group.setdefault(r.group_id, []).append(r)

        parts = [f"{len(rs)} in G{g}" for g, rs in sorted(by_group.items())]
        self._ov_status.config(text=f"STUCK: {', '.join(parts)}", fg=RED)
        self._ov_action.config(text="", fg=GRAY)

        lines = []
        for r in stuck[:8]:
            lines.append(f"R{r.id:02d}: {r.failure_reason}")
        if len(stuck) > 8:
            lines.append(f"...and {len(stuck) - 8} more")
        sep = "  |  " if len(stuck) <= 4 else "\n"
        self._ov_detail.config(text=sep.join(lines), fg=DIM)

    def _show_all_autonomous(self):
        self._ov_status.config(text="ALL ROBOTS AUTONOMOUS", fg="#1a4d2e")
        self._ov_action.config(text="", fg=GRAY)
        self._ov_detail.config(text="")
        self._mujoco_label.config(
            text="MuJoCo view: Fleet patrol (autonomous)", fg=DIM)

    # -----------------------------------------------------------------
    #  EVENT HANDLERS
    # -----------------------------------------------------------------

    def _on_key(self, event):
        if isinstance(self.root.focus_get(), tk.Entry):
            return
        ch = event.char.lower() if event.char else ""
        if ch == "a":
            self._toggle_autoplay()
        elif ch == "q":
            self._quit()
        elif ch in EVIDENCE_CHARTS:
            self._show_evidence_chart(ch)

    def _show_evidence_chart(self, key):
        fname, title = EVIDENCE_CHARTS[key]
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path):
            print(f"  Chart not found: {path}")
            return
        img = plt.imread(path)
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(img)
        ax.set_axis_off()
        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.tight_layout()
        plt.show(block=False)

    def _on_signal(self, label):
        """Brain signal button -> single robot override."""
        if self._pipeline_running:
            return
        self._run_decode(label, cmd_type="single")

    def _on_chat_focus(self, event):
        if self._chat_placeholder:
            self._chat.delete(0, tk.END)
            self._chat.config(fg=TEXT)
            self._chat_placeholder = False

    def _on_chat_blur(self, event):
        if not self._chat.get().strip():
            self._chat.delete(0, tk.END)
            self._chat.insert(0,
                "Try: 'group 3 fix', 'group 1 left', 'fix all', or 'turn left'")
            self._chat.config(fg=DIM)
            self._chat_placeholder = True

    def _on_chat_submit(self):
        if self._pipeline_running:
            return
        text = self._chat.get().strip()
        if not text or self._chat_placeholder:
            return

        text_lower = text.lower()
        self._chat.delete(0, tk.END)
        self._chat_placeholder = False

        cmd_type, group, param = self._parse_chat_command(text_lower)

        if cmd_type == "unknown":
            self._chat_result.config(
                text="Unknown command. Try: 'group 3 fix', 'group 1 left', 'fix all', or 'turn left'",
                fg=RED)
            return

        if cmd_type == "fix_all":
            stuck = [r for r in self.robots if r.state == "stuck"]
            if not stuck:
                self._chat_result.config(text="No stuck robots to fix.", fg=YELLOW)
                return
            label = random.choice(list(DEMO_FILES.keys()))
            self._run_decode(label, cmd_type="fix_all")

        elif cmd_type == "group_fix":
            if group < 1 or group > NUM_GROUPS:
                self._chat_result.config(
                    text="Invalid group. Groups are 1-10.", fg=RED)
                return
            stuck = [r for r in self.robots
                     if r.group_id == group and r.state == "stuck"]
            if not stuck:
                self._chat_result.config(
                    text=f"No stuck robots in Group {group}.", fg=YELLOW)
                return
            label = random.choice(list(DEMO_FILES.keys()))
            self._run_decode(label, cmd_type="group_fix", group=group)

        elif cmd_type == "group_direction":
            if group < 1 or group > NUM_GROUPS:
                self._chat_result.config(
                    text="Invalid group. Groups are 1-10.", fg=RED)
                return
            stuck = [r for r in self.robots
                     if r.group_id == group and r.state == "stuck"]
            if not stuck:
                self._chat_result.config(
                    text=f"No stuck robots in Group {group}.", fg=YELLOW)
                return
            label = CHAT_DIRECTION_MAP.get(param)
            if not label:
                self._chat_result.config(
                    text=f"Unknown direction: {param}", fg=RED)
                return
            self._run_decode(label, cmd_type="group_direction", group=group)

        elif cmd_type == "single":
            self._run_decode(param, cmd_type="single")

    def _parse_chat_command(self, text):
        """Parse chat input. Returns (cmd_type, group, param)."""
        # Fix all
        if re.match(r"fix\s+all|fix\s+everything|override\s+all", text):
            return ("fix_all", None, None)

        # Group fix: "group 3 fix" or "fix group 3" or "group 3 resolve"
        m = re.match(
            r"(?:group\s*(\d+)\s+(?:fix|resolve)|(?:fix|resolve)\s+group\s*(\d+))",
            text)
        if m:
            group = int(m.group(1) or m.group(2))
            return ("group_fix", group, None)

        # Group direction: "group 3 left"
        m = re.match(
            r"group\s*(\d+)\s+(left|right|forward|backward|back|stop)", text)
        if m:
            return ("group_direction", int(m.group(1)), m.group(2))

        # Simple direction (existing behavior)
        for cmd, lbl in CHAT_COMMANDS:
            if cmd in text:
                return ("single", None, lbl)

        return ("unknown", None, None)

    # -----------------------------------------------------------------
    #  PIPELINE DECODE
    # -----------------------------------------------------------------

    def _run_decode(self, label, cmd_type="single", group=None):
        """Start pipeline decode in a background thread."""
        self._pipeline_running = True
        self._pipeline_result = None
        self._decode_cmd_type = cmd_type
        self._decode_group = group
        self._decode_label = label

        if cmd_type == "single":
            stuck = [r for r in self.robots if r.state == "stuck"]
            self._decode_target = stuck[0] if stuck else random.choice(self.robots)

        # Show decoding status
        if cmd_type == "single":
            self._ov_status.config(text=f"DECODING: {label}...", fg=YELLOW)
        elif cmd_type == "group_direction":
            self._ov_status.config(
                text=f"DECODING: {label} for Group {group}...", fg=YELLOW)
        elif cmd_type == "group_fix":
            self._ov_status.config(
                text=f"DECODING: trigger for Group {group} fix...", fg=YELLOW)
        elif cmd_type == "fix_all":
            self._ov_status.config(
                text="DECODING: trigger for fleet-wide fix...", fg=YELLOW)
        self._ov_action.config(text="...", fg=YELLOW)
        self._ov_detail.config(text="")

        thread = threading.Thread(
            target=self._pipeline_worker, args=(label,), daemon=True)
        thread.start()
        self._poll_decode(thread)

    def _pipeline_worker(self, label):
        """Run the real pipeline in a background thread."""
        try:
            fname = DEMO_FILES.get(label)
            if not fname:
                return
            npz_path = os.path.join(DATA_DIR, fname)
            if not os.path.exists(npz_path):
                return

            pipe = ThoughtLinkPipeline()
            pipe.stage1_model = PIPELINE.stage1_model
            pipe.stage2_model = PIPELINE.stage2_model
            pipe.DIRECTION_TO_ACTION = PIPELINE.DIRECTION_TO_ACTION

            arr = np.load(npz_path, allow_pickle=True)
            eeg_raw = arr["feature_eeg"]

            results = []
            for item in pipe.process_file(npz_path):
                if len(item) == 4:
                    results.append(item)
                else:
                    results.append((*item, "N/A"))

            if not results:
                return

            actions = [r[0] for r in results]
            dominant = Counter(actions).most_common(1)[0][0]
            metrics = pipe.get_metrics()
            last_phase = results[-1][3]
            s1_status = "ACTIVE" if dominant != "STOP" else "REST"

            self._pipeline_result = {
                "action": dominant,
                "label": label,
                "latency": metrics["avg_latency_ms"],
                "confidence": metrics["avg_confidence"],
                "phase": last_phase,
                "n_windows": len(results),
                "s1_status": s1_status,
                "eeg_raw": eeg_raw,
            }
        except Exception as e:
            print(f"  Pipeline error: {e}")
            self._pipeline_result = None

    def _poll_decode(self, thread):
        """Poll for pipeline completion, then dispatch result."""
        if thread.is_alive():
            self.root.after(50, lambda: self._poll_decode(thread))
            return

        self._pipeline_running = False
        result = self._pipeline_result

        if result is None:
            self._ov_status.config(text="DECODE FAILED", fg=RED)
            self._ov_action.config(text="", fg=GRAY)
            return

        cmd_type = self._decode_cmd_type
        group = self._decode_group

        if cmd_type == "single":
            self._apply_single_override(result)
        elif cmd_type == "group_direction":
            self._apply_group_direction(result, group)
        elif cmd_type == "group_fix":
            self._apply_group_fix(result, group)
        elif cmd_type == "fix_all":
            self._apply_fix_all(result)

    # -----------------------------------------------------------------
    #  OVERRIDE APPLICATION
    # -----------------------------------------------------------------

    def _apply_single_override(self, result):
        """Layer 1: Override a single stuck robot."""
        target = self._decode_target
        action = result["action"]
        label = result["label"]
        lat = result["latency"]
        conf = result["confidence"]
        color = ACTION_COLORS.get(action, GRAY)

        self._ov_status.config(
            text=f"OVERRIDE SENT: {action} \u2192 Robot #{target.id:02d}", fg=GREEN)
        self._ov_action.config(text=action, fg=color)
        self._ov_detail.config(
            text=f"Decoded: {label} EEG ({lat:.0f}ms, conf {conf:.2f})",
            fg=TEXT)

        target.state = "override"
        target.override_action = action
        target.override_timer = 3.0
        target.flash_tick = 0

        self.total_robots_overridden += 1
        self.total_commands_issued += 1
        self._update_efficiency()
        self._update_metrics(result)
        self._draw_eeg(result["eeg_raw"])

        if self.mujoco:
            self.mujoco.set_action(action)
            self._current_mujoco_action = action

        self._mujoco_label.config(
            text=f"MuJoCo view: Robot #{target.id:02d} (close-up)", fg=TEXT)

        self._chat_result.config(
            text=f"Decoded: {label} \u2192 {action} ({lat:.0f}ms) "
                 f"\u2192 Robot #{target.id:02d}",
            fg=GREEN)

    def _apply_group_direction(self, result, group):
        """Layer 2: Send same decoded action to all stuck robots in a group."""
        stuck = [r for r in self.robots
                 if r.group_id == group and r.state == "stuck"]
        if not stuck:
            self._ov_status.config(
                text=f"No stuck robots in Group {group}", fg=YELLOW)
            return

        action = result["action"]
        label = result["label"]
        lat = result["latency"]
        color = ACTION_COLORS.get(action, GRAY)

        ids = ", ".join(f"R{r.id:02d}" for r in stuck)
        self._ov_status.config(
            text=f"GROUP {group} OVERRIDE: {action} \u2192 "
                 f"{len(stuck)} robots ({ids})",
            fg=GREEN)
        self._ov_action.config(text=action, fg=color)
        self._ov_detail.config(
            text=f"Decoded: {label} EEG ({lat:.0f}ms) | Same action to all",
            fg=TEXT)

        for robot in stuck:
            robot.state = "override"
            robot.override_action = action
            robot.override_timer = 3.0
            robot.flash_tick = 0

        self.total_robots_overridden += len(stuck)
        self.total_commands_issued += 1
        self._update_efficiency()
        self._update_metrics(result)
        self._draw_eeg(result["eeg_raw"])

        if self.mujoco:
            self.mujoco.set_action(action)
            self._current_mujoco_action = action

        self._mujoco_label.config(
            text=f"MuJoCo view: Robot #{stuck[-1].id:02d} (close-up)", fg=TEXT)

        self._chat_result.config(
            text=f"Group {group}: {label} \u2192 {action} "
                 f"\u2192 {len(stuck)} robots",
            fg=GREEN)

    def _apply_group_fix(self, result, group):
        """Layer 3: Individualized actions based on each robot's failure reason."""
        stuck = [r for r in self.robots
                 if r.group_id == group and r.state == "stuck"]
        if not stuck:
            self._ov_status.config(
                text=f"No stuck robots in Group {group}", fg=YELLOW)
            return

        label = result["label"]
        lat = result["latency"]

        self._ov_status.config(
            text=f"GROUP {group} OVERRIDE ({len(stuck)} robots) "
                 f"\u2014 Context-Aware:",
            fg=GREEN)

        lines = []
        last_action = "STOP"
        for r in stuck[:8]:
            a = FAILURE_ACTIONS.get(r.failure_reason, "STOP")
            lines.append(f"  R{r.id:02d}: {r.failure_reason} \u2192 {a}")
            last_action = a
        if len(stuck) > 8:
            lines.append(f"  ...and {len(stuck) - 8} more")
        lines.append(
            f"  Triggered by: {label} EEG ({lat:.0f}ms) | "
            f"1 command \u2192 {len(stuck)} individualized actions")

        self._ov_action.config(text="CONTEXT-AWARE", fg=GREEN)
        self._ov_detail.config(text="\n".join(lines), fg=TEXT)

        for robot in stuck:
            a = FAILURE_ACTIONS.get(robot.failure_reason, "STOP")
            robot.state = "override"
            robot.override_action = a
            robot.override_timer = 3.0
            robot.flash_tick = 0
            last_action = a

        self.total_robots_overridden += len(stuck)
        self.total_commands_issued += 1
        self._update_efficiency()
        self._update_metrics(result)
        self._draw_eeg(result["eeg_raw"])

        if self.mujoco:
            self.mujoco.set_action(last_action)
            self._current_mujoco_action = last_action

        self._mujoco_label.config(
            text=f"MuJoCo view: Robot #{stuck[-1].id:02d} (close-up)", fg=TEXT)

        self._chat_result.config(
            text=f"Group {group} fix: 1 decode \u2192 "
                 f"{len(stuck)} individualized actions",
            fg=GREEN)

    def _apply_fix_all(self, result):
        """Layer 3+: Individualized actions for ALL stuck robots."""
        stuck = [r for r in self.robots if r.state == "stuck"]
        if not stuck:
            self._ov_status.config(text="No stuck robots to fix.", fg=YELLOW)
            return

        label = result["label"]
        lat = result["latency"]

        by_group = {}
        for r in stuck:
            by_group.setdefault(r.group_id, []).append(r)

        self._ov_status.config(
            text=f"FLEET-WIDE OVERRIDE ({len(stuck)} robots) "
                 f"\u2014 Context-Aware:",
            fg=GREEN)

        lines = []
        count = 0
        last_action = "STOP"
        for g in sorted(by_group.keys()):
            for r in by_group[g]:
                if count < 8:
                    a = FAILURE_ACTIONS.get(r.failure_reason, "STOP")
                    lines.append(
                        f"  R{r.id:02d} (G{g}): {r.failure_reason} \u2192 {a}")
                count += 1
        if len(stuck) > 8:
            lines.append(f"  ...and {len(stuck) - 8} more")
        lines.append(
            f"  Triggered by: {label} EEG ({lat:.0f}ms) | "
            f"1 command \u2192 {len(stuck)} individualized actions")

        self._ov_action.config(text="FLEET FIX", fg=GREEN)
        self._ov_detail.config(text="\n".join(lines), fg=TEXT)

        for robot in stuck:
            a = FAILURE_ACTIONS.get(robot.failure_reason, "STOP")
            robot.state = "override"
            robot.override_action = a
            robot.override_timer = 3.0
            robot.flash_tick = 0
            last_action = a

        self.total_robots_overridden += len(stuck)
        self.total_commands_issued += 1
        self._update_efficiency()
        self._update_metrics(result)
        self._draw_eeg(result["eeg_raw"])

        if self.mujoco:
            self.mujoco.set_action(last_action)
            self._current_mujoco_action = last_action

        self._mujoco_label.config(
            text=f"MuJoCo view: Robot #{stuck[-1].id:02d} (close-up)", fg=TEXT)

        self._chat_result.config(
            text=f"Fleet fix: 1 decode \u2192 "
                 f"{len(stuck)} individualized actions",
            fg=GREEN)

    # -----------------------------------------------------------------
    #  HELPERS
    # -----------------------------------------------------------------

    def _update_metrics(self, result):
        lat = result["latency"]
        conf = result["confidence"]
        action = result["action"]

        lat_color = GREEN if lat < 50 else YELLOW if lat < 100 else RED
        self._m_lat.config(text=f"{lat:.0f}ms", fg=lat_color)

        conf_color = GREEN if conf > 0.5 else YELLOW
        self._m_conf.config(text=f"{conf:.2f}", fg=conf_color)
        bar_w = max(1, int(conf * 100))
        self._conf_bar.coords(self._conf_fill, 0, 0, bar_w, 16)
        self._conf_bar.itemconfig(self._conf_fill, fill=conf_color)

        self._m_pipe.config(
            text=f"Stage 1: {result['s1_status']} \u2192 Stage 2: {action}",
            fg=TEXT)
        self._m_win.config(
            text=f"{result['n_windows']}/{result['n_windows']}", fg=TEXT)

    def _update_efficiency(self):
        if self.total_commands_issued == 0:
            self._m_eff.config(
                text="Session: Awaiting first command...", fg=DIM)
            return
        ratio = self.total_robots_overridden / self.total_commands_issued
        if self.total_robots_overridden > 0:
            reduction = ((self.total_robots_overridden - self.total_commands_issued)
                         / self.total_robots_overridden) * 100
        else:
            reduction = 0
        self._m_eff.config(
            text=f"Session: {self.total_robots_overridden} robots fixed with "
                 f"{self.total_commands_issued} commands | "
                 f"Efficiency: {ratio:.1f}x leverage "
                 f"({reduction:.0f}% fewer commands)",
            fg=GREEN if ratio > 1 else DIM)

    def _draw_eeg(self, eeg_raw):
        ax = self._eeg_ax
        ax.clear()
        ax.set_facecolor(BG2)

        fs = 500
        center = eeg_raw.shape[0] // 2
        start = max(0, center - fs)
        end = min(eeg_raw.shape[0], center + fs)
        segment = eeg_raw[start:end]

        t = np.arange(len(segment)) / fs
        n_ch = segment.shape[1]
        for ch in range(n_ch):
            sig = segment[:, ch]
            sig_norm = (sig - sig.mean()) / (sig.std() + 1e-8) * 0.5
            offset = (n_ch - 1 - ch) * 1.2
            ax.plot(t, sig_norm + offset, color=CHANNEL_COLORS[ch],
                    linewidth=0.6, alpha=0.9)

        ax.set_xlim(0, t[-1] if len(t) > 0 else 2)
        ax.set_ylim(-1, n_ch * 1.2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(BORDER)

        self._eeg_fig.tight_layout(pad=0.3)
        self._eeg_canvas.draw()

    # -----------------------------------------------------------------
    #  AUTO-PLAY
    # -----------------------------------------------------------------

    def _toggle_autoplay(self):
        self.autoplay = not self.autoplay
        if self.autoplay:
            self._autoplay_step = 0
            self._autoplay_timer = 3.0
            self._autoplay_retry = False
            self.root.title(
                "ThoughtLink | AUTO-PLAY ON | Press A to stop, Q to quit")
        else:
            self.root.title(
                "ThoughtLink: 100-Robot Fleet | A=auto-play  Q=quit")

    def _autoplay_tick(self, dt):
        """Called each frame during auto-play when pipeline is idle."""
        self._autoplay_timer -= dt
        if self._autoplay_timer > 0:
            return

        if self._autoplay_step >= 5:
            # Auto-play complete
            self.autoplay = False
            if self.total_commands_issued > 0:
                ratio = self.total_robots_overridden / self.total_commands_issued
                self._chat_result.config(
                    text=f"Auto-play complete: "
                         f"{self.total_robots_overridden} robots fixed with "
                         f"{self.total_commands_issued} commands "
                         f"({ratio:.1f}x leverage)",
                    fg=GREEN)
            self.root.title(
                "ThoughtLink: 100-Robot Fleet | A=auto-play  Q=quit")
            return

        # Ensure stuck robots exist
        stuck = [r for r in self.robots if r.state == "stuck"]
        if not stuck:
            if not self._autoplay_retry:
                self._autoplay_retry = True
                self._force_robots_stuck()
                self._autoplay_timer = 2.0
                return
            else:
                self._force_robots_stuck()
                self._autoplay_retry = False

        step = self._autoplay_step

        if step == 0:   # Layer 3: group fix
            g = self._find_group_with_stuck()
            if g:
                label = random.choice(list(DEMO_FILES.keys()))
                self._run_decode(label, cmd_type="group_fix", group=g)
        elif step == 1: # Layer 2: group direction
            g = self._find_group_with_stuck()
            if g:
                self._run_decode("Left Fist", cmd_type="group_direction",
                                 group=g)
        elif step == 2: # Layer 1: single button
            self._run_decode("Right Fist", cmd_type="single")
        elif step == 3: # Layer 3 again
            g = self._find_group_with_stuck()
            if g:
                label = random.choice(list(DEMO_FILES.keys()))
                self._run_decode(label, cmd_type="group_fix", group=g)
        elif step == 4: # Layer 3 fleet-wide
            label = random.choice(list(DEMO_FILES.keys()))
            self._run_decode(label, cmd_type="fix_all")

        self._autoplay_step += 1
        self._autoplay_retry = False
        self._autoplay_timer = 5.0

    def _find_group_with_stuck(self):
        groups = list(set(r.group_id for r in self.robots
                         if r.state == "stuck"))
        return random.choice(groups) if groups else None

    def _force_robots_stuck(self):
        """Force 3-5 random robots to get stuck (for auto-play)."""
        autonomous = [r for r in self.robots if r.state == "autonomous"]
        n = min(random.randint(3, 5), len(autonomous))
        if n == 0:
            return
        for robot in random.sample(autonomous, n):
            robot.state = "stuck"
            robot.flash_tick = 0
            robot.failure_reason = random.choice(FAILURE_REASONS)
        self._update_stuck_display()

    # -----------------------------------------------------------------
    #  QUIT
    # -----------------------------------------------------------------

    def _quit(self):
        self.autoplay = False
        if self.mujoco:
            self.mujoco.stop()
        self.root.destroy()


# =====================================================================
#  MAIN
# =====================================================================

def main():
    print("=" * 60)
    print("  ThoughtLink: Multi-Robot Orchestration Demo")
    print("  Hack Nation 2026 | Challenge 9")
    print("=" * 60)
    print()

    # Start MuJoCo (optional -- opens its own window)
    mujoco = MuJoCoRunner()
    mujoco.start()

    # Create tkinter window on right half of screen
    root = tk.Tk()
    root.title("ThoughtLink: 100-Robot Fleet | A=auto-play  Q=quit")
    root.configure(bg=BG)

    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    win_w = screen_w // 2
    win_h = screen_h - 80
    root.geometry(f"{win_w}x{win_h}+{screen_w // 2}+0")
    root.minsize(700, 850)

    panel = ControlPanel(root, mujoco)

    def on_close():
        panel._quit()

    root.protocol("WM_DELETE_WINDOW", on_close)
    print("DONE")
    root.mainloop()


if __name__ == "__main__":
    main()
