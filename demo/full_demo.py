"""
ThoughtLink: Multi-Robot Orchestration Demo
=============================================
10 robots operate autonomously. One gets stuck. A human operator
decodes a brain signal to override that robot. The robot resumes.
This is "one-to-many supervision" and "intent-level control."

Window 1 (left):  MuJoCo G1 humanoid — the overridden robot in 3D
Window 2 (right): Tkinter control panel — fleet view, metrics, controls

Controls:
  Click a signal button or type a command to override the stuck robot.
  Press A for auto-play (cycles all 5 signals, 5s each).
  Press Q to quit.

Usage: python demo/full_demo.py
"""
import sys
import os
import time
import random
import math
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
    """One robot in the 2D fleet view."""

    def __init__(self, robot_id, x, y):
        self.id = robot_id
        self.x = x
        self.y = y
        self.heading = random.uniform(0, 360)
        self.speed = random.uniform(0.15, 0.25)
        self.state = "autonomous"          # autonomous | stuck | override
        self.dir_timer = random.uniform(3, 5)
        self.flash_tick = 0
        self.override_timer = 0.0
        self.override_action = None

    def update(self, dt):
        self.flash_tick += 1

        if self.state == "autonomous":
            rad = math.radians(self.heading)
            self.x += self.speed * math.cos(rad) * dt * 30
            self.y += self.speed * math.sin(rad) * dt * 30
            # Bounce off boundaries
            if self.x < 5 or self.x > 95:
                self.heading = 180 - self.heading
                self.x = max(5, min(95, self.x))
            if self.y < 5 or self.y > 95:
                self.heading = -self.heading
                self.y = max(5, min(95, self.y))
            # Random direction changes
            self.dir_timer -= dt
            if self.dir_timer <= 0:
                self.heading += random.uniform(-90, 90)
                self.dir_timer = random.uniform(3, 5)

        elif self.state == "override":
            # Move in the override direction
            if self.override_action and self.override_action != "STOP":
                dx, dy = {"LEFT": (-1, 0), "RIGHT": (1, 0),
                          "FORWARD": (0, -1), "BACKWARD": (0, 1),
                          }.get(self.override_action, (0, 0))
                self.x += dx * 0.5 * dt * 30
                self.y += dy * 0.5 * dt * 30
                self.x = max(5, min(95, self.x))
                self.y = max(5, min(95, self.y))
            self.override_timer -= dt
            if self.override_timer <= 0:
                self.state = "autonomous"
                self.override_action = None
                self.dir_timer = random.uniform(3, 5)

        # stuck: don't move (handled by get_color flash)

    def get_color(self):
        if self.state == "autonomous":
            return GREEN
        elif self.state == "stuck":
            return RED if self.flash_tick % 30 < 15 else "#3d1111"
        elif self.state == "override":
            # White flash for the first second, then action color
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
    """Right-half tkinter window: fleet view, override, metrics, controls."""

    def __init__(self, root, mujoco_runner):
        self.root = root
        self.mujoco = mujoco_runner if mujoco_runner and mujoco_runner.available else None

        # Fleet state
        self.robots = []
        for i in range(10):
            angle = i * 2 * math.pi / 10
            x = 50 + 30 * math.cos(angle)
            y = 50 + 22 * math.sin(angle)
            self.robots.append(FleetRobot(i + 1, x, y))

        self.stuck_robot = None
        self._override_target = None
        self._current_mujoco_action = "FORWARD"

        # Pipeline state
        self._pipeline_running = False
        self._pipeline_result = None

        # Auto-play state
        self.autoplay = False
        self._autoplay_idx = 0
        self._autoplay_timer = 0.0

        # Build UI
        self._build_ui()

        # Start loops
        self._last_time = time.time()
        self._update_fleet()
        self.root.after(8000, self._make_robot_stuck)

        # Send default FORWARD to MuJoCo
        if self.mujoco:
            self.mujoco.set_action("FORWARD")

        # Key bindings (only when not typing in chat)
        self.root.bind("<Key>", self._on_key)

    # -----------------------------------------------------------------
    #  UI BUILDING
    # -----------------------------------------------------------------

    def _build_ui(self):
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill=tk.BOTH, expand=True)
        self._main = main

        # Title
        tk.Label(main, text="ThoughtLink: Multi-Robot Orchestration",
                 font=(FONT, 14, "bold"), fg=TEXT, bg=BG).pack(pady=(5, 2))

        self._build_fleet(main)
        self._build_override(main)
        self._build_metrics(main)
        self._build_controls(main)
        self._build_chat(main)
        self._build_eeg(main)

    def _build_fleet(self, parent):
        """SECTION 1: Fleet overview canvas."""
        frame = tk.Frame(parent, bg=BG2,
                         highlightbackground=BORDER, highlightthickness=1)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.fleet_counter = tk.Label(
            frame, text="Fleet: 10 | Autonomous: 10 | Override: 0",
            font=(FONT, 14, "bold"), fg=TEXT, bg=BG2)
        self.fleet_counter.pack(pady=(4, 0))

        self.canvas = tk.Canvas(frame, bg=BG2, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Create canvas items for each robot (oval + number)
        self._ovals = []
        self._labels = []
        for robot in self.robots:
            ov = self.canvas.create_oval(0, 0, 12, 12, fill=GREEN, outline="")
            lb = self.canvas.create_text(0, 0, text=str(robot.id),
                                         fill="white", font=(FONT, 7, "bold"))
            self._ovals.append(ov)
            self._labels.append(lb)

        # Stuck indicator items (hidden initially)
        self._stuck_line = self.canvas.create_line(
            0, 0, 0, 0, fill=RED, width=2, dash=(4, 2), state="hidden")
        self._stuck_text = self.canvas.create_text(
            0, 0, text="AWAITING OVERRIDE",
            fill=RED, font=(FONT, 10, "bold"), state="hidden")

        tk.Label(frame, text="38 robots/core at 1Hz | Demo: 10 robots",
                 font=(FONT, 9), fg=DIM, bg=BG2).pack(pady=(0, 4))

    def _build_override(self, parent):
        """SECTION 2: Current override status."""
        frame = tk.Frame(parent, bg=BG2,
                         highlightbackground=BORDER, highlightthickness=1,
                         height=170)
        frame.pack(fill=tk.X, padx=8, pady=4)
        frame.pack_propagate(False)

        self._ov_status = tk.Label(
            frame, text="ALL ROBOTS AUTONOMOUS",
            font=(FONT, 22, "bold"), fg="#1a4d2e", bg=BG2)
        self._ov_status.pack(pady=(10, 0))

        self._ov_action = tk.Label(
            frame, text="", font=(FONT, 48, "bold"), fg=GRAY, bg=BG2)
        self._ov_action.pack()

        self._ov_detail = tk.Label(
            frame, text="", font=(FONT, 11), fg=DIM, bg=BG2)
        self._ov_detail.pack(pady=(0, 8))

    def _build_metrics(self, parent):
        """SECTION 3: Live metrics."""
        frame = tk.Frame(parent, bg=BG2,
                         highlightbackground=BORDER, highlightthickness=1,
                         height=115)
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
        self._chat.insert(0, "Type command... (e.g. 'turn left', 'go forward', 'stop')")
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
        """SECTION 6: EEG preview + info."""
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
                 text="Keys: A=Auto-play | Q=Quit | 1=Latency Tradeoffs | 2=Scalability | 3=Failure Analysis",
                 font=(FONT, 9), fg=DIM, bg=BG2).pack(pady=(2, 5))

    # -----------------------------------------------------------------
    #  FLEET ANIMATION (30 fps)
    # -----------------------------------------------------------------

    def _update_fleet(self):
        now = time.time()
        dt = min(now - self._last_time, 0.1)  # cap to avoid jumps
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

        # Redraw robot positions and colors
        auto_n = 0
        ovr_n = 0
        for i, robot in enumerate(self.robots):
            cx = robot.x / 100 * cw
            cy = robot.y / 100 * ch
            r = 6
            self.canvas.coords(self._ovals[i], cx - r, cy - r, cx + r, cy + r)
            self.canvas.coords(self._labels[i], cx, cy)
            self.canvas.itemconfig(self._ovals[i], fill=robot.get_color())
            if robot.state == "autonomous":
                auto_n += 1
            else:
                ovr_n += 1

        # Stuck indicator line + text
        if self.stuck_robot and self.stuck_robot.state == "stuck":
            sx = self.stuck_robot.x / 100 * cw
            sy = self.stuck_robot.y / 100 * ch
            tx, ty = cw * 0.75, 18
            self.canvas.coords(self._stuck_line, sx, sy, tx, ty)
            self.canvas.coords(self._stuck_text, tx, ty)
            self.canvas.itemconfig(self._stuck_line, state="normal")
            self.canvas.itemconfig(self._stuck_text, state="normal")
        else:
            self.canvas.itemconfig(self._stuck_line, state="hidden")
            self.canvas.itemconfig(self._stuck_text, state="hidden")

        # Counter
        self.fleet_counter.config(
            text=f"Fleet: 10 | Autonomous: {auto_n} | Override: {ovr_n}")

        # Check if override period ended → clear display
        if self._override_target and self._override_target.state == "autonomous":
            self._show_all_autonomous()
            self._override_target = None
            if self.mujoco:
                self.mujoco.set_action("FORWARD")
                self._current_mujoco_action = "FORWARD"

        # MuJoCo: re-send action + track camera
        if self.mujoco:
            self.mujoco.set_action(self._current_mujoco_action)
            self.mujoco.update_camera()

        # Auto-play timer
        if self.autoplay and not self._pipeline_running:
            self._autoplay_timer -= dt
            if self._autoplay_timer <= 0:
                self._autoplay_next()

        self.root.after(33, self._update_fleet)

    def _make_robot_stuck(self):
        """Every 8 seconds, make a random autonomous robot stuck."""
        if self.stuck_robot is None or self.stuck_robot.state != "stuck":
            autonomous = [r for r in self.robots if r.state == "autonomous"]
            if autonomous:
                robot = random.choice(autonomous)
                robot.state = "stuck"
                robot.flash_tick = 0
                self.stuck_robot = robot
                self._ov_status.config(
                    text=f"ROBOT #{robot.id} NEEDS OVERRIDE", fg=RED)
                self._ov_action.config(text="", fg=GRAY)
                self._ov_detail.config(text="")
        self.root.after(8000, self._make_robot_stuck)

    # -----------------------------------------------------------------
    #  EVENT HANDLERS
    # -----------------------------------------------------------------

    def _on_key(self, event):
        """Global key handler (ignored when typing in chat)."""
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
        """Open a matplotlib window showing a bonus evidence chart."""
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
        """Brain signal button clicked."""
        if self._pipeline_running:
            return
        self._run_decode(label)

    def _on_chat_focus(self, event):
        if self._chat_placeholder:
            self._chat.delete(0, tk.END)
            self._chat.config(fg=TEXT)
            self._chat_placeholder = False

    def _on_chat_blur(self, event):
        if not self._chat.get().strip():
            self._chat.delete(0, tk.END)
            self._chat.insert(0,
                "Type command... (e.g. 'turn left', 'go forward', 'stop')")
            self._chat.config(fg=DIM)
            self._chat_placeholder = True

    def _on_chat_submit(self):
        if self._pipeline_running:
            return
        text = self._chat.get().strip().lower()
        if not text or self._chat_placeholder:
            return

        # Match command (longest first)
        label = None
        for cmd, lbl in CHAT_COMMANDS:
            if cmd in text:
                label = lbl
                break

        if label is None:
            self._chat_result.config(
                text=f"Unknown command: '{text}'. Try 'left', 'forward', 'stop'...",
                fg=RED)
            return

        self._chat.delete(0, tk.END)
        self._chat_placeholder = False
        self._run_decode(label)

    # -----------------------------------------------------------------
    #  PIPELINE DECODE
    # -----------------------------------------------------------------

    def _run_decode(self, label):
        """Start pipeline decode in a background thread."""
        self._pipeline_running = True
        self._pipeline_result = None

        # Pick target robot
        if self.stuck_robot and self.stuck_robot.state == "stuck":
            target = self.stuck_robot
        else:
            target = random.choice(self.robots)

        self._decode_target = target
        self._decode_label = label

        # Show decoding status
        self._ov_status.config(
            text=f"DECODING: {label}...", fg=YELLOW)
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

            # Fresh pipeline instance, shared models
            pipe = ThoughtLinkPipeline()
            pipe.stage1_model = PIPELINE.stage1_model
            pipe.stage2_model = PIPELINE.stage2_model
            pipe.DIRECTION_TO_ACTION = PIPELINE.DIRECTION_TO_ACTION

            # Load raw EEG for display
            arr = np.load(npz_path, allow_pickle=True)
            eeg_raw = arr["feature_eeg"]

            # Consume all windows
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
        """Poll for pipeline completion, then apply result."""
        if thread.is_alive():
            self.root.after(50, lambda: self._poll_decode(thread))
            return

        self._pipeline_running = False
        result = self._pipeline_result
        target = self._decode_target

        if result is None:
            self._ov_status.config(text="DECODE FAILED", fg=RED)
            self._ov_action.config(text="", fg=GRAY)
            return

        action = result["action"]
        label = result["label"]
        color = ACTION_COLORS.get(action, GRAY)
        lat = result["latency"]
        conf = result["confidence"]

        # --- Update override section ---
        self._ov_status.config(
            text=f"OVERRIDE SENT: {action} \u2192 Robot #{target.id}", fg=GREEN)
        self._ov_action.config(text=action, fg=color)
        self._ov_detail.config(
            text=f"Brain Signal: {label} \u2192 {action} | Phase: {result['phase']}",
            fg=TEXT)

        # --- Update metrics ---
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

        # --- Update chat result ---
        self._chat_result.config(
            text=f"Decoded: {label} EEG \u2192 {action} "
                 f"({lat:.0f}ms, conf {conf:.2f}) \u2192 Robot #{target.id}",
            fg=GREEN)

        # --- Update fleet robot ---
        target.state = "override"
        target.override_action = action
        target.override_timer = 3.0
        target.flash_tick = 0
        if self.stuck_robot == target:
            self.stuck_robot = None
        self._override_target = target

        # --- Update MuJoCo ---
        if self.mujoco:
            self.mujoco.set_action(action)
            self._current_mujoco_action = action

        # --- Update EEG preview ---
        self._draw_eeg(result["eeg_raw"])

    def _show_all_autonomous(self):
        """Reset override display to idle state."""
        self._ov_status.config(text="ALL ROBOTS AUTONOMOUS", fg="#1a4d2e")
        self._ov_action.config(text="", fg=GRAY)
        self._ov_detail.config(text="")

    # -----------------------------------------------------------------
    #  EEG PREVIEW
    # -----------------------------------------------------------------

    def _draw_eeg(self, eeg_raw):
        """Draw 6-channel EEG waveform in the preview canvas."""
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
            self._autoplay_idx = 0
            self._autoplay_timer = 1.0  # first signal after 1s
            self.root.title(
                "ThoughtLink | AUTO-PLAY ON | Press A to stop, Q to quit")
        else:
            self.root.title(
                "ThoughtLink: Multi-Robot Orchestration | A=auto-play  Q=quit")

    def _autoplay_next(self):
        """Trigger the next signal in the auto-play cycle."""
        labels = list(DEMO_FILES.keys())

        # Ensure a stuck robot exists
        if self.stuck_robot is None or self.stuck_robot.state != "stuck":
            autonomous = [r for r in self.robots if r.state == "autonomous"]
            if autonomous:
                robot = random.choice(autonomous)
                robot.state = "stuck"
                robot.flash_tick = 0
                self.stuck_robot = robot
                self._ov_status.config(
                    text=f"ROBOT #{robot.id} NEEDS OVERRIDE", fg=RED)

        label = labels[self._autoplay_idx % len(labels)]
        self._autoplay_idx += 1
        self._autoplay_timer = 5.0
        self._run_decode(label)

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

    # Start MuJoCo (optional — opens its own window)
    mujoco = MuJoCoRunner()
    mujoco.start()

    # Create tkinter window on right half of screen
    root = tk.Tk()
    root.title("ThoughtLink: Multi-Robot Orchestration | A=auto-play  Q=quit")
    root.configure(bg=BG)

    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    win_w = screen_w // 2
    win_h = screen_h - 80
    root.geometry(f"{win_w}x{win_h}+{screen_w // 2}+0")
    root.minsize(700, 750)

    panel = ControlPanel(root, mujoco)

    def on_close():
        panel._quit()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
