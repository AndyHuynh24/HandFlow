import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import tensorflow as tf
import cv2
import numpy as np
import time
import mediapipe as mp
import pyautogui
import keyboard
import threading
from tkinter import filedialog
import subprocess
import platform
import sys

SETTINGS_FILE = "gesture_mapping.json"
MAX_ACTIONS = 10

if sys.platform == "darwin":  # macOS
    from Quartz.CoreGraphics import CGEventCreateMouseEvent, CGEventPost, kCGHIDEventTap
    from Quartz.CoreGraphics import kCGMouseButtonLeft, kCGEventMouseMoved
elif sys.platform.startswith("win"):  # Windows
    import ctypes
    user32 = ctypes.windll.user32

GESTURES = [
    "Left_swipeleft",
    "Left_zoom",
    "Left_pointyclick",
    "Left_middleclick",
    "Left_ringclick",
    "Left_pinkyclick", 
    "Right_swiperight",
    "Right_zoom", 
    "Right_pointyclick", 
    "Right_middleclick", 
    "Right_ringclick", 
    "Right_pinkyclick",
]

ACTIONS = [
    "None",
    "Custom Text",
    "Custom Shortcut",
    "Scroll Up",
    "Scroll Down",
    "Zoom In",
    "Zoom Out",
    "Open File/App",
    "Open Folder",
    "leftclick", 
    "rightclick"
]



#mediapipe implementation
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
actions_right = np.array(['none', 'nonezoom', 'swiperight', 'zoom', 'pointyclick', 'middleclick', 'ringclick', 'pinkyclick'])
actions_left = np.array(['none', 'nonezoom', 'swipeleft', 'zoom', 'pointyclick', 'middleclick', 'ringclick', 'pinkyclick'])

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    image = cv2.flip(image, 1)
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    if not results.multi_hand_landmarks:
        return
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

def extract_keypoints_normalized(results):
    if results.multi_hand_landmarks:
        right_hand = None
        left_hand = None
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[idx].classification[0].label
            kp = np.array([[res.x, res.y, res.z, getattr(res, "visibility", 0.0)] for res in hand_landmarks.landmark])
            origin = kp[0, :3]
            kp[:, :3] -= origin
            max_dist = np.max(np.linalg.norm(kp[:, :3], axis=1))
            if max_dist > 0:
                kp[:, :3] /= max_dist
            if (hand_label == 'Right'):
                right_hand = (kp.flatten())
            else:
                left_hand = kp.flatten()
            
        #print("righthands", right_hand)
        #print("lefthands", left_hand)
        return right_hand, left_hand
    else:
        return np.zeros(21*4), np.zeros(21*4)

colors = [(245,117,16), (117,245,16), (16,117,245), (245,117,16), (117,245,16), (16,117,245), (245,117,16), (245,117,16), (245,117,16), (245,117,16), (245,117,16), (245,117,16)]
def prob_viz(res_right, res_left, actions_right, actions_left, input_frame, colors, right_lock, left_lock):
    output_frame = input_frame.copy()
    h, w, _ = output_frame.shape  # image height, width
    if right_lock:
        for num, prob in enumerate(res_right):
            bar_length = int(prob * 100)
            margin = 10

            # Right-aligned rectangle
            x2 = w - margin
            x1 = x2 - bar_length
            y1 = 60 + num * 40
            y2 = 90 + num * 40

            cv2.rectangle(output_frame, (x1, y1), (x2, y2), colors[num], -1)

            # Right-aligned text
            text = actions_right[num]
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            text_x = w - text_w - margin
            text_y = 85 + num * 40

            cv2.putText(output_frame, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    if left_lock: 
        for num, prob in enumerate(res_left):
            cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
            cv2.putText(output_frame, actions_right[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    

    
    return output_frame


# Dummy scrollers
class ContinuousScroller:
    def __init__(self, speed=-6, interval=0.02, start_delay=2):
        self.speed = speed
        self.interval = interval
        self.start_delay = start_delay
        self._scrolling = False
        self._thread = None
        self._lock = threading.Lock()
        self._first_start = True

    def reset_first_start(self):
        self._first_start = True

    def _scroll_worker(self):
        if not self._first_start:
            start_time = time.time()
            while time.time() - start_time < self.start_delay:
                with self._lock:
                    if not self._scrolling:
                        return
                time.sleep(0.01)
        self._first_start = False
        while True:
            with self._lock:
                if not self._scrolling: break
            pyautogui.scroll(self.speed)
            time.sleep(self.interval)

    def start_scroll(self):
        with self._lock:
            if not self._scrolling:
                self._scrolling = True
                self._thread = threading.Thread(target=self._scroll_worker, daemon=True)
                self._thread.start()

    def stop_scroll(self):
        with self._lock:
            self._scrolling = False
        if self._thread:
            self._thread.join()
            self._thread = None

cscroller = ContinuousScroller(speed=-8, interval=0.01, start_delay=2)
upScroller = ContinuousScroller(speed=8, interval=0.01, start_delay=2)
import math, threading, sys, pyautogui
from collections import deque

try:
    from Quartz import (
        CGEventCreateMouseEvent, kCGEventMouseMoved, kCGMouseButtonLeft,
        CGEventPost, kCGHIDEventTap, CGDisplayBounds,
        CGGetActiveDisplayList
    )
except ImportError:
    pass


import math, threading, sys, pyautogui
from collections import deque
try:
    from Quartz import (
        CGEventCreateMouseEvent, kCGEventMouseMoved, kCGMouseButtonLeft,
        CGEventPost, kCGHIDEventTap, CGDisplayBounds, CGGetActiveDisplayList
    )
except ImportError:
    pass


class MouseFollower:
    def __init__(self, smoothing=0.25, base_sensitivity=1):
        """
        smoothing: smoothing factor (0–1)
        base_sensitivity: master multiplier for motion speed
        """
        self.latest_landmark = None
        self.is_moving = False
        self.mouse_thread = None
        self.prev_x = self.prev_y = None
        self.smoothing = smoothing
        self.base_sensitivity = base_sensitivity

        # state
        self._origin_landmark = None
        self._origin_mouse = None
        self._velocity = 0.0
        self._hover_lock = 0.0
        self._lock_decay = 0.9
        self._smooth_buffer = deque(maxlen=5)
        self._z_smooth = None
        self._still_frames = 0            # counts frames of stillness
        self._still_threshold = 0.005    # below this movement, consider still
        self._still_lock = False          # freeze flag

        # display info
        self._virtual_bounds = self._compute_total_bounds()

        # platform detection
        if sys.platform == "darwin":
            self._move_mode = "mac"
        elif sys.platform.startswith("win"):
            self._move_mode = "win"
        else:
            self._move_mode = "pyautogui"

    # ---------------- Public ----------------
    def update_landmark(self, x, y, z=0, w=None, h=None):
        """Update normalized coordinates (x,y∈[0,1]) and z≈2e-7→8e-7."""
        self.latest_landmark = (x, y, z, w, h)

    def start(self):
        if not self.is_moving:
            self.is_moving = True
            self._origin_landmark = None
            self._origin_mouse = None
            self.mouse_thread = threading.Thread(target=self._mouse_follow_loop, daemon=True)
            self.mouse_thread.start()
            print("[MouseFollower] Mouse follow started.")

    def stop(self):
        self.is_moving = False
        if self.mouse_thread:
            self.mouse_thread.join(timeout=0.2)
            self.mouse_thread = None
        self._origin_landmark = None
        self._origin_mouse = None
        self.prev_x = self.prev_y = None
        self.latest_landmark = None
        self._velocity = 0
        self._hover_lock = 0
        self._smooth_buffer.clear()
        print("[MouseFollower] Mouse follow stopped.")

    # ---------------- Internal ----------------
    def _compute_total_bounds(self):
        """Compute combined display bounds (min_x,min_y,width,height)."""
        if sys.platform == "darwin":
            max_displays = 16
            err, display_ids, count = CGGetActiveDisplayList(max_displays, None, None)
            if err == 0 and count > 0:
                bounds = [CGDisplayBounds(display_ids[i]) for i in range(count)]
                min_x = min(b.origin.x for b in bounds)
                min_y = min(b.origin.y for b in bounds)
                max_x = max(b.origin.x + b.size.width for b in bounds)
                max_y = max(b.origin.y + b.size.height for b in bounds)
                return (min_x, min_y, max_x - min_x, max_y - min_y)
        sw, sh = pyautogui.size()
        return (0, 0, sw, sh)

    def _move_mouse_native(self, norm_x, norm_y):
        """Convert normalized coords (0–1) to pixels using virtual desktop bounds."""
        min_x, min_y, total_w, total_h = self._virtual_bounds
        px = int(min_x + max(0, min(1, norm_x)) * (total_w - 1))
        py = int(min_y + max(0, min(1, norm_y)) * (total_h - 1))

        if self._move_mode == "mac":
            event = CGEventCreateMouseEvent(None, kCGEventMouseMoved, (px, py), kCGMouseButtonLeft)
            CGEventPost(kCGHIDEventTap, event)
        elif self._move_mode == "win":
            import ctypes
            ctypes.windll.user32.SetCursorPos(px, py)
        else:
            pyautogui.moveTo(px, py)

    def _mouse_follow_loop(self):
        import numpy as np
        poll_interval = 0.002
        velocity_smooth = 0.25
        precision_gain = 1.4
        exp_high, exp_low = 0.92, 0.25

        # refined thresholds
        inner_deadzone = 0.005
        outer_deadzone = 0.014
        move_activation = 0.06

        # Still lock params
        stable_frames = 0
        stable_lock_threshold = 10
        stable_unlock_threshold = 2
        last_stable_target = None
        still_locked = False

        # Smooth transition control
        jitter_window = deque(maxlen=8)
        jitter_limit = 0.0038
        adaptive_alpha = 0.4

        # Motion blend tuning
        move_smooth = 0.65   # blending factor for smooth directional change
        unlock_boost = 1.5   # boost cursor reactivity right after unlocking
        active = False

        while self.is_moving:
            if not self.latest_landmark:
                threading.Event().wait(poll_interval)
                continue

            lx, ly, lz = self.latest_landmark[:3]

            # init
            if self._origin_landmark is None:
                self._origin_landmark = (lx, ly, lz)
                cur = pyautogui.position()
                min_x, min_y, total_w, total_h = self._virtual_bounds
                self._origin_mouse = ((cur.x - min_x) / total_w, (cur.y - min_y) / total_h)
                self.prev_x, self.prev_y = self._origin_mouse
                self._z_smooth = lz
                self._smooth_buffer.append(self._origin_mouse)
                continue

            # smooth z and depth factor
            self._z_smooth = 0.9 * self._z_smooth + 0.1 * lz
            z = max(2e-7, min(8e-7, self._z_smooth))
            z_norm = (z - 2e-7) / (8e-7 - 2e-7)
            inv_norm = (1.0 - z_norm) ** 1.2
            depth_factor = 0.5 + (2.0 - 0.5) * inv_norm

            dx = lx - self._origin_landmark[0]
            dy = ly - self._origin_landmark[1]
            dist_norm = math.hypot(dx, dy)

            # activation
            if not active and dist_norm > move_activation:
                active = True
            elif active and dist_norm < inner_deadzone:
                active = False

            if not active:
                threading.Event().wait(poll_interval)
                continue

            # adaptive jitter smoothing
            jitter_window.append(dist_norm)
            avg_jitter = np.std(jitter_window) if len(jitter_window) > 3 else 0
            if avg_jitter < jitter_limit:
                adaptive_alpha = 0.9
            else:
                adaptive_alpha = 0.35 + (0.3 * min(1, avg_jitter / 0.01))

            # deadzones
            if dist_norm < inner_deadzone:
                dx = dy = 0
                dist_norm = 0
            elif dist_norm < outer_deadzone:
                factor = (dist_norm - inner_deadzone) / (outer_deadzone - inner_deadzone)
                dx *= factor * 0.5
                dy *= factor * 0.5

            # scaling
            scale = (dist_norm ** 0.5) * precision_gain + 0.001
            move_scale = scale * self.base_sensitivity * depth_factor
            target_x = self._origin_mouse[0] + dx * move_scale
            target_y = self._origin_mouse[1] + dy * move_scale

            # velocity adaptive smoothing
            diff = math.hypot(target_x - self.prev_x, target_y - self.prev_y)
            self._velocity = (1 - velocity_smooth) * self._velocity + velocity_smooth * diff
            speed_norm = min(self._velocity / 0.008, 1.0)
            base_alpha = exp_high - (exp_high - exp_low) * speed_norm
            alpha = base_alpha * adaptive_alpha

            # Still lock filter — smoother version
            if diff < 0.0009:
                stable_frames += 1
            else:
                stable_frames = max(0, stable_frames - 1)

            if not still_locked and stable_frames > stable_lock_threshold:
                still_locked = True
                last_stable_target = (self.prev_x, self.prev_y)
            elif still_locked and stable_frames < stable_unlock_threshold:
                still_locked = False

            # React faster right after unlocking
            if still_locked and last_stable_target is not None:
                target_x, target_y = last_stable_target
                alpha = 1.0
            elif not still_locked and stable_frames < stable_unlock_threshold:
                alpha = min(1.0, alpha * unlock_boost)

            # Smooth directional blending
            smoothed_x = self.prev_x * (1 - move_smooth) + target_x * move_smooth
            smoothed_y = self.prev_y * (1 - move_smooth) + target_y * move_smooth
            final_x = self.prev_x + (smoothed_x - self.prev_x) * alpha
            final_y = self.prev_y + (smoothed_y - self.prev_y) * alpha

            self._smooth_buffer.append((final_x, final_y))
            avg_x = sum(p[0] for p in self._smooth_buffer) / len(self._smooth_buffer)
            avg_y = sum(p[1] for p in self._smooth_buffer) / len(self._smooth_buffer)

            # micro-hysteresis filter
            if abs(avg_x - self.prev_x) < 0.00025 and abs(avg_y - self.prev_y) < 0.00025:
                avg_x, avg_y = self.prev_x, self.prev_y

            self._move_mouse_native(avg_x, avg_y)
            self.prev_x, self.prev_y = avg_x, avg_y
            threading.Event().wait(poll_interval)





mouseFollower = MouseFollower() 

class GestureMapperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Gesture Action Mapper")
        self.root.geometry("900x600")

        # Data holders
        self.mapping = {}             # gesture -> list of actions
        self.custom_values = {}       # gesture -> list of custom values
        self.widgets = {}             # gesture -> list of StringVars
        self.custom_labels = {}       # gesture -> list of labels

        # UI
        self.create_ui()
        self.load_settings()

        #mouse event
        self.is_moving = False
        self.mouse_thread = None
        self.latest_landmark = None

    # ---------- UI ----------
    def create_ui(self):
        tk.Label(self.root, text="Map Hand Gestures to Actions (up to 4 per gesture)",
                 font=("Arial", 14, "bold")).pack(pady=10)

        # Scrollable container
        container = tk.Frame(self.root)
        container.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        canvas = tk.Canvas(container)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Gesture frames
        self.action_frames = {}

        for gesture in GESTURES:
            gesture_frame = tk.Frame(scrollable_frame, relief="groove", bd=1, padx=5, pady=5)
            gesture_frame.pack(fill="x", pady=5)

            top_row = tk.Frame(gesture_frame)
            top_row.pack(fill="x")

            tk.Label(top_row, text=gesture.capitalize() + ":", width=15, anchor="w").pack(side="left")
            actions_frame = tk.Frame(gesture_frame)
            actions_frame.pack(fill="x", pady=3)

            self.action_frames[gesture] = actions_frame
            self.widgets[gesture] = []
            self.custom_labels[gesture] = []
            self.mapping[gesture] = []
            self.custom_values[gesture] = []

            # function to add a row
            def add_action_dropdown(index, g=gesture, parent_frame=actions_frame):
                if len(self.widgets[g]) >= MAX_ACTIONS:
                    messagebox.showwarning("Limit reached", f"Max {MAX_ACTIONS} actions per gesture")
                    return

                row = tk.Frame(parent_frame)
                row.pack(fill="x", pady=2)

                var = tk.StringVar(value="None")
                combo = ttk.Combobox(row, textvariable=var, values=ACTIONS, state="readonly", width=25)
                combo.pack(side="left", padx=2)

                lbl = tk.Label(row, text="", width=30, anchor="w", relief="groove")
                lbl.pack(side="left", padx=2)

                delete_btn = tk.Button(row, text="X", fg="red", command=lambda r=row, i=index, g=g: self.delete_action(g, i, r))
                delete_btn.pack(side="left", padx=2)

                combo.bind("<<ComboboxSelected>>", lambda e, g=g, i=index, v=var: self.on_action_selected(g, v, i))

                self.widgets[g].append(var)
                self.custom_labels[g].append(lbl)
                self.mapping[g].append("None")
                self.custom_values[g].append("")

            # first dropdown
            add_action_dropdown(0)
            tk.Button(top_row, text="Add another action",
                      command=lambda g=gesture: add_action_dropdown(len(self.widgets[g]), g, self.action_frames[g])
                      ).pack(side="left", padx=5)

        # Bottom buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(side="bottom", pady=10)
        tk.Button(btn_frame, text="Save Settings", command=self.save_settings).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Load Settings", command=self.load_settings).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Run Detection", command=self.run_detection).pack(side="left", padx=5)

        # Mousewheel scrolling
        def _on_mouse_wheel(event):
            direction = -1 if event.delta > 0 else 1
            canvas.yview_scroll(direction, "units")
        canvas.bind_all("<MouseWheel>", _on_mouse_wheel)

    # ---------- Action handlers ----------
    def on_action_selected(self, gesture, var, index):
        choice = var.get()
        self.mapping[gesture][index] = choice
        self.custom_values[gesture][index] = ''
        self.custom_labels[gesture][index].config(text='')

        if choice == "Custom Text":
            popup = tk.Toplevel(self.root)
            popup.title(f"Custom text for {gesture} action {index+1}")
            entry = tk.Entry(popup)
            entry.pack()
            def save_text():
                txt = entry.get().strip()
                if txt:
                    self.custom_values[gesture][index] = txt
                    self.custom_labels[gesture][index].config(text=txt)
                popup.destroy()
            tk.Button(popup, text="Save", command=save_text).pack()

        elif choice == "Custom Shortcut":
            self.open_custom_shortcut_popup(gesture, index)

        elif choice == "Open File/App":
            self.open_custom_file_popup(gesture, index)

    def open_custom_file_popup(self, gesture, index):
        file_path = filedialog.askopenfilename(title=f"Select file/application for {gesture} action {index+1}")
        if file_path:
            self.mapping[gesture][index] = "Open File/App"
            self.custom_values[gesture][index] = file_path
            self.custom_labels[gesture][index].config(text=os.path.basename(file_path))

    def open_custom_shortcut_popup(self, gesture, index):
        popup = tk.Toplevel(self.root)
        popup.title(f"Custom shortcut for {gesture} action {index+1}")
        popup.geometry("360x150")
        popup.grab_set()
        tk.Label(popup, text="Press your shortcut keys (while this window is focused):").pack(pady=10)

        shortcut_var = tk.StringVar()
        entry = tk.Entry(popup, textvariable=shortcut_var, width=40, state="readonly", justify="center")
        entry.pack(pady=5)
        entry.focus_set()

        pressed_keys, order = set(), []
        KEY_TRANSLATE = {
            "Meta_L": "command", "Meta_R": "command",
            "Control_L": "ctrl", "Control_R": "ctrl",
            "Shift_L": "shift", "Shift_R": "shift",
            "Alt_L": "alt", "Alt_R": "alt"
        }

        def on_press(event):
            key = KEY_TRANSLATE.get(event.keysym, event.keysym).lower()
            if key not in pressed_keys:
                pressed_keys.add(key)
                order.append(key)
                shortcut_var.set("+".join(order))

        def on_release(event):
            key = KEY_TRANSLATE.get(event.keysym, event.keysym).lower()
            if key in pressed_keys:
                pressed_keys.remove(key)

        entry.bind("<KeyPress>", on_press)
        entry.bind("<KeyRelease>", on_release)

        def save():
            if order:
                self.mapping[gesture][index] = "Custom Shortcut"
                self.custom_values[gesture][index] = "+".join(order)
                self.custom_labels[gesture][index].config(text=self.custom_values[gesture][index])
            popup.destroy()

        tk.Button(popup, text="Save", command=save).pack(pady=10)

    def delete_action(self, gesture, index, row_widget):
        if index < len(self.widgets[gesture]):
            self.widgets[gesture].pop(index)
            self.custom_labels[gesture].pop(index)
            self.mapping[gesture].pop(index)
            self.custom_values[gesture].pop(index)
            row_widget.destroy()

    # ---------- Save / Load ----------
    def save_settings(self):
        data = {"mapping": self.mapping, "custom_values": self.custom_values}
        with open(SETTINGS_FILE, "w") as f:
            json.dump(data, f, indent=4)
        messagebox.showinfo("Saved", "Gesture mappings saved successfully!")

    def load_settings(self):
        if not os.path.exists(SETTINGS_FILE):
            return
        with open(SETTINGS_FILE, "r") as f:
            data = json.load(f)

        self.mapping = data.get("mapping", {g: [] for g in GESTURES})
        self.custom_values = data.get("custom_values", {g: [] for g in GESTURES})

        for gesture in GESTURES:
            actions_list = self.mapping.get(gesture, [])
            values_list = self.custom_values.get(gesture, [])

            # Clear old UI
            for widget in self.action_frames[gesture].winfo_children():
                widget.destroy()
            self.widgets[gesture], self.custom_labels[gesture] = [], []

            for idx, val in enumerate(actions_list):
                row = tk.Frame(self.action_frames[gesture])
                row.pack(fill="x", pady=2)

                var = tk.StringVar(value=val)
                combo = ttk.Combobox(row, textvariable=var, values=ACTIONS, state="readonly", width=25)
                combo.pack(side="left", padx=2)

                lbl = tk.Label(row, text=values_list[idx] if idx < len(values_list) else "", width=30, anchor="w", relief="groove")
                lbl.pack(side="left", padx=2)

                delete_btn = tk.Button(row, text="X", fg="red", command=lambda r=row, i=idx, g=gesture: self.delete_action(g, i, r))
                delete_btn.pack(side="left", padx=2)

                combo.bind("<<ComboboxSelected>>", lambda e, g=gesture, i=idx, v=var: self.on_action_selected(g, v, i))

                self.widgets[gesture].append(var)
                self.custom_labels[gesture].append(lbl)

    # ---------- Perform actions ----------
    def perform_action(self, gesture_name):
        print("gesture", gesture_name)
        actions_list = self.mapping.get(gesture_name, [])
        for idx, action_choice in enumerate(actions_list):
            if idx > 0:
                time.sleep(0.17)  # small delay between multiple actions

            if action_choice == "None":
                continue
            elif action_choice == "leftclick":
                pyautogui.click()
            elif action_choice == "rightclick": 
                pyautogui.rightClick()
            elif action_choice == "Custom Text":
                txt = self.custom_values[gesture_name][idx]
                if txt:
                    pyautogui.write(txt, interval=0)

            elif action_choice == "Custom Shortcut":
                shortcut = self.custom_values[gesture_name][idx]
                if shortcut:
                    pyautogui.hotkey(*shortcut.split("+"))

            elif action_choice == "Open File/App":
                filepath = self.custom_values[gesture_name][idx]
                if filepath and os.path.exists(filepath):
                    if os.name == 'nt':  # Windows
                        os.startfile(filepath)
                    elif os.name == 'posix':  # macOS or Linux
                        import subprocess
                        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", filepath])
                    print(f"Opened file/app: {filepath}")
                else:
                    print(f"File/App path invalid or not set: {filepath}")

            elif action_choice == 'Scroll Down':
                cscroller.reset_first_start()
                cscroller.start_scroll()

            elif action_choice == 'Scroll Up':
                upScroller.reset_first_start()
                upScroller.start_scroll()

            # elif action_choice == "LeftClick Mouse":
            #     # ⚡ When your detection gives (x,y), call move_mouse_to_landmark(x,y,w,h)
            #     print("Mouse control mode active – move using landmarks.")


            else:
                print("Execute named action:", action_choice)


    # ---------- Detection placeholder ----------
    def run_detection_thread(self):
        threading.Thread(target=self.run_detection, daemon=True).start()

    def run_detection(self):
        RIGHT_MODEL_FILE = 'ModelTraining/right_action.h5'
        LEFT_MODEL_FILE = 'ModelTraining/left_action.h5'
        if os.path.exists(RIGHT_MODEL_FILE) and os.path.exists(LEFT_MODEL_FILE):
            self.right_model = tf.keras.models.load_model(RIGHT_MODEL_FILE)
            self.left_model = tf.keras.models.load_model(LEFT_MODEL_FILE)
            print(f"✅ Loaded model from {RIGHT_MODEL_FILE}")
            print(f"✅ Loaded model from {LEFT_MODEL_FILE}")
        else:
            print('error unable to load file')
            return

        # Detection loop (unchanged logic except perform_action call)
        right_sequence = []
        right_lock = False
        left_sequence = []
        left_lock = False

        right_sentence = ['none']
        left_sentence = ['none']
        threshold = 0.5
        right_predictions = []
        left_predictions = []

        res_right = []
        res_left = []

        cap = cv2.VideoCapture(0)
        prev_time = 0

        with mp_hands.Hands(min_detection_confidence=0.5,
                            min_tracking_confidence=0.2,
                            max_num_hands=2) as hands:

            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    break

                image, results = mediapipe_detection(frame, hands)
                h, w, _ = image.shape
                draw_landmarks(image, results)


                #if results.multi_hand_landmarks:
                    #for hand_landmarks in results.multi_hand_landmarks:
                        #print(hand_landmarks)
                
                    #print('Handedness:', results.multi_handedness)

                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
                prev_time = curr_time
                cv2.putText(image, f'FPS: {int(fps)}', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                right_keypoints, left_keypoints = extract_keypoints_normalized(results)
                if (right_keypoints is not None) :
                    right_sequence.append(right_keypoints)
                    right_sequence = right_sequence[-16:]
                    right_lock = True
                if (left_keypoints is not None) :
                    left_sequence.append(left_keypoints)
                    left_sequence = left_sequence[-16:]
                    left_lock = True

                if results.multi_hand_landmarks:        
                    if  (right_lock and len(right_sequence) == 16):
                        if (mouseFollower.is_moving):
                            right_hand_landmarks = None
                            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                                hand_label = results.multi_handedness[idx].classification[0].label
                                if hand_label == "Right":
                                    right_hand_landmarks = hand_landmarks
                                    break  # stop after finding the right hand
                        
                            #print("right_hand", right_hand_landmarks.landmark[4].x) 
                            mouseFollower.update_landmark(right_hand_landmarks.landmark[0].x, right_hand_landmarks.landmark[0].y)

                        # quieter predict
                        res_right = self.right_model.predict(np.expand_dims(right_sequence, axis=0), verbose=0)[0]
                        right_predictions.append(np.argmax(res_right))

                        # stability check - use most common of last N predictions for reliability
                        from collections import Counter
                        last_preds = right_predictions[-5:]
                        most_common = Counter(last_preds).most_common(1)[0][0] if last_preds else None
                    
                        if most_common == np.argmax(res_right):
                            if res_right[np.argmax(res_right)] > threshold:
                                gesture_name = "Right_" + actions_right[np.argmax(res_right)]
                                if len(right_sentence) > 0:
                                    #print("rightkeypoints", right_keypoints.shape)
                                    if (gesture_name != right_sentence[-1]):
                                        # if (mouseFollower.is_moving):
                                        #     if gesture_name == 'Right_leftclick': 
                                        #         pyautogui.click()
                                        #     elif gesture_name == 'Right_rightclick': 
                                        #         pyautogui.rightClick()
                                        #if gesture_name not in ['nonezoom']:
                                            # call class method that resolves mapping and custom values
                                        if (gesture_name == "Right_pinkyclick"):
                                            print("is mouse moving (before)", mouseFollower.is_moving) 
                                            if (mouseFollower.is_moving):
                                                mouseFollower.stop()
                                            else:
                                                mouseFollower.start()
                                            
                                            print("is mouse moving:", mouseFollower.is_moving)
                                        elif (gesture_name == "Right_ringclick"):
                                            mouseFollower.stop()
                                            
                                            print("is mouse moving:", mouseFollower.is_moving)
                                        self.perform_action(gesture_name)
                                        if (gesture_name == 'Right_nonezoom' or gesture_name == 'Right_none'):
                                            if (right_sentence[-1] == "Right_ringclick"):
                                                mouseFollower.start()
                                            cscroller.stop_scroll()
                                            upScroller.stop_scroll()
                                            
                                            print('stop')

                                        right_sentence.append(gesture_name)
                                else:
                                    if gesture_name not in ['nonezoom']:
                                        self.perform_action(gesture_name)
                                        right_sentence.append(gesture_name)
                        if len(right_sentence) > 5:
                            right_sentence = right_sentence[-3:]

                    if  (left_lock and len(left_sequence) == 16) :
                        # quieter predict
                        res_left = self.left_model.predict(np.expand_dims(left_sequence, axis=0), verbose=0)[0]
                        left_predictions.append(np.argmax(res_left))

                        # stability check - use most common of last N predictions for reliability
                        from collections import Counter
                        last_preds = left_predictions[-5:]
                        most_common = Counter(last_preds).most_common(1)[0][0] if last_preds else None
                    
                        if most_common == np.argmax(res_left):
                            if res_left[np.argmax(res_left)] > threshold:
                                gesture_name = "Left_" + actions_left[np.argmax(res_left)]
                                if len(left_sentence) > 0:
                                    if (gesture_name != left_sentence[-1]):
                                        #if gesture_name not in ['nonezoom']:
                                            # call class method that resolves mapping and custom values
                                        self.perform_action(gesture_name)
                                        if (gesture_name == 'Left_nonezoom' or gesture_name == 'Left_none') :
                                            cscroller.stop_scroll()
                                            upScroller.stop_scroll()
                                            print('stop')

                                        left_sentence.append(gesture_name)
                                else:
                                    if gesture_name not in ['nonezoom']:
                                        self.perform_action(gesture_name)
                                        left_sentence.append(gesture_name)
                                

                        if len(left_sentence) > 5:
                            left_sentence = left_sentence[-3:]

                    image = prob_viz(res_right, res_left,  actions_right, actions_left, image, colors, right_lock, left_lock) 
                    right_lock = False
                    left_lock = False

                else: 
                    if (right_sentence[-1] != 'none'):
                        cscroller.stop_scroll()
                        upScroller.stop_scroll()
                        mouseFollower.stop()
                        right_sentence.append('none')
                    if len(right_sentence) > 5:
                        right_sentence = right_sentence[-3:]

                    if (left_sentence[-1] != 'none'):
                        cscroller.stop_scroll()
                        upScroller.stop_scroll()
                        left_sentence.append('none')
                    if len(left_sentence) > 5:
                        left_sentence = left_sentence[-3:]

                # Draw full-width rectangle at top
                cv2.rectangle(image, (0, 0), (w, 40), (245, 117, 16), -1)

                # Left sentence (aligned left, same height)
                cv2.putText(image, ' '.join(left_sentence), (10, 30),++
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Right sentence (aligned right, same height)
                right_text = ' '.join(right_sentence)
                (text_w, text_h), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                x_right = w - text_w - 10  # 10px margin from right
                cv2.putText(image, right_text, (x_right, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow('OpenCV Feed', image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    root = tk.Tk()
    app = GestureMapperApp(root)
    root.mainloop()
