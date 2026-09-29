"""
Real-time OpenCV Drosophila Escape Reflex Simulation with Live Webcam Looming Detection.
Uses Leaky Integrate-and-Fire (LIF) dynamics with Berkeley (2024) biophysical parameters.
"""

import cv2
import numpy as np
import time
import pandas as pd
from collections import deque

class FlyEscapeVisualizer:
    def __init__(self):
        # 1. Load circuit connectivity
        print("[Init] Setting up MaleCNS v1.0 Escape Circuit...")
        # Pre-configured core circuit IDs:
        # Sensory: Optic Lobe / Visual Looming detector neurons (PVLP / DNp70)
        self.sensory_ids = [531985, 12215, 10580, 10074, 11133]
        self.sensory_names = ["PVLP010_R", "PVLP122_R", "DNp70_L", "PVLP010_L", "DNp70_R"]
        
        # Command Interneurons: Giant Fibers
        self.gf_ids = [10001, 10010]
        self.gf_names = ["DNp01(GF)_R", "DNp01(GF)_L"]
        
        # Motor / Muscle targets: Tergotrochanteral muscle (TTMn) & Interneurons (PSI)
        self.motor_ids = [800146, 802595]
        self.motor_names = ["TTMn_R (Jump)", "GFC2_L (Wing)"]

        self.neurons = self.sensory_names + self.gf_names + self.motor_names
        self.N = len(self.neurons)

        # 2. LIF Simulation Parameters (Shiu et al. 2024)
        self.tau_m = 20.0       # ms
        self.V_rest = 0.0
        self.V_th = 1.0
        self.V_reset = 0.0
        self.dt = 1.0           # ms per step
        self.ref_steps = 2      # 2 ms refractory
        
        self.V = np.zeros(self.N, dtype=np.float32)
        self.ref_counter = np.zeros(self.N, dtype=int)
        
        # Synaptic weight matrix: Sensory -> GF -> Motor
        self.W = np.zeros((self.N, self.N), dtype=np.float32)
        # Sensory -> GF connections
        for s_idx in range(5):
            self.W[5, s_idx] = 0.45  # to GF_R
            self.W[6, s_idx] = 0.45  # to GF_L
        # GF -> Motor
        self.W[7, 5] = 0.9          # GF_R -> TTMn
        self.W[7, 6] = 0.9          # GF_L -> TTMn
        self.W[8, 5] = 0.8          # GF_R -> GFC2
        self.W[8, 6] = 0.8          # GF_L -> GFC2

        # 3. Voltage History for oscilloscope graph (150 frames)
        self.history_len = 150
        self.v_history = {name: deque([0.0]*self.history_len, maxlen=self.history_len) for name in self.neurons}
        self.spike_events = deque(maxlen=20)

        # 4. Animation state of fly
        self.fly_state = "RESTING" # RESTING, JUMPING, ESCAPING
        self.fly_y_offset = 0
        self.wing_angle = 20
        self.escape_timer = 0

        # Previous frame for looming detection
        self.prev_gray = None
        self.looming_energy = 0.0

    def update_lif(self, sensory_drive):
        """Update LIF neurons for current time step"""
        I_ext = np.zeros(self.N, dtype=np.float32)
        # Apply looming drive to sensory neurons
        for i in range(5):
            I_ext[i] = sensory_drive * (0.8 + 0.4 * np.random.rand())

        spikes = np.zeros(self.N, dtype=bool)
        
        # Run 3 micro-steps per video frame for high temporal resolution
        for _ in range(3):
            I_syn = self.W @ spikes.astype(np.float32)
            dV = (-(self.V - self.V_rest) / self.tau_m + I_syn + I_ext) * self.dt
            V_new = self.V + dV
            
            is_ref = self.ref_counter > 0
            V_new[is_ref] = self.V_reset
            self.ref_counter[is_ref] -= 1
            
            fired = (V_new >= self.V_th) & (~is_ref)
            V_new[fired] = self.V_reset
            self.ref_counter[fired] = self.ref_steps
            
            self.V = V_new
            spikes |= fired

        # Record history
        for i, name in enumerate(self.neurons):
            self.v_history[name].append(float(self.V[i]) if not spikes[i] else 1.2)

        # Check if Giant Fiber fired -> trigger escape reflex
        if spikes[5] or spikes[6]:
            self.fly_state = "ESCAPING"
            self.escape_timer = 25
            self.spike_events.appendleft((time.time(), "GIANT FIBER FIRED (DNp01)"))

        return spikes

    def detect_looming(self, frame):
        """Detect looming motion (approaching hand/object expanding in FOV)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0

        # Frame difference + threshold
        delta = cv2.absdiff(self.prev_gray, gray)
        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Optical change / motion area
        motion_pixels = np.count_nonzero(thresh)
        total_pixels = gray.shape[0] * gray.shape[1]
        motion_ratio = motion_pixels / total_pixels

        self.prev_gray = gray
        
        # Non-linear expansion response (looming threshold)
        if motion_ratio > 0.08:
            drive = min(2.5, (motion_ratio - 0.08) * 10.0 + 0.6)
        else:
            drive = 0.0
        return drive

    def draw_fly(self, canvas, center_x, center_y):
        """Draw basic 2D biological fly schematic with wings and legs"""
        y_off = 0
        if self.fly_state == "ESCAPING":
            self.escape_timer -= 1
            y_off = -int((25 - self.escape_timer) * 4)
            self.wing_angle = 65 if (self.escape_timer % 4 < 2) else 10
            if self.escape_timer <= 0:
                self.fly_state = "RESTING"
        else:
            self.wing_angle = 20
            y_off = 0

        cx = center_x
        cy = center_y + y_off

        # Draw Shadow / Base
        cv2.ellipse(canvas, (cx, center_y + 80), (45, 12), 0, 0, 360, (40, 40, 40), -1)

        # Wings (Left & Right)
        wing_color = (210, 230, 240) if self.fly_state != "ESCAPING" else (80, 255, 255)
        # Left Wing
        pts_left = np.array([
            [cx - 15, cy - 10],
            [cx - 90, cy - int(self.wing_angle * 1.5)],
            [cx - 110, cy - int(self.wing_angle * 1.2) + 20],
            [cx - 40, cy + 20]
        ], np.int32)
        cv2.fillPoly(canvas, [pts_left], wing_color)
        cv2.polylines(canvas, [pts_left], True, (150, 180, 200), 2)

        # Right Wing
        pts_right = np.array([
            [cx + 15, cy - 10],
            [cx + 90, cy - int(self.wing_angle * 1.5)],
            [cx + 110, cy - int(self.wing_angle * 1.2) + 20],
            [cx + 40, cy + 20]
        ], np.int32)
        cv2.fillPoly(canvas, [pts_right], wing_color)
        cv2.polylines(canvas, [pts_right], True, (150, 180, 200), 2)

        # Legs (6 thoracic legs)
        leg_color = (60, 60, 70) if self.fly_state != "ESCAPING" else (0, 180, 255)
        # Front legs
        cv2.line(canvas, (cx - 20, cy - 5), (cx - 60, cy - 35), leg_color, 3)
        cv2.line(canvas, (cx + 20, cy - 5), (cx + 60, cy - 35), leg_color, 3)
        # Middle legs (Jump / TTMn extensor)
        ext_y = 20 if self.fly_state != "ESCAPING" else 55
        cv2.line(canvas, (cx - 25, cy + 15), (cx - 75, cy + ext_y), leg_color, 4)
        cv2.line(canvas, (cx + 25, cy + 15), (cx + 75, cy + ext_y), leg_color, 4)
        # Hind legs
        cv2.line(canvas, (cx - 20, cy + 40), (cx - 65, cy + 75), leg_color, 3)
        cv2.line(canvas, (cx + 20, cy + 40), (cx + 65, cy + 75), leg_color, 3)

        # Abdomen
        cv2.ellipse(canvas, (cx, cy + 45), (26, 45), 0, 0, 360, (50, 65, 80), -1)
        for stripe_y in range(cy + 20, cy + 80, 12):
            cv2.line(canvas, (cx - 20, stripe_y), (cx + 20, stripe_y), (35, 45, 55), 2)

        # Thorax
        cv2.circle(canvas, (cx, cy), 28, (65, 85, 105), -1)

        # Head & Compound Eyes
        cv2.circle(canvas, (cx, cy - 38), 20, (55, 70, 85), -1)
        # Red compound eyes (Optic lobes)
        eye_color = (40, 40, 230) if self.fly_state != "ESCAPING" else (0, 120, 255)
        cv2.circle(canvas, (cx - 16, cy - 42), 11, eye_color, -1)
        cv2.circle(canvas, (cx + 16, cy - 42), 11, eye_color, -1)

        # Status text overlay
        status_label = "STATUS: IDLE / SENSING" if self.fly_state != "ESCAPING" else ">>> ESCAPE REFLEX TRIGGERED (JUMP & FLIGHT) <<<"
        color = (100, 255, 100) if self.fly_state != "ESCAPING" else (0, 0, 255)
        cv2.putText(canvas, status_label, (cx - 180, cy - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    def draw_oscilloscope(self, canvas, x_start, y_start, width, height):
        """Draw real-time multi-channel spike and membrane voltage oscilloscope"""
        cv2.rectangle(canvas, (x_start, y_start), (x_start + width, y_start + height), (25, 25, 30), -1)
        cv2.rectangle(canvas, (x_start, y_start), (x_start + width, y_start + height), (80, 80, 90), 1)
        
        cv2.putText(canvas, "LIVE NEURON MEMBRANE POTENTIAL (LIF OSCILLOSCOPE)", (x_start + 12, y_start + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 255), 1)

        channels = [
            ("Sensory (PVLP / DNp70)", self.v_history["PVLP010_R"], (255, 200, 50)),
            ("Giant Fiber (DNp01_R)", self.v_history["DNp01(GF)_R"], (50, 80, 255)),
            ("Motor TTMn (Jump)", self.v_history["TTMn_R (Jump)"], (50, 255, 100))
        ]

        ch_height = (height - 40) // len(channels)
        for idx, (label, hist, color) in enumerate(channels):
            cy = y_start + 45 + idx * ch_height
            cv2.putText(canvas, label, (x_start + 12, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
            
            # Baseline & threshold lines
            cv2.line(canvas, (x_start + 220, cy + 30), (x_start + width - 15, cy + 30), (50, 50, 60), 1)
            cv2.line(canvas, (x_start + 220, cy + 5), (x_start + width - 15, cy + 5), (70, 70, 85), 1) # V_th

            # Draw trace
            pts = []
            xs = np.linspace(x_start + 220, x_start + width - 15, self.history_len)
            for i, val in enumerate(hist):
                y_val = cy + 30 - int(val * 25)
                pts.append((int(xs[i]), y_val))
            
            if len(pts) > 1:
                cv2.polylines(canvas, [np.array(pts, np.int32)], False, color, 2)

    def run(self):
        print("\n[Start] Opening Camera (index 0)...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("[Warning] Could not open webcam 0. Running in Looming-Simulation Synthetic Mode.")
            use_cam = False
        else:
            use_cam = True
            print("[Info] Camera opened successfully. Wave your hand towards the camera to test escape reflex!")

        cv2.namedWindow("MaleCNS v1.0 - Drosophila Escape Reflex Simulation", cv2.WINDOW_AUTOSIZE)

        canvas_w, canvas_h = 1100, 700

        while True:
            # Create dark dashboard canvas
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            canvas[:] = (20, 20, 25)

            # Header
            cv2.putText(canvas, "MaleCNS v1.0: Whole-Brain Escape Circuit (LIF Model - UC Berkeley 2024)",
                        (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.line(canvas, (20, 45), (canvas_w - 20, 45), (60, 60, 70), 1)

            # 1. Camera Looming / Threat Detection
            if use_cam:
                ret, frame = cap.read()
                if ret:
                    frame_resized = cv2.resize(frame, (320, 240))
                    sensory_drive = self.detect_looming(frame_resized)
                    # Embed camera feed
                    canvas[60:300, 30:350] = frame_resized
                    cv2.rectangle(canvas, (30, 60), (350, 300), (80, 80, 90), 2)
                    cv2.putText(canvas, "WEBCAM LOOMING DETECTOR", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                else:
                    sensory_drive = 0.0
            else:
                # Synthetic pulse every 4 seconds
                t_sec = time.time() % 4.0
                sensory_drive = 2.0 if t_sec < 0.3 else 0.0
                cv2.rectangle(canvas, (30, 60), (350, 300), (35, 35, 45), -1)
                cv2.putText(canvas, "SYNTHETIC LOOMING INPUT", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

            # Threat meter
            cv2.putText(canvas, f"Looming Threat Energy: {sensory_drive:.2f}", (30, 325),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
            bar_w = int(min(sensory_drive / 2.5, 1.0) * 320)
            cv2.rectangle(canvas, (30, 335), (350, 350), (40, 40, 50), -1)
            bar_color = (0, 255, 100) if sensory_drive < 1.0 else ((0, 200, 255) if sensory_drive < 1.8 else (0, 0, 255))
            cv2.rectangle(canvas, (30, 335), (30 + bar_w, 350), bar_color, -1)

            # 2. Update Neural LIF Model
            spikes = self.update_lif(sensory_drive)

            # 3. Draw Fly Anatomical Structure & Motion
            fly_box_x = 720
            fly_box_y = 180
            cv2.rectangle(canvas, (540, 60), (canvas_w - 30, 360), (30, 30, 38), -1)
            cv2.rectangle(canvas, (540, 60), (canvas_w - 30, 360), (70, 70, 80), 1)
            cv2.putText(canvas, "DROSOPHILA MOTOR EMBODIMENT", (555, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 240), 1)
            self.draw_fly(canvas, fly_box_x, fly_box_y)

            # Circuit Pathway indicator in between
            cv2.putText(canvas, "CIRCUIT PATHWAY:", (375, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
            # Sensory Node
            s_col = (0, 255, 255) if sensory_drive > 0.5 else (100, 100, 100)
            cv2.circle(canvas, (450, 140), 16, s_col, -1)
            cv2.putText(canvas, "SENSORY", (422, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.35, s_col, 1)
            cv2.arrowedLine(canvas, (450, 180), (450, 210), (120, 120, 120), 2)
            
            # Giant Fiber Node
            gf_col = (0, 0, 255) if (spikes[5] or spikes[6]) else (80, 80, 100)
            cv2.circle(canvas, (450, 230), 18, gf_col, -1)
            cv2.putText(canvas, "GIANT FIBER", (412, 262), cv2.FONT_HERSHEY_SIMPLEX, 0.35, gf_col, 1)
            cv2.arrowedLine(canvas, (450, 275), (450, 305), (120, 120, 120), 2)
            
            # Motor Node
            m_col = (0, 255, 0) if (spikes[7] or spikes[8] or self.fly_state == "ESCAPING") else (80, 80, 100)
            cv2.circle(canvas, (450, 325), 16, m_col, -1)
            cv2.putText(canvas, "MOTOR/JUMP", (410, 352), cv2.FONT_HERSHEY_SIMPLEX, 0.35, m_col, 1)

            # 4. Oscilloscope (Bottom Section)
            self.draw_oscilloscope(canvas, 30, 380, canvas_w - 60, 290)

            # Instructions
            cv2.putText(canvas, "Press 'q' or ESC to Exit | Approach camera quickly to trigger escape jump",
                        (35, 688), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 130, 140), 1)

            cv2.imshow("MaleCNS v1.0 - Drosophila Escape Reflex Simulation", canvas)
            key = cv2.waitKey(15) & 0xFF
            if key == ord('q') or key == 27:
                break

        if use_cam:
            cap.release()
        cv2.destroyAllWindows()
        print("[Exit] Simulation closed.")

if __name__ == "__main__":
    sim = FlyEscapeVisualizer()
    sim.run()
