"""
Real-time OpenCV Drosophila Escape Reflex Simulation with Live Webcam Looming Detection.
Uses Leaky Integrate-and-Fire (LIF) dynamics with Berkeley (2024) biophysical parameters.
"""

import cv2
import numpy as np
import time
from collections import deque

class FlyEscapeVisualizer:
    def __init__(self):
        print("[Init] Initializing MaleCNS v1.0 Escape Reflex Model...")
        # Core Escape Reflex Circuit IDs & Labels:
        self.sensory_names = ["PVLP010_R", "PVLP122_R", "DNp70_L", "PVLP010_L", "DNp70_R"]
        self.gf_names = ["DNp01(GF)_R", "DNp01(GF)_L"]
        self.motor_names = ["TTMn_R (Jump Muscle)", "GFC2_L (Wing Motor)"]

        self.neurons = self.sensory_names + self.gf_names + self.motor_names
        self.N = len(self.neurons)

        # LIF Biophysical Parameters (Shiu et al. Nature 2024 / Berkeley)
        self.tau_m = 20.0       # ms membrane time constant
        self.V_rest = 0.0
        self.V_th = 1.0
        self.V_reset = 0.0
        self.dt = 1.0           # ms integration step
        self.ref_steps = 2      # refractory window
        
        self.V = np.zeros(self.N, dtype=np.float32)
        self.ref_counter = np.zeros(self.N, dtype=int)
        
        # Synaptic weight matrix: Sensory -> GF -> Motor
        self.W = np.zeros((self.N, self.N), dtype=np.float32)
        for s_idx in range(5):
            self.W[5, s_idx] = 0.50  # to GF_R
            self.W[6, s_idx] = 0.50  # to GF_L
        self.W[7, 5] = 0.95          # GF_R -> TTMn
        self.W[7, 6] = 0.95          # GF_L -> TTMn
        self.W[8, 5] = 0.85          # GF_R -> GFC2
        self.W[8, 6] = 0.85          # GF_L -> GFC2

        # Voltage History for multi-channel oscilloscope (200 points)
        self.history_len = 200
        self.v_history = {name: deque([0.0]*self.history_len, maxlen=self.history_len) for name in self.neurons}

        # Fly embodiment animation state
        self.fly_state = "IDLE"
        self.escape_timer = 0
        self.wing_angle = 20
        self.jump_y = 0

        # Background subtractor for robust looming detection
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25, detectShadows=False)
        self.looming_energy = 0.0

    def update_lif(self, sensory_drive):
        """Update LIF neurons with incoming sensory drive"""
        I_ext = np.zeros(self.N, dtype=np.float32)
        for i in range(5):
            I_ext[i] = sensory_drive * (0.9 + 0.3 * np.random.rand())

        spikes = np.zeros(self.N, dtype=bool)
        
        # 3 micro-steps per frame
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

        for i, name in enumerate(self.neurons):
            self.v_history[name].append(1.3 if spikes[i] else float(self.V[i]))

        # Escape Trigger on Giant Fiber firing
        if spikes[5] or spikes[6]:
            self.fly_state = "ESCAPING"
            self.escape_timer = 30

        return spikes

    def detect_looming(self, frame):
        """Detect looming motion in camera frame"""
        fg_mask = self.bg_sub.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        motion_ratio = np.count_nonzero(fg_mask) / (frame.shape[0] * frame.shape[1])
        
        # Looming activation threshold
        if motion_ratio > 0.04:
            drive = min(3.0, (motion_ratio - 0.04) * 12.0 + 0.7)
        else:
            drive = 0.0
        return drive, fg_mask

    def draw_fly(self, canvas, center_x, center_y):
        """Render biological fly structure and escape jump animation"""
        if self.fly_state == "ESCAPING":
            self.escape_timer -= 1
            # Upward ballistic jump trajectory
            self.jump_y = -int(np.sin((30 - self.escape_timer) / 30.0 * np.pi) * 85)
            self.wing_angle = 75 if (self.escape_timer % 4 < 2) else 15
            if self.escape_timer <= 0:
                self.fly_state = "IDLE"
                self.jump_y = 0
        else:
            self.wing_angle = 20
            self.jump_y = 0

        cx = center_x
        cy = center_y + self.jump_y

        # Shadow
        cv2.ellipse(canvas, (cx, center_y + 85), (55, 14), 0, 0, 360, (30, 30, 35), -1)

        # Wings
        wing_color = (210, 230, 245) if self.fly_state != "ESCAPING" else (80, 255, 255)
        # Left Wing
        pts_l = np.array([[cx - 15, cy - 10], [cx - 95, cy - int(self.wing_angle * 1.6)],
                          [cx - 120, cy - int(self.wing_angle * 1.3) + 25], [cx - 40, cy + 25]], np.int32)
        cv2.fillPoly(canvas, [pts_l], wing_color)
        cv2.polylines(canvas, [pts_l], True, (160, 190, 210), 2)

        # Right Wing
        pts_r = np.array([[cx + 15, cy - 10], [cx + 95, cy - int(self.wing_angle * 1.6)],
                          [cx + 120, cy - int(self.wing_angle * 1.3) + 25], [cx + 40, cy + 25]], np.int32)
        cv2.fillPoly(canvas, [pts_r], wing_color)
        cv2.polylines(canvas, [pts_r], True, (160, 190, 210), 2)

        # Thoracic Legs
        leg_col = (70, 75, 85) if self.fly_state != "ESCAPING" else (0, 180, 255)
        # Front legs
        cv2.line(canvas, (cx - 20, cy - 5), (cx - 65, cy - 40), leg_col, 3)
        cv2.line(canvas, (cx + 20, cy - 5), (cx + 65, cy - 40), leg_col, 3)
        # Middle legs (Jump / TTMn extensor)
        ext = 22 if self.fly_state != "ESCAPING" else 65
        cv2.line(canvas, (cx - 25, cy + 15), (cx - 85, cy + ext), leg_col, 4)
        cv2.line(canvas, (cx + 25, cy + 15), (cx + 85, cy + ext), leg_col, 4)
        # Hind legs
        cv2.line(canvas, (cx - 20, cy + 45), (cx - 70, cy + 80), leg_col, 3)
        cv2.line(canvas, (cx + 20, cy + 45), (cx + 70, cy + 80), leg_col, 3)

        # Abdomen
        cv2.ellipse(canvas, (cx, cy + 45), (28, 48), 0, 0, 360, (55, 70, 85), -1)
        for sy in range(cy + 20, cy + 85, 12):
            cv2.line(canvas, (cx - 22, sy), (cx + 22, sy), (35, 45, 55), 2)

        # Thorax
        cv2.circle(canvas, (cx, cy), 30, (75, 95, 115), -1)

        # Head & Eyes
        cv2.circle(canvas, (cx, cy - 40), 22, (60, 75, 90), -1)
        eye_col = (35, 40, 220) if self.fly_state != "ESCAPING" else (0, 120, 255)
        cv2.circle(canvas, (cx - 18, cy - 44), 12, eye_col, -1)
        cv2.circle(canvas, (cx + 18, cy - 44), 12, eye_col, -1)

        # Banner
        status_txt = "STATUS: SENSORY MONITORING (RESTING)" if self.fly_state != "ESCAPING" else ">>> ESCAPE REFLEX TRIGGERED (JUMP & FLAP) <<<"
        txt_col = (120, 255, 120) if self.fly_state != "ESCAPING" else (0, 0, 255)
        cv2.putText(canvas, status_txt, (cx - 200, cy - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.52, txt_col, 2)

    def draw_oscilloscope(self, canvas, x, y, w, h):
        """Draw live LIF membrane voltage oscilloscope"""
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (24, 25, 30), -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (70, 75, 85), 1)
        
        cv2.putText(canvas, "LIVE NEURON SPIKE & MEMBRANE POTENTIAL (LIF OSCILLOSCOPE)", (x + 15, y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 230, 255), 1)

        traces = [
            ("Sensory Neurons (PVLP010 / DNp70)", self.v_history["PVLP010_R"], (255, 200, 40)),
            ("Giant Fiber Command (DNp01_R)", self.v_history["DNp01(GF)_R"], (50, 90, 255)),
            ("Motor TTMn (Leg Jump Muscle)", self.v_history["TTMn_R (Jump Muscle)"], (60, 255, 120))
        ]

        th = (h - 40) // len(traces)
        for idx, (label, hist, col) in enumerate(traces):
            cy = y + 45 + idx * th
            cv2.putText(canvas, label, (x + 15, cy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
            
            # Baseline and threshold
            cv2.line(canvas, (x + 300, cy + 32), (x + w - 20, cy + 32), (50, 50, 60), 1)
            cv2.line(canvas, (x + 300, cy + 8), (x + w - 20, cy + 8), (70, 70, 85), 1)

            pts = []
            xs = np.linspace(x + 300, x + w - 20, self.history_len)
            for i, val in enumerate(hist):
                yv = cy + 32 - int(val * 24)
                pts.append((int(xs[i]), yv))
            if len(pts) > 1:
                cv2.polylines(canvas, [np.array(pts, np.int32)], False, col, 2)

    def run(self):
        print("\n[Start] Initializing Camera with DirectShow backend (cv2.CAP_DSHOW)...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print("[Fallback] DirectShow failed, trying default backend...")
            cap = cv2.VideoCapture(0)

        use_cam = cap.isOpened()
        if use_cam:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print("[Success] Camera is active! Wave your hand or approach the camera to trigger the escape reflex.")
        else:
            print("[Warning] No camera detected. Running in synthetic stimulation mode.")

        win_title = "MaleCNS v1.0 - Drosophila Escape Reflex Simulation"
        cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_title, 1200, 780)

        canvas_w, canvas_h = 1200, 780

        while True:
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            canvas[:] = (18, 19, 23)

            # Title Header
            cv2.putText(canvas, "MaleCNS v1.0: Whole-Brain Escape Circuit (LIF Model - UC Berkeley 2024)",
                        (25, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
            cv2.line(canvas, (25, 48), (canvas_w - 25, 48), (60, 65, 75), 1)

            # 1. Camera Looming Feed (Left Box: 440 x 330)
            if use_cam:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1) # Mirror for natural interaction
                    sensory_drive, fg_mask = self.detect_looming(frame)
                    
                    cam_view = cv2.resize(frame, (420, 315))
                    canvas[65:380, 30:450] = cam_view
                    cv2.rectangle(canvas, (30, 65), (450, 380), (80, 85, 95), 2)
                    cv2.putText(canvas, "LIVE CAMERA FEED (LOOMING SENSOR)", (45, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                else:
                    sensory_drive = 0.0
            else:
                sensory_drive = 2.0 if (time.time() % 4.0 < 0.3) else 0.0
                cv2.rectangle(canvas, (30, 65), (450, 380), (35, 35, 45), -1)
                cv2.putText(canvas, "SYNTHETIC LOOMING INPUT", (60, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1)

            # Threat meter
            cv2.putText(canvas, f"Looming Threat Intensity: {sensory_drive:.2f}", (30, 405),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1)
            bar_w = int(min(sensory_drive / 3.0, 1.0) * 420)
            cv2.rectangle(canvas, (30, 415), (450, 432), (40, 40, 50), -1)
            bar_col = (0, 255, 120) if sensory_drive < 1.0 else ((0, 200, 255) if sensory_drive < 2.0 else (0, 0, 255))
            cv2.rectangle(canvas, (30, 415), (30 + bar_w, 432), bar_col, -1)

            # 2. LIF Simulation Step
            spikes = self.update_lif(sensory_drive)

            # 3. Circuit Pathway Flow (Middle Column)
            mid_x = 540
            cv2.putText(canvas, "CIRCUIT:", (mid_x - 30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 200), 1)
            
            # Sensory Node
            s_col = (0, 255, 255) if sensory_drive > 0.4 else (90, 90, 100)
            cv2.circle(canvas, (mid_x, 150), 20, s_col, -1)
            cv2.putText(canvas, "Sensory", (mid_x - 25, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.38, s_col, 1)
            cv2.arrowedLine(canvas, (mid_x, 195), (mid_x, 230), (130, 130, 140), 2)

            # Giant Fiber Node
            gf_active = spikes[5] or spikes[6]
            gf_col = (0, 0, 255) if gf_active else (90, 90, 100)
            cv2.circle(canvas, (mid_x, 255), 22, gf_col, -1)
            cv2.putText(canvas, "Giant Fiber", (mid_x - 35, 292), cv2.FONT_HERSHEY_SIMPLEX, 0.38, gf_col, 1)
            cv2.arrowedLine(canvas, (mid_x, 302), (mid_x, 340), (130, 130, 140), 2)

            # Motor Output Node
            m_col = (0, 255, 100) if (spikes[7] or spikes[8] or self.fly_state == "ESCAPING") else (90, 90, 100)
            cv2.circle(canvas, (mid_x, 365), 20, m_col, -1)
            cv2.putText(canvas, "Motor/Jump", (mid_x - 35, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.38, m_col, 1)

            # 4. Fly Embodiment Box (Right Box)
            cv2.rectangle(canvas, (640, 65), (canvas_w - 30, 435), (28, 29, 36), -1)
            cv2.rectangle(canvas, (640, 65), (canvas_w - 30, 435), (70, 75, 85), 1)
            cv2.putText(canvas, "DROSOPHILA MOTOR EMBODIMENT", (660, 92),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 230, 250), 1)
            self.draw_fly(canvas, 915, 250)

            # 5. Bottom Oscilloscope
            self.draw_oscilloscope(canvas, 30, 455, canvas_w - 60, 275)

            # Footer instructions
            cv2.putText(canvas, "Quickly approach or wave your hand near camera to trigger looming reflex | Press 'q' or ESC to Exit",
                        (35, 762), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (140, 150, 160), 1)

            cv2.imshow(win_title, canvas)
            key = cv2.waitKey(12) & 0xFF
            if key == ord('q') or key == 27:
                break

        if use_cam:
            cap.release()
        cv2.destroyAllWindows()
        print("[Exit] Simulation window closed.")

if __name__ == "__main__":
    sim = FlyEscapeVisualizer()
    sim.run()
