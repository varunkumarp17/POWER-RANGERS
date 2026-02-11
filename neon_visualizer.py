import cv2
import numpy as np
import math
import time
import random
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

# ---------------- MEDIAPIPE ----------------
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
RunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)
landmarker = HandLandmarker.create_from_options(options)

# ---------------- SETTINGS ----------------
modes = ["BARS", "CIRCLE", "WAVE", "TUNNEL"]
mode_index = 0
mode_anim = 0

neon_colors = [
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
    (0, 255, 0),
    (255, 0, 0)
]
color_index = 0

energy = 0.4
phase = 0
last_pinch = 0
last_swipe = 0
prev_x = None

particles = []

# ---------------- HELPERS ----------------
def open_palm(hand):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    return all(hand[t].y < hand[p].y for t, p in zip(tips, pips))

def pinch(hand):
    t = hand[4]
    i = hand[8]
    return math.hypot(t.x - i.x, t.y - i.y) < 0.04

def spawn_particles(x, y, color):
    for _ in range(30):
        particles.append({
            "x": x,
            "y": y,
            "vx": random.uniform(-4, 4),
            "vy": random.uniform(-4, 4),
            "life": random.randint(20, 40),
            "size": random.randint(3, 6),
            "color": color
        })

# ---------------- MAIN LOOP ----------------
timestamp = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, timestamp)
    timestamp += 1

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        x = int(hand[8].x * w)
        y = int(hand[8].y * h)

        if open_palm(hand):
            energy = min(1.0, energy + 0.02)
        else:
            energy = max(0.3, energy - 0.01)

        if pinch(hand) and time.time() - last_pinch > 0.5:
            color_index = (color_index + 1) % len(neon_colors)
            spawn_particles(x, y, neon_colors[color_index])
            last_pinch = time.time()

        if prev_x:
            dx = x - prev_x
            if abs(dx) > 120 and time.time() - last_swipe > 0.6:
                mode_index = (mode_index + (1 if dx > 0 else -1)) % len(modes)
                mode_anim = 30
                last_swipe = time.time()
        prev_x = x

        phase += (x - w // 2) * 0.0003
    else:
        prev_x = None

    color = neon_colors[color_index]

    # ---------------- VISUAL MODES ----------------
    if modes[mode_index] == "TUNNEL":
        for i in range(40):
            depth = i / 40
            r = int((1 - depth) * energy * 400)
            thickness = int(6 * (1 - depth))
            cv2.circle(canvas, (w//2, h//2), r, color, thickness)

    elif modes[mode_index] == "BARS":
        for i in range(60):
            depth = i / 60
            x = int(i * w / 60)
            y = int((math.sin(i*0.4 + phase)+1) * energy * h * 0.4)
            cv2.line(canvas, (x, h//2-y), (x, h//2+y),
                     color, int(2 + 6*(1-depth)))

    elif modes[mode_index] == "CIRCLE":
        for i in range(80):
            angle = i * 2 * math.pi / 80
            r = int((math.sin(phase+i*0.3)+1) * energy * 250)
            cx = int(w//2 + r * math.cos(angle))
            cy = int(h//2 + r * math.sin(angle))
            cv2.circle(canvas, (cx, cy), 4, color, -1)

    elif modes[mode_index] == "WAVE":
        for x in range(0, w, 6):
            y = int(h//2 + math.sin(x*0.02 + phase) * energy * 250)
            cv2.circle(canvas, (x, y), 4, color, -1)

    # ---------------- PARTICLES ----------------
    for p in particles[:]:
        p["x"] += p["vx"]
        p["y"] += p["vy"]
        p["life"] -= 1
        if p["life"] <= 0:
            particles.remove(p)
            continue
        cv2.circle(canvas, (int(p["x"]), int(p["y"])),
                   p["size"], p["color"], -1)

    # Glow
    blur = cv2.GaussianBlur(canvas, (31, 31), 0)
    frame = cv2.addWeighted(frame, 0.25, blur, 0.75, 0)

    # Mode animation
    if mode_anim > 0:
        cv2.putText(frame, f"MODE: {modes[mode_index]}",
                    (w//2 - 160, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (255, 255, 255), 3)
        mode_anim -= 1

    cv2.imshow("Neon Visualizer PRO MAX", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



