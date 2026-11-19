"""
Pose Game - kiosk mode with a looping demo.

When started, the game plays itself in a loop (attract mode):

  * an animated player with a real skeleton (fixed bone lengths, bending
    elbows and knees) tries to match each animal pose, sometimes missing
    joints and correcting them, like a real person would,
  * round results are shown on top of the picture (no black screens),
  * every SCOREBOARD_EVERY seconds of play the top-5 leaderboard from
    highscores.json pops up for SCOREBOARD_SHOW seconds,
  * after the last level the playlist is shuffled and the demo starts again.

To play for real, start typing a name (or press Enter) and press Enter to
start. The camera game runs through the playlist, saves the score to
highscores.json, shows the leaderboard and then goes back to the demo.
Demo scores are never saved.

Usage:
    python pose_game.py                           # Christmas playlist (default)
    python pose_game.py annotations               # every level in annotations/
    python pose_game.py playlist_1.json           # the animal playlist
    python pose_game.py playlist_2.json           # shorter animal playlist
    python pose_game.py annotations_animals       # every level in a folder
    python pose_game.py --fullscreen              # start in fullscreen

Keys:
    demo:        type a name / Enter - play    Esc - quit    F11 - fullscreen
    name entry:  Enter - start    Backspace - delete    Esc - back to demo
    game:        Esc or Q - stop and go back to the demo
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # quieter MediaPipe start-up

import cv2
import numpy as np
import glob
import json
import math
import random
import sys
import threading
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANNOTATION_DIRS = ["annotations", "annotations_animals"]
IMAGE_DIRS = [os.path.join("images", "animals"), "images"]
HIGHSCORES_FILE = os.path.join(BASE_DIR, "highscores.json")
MODEL_PATH = os.path.join(BASE_DIR, "pose_landmarker_lite.task")
DEFAULT_LEVELS = "playlist.json"   # Christmas playlist (or a folder) played by default
CAMERA_INDEX = 0

WINDOW = "Pose Game"
W, H = 640, 480                # same canvas size as the real game
WINDOW_SIZE = (960, 720)       # initial window size (window is resizable)
FPS = 30

MATCH_THRESHOLD = 0.12         # max distance for a joint to count as matched
ROUND_DURATION = 10            # seconds per level
RESULT_DURATION = 3            # how long a round result stays on screen
FINAL_BOARD_DURATION = 8       # leaderboard after a real game
SCOREBOARD_EVERY = 5           # seconds of demo play between leaderboard pop-ups
SCOREBOARD_SHOW = 2            # how long the leaderboard stays visible
NAME_IDLE_TIMEOUT = 30         # back to the demo if nobody types for this long
MAX_NAME = 15
TOP_N = 5

KEY_ENTER, KEY_ESC, KEY_BACKSPACE, KEY_SPACE = 13, 27, 8, 32
KEY_F11 = 0x7A0000             # cv2.waitKeyEx code on Windows

POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),
    (17,19),(12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32)
]


# --------------------------------------------------------------------------
# Loading levels and scores
# --------------------------------------------------------------------------

def find_file(name, dirs):
    if os.path.isabs(name):
        return name if os.path.isfile(name) else None
    for d in [""] + dirs:
        p = os.path.join(BASE_DIR, d, name)
        if os.path.isfile(p):
            return p
    return None


def load_levels(arg):
    path = arg if os.path.isabs(arg) else os.path.join(BASE_DIR, arg)
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.json")))
    else:
        found = find_file(arg, ANNOTATION_DIRS)
        if not found:
            print(f"Could not find {arg}")
            return []
        with open(found) as f:
            data = json.load(f)
        if isinstance(data, list):
            files = []
            for name in data:
                p = find_file(name, ANNOTATION_DIRS)
                if p:
                    files.append(p)
                else:
                    print(f"Could not find level {name}, skipping.")
        else:
            files = [found]

    levels = []
    for ann_file in files:
        with open(ann_file) as f:
            data = json.load(f)
        if "image" not in data or "landmarks" not in data:
            continue
        image_name = os.path.basename(data["image"])
        image_path = find_file(image_name, IMAGE_DIRS)
        img = cv2.imread(image_path) if image_path else None
        if img is None:
            print(f"Could not load image {image_name}, skipping.")
            continue
        levels.append({
            "name": os.path.splitext(os.path.basename(ann_file))[0],
            "image": cv2.resize(img, (W, H)),
            "landmarks": {int(k): v for k, v in data["landmarks"].items()},
        })
    return levels


def load_highscores():
    try:
        with open(HIGHSCORES_FILE) as f:
            scores = json.load(f)
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:TOP_N]
    except (OSError, ValueError, KeyError, TypeError):
        return []


def update_highscores(name, score):
    scores = load_highscores()
    scores.append({"name": name, "score": score})
    scores.sort(key=lambda x: x["score"], reverse=True)
    scores = scores[:TOP_N]
    with open(HIGHSCORES_FILE, "w") as f:
        json.dump(scores, f, indent=2)
    return scores


def calculate_score(matched, total):
    if total == 0 or matched == 0:
        return 0
    ratio = matched / total
    multiplier = 2.0 if ratio == 1.0 else (1.5 if ratio >= 0.5 else 1.0)
    return int(matched * multiplier)


def count_matched(full, landmarks):
    if full is None:
        return 0
    matched = 0
    for i, ann in landmarks.items():
        if 0 <= i < len(full):
            p = full[i]
            if math.hypot(p[0] - ann["x"], p[1] - ann["y"]) < MATCH_THRESHOLD:
                matched += 1
    return matched


# --------------------------------------------------------------------------
# Skeleton model of the demo player
#
# The body is described by a few parameters:
#   x, y  - hip centre in the image          s  - size (distance to camera)
#   rot   - torso lean                       f  - facing (+1 front, -1 back,
#   nk    - head tilt                             0 = turned sideways)
# Wrists are aimed in body coordinates and ankles in image coordinates;
# elbows and knees are solved with two-bone inverse kinematics, so bones
# keep their length and joints bend naturally.
# --------------------------------------------------------------------------

HIP_W, SHOULDER_W, TORSO, NECK = 0.04, 0.07, 0.25, 0.13
UPPER_ARM, FOREARM, THIGH, SHIN = 0.12, 0.11, 0.16, 0.16

LOCAL = {11: (SHOULDER_W, -TORSO), 12: (-SHOULDER_W, -TORSO),
         23: (HIP_W, 0.0), 24: (-HIP_W, 0.0)}
REST_WRIST = {15: (SHOULDER_W + 0.035, -TORSO + UPPER_ARM + FOREARM - 0.01),
              16: (-SHOULDER_W - 0.035, -TORSO + UPPER_ARM + FOREARM - 0.01)}
ARMS = ((11, 13, 15), (12, 14, 16))
LEGS = ((23, 25, 27), (24, 26, 28))
TORSO_IDS = (11, 12, 23, 24)

FACE_OFFSETS = {
    1: (0.008, -0.015), 2: (0.015, -0.016), 3: (0.022, -0.015),
    4: (-0.008, -0.015), 5: (-0.015, -0.016), 6: (-0.022, -0.015),
    7: (0.035, -0.005), 8: (-0.035, -0.005),
    9: (0.010, 0.020), 10: (-0.010, 0.020),
}


def rot2(a, v):
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def unit(v):
    n = math.hypot(v[0], v[1])
    return v / n if n > 1e-6 else np.zeros(2)


def to_world(p, local):
    return (np.array([p["x"], p["y"]])
            + p["s"] * rot2(p["rot"], (local[0] * p["f"], local[1])))


def to_local(p, world):
    d = rot2(-p["rot"], (np.asarray(world, float) - (p["x"], p["y"])) / p["s"])
    f = p["f"] if abs(p["f"]) > 0.05 else math.copysign(0.05, p["f"] or 1.0)
    return np.array([d[0] / f, d[1]])


def heel_offset(side, p):
    return p["s"] * np.array([-0.008 * side * p["f"], 0.02])


def toe_offset(side, p):
    return p["s"] * np.array([0.03 * side * p["f"], 0.025])


def rest_ankle(p, hip):
    """Foot on the ground straight below the hip."""
    h = to_world(p, LOCAL[hip])
    dx = h[0] - p["x"]
    return np.array([h[0] + 0.3 * dx, h[1] + p["s"] * (THIGH + SHIN - 0.005)])


def ik(root, goal, l1, l2, outward, amount):
    """Two-bone IK: returns (middle joint, end joint).

    The middle joint bends towards `outward`; when the limb points along
    that direction it is drawn straight (the joint points at the camera),
    which keeps the motion continuous.
    """
    d = goal - root
    dist = math.hypot(d[0], d[1])
    u = d / dist if dist > 1e-6 else np.array([0.0, 1.0])
    dc = min(max(dist, abs(l1 - l2) + 1e-4), (l1 + l2) * 0.999)
    a = (l1 * l1 - l2 * l2 + dc * dc) / (2 * dc)
    h = math.sqrt(max(l1 * l1 - a * a, 0.0))
    perp = np.array([-u[1], u[0]])
    b = max(-1.0, min(1.0, 2.0 * float(perp @ outward))) * amount
    return root + u * a + perp * h * b, root + u * dc


def build_pose(p, wrists_local, ankles_world):
    s, rot = p["s"], p["rot"]
    core = {i: to_world(p, loc) for i, loc in LOCAL.items()}
    hip_c = np.array([p["x"], p["y"]])
    sh_c = to_world(p, (0.0, -TORSO))
    a = rot + p["nk"]
    core[0] = sh_c + s * NECK * np.array([math.sin(a), -math.cos(a)])
    down = np.array([-math.sin(rot), math.cos(rot)])
    for sh, el, wr in ARMS:
        out = unit(unit(core[sh] - sh_c) + 0.6 * down)
        core[el], core[wr] = ik(core[sh], to_world(p, wrists_local[wr]),
                                s * UPPER_ARM, s * FOREARM, out, 1.0)
    for hip, kn, an in LEGS:
        out = unit(core[hip] - hip_c)
        core[kn], core[an] = ik(core[hip], ankles_world[an],
                                s * THIGH, s * SHIN, out, 0.35)
    return core


def derive_full(core, p):
    """All 33 MediaPipe landmarks: face, hands and feet follow the core."""
    full = [None] * 33
    for i, v in core.items():
        full[i] = v
    s, f = p["s"], p["f"]
    fs = 1.0 if f >= 0 else -1.0
    head = p["rot"] + p["nk"]
    for i, (ox, oy) in FACE_OFFSETS.items():
        full[i] = core[0] + s * rot2(head, (ox * f, oy))
    for wrist, elbow, pinky, index, thumb, side in ((15, 13, 17, 19, 21, 1),
                                                    (16, 14, 18, 20, 22, -1)):
        d = unit(core[wrist] - core[elbow])
        if not d.any():
            d = np.array([0.0, 1.0])
        perp = np.array([-d[1], d[0]]) * side * fs
        full[pinky] = core[wrist] + d * 0.035 * s + perp * 0.012 * s
        full[index] = core[wrist] + d * 0.04 * s - perp * 0.006 * s
        full[thumb] = core[wrist] + d * 0.02 * s - perp * 0.02 * s
    for ankle, heel, toe, side in ((27, 29, 31, 1), (28, 30, 32, -1)):
        full[heel] = core[ankle] + heel_offset(side, p)
        full[toe] = core[ankle] + toe_offset(side, p)
    return full


def limb_goals(p, ann):
    """Where the limbs should go for body parameters p."""
    wl = {}
    for wr, hand in ((15, (17, 19, 21)), (16, (18, 20, 22))):
        g = ann.get(wr)
        if g is None:
            pts = [ann[i] for i in hand if i in ann]
            g = np.mean(pts, axis=0) if pts else None
        wl[wr] = to_local(p, g) if g is not None else np.array(REST_WRIST[wr])
    aw = {}
    for an, hip, heel, toe, side in ((27, 23, 29, 31, 1), (28, 24, 30, 32, -1)):
        if an in ann:
            aw[an] = ann[an]
        elif heel in ann:
            aw[an] = ann[heel] - heel_offset(side, p)
        elif toe in ann:
            aw[an] = ann[toe] - toe_offset(side, p)
        else:
            aw[an] = rest_ankle(p, hip)
    nk = 0.0
    if 0 in ann:
        d = ann[0] - to_world(p, (0.0, -TORSO))
        a = math.atan2(d[0], -d[1]) - p["rot"]
        a = (a + math.pi) % (2 * math.pi) - math.pi
        nk = max(-0.4, min(0.4, a))
    return wl, aw, nk


def _fit_cost(p, ann):
    wl, aw, nk = limb_goals(p, ann)
    p["nk"] = nk
    full = derive_full(build_pose(p, wl, aw), p)
    cost = 0.0
    for i, t in ann.items():
        if 0 <= i < 33:
            cost += min(math.hypot(*(full[i] - t)), 0.3) ** 2
    cost += (0.02 * p["rot"] ** 2 + 0.03 * math.log(p["s"]) ** 2
             + 0.005 * nk ** 2 + (0.003 if p["f"] < 0 else 0.0))
    return cost, wl, aw


def fit_level(landmarks):
    """Best pose a human body can make for a level's target landmarks."""
    ann = {i: np.array([v["x"], v["y"]], float) for i, v in landmarks.items()}
    torso = [i for i in TORSO_IDS if i in ann]

    def place(p):
        if torso:
            off = np.mean([ann[i] - to_world(p, LOCAL[i]) for i in torso], axis=0)
        elif 0 in ann:
            off = ann[0] - to_world(p, (0.0, -TORSO - NECK)) + (p["x"], p["y"])
        else:
            off = np.array([0.5, 0.58])
        p["x"], p["y"] = float(off[0]), float(off[1])

    best = None
    for f in (1.0, -1.0):
        for rot in np.linspace(-0.45, 0.45, 11):
            for s in np.geomspace(0.6, 1.7, 12):
                p = {"x": 0.0, "y": 0.0, "s": float(s), "rot": float(rot),
                     "f": f, "nk": 0.0}
                place(p)
                cost = _fit_cost(p, ann)[0]
                if best is None or cost < best[0]:
                    best = (cost, p)

    # Small random refinement around the best grid point.
    cost, p = best
    rng = random.Random(0)
    for _ in range(150):
        q = dict(p)
        q["x"] += rng.gauss(0, 0.015)
        q["y"] += rng.gauss(0, 0.015)
        q["s"] = min(1.7, max(0.6, q["s"] * math.exp(rng.gauss(0, 0.04))))
        q["rot"] = min(0.45, max(-0.45, q["rot"] + rng.gauss(0, 0.04)))
        c = _fit_cost(q, ann)[0]
        if c < cost:
            cost, p = c, q

    _, wl, aw = _fit_cost(p, ann)
    targeted = set()
    if ann.keys() & {15, 17, 19, 21}:
        targeted.add(15)
    if ann.keys() & {16, 18, 20, 22}:
        targeted.add(16)
    if ann.keys() & {27, 29, 31}:
        targeted.add(27)
    if ann.keys() & {28, 30, 32}:
        targeted.add(28)
    return {"p": p, "wl": wl, "aw": aw, "targeted": targeted}


class Spring:
    """Critically damped spring: smooth, human-like ease in and out."""

    def __init__(self, value, omega):
        self.x = np.array(value, float)
        self.v = np.zeros_like(self.x)
        self.target = self.x.copy()
        self.omega = omega

    def step(self, dt):
        w = self.omega
        a = w * w * (self.target - self.x) - 2 * w * self.v
        self.v += a * dt
        self.x += self.v * dt


class FakePlayer:
    def __init__(self):
        p = {"x": 0.5, "y": 0.58, "s": 1.0, "rot": 0.0, "f": 1.0, "nk": 0.0}
        self.pos = Spring((p["x"], p["y"]), 2.2)
        self.size = Spring(p["s"], 2.0)
        self.lean = Spring(p["rot"], 2.5)
        self.facing = Spring(p["f"], 1.8)
        self.neck = Spring(p["nk"], 4.0)
        self.wrists = {k: Spring(v, 4.5) for k, v in REST_WRIST.items()}
        self.ankles = {an: Spring(rest_ankle(p, hip), 3.2)
                       for hip, _, an in LEGS}
        self.springs = ([self.pos, self.size, self.lean, self.facing, self.neck]
                        + list(self.wrists.values()) + list(self.ankles.values()))
        self.t = 0.0
        self.events = []
        self.wave = [random.uniform(0, 2 * math.pi) for _ in range(8)]

    # -- goals -------------------------------------------------------------
    def _set_goal(self, p, wl, aw):
        self.pos.target = np.array([p["x"], p["y"]])
        self.size.target = np.array(p["s"])
        self.lean.target = np.array(p["rot"])
        self.facing.target = np.array(p["f"])
        self.neck.target = np.array(p["nk"])
        for k, v in wl.items():
            self.wrists[k].target = np.array(v, float)
        for k, v in aw.items():
            self.ankles[k].target = np.array(v, float)

    def begin_round(self, fit):
        """Plan how this round's attempt will go."""
        self.t = 0.0
        skill = random.uniform(0.5, 1.0)
        body_err = {"x": random.gauss(0, 0.03), "y": random.gauss(0, 0.02),
                    "s": random.gauss(0, 0.06), "rot": random.gauss(0, 0.08),
                    "nk": random.gauss(0, 0.12)}
        limb_err = {}
        for k in (15, 16, 27, 28):
            ang = random.uniform(0, 2 * math.pi)
            if k in fit["targeted"] and random.random() > skill:
                mag, fixable = random.uniform(0.12, 0.2), random.random() < 0.6
            else:
                mag, fixable = random.uniform(0.005, 0.025), False
            if k in (27, 28):
                mag *= 0.7
            limb_err[k] = (np.array([math.cos(ang), math.sin(ang)]) * mag, fixable)

        react = random.uniform(0.3, 0.9)      # time to look at the picture
        fix = random.uniform(3.5, 6.5)        # watches the skeleton and adjusts
        self.events = [
            (react, lambda: self._aim(fit, body_err, limb_err, 1.0, False)),
            (fix, lambda: self._aim(fit, body_err, limb_err, 0.35, True)),
        ]

    def _aim(self, fit, body_err, limb_err, strength, corrected):
        p = dict(fit["p"])
        p["x"] += body_err["x"] * strength
        p["y"] += body_err["y"] * strength
        p["s"] *= math.exp(body_err["s"] * strength)
        p["rot"] += body_err["rot"] * strength
        p["nk"] += body_err["nk"] * strength
        wl, aw = {}, {}
        for k, goal in fit["wl"].items():
            err, fixable = limb_err[k]
            wl[k] = goal + err * (0.15 if corrected and fixable else 1.0)
        for hip, _, k in LEGS:
            err, fixable = limb_err[k]
            base = fit["aw"][k] if k in fit["targeted"] else rest_ankle(p, hip)
            aw[k] = base + err * (0.15 if corrected and fixable else 1.0)
        self._set_goal(p, wl, aw)

    def relax(self, delay=0.6):
        """Drop the arms and stand normally (end of a round)."""
        def go():
            p = {"x": float(self.pos.target[0]), "y": float(self.pos.target[1]),
                 "s": float(self.size.target), "rot": 0.0, "f": 1.0, "nk": 0.0}
            self._set_goal(p, {k: np.array(v) for k, v in REST_WRIST.items()},
                           {an: rest_ankle(p, hip) for hip, _, an in LEGS})
        self.events = [(self.t + delay, go)]

    # -- animation ---------------------------------------------------------
    def update(self, dt):
        self.t += dt
        due = [e for e in self.events if e[0] <= self.t]
        self.events = [e for e in self.events if e[0] > self.t]
        for _, fn in due:
            fn()
        steps = max(1, int(math.ceil(dt * 120)))
        for _ in range(steps):
            for sp in self.springs:
                sp.step(dt / steps)

    def landmarks(self):
        t, w = self.t, self.wave
        p = {"x": float(self.pos.x[0]) + 0.004 * math.sin(0.8 * t + w[0]),  # weight shift
             "y": float(self.pos.x[1]) + 0.002 * math.sin(1.7 * t + w[1]),  # breathing
             "s": max(0.3, float(self.size.x)),
             "rot": float(self.lean.x) + 0.01 * math.sin(0.6 * t + w[2]),
             "f": max(-1.0, min(1.0, float(self.facing.x))),
             "nk": float(self.neck.x) + 0.03 * math.sin(0.7 * t + w[3])}
        wl = {k: sp.x + 0.005 * np.array([math.sin(1.3 * t + w[4 + j]),
                                          math.sin(1.1 * t + w[6 + j])])
              for j, (k, sp) in enumerate(self.wrists.items())}
        aw = {}
        for k, sp in self.ankles.items():
            lift = min(0.05, 0.12 * math.hypot(*sp.v)) * p["s"]   # steps lift the foot
            aw[k] = sp.x - (0.0, lift)
        return derive_full(build_pose(p, wl, aw), p)


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------

def draw_animal_joints(frame, landmarks):
    for ann in landmarks.values():
        px, py = int(ann["x"] * W), int(ann["y"] * H)
        cv2.circle(frame, (px, py), 7, (0, 220, 255), -1)
        cv2.putText(frame, ann["name"], (px + 9, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)


def draw_user_skeleton(frame, full, landmarks):
    if full is None:
        return
    points = [(int(p[0] * W), int(p[1] * H)) for p in full]
    for a, b in POSE_CONNECTIONS:
        if a < len(points) and b < len(points):
            cv2.line(frame, points[a], points[b], (200, 200, 200), 2)
    for i, ann in landmarks.items():
        if not 0 <= i < len(full):
            continue
        p = full[i]
        dist = math.hypot(p[0] - ann["x"], p[1] - ann["y"])
        color = (0, 255, 0) if dist < MATCH_THRESHOLD else (0, 0, 255)
        cv2.circle(frame, points[i], 7, color, -1)


def put_center(frame, text, y, scale, color, thickness):
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.putText(frame, text, ((W - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness)


def dark_box(frame, x1, y1, x2, y2, alpha=0.7, border=True):
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
    roi = frame[y1:y2, x1:x2]
    frame[y1:y2, x1:x2] = (roi * (1 - alpha)).astype(np.uint8)
    if border:
        cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), (255, 200, 0), 2)


def render_play(level, full, matched, best, total_score, img_num, n_images,
                remaining, label, blink, hint):
    frame = level["image"].copy()
    landmarks = level["landmarks"]
    draw_animal_joints(frame, landmarks)
    draw_user_skeleton(frame, full, landmarks)

    timer_color = (0, 0, 255) if remaining <= 10 else (0, 220, 255)
    cv2.putText(frame, f"Image {img_num}/{n_images}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"{int(math.ceil(remaining))}s",
                (W - 80, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, timer_color, 3)
    cv2.putText(frame, f"Joints: {matched}/{len(landmarks)}  Best: {best}",
                (10, H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Score: {total_score}",
                (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
    if label and (not blink or int(time.time() * 2) % 2 == 0):
        put_center(frame, label, 32, 0.9, (0, 0, 255) if blink else (255, 255, 255), 2)
    (tw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(frame, hint, (W - tw - 10, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    return frame


def draw_play_prompt(frame):
    dark_box(frame, 110, 44, W - 110, 74, alpha=0.6, border=False)
    put_center(frame, "Type your name and press ENTER to play", 65, 0.55, (0, 255, 255), 1)


def draw_name_entry(frame, name):
    dark_box(frame, 80, 130, W - 80, 350, alpha=0.8)
    put_center(frame, "ENTER YOUR NAME", 185, 1.1, (255, 200, 0), 3)
    cursor = "_" if int(time.time() * 2) % 2 == 0 else " "
    put_center(frame, name + cursor, 255, 1.0, (255, 255, 255), 2)
    put_center(frame, "ENTER: start   Backspace: delete   Esc: back", 325,
               0.5, (200, 200, 200), 1)


def draw_message(frame, text):
    dark_box(frame, 90, 200, W - 90, 280, alpha=0.8)
    put_center(frame, text, 250, 0.75, (255, 255, 255), 2)


def draw_round_result(frame, round_score, matched, total, cumulative, img_num, n_images):
    ratio = matched / total if total > 0 else 0
    mult = "x2.0" if ratio == 1.0 else ("x1.5" if ratio >= 0.5 else "x1.0")
    dark_box(frame, 70, 110, W - 70, 370)
    put_center(frame, f"Round {img_num}/{n_images} complete!", 165, 0.95, (255, 255, 255), 2)
    put_center(frame, f"Best match: {matched}/{total}  (multiplier {mult})", 225, 0.7, (0, 220, 255), 2)
    put_center(frame, f"Round score: {round_score}", 285, 0.9, (0, 255, 0), 2)
    put_center(frame, f"Total score: {cumulative}", 340, 0.9, (255, 200, 0), 2)


def draw_scoreboard(frame, scores, highlight=None, footer=None):
    rows = max(len(scores), 1)
    top = 85
    bottom = top + 80 + rows * 45 + (40 if footer else 0)
    dark_box(frame, W // 2 - 210, top, W // 2 + 210, bottom, alpha=0.75)
    put_center(frame, "LEADERBOARD", top + 50, 1.2, (255, 200, 0), 3)
    if not scores:
        put_center(frame, "No scores yet", top + 100, 0.8, (255, 255, 255), 2)
    marked = False
    for i, entry in enumerate(scores):
        is_player = (not marked and highlight is not None
                     and (entry["name"], entry["score"]) == highlight)
        marked = marked or is_player
        put_center(frame, f"{i + 1}.  {entry['name']}  -  {entry['score']} pts",
                   top + 100 + i * 45, 0.8,
                   (0, 255, 255) if is_player else (255, 255, 255), 2)
    if footer:
        put_center(frame, footer, bottom - 20, 0.6, (200, 200, 200), 1)


def draw_run_complete(frame, total_score):
    dark_box(frame, 70, 140, W - 70, 340)
    put_center(frame, "Demo run complete!", 195, 1.0, (255, 255, 255), 2)
    put_center(frame, f"Total score: {total_score}", 255, 0.95, (255, 200, 0), 2)
    put_center(frame, "Starting again...", 310, 0.7, (0, 220, 255), 2)


# --------------------------------------------------------------------------
# Demo loop and real game
# --------------------------------------------------------------------------

def scoreboard_visible(play_clock):
    return play_clock % (SCOREBOARD_EVERY + SCOREBOARD_SHOW) >= SCOREBOARD_EVERY


class Demo:
    HINT = "Esc: quit  F11: fullscreen"

    def __init__(self, levels):
        self.order = list(levels)
        self.player = FakePlayer()
        self.fits = {}
        self.play_clock = 0.0
        self.scores = load_highscores()
        self.board_on = False
        self.idx = 0
        self.total = 0
        self._start_round()

    def _start_round(self):
        level = self.order[self.idx]
        if level["name"] not in self.fits:
            self.fits[level["name"]] = fit_level(level["landmarks"])
        self.player.begin_round(self.fits[level["name"]])
        self.phase, self.t, self.best, self.round_score = "play", 0.0, 0, 0

    def step(self, dt):
        self.t += dt
        self.player.update(dt)
        level = self.order[self.idx]
        landmarks = level["landmarks"]
        full = self.player.landmarks()
        matched = count_matched(full, landmarks)
        n = len(self.order)

        if self.phase == "play":
            self.best = max(self.best, matched)
            self.play_clock += dt
            remaining = max(0.0, ROUND_DURATION - self.t)
            frame = render_play(level, full, matched, self.best, self.total,
                                self.idx + 1, n, remaining, "", False, self.HINT)
            if scoreboard_visible(self.play_clock):
                if not self.board_on:
                    self.scores = load_highscores()
                    self.board_on = True
                draw_scoreboard(frame, self.scores)
            else:
                self.board_on = False
            if self.t >= ROUND_DURATION:
                self.round_score = calculate_score(self.best, len(landmarks))
                self.total += self.round_score
                self.phase, self.t = "result", 0.0
                self.player.relax()
        elif self.phase == "result":
            frame = render_play(level, full, matched, self.best, self.total,
                                self.idx + 1, n, 0, "", False, self.HINT)
            draw_round_result(frame, self.round_score, self.best, len(landmarks),
                              self.total, self.idx + 1, n)
            if self.t >= RESULT_DURATION:
                if self.idx + 1 < n:
                    self.idx += 1
                    self._start_round()
                else:
                    self.phase, self.t = "end", 0.0
        else:
            frame = render_play(level, full, matched, self.best, self.total,
                                self.idx + 1, n, 0, "", False, self.HINT)
            draw_run_complete(frame, self.total)
            if self.t >= RESULT_DURATION:
                random.shuffle(self.order)
                self.idx, self.total = 0, 0
                self._start_round()
        return frame


def load_pose_model():
    """MediaPipe pose landmarker for the camera game, or None if unavailable."""
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH))
        return mp, vision.PoseLandmarker.create_from_options(options)
    except Exception as e:   # missing package or model file
        print(f"Camera game disabled: {e}")
        return None


class RealGame:
    HINT = "Esc / Q: stop game"

    def __init__(self, levels, name, pose_model, cap):
        self.levels, self.name, self.cap = levels, name, cap
        self.mp, self.detector = pose_model
        self.idx, self.total = 0, 0
        self.phase, self.t, self.best, self.round_score = "play", 0.0, 0, 0
        self.full = None
        self.scores = []
        self.done = False

    def step(self, dt):
        self.t += dt
        level = self.levels[self.idx]
        landmarks = level["landmarks"]
        n = len(self.levels)

        if self.phase == "play":
            ret, cam = self.cap.read()
            if not ret:
                self.done = True
                frame = level["image"].copy()
                draw_message(frame, "Camera stopped")
                return frame
            rgb = cv2.cvtColor(cv2.flip(cam, 1), cv2.COLOR_BGR2RGB)
            result = self.detector.detect(
                self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
            self.full = ([np.array([lm.x, lm.y]) for lm in result.pose_landmarks[0]]
                         if result.pose_landmarks else None)
            matched = count_matched(self.full, landmarks)
            self.best = max(self.best, matched)
            remaining = max(0.0, ROUND_DURATION - self.t)
            frame = render_play(level, self.full, matched, self.best, self.total,
                                self.idx + 1, n, remaining, self.name, False, self.HINT)
            if self.t >= ROUND_DURATION:
                self.round_score = calculate_score(self.best, len(landmarks))
                self.total += self.round_score
                self.phase, self.t = "result", 0.0
        elif self.phase == "result":
            matched = count_matched(self.full, landmarks)
            frame = render_play(level, self.full, matched, self.best, self.total,
                                self.idx + 1, n, 0, self.name, False, self.HINT)
            draw_round_result(frame, self.round_score, self.best, len(landmarks),
                              self.total, self.idx + 1, n)
            if self.t >= RESULT_DURATION:
                if self.idx + 1 < n:
                    self.idx += 1
                    self.phase, self.t, self.best = "play", 0.0, 0
                else:
                    self.scores = update_highscores(self.name, self.total)
                    self.phase, self.t = "board", 0.0
        else:
            frame = level["image"].copy()
            draw_scoreboard(frame, self.scores, (self.name, self.total),
                            f"{self.name}: {self.total} pts  -  press any key")
            if self.t >= FINAL_BOARD_DURATION:
                self.done = True
        return frame

    def close(self):
        self.cap.release()


class Display:
    def __init__(self, fullscreen):
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, *WINDOW_SIZE)
        self.fullscreen = False
        if fullscreen:
            self.toggle_fullscreen()
        self.last = time.time()

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if self.fullscreen
                              else cv2.WINDOW_NORMAL)

    def show(self, frame):
        """Show a frame at about FPS. Returns the key pressed (-1 for none),
        or None when the window was closed."""
        cv2.imshow(WINDOW, frame)
        wait = max(1, int((1.0 / FPS - (time.time() - self.last)) * 1000))
        key = cv2.waitKeyEx(wait)
        self.last = time.time()
        try:
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return None
        except cv2.error:
            return None
        return key


class PoseModelLoader:
    """Loads MediaPipe in the background (takes ~15 s) while the demo plays."""

    def __init__(self):
        self.model = None
        self.thread = threading.Thread(target=self._load, daemon=True)
        self.thread.start()

    def _load(self):
        self.model = load_pose_model()

    def wait(self, display, frame):
        """Return the model, keeping the window responsive while it loads."""
        if self.thread.is_alive():
            waiting = frame.copy()
            draw_message(waiting, "Loading camera game...")
            while self.thread.is_alive():
                display.show(waiting)
                self.thread.join(0.05)
        return self.model


def start_game(display, frame, levels, name, loader):
    pose_model = loader.wait(display, frame)
    if pose_model is None:
        return None, "Camera game is not available"
    waiting = frame.copy()
    draw_message(waiting, "Starting camera...")
    display.show(waiting)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        cap.release()
        return None, "Camera not found"
    return RealGame(levels, name, pose_model, cap), ""


def run():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    levels = load_levels(args[0] if args else DEFAULT_LEVELS)
    if not levels:
        print("No levels found.")
        return
    print(f"Loaded {len(levels)} levels. Press Esc in the window to quit.")
    loader = PoseModelLoader()

    display = Display("--fullscreen" in sys.argv)
    demo = Demo(levels)
    state, name, game = "demo", "", None
    message, message_until = "", 0.0
    last = last_key = time.time()

    while True:
        now = time.time()
        dt = min(now - last, 0.1)
        last = now

        if state == "game":
            frame = game.step(dt)
            if game.done:
                game.close()
                game, state = None, "demo"
        else:
            frame = demo.step(dt)
            if state == "name":
                draw_name_entry(frame, name)
                if now - last_key > NAME_IDLE_TIMEOUT:
                    state = "demo"
            else:
                draw_play_prompt(frame)
            if now < message_until:
                draw_message(frame, message)

        key = display.show(frame)
        if key is None:
            break
        if key == -1:
            continue
        if key == KEY_F11:
            display.toggle_fullscreen()
            continue
        last_key = time.time()

        if state == "demo":
            if key == KEY_ESC:
                break
            if key in (KEY_ENTER, KEY_SPACE) or 33 <= key <= 126:
                name = chr(key) if 33 <= key <= 126 else ""
                state = "name"
        elif state == "name":
            if key == KEY_ESC:
                state = "demo"
            elif key == KEY_ENTER:
                game, error = start_game(display, frame, levels,
                                         name.strip() or "Player", loader)
                if game:
                    state = "game"
                else:
                    state = "demo"
                    message, message_until = error, time.time() + 4
                last = time.time()
            elif key == KEY_BACKSPACE:
                name = name[:-1]
            elif 32 <= key <= 126 and len(name) < MAX_NAME:
                name += chr(key)
        elif state == "game":
            if game.phase == "board" or key in (KEY_ESC, ord('q'), ord('Q')):
                game.done = True

    if game:
        game.close()


def main():
    try:
        run()
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
