"""
Pose Game - looping demo (attract mode).

Plays the Pose Game by itself with an animated "fake" player, so it needs no
camera, no MediaPipe model and no keyboard input. It runs in a loop forever:

  * each animal level is played for ROUND_DURATION seconds by a drawn
    skeleton that moves into the animal pose (sometimes it misses joints,
    like a real player would),
  * the round result is shown on top of the picture (no black screens),
  * every SCOREBOARD_EVERY seconds of play the top-5 leaderboard from
    highscores.json pops up for SCOREBOARD_SHOW seconds,
  * after the last level the playlist is shuffled and the demo starts again.

Demo scores are never written to highscores.json.

Usage:
    python pose_game_demo.py                    # levels from playlist.json
    python pose_game_demo.py playlist_2.json    # another playlist
    python pose_game_demo.py annotations_animals  # every level in a folder

Keys:  Q / Esc - quit    F - toggle fullscreen
"""
import cv2
import numpy as np
import glob
import json
import math
import os
import random
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANNOTATION_DIRS = ["annotations_animals", "annotations"]
IMAGE_DIRS = [os.path.join("images", "animals"), "images"]
HIGHSCORES_FILE = "highscores.json"
DEFAULT_PLAYLIST = "playlist.json"

WINDOW = "Pose Game - Demo"
W, H = 640, 480                # same canvas size as the real game
WINDOW_SIZE = (960, 720)       # initial window size (window is resizable)
FPS = 30

MATCH_THRESHOLD = 0.12         # same as pose_game.py
ROUND_DURATION = 10            # same as pose_game.py
RESULT_DURATION = 3            # how long the round result stays on screen
SCOREBOARD_EVERY = 5           # seconds of play between leaderboard pop-ups
SCOREBOARD_SHOW = 2            # how long the leaderboard stays visible
TOP_N = 5
PLAYER_NAME = "DEMO"

POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),
    (17,19),(12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32)
]

# Standing pose of the fake player (normalised image coordinates).
# Only the "core" joints are animated; face, hands and feet follow them.
NEUTRAL = {
    0: (0.50, 0.20),
    11: (0.57, 0.33), 12: (0.43, 0.33),
    13: (0.60, 0.45), 14: (0.40, 0.45),
    15: (0.61, 0.56), 16: (0.39, 0.56),
    23: (0.54, 0.58), 24: (0.46, 0.58),
    25: (0.545, 0.74), 26: (0.455, 0.74),
    27: (0.55, 0.90), 28: (0.45, 0.90),
}
CORE = list(NEUTRAL.keys())

# Offsets of the face points from the nose (at normal body size).
FACE_OFFSETS = {
    1: (0.008, -0.015), 2: (0.015, -0.016), 3: (0.022, -0.015),
    4: (-0.008, -0.015), 5: (-0.015, -0.016), 6: (-0.022, -0.015),
    7: (0.035, -0.005), 8: (-0.035, -0.005),
    9: (0.010, 0.020), 10: (-0.010, 0.020),
}
# Non-core joint -> the core joint it hangs from.
PARENT = {i: 0 for i in FACE_OFFSETS}
PARENT.update({17: 15, 19: 15, 21: 15, 18: 16, 20: 16, 22: 16,
               29: 27, 31: 27, 30: 28, 32: 28})


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
        with open(os.path.join(BASE_DIR, HIGHSCORES_FILE)) as f:
            scores = json.load(f)
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:TOP_N]
    except (OSError, ValueError, KeyError, TypeError):
        return []


def calculate_score(matched, total):
    if total == 0 or matched == 0:
        return 0
    ratio = matched / total
    multiplier = 2.0 if ratio == 1.0 else (1.5 if ratio >= 0.5 else 1.0)
    return int(matched * multiplier)


# --------------------------------------------------------------------------
# Fake player
# --------------------------------------------------------------------------

def _v(p):
    return np.array(p, dtype=float)


def bend(a, b, amount):
    """Point between a and b, pushed sideways (an elbow or a knee)."""
    mid = (a + b) / 2
    d = b - a
    n = np.linalg.norm(d)
    if n < 1e-6:
        return mid
    return mid + np.array([-d[1], d[0]]) / n * amount


def derive_full(core):
    """Build all 33 MediaPipe landmarks from the core joints."""
    full = [None] * 33
    for i, p in core.items():
        full[i] = p
    s = np.linalg.norm(core[11] - core[12]) / 0.14
    s = min(max(s, 0.4), 2.5)

    for i, off in FACE_OFFSETS.items():
        full[i] = core[0] + _v(off) * s

    for wrist, elbow, pinky, index, thumb, side in ((15, 13, 17, 19, 21, 1),
                                                    (16, 14, 18, 20, 22, -1)):
        d = core[wrist] - core[elbow]
        n = np.linalg.norm(d)
        d = d / n if n > 1e-6 else np.array([0.0, 1.0])
        perp = np.array([-d[1], d[0]]) * side
        full[pinky] = core[wrist] + d * 0.035 * s + perp * 0.012 * s
        full[index] = core[wrist] + d * 0.04 * s - perp * 0.006 * s
        full[thumb] = core[wrist] + d * 0.02 * s - perp * 0.02 * s

    for ankle, heel, toe, side in ((27, 29, 31, 1), (28, 30, 32, -1)):
        full[heel] = core[ankle] + _v((-0.008 * side, 0.02)) * s
        full[toe] = core[ankle] + _v((0.03 * side, 0.025)) * s
    return full


def default_offset(i):
    """Typical offset of a non-core joint from its parent at normal size."""
    if i in FACE_OFFSETS:
        return _v(FACE_OFFSETS[i])
    neutral_full = derive_full({k: _v(p) for k, p in NEUTRAL.items()})
    return neutral_full[i] - _v(NEUTRAL[PARENT[i]])


def target_pose(landmarks):
    """Core-joint pose that fits the level's target landmarks."""
    target = {i: _v(p) for i, p in NEUTRAL.items()}
    ann = {i: _v((v["x"], v["y"])) for i, v in landmarks.items()}

    goals = {i: p for i, p in ann.items() if i in target}
    # A target on a face/hand/foot point moves its parent joint there too.
    for i, p in ann.items():
        if i in PARENT and PARENT[i] not in goals:
            goals[PARENT[i]] = p - default_offset(i)

    if goals:
        shift = np.mean([goals[i] - target[i] for i in goals], axis=0)
        for i in target:
            if i not in goals:
                target[i] = target[i] + shift
        target.update(goals)

    for elbow, sh, wr, side in ((13, 11, 15, 1), (14, 12, 16, -1)):
        if elbow not in goals:
            target[elbow] = bend(target[sh], target[wr], 0.03 * side)
    for knee, hip, ank, side in ((25, 23, 27, -1), (26, 24, 28, 1)):
        if knee not in goals:
            target[knee] = bend(target[hip], target[ank], 0.02 * side)

    return {i: np.clip(p, 0.02, 0.98) for i, p in target.items()}


def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


class FakePlayer:
    def __init__(self):
        self.pose = {i: _v(p) for i, p in NEUTRAL.items()}
        self.start_round({})

    def start_round(self, landmarks):
        self.start = {i: p.copy() for i, p in self.pose.items()}
        self.target = target_pose(landmarks)
        self.ann = {i: _v((v["x"], v["y"])) for i, v in landmarks.items()}

        skill = random.uniform(0.55, 1.0)
        ids = set(CORE) | set(self.ann)
        self.err, self.fix_at = {}, {}
        for i in ids:
            angle = random.uniform(0, 2 * math.pi)
            if i in self.ann and random.random() > skill:
                mag = random.uniform(0.14, 0.26)       # misses this joint
                # Sometimes the player notices and corrects it later.
                self.fix_at[i] = (random.uniform(4.0, 8.5)
                                  if random.random() < 0.5 else math.inf)
            else:
                mag = random.uniform(0.0, 0.05)
                self.fix_at[i] = math.inf
            self.err[i] = np.array([math.cos(angle), math.sin(angle)]) * mag
        self.delay = {i: random.uniform(0.0, 1.2) for i in ids}
        self.move = {i: random.uniform(1.2, 2.5) for i in ids}
        self.phase = {i: (random.uniform(0, 6.3), random.uniform(0, 6.3))
                      for i in ids}

    def _error(self, i, t):
        e = self.err[i]
        if t > self.fix_at[i]:
            e = e * max(0.15, 1.0 - (t - self.fix_at[i]))
        return e

    def _jitter(self, i, t):
        a, b = self.phase[i]
        return np.array([math.sin(2.1 * t + a), math.cos(1.7 * t + b)]) * 0.008

    def _progress(self, i, t):
        return smoothstep((t - 0.6 - self.delay[i]) / self.move[i])

    def update(self, t):
        """Return the 33 landmarks of the fake player t seconds into the round."""
        sway = np.array([math.sin(1.3 * t) * 0.006, math.sin(2.6 * t) * 0.003])
        core = {}
        for i in CORE:
            goal = self.target[i] + self._error(i, t)
            p = self._progress(i, t)
            pos = self.start[i] + (goal - self.start[i]) * p
            core[i] = np.clip(pos + self._jitter(i, t) + sway, 0.01, 0.99)
        self.pose = core

        full = derive_full(core)
        # Pull targeted face/hand/foot points the rest of the way.
        for i, goal in self.ann.items():
            if i in CORE or not 0 <= i < 33:
                continue
            goal = goal + self._error(i, t) + self._jitter(i, t) + sway
            full[i] = full[i] + (goal - full[i]) * self._progress(i, t)
        return full


def count_matched(full, landmarks):
    matched = 0
    for i, ann in landmarks.items():
        if 0 <= i < len(full):
            p = full[i]
            if math.hypot(p[0] - ann["x"], p[1] - ann["y"]) < MATCH_THRESHOLD:
                matched += 1
    return matched


# --------------------------------------------------------------------------
# Drawing (same look as pose_game.py)
# --------------------------------------------------------------------------

def draw_animal_joints(frame, landmarks):
    for ann in landmarks.values():
        px, py = int(ann["x"] * W), int(ann["y"] * H)
        cv2.circle(frame, (px, py), 7, (0, 220, 255), -1)
        cv2.putText(frame, ann["name"], (px + 9, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)


def draw_user_skeleton(frame, full, landmarks):
    points = [(int(p[0] * W), int(p[1] * H)) for p in full]
    for a, b in POSE_CONNECTIONS:
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


def dark_box(frame, x1, y1, x2, y2, alpha=0.7):
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
    roi = frame[y1:y2, x1:x2]
    frame[y1:y2, x1:x2] = (roi * (1 - alpha)).astype(np.uint8)
    cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), (255, 200, 0), 2)


def render_play(level, full, matched, best, total_score, img_num, n_images, remaining):
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

    # Blinking demo label so viewers know nobody is playing right now.
    if int(time.time() * 2) % 2 == 0:
        put_center(frame, "DEMO", 32, 0.9, (0, 0, 255), 2)
    cv2.putText(frame, "Q: quit  F: fullscreen", (W - 205, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    return frame


def draw_round_result(frame, round_score, matched, total, cumulative, img_num, n_images):
    ratio = matched / total if total > 0 else 0
    mult = "x2.0" if ratio == 1.0 else ("x1.5" if ratio >= 0.5 else "x1.0")
    dark_box(frame, 70, 110, W - 70, 370)
    put_center(frame, f"Round {img_num}/{n_images} complete!", 165, 0.95, (255, 255, 255), 2)
    put_center(frame, f"Best match: {matched}/{total}  (multiplier {mult})", 225, 0.7, (0, 220, 255), 2)
    put_center(frame, f"Round score: {round_score}", 285, 0.9, (0, 255, 0), 2)
    put_center(frame, f"Total score: {cumulative}", 340, 0.9, (255, 200, 0), 2)


def draw_scoreboard(frame, scores):
    rows = max(len(scores), 1)
    top = 85
    bottom = top + 80 + rows * 45
    dark_box(frame, W // 2 - 210, top, W // 2 + 210, bottom, alpha=0.75)
    put_center(frame, "LEADERBOARD", top + 50, 1.2, (255, 200, 0), 3)
    if not scores:
        put_center(frame, "No scores yet", top + 100, 0.8, (255, 255, 255), 2)
    for i, entry in enumerate(scores):
        put_center(frame, f"{i + 1}.  {entry['name']}  -  {entry['score']} pts",
                   top + 100 + i * 45, 0.8, (255, 255, 255), 2)


def draw_run_complete(frame, total_score):
    dark_box(frame, 70, 140, W - 70, 340)
    put_center(frame, "Demo run complete!", 195, 1.0, (255, 255, 255), 2)
    put_center(frame, f"Total score: {total_score}", 255, 0.95, (255, 200, 0), 2)
    put_center(frame, "Starting again...", 310, 0.7, (0, 220, 255), 2)


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

class Display:
    def __init__(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, *WINDOW_SIZE)
        self.fullscreen = False
        self.last = time.time()

    def show(self, frame):
        """Show a frame at about FPS. Returns False when the user quits."""
        cv2.imshow(WINDOW, frame)
        wait = max(1, int((1.0 / FPS - (time.time() - self.last)) * 1000))
        key = cv2.waitKey(wait) & 0xFF
        self.last = time.time()
        if key in (ord('q'), ord('Q'), 27):
            return False
        if key in (ord('f'), ord('F')):
            self.fullscreen = not self.fullscreen
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if self.fullscreen
                                  else cv2.WINDOW_NORMAL)
        try:
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return False   # window closed with the X button
        except cv2.error:
            return False
        return True


def scoreboard_visible(play_clock):
    return play_clock % (SCOREBOARD_EVERY + SCOREBOARD_SHOW) >= SCOREBOARD_EVERY


def run():
    arg = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PLAYLIST
    levels = load_levels(arg)
    if not levels:
        print("No levels found.")
        return
    print(f"Loaded {len(levels)} levels. Press Q or Esc in the window to quit.")

    display = Display()
    player = FakePlayer()
    order = list(levels)
    play_clock = 0.0   # counts only gameplay time, drives the leaderboard pop-up

    while True:
        scores = load_highscores()
        total_score = 0
        level = order[0]

        for img_num, level in enumerate(order, 1):
            landmarks = level["landmarks"]
            player.start_round(landmarks)
            best = 0
            t0 = last = time.time()

            # Gameplay
            while True:
                now = time.time()
                t = now - t0
                play_clock += now - last
                last = now
                if t >= ROUND_DURATION:
                    break
                full = player.update(t)
                matched = count_matched(full, landmarks)
                best = max(best, matched)
                frame = render_play(level, full, matched, best, total_score,
                                    img_num, len(order), ROUND_DURATION - t)
                if scoreboard_visible(play_clock):
                    draw_scoreboard(frame, scores)
                if not display.show(frame):
                    return

            round_score = calculate_score(best, len(landmarks))
            total_score += round_score

            # Round result over the picture; the player keeps holding the pose.
            t_end = time.time() + RESULT_DURATION
            while time.time() < t_end:
                full = player.update(time.time() - t0)
                matched = count_matched(full, landmarks)
                frame = render_play(level, full, matched, best, total_score,
                                    img_num, len(order), 0)
                draw_round_result(frame, round_score, best, len(landmarks),
                                  total_score, img_num, len(order))
                if not display.show(frame):
                    return

        # End of the playlist: show the total, shuffle and start again.
        t_end = time.time() + RESULT_DURATION
        while time.time() < t_end:
            frame = level["image"].copy()
            draw_run_complete(frame, total_score)
            if not display.show(frame):
                return
        random.shuffle(order)


def main():
    try:
        run()
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
