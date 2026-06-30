import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json
import sys
import math
import os
import time
import glob
import numpy as np

MODEL_PATH = "pose_landmarker_lite.task"
MATCH_THRESHOLD = 0.12
ROUND_DURATION = 30
HIGHSCORES_FILE = "highscores.json"
TOP_N = 5

POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),
    (17,19),(12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32)
]


def load_highscores():
    if os.path.exists(HIGHSCORES_FILE):
        with open(HIGHSCORES_FILE) as f:
            return json.load(f)
    return []


def save_highscores(scores):
    with open(HIGHSCORES_FILE, 'w') as f:
        json.dump(scores, f, indent=2)


def update_highscores(name, score):
    scores = load_highscores()
    scores.append({"name": name, "score": score})
    scores.sort(key=lambda x: x["score"], reverse=True)
    scores = scores[:TOP_N]
    save_highscores(scores)
    return scores


def calculate_score(matched, total):
    if total == 0 or matched == 0:
        return 0
    ratio = matched / total
    multiplier = 2.0 if ratio == 1.0 else (1.5 if ratio >= 0.5 else 1.0)
    return int(matched * multiplier)


def draw_animal_joints(frame, annotations, h, w):
    for ann in annotations.values():
        px, py = int(ann["x"] * w), int(ann["y"] * h)
        cv2.circle(frame, (px, py), 7, (0, 220, 255), -1)
        cv2.putText(frame, ann["name"], (px + 9, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)


def draw_user_skeleton(frame, landmarks, annotations, h, w):
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in POSE_CONNECTIONS:
        if a < len(points) and b < len(points):
            cv2.line(frame, points[a], points[b], (200, 200, 200), 2)
    for lm_id_str, ann in annotations.items():
        lm_id = int(lm_id_str)
        if lm_id >= len(landmarks):
            continue
        lm = landmarks[lm_id]
        px, py = int(lm.x * w), int(lm.y * h)
        dist = math.sqrt((lm.x - ann["x"]) ** 2 + (lm.y - ann["y"]) ** 2)
        color = (0, 255, 0) if dist < MATCH_THRESHOLD else (0, 0, 255)
        cv2.circle(frame, (px, py), 7, color, -1)


def count_matched(landmarks, annotations):
    matched = 0
    for lm_id_str, ann in annotations.items():
        lm_id = int(lm_id_str)
        if lm_id >= len(landmarks):
            continue
        lm = landmarks[lm_id]
        if math.sqrt((lm.x - ann["x"]) ** 2 + (lm.y - ann["y"]) ** 2) < MATCH_THRESHOLD:
            matched += 1
    return matched


def show_round_result(w, h, round_score, matched, total, cumulative_score, img_num, total_images):
    ratio = matched / total if total > 0 else 0
    mult_str = "x2.0" if ratio == 1.0 else ("x1.5" if ratio >= 0.5 else "x1.0")

    panel = np.zeros((h, w, 3), dtype=np.uint8)
    cy = h // 4

    cv2.putText(panel, f"Round {img_num}/{total_images} complete!",
                (w // 2 - 200, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cy += 70
    cv2.putText(panel, f"Best match: {matched}/{total}  (multiplier {mult_str})",
                (w // 2 - 230, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
    cy += 65
    cv2.putText(panel, f"Round score: {round_score}",
                (w // 2 - 130, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cy += 65
    cv2.putText(panel, f"Total score: {cumulative_score}",
                (w // 2 - 140, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)

    cv2.imshow("Pose Game", panel)


def show_leaderboard(w, h, scores, player_name, player_score):
    panel = np.zeros((h, w, 3), dtype=np.uint8)

    cv2.putText(panel, "LEADERBOARD", (w // 2 - 160, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 200, 0), 3)

    for i, entry in enumerate(scores):
        y = 150 + i * 65
        is_current = entry["name"] == player_name and entry["score"] == player_score
        color = (0, 255, 255) if is_current else (255, 255, 255)
        cv2.putText(panel, f"{i + 1}.  {entry['name']}  —  {entry['score']} pts",
                    (w // 2 - 210, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

    cv2.putText(panel, "Press any key to exit.", (w // 2 - 150, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

    cv2.imshow("Pose Game", panel)
    cv2.waitKey(0)


def load_annotation_files(arg):
    if os.path.isdir(arg):
        return sorted(glob.glob(os.path.join(arg, "*_annotations.json")))
    with open(arg) as f:
        data = json.load(f)
    if isinstance(data, list):
        base = os.path.dirname(arg)
        return [p if os.path.isabs(p) else os.path.join(base, p) for p in data]
    return [arg]


def input_player_name_cv2(w, h):
    """Displays an input panel inside the OpenCV window to type your name."""
    name = ""
    # Ensure window exists to capture immediate keyboard input
    cv2.namedWindow("Pose Game")

    while True:
        panel = np.zeros((h, w, 3), dtype=np.uint8)
        
        cv2.putText(panel, "ENTER YOUR NAME", (w // 2 - 180, h // 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 200, 0), 3)
        
        # Draw the current typed name string
        cv2.putText(panel, name + "_", (w // 2 - 150, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        cv2.putText(panel, "Press ENTER to confirm  |  Backspace to delete", (w // 2 - 220, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        cv2.imshow("Pose Game", panel)
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == 13:  # Enter Key code
            break
        elif key == 8:  # Backspace Key code
            name = name[:-1]
        elif 32 <= key <= 126:  # Normal readable characters
            if len(name) < 15:  # Restrict length limit to fit on screen
                name += chr(key)
                
    return name.strip() if name.strip() else "Player"


def main():
    if len(sys.argv) < 2:
        print("No argument provided. Falling back to default playlist.json")
        raw = ["playlist.json"] 
    else:
        raw = sys.argv[1:]
    name_arg = next((raw[i + 1] for i, a in enumerate(raw) if a == "--name" and i + 1 < len(raw)), None)
    args = [a for i, a in enumerate(raw) if a != "--name" and (i == 0 or raw[i - 1] != "--name")]

    annotation_files = []
    for arg in args:
        annotation_files.extend(load_annotation_files(arg))

    if not annotation_files:
        print("No annotation files found.")
        return

    # Fallback default canvas geometry for resolution before camera configuration
    cam_w, cam_h = 640, 480

    if name_arg:
        player_name = name_arg
    else:
        player_name = input_player_name_cv2(cam_w, cam_h)

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(base_options=base_options)
    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    time.sleep(1.0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Refresh correct resolution after camera spins up
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if cam_w == 0 or cam_h == 0:
        cam_w, cam_h = 640, 480
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)

    total_score = 0

    for img_num, ann_file in enumerate(annotation_files, 1):
        with open(ann_file) as f:
            data = json.load(f)

        image_path = data["image"]
        if not os.path.isabs(image_path):
            image_path = os.path.join(os.path.dirname(ann_file), os.path.basename(image_path))

        animal_img = cv2.imread(image_path)
        if animal_img is None:
            print(f"Could not load {image_path}, skipping.")
            continue

        annotations = data["landmarks"]
        total_joints = len(annotations)
        animal_bg = cv2.resize(animal_img, (cam_w, cam_h))

        best_matched = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            elapsed = time.time() - start_time
            remaining = max(0.0, ROUND_DURATION - elapsed)

            panel = animal_bg.copy()
            draw_animal_joints(panel, annotations, cam_h, cam_w)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)

            matched = 0
            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                draw_user_skeleton(panel, landmarks, annotations, cam_h, cam_w)
                matched = count_matched(landmarks, annotations)
                best_matched = max(best_matched, matched)

            # HUD
            timer_color = (0, 0, 255) if remaining <= 10 else (0, 220, 255)
            cv2.putText(panel, f"Image {img_num}/{len(annotation_files)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(panel, f"{int(remaining)}s",
                        (cam_w - 80, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, timer_color, 3)
            cv2.putText(panel, f"Joints: {matched}/{total_joints}  Best: {best_matched}",
                        (10, cam_h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(panel, f"Score: {total_score}",
                        (10, cam_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

            cv2.imshow("Pose Game", panel)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

            if remaining <= 0:
                break

        round_score = calculate_score(best_matched, total_joints)
        total_score += round_score

        show_round_result(cam_w, cam_h, round_score, best_matched, total_joints,
                          total_score, img_num, len(annotation_files))

        deadline = time.time() + 3
        while time.time() < deadline:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()

    scores = update_highscores(player_name, total_score)
    show_leaderboard(cam_w, cam_h, scores, player_name, total_score)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()