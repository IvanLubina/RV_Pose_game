import cv2
import json
import sys
import os

LANDMARKS_TO_ANNOTATE = [
    (0,  "Nose"),
    #(11, "Left Shoulder"),
    (12, "Right Shoulder"),
    #(13, "Left Elbow"),
    #(14, "Right Elbow"),
    (15, "Left Wrist"),
    (16, "Right Wrist"),
    (23, "Left Hip"),
    #(24, "Right Hip"),
    #(25, "Left Knee"),
    #(26, "Right Knee"),
    #(27, "Left Ankle"),
    #(28, "Right Ankle"),
]

image = None
base_image = None
annotations = {}
current_idx = 0


def update_display():
    display = base_image.copy()
    h, w = display.shape[:2]

    for lm_id, pos in annotations.items():
        px, py = int(pos["x"] * w), int(pos["y"] * h)
        cv2.circle(display, (px, py), 7, (0, 255, 0), -1)
        cv2.putText(display, pos["name"], (px + 9, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    if current_idx < len(LANDMARKS_TO_ANNOTATE):
        label = LANDMARKS_TO_ANNOTATE[current_idx][1]
        cv2.putText(display, f"Click: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
        cv2.putText(display, "U = undo  |  ESC = quit", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    else:
        cv2.putText(display, "All joints placed! Press S to save.", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Annotate", display)


def click_handler(event, x, y, flags, param):
    global current_idx
    if event == cv2.EVENT_LBUTTONDOWN and current_idx < len(LANDMARKS_TO_ANNOTATE):
        h, w = base_image.shape[:2]
        lm_id, lm_name = LANDMARKS_TO_ANNOTATE[current_idx]
        annotations[str(lm_id)] = {"x": x / w, "y": y / h, "name": lm_name}
        print(f"  {lm_name}: ({x/w:.3f}, {y/h:.3f})")
        current_idx += 1
        update_display()


def main():
    global base_image, current_idx

    if len(sys.argv) < 2:
        image_path = "dog.jpg" 
        print(f"No image provided. Falling back to default: {image_path}")
    else:
        image_path = sys.argv[1]

    base_image = cv2.imread(image_path)
    if base_image is None:
        print(f"Could not load image: {image_path}")
        return

    print("Annotation tool — click each joint in the order shown.")
    print("U = undo last  |  S = save  |  ESC = quit\n")

    cv2.namedWindow("Annotate", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Annotate", click_handler)
    update_display()

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord('s'):
            out_path = os.path.splitext(image_path)[0] + "_annotations.json"
            with open(out_path, 'w') as f:
                json.dump({"image": image_path, "landmarks": annotations}, f, indent=2)
            print(f"\nSaved annotations to: {out_path}")
            break

        elif key == ord('u'):
            if annotations:
                last_key = list(annotations.keys())[-1]
                removed = annotations.pop(last_key)
                current_idx -= 1
                print(f"  Undid: {removed['name']}")
                update_display()

        elif key == 27:
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
