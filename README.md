# Pose Mimicry & Annotation Toolset

An interactive computer vision project built with **OpenCV** and **Google MediaPipe**. Players stand in front of a webcam and try to copy the pose shown in a picture (a dabbing Santa, a disco Christmas tree, a rearing horse...) by matching their joints to target points marked on the image. The project has three parts: a webcam pose tester, a tool for marking target joints on images, and the Pose Game itself, which runs as a kiosk with a self-playing demo.

---

## 🚀 Quick start

```bash
pip install opencv-python mediapipe numpy
python test.py        # first run only: downloads the pose model and checks the camera (Q to exit)
python pose_game.py   # starts the demo; type your name and press ENTER to play
```

Requires Python 3.9 or newer (tested with Python 3.12) and a webcam for playing. The demo itself works without a camera.

---

## 🛠 Features

- **`test.py`**: Diagnostics script. Downloads the MediaPipe Pose Landmarker model (`pose_landmarker_lite.task`) if it is missing, opens the webcam and draws the detected skeleton live.
- **`annotate.py`**: Level editor. Click target joints (nose, shoulders, wrists, hips...) on any image and save them as a JSON level.
- **`pose_game.py`**: The game, running as a kiosk:
  - a looping, self-playing demo with an animated human skeleton,
  - name entry over the demo to start a real webcam game,
  - 10-second rounds, live HUD and scoring multipliers for accuracy,
  - a top 5 leaderboard (`highscores.json`) that pops up during the demo and after every game.

---

## 📖 User guide

### `test.py`: check the camera and model

Run `python test.py`. If `pose_landmarker_lite.task` is not in the folder yet, it is downloaded first. A window then shows the camera image with the detected skeleton drawn on top. Press **Q** to exit.

Run it once before the first game: `pose_game.py` needs the model file, and without it only the demo works.

### `annotate.py`: make a level

1. Put the image in the `images` folder.
2. Run `python annotate.py <image_name>`, for example `python annotate.py santa.jpg`. (Without a name it tries `dog.jpg`.)
3. The top left corner says which joint to click next. Left-click on that spot in the image.
4. **U** undoes the last point. When all joints are placed, press **S** to save. **ESC** quits without saving.

The level is saved as `annotations/<image name>_annotations.json`, and the game plays it automatically the next time it starts.

Things to keep in mind:
- **Which joints are marked** is set by the `LANDMARKS_TO_ANNOTATE` list at the top of `annotate.py`. Comment or uncomment lines to choose them. More joints make a level harder. The included levels use 9 joints: nose, both shoulders, both wrists, both hips and both ankles.
- **Left and right work like a mirror.** The game shows the player mirrored, so mark the *Left* joints on the right side of the picture and the *Right* joints on the left side, as you would for a person facing you.

### `pose_game.py`: the game

Start it with:

```bash
python pose_game.py [levels] [--fullscreen]
```

| Command | What it plays |
| --- | --- |
| `python pose_game.py` | every level in the `annotations` folder (currently the 3 Christmas levels) |
| `python pose_game.py playlist.json` | the 16 animal levels |
| `python pose_game.py playlist_2.json` | a shorter set of 5 animal levels |
| `python pose_game.py annotations_animals` | every level in another folder |
| `--fullscreen` | starts in fullscreen (can be combined with any of the above) |

**Demo.** When the game starts, the demo begins right away; no camera is needed. An animated player with a human skeleton (fixed bone lengths, bending elbows and knees) tries to copy each picture. It looks at the picture, moves into the closest pose a human body can make, corrects some of its mistakes, and relaxes when the round ends. Round results are shown over the picture. After every 5 seconds of play the top 5 leaderboard pops up for 2 seconds. After the last level the levels are shuffled and the demo starts again. Demo scores are never saved.

**Playing.** While the demo runs, start typing your name (or press **ENTER**). The name box appears over the demo: **BACKSPACE** deletes, **ESC** goes back to the demo, and **ENTER** starts the game. If nobody types for 30 seconds, the demo comes back on its own.

In the game, each picture is shown with yellow target points, and you have 10 seconds to match them with your body. Your joints turn green when they are close enough to their target and red when they are not. The HUD at the bottom shows how many joints you match and your score, and the timer counts down in the top right corner. After each round the result is shown over the picture. After the last round your score is saved and the leaderboard is shown with your row highlighted; after 8 seconds (or any key) the game goes back to the demo. **ESC** or **Q** stops a game early, and the score is not saved.

**Keys.**

| Key | Where | Action |
| --- | --- | --- |
| letters / ENTER | demo | start entering a name |
| ENTER | name entry | start the game |
| BACKSPACE | name entry | delete a letter |
| ESC | name entry | back to the demo |
| ESC or Q | game | stop the game |
| any key | final leaderboard | back to the demo |
| ESC (or the window's X) | demo | close the program |
| F11 | anywhere | fullscreen on/off |

The MediaPipe model takes about 15 seconds to load, so it loads in the background while the demo plays. If a game is started before it is ready, "Loading camera game..." is shown for a moment.

### Scoring

During a round the game measures the distance between each of your detected joints and its target. A joint counts as matched when it is close enough (within 12% of the picture size). The game remembers the highest number of joints you matched at the same time during the round.

At the end of the round, that best match count is multiplied by:
- **x2.0** if you matched all joints,
- **x1.5** if you matched at least half of them,
- **x1.0** otherwise.

Round scores add up to your total. At the end of the game the total is compared with `highscores.json`, and if it is high enough your name is saved in the top 5.

---

## 🧩 Level and playlist files

A level is a JSON file that names its image and lists the target joints, using MediaPipe landmark numbers (0 = nose, 11/12 = shoulders, 15/16 = wrists, 23/24 = hips, 27/28 = ankles). Coordinates go from 0 to 1, measured from the top left corner of the image:

```json
{
  "image": "horse.png",
  "landmarks": {
    "0":  { "x": 0.696, "y": 0.206, "name": "Nose" },
    "11": { "x": 0.652, "y": 0.452, "name": "Left Shoulder" },
    "12": { "x": 0.573, "y": 0.454, "name": "Right Shoulder" }
  }
}
```

A playlist is a JSON list of level file names, played in that order:

```json
["bug.json", "liger.json", "lizard.json", "rio.json", "spider.json"]
```

Level files are looked up in `annotations/` and `annotations_animals/`, and images in `images/animals/` and `images/`.

---

## 📁 Repository directory structure

```text
├── annotations/              # Levels played by default; annotate.py saves here (3 Christmas levels)
├── annotations_animals/      # 16 animal levels used by the playlists
├── images/                   # Images for annotate.py and the Christmas levels
│   └── animals/              # Images for the animal levels
├── test.py                   # Camera and model check, downloads the model
├── annotate.py               # Level editor
├── pose_game.py              # The game: looping demo + webcam game
├── playlist.json             # All 16 animal levels
├── playlist_2.json           # Shorter animal playlist (5 levels)
├── highscores.json           # Top 5 leaderboard
└── pose_landmarker_lite.task # MediaPipe pose model (downloaded by test.py)
```

---

## 📦 Installation & requirements

- Python 3.9 or newer (tested with Python 3.12)
- A webcam (only needed for playing and for `test.py`)
- Packages:

```bash
pip install opencv-python mediapipe numpy
```
