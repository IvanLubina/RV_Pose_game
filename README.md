# Pose Mimicry & Annotation Toolset

An interactive computer vision project built with **OpenCV** and **Google MediaPipe**. This project features a standard human pose tester, a custom landmark annotator, and an engaging "Pose Game" where players attempt to mimic target coordinates (e.g., matching human joint locations to custom positions on animal images or template poses) using a live webcam feed.

---

## 🛠 Features

'test.py': A quick-start diagnostics script that downloads the MediaPipe Pose Landmarker model automatically, initializes your webcam, and renders live skeleton landmarks overlayed on your screen.

'annotate.py': A lightweight annotation utility allowing you to overlay and map custom target landmark locations (e.g., Nose, Right Shoulder, Wrists, Hips) onto any local image file, generating custom JSON profiles.

'pose_game.py': A complete gamified experience that runs as a kiosk: a looping self-playing demo with an animated human skeleton, in-app name entry via OpenCV canvas to start a real webcam game, a 10-second round timer, multi-tier scoring multipliers based on joint matching precision, interactive HUDs, and a persistent local leaderboard ('highscores.json') that pops up during the demo.

---

## 📖 User guide

'test.py' is a script to test 'pose_landmarker_lite.task' model that recognises joints and camera quickly. If a user thinks everything works, running this script is not important. In case user wants to run 'test.py', camera must be plugged in, otherwise the script will just download 'pose_landmarker_lite.task'. When the cammera is plugged in and 'test.py' runs, a window will pop up that shows everything that camera recieves. When done with testing, press Q to exit.

'annotate.py' is a script that is needed for setting levels (annotations of joints on the image). On the very top of that script is list named 'LANDMARKS_TO_ANNOTATE' which contains jonits that are going to be annotated. Keep in mind that the more joints are annotaded, the more difficult it will be to match all joints in the game. To annotate an image of choice, put the image in images folder and run the script (in terminal) with the following line: python annotate.py <image_name>
If user runs the script without writting specific image name, the image chosen to be annotated will be 'dog.jpg'. While annotating, on the top left corner will be a message that says which joint is to be marked. Annotations are marked by clicking left mouse button on the spot of user's choice. After marking all of the joints are marked, there will be message "Press S to save".

'pose_game.py' is the main script. It runs as a kiosk: a demo plays the game by itself in a loop, and anyone can start a real webcam game by typing their name. To launch it, run the script in the terminal with the following line: python pose_game.py [playlist_or_folder] [--fullscreen]. Without an argument it loads 'playlist.json'; you can also pass another playlist (for example, python pose_game.py playlist_2.json) or a folder of level files (for example, python pose_game.py annotations_animals). In 'playlist.json' users can change (add, or remove) which levels are going to be played. Level files are looked up in the 'annotations_animals' and 'annotations' folders, and images in 'images/animals' and 'images', so levels made with 'annotate.py' work without moving any files.

Demo: when the script starts, a window opens and the demo begins right away, no camera needed. An animated player with a human skeleton (fixed bone lengths, bending elbows and knees) tries to match each animal pose: it looks at the picture, moves into the closest pose a human body can make, corrects some of its mistakes, and relaxes when the round ends. Round results are shown on top of the picture, every 5 seconds of play the top 5 leaderboard pops up for 2 seconds, and after the last level the levels are shuffled and the demo starts again. Demo scores are never saved.

Playing: start typing your name (or press ENTER) while the demo runs. The name box appears over the demo; BACKSPACE deletes, ESC goes back to the demo, and ENTER starts the game. If nobody types for 30 seconds, the demo comes back on its own. The webcam game loads each image backdrop one by one, and the player has 10 seconds per round to match their body joints with the yellow target landmarks shown on the screen. The HUD at the bottom tracks the live score and joint matching precision, while the timer ticks down in the top right corner. After every round the result is shown over the picture. After all rounds are completed, the score is saved and the leaderboard is shown with the player's row highlighted; after 8 seconds (or any key) the game goes back to the demo. Press ESC or Q to stop a game early (the score is not saved).

Keys: ESC in the demo (or the window's X) closes the program, and F11 switches fullscreen on and off. The MediaPipe model takes about 15 seconds to load, so it loads in the background while the demo plays; if a game is started before it is ready, "Loading camera game..." is shown for a moment.

The scoring system in 'pose_game.py' is designed to reward both precision and completion. During gameplay, the script constantly tracks the mathematical distance between your webcam's detected joints and the target annotations. If a user's joint comes within a specific proximity threshold of the target landmark, the marker turns green and counts as a successful match; otherwise, it remains red. The game actively remembers the highest number of joints you managed to match simultaneously at any point during the 10-second round.
When the round ends, a final score is calculated by taking your best match count and applying a multiplier based on your accuracy ratio. Achieving a perfect 100% match across all joints grants a x2.0 point multiplier, while matching at least half of the required joints awards a x1.5 multiplier. Any performance below the 50% accuracy threshold receives a standard x1.0 multiplier. These points accumulate across all loaded images to form your final total score. At the end of the game, this score is compared against 'highscores.json', and if it is high enough, your name will be permanently saved into the local Top 5 leaderboard.

---
## 📁 Repository Directory Structure
To ensure the scripts track path references correctly, maintain the following directory tree layout in your workspace root:

```text
├── annotations/             # Levels saved by annotate.py (*_annotations.json)
├── annotations_animals/     # Animal levels used by the playlists
├── images/                  # Drop your image backdrops here (used by annotate.py)
│   └── animals/             # Animal backdrops used by the playlists
├── test.py                  # Pipeline tracking diagnostic tool
├── annotate.py              # Landmark design utility
├── pose_game.py             # Main game: looping demo + webcam game
├── playlist.json            # Default playlist (all animal levels)
├── playlist_2.json          # Shorter playlist (5 levels)
├── highscores.json          # Top 5 leaderboard
└── pose_landmarker_lite.task # Automatically downloaded pipeline asset
```
---

## 🚀 Installation & Requirements

Ensure you have Python 3.8+ installed. You can install all necessary prerequisites via `pip`:

```bash
pip install opencv-python mediapipe numpy
```
