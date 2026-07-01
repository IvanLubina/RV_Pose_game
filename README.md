# Pose Mimicry & Annotation Toolset

An interactive computer vision project built with **OpenCV** and **Google MediaPipe**. This project features a standard human pose tester, a custom landmark annotator, and an engaging "Pose Game" where players attempt to mimic target coordinates (e.g., matching human joint locations to custom positions on animal images or template poses) using a live webcam feed.

---

## 🛠 Features

'test.py': A quick-start diagnostics script that downloads the MediaPipe Pose Landmarker model automatically, initializes your webcam, and renders live skeleton landmarks overlayed on your screen.

'annotate.py': A lightweight annotation utility allowing you to overlay and map custom target landmark locations (e.g., Nose, Right Shoulder, Wrists, Hips) onto any local image file, generating custom JSON profiles.

'pose_game.py': A complete gamified experience featuring in-app name entry via OpenCV canvas, a 30-second round timer, multi-tier scoring multipliers based on joint matching precision, interactive HUDs, and a persistent local leaderboard ('highscores.json').

---

## 📖 User guide

'test.py' is a script to test 'pose_landmarker_lite.task' model that recognises joints and camera quickly. If a user thinks everything works, running this script is not important. In case user wants to run 'test.py', camera must be plugged in, otherwise the script will just download 'pose_landmarker_lite.task'. When the cammera is plugged in and 'test.py' runs, a window will pop up that shows everything that camera recieves. When done with testing, press Q to exit.

'annotate.py' is a script that is needed for setting levels (annotations of joints on the image). On the very top of that script is list named 'LANDMARKS_TO_ANNOTATE' which contains jonits that are going to be annotated. Keep in mind that the more joints are annotaded, the more difficult it will be to match all joints in the game. To annotate an image of choice, put the image in images folder and run the script (in terminal) with the following line: python annotate.py <image_name>
If user runs the script without writting specific image name, the image chosen to be annotated will be 'dog.jpg'. While annotating, on the top left corner will be a message that says which joint is to be marked. Annotations are marked by clicking left mouse button on the spot of user's choice. After marking all of the joints are marked, there will be message "Press S to save".

'pose_game.py' is the main script that starts the actual gameplay where players attempt to mimic the saved landmark patterns using their webcam. To play the game, ensure you have already generated annotation files using 'annotate.py' and that your images are in the images folder and your JSON files are in the annotations folder. To launch the game, run the script in the terminal with the following line: python pose_game.py <annotation_file_or_folder> (for example, python pose_game.py annotations to load all levels). If a user runs the script without specifying an argument, it will automatically fall back to loading 'playlist.json'. In 'playlist.json' users can change (add, or remove) which images are going to be in the round.
When the script starts, a window will pop up prompting the player to enter their name directly using the keyboard, which is confirmed by pressing ENTER. The game will then load each image backdrop one by one, and the player has 30 seconds per round to match their body joints with the yellow target landmarks shown on the screen. The HUD at the bottom tracks the live score and joint matching precision, while a color-coded timer ticks down in the top right corner. Scores are calculated at the end of each round with point multipliers (up to x2.0) awarded for high tracking accuracy. After all rounds are completed, a persistent local leaderboard window will display the top high scores. At any point during the game, press Q to exit.

The scoring system in 'pose_game.py' is designed to reward both precision and completion. During gameplay, the script constantly tracks the mathematical distance between your webcam's detected joints and the target annotations. If a user's joint comes within a specific proximity threshold of the target landmark, the marker turns green and counts as a successful match; otherwise, it remains red. The game actively remembers the highest number of joints you managed to match simultaneously at any point during the 30-second round.
When the round ends, a final score is calculated by taking your best match count and applying a multiplier based on your accuracy ratio. Achieving a perfect 100% match across all joints grants a x2.0 point multiplier, while matching at least half of the required joints awards a x1.5 multiplier. Any performance below the 50% accuracy threshold receives a standard x1.0 multiplier. These points accumulate across all loaded images to form your final total score. At the end of the game, this score is compared against 'highscores.json', and if it is high enough, your name will be permanently saved into the local Top 5 leaderboard.

---

## 🚀 Installation & Requirements

Ensure you have Python 3.8+ installed. You can install all necessary prerequisites via `pip`:

```bash
pip install opencv-python mediapipe numpy

