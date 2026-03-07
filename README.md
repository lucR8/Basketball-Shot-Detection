# Basketball Shot Detection

Computer vision project designed to automatically detect basketball shot attempts and classify their outcomes (Made / Miss / Unknown) from a single video stream.

Unlike standard object detection tasks, this project focuses on temporal event inference: a shot is not treated as a single detection, but as a short sequence of visual cues including shooting motion, ball trajectory, and rim interaction.

---
 
## Overview

The pipeline is built to :

* detect key objects in each frame (`ball`, `rim`, `person`, `shoot`),
* track the ball over time,
* detect shot attempts from temporal patterns,
* classify each attempt as Made, Miss, or Unknown,
* generate annotated video outputs and detailed debug information.

The project is modular and designed for reproducible experimentation on fixed-camera basketball videos.

---

## Main Features

* YOLO-based object detection
* Ball tracking with temporal continuity
* Shot attempt detection using a finite-state machine (FSM)
* Made / Miss / Unknown classification
* Rim stabilization for fixed-camera videos
* Debug overlays and detailed console logs
* Modular pipeline structure

---

## Outputs

For each input video, the system produces the following outputs.

### Annotated video

The generated video can include:

* ball trajectory,
* rim reference and spatial gates,
* attempt state and debug information,
* scoreboard (Attempts / Made / Miss / Unknown).

### Console logs

The console output includes:

* global statistics,
* gate reason histograms,
* detailed event decisions for recent detections.

---

## Requirements

* Python 3.9+
* Git
* NVIDIA GPU with CUDA (optional but recommended)

All Python dependencies are listed in `requirements.txt`.

---

## Installation

### Clone the repository

```bash
git clone https://github.com/USERNAME/basketball-shot-detection.git
cd basketball-shot-detection
```

### Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Prepare the input video

Place the input video file in the `data/input/` directory.

The current pipeline assumes a fixed camera and is not designed for moving-camera footage.

### Run inference

```bash
python src/main.py
```

### Output location

The annotated output video is saved in:

```text
data/output/output_main.mp4
```

Input and output paths, as well as inference parameters, can be adjusted directly in `src/main.py`.

Detailed logs and debug information are printed to the console during execution.

---

## Project Structure

```text
src/
├─ main.py              # Entry point
├─ detect/              # YOLO inference and filtering
├─ track/               # Ball tracking and rim stabilization
├─ events/
│  ├─ attempt/          # Attempt detection
│  └─ made/             # Made / Miss / Unknown logic
├─ video/               # Video I/O, overlays, visualization
├─ utils/               # Shared utilities
```

---

## Method

The processing pipeline follows these main steps:

```text
Video
 → YOLO detection (ball / rim / person / shoot)
 → FakeBallSuppressor
 → RimStabilizer
 → BallTracker
 → AttemptDetector
 → MadeDetector
 → Video overlays and logs
 → Output video
```

The core logic combines object detection, temporal tracking, and rule-based event inference to identify shot attempts and classify outcomes.

---

## Training a Custom Model

To train a custom YOLO model for ball, rim, person, and shoot detection:

1. Prepare and annotate a dataset with bounding boxes for the required classes.
2. Export the dataset in YOLO format.
3. Train a YOLO model using a compatible training framework.
4. Update the `YOLO_WEIGHTS` path in `src/main.py` to use the new weights.
5. Run the inference pipeline with the updated model.

Example:

```python
YOLO_WEIGHTS = r"models\your_trained_model.pt"
```

---

## Limitations

* The pipeline assumes a fixed camera and may degrade with camera motion
* Performance can drop with occlusions, poor lighting, or unusual camera angles
* No player attribution is implemented yet
* Some thresholds depend on input resolution
* Rim detection quality has a strong impact on the final result

---

## Future Work

* Add player attribution for each shot attempt
* Generate shot charts via homography
* Improve multi-resolution normalization
* Support real-time or streaming inference

---

## Disclaimer

This project was developed for academic and research purposes.

Performance depends on several factors, including camera angle, video resolution, lighting conditions, and object visibility.
