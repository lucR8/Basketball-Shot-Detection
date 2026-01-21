# 🏀 Basketball Shot Detection

Computer Vision project to **automatically detect basketball shot attempts and classify their outcomes (Made / Miss / Unknown)** from a single video stream.

Unlike standard object detection tasks, this project focuses on **temporal event inference**:  
a basketball shot is not a single detection, but a **short, structured sequence of visual cues** (shooting motion, ball trajectory, rim interaction).

---

## ✨ Features

- YOLO-based detection (`ball`, `rim`, `person`, `shoot`)
- Robust **ball tracking** (YOLO + temporal continuity)
- **Shot attempt detection** using a finite-state machine (FSM)
- **Made / Miss / Unknown** classification
- Rim stabilization for fixed-camera videos
- Extensive **debug overlays and logs**
- Reproducible, modular pipeline

---

## 📦 Outputs

For each input video, the system produces:

### 🎥 Annotated video
- ball trajectory
- rim reference and spatial gates
- attempt state + debug information
- scoreboard (Attempts / Made / Miss / Unknown)

### 🧾 Console logs
- global statistics
- gate reason histograms
- detailed event decisions (last events)

---

## 1. Requirements

- Python **3.9+**
- Git
- (Optional but recommended) NVIDIA GPU with CUDA

All Python dependencies are listed in `requirements.txt`.

---

## 2. Installation

### 2.1 Clone the repository

```bash
git clone https://github.com/USERNAME/basketball-shot-detection.git
cd basketball-shot-detection
```

## 2.2 Create and activate a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

## 2.3 Install dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Running the project

### 3.1 Prepare input video

Place your input video file in the `data/input/` directory.
    ⚠️ The current pipeline assumes a fixed camera (no camera motion).

### 3.2 Run inference

```bash
python src/main.py
```

### 3.3 View outputs
The annotated output video will be saved in `data/output/output_main.mp4`.
You can adjust input/output paths and other parameters directly in `src/main.py`.
Detailed logs and debug information will be printed to the console.

---
## 4. Project Structure

src/
├─ main.py              # Entry point (pipeline orchestration)
├─ detect/              # YOLO inference + filtering
├─ track/               # Ball tracker & rim stabilizer
├─ events/
│  ├─ attempt/          # Attempt detection (FSM, arming, release)
│  └─ made/             # Made / Miss / Unknown logic
├─ video/               # Video IO, overlays, visualization
├─ utils/               # Shared utilities and helpers

---
## 5. Method overview

Video
 → YOLO detection (ball / rim / person / shoot)
 → FakeBallSuppressor
 → RimStabilizer (fixed camera)
 → BallTracker (temporal continuity)
 → AttemptDetector (FSM + locks)
 → MadeDetector (rim plane pass + swish confirmation)
 → Video overlay + logs
    → Output video

---
## 6. Train your own model

To train your own YOLO model for ball, rim, person, and shoot detection, follow these steps:
1. **Prepare Dataset**: Collect and annotate images/videos with bounding boxes for the classes: `ball`, `rim`, `person`, and `shoot`. You can use tools like [Roboflow](https://roboflow.com/) for annotation and dataset management.
2. **Export Dataset**: Export your annotated dataset in YOLO format.
3. **Train**: Use a YOLO training framework (e.g., [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)) to train your model on the exported dataset.
4. **Update Weights Path**: After training, update the `YOLO_WEIGHTS` path in `src/main.py` to point to your newly trained model weights.
```python
YOLO_WEIGHTS = r"models\your_trained_model.pt"
```
5. **Run Inference**: Execute the main pipeline to see the results with your custom-trained model.
---

## 8. Limitations
- Assumes a fixed camera; performance may degrade with camera motion.
- May struggle with occlusions, extreme lighting, or unconventional camera angles.
- No player attribution yet
- Thresholds are resolution-dependent
- Rim detection quality strongly impacts performance

---

## 9. Future Work
- Implement player attribution for each shot attempt.
- Shot chart via homography
- Multi-resolution normalization
- Real-time / streaming support

---

## 10. Disclaimer

This project is developed for academic and research purposes.
Performance depends on:
- camera angle,
- video resolution,
- lighting conditions,
- ball and rim visibility.

