# Basketball Shot Detection 🏀

Computer Vision project to automatically detect basketball shot attempts and made shots from video.

The system takes a video as input and outputs:
- an annotated video
- a JSON file with detections and events
- basic shooting statistics (attempts / made)

---

## 1. Requirements

- Python **3.9+**
- Git

All Python dependencies are listed in `requirements.txt`.

---

## 2. Installation


### 2.1 Clone the repository

```bash
git clone https://github.com/USERNAME/basketball-shot-detection.git
cd basketball-shot-detection
```

### 2.3 Install dependencies
```bash
pip install -r requirements.txt
```

## 3. Running the project
### 3.1 Add a video

Place a video file here:

```bash
data/input/sample.mp4
```

### 3.2 Run inference
```bash
python src/main.py --config configs/default.yaml
```

Outputs:
`data/output/annotated.mp4`
`data/output/detections.json`

## 4. Project structure
src/
├─ main.py            # Entry point
├─ detect/           # YOLO detection
├─ track/            # Ball tracking
├─ events/           # Shot attempt / made logic
├─ video/            # Video IO and visualization

## 5. Configuration

All parameters (model, thresholds, paths) are defined in:
```bash
configs/default.yaml
```
## 6. Roadmap

YOLO-based detection

Ball tracking

Shot attempt detection

Made vs miss classification

Player tracking (optional)

Shot chart visualization

7. Disclaimer

This project is developed for academic purposes.
Performance depends on camera angle, video quality, and visibility of the ball and rim.