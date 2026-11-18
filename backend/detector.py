"""
Sentinel - detection worker
===========================
One CameraWorker = one camera. It runs in its own thread:

    video source -> grab frame -> YOLO finds objects -> send results

It does two things with every frame:
  1. keeps the newest frame as a JPEG in `latest_jpeg` (main.py streams
     this to the browser so the video and the boxes stay in sync)
  2. sends a "frame" message over the websocket with the boxes found

The dashboard decides which boxes are inside your drawn zones, so this
file doesn't need to know anything about zones.
"""

import os
import threading
import time
from datetime import datetime, timezone

import cv2
import numpy as np

# YOLO class ids we care about (COCO dataset): people and vehicles
TARGET_CLASSES = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck
DISPLAY_W, DISPLAY_H = 640, 360   # the video panel on the dashboard is this shape (16:9)
DETECT_SIZE = 800          # longest side of the picture YOLO looks at (bigger = finds smaller things, but slower)
CONFIDENCE = 0.25          # ignore detections the model is less sure about
JPEG_QUALITY = 70

_model = None
_model_lock = threading.Lock()   # only one thread uses the model at a time


def get_model():
    """Load YOLO once and share it between cameras."""
    global _model
    with _model_lock:
        if _model is None:
            from ultralytics import YOLO   # imported here so the server still
            _model = YOLO("yolov8n.pt")    # starts if ultralytics isn't installed
        return _model


def prepare(frame):
    """
    Works for ANY video shape (landscape, portrait phone videos, ...).
      detect_img : what YOLO looks at (longest side = DETECT_SIZE)
      shown      : what the dashboard displays - the same picture,
                   centred on a black 16:9 canvas so nothing is stretched
      scale, ox, oy : how to move a box from detect_img to the shown picture
    """
    h, w = frame.shape[:2]
    k = DETECT_SIZE / max(w, h)
    dw, dh = max(1, int(w * k)), max(1, int(h * k))
    detect_img = cv2.resize(frame, (dw, dh))

    scale = min(DISPLAY_W / dw, DISPLAY_H / dh)
    cw, ch = max(1, int(dw * scale)), max(1, int(dh * scale))
    ox, oy = (DISPLAY_W - cw) // 2, (DISPLAY_H - ch) // 2

    shown = np.zeros((DISPLAY_H, DISPLAY_W, 3), np.uint8)
    shown[oy:oy + ch, ox:ox + cw] = cv2.resize(detect_img, (cw, ch))
    return detect_img, shown, scale, ox, oy


STREAM_MAX_HEIGHT = 720      # for YouTube links: sharper picture helps find small things
MOTION_WARMUP_FRAMES = 15    # the background learner needs a moment at the start


def new_motion_detector():
    return cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=30, detectShadows=True)


def motion_detections(subtractor, detect_img, scale, ox, oy):
    """
    Finds things that MOVE (camera must stay still). It doesn't know what
    they are - it works on cartoons, night footage, anything.
    """
    mask = subtractor.apply(detect_img)
    mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]      # ignore shadows
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    dh, dw = detect_img.shape[:2]
    min_area = 0.001 * dw * dh                                       # skip tiny specks
    found = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        found.append({
            "bbox": [int(x * scale + ox), int(y * scale + oy),
                     int((x + w) * scale + ox), int((y + h) * scale + oy)],
            "confidence": 0.99,
            "label": "motion",
        })
    return found


def make_demo_frame(t):
    """
    A pretend road with a car driving across and a person walking down.
    Used when you type the word  demo  instead of a video. It needs no YOLO,
    so it proves the rest of Sentinel (boxes, lines, alerts, sound) works.
    """
    img = np.full((DISPLAY_H, DISPLAY_W, 3), 55, np.uint8)
    cv2.rectangle(img, (0, 120), (DISPLAY_W, 300), (75, 75, 75), -1)        # road
    for x in range(0, DISPLAY_W, 80):
        cv2.line(img, (x, 210), (x + 40, 210), (200, 200, 200), 2)          # lane marks

    cx = int(((t % 8.0) / 8.0) * (DISPLAY_W + 160)) - 160                   # car: left -> right, 8 s
    car = [cx, 170, cx + 130, 240]
    cv2.rectangle(img, (car[0], car[1]), (car[2], car[3]), (40, 60, 220), -1)

    py = int(((t % 6.0) / 6.0) * (DISPLAY_H + 80)) - 80                     # person: top -> bottom, 6 s
    person = [300, py, 330, py + 80]
    cv2.rectangle(img, (person[0], person[1]), (person[2], person[3]), (60, 200, 90), -1)

    found = []
    for box, label in ((car, "car"), (person, "person")):
        x1 = max(0, box[0]); y1 = max(0, box[1])
        x2 = min(DISPLAY_W, box[2]); y2 = min(DISPLAY_H, box[3])
        if x2 - x1 > 10 and y2 - y1 > 10:
            found.append({"bbox": [x1, y1, x2, y2], "confidence": 0.9, "label": label})
    return img, found


def resolve_source(source):
    """
    Turn whatever the user pasted into something OpenCV can open.
      - a path to a video file on this computer -> used as is
      - a web link (YouTube etc.)               -> yt-dlp finds the real stream
    """
    source = source.strip().strip('"')
    if os.path.exists(source):
        return source

    import yt_dlp
    options = {
        # we only need the picture, so accept video-only streams too
        "format": f"bv*[height<={STREAM_MAX_HEIGHT}]/b[height<={STREAM_MAX_HEIGHT}]/bv*/b",
        "quiet": True,
        "noplaylist": True,
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(source, download=False)
    except Exception as err:
        text = str(err)
        if "format is not available" in text or "JavaScript" in text:
            raise RuntimeError(
                "YouTube would not give a video stream. Most likely yt-dlp needs a "
                "JavaScript runtime. In PowerShell (with (venv) showing) run: "
                "pip install -U \"yt-dlp[default]\" deno   then restart the server. "
                "Or use a saved video file instead."
            )
        raise

    if info.get("url"):
        return info["url"]
    for fmt in info.get("requested_formats") or []:
        if fmt.get("url") and fmt.get("vcodec") not in (None, "none"):
            return fmt["url"]
    raise RuntimeError("Found the video page but no playable stream in it.")


class CameraWorker(threading.Thread):
    def __init__(self, cam_id, source, loop, send):
        super().__init__(daemon=True)
        self.cam_id = cam_id
        self.source = source
        self.loop = loop          # the server's asyncio loop
        self.send = send          # async function that broadcasts a dict
        self.latest_jpeg = None
        self._stop_flag = threading.Event()
        self._input = None                      # newest picture for YOLO to look at
        self._input_event = threading.Event()   # "there is a new picture"
        self._dets = []                         # newest boxes YOLO has found
        self._weak = ""                         # text about faint guesses

    def stop(self):
        self._stop_flag.set()

    # -- helpers ---------------------------------------------------------
    def _emit(self, message):
        import asyncio
        asyncio.run_coroutine_threadsafe(self.send(message), self.loop)

    def _status(self, text):
        print(f"[{self.cam_id}] {text}")
        self._emit({"type": "status", "camera_id": self.cam_id, "message": text})

    def _detect(self, model, detect_img, scale, ox, oy):
        result = model.predict(
            detect_img, conf=CONFIDENCE, classes=TARGET_CLASSES,
            imgsz=DETECT_SIZE, verbose=False,
        )[0]
        found = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            found.append({
                # moved onto the dashboard's picture
                "bbox": [int(x1 * scale + ox), int(y1 * scale + oy),
                         int(x2 * scale + ox), int(y2 * scale + oy)],
                "confidence": round(float(box.conf[0]), 2),
                "label": result.names[int(box.cls[0])],
            })
        return found

    def _weak_guesses(self, model, detect_img):
        """When nothing is found: what does the model see at very low confidence?"""
        result = model.predict(detect_img, conf=0.05, imgsz=DETECT_SIZE, verbose=False)[0]
        best = {}
        for box in result.boxes:
            name = result.names[int(box.cls[0])]
            best[name] = max(best.get(name, 0.0), float(box.conf[0]))
        top = sorted(best.items(), key=lambda kv: -kv[1])[:2]
        if not top:
            return "sees nothing even at very low confidence"
        return "weak guesses: " + ", ".join(f"{n} {c:.2f}" for n, c in top)

    # -- built-in demo (type the word "demo" instead of a video) ----------
    def _run_demo(self):
        self._status("Running (built-in demo, not real detection)")
        started_at = time.time()
        last_report = 0.0
        while not self._stop_flag.is_set():
            tick = time.time()
            shown, detections = make_demo_frame(tick - started_at)
            ok, buf = cv2.imencode(".jpg", shown, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                self.latest_jpeg = buf.tobytes()
            self._emit({
                "type": "frame", "camera_id": self.cam_id,
                "frame_width": DISPLAY_W, "frame_height": DISPLAY_H,
                "detections": detections,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            if tick - last_report > 1.0:
                last_report = tick
                self._emit({"type": "status", "camera_id": self.cam_id,
                            "message": f"Running (demo) - {len(detections)} objects found"})
            time.sleep(max(0.0, 0.08 - (time.time() - tick)))

    # -- main loop -------------------------------------------------------
    def _detection_loop(self, model):
        """Runs YOLO in the background so the video never has to wait for it."""
        last_weak = 0.0
        while not self._stop_flag.is_set():
            if not self._input_event.wait(0.2):
                continue
            self._input_event.clear()
            item = self._input
            if item is None:
                continue
            detect_img, scale, ox, oy = item
            try:
                dets = self._detect(model, detect_img, scale, ox, oy)
                self._dets = dets
                if dets:
                    self._weak = ""
                elif time.time() - last_weak > 3.0:
                    last_weak = time.time()
                    self._weak = self._weak_guesses(model, detect_img)
            except Exception as err:
                self._status(f"Error: {err}")
                time.sleep(1.0)

    def run(self):
        try:
            if self.source.strip().strip('"').lower() == "demo":
                self._run_demo()
                return
            source = self.source.strip()
            motion_mode = source.lower().startswith("motion:")
            if motion_mode:
                source = source[len("motion:"):]

            self._status("Finding the video...")
            stream = resolve_source(source)

            model = None
            if not motion_mode:
                self._status("Loading the detection model (first time can take a minute)...")
                model = get_model()
                threading.Thread(target=self._detection_loop, args=(model,), daemon=True).start()

            announced = False
            last_report = 0.0

            while not self._stop_flag.is_set():
                cap = cv2.VideoCapture(stream)
                if not cap.isOpened():
                    self._status("Error: could not open the video")
                    return

                subtractor = new_motion_detector() if motion_mode else None
                frames_seen = 0
                frame_no = 0
                self._dets = []
                self._weak = ""

                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                if fps > 60 or fps < 1:
                    fps = 25
                frame_time = 1.0 / fps
                if not announced:
                    self._status("Running (motion mode)" if motion_mode else "Running")   # print this only once
                    announced = True

                while not self._stop_flag.is_set():
                    started = time.time()
                    ok, frame = cap.read()
                    if not ok:
                        break   # video ended -> outer loop reopens it

                    detect_img, shown, scale, ox, oy = prepare(frame)
                    if motion_mode:
                        found = motion_detections(subtractor, detect_img, scale, ox, oy)
                        frames_seen += 1
                        self._dets = found if frames_seen > MOTION_WARMUP_FRAMES else []
                    else:
                        # hand the picture to the background YOLO thread and carry on
                        self._input = (detect_img, scale, ox, oy)
                        self._input_event.set()
                    detections = self._dets

                    # once a second, tell the dashboard how many objects we see
                    if time.time() - last_report > 1.0:
                        last_report = time.time()
                        text = f"Running - {len(detections)} objects found"
                        if self._weak and not detections:
                            text += f" | {self._weak}"
                        self._emit({"type": "status", "camera_id": self.cam_id,
                                    "message": text})

                    ok2, buf = cv2.imencode(
                        ".jpg", shown, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                    )
                    if ok2:
                        self.latest_jpeg = buf.tobytes()

                    frame_no += 1
                    if frame_no % 2 == 0:      # boxes at half the video's frame rate is plenty
                        self._emit({
                            "type": "frame",
                            "camera_id": self.cam_id,
                            "frame_width": DISPLAY_W,
                            "frame_height": DISPLAY_H,
                            "detections": detections,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })

                    # keep the video at its real speed
                    spent = time.time() - started
                    if spent < frame_time:
                        time.sleep(frame_time - spent)
                    else:
                        behind = int(spent / frame_time) - 1
                        for _ in range(max(0, min(behind, 60))):
                            cap.grab()

                cap.release()
        except Exception as err:
            self._status(f"Error: {err}")
