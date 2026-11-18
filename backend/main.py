"""
Sentinel - backend
==================
Run it with:
    uvicorn main:app --reload
Then open http://localhost:8000

What it does:
  - serves the dashboard (the frontend folder)
  - POST /api/cameras/<cam_id>/source   starts detection on a video
  - GET  /video/<cam_id>                streams that video to the browser
  - /ws                                 pushes detections to the dashboard

Message shapes sent over the websocket:
  {"type": "frame",  "camera_id", "frame_width", "frame_height",
                     "detections": [{"bbox": [x1,y1,x2,y2], "confidence", "label"}], "timestamp"}
  {"type": "status", "camera_id", "message"}
"""

import asyncio
import json

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="Sentinel")

CAMERAS = ["cam_1", "cam_2"]
workers = {}   # camera id -> running CameraWorker


@app.get("/")
async def landing():
    return FileResponse("../frontend/landing.html")


@app.get("/app")
async def dashboard():
    return FileResponse("../frontend/index.html")


class ConnectionManager:
    """Keeps track of connected dashboards and broadcasts to all of them."""

    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict):
        dead = []
        for ws in list(self.active):
            try:
                await ws.send_text(json.dumps(message))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


class Source(BaseModel):
    url: str


@app.post("/api/cameras/{cam_id}/source")
async def set_source(cam_id: str, body: Source):
    if cam_id not in CAMERAS:
        raise HTTPException(404, f"Unknown camera {cam_id}")

    try:
        from detector import CameraWorker
    except ImportError as err:
        raise HTTPException(
            503,
            "Detection libraries are not installed yet. In PowerShell run: "
            f"pip install -r requirements.txt   ({err})",
        )
    except Exception as err:
        # for example a typo you made while editing detector.py
        raise HTTPException(
            500, f"detector.py has a problem - {type(err).__name__}: {err}"
        )

    old = workers.pop(cam_id, None)
    if old:
        old.stop()

    try:
        worker = CameraWorker(
            cam_id, body.url, asyncio.get_running_loop(), manager.broadcast
        )
        workers[cam_id] = worker
        worker.start()
    except Exception as err:
        workers.pop(cam_id, None)
        raise HTTPException(
            500, f"Could not start the detector - {type(err).__name__}: {err}"
        )
    return {"ok": True}


@app.get("/video/{cam_id}")
async def video(cam_id: str):
    worker = workers.get(cam_id)
    if worker is None:
        raise HTTPException(404, "No video started for this camera")

    async def frames():
        last = None
        while workers.get(cam_id) is worker:
            jpg = worker.latest_jpeg
            if jpg is not None and jpg is not last:
                last = jpg
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + jpg + b"\r\n")
            await asyncio.sleep(0.03)

    return StreamingResponse(
        frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )
