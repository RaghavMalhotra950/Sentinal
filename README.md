# Border Watch — multi-camera zone intrusion alerting

A weekend hackathon project: watch video streams (YouTube live/recorded),
detect objects with YOLO, and raise an alert when something enters a
user-defined "sensitive zone."

## Live demo

- Landing page: https://raghavmalhotra950.github.io/Sentinal/
- Interactive dashboard: https://raghavmalhotra950.github.io/Sentinal/app/

The GitHub Pages build runs the frontend in browser-based demo mode, so zone
drawing, simulated detections, alerts, sound, and screen flashes work online.
GitHub Pages cannot run the Python FastAPI/YOLO backend; live video detection
requires deploying `backend/` to a Python-capable host.

## Team

- **Person A** — stream ingestion (YouTube → OpenCV), YOLOv8 detection,
  zone intrusion logic. Owns `backend/main.py`.
- **Person B** — dashboard, zone-drawing UI, alerts (sound/flash/webhook).
  Owns `frontend/index.html`.

Both sides only need to agree on one thing: the **message contract**
documented at the top of `backend/main.py`. As long as detection events
keep that shape, you can each work independently.

## Quick start (for Person B, right now, first project — step by step)

You don't need to wait for the real backend to start building. Here's how
to get moving in the next 10 minutes:

1. **Install Python** if you don't have it (3.10+). Check with:
   ```
   python3 --version
   ```

2. **Open a terminal in this project folder**, then set up a virtual
   environment (keeps this project's packages separate from everything
   else on your machine):
   ```
   cd border-watch/backend
   python3 -m venv venv
   source venv/bin/activate        # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the stub backend:**
   ```
   uvicorn main:app --reload
   ```
   You should see it say it's running on `http://127.0.0.1:8000`.

4. **Open your browser** to `http://localhost:8000`. You'll see the
   dashboard, already receiving fake detection events from the stub
   backend every couple seconds — that's `fake_detection_loop()` in
   `main.py` standing in for the real thing.

5. **Try it out:**
   - Click **"Draw zone"** on a camera panel, then click a few points on
     the video area to draw a polygon (a "sensitive area"). Click "Draw
     zone" again to finish.
   - Watch the alert log on the right — when a fake detection's box
     center falls inside your zone, it'll log as an "intrusion," beep,
     and flash the screen red.
   - Toggle **Demo mode** in Controls to switch between the real
     websocket feed and a fully client-side simulation — useful if
     Person A's backend isn't running yet, or you're working on a plane.

That's it — you have a working, clickable thing to demo within your first
hour. Everything past this point is refinement.

## What to build next (roughly in order)

1. Make the video panel show something real — either:
   - embed the actual YouTube video as a background layer under the
     transparent canvas (simplest, looks great, but you can't draw
     bounding boxes *from* a YouTube iframe — Person A's backend handles
     the actual frame analysis separately), or
   - once Person A has frame images flowing, render those frames to the
     canvas directly (gives you a real "AI is watching this" feel).
   Ask Person A which they're doing server-side — it changes this.
2. Polish the zone-drawing UX (undo last point, multiple zones per
   camera, zone labels).
3. Wire up a real alert channel — a Discord webhook is the fastest for a
   demo. **Do this through the backend, not the browser** — a webhook URL
   in client-side JS is visible to anyone who opens dev tools. Send the
   alert event to a backend route, have the backend call Discord.
4. Add a "severity" concept if you have time — e.g. how long an object
   has been in the zone, or how close to the border of the zone.

## Things to not worry about right now

- Multiple simultaneous live streams. One working stream, well-integrated
  and reliable for your live demo, beats three flaky ones. Have a backup
  recorded video ready in case a livestream goes down mid-demo.
- A fancy zone-drawing UI (freehand curves, etc). Straight-line polygons
  are enough.
- Authentication/security. Not what judges are scoring in a weekend demo.

## Message contract (the thing both of you must agree on)

See the full spec in the docstring at the top of `backend/main.py`. Short
version: every detection is a JSON object with `camera_id`, `bbox`,
`frame_width`/`frame_height`, `confidence`, `in_zone`, `zone_id`, and
`timestamp`. If Person A needs to change this shape, tell Person B before
changing it — the frontend is built against it.
