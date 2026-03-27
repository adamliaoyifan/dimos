# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""VLN Web Input — web interface with text input, image upload, and camera feed.

Provides a browser-based UI at http://localhost:<port> where users can:
- Type navigation goals as text  (e.g. "find the glasses box in CTO office")
- Upload a reference image of the target object
- See the live camera feed and navigation status
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from threading import Thread
from typing import Any

import cv2
import numpy as np
import reactivex as rx
from reactivex.disposable import Disposable
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import uvicorn

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.core.transport import pLCMTransport
from dimos.msgs.sensor_msgs.Image import Image
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class VLNWebConfig(ModuleConfig):
    port: int = 5556


class VLNWebInput(Module[VLNWebConfig]):
    """Web interface for VLN task input.

    Serves a page with:
    - Text input box for navigation goals
    - Image upload for reference target images
    - Live camera feed from the robot
    - Navigation status display
    """

    default_config = VLNWebConfig

    color_image: In[Image]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._app = FastAPI(title="VLN Navigation Interface")
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self._server: uvicorn.Server | None = None
        self._thread: Thread | None = None
        self._human_transport: pLCMTransport[str] | None = None
        self._latest_frame: np.ndarray | None = None
        self._uploaded_image: np.ndarray | None = None
        self._status: str = "idle"
        self._setup_routes()

    def _setup_routes(self) -> None:
        @self._app.get("/", response_class=HTMLResponse)
        async def index() -> HTMLResponse:
            return HTMLResponse(content=_VLN_HTML)

        @self._app.post("/navigate")
        async def navigate(
            goal: str = Form(...),
            image: UploadFile | None = File(None),
        ) -> JSONResponse:
            """Accept a text goal and optional reference image."""
            # Handle optional image upload
            if image and image.filename:
                data = await image.read()
                arr = np.frombuffer(data, np.uint8)
                self._uploaded_image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                logger.info(f"[VLN-Web] Received reference image: {image.filename}")

            if goal and self._human_transport:
                self._human_transport.publish(goal)
                self._status = f"navigating: {goal}"
                logger.info(f"[VLN-Web] Goal submitted: {goal}")
                return JSONResponse({"success": True, "goal": goal})

            return JSONResponse({"success": False, "message": "No goal provided"})

        @self._app.post("/stop")
        async def stop_nav() -> JSONResponse:
            if self._human_transport:
                self._human_transport.publish("stop navigation")
                self._status = "stopped"
            return JSONResponse({"success": True})

        @self._app.get("/status")
        async def status() -> JSONResponse:
            return JSONResponse({"status": self._status})

        @self._app.get("/camera_feed")
        async def camera_feed() -> StreamingResponse:
            """MJPEG stream of the robot camera."""
            return StreamingResponse(
                self._mjpeg_generator(),
                media_type="multipart/x-mixed-replace; boundary=frame",
            )

        @self._app.get("/uploaded_image")
        async def get_uploaded() -> JSONResponse:
            """Return the uploaded reference image as base64."""
            if self._uploaded_image is None:
                return JSONResponse({"image": None})
            _, buf = cv2.imencode(".jpg", self._uploaded_image)
            b64 = base64.b64encode(buf.tobytes()).decode()
            return JSONResponse({"image": f"data:image/jpeg;base64,{b64}"})

    def _mjpeg_generator(self):  # type: ignore[no-untyped-def]
        import time

        while True:
            if self._latest_frame is not None:
                _, buf = cv2.imencode(".jpg", self._latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            time.sleep(0.1)

    @rpc
    def start(self) -> None:
        super().start()
        self._human_transport = pLCMTransport("/human_input")
        self._disposables.add(Disposable(self.color_image.subscribe(self._on_image)))

        config = uvicorn.Config(
            self._app,
            host="0.0.0.0",
            port=self.config.port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)
        self._thread = Thread(target=self._server.run, daemon=True)
        self._thread.start()
        logger.info(f"[VLN-Web] Interface running at http://localhost:{self.config.port}")

    @rpc
    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._human_transport:
            self._human_transport.lcm.stop()
        super().stop()

    def _on_image(self, image: Image) -> None:
        if hasattr(image, "data") and image.data is not None:
            self._latest_frame = cv2.cvtColor(image.data, cv2.COLOR_RGB2BGR)

    def get_uploaded_image(self) -> np.ndarray | None:
        """Return the most recently uploaded reference image (BGR numpy array)."""
        return self._uploaded_image


# ---------------------------------------------------------------------------
# Inline HTML for the VLN interface
# ---------------------------------------------------------------------------
_VLN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VLN Navigation</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, -apple-system, sans-serif; background: #1a1a2e; color: #eee; }
  .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
  h1 { text-align: center; margin-bottom: 20px; color: #00d4ff; font-size: 1.5rem; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .panel { background: #16213e; border-radius: 12px; padding: 20px; }
  .panel h2 { font-size: 1rem; color: #00d4ff; margin-bottom: 12px; }
  #camera { width: 100%; border-radius: 8px; background: #000; min-height: 300px; }
  .input-group { margin-bottom: 12px; }
  .input-group label { display: block; font-size: 0.85rem; color: #aaa; margin-bottom: 4px; }
  input[type=text] {
    width: 100%; padding: 10px 14px; border: 1px solid #333; border-radius: 8px;
    background: #0f3460; color: #fff; font-size: 1rem; outline: none;
  }
  input[type=text]:focus { border-color: #00d4ff; }
  input[type=file] { color: #aaa; font-size: 0.85rem; margin-top: 4px; }
  .btn {
    padding: 10px 24px; border: none; border-radius: 8px; font-size: 1rem;
    cursor: pointer; font-weight: 600; transition: background 0.2s;
  }
  .btn-go { background: #00d4ff; color: #1a1a2e; margin-right: 8px; }
  .btn-go:hover { background: #00b8d9; }
  .btn-stop { background: #e94560; color: #fff; }
  .btn-stop:hover { background: #c73e54; }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  #status { margin-top: 16px; padding: 10px; background: #0f3460; border-radius: 8px; font-size: 0.9rem; }
  #status .label { color: #aaa; }
  #ref-preview { max-width: 200px; max-height: 150px; border-radius: 6px; margin-top: 8px; display: none; }
  .actions { display: flex; gap: 8px; margin-top: 12px; }
  #log { margin-top: 12px; max-height: 200px; overflow-y: auto; font-size: 0.8rem; color: #888; }
  #log div { padding: 2px 0; border-bottom: 1px solid #1a1a2e; }
  @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="container">
  <h1>VLN Navigation Interface</h1>
  <div class="grid">
    <div class="panel">
      <h2>Camera Feed</h2>
      <img id="camera" src="/camera_feed" alt="Camera feed">
    </div>
    <div class="panel">
      <h2>Navigation Target</h2>
      <form id="nav-form" enctype="multipart/form-data">
        <div class="input-group">
          <label for="goal">Text goal</label>
          <input type="text" id="goal" name="goal"
                 placeholder="e.g. find the glasses box in CTO office room" autofocus>
        </div>
        <div class="input-group">
          <label for="image">Reference image (optional)</label>
          <input type="file" id="image" name="image" accept="image/*">
          <img id="ref-preview" alt="preview">
        </div>
        <div class="actions">
          <button type="submit" class="btn btn-go" id="btn-go">Navigate</button>
          <button type="button" class="btn btn-stop" id="btn-stop">Stop</button>
        </div>
      </form>
      <div id="status"><span class="label">Status:</span> <span id="status-text">idle</span></div>
      <div id="log"></div>
    </div>
  </div>
</div>
<script>
const form = document.getElementById('nav-form');
const goalEl = document.getElementById('goal');
const imageEl = document.getElementById('image');
const preview = document.getElementById('ref-preview');
const statusText = document.getElementById('status-text');
const logEl = document.getElementById('log');

function log(msg) {
  const d = document.createElement('div');
  d.textContent = new Date().toLocaleTimeString() + ' ' + msg;
  logEl.prepend(d);
  if (logEl.children.length > 50) logEl.lastChild.remove();
}

imageEl.addEventListener('change', () => {
  const file = imageEl.files[0];
  if (file) { preview.src = URL.createObjectURL(file); preview.style.display = 'block'; }
  else { preview.style.display = 'none'; }
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData();
  fd.append('goal', goalEl.value);
  if (imageEl.files[0]) fd.append('image', imageEl.files[0]);
  try {
    const r = await fetch('/navigate', { method: 'POST', body: fd });
    const j = await r.json();
    if (j.success) { statusText.textContent = 'navigating: ' + j.goal; log('Sent: ' + j.goal); }
    else { log('Error: ' + j.message); }
  } catch(err) { log('Network error: ' + err); }
});

document.getElementById('btn-stop').addEventListener('click', async () => {
  await fetch('/stop', { method: 'POST' });
  statusText.textContent = 'stopped';
  log('Navigation stopped');
});

setInterval(async () => {
  try {
    const r = await fetch('/status');
    const j = await r.json();
    statusText.textContent = j.status;
  } catch(e) {}
}, 2000);
</script>
</body>
</html>"""

__all__ = ["VLNWebInput"]
