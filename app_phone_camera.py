from flask import Flask, Response, request, jsonify, render_template_string
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import threading

app = Flask(__name__)

# Load YOLO once. This app does not open the laptop webcam.
model = YOLO("yolov8n.pt")
model_lock = threading.Lock()

OBSTACLE_CLASSES = {
    "person", "bottle", "chair", "car", "dog",
    "cat", "refrigerator", "laptop", "backpack",
    "suitcase", "cup", "book", "cell phone"
}

PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Robot - Phone Camera</title>
<style>
body {
    margin: 0; padding: 18px;
    background: #101827; color: white;
    font-family: Arial, sans-serif; text-align: center;
}
.container {
    max-width: 720px; margin: auto; padding: 18px;
    background: #1e293b; border-radius: 16px;
}
h1 { color: #38bdf8; }
video, canvas {
    width: 100%; max-width: 640px;
    border-radius: 12px; background: black;
}
canvas { display: none; }
button, select {
    margin: 5px; padding: 12px;
    border: 0; border-radius: 8px; font-size: 16px;
}
button { background: #0284c7; color: white; }
#status { font-weight: bold; color: #86efac; }
</style>
</head>
<body>
<div class="container">
<h1>AI Object Detection Robot</h1>
<p>Phone camera with YOLO detection</p>

<video id="video" autoplay playsinline muted></video>
<canvas id="canvas"></canvas>

<div>
<select id="camera">
    <option value="environment">Back Camera</option>
    <option value="user">Front Camera</option>
</select>
</div>
<div>
<button id="start">Start Camera</button>
<button id="switch">Switch Camera</button>
<button id="stop">Stop Camera</button>
</div>

<p id="status">Camera stopped</p>
<p id="objects">Detected objects: —</p>
<p id="obstacle">Obstacle status: —</p>
</div>

<script>
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const cameraSelect = document.getElementById("camera");
const statusText = document.getElementById("status");
const objectsText = document.getElementById("objects");
const obstacleText = document.getElementById("obstacle");

let stream = null;
let running = false;
let busy = false;
let lastSent = 0;

async function startCamera() {
    stopCamera();

    if (!navigator.mediaDevices ||
        !navigator.mediaDevices.getUserMedia) {
        statusText.textContent =
            "Camera unavailable. Open this site using HTTPS.";
        return;
    }

    try {
        stream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: {
                facingMode: { ideal: cameraSelect.value },
                width: { ideal: 640 },
                height: { ideal: 480 }
            }
        });

        video.srcObject = stream;
        await video.play();
        running = true;
        statusText.textContent = "Camera running";
        requestAnimationFrame(processFrame);
    } catch (error) {
        statusText.textContent =
            "Camera error: " + error.name +
            ". Allow camera permission and use HTTPS.";
    }
}

function stopCamera() {
    running = false;

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    video.srcObject = null;
    statusText.textContent = "Camera stopped";
}

async function processFrame(timestamp) {
    if (!running) return;

    if (timestamp - lastSent > 350 && !busy &&
        video.readyState >= 2) {
        lastSent = timestamp;
        busy = true;

        try {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            const image = canvas.toDataURL("image/jpeg", 0.7);

            const response = await fetch("/detect", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image })
            });

            const data = await response.json();

            if (data.error) {
                statusText.textContent = data.error;
            } else {
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

                for (const item of data.detections) {
                    ctx.strokeStyle = item.obstacle ? "#ff4040" : "#38bdf8";
                    ctx.lineWidth = 3;
                    ctx.strokeRect(item.x1, item.y1,
                                   item.x2 - item.x1,
                                   item.y2 - item.y1);

                    ctx.fillStyle = ctx.strokeStyle;
                    ctx.font = "16px Arial";
                    ctx.fillText(
                        item.name + " " + item.confidence,
                        item.x1, Math.max(18, item.y1 - 5)
                    );
                }

                video.style.display = "none";
                canvas.style.display = "block";

                objectsText.textContent =
                    "Detected objects: " +
                    (data.objects.join(", ") || "None");

                obstacleText.textContent =
                    "Obstacle status: " +
                    (data.obstacle ? "OBSTACLE AHEAD" : "PATH CLEAR");

                obstacleText.style.color =
                    data.obstacle ? "#ff6464" : "#86efac";
            }
        } catch (error) {
            statusText.textContent = "Detection connection error";
        } finally {
            busy = false;
        }
    }

    if (running) requestAnimationFrame(processFrame);
}

document.getElementById("start").onclick = startCamera;
document.getElementById("stop").onclick = stopCamera;
document.getElementById("switch").onclick = async () => {
    cameraSelect.value =
        cameraSelect.value === "environment" ? "user" : "environment";
    await startCamera();
};
cameraSelect.onchange = startCamera;

window.addEventListener("pagehide", stopCamera);
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(PAGE)


@app.route("/detect", methods=["POST"])
def detect():
    try:
        payload = request.get_json(silent=True) or {}
        image_data = payload.get("image", "")

        if "," not in image_data:
            return jsonify({"error": "No camera frame received"}), 400

        raw = base64.b64decode(image_data.split(",", 1)[1])
        frame = cv2.imdecode(
            np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR
        )

        if frame is None:
            return jsonify({"error": "Invalid camera frame"}), 400

        height, width = frame.shape[:2]
        detections = []
        detected_objects = []
        obstacle_ahead = False

        with model_lock:
            result = model(frame, verbose=False)[0]

        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence < 0.35:
                continue

            class_id = int(box.cls[0])
            name = model.names[class_id]
            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            is_obstacle = (
                name in OBSTACLE_CLASSES
                and width // 3 <= (x1 + x2) // 2 <= 2 * width // 3
            )

            if is_obstacle:
                obstacle_ahead = True

            if name not in detected_objects:
                detected_objects.append(name)

            detections.append({
                "name": name,
                "confidence": round(confidence, 2),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "obstacle": is_obstacle
            })

        return jsonify({
            "detections": detections,
            "objects": detected_objects,
            "obstacle": obstacle_ahead
        })

    except Exception as error:
        app.logger.exception("Detection failed")
        return jsonify({"error": "Detection failed on server"}), 500


if __name__ == "__main__":
    import os

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
        debug=False,
        threaded=True
    )
