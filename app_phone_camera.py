from flask import Flask, render_template_string

app = Flask(__name__)

PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Object Detection Robot</title>

<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@2.2.3/dist/coco-ssd.min.js"></script>

<style>
* { box-sizing: border-box; }

body {
    margin: 0;
    padding: 18px;
    background: #101827;
    color: white;
    font-family: Arial, sans-serif;
    text-align: center;
}

.container {
    max-width: 720px;
    margin: auto;
    padding: 18px;
    background: #1e293b;
    border-radius: 16px;
}

h1 { color: #38bdf8; }

.camera-box {
    position: relative;
    width: 100%;
    max-width: 640px;
    margin: auto;
    background: black;
    border-radius: 12px;
    overflow: hidden;
}

video {
    display: block;
    width: 100%;
    height: auto;
}

canvas {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
}

button, select {
    margin: 5px;
    padding: 12px;
    border: 0;
    border-radius: 8px;
    font-size: 16px;
}

button {
    background: #0284c7;
    color: white;
    cursor: pointer;
}

button:disabled {
    opacity: 0.55;
    cursor: not-allowed;
}

#status { color: #86efac; font-weight: bold; }
#obstacle { font-weight: bold; }
</style>
</head>

<body>
<div class="container">
    <h1>AI Object Detection Robot</h1>
    <p>Browser-based AI camera detection</p>

    <div class="camera-box">
        <video id="video" autoplay playsinline muted></video>
        <canvas id="canvas"></canvas>
    </div>

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

    <p id="status">Loading AI model...</p>
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
const startButton = document.getElementById("start");
const switchButton = document.getElementById("switch");

const OBSTACLE_CLASSES = new Set([
    "person", "bottle", "chair", "car", "dog", "cat",
    "refrigerator", "laptop", "backpack", "suitcase",
    "cup", "book", "cell phone", "bus", "truck",
    "bicycle", "motorcycle", "potted plant", "bench"
]);

let model = null;
let stream = null;
let running = false;
let busy = false;
let session = 0;
let lastDetection = 0;

async function loadModel() {
    try {
        statusText.textContent = "Loading browser AI model...";
        model = await cocoSsd.load({ base: "lite_mobilenet_v2" });
        statusText.textContent = "AI model ready. Start the camera.";
    } catch (error) {
        statusText.textContent =
            "AI model failed to load. Check your internet connection and refresh.";
        console.error("Model loading error:", error);
    }
}

async function startCamera() {
    stopCamera();

    if (!model) {
        statusText.textContent = "Please wait for the AI model to load.";
        return;
    }

    if (!navigator.mediaDevices?.getUserMedia) {
        statusText.textContent = "Camera requires HTTPS and browser permission.";
        return;
    }

    const currentSession = session;

    try {
        stream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: {
                facingMode: { ideal: cameraSelect.value },
                width: { ideal: 640 },
                height: { ideal: 480 }
            }
        });

        if (currentSession !== session) {
            stream.getTracks().forEach(track => track.stop());
            return;
        }

        video.srcObject = stream;
        await video.play();

        if (currentSession !== session) return;

        running = true;
        lastDetection = 0;
        statusText.textContent = "Camera running. Detecting objects...";
        requestAnimationFrame(processFrame);
    } catch (error) {
        statusText.textContent = "Camera error: " + error.message;
        console.error("Camera error:", error);
    }
}

function stopCamera() {
    session++;
    running = false;

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    video.srcObject = null;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    objectsText.textContent = "Detected objects: —";
    obstacleText.textContent = "Obstacle status: —";
    obstacleText.style.color = "white";
    statusText.textContent = model
        ? "Camera stopped"
        : "Loading AI model...";
}

async function processFrame(timestamp) {
    if (!running || busy) {
        if (running) requestAnimationFrame(processFrame);
        return;
    }

    if (video.readyState < 2 || timestamp - lastDetection < 250) {
        requestAnimationFrame(processFrame);
        return;
    }

    busy = true;
    lastDetection = timestamp;
    const currentSession = session;

    try {
        const predictions = await model.detect(video);

        if (!running || currentSession !== session) return;

        const width = video.videoWidth;
        const height = video.videoHeight;

        if (!width || !height) return;

        if (canvas.width !== width || canvas.height !== height) {
            canvas.width = width;
            canvas.height = height;
        }

        ctx.clearRect(0, 0, width, height);

        const detectedNames = new Set();
        let obstacleAhead = false;

        for (const item of predictions) {
            if (item.score < 0.45) continue;

            const [x, y, w, h] = item.bbox;
            const name = item.class;
            const centerX = x + w / 2;

            const isObstacle =
                OBSTACLE_CLASSES.has(name) &&
                centerX >= width / 3 &&
                centerX <= (2 * width) / 3;

            if (isObstacle) obstacleAhead = true;
            detectedNames.add(name);

            ctx.strokeStyle = isObstacle ? "#ff4040" : "#38bdf8";
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, w, h);

            const label =
                name + " " + Math.round(item.score * 100) + "%";
            ctx.font = "16px Arial";

            const labelY = Math.max(18, y - 5);
            ctx.fillStyle = isObstacle ? "#ff4040" : "#0284c7";
            ctx.fillRect(
                x,
                labelY - 17,
                Math.min(ctx.measureText(label).width + 8, width - x),
                21
            );

            ctx.fillStyle = "white";
            ctx.fillText(label, x + 4, labelY - 2);
        }

        objectsText.textContent =
            "Detected objects: " +
            ([...detectedNames].join(", ") || "None");

        obstacleText.textContent =
            "Obstacle status: " +
            (obstacleAhead ? "OBSTACLE AHEAD" : "PATH CLEAR");

        obstacleText.style.color =
            obstacleAhead ? "#ff6464" : "#86efac";

        statusText.textContent = "Browser AI detection running";
    } catch (error) {
        if (currentSession === session) {
            statusText.textContent = "Detection error: " + error.message;
            console.error("Detection error:", error);
        }
    } finally {
        busy = false;
        if (running && currentSession === session) {
            requestAnimationFrame(processFrame);
        }
    }
}

startButton.onclick = startCamera;
document.getElementById("stop").onclick = stopCamera;

switchButton.onclick = async () => {
    cameraSelect.value =
        cameraSelect.value === "environment" ? "user" : "environment";
    await startCamera();
};

cameraSelect.onchange = startCamera;
window.addEventListener("pagehide", stopCamera);

loadModel();
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(PAGE)


if __name__ == "__main__":
    import os
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
        debug=False,
        threaded=True
    )
