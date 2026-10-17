from flask import Flask, Response, jsonify
from threading import Lock
import cv2
import atexit

from yolo_obstacle_detector_portfolio import YOLOObstacleDetector

app = Flask(__name__)

detector = None
detector_lock = Lock()


def get_detector():
    global detector

    if detector is None:
        detector = YOLOObstacleDetector()

    return detector


def generate_frames():
    while True:
        try:
            with detector_lock:
                camera = get_detector()
                obstacle, frame, status, objects = camera.detect()

                if frame is None:
                    continue

                success, buffer = cv2.imencode(".jpg", frame)

            if not success:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

        except GeneratorExit:
            break
        except Exception as error:
            print("Camera error:", error)
            break


@app.route("/")
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport"
              content="width=device-width, initial-scale=1">
        <title>AI Object Detection Robot</title>
        <style>
            body {
                margin: 0;
                padding: 20px;
                background: #101827;
                color: white;
                font-family: Arial, sans-serif;
                text-align: center;
            }
            .container {
                max-width: 850px;
                margin: auto;
                background: #1e293b;
                padding: 20px;
                border-radius: 16px;
            }
            h1 { color: #38bdf8; }
            img {
                width: 100%;
                max-width: 720px;
                border-radius: 12px;
                background: #000;
            }
            .note {
                color: #cbd5e1;
                line-height: 1.6;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>AI Object Detection Robot</h1>
            <p>Live YOLO object detection</p>
            <img src="/video_feed" alt="Live camera detection feed">
            <p class="note">
                Detected objects and obstacle status are displayed
                on the camera image.
            </p>
        </div>
    </body>
    </html>
    """


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def close_detector():
    global detector

    if detector is not None:
        detector.close()
        detector = None


atexit.register(close_detector)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True
    )