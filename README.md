# AI Object Detection Robot — Web App

A computer vision project that uses YOLO, OpenCV, and Flask to detect objects through a live camera feed in a web browser.

## Features

- Real-time object detection using YOLO
- Live camera preview with detection bounding boxes
- Object labels and confidence scores
- Front and rear camera selection on supported phones
- Mobile-friendly web interface
- Obstacle status display

## Technologies Used

- Python
- Flask
- Ultralytics YOLO
- OpenCV
- NumPy
- HTML, CSS, and JavaScript

## How It Works

The web application receives camera frames and uses a YOLO object detection model to identify supported objects. Detected objects are displayed with bounding boxes and labels in the browser.

## Running the Project

1. Install Python.
2. Install the dependencies listed in `requirements.txt`.
3. Ensure the YOLO model file is available.
4. Run the appropriate Flask application.
5. Open the web address displayed in the terminal.

Phone camera access requires HTTPS or localhost. Local HTTPS certificates must be configured separately.

## Safety Note

This is a software demonstration. Object detection and obstacle status are not safety-certified for controlling a physical robot.

## Author

Muhammed Miraz P