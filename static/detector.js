// Browser-based object detection using TensorFlow.js and COCO-SSD
let detectorModel = null;
let detectorBusy = false;

async function loadObjectDetector() {
    const status = document.getElementById("detectionStatus");

    try {
        status.textContent = "Loading browser AI model...";

        detectorModel = await cocoSsd.load();

        status.textContent = "AI detector ready";
        return true;
    } catch (error) {
        console.error("Detector loading failed:", error);
        status.textContent = "Could not load AI model";
        return false;
    }
}

async function detectObjectsInVideo(videoElement) {
    if (!detectorModel || detectorBusy) return;

    if (!videoElement || videoElement.readyState < 2) return;

    detectorBusy = true;

    try {
        const predictions = await detectorModel.detect(videoElement);

        const names = [...new Set(
            predictions
                .filter(item => item.score >= 0.45)
                .map(item => item.class)
        )];

        const detectedElement = document.getElementById("detectedObjects");
        const obstacleElement = document.getElementById("obstacleStatus");

        if (detectedElement) {
            detectedElement.textContent =
                names.length ? names.join(", ") : "No objects detected";
        }

        if (obstacleElement) {
            obstacleElement.textContent =
                names.length ? "Object detected" : "Path appears clear";
        }
    } catch (error) {
        console.error("Object detection failed:", error);
    } finally {
        detectorBusy = false;
    }
}
