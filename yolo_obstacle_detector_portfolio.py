import cv2
from ultralytics import YOLO


class YOLOObstacleDetector:
    def __init__(self, camera_index=0):
        print("Loading YOLO model...")
        self.model = YOLO("yolov8n.pt")
        self.camera = cv2.VideoCapture(camera_index)

        if not self.camera.isOpened():
            raise RuntimeError("Could not open camera")

        self.obstacle_classes = {
            "person", "bottle", "chair", "car", "dog",
            "cat", "refrigerator", "laptop", "backpack",
            "suitcase", "cup", "book", "cell phone"
        }

    def detect(self):
        success, frame = self.camera.read()

        if not success:
            return False, None, "CAMERA ERROR", []

        results = self.model(frame, verbose=False)
        detected_objects = []
        obstacle_ahead = False
        height, width = frame.shape[:2]

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            name = self.model.names[class_id]
            confidence = float(box.conf[0])

            if confidence < 0.35:
                continue

            if name not in detected_objects:
                detected_objects.append(name)

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            center_x = (x1 + x2) // 2

            if (
                name in self.obstacle_classes
                and width // 3 <= center_x <= 2 * width // 3
            ):
                obstacle_ahead = True

        annotated = results[0].plot()

        if detected_objects:
            objects_text = ", ".join(
                name.upper() for name in detected_objects
            )
        else:
            objects_text = "NONE"

        status = (
            f"OBJECTS: {objects_text} | "
            f"{'OBSTACLE AHEAD' if obstacle_ahead else 'PATH CLEAR'}"
        )

        cv2.putText(
            annotated,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255) if obstacle_ahead else (0, 255, 0),
            2
        )

        return obstacle_ahead, annotated, status, detected_objects

    def close(self):
        self.camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = YOLOObstacleDetector()

    try:
        while True:
            obstacle, frame, status, objects = detector.detect()

            if frame is None:
                print(status)
                break

            print(status)
            cv2.imshow("YOLO Live Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        detector.close()